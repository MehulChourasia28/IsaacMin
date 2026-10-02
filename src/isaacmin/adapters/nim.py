"""Hosted NVIDIA adapter with bounded tools, image payloads and sanitized evidence."""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from isaacmin.assets.network import ServiceError, atomic_json, digest, retry_delay, utcnow

PLANNER = "nvidia/nemotron-3-ultra-550b-a55b"
VISION = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
BASE = "https://integrate.api.nvidia.com/v1"


def planner_model(workspace: Path) -> str:
    """Explicit user-authorized NVIDIA selection; no hidden provider fallback."""
    path=Path(workspace)/'state/planner_selection.json'
    if not path.is_file():return PLANNER
    selection=json.loads(path.read_text())
    model=selection['model']
    if selection.get('user_authorized_nvidia_selection') is not True or not re.fullmatch(r'nvidia/[a-z0-9-]+',model):
        raise ServiceError('model_selection_invalid','Planner selection requires recorded NVIDIA-only authority')
    return model
DOCS = {
    "planner": "https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-ultra-550b-a55b-infer",
    "vision": "https://docs.api.nvidia.com/nim/re/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-infer",
    "polling": "https://docs.api.nvidia.com/nim/re/reference/nvidia-nemotron-3-nano-omni-30b-a3b-reasoning-statuspolling",
}


def snapshot_official_references(workspace: Path):
    references = {**DOCS, "planner_catalogue": "https://build.nvidia.com/" + PLANNER,
                  "vision_catalogue": "https://build.nvidia.com/" + VISION}
    records = {}
    # Separate unauthenticated client: NVIDIA_API_KEY is sent exclusively to BASE.
    with httpx.Client(timeout=httpx.Timeout(30, connect=15), follow_redirects=True,
                      headers={"User-Agent": "IsaacMin/0.1"}) as client:
        for role, url in references.items():
            try:
                response = client.get(url)
                records[role] = {"url": url, "retrieved_at_utc": utcnow(), "http_status": response.status_code,
                                 "response_sha256": hashlib.sha256(response.content).hexdigest(),
                                 "response_bytes": len(response.content)}
            except httpx.HTTPError:
                records[role] = {"url": url, "status": "network_unavailable", "retrieved_at_utc": utcnow()}
    atomic_json(Path(workspace) / "evidence/services/nvidia_official_references.json", records)
    return records


def load_key(workspace: Path) -> str:
    value = os.environ.get("NVIDIA_API_KEY", "")
    secret = workspace / ".env"
    if not value and secret.is_file():
        if secret.is_symlink() or secret.stat().st_mode & 0o077:
            raise ServiceError("unsafe_secret_file", "Project .env must be a regular file with mode 0600")
        for line in secret.read_text().splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                value = line.partition("=")[2].strip().strip("\"'")
                break
    if not value:
        raise ServiceError("missing_key", "NVIDIA_API_KEY is absent from environment and protected project .env")
    if "\n" in value or "\r" in value:
        raise ServiceError("invalid_key", "NVIDIA_API_KEY has invalid format")
    return value


def redact(value, secret=""):
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if k.lower() in {"authorization", "api_key", "nvidia_api_key", "reasoning", "reasoning_content"} else redact(v, secret)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, str):
        if secret:
            value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"nvapi-[A-Za-z0-9_-]+", "[REDACTED]", value)
        return re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", value, flags=re.I)
    return value


class NIMClient:
    def __init__(self, workspace: Path, max_attempts=5, deadline=900, allow_access_reprobe=False):
        self.workspace = Path(workspace)
        self._key = load_key(self.workspace)
        self.max_attempts, self.deadline = max_attempts, deadline
        self.allow_access_reprobe = allow_access_reprobe
        self.client = httpx.Client(timeout=httpx.Timeout(300, connect=15), follow_redirects=False,
                                   headers={"Authorization": "Bearer " + self._key,
                                            "Content-Type": "application/json", "Accept": "application/json"})
        self.trace_path = self.workspace / "evidence/services/nim_calls.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    def _trace(self, data):
        with self.trace_path.open("a") as f:
            f.write(json.dumps(redact(data, self._key), sort_keys=True) + "\n")

    def _availability(self, model, status, **metadata):
        path = self.workspace / "state/nim_availability.json"
        current = json.loads(path.read_text()) if path.exists() else {"models": {}}
        current["models"][model] = {"status": status, "checked_at_utc": utcnow(), **metadata}
        atomic_json(path, current)

    def chat(self, model, messages, **options):
        lock_path = self.workspace / "state/nim_request.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with lock_path.open("a") as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() - started > self.deadline:
                        raise ServiceError("waiting_provider", "Another process holds the single NVIDIA request slot", True)
                    time.sleep(0.25)
            try:
                return self._chat_unlocked(model, messages, **options)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _chat_unlocked(self, model, messages, **options):
        if model not in (PLANNER, VISION, planner_model(self.workspace)):
            raise ServiceError("model_substitution_forbidden", "Only the configured specification models are allowed")
        availability = self.workspace / "state/nim_availability.json"
        if availability.exists() and not self.allow_access_reprobe:
            previous = json.loads(availability.read_text()).get("models", {}).get(model, {})
            if previous.get("code") in ("invalid_key", "access_restricted"):
                raise ServiceError(previous["code"], "Previous NVIDIA authentication/access denial retained; explicit access reprobe required after account/credential change")
        payload = {"model": model, "messages": messages, "stream": False, "max_tokens": 768, **options}
        if self._key in json.dumps(payload):
            raise ServiceError("secret_in_payload", "NVIDIA credentials are forbidden in model messages and generation options")
        if model == PLANNER:
            payload.setdefault("reasoning_effort", "none")
        elif model == VISION:
            payload.setdefault("reasoning_budget", 128)
        call_id = str(uuid.uuid4())
        start = time.monotonic()
        # Durable checkpoint stores identity and shape, never request text/images or secrets.
        self._trace({"event": "request_checkpoint", "call_id": call_id, "model": model, "time": utcnow(),
                     "message_count": len(messages), "max_tokens": payload["max_tokens"],
                     "options": {k: v for k, v in payload.items() if k not in ("messages", "tools")}})
        r = None
        for attempt in range(self.max_attempts):
            remaining = self.deadline - (time.monotonic() - start)
            if remaining <= 0:
                raise ServiceError("request_deadline", "NVIDIA request deadline exhausted", True)
            try:
                r = self.client.post(BASE + "/chat/completions", json=payload,
                                     timeout=httpx.Timeout(min(300, remaining), connect=min(15, remaining)))
                while r.status_code == 202:
                    body = r.json()
                    request_id = r.headers.get("nvcf-reqid") or body.get("requestId")
                    try:
                        request_id = str(uuid.UUID(request_id))
                    except (ValueError, TypeError, AttributeError):
                        raise ServiceError("pending_schema", "NVIDIA pending response lacks a valid request ID", True) from None
                    delay = max(1, retry_delay(r.headers.get("retry-after"), 0))
                    if time.monotonic() - start + delay >= self.deadline:
                        raise ServiceError("request_deadline", "NVIDIA pending result exceeded deadline", True)
                    time.sleep(delay)
                    remaining = self.deadline - (time.monotonic() - start)
                    r = self.client.get(BASE + "/status/" + request_id,
                                        timeout=httpx.Timeout(min(300, remaining), connect=min(15, remaining)))
                if r.status_code == 429 or r.status_code >= 500:
                    delay = retry_delay(r.headers.get("retry-after"), attempt)
                    if attempt + 1 < self.max_attempts and time.monotonic() - start + delay < self.deadline:
                        self._trace({'event':'transport_retry','call_id':call_id,'model':model,
                                     'http_status':r.status_code,'attempt':attempt+1,'delay_seconds':delay})
                        time.sleep(delay)
                        continue
                break
            except httpx.HTTPError:
                if attempt + 1 == self.max_attempts:
                    self._trace({"event": "failure", "call_id": call_id, "code": "network"})
                    raise ServiceError("network", "NVIDIA connection failed; credentials and headers suppressed", True) from None
                delay=retry_delay(None,attempt)
                self._trace({'event':'transport_retry','call_id':call_id,'model':model,
                             'code':'network','attempt':attempt+1,'delay_seconds':delay})
                time.sleep(delay)
        elapsed = time.monotonic() - start
        if r is None:
            raise ServiceError("network", "NVIDIA request produced no response", True)
        if r.status_code != 200:
            code = {400: "invalid_payload", 401: "invalid_key", 403: "access_restricted", 404: "model_unavailable",
                    422: "invalid_payload", 429: "rate_limit"}.get(r.status_code, "provider_outage" if r.status_code >= 500 else "provider_http")
            # Do not persist arbitrary response bodies: providers sometimes echo submitted input.
            self._trace({"event": "failure", "call_id": call_id, "http_status": r.status_code,
                         "model": model, "latency_seconds": elapsed, "code": code})
            self._availability(model, "blocked", http_status=r.status_code, code=code, call_id=call_id)
            raise ServiceError(code, f"NVIDIA {model} returned HTTP {r.status_code}", r.status_code == 429 or r.status_code >= 500)
        try:
            obj = r.json()
            choice = obj["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError()
        except (ValueError, KeyError, IndexError, TypeError):
            raise ServiceError("malformed_response", "NVIDIA response lacks a valid completion message") from None
        record = {"event": "response", "call_id": call_id, "http_status": 200, "model": obj.get("model", model),
                  "configured_model": model, "latency_seconds": elapsed, "finish_reason": choice.get("finish_reason"),
                  "usage": obj.get("usage"), "message": {k: v for k, v in message.items() if k in ("role", "content", "tool_calls")}}
        content = record["message"].get("content")
        if isinstance(content, str):
            record["message"]["content"] = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        self._trace(record)
        self._availability(model, "available", http_status=200, call_id=call_id)
        return redact(record, self._key)

    def inspect_image(self, path: Path, prompt: str, **options):
        return self.inspect_images([path], prompt, **options)

    def inspect_images(self, paths, prompt: str, **options):
        """Ordered real image parts; multi-image comparison is separately qualified."""
        paths=[Path(p) for p in paths]
        if not 1<=len(paths)<=2 or sum(p.stat().st_size for p in paths)>20*1024**2:
            raise ServiceError('image_limit','Visual requests accept one or two actual images within20MiB total')
        content=[{'type':'text','text':prompt}]
        for path in paths:
            content.append(self._image_part(path))
        return self.chat(VISION,[{'role':'user','content':content}],**options)

    def _image_part(self, path):
        with Image.open(path) as image:
            image.load()
            if image.width * image.height > 32_000_000:
                raise ServiceError("image_limit", "Image exceeds bounded visual payload size")
            mime = Image.MIME.get(image.format)
        if mime not in ("image/png", "image/jpeg") or path.stat().st_size > 20 * 1024**2:
            raise ServiceError("image_format", "Vision accepts bounded real PNG/JPEG files")
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


TOOL = {"type": "function", "function": {"name": "inspect_test_tile", "description": "Read bounded diagnostic tile metadata.",
        "parameters": {"type": "object", "properties": {"tile_id": {"type": "string", "enum": ["diagnostic_tile"]}},
                       "required": ["tile_id"], "additionalProperties": False}}}


def inspect_test_tile(arguments, *, fail=False, nonce="TEST"):
    if not isinstance(arguments, dict) or set(arguments) != {"tile_id"} or arguments["tile_id"] != "diagnostic_tile":
        raise ServiceError("invalid_tool_arguments", "tile_id must equal the registered diagnostic tile")
    if fail:
        return {"status": "failed", "code": "test_missing_material", "retryable": False, "observation_nonce": nonce}
    return {"status": "success", "tile_id": "diagnostic_tile", "observed_ground_gap_mm": 7,
            "observation_nonce": nonce, "qualification": "not_run"}


def probe_native_tools(client: NIMClient):
    messages = [{"role": "user", "content": "Use inspect_test_tile for diagnostic_tile. Do not guess its result. Then report observed_ground_gap_mm and observation_nonce from the tool result."}]
    first = client.chat(PLANNER, messages, tools=[TOOL], tool_choice={"type": "function", "function": {"name": "inspect_test_tile"}})
    message = first["message"]
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        raise ServiceError("native_tools_unavailable", "Planner did not return exactly one native tool call")
    call = calls[0]
    if call.get("function", {}).get("name") != "inspect_test_tile" or not isinstance(call.get("id"), str):
        raise ServiceError("invalid_tool_call", "Planner returned an unregistered tool or missing call ID")
    try:
        args = json.loads(call["function"]["arguments"])
    except (ValueError, KeyError, TypeError):
        raise ServiceError("invalid_tool_arguments", "Native tool arguments were not JSON") from None
    nonce = uuid.uuid4().hex[:12]
    result = inspect_test_tile(args, nonce=nonce)
    next_messages = messages + [message, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}]
    second = client.chat(PLANNER, next_messages, tools=[TOOL])
    content = second["message"].get("content") or ""
    if nonce not in content or "7" not in content:
        raise ServiceError("ungrounded_tool_response", "Planner did not incorporate actual local tool observation")
    fail_nonce = uuid.uuid4().hex[:12]
    fail_result = inspect_test_tile(args, fail=True, nonce=fail_nonce)
    failure_messages = messages + [message, {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(fail_result)}]
    failed = client.chat(PLANNER, failure_messages, tools=[TOOL])
    failure_text = failed["message"].get("content") or ""
    handled = "test_missing_material" in failure_text or any(w in failure_text.lower() for w in ("fail", "missing material", "unable"))
    if not handled:
        raise ServiceError("tool_failure_unhandled", "Planner did not acknowledge local tool failure")
    try:
        inspect_test_tile({"tile_id": "../../etc/passwd"})
    except ServiceError:
        invalid_rejected = True
    else:
        invalid_rejected = False
    return {"status": "pass", "protocol": "native_tool_call", "first_call_id": first["call_id"],
            "result_call_id": second["call_id"], "failure_call_id": failed["call_id"],
            "grounded_nonce": nonce, "invalid_arguments_rejected": invalid_rejected, "tool_failure_acknowledged": handled}


def probe_json_actions(client: NIMClient):
    messages = [{"role": "user", "content": 'Return only JSON {"action":"inspect_test_tile","arguments":{"tile_id":"diagnostic_tile"}}. This is a bounded action request.'}]
    r = client.chat(PLANNER, messages)
    try:
        obj = json.loads(r["message"]["content"])
        if set(obj) != {"action", "arguments"} or obj["action"] != "inspect_test_tile":
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ServiceError("invalid_structured_action", "Model action failed local strict schema") from None
    nonce = uuid.uuid4().hex[:12]
    result = inspect_test_tile(obj["arguments"], nonce=nonce)
    followup = client.chat(PLANNER, messages + [r["message"], {"role": "user", "content": "Local tool result: " + json.dumps(result) + ". Return only JSON with keys gap_mm and nonce copied from result."}])
    try:
        answer = json.loads(followup["message"]["content"])
        valid = answer == {"gap_mm": 7, "nonce": nonce}
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise ServiceError("ungrounded_action_response", "JSON action loop failed to use tool result")
    return {"status": "pass", "protocol": "application_validated_json", "server_enforced_schema": "not_claimed",
            "call_ids": [r["call_id"], followup["call_id"]]}


def probe_nim(workspace: Path, *, recheck_access=False):
    workspace = Path(workspace).resolve()
    report = {"schema_version": 1, "created_at_utc": utcnow(), "planner_model": PLANNER, "vision_model": VISION,
              "allow_model_substitution": False, "official_references": DOCS, "probes": {},
              "critic_isaac_calibration": {"status": "not_run", "reason": "Requires labelled actual Isaac captures; diagnostic image is capability evidence only"},
              "planner_project_scenarios": {"status": "not_run", "required_count": 20}}
    target = workspace / "state/nim_capabilities.json"
    if target.exists():
        previous = json.loads(target.read_text())
        history_id = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
        atomic_json(workspace / "evidence/services/nim_history" / (history_id + ".json"), previous)
    try:
        client = NIMClient(workspace, allow_access_reprobe=recheck_access)
    except ServiceError as e:
        report.update(status="blocked", error=e.record())
        atomic_json(target, report)
        return report
    def run(name, callback):
        try:
            result = callback()
        except ServiceError as e:
            result = {"status": "blocked" if e.code in ("invalid_key", "access_restricted", "model_unavailable", "network") else "fail", "error": e.record()}
        report["probes"][name] = result
        atomic_json(target, report)
        return result
    def text_probe(model):
        r = client.chat(model, [{"role": "user", "content": "Reply with exactly ISAACMIN_OK."}])
        content = r["message"].get("content") or ""
        return {"status": "pass" if "ISAACMIN_OK" in content else "fail", **r}
    planner = run("planner_text", lambda: text_probe(PLANNER))
    vision = run("vision_text", lambda: text_probe(VISION))
    if planner["status"] == "pass":
        run("native_tools", lambda: probe_native_tools(client))
        run("structured_actions", lambda: probe_json_actions(client))
    else:
        report["probes"]["native_tools"] = {"status": "blocked", "reason": "Planner access probe failed"}
        report["probes"]["structured_actions"] = {"status": "blocked", "reason": "Planner access probe failed"}
    report["probes"]["server_json_schema"] = {"status": "not_run", "reason": "Current model-specific hosted reference does not document response_format/json_schema; no server-enforced claim"}
    if vision["status"] == "pass":
        path = workspace / "evidence/services/vision_diagnostic.png"
        image = Image.new("RGB", (384, 256), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 28, 138, 143), fill=(225, 30, 25))
        draw.ellipse((245, 145, 340, 240), fill=(20, 50, 230))
        image.save(path)
        def image_probe():
            r = client.inspect_image(path, "Describe the two colored shapes and their positions in this image. Do not infer unseen objects. Answer briefly.")
            content = (r["message"].get("content") or "").lower()
            grounded = all(w in content for w in ("red", "blue", "left", "right")) and ("square" in content or "rectangle" in content) and "circle" in content
            return {"status": "pass" if grounded else "fail", "diagnostic_sha256": digest(path),
                    "image_size": [384, 256], "payload": "image_url_data_uri_png", "call": r,
                    "qualification_scope": "image capability only; not terrain realism"}
        run("vision_image", image_probe)
    else:
        report["probes"]["vision_image"] = {"status": "blocked", "reason": "Vision access probe failed"}
    report["status"] = "external_tool_verified" if all(report["probes"][k]["status"] == "pass" for k in ("planner_text", "vision_text", "vision_image")) and any(report["probes"][k]["status"] == "pass" for k in ("native_tools", "structured_actions")) else "blocked"
    atomic_json(target, redact(report))
    return report
