"""Bounded, resumable downloads. No NVIDIA credentials enter this module."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import stat
import time
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, retryable: bool = False):
        self.code, self.retryable = code, retryable
        super().__init__(message)

    def record(self):
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def atomic_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def retry_delay(value: str | None, attempt: int) -> float:
    if value:
        try:
            return max(0, float(value))
        except ValueError:
            try:
                return max(0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    return min(32, 2 ** attempt) + random.random()


def safe_relative(name: str) -> Path:
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or ".." in p.parts or ":" in name or "\x00" in name:
        raise ServiceError("unsafe_path", "Unsafe provider/archive relative path")
    return Path(*p.parts)


def validate_url(url: str, hosts: tuple[str, ...]):
    p = urlsplit(url)
    if p.scheme != "https" or p.username or p.password or p.port not in (None, 443):
        raise ServiceError("unsafe_url", "Only credential-free HTTPS provider URLs are permitted")
    if not p.hostname or not any(p.hostname == h or p.hostname.endswith("." + h) for h in hosts):
        raise ServiceError("unsafe_url", "URL host is outside this provider's allowlist")
    return url


class PublicClient:
    """Metadata requests serialized by the caller; transfers are independently resumable."""
    def __init__(self, cache: Path, hosts: tuple[str, ...], attempts=5, budget_bytes=30 * 1024**3):
        self.cache, self.hosts, self.attempts = Path(cache), hosts, attempts
        self.budget_bytes, self.downloaded = budget_bytes, 0
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client = httpx.Client(headers={"User-Agent": "IsaacMin/0.1", "Accept-Encoding": "identity"},
                                   timeout=httpx.Timeout(120, connect=15), follow_redirects=False)

    def _request(self, method, url, **kwargs):
        for _ in range(6):
            validate_url(url, self.hosts)
            r = self.client.request(method, url, **kwargs)
            if r.status_code in (301, 302, 303, 307, 308):
                if "location" not in r.headers:
                    raise ServiceError("schema_drift", "Redirect lacks location")
                url = str(r.url.join(r.headers["location"]))
                continue
            return r
        raise ServiceError("redirect_loop", "Too many provider redirects")

    def metadata(self, url: str, *, refresh=False):
        validate_url(url, self.hosts)
        cache_path = self.cache / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if cache_path.exists() and not refresh:
            obj = json.loads(cache_path.read_text())
            if time.time() - obj["retrieved_timestamp"] < 86400:
                return obj["data"], obj
        for attempt in range(self.attempts):
            try:
                r = self._request("GET", url)
                if r.status_code == 429 or r.status_code >= 500:
                    delay = retry_delay(r.headers.get("retry-after"), attempt)
                    if delay > 60:
                        raise ServiceError("rate_limited", f"Provider retry is deferred for {delay:.0f} seconds", True)
                    if attempt + 1 < self.attempts:
                        time.sleep(delay)
                        continue
                if r.status_code != 200:
                    raise ServiceError("provider_http", f"Public provider returned HTTP {r.status_code}", r.status_code >= 500 or r.status_code == 429)
                if len(r.content) > 32 * 1024**2:
                    raise ServiceError("resource_limit", "Metadata response exceeds 32 MiB")
                try:
                    data = r.json()
                except ValueError as e:
                    raise ServiceError("schema_drift", "Provider response is not JSON") from e
                obj = {"url": url, "retrieved_at_utc": utcnow(), "retrieved_timestamp": time.time(),
                       "response_sha256": hashlib.sha256(r.content).hexdigest(), "data": data}
                atomic_json(cache_path, obj)
                return data, obj
            except httpx.HTTPError:
                if attempt + 1 == self.attempts:
                    raise ServiceError("network", "Public provider connection failed", True) from None
                time.sleep(retry_delay(None, attempt))
        raise ServiceError("provider_unavailable", "Provider retries exhausted", True)

    def download(self, file: dict, target: Path):
        url, expected = file.get("url"), file.get("size")
        if not isinstance(url, str) or type(expected) is not int or expected <= 0:
            raise ServiceError("schema_drift", "Download requires URL and positive integer size")
        validate_url(url, self.hosts)
        if expected > self.budget_bytes or self.downloaded + expected > self.budget_bytes:
            raise ServiceError("resource_limit", "Initial binary budget exhausted; acquire after target qualification")
        target.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(target.parent).free - expected < shutil.disk_usage(target.parent).total * 0.2:
            raise ServiceError("resource_limit", "Download would violate 20 percent disk reserve")
        def verify(path):
            if path.stat().st_size != expected:
                raise ServiceError("corrupt_download", "Downloaded size does not match provider manifest")
            if file.get("md5") and digest(path, "md5") != file["md5"]:
                raise ServiceError("corrupt_download", "Downloaded MD5 does not match provider manifest")
            return digest(path)
        if target.exists():
            sha = verify(target)
            meta_path = target.with_suffix(target.suffix + ".download.json")
            if meta_path.exists():
                previous = json.loads(meta_path.read_text())
                if previous.get("sha256") != sha or previous.get("source_url") != url:
                    raise ServiceError("corrupt_download", "Cached asset identity changed")
            elif not file.get("md5"):
                raise ServiceError("unvalidated_cache", "Cached bytes lack upstream checksum and prior local provenance")
            return {"path": str(target), "size_bytes": expected, "sha256": sha,
                    "source_url": url, "cache_hit": True, "md5": file.get("md5")}
        part = target.with_suffix(target.suffix + ".part")
        state = part.with_suffix(part.suffix + ".json")
        identity = {k: file.get(k) for k in ("url", "size", "md5")}
        if state.exists() and json.loads(state.read_text()).get("identity") != identity:
            raise ServiceError("stale_partial", "Partial download belongs to another upstream revision")
        partial_state = json.loads(state.read_text()) if state.exists() else {"identity": identity}
        atomic_json(state, partial_state)
        resumed_bytes = 0
        for attempt in range(self.attempts):
            offset = part.stat().st_size if part.exists() else 0
            if offset == expected:
                break
            if offset > expected:
                raise ServiceError("corrupt_download", "Partial file exceeds declared size")
            validator = partial_state.get("etag") or partial_state.get("last_modified")
            if offset and not file.get("md5") and not validator:
                # No integrity mechanism can prove an unvalidated suffix shares the prefix's revision.
                part.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            if offset and validator:
                headers["If-Range"] = validator
            try:
                current_url = url
                for redirect in range(6):
                    validate_url(current_url, self.hosts)
                    with self.client.stream("GET", current_url, headers=headers) as r:
                        if r.status_code in (301, 302, 303, 307, 308):
                            current_url = str(r.url.join(r.headers["location"]))
                            continue
                        if r.status_code == 429 or r.status_code >= 500:
                            delay = retry_delay(r.headers.get("retry-after"), attempt)
                            if delay > 60:
                                raise ServiceError("rate_limited", f"Provider asks retry after {delay:.0f} seconds", True)
                            time.sleep(delay)
                            break
                        if r.status_code not in (200, 206):
                            raise ServiceError("provider_http", f"Asset download returned HTTP {r.status_code}")
                        mode = "wb"
                        if r.status_code == 206:
                            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("content-range", ""))
                            if not match or int(match[1]) != offset or int(match[3]) != expected:
                                raise ServiceError("invalid_range", "Provider range response does not match partial file")
                            mode = "ab"
                            resumed_bytes = max(resumed_bytes, offset)
                        else:
                            offset = 0
                        partial_state.update(etag=r.headers.get("etag"), last_modified=r.headers.get("last-modified"))
                        atomic_json(state, partial_state)
                        with part.open(mode) as f:
                            for block in r.iter_bytes(1024 * 1024):
                                offset += len(block)
                                if offset > expected:
                                    raise ServiceError("corrupt_download", "Response exceeds declared asset size")
                                f.write(block)
                            f.flush()
                            os.fsync(f.fileno())
                        break
                else:
                    raise ServiceError("redirect_loop", "Too many binary redirects")
                if part.exists() and part.stat().st_size == expected:
                    break
            except httpx.HTTPError:
                if attempt + 1 == self.attempts:
                    raise ServiceError("network", "Asset connection interrupted; verified partial retained", True) from None
                time.sleep(retry_delay(None, attempt))
        if not part.exists() or part.stat().st_size != expected:
            raise ServiceError("incomplete_download", "Partial asset retained for retry", True)
        try:
            sha = verify(part)
        except ServiceError:
            quarantine = part.with_name(part.name + ".corrupt-" + digest(part)[:12])
            os.replace(part, quarantine)
            raise
        os.replace(part, target)
        state.unlink(missing_ok=True)
        self.downloaded += expected
        record = {"path": str(target), "size_bytes": expected, "sha256": sha, "md5": file.get("md5"),
                  "source_url": url, "retrieved_at_utc": utcnow(), "resumed_bytes": resumed_bytes, "cache_hit": False}
        atomic_json(target.with_suffix(target.suffix + ".download.json"), record)
        return record


def extract_zip(archive: Path, destination: Path, max_bytes=4 * 1024**3):
    """Validate every member before writing. Extract only to a fresh staging directory."""
    if destination.exists():
        raise ServiceError("existing_destination", "Extraction destination already exists")
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    forbidden = {".py", ".sh", ".bat", ".cmd", ".ps1", ".exe", ".dll", ".so", ".js", ".msi"}
    try:
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            if len(infos) > 10000 or sum(i.file_size for i in infos) > max_bytes:
                raise ServiceError("archive_limit", "ZIP expansion exceeds bounded limits")
            names = set()
            for i in infos:
                p = safe_relative(i.filename)
                mode = i.external_attr >> 16
                if stat.S_ISLNK(mode) or (mode & 0o111 and not i.is_dir()) or p.suffix.lower() in forbidden:
                    raise ServiceError("unsafe_archive", "Archive contains executable or symbolic-link entry")
                if str(p).casefold() in names:
                    raise ServiceError("unsafe_archive", "Archive contains duplicate member paths")
                names.add(str(p).casefold())
            staging.mkdir(parents=True)
            for i in infos:
                p = staging / safe_relative(i.filename)
                if i.is_dir():
                    p.mkdir(parents=True, exist_ok=True)
                else:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(i) as src, p.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
            records = [{"path": str(p.relative_to(staging)), "sha256": digest(p), "size_bytes": p.stat().st_size}
                       for p in sorted(staging.rglob("*")) if p.is_file()]
        os.replace(staging, destination)
        return records
    except (zipfile.BadZipFile, RuntimeError) as e:
        if isinstance(e, ServiceError):
            raise
        raise ServiceError("corrupt_archive", "ZIP failed CRC/decompression validation") from None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
