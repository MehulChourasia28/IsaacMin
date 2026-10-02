from pathlib import Path
import hashlib
import json
import stat
import zipfile

import httpx
import pytest

from isaacmin.assets.network import PublicClient, ServiceError, extract_zip, safe_relative, validate_url
from isaacmin.assets.providers import parse_ambient, parse_poly_catalogue, poly_variant
from isaacmin.adapters.nim import inspect_test_tile, load_key, redact, NIMClient, PLANNER


def test_real_provider_fixtures_have_supported_schema():
    root = Path(__file__).resolve().parents[1] / "evidence/services/provider_fixtures"
    if not root.exists():
        pytest.skip("Live fixtures are acquired by services bootstrap")
    models = parse_poly_catalogue(json.loads((root / "poly_models.json").read_text()))
    assert models["pine_sapling_small"]["type"] == 2
    ambient = parse_ambient(json.loads((root / "ambient_ground.json").read_text()))
    assert any(a["downloads"] for a in ambient)
    f = json.loads((root / "poly_files_pine_sapling_small.json").read_text())
    assert poly_variant(f, "blend", "1k", "blend")["include"]


@pytest.mark.parametrize("payload", [{}, {"totalResults": 2, "assets": [{}]},
    {"totalResults": 1, "assets": [{"id": "Good", "downloads": {"1K": {}}}]}])
def test_ambient_schema_drift_is_not_empty_catalogue(payload):
    with pytest.raises(ServiceError, match="changed|array"):
        parse_ambient(payload)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../../escape", "C:/evil", "a\\evil"])
def test_relative_paths_reject_traversal(name):
    with pytest.raises(ServiceError):
        safe_relative(name)


def test_archive_checks_entire_archive_before_extracting(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("innocent.png", b"123")
        f.writestr("../outside", b"bad")
    with pytest.raises(ServiceError):
        extract_zip(z, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "outside").exists()


def test_archive_symlink_and_executable_rejected(tmp_path):
    for name in ("run.py", "link"):
        z = tmp_path / (name + ".zip")
        with zipfile.ZipFile(z, "w") as f:
            entry = zipfile.ZipInfo(name)
            if name == "link":
                entry.create_system = 3
                entry.external_attr = (stat.S_IFLNK | 0o777) << 16
            f.writestr(entry, b"../x")
        with pytest.raises(ServiceError):
            extract_zip(z, tmp_path / name)


def test_download_rejects_same_size_corruption_and_no_promotion(tmp_path):
    expected = b"realasset"
    bad = b"badassets"
    c = PublicClient(tmp_path / "cache", ("assets.example",), attempts=1)
    c.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=bad)))
    with pytest.raises(ServiceError) as failure:
        c.download({"url": "https://assets.example/data", "size": len(expected),
                    "md5": hashlib.md5(expected).hexdigest()}, tmp_path / "asset.bin")
    assert failure.value.code == "corrupt_download"
    assert not (tmp_path / "asset.bin").exists()


def test_resume_validates_content_range(tmp_path):
    full = b"abcdefgh"
    target = tmp_path / "asset.bin"
    target.with_suffix(".bin.part").write_bytes(full[:3])
    c = PublicClient(tmp_path / "cache", ("assets.example",), attempts=1)
    c.client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(206,
        headers={"Content-Range": "bytes 4-7/8"}, content=full[4:])))
    with pytest.raises(ServiceError) as failure:
        c.download({"url": "https://assets.example/data", "size": 8, "md5": hashlib.md5(full).hexdigest()}, target)
    assert failure.value.code == "invalid_range"
    assert not target.exists()


def test_successful_resume_preserves_full_file_hash(tmp_path):
    full = b"abcdefgh"
    target = tmp_path / "asset.bin"
    target.with_suffix(".bin.part").write_bytes(full[:3])
    def handle(request):
        assert request.headers["Range"] == "bytes=3-"
        return httpx.Response(206, headers={"Content-Range": "bytes 3-7/8"}, content=full[3:])
    c = PublicClient(tmp_path / "cache", ("assets.example",), attempts=1)
    c.client = httpx.Client(transport=httpx.MockTransport(handle))
    r = c.download({"url": "https://assets.example/data", "size": 8, "md5": hashlib.md5(full).hexdigest()}, target)
    assert target.read_bytes() == full
    assert r["resumed_bytes"] == 3
    assert r["sha256"] == hashlib.sha256(full).hexdigest()


def test_cached_metadata_does_not_hide_expiry(tmp_path):
    c = PublicClient(tmp_path, ("assets.example",), attempts=1)
    calls = []
    def handle(r):
        calls.append(r)
        return httpx.Response(200, json={"live": len(calls)})
    c.client = httpx.Client(transport=httpx.MockTransport(handle))
    url = "https://assets.example/catalogue"
    assert c.metadata(url)[0]["live"] == 1
    assert c.metadata(url)[0]["live"] == 1
    file = next(tmp_path.glob("*.json"))
    entry = json.loads(file.read_text())
    entry["retrieved_timestamp"] = 0
    file.write_text(json.dumps(entry))
    assert c.metadata(url)[0]["live"] == 2


@pytest.mark.parametrize("url", ["http://assets.example/x", "https://evil.example/x", "https://user:password@assets.example/x", "https://assets.example.evil/x"])
def test_provider_urls_never_send_data_to_other_hosts(url):
    with pytest.raises(ServiceError):
        validate_url(url, ("assets.example",))


def test_secret_redaction_and_protected_loading(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    fake = "nvapi-diagnostic-not-a-real-key"
    p = tmp_path / ".env"
    p.write_text("NVIDIA_API_KEY=" + fake + "\n")
    p.chmod(0o644)
    with pytest.raises(ServiceError):
        load_key(tmp_path)
    p.chmod(0o600)
    assert load_key(tmp_path) == fake
    sanitized = json.dumps(redact({"Authorization": "Bearer " + fake, "failure": fake, "reasoning_content": "hidden"}, fake))
    assert fake not in sanitized and "hidden" not in sanitized


@pytest.mark.parametrize("args", [{"tile_id": "../x"}, {"tile_id": "diagnostic_tile", "shell": "true"}, {}, []])
def test_bounded_native_tool_rejects_invalid_arguments(args):
    with pytest.raises(ServiceError):
        inspect_test_tile(args)


def test_persistent_access_denial_does_not_retry_network(tmp_path,monkeypatch):
    monkeypatch.setenv('NVIDIA_API_KEY','nvapi-diagnostic-not-a-real-key')
    state=tmp_path/'state';state.mkdir()
    (state/'nim_availability.json').write_text(json.dumps({'models':{PLANNER:{'code':'access_restricted'}}}))
    c=NIMClient(tmp_path)
    c.client=httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('Denied access was retried')))
    with pytest.raises(ServiceError) as error:
        c.chat(PLANNER,[{'role':'user','content':'hello'}])
    assert error.value.code=='access_restricted'


def test_key_cannot_be_accidentally_uploaded_as_prompt(tmp_path,monkeypatch):
    fake='nvapi-diagnostic-not-a-real-key'
    monkeypatch.setenv('NVIDIA_API_KEY',fake)
    c=NIMClient(tmp_path)
    c.client=httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail('Secret payload reached network')))
    with pytest.raises(ServiceError) as error:
        c.chat(PLANNER,[{'role':'user','content':fake}])
    assert error.value.code=='secret_in_payload'
