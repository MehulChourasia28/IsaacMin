import hashlib

from isaacmin.validation.provenance import scan_delivery_file


def test_delivery_scanner_detects_token_across_binary_chunk_boundary(tmp_path):
    path = tmp_path / "content.usdc"
    # Deliberately noncredential test data; ensure the scanner never returns it.
    data = b"PXR-USDC\x00" + b"nvapi-" + b"f"*48 + b"\x00data"
    path.write_bytes(data)
    result = scan_delivery_file(path, chunk_bytes=13)
    assert result["credential_pattern_found"]
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert b"f"*48 not in repr(result).encode()


def test_delivery_scanner_does_not_treat_documented_variable_name_as_secret(tmp_path):
    path = tmp_path / "instructions.txt"
    path.write_text("Read NVIDIA_API_KEY from a masked prompt. No value is embedded.")
    assert not scan_delivery_file(path)["credential_pattern_found"]
    secret_name = tmp_path / ".env.local"
    secret_name.write_bytes(b"")
    assert scan_delivery_file(secret_name)["forbidden_filename"]
