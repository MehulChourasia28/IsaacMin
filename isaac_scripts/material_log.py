"""Reject observed native material compilation/import errors before accepting RGB."""
import hashlib
import json
from pathlib import Path


def material_errors(text):
    records = []
    for line in text.splitlines():
        error = any(level in line for level in ('[Error]', '[Fatal]'))
        material = any(name in line for name in ('MDLC:COMPILER', 'USD_MDL:',
            '[omni.usdMdl]', '[omni.rtx.materials]', '[rtx.neuraylib.plugin]'))
        missing = any(message in line.lower() for message in (
            'failed to load texture', 'unable to load texture', 'could not load texture',
            "texture file referenced in the material body wasn't resolved properly"))
        geometry_failure = ('[rtx.scenedb.plugin]' in line and error) or (
            '[rtx.hydra]' in line and 'Unable to create ' in line and 'instances of geometry' in line)
        if (error and material) or missing or geometry_failure:
            records.append(line)
    return records


class MaterialLogGuard:
    def __init__(self, log_path, output_path):
        self.log = Path(log_path)
        self.output = Path(output_path)
        self.offset = 0
        self.partial = b''
        self.digest = hashlib.sha256()
        self.errors = []

    def check(self):
        if not self.log.is_file():
            raise RuntimeError('Native material log is missing; import cannot be checked')
        if self.log.stat().st_size < self.offset:
            raise RuntimeError('Native material log was truncated during capture')
        with self.log.open('rb') as stream:
            stream.seek(self.offset)
            while data := stream.read(65536):
                self.offset += len(data)
                self.digest.update(data)
                lines = (self.partial + data).split(b'\n')
                self.partial = lines.pop()
                found = material_errors(b'\n'.join(lines).decode('utf8', errors='replace'))
                self.errors.extend(found[:max(0, 1000-len(self.errors))])
        result = {'status': 'fail' if self.errors else 'no_observed_native_material_errors',
            'checked_log_prefix_bytes': self.offset, 'checked_log_prefix_sha256': self.digest.hexdigest(),
            'errors': self.errors, 'errors_recorded_limit': 1000,
            'scope': 'Native log diagnostics only; successful compilation does not qualify appearance'}
        self.output.write_text(json.dumps(result, indent=2) + '\n')
        if self.errors:
            raise RuntimeError('Native material compilation/import failed; see ' + str(self.output))
        return result
