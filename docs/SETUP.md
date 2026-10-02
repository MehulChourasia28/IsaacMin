# Running IsaacMin

**Target:** NVIDIA DGX Spark / GB10, Linux ARM64, 128 GB shared CPU/GPU memory.
The Python application requires Python 3.12+. Native construction and rendering
use separately installed HighMap, Blender, OpenUSD and Isaac Sim workers.

## On the provisioned demo machine

From the project directory:

```bash
env -u NVIDIA_API_KEY .venv/bin/python -m isaacmin demo --port 8765
```

Open **http://127.0.0.1:8765**. Choose a Minecraft preset and select **Create Isaac
world**. Saved worlds → **View result** shows Minecraft/Isaac comparisons, view
filters, additional capture controls and the portable download.

Supply the NVIDIA key through `NVIDIA_API_KEY` or run the masked local prompt:

```bash
.venv/bin/python -m isaacmin secret
```

The prompt creates a private `.env` with mode `0600`. Never put a key in source
code or a command argument. The server binds to loopback; browsing saved images
and downloading a completed world do not require a running Isaac process.

## From a source checkout

This repository contains the application, agent integration, workers, recipes,
locks, tests and selected demo images. Large local runtime data is intentionally
not in Git. A fresh clone does **not** contain the provisioned four-preset demo.

Install the Python application for development:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m isaacmin --help
```

For native operation, provision the following local components:

| Component | Location / tooling |
| --- | --- |
| Original Java saves and preset registration | `state/source_worlds.json`, `state/demo_presets.json`; preset metadata is in [`configs/demo_presets.example.json`](../configs/demo_presets.example.json) |
| Qualified native worker installations | `.tools/`; `scripts/bootstrap_native.py`, `scripts/bootstrap_precision.py` and `scripts/run_native_compatibility.py` |
| OpenClaw and its private Node runtime | `.venv/bin/python scripts/bootstrap_openclaw.py`; locks in `configs/openclaw/` |
| NVIDIA planner selection and access qualification | `state/planner_selection.json`, `evidence/demo/planner_capability.json` |
| Licensed assets and prepared USD libraries | `assets/`, selected `state/surface_asset_library*.json`, their dependency-closed library directories |
| Preset scenes, Chunky previews and capture receipts | Registered local build paths and `artifacts/demo/` |

The bootstrap scripts support individual setup tasks; they do not yet reconstruct
the complete demo workspace from this Git repository alone. Existing native
installations, saved worlds and runtime registrations are preserved on the demo
machine. Full clone-to-demo provisioning is a separate packaging task.

With the native runtime and an asset library provisioned, the exterior CLI is:

```bash
.venv/bin/python -m isaacmin build-outdoor \
  --world /path/to/java-save --center X Z --extent 256 --json
```

Omit `--center` to use the matching registered centre or the save's spawn. Input
saves are read-only. Package a completed build with `package-outdoor --build PATH`.
Extract a downloaded world ZIP in full and open `scene/world.usda` in the pinned
Isaac runtime, keeping its textures, materials and other dependencies together.

## Tests

```bash
.venv/bin/python -m pytest -q
```

Native tests require their isolated tools; unavailable native checks skip
explicitly. Fixture tests do not qualify rendered-world realism or robot contact.
