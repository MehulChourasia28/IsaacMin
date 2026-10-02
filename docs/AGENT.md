# The IsaacMin agent

The live demo uses **OpenClaw 2026.9.7** with NVIDIA-hosted
`nvidia/nemotron-3-super-120b-a12b`. The studio assistant and world-conversion
dispatch both run through this harness. NemoClaw is not installed in this build.

The assistant interprets a request, selects tools, reads their results and takes
the requested action. It can inspect geography, materials, source identity,
captures, download readiness, progress and recorded failures. Once a conversion
is submitted, persistent workers carry it through to delivery without keeping
the browser open or requiring approval at each stage.

| Tools | What the agent can do |
| --- | --- |
| `list_presets`, `inspect_preset` | Discover and compare the registered worlds |
| `inspect_world` | Read source, terrain, water, asset, capture, qualification and delivery evidence |
| `get_capabilities` | Explain supported actions, camera conventions, timing and limits |
| `list_jobs`, `inspect_job` | Check actual progress, errors and completed results |
| `start_job` | Start a conversion, verify a saved preset or prepare its download |
| `request_views` | Create 1–6 ground/aerial images with optional coordinates, heading and height |
| `control_job` | Resume eligible interrupted/failed work or cancel a queued job |
| `run_pipeline_stage` | Dispatch the next admitted construction operation during a conversion |

## Long-running execution

OpenClaw's tool loop connects to a project-local adapter. NVIDIA credentials stay
in the adapter; the browser, OpenClaw process and native asset workers do not
receive the provider key. Native tools run in separate processes. Persistent jobs,
tool receipts and dependency-bound checkpoints preserve progress across browser
closures and support resumption after interruption.

The construction controller checks stage order and checkpoint identity before
accepting a dispatch. A stage finishes only when its native completion receipt
exists. Tool schemas constrain the agent's actions; it cannot rewrite quality
settings, alter a source save or execute arbitrary shell commands.

## Example requests

- “Create a new Isaac world from this preset.”
- “Which biomes and materials are in my latest conversion?”
- “Create three aerial views of this world.”
- “Capture one aerial at 180 metres, heading 120 degrees.”
- “Why did this job stop? Resume it.”
- “Is the world ready to download?”

New views reuse the existing scene. Ground views use a 0.6 m camera height;
aerial height is configurable from 20–600 m. Camera requests are checked against
the constructed region and supported ground locations.

Actions require a request; questions alone do not launch expensive jobs. The
agent does not inspect image pixels, perform automatic visual repair, register
arbitrary uploads, edit appearance or drive a robot. Errors remain visible and
recovery is bounded. Each assistant request is independent; persisted job state
is not unrestricted conversational memory.
