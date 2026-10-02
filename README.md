# IsaacMin

**Minecraft worlds → realistic Isaac Sim environments. Built by an AI agent.**

Robots learning long-range navigation need far more varied terrain than a handful
of hand-built simulation maps can provide. Creating those environments is slow,
expensive and difficult to scale.

IsaacMin taps into **Minecraft's extraordinary terrain generator as a virtually
unlimited source of simulation worlds**. Mountains, forests, rivers, deserts and
frozen landscapes become detailed, navigable 3D environments in NVIDIA Isaac Sim:
natural terrain, real vegetation, physical materials, ground collision and sensor
captures. Minecraft supplies the geography. IsaacMin brings it into robotics.

| Snowy hillside · Isaac Sim | River reflections · Isaac Sim |
| --- | --- |
| ![Snowy hillside and trees rendered in Isaac Sim](docs/media/mountain-slope.png) | ![Riverbank and water reflections rendered in Isaac Sim](docs/media/riverbank.png) |

## An agent that does the work

Powered by **NVIDIA Nemotron on OpenClaw**, the IsaacMin agent turns one request
into a long-running world-building job. It uses specialised tools to inspect
worlds, understand job state, initiate conversions, verify outputs, prepare
downloads and create new camera views. It selects tools and arguments from your
request and the evidence those tools return.

Once a conversion starts, the agent drives terrain reconstruction, scene assembly,
collision generation, Isaac rendering and delivery **without human intervention
between stages**. Durable jobs and checkpoints keep progress independent of the
browser and let interrupted work resume without throwing away completed stages.

Ask it to **“Create this world,” “Show me three aerial views,”** or **“Resume my
interrupted conversion.”** It acts through tools and returns real job results.

## The demo

Choose from four Minecraft presets, watch the world take shape, compare Minecraft
and Isaac side by side, then download the portable USD world. Request more ground
or aerial views whenever you need them.

![Minecraft and generated Isaac world side by side in the studio](docs/media/world-comparison.png)

**Built on DGX Spark / GB10 / Linux ARM64**, with HighMap, Blender, Isaac Sim,
OpenClaw and NVIDIA-hosted Nemotron. [Setup](docs/SETUP.md) ·
[Agent tools](docs/AGENT.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Asset credits](ASSET_CREDITS.md)

*Hackathon scope: four 256 m demo regions. Continuous kilometre-scale delivery and
full navigation/realism qualification remain future work. Source saves, native
runtimes and large asset libraries are provisioned separately from this source repo.*
