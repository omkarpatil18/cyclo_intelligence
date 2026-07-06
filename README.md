# Cyclo Intelligence

Open-source full-stack Physical AI platform — data recording, conversion,
training, inference, and robot execution in a single repository.

For detailed usage and tutorials, please refer to the documentation below.
  - [Documentation for AI Worker](https://ai.robotis.com/)

## Clone

```bash
git clone --recurse-submodules https://github.com/ROBOTIS-GIT/cyclo_intelligence.git
cd cyclo_intelligence
```

### Updating an existing checkout

After switching branches or pulling a new release, sync the pinned
submodule commits from the repository root:

```bash
git pull --ff-only
git submodule update --init --recursive
```

## Folders at a glance

| Folder | Role |
| --- | --- |
| [`shared/`](shared/) | Robot configs, IO helpers, logger |
| [`cyclo_brain/`](cyclo_brain/) | Training + inference (per-backend containers under `policy/`) |
| [`cyclo_data/`](cyclo_data/) | Data recording / conversion / hub upload (ROS2 node) |
| [`orchestrator/`](orchestrator/) | Session state, UI, behaviour-tree control (ships React UI) |
| [`interfaces/`](interfaces/) | ROS2 msg / srv definitions |
| [`docker/`](docker/) | Unified compose, s6-services, Dockerfiles (arm64 / amd64) |
| [`docs/`](docs/) | Architecture |

## Prerequisites

- Docker 24+ with the Compose plugin (`docker compose version`)
- NVIDIA Container Toolkit configured as the default runtime
  (`docker info | grep "Default Runtime: nvidia"`)
- ~35 GB free disk for the three pre-built images
- Ports 80, 8100, 9090 free on the host

No Docker Hub login is required — the published images
(`robotis/cyclo-intelligence`, `robotis/groot-zenoh`,
`robotis/lerobot-zenoh`) are public and pulled anonymously by default.

## Quick start (Jetson / ARM64 — same on AMD64)

```bash
docker/container.sh start          # pull + start unified image (no rebuild)
docker/container.sh status         # check s6 service state
# UI:          http://localhost/
# control API: http://localhost/api/health
docker/container.sh start-lerobot  # policy on demand (LOAD via UI)
docker/container.sh start-groot    # policy on demand (LOAD via UI)
docker/container.sh stop           # tear everything down
```

`docker/container.sh` auto-detects `uname -m`, so the same commands work on
both Jetson and an AMD64 workstation. The default `start*` flow uses the
pre-built image from Docker Hub. Pass `--build` to rebuild from local
Dockerfile (only needed when iterating on Dockerfile changes).

### ROS/Zenoh runtime env

Runtime ROS/Zenoh settings are shared through:

```bash
docker/workspace/config/ros_zenoh.env
```

`container.sh start*` creates this file from the shared default
[`docker/config/ros_zenoh.default.env`](docker/config/ros_zenoh.default.env) if
it is missing. Edit the workspace runtime file when `ROS_DOMAIN_ID` or the
Zenoh router IP changes. s6-managed services and interactive shells source the
same file so manual ROS commands and policy servers use one configuration
source.

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for runtime topology
and data flow.

## Submodules (pinned commit)

- `cyclo_brain/sdk/zenoh_ros2_sdk/` ← [ROBOTIS-GIT/zenoh_ros2_sdk](https://github.com/ROBOTIS-GIT/zenoh_ros2_sdk)
- `cyclo_brain/policy/lerobot/lerobot/` ← [huggingface/lerobot](https://github.com/huggingface/lerobot)
- `cyclo_brain/policy/groot/Isaac-GR00T/` ← [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)

These paths are pinned to exact commits by the parent repository. If a
submodule directory is empty or stale, run
`git submodule update --init --recursive` before building images.

## Related

  - [AI Worker ROS 2 Packages](https://github.com/ROBOTIS-GIT/ai_worker)
  - [Simulation Models](https://github.com/ROBOTIS-GIT/robotis_mujoco_menagerie)
  - [Tutorial Videos](https://www.youtube.com/@ROBOTISOpenSourceTeam)
  - [AI Models & Datasets](https://huggingface.co/ROBOTIS)
  - [Docker Images](https://hub.docker.com/r/robotis/ros/tags)

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
