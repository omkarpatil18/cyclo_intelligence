# Cyclo Intelligence — Codebase Notes

Working notes distilled from a full read of this repo and its sister repos
(`ai_worker`, `cyclo_control`, `physical_ai_tools`, `cyclo_lab`) on 2026-08-24,
at commit `9c2ac91` (Cyclo 1.3.0, LeRobot 0.6.1). Line numbers are from that
commit and will drift.

---

## 1. One-paragraph summary

Cyclo Intelligence is ROBOTIS' Physical-AI platform for the AI Worker (FFW)
robots: teleop recording → LeRobot dataset → policy training → policy inference
→ robot execution. It runs as Docker containers supervised by s6-overlay, with
ROS 2 Jazzy over `rmw_zenoh_cpp` (`ROS_DOMAIN_ID=30`) on the host and a
pure-Python `zenoh_ros2_sdk` inside GPU policy containers. Everything couples
to the robot through a **topic contract** declared in
`shared/shared/robot_configs/<robot>_config.yaml`; nothing in the robot-side
repos knows this platform exists.

## 2. System topology

```
 ┌──────────────── cyclo_intelligence ─────────────────────────────────────┐
 │  UI (React, :7080) ──rosbridge :7090──▶ orchestrator (ROS2 node)        │
 │        │ HTTP /api                     │  /task/command, /training/*    │
 │        ▼                               ├──▶ cyclo_data (record/convert) │
 │  supervisor_api (FastAPI :7100)        ├──▶ bt_node (behavior trees)    │
 │   s6 services + docker backends        └──▶ policy container via zenoh │
 │                                             /<backend>/inference_command│
 │  ┌─ policy container (lerobot | groot) ────────────────────────────┐   │
 │  │ main_runtime:  ServiceHandler → ControlLoop → ActionChunkProcessor│   │
 │  │      │ /<backend>/engine_command (EngineCommand.srv, seq_id)     │   │
 │  │ engine_process: EngineWorker → InferenceEngine backend           │   │
 │  │ RobotClient (zenoh_ros2_sdk) ── publishes ──┐                     │   │
 │  └─────────────────────────────────────────────┼─────────────────────┘   │
 └────────────────────────────────────────────────┼─────────────────────────┘
        topic contract: /leader/*/joint_trajectory, /cmd_vel, /joint_states, cameras
 ┌── ai_worker (FFW robot) ───────────┐   ┌── cyclo_control ──────────────┐
 │ ros2_control: arm_l/r, head, lift, │   │ QP differential-IK (OSQP)     │
 │ swerve_drive (/cmd_vel, 1 s t/o)   │◀──│ publishes same /leader topics │
 │ leader arm broadcaster (teleop)    │   │ (alternative command source)  │
 └────────────────────────────────────┘   └───────────────────────────────┘
        cyclo_lab (Isaac Lab) impersonates the robot on the same topics over DDS.
        physical_ai_tools = the monolithic predecessor (same service names).
```

**Key design fact:** teleop leader, `cyclo_control`, `cyclo_lab`, and the AI
policy are all *alternative publishers* on the same `/leader/...` topics.
There is no mode-switch topic — whoever publishes wins.

## 3. Repository layout

| Path | Role |
|---|---|
| `orchestrator/` | Control-plane ROS2 node, behavior-tree runtime, React UI, launch files |
| `cyclo_data/` | Data-plane ROS2 node: recording, conversion, HF hub, editing, replay server |
| `shared/` | Per-robot YAML configs + `schema.py`, URDF/meshes, atomic file IO |
| `interfaces/` | `msg/` and `srv/` definitions (ament_cmake) |
| `cyclo_brain/sdk/` | `zenoh_ros2_sdk` (submodule), `robot_client`, `action_chunk_processing` |
| `cyclo_brain/policy/` | `common/runtime` (Main + Engine processes), `lerobot/`, `groot/` backends |
| `cyclo_brain/docs/architecture.html` | Visual diagram of the policy-container runtime |
| `docker/` | compose, s6 services, `supervisor_api`, `container.sh`, zenoh cache seeding |
| `install.sh` | Clones with submodules; on `ffw*` hosts installs to `/mnt/ssd` + fstab bind mount |

Submodules: `cyclo_brain/policy/lerobot/lerobot`, `cyclo_brain/policy/groot/Isaac-GR00T`,
`cyclo_brain/sdk/zenoh_ros2_sdk`.

## 4. Control plane — `orchestrator/`

- `orchestrator/orchestrator/orchestrator_node.py:92` `OrchestratorNode` (~2900 lines,
  `MultiThreadedExecutor`). Owns session state (`on_recording` / `on_inference`),
  robot-type params, HF user/endpoints, training lifecycle, policy-container dispatch.
  Three locks (`_state_lock`, `_recording_command_lock`, `_inference_lifecycle_lock`)
  and split callback groups avoid service-in-service deadlock.
- **Services hosted** (`_init_ros_service`, `:530`): `/task/command` (`SendCommand`, 25
  commands), `/get_robot_types`, `/get_robot_info`, `/set_robot_type`, `/register_hf_user`,
  `/get_registered_hf_user`, `/huggingface/list_endpoints|select_endpoint`,
  `/training/command`, `/training/get_*`, `/replay/get_data`, `/browse_file`.
  Publishes `/training/status`.
- `internal/communication/communicator.py:48` — builds the topic inventory from the robot
  YAML; publishes `/task/inference_status` (`InferenceStatus`) and `heartbeat`; subscribes
  the leader joystick `tact_trigger` for hands-free record start/stop.
- `internal/communication/cyclo_data_client.py:57` — clients for `/data/recording`,
  `/data/convert`, `/data/convert/status`, `/data/hub`, `/data/edit`; subscribes `/data/status`.
- `internal/communication/container_service_client.py:118` — prefix-parameterised policy
  container client: `/{prefix}/inference_command|train|stop|status`, topic `/{prefix}/progress`.
  Prefix chosen by `_determine_service_prefix` (`orchestrator_node.py:2515`):
  `TaskInfo.service_type` → `policy_path/config.json` type → default `/groot`.
- `internal/communication/inference_mode.py` — maps `TaskInfo.inference_mode` to
  `publish_to_robot` (the dry-run safety gate).
- Training: `training/zenoh_training_manager.py:36` wraps `ContainerServiceClient`;
  `user_training_interaction_callback` (`orchestrator_node.py:899`) spawns a thread and a
  0.5 s timer republishes `/training/status`.
- **Behavior trees** (`orchestrator/orchestrator/bt/`): `bt_node.py:48` ticks at 30 Hz;
  services `/bt/set_running`, `/bt/load_and_run`, `/bt/nodes/catalog`; topics `/bt/status`,
  `/bt/active_nodes`. `node_registry.py` auto-discovers `BTNode` subclasses for the UI
  palette; `bt_nodes_loader.py` parses BehaviorTree.CPP-v4 XML. Actions: `SendCommand`,
  `JointControl`, `Rotate`, `Wait`; controls: `Sequence`, `Loop`. Restricted to
  `ffw_sg2_rev1`. `launch/bt_node.launch.py` flattens the VLA-semantic YAML into BT params.
- **UI** (`orchestrator/ui/`, CRA + Redux Toolkit): pages Home / Record / Training /
  Inference / EditDataset / BTManager / Navigation / Replay. ROS via rosbridge WebSocket
  (`src/utils/rosConnectionManager.js`), HTTP `/api/` → supervisor_api, `/data-api/` →
  video file server. Ports in `ui/public/cyclo-config.js`: UI 7080, rosbridge 7090,
  video-file-server 7082, web_video_server 7085, supervisor_api 7100.
- Launch: `cyclo_intelligence_bringup.launch.py` (orchestrator + `cyclo_data_node`),
  `orchestrator_bringup.launch.py` (adds rosbridge, C++ recorder, web_video_server).

## 5. Data plane — `cyclo_data/`

- `cyclo_data/cyclo_data/cyclo_data_node.py:35` publishes umbrella `/data/status` and hosts:
  - `services/recording_service.py:73` — `/data/recording` (`RecordingCommand`:
    START/STOP/SEGMENT/FINISH_EPISODE/DISCARD/REFRESH_TOPICS), publishes
    `/data/recording/status` at 5 Hz.
  - `services/conversion_service.py:61` — `/data/convert`, `/data/convert/status`.
  - `services/hub_service.py:58` — `/data/hub` (HF up/download), publishes `/huggingface/status`.
  - `services/edit_service.py:35` — `/data/edit`.
- **Recording format v2**: state/action/tf → MCAP via the C++ recorder
  (`recorder/rosbag_recorder/src/service_bag_recorder.cpp:82`, service
  `/rosbag_recorder/send_command`); images → per-camera MJPEG-in-MP4 via
  `recorder/video_recorder.py:262` plus `camera_info` snapshot. Images are **not** in the
  MCAP. Output: `/workspace/rosbag2/Task_<num>_<name>_MCAP/<episode>/` (segmented mode:
  `<full_idx>/segments/<subtask_idx>/`, archived into `<full_idx>/` on FINISH_EPISODE).
- `recorder/session_manager.py:167` `DataManager` — episode/subtask numbering, archiving,
  metadata. `recorder/rosbag_control.py:39` also publishes `/task/action_event`.
- Converters: `converter/rosbag2mp4.py:111`, `to_lerobot_v21.py:160`,
  `to_lerobot_v30.py:301` (parquet); run in a separate process
  (`pipeline_worker.py:162`). Outputs go to `/workspace/lerobot/<task>_lerobot_v21` / `_lerobot_v30`;
  the Stage-1 `<episode>_converted/` intermediates are deleted afterwards.
- `reader/metadata_manager.py:27` — `episode_info.json`, task markers, trim/exclude regions.
- `visualization/video_file_server.py:63` — plain HTTP on :7082 for replay data, task-marker
  PUT, BT launch/save endpoints; started from `orchestrator_node.py:497`.

## 6. Shared config & interfaces

- `shared/shared/robot_configs/schema.py` is the single reader of the VLA-semantic YAML:
  `observation.images.<cam>{topic,msg_type,rotation_deg}`,
  `observation.state.<group>{topic,msg_type,joint_names}`,
  `action.<modality>{topic,msg_type,joint_names}`, `joystick`, `urdf_path`, `recording.extra_topics`.
  Robots: `ffw_sg2_rev1`, `ffw_bg2_rev4`, `ffw_sh5_rev1`, `f1`, `f2`, `omy_f3m`, `omx_f`.
  Mobile-base robots declare `action.mobile` → `/cmd_vel` as `geometry_msgs/msg/Twist`;
  everything else is `JointTrajectory`.
- Key srvs (`interfaces/srv/`):
  - `InferenceCommand` — LOAD/START/PAUSE/RESUME/STOP/UNLOAD/UPDATE_INSTRUCTION;
    `model_path`, `embodiment_tag`, `robot_type`, `task_instruction`, `publish_to_robot`,
    `action_request_mode`, `acceleration_mode`, `acceleration_engine_path` →
    `success`, `message`, `action_keys[]`.
  - `EngineCommand` — internal Main↔Engine; LOAD_POLICY/GET_ACTION/UNLOAD_POLICY with
    `seq_id` echoed; returns row-major `action_list` + `chunk_size` + `action_dim`.
  - `SendCommand` (UI → orchestrator), `RecordingCommand`, `StartConversion`, `HfOperation`,
    `EditDataset`, `TrainModel`, `StopTraining`, `TrainingStatus`, `LoadAndRunTree`,
    `GetNodeCatalog`, `GetReplayData`, `BrowseFile`, `SetRobotType`.
- Key msgs: `TaskInfo` (task, instructions, policy_path, service_type, inference_mode,
  action_request_mode, acceleration_*, control_hz, inference_hz, chunk_align_window_s),
  `RecordingStatus`, `InferenceStatus` (READY/LOADING/INFERENCING/PAUSED),
  `DataOperationStatus`, `TrainingInfo`, `TrainingProgress`, `ActionChunk`.

## 7. Docker / runtime supervision

- `docker/docker-compose.yml`: `cyclo_intelligence` (`robotis/cyclo-intelligence:1.3.0`,
  host net + IPC, nvidia, repo bind-mounted at `/root/ros2_ws/src`, docker.sock) plus
  on-demand `lerobot` (`robotis/lerobot-zenoh:1.4.0`) and `groot`
  (`robotis/groot-zenoh:1.3.4`) containers. Policy containers bind-mount
  `cyclo_brain/sdk/*`, `policy/common/runtime` → `/policy_runtime`,
  `policy/<backend>/<backend>_engine` → `/app`, `shared/robot_configs` → `/orchestrator_config`.
- s6 longruns (`docker/s6-services/`): `cyclo_intelligence`, `orchestrator`, `cyclo_data`,
  `bt_node` (on demand), `nginx`, `supervisor_api`, `s6-agent`. Policy containers run
  `main-runtime` and `engine-process` longruns with
  `PYTHONPATH=/app:/policy_runtime:/zenoh_sdk:/robot_client_sdk:/action_chunk_processing_sdk`.
- `docker/supervisor_api/app.py` (FastAPI behind nginx `/api/`): `/health`, `/workspace`,
  `/services/{name}/{status,start,stop}`, `/backends/{name}/{pull,start,restart,recreate,stop,status}`,
  `/backends/groot/trt/{status,build}`, `/navigation/*`.
- `docker/scripts/init_zenoh_cache.sh` seeds the zenoh message-definition cache; new
  msg/srv types must be reflected there or the container side will fail to decode.

## 8. Brain — `cyclo_brain/`

### 8.1 SDKs
- `sdk/zenoh_ros2_sdk` — pure-Python ROS2-over-zenoh (`ROS2Publisher`, `ROS2Subscriber`,
  `ROS2ServiceClient/Server`, `MessageRegistry`, liveliness tokens). Wire-compatible with
  the host's `rmw_zenoh_cpp`, so containers need no ROS install.
- `sdk/robot_client/robot_client/robot_client.py:161` `RobotClient(robot_type, enable_command_publishers, enable_preview_publisher)`
  — the single robot I/O abstraction, configured from the YAML. Subscribes cameras,
  follower joint states, odom. When `enable_command_publishers=True` it creates one
  publisher per `action.*` modality keyed `leader_<modality>` (`_init_command_publishers:304`)
  and a preview publisher on `/inference/trajectory_preview` (`:330`).
  API: `get_observation`, `get_images`, `get_joint_positions`, `publish_action`,
  `publish_action_preview`, `publish_idle_action`, `wait_for_ready`, `close`.
  `_resolve_action_key` maps a policy's `"odometry"` key to the `"mobile"` group.
  Env: `CMD_VEL_LINEAR_DEADBAND`, `CMD_VEL_ANGULAR_DEADBAND`.
- `sdk/robot_client/robot_client/service_server.py:121` `RobotServiceServer` — hosts the
  training-side `/<name>/train|stop|status` + `/<name>/training_progress`.
- `sdk/action_chunk_processing/action_chunk_processor.py:28` `ActionChunkProcessor` —
  `push_chunk((T,D))` → alignment (`l2` | `none` | `rtc`) against the last output →
  interpolation to `control_hz` → 0.2 s blend → deque. `pop_action()` returns `None` when
  empty (never repeats). `clear()` on pause/stop/mode change.

### 8.2 Policy container runtime — two processes

```
Orchestrator ── /<backend>/inference_command ──▶
══ MAIN PROCESS  (python3 -m main_runtime) ═════════════════════════════
 MainRuntime            main_runtime/main.py:49      env: INFERENCE_HZ=15, CONTROL_HZ=100,
   └─ ServiceHandler    service_handler.py:19             REFILL_MARGIN_S=0.2, ACTION_REQUEST_MODE=async,
       └─ ControlLoop   control_loop.py:60                GET_ACTION_TIMEOUT_S=5, ACTION_ALIGNMENT_MODE=l2
           ├─ RobotClient (command publishers ON)
           ├─ ActionChunkProcessor
           └─ InferenceRequester  inference_requester.py:27   (seq_id guard, one in flight)
               └─ ZenohEngineCommandClient  zenoh_client.py:29
                       │  /<backend>/engine_command  — zenoh service call across processes
══ ENGINE PROCESS  (python3 -m engine_process) ═════════════════════════
 EngineWorker           engine_process/worker.py:58
   └─ resolve_engine()  worker.py:174 — importlib.import_module("<backend>_engine").create_engine()
        └─ InferenceEngine (ABC)  engine.py:35 — load_policy / get_action_chunk / cleanup / is_ready
             ├── LeRobotEngine   policy/lerobot/lerobot_engine/engine.py:92 (explicit subclass; mixins)
             └── GR00TInference  policy/groot/runtime/inference_engine.py:224 (duck-typed, no base class)
        each engine owns a second, read-only RobotClient for observations
```

- **`ControlLoop.tick()`** (`control_loop.py:226`) runs at `processor.output_hz`:
  pop one action → always `publish_action_preview` → if `publish_to_robot`:
  `publish_action`; if buffer empty and `publish_to_robot`: `publish_idle_action`.
  Then `_should_request_actions` (`:339`): in `async` mode refill when
  `buffer_size < ceil((refill_margin_s + latency_EMA) * output_hz)`; in `sync` mode only
  when the buffer is empty. Requests run on a daemon thread; a `generation` counter
  discards chunks that arrive after pause/stop/mode change.
- **`publish_idle_action`** (`robot_client.py:616`): publishes a zero `Twist` on every
  Twist-typed action topic (in practice only `/cmd_vel`); JointTrajectory groups are
  skipped because their controllers hold the last point. Exists because the swerve base
  only has a 1 s `cmd_vel_timeout`. Not fired on STOP/PAUSE (the loop stops ticking).
  Introduced in commit `43b4838` with `sync` request mode.
- **`InferenceRequester`**: refuses a second GET_ACTION while one is in flight; discards
  responses whose `seq_id` doesn't match (a Main-side timeout doesn't stop the Engine).
- **Backends**: `LeRobotEngine` is composed from `loading.py`, `io_mapping.py`,
  `preprocessing.py`, `prediction.py`, `optimization.py`, `image_preprocessing.py`.
  `groot_engine` is a thin shim re-exporting `GR00TInference` (Gr00tPolicy, modality
  mapping, optional DiT-only TensorRT, HF token sync via `runtime/hf_token_sync.py`).
  Adding a backend = implement the four `InferenceEngine` members + `create_engine()`,
  mount it at `/app`, set `POLICY_BACKEND`.

## 9. End-to-end flows

1. **Setup** — UI → `/set_robot_type` → orchestrator loads the YAML, builds the topic
   inventory, starts heartbeat.
2. **Record** — `/task/command PREPARE_SESSION/START_RECORD` (or joystick trigger →
   `handle_joystick_trigger`, `orchestrator_node.py:2806`) → `/data/recording` →
   `RecordingService` starts `DataManager`, the C++ MCAP recorder, and `VideoRecorder`.
3. **Convert** — `CONVERT_MP4` → orchestrator resolves paths (`:1603`) → `/data/convert`
   → `Mp4ConversionWorker` → `/workspace/lerobot/<task>_lerobot_v21|_v30`. Optional `/data/hub` upload.
4. **Train** — `/training/command` → `ZenohTrainingManager` → `/lerobot/train`
   (`policy/lerobot/training.py:41 run_training`); progress on `/lerobot/training_progress`.
   Checkpoints in `/workspace/model/lerobot`.
5. **Infer** — `START_INFERENCE` → orchestrator picks backend prefix, ensures container is
   up via supervisor_api → `LOAD` then `START` on `/<backend>/inference_command` with
   `publish_to_robot` from `inference_mode` → Main loop pulls chunks from Engine, buffers,
   publishes preview always and real commands only in robot mode. `*_INFERENCE_RECORD`
   loops rollouts back into the recorder.

## 10. Sister repos (siblings under `~/src/`)

### `ai_worker` — the FFW robot (v2.2.5)
- Variants: `ffw_sg2` (swerve + gripper), `ffw_bg2` (bench), `ffw_sh5`/`bh5` (HX5-D20
  hands), `ffw_f1`/`f2`, `ffw_lg2_leader` (teleop master).
- Follower controllers (`ffw_bringup/config/ffw_sg2_rev1_follower/ffw_sg2_follower_ai_hardware_controller.yaml`):
  `arm_l/r_controller` (JTC, `arm_{l,r}_joint1..7` + `gripper_{l,r}_joint1`),
  `head_controller`, `lift_controller`, `swerve_drive_controller`, `ffw_robot_manager`.
- **Policy injection point** — launch-time remaps (`ffw_sg2_follower_ai.launch.py:154-177`):
  `/leader/joint_trajectory_command_broadcaster_{left,right}/joint_trajectory` → arms,
  `/leader/joystick_controller_left/joint_trajectory` → head,
  `/leader/joystick_controller_right/joint_trajectory` → lift, `/cmd_vel` → base.
  These match the `action.*` topics in the cyclo_intelligence YAML exactly.
- Teleop↔AI switching is implicit: the leader broadcaster defaults to `STOPPED` and
  publishes nothing until both triggers are squeezed for 2 s
  (`joint_trajectory_command_broadcaster.cpp:391-434`).
- Safety: no arm watchdog (JTC holds last point); base `cmd_vel_timeout: 1.0`
  (`swerve_drive_controller.cpp:642`). No "idle", "brake", or "ready pose" concepts.
- Flag: H5 launches remap `hand_{l,r}_controller` to `..._{left,right}_hand/joint_trajectory`
  but nothing in the repo publishes those topics.

### `cyclo_control` — motion controller (v0.3.2)
- QP differential-IK at 100 Hz over OSQP (`cyclo_motion_controller_core/.../qp_base.hpp:45`).
  Decision vector `[q̇, slack_qmin, slack_qmax, slack_sing, slack_collision]`; CBF joint
  limits and self-collision constraints (`vr_controller.cpp:178-216`).
- Modes: `movel`, `movej`, `bimanual_movel` (rigid-grasp equality), `bimanual_movej`,
  `vr`, `leader`. Topics only, no services/actions.
- Publishes the same `/leader/.../joint_trajectory` topics — a sibling command source to
  the policy, not a layer beneath it. Zero references to cyclo_intelligence.
- Flag: singularity slack is penalised (`vr_controller.cpp:143`) but its constraint row is
  never written — manipulability avoidance is dead code.

### `physical_ai_tools` — predecessor (v0.8.2)
- Monolithic `physical_ai_server` node; same `/task/command`, `TaskInfo`, `TaskStatus`.
- Inference was an in-process rclpy timer calling `policy.select_action()` single-step
  (`physical_ai_server.py:551`): no chunking, no dry-run gate, no idle action; the ZMQ
  remote-inference pair was dead code. cyclo_intelligence replaced it with the
  containerised Main/Engine split over zenoh.
- `physical_ai_manager` (React UI) became `orchestrator/ui`; `rosbag_recorder` became
  `cyclo_data/recorder/rosbag_recorder`; `physical_ai_bt` became `orchestrator/bt`.

### `cyclo_lab` — Isaac Lab sim (v2.0.3)
- RL/IL tasks for OMY / OMX / FFW_BG2 / FFW_SG2 / K1. No code dependency on the others;
  impersonates the robot at the topic level over CycloneDDS
  (`scripts/sim2real/imitation_learning/dds_sdk/ffw_sg2_sdk.py:69-105`). Hands datasets
  over via `isaaclab2lerobot.py`.

## 11. Non-obvious things to keep in mind

1. **Two `RobotClient`s per policy container** — Main's has command publishers, Engine's
   is read-only. New command paths belong in Main.
2. **`STOP`/`PAUSE` stop the tick entirely**, so the zero-`cmd_vel` idle publish does not
   fire on stop; the base then relies on the 1 s controller timeout. An immediate halt
   on STOP would be a change in `ControlLoop.stop()`.
3. **Host↔container boundary is `rmw_zenoh_cpp` ↔ `zenoh_ros2_sdk`.** New msg/srv types
   must land in the zenoh definition cache (`docker/scripts/init_zenoh_cache.sh`).
4. **`GR00TInference` does not inherit `InferenceEngine`** — it duck-types it.
   `EngineWorker` never does `isinstance`, so both work.
5. **Least mature areas**: BT runtime (single robot), H5 hand topics (no in-repo
   publisher), ZMQ leftovers in physical_ai_tools.
6. **Dry-run mode** (`publish_to_robot=False`) still publishes
   `/inference/trajectory_preview` for the 3D viewer, so you can validate a policy
   visually with the robot torqued but uncommanded.

## 12. Useful entry points

| Want to… | Start at |
|---|---|
| Add a UI command | `interfaces/srv/SendCommand.srv` → `orchestrator_node.py` handler → `ui/src/features/tasks` |
| Add a robot | `shared/shared/robot_configs/<robot>_config.yaml` + URDF; validate with `schema.py` |
| Add a policy backend | implement `InferenceEngine` (`cyclo_brain/policy/common/runtime/engine.py`) + `create_engine()`; add compose service |
| Change action timing/safety | `cyclo_brain/policy/common/runtime/main_runtime/control_loop.py`, `sdk/action_chunk_processing` |
| Change what gets recorded | `cyclo_data/cyclo_data/services/recording_service.py`, `recorder/video_recorder.py` |
| Change dataset format | `cyclo_data/cyclo_data/converter/to_lerobot_v30.py` |
| Unit tests for the runtime | `cyclo_brain/policy/common/runtime/tests/` (pure-Python, no SDK needed) |

---

## Appendix A. ROS 2 entity wiring (added 2026-08-24)

No ROS 2 **actions** are used anywhere in cyclo_intelligence; long-running work is
"service returns job_id + status topic + poll/cancel service" (probably because
`zenoh_ros2_sdk` implements pub/sub/services only).

### A.1 orchestrator_node

| Kind | Name | Type | Notes |
|---|---|---|---|
| SRV | `/task/command` | `SendCommand` | 25-command dispatcher, `orchestrator_node.py:1089` |
| SRV | `/set_robot_type`, `/get_robot_types`, `/get_robot_info` | | loads YAML, builds `Communicator` |
| SRV | `/register_hf_user`, `/get_registered_hf_user`, `/huggingface/list_endpoints`, `/huggingface/select_endpoint` | | `HFEndpointStore` |
| SRV | `/training/command` | `SendTrainingCommand` | START / FINISH |
| SRV | `/training/get_available_policy`, `get_user_list`, `get_dataset_list`, `get_model_weight_list`, `get_training_info` | | |
| SRV | `/replay/get_data`, `/browse_file` | | |
| SRV | `/image/get_available_list`, `/dataset/get_info`, `/bt/list_trees`, `/browse_file` (dup) | | via `Communicator` |
| PUB | `/training/status` | `TrainingStatus` | 0.5 Hz while training |
| PUB | `/task/inference_status` | `InferenceStatus` | one-shot on LOAD/START/PAUSE/RESUME/STOP |
| PUB | `heartbeat` | `std_msgs/Empty` | |
| SUB | `/data/status` | `DataOperationStatus` | debug-log only |
| SUB | `<joystick.trigger_topic>` | `std_msgs/String` | right = start/stop segment, left = cancel |
| SUB | `/<backend>/progress` | `TrainingProgress` | while training |
| CLI | `/data/recording`, `/data/convert`, `/data/convert/status`, `/data/hub`, `/data/edit` | | `CycloDataClient` |
| CLI | `/<backend>/inference_command`, `/train`, `/stop`, `/status` | | `ContainerServiceClient`, prefix `/lerobot` or `/groot` |

Key behaviours: `START_INFERENCE` is async (LOAD→START on a daemon thread, UI
watches `/task/inference_status`); `STOP_INFERENCE` sends **PAUSE** (model stays
loaded); only `FINISH` sends STOP+UNLOAD. Orchestrator makes no HTTP calls —
container start/stop is UI → `supervisor_api` `/api/backends/<b>/start`.

### A.2 cyclo_data node

| Kind | Name | Type | Notes |
|---|---|---|---|
| SRV | `/data/recording` | `RecordingCommand` | 17 commands |
| SRV | `/data/convert`, `/data/convert/status` | `StartConversion`, `GetConversionStatus` | |
| SRV | `/data/hub` | `HfOperation` | UI calls this directly |
| SRV | `/data/edit` | `EditDataset` | UI calls this directly |
| PUB | `/data/status` | `DataOperationStatus` | umbrella progress |
| PUB | `/data/recording/status` | `RecordingStatus` | 5 Hz, UI subscribes directly |
| PUB | `/task/action_event` | `std_msgs/String` | "start" / "finish" / "cancel" |
| PUB | `/huggingface/status` | `HFOperationStatus` | |
| SUB | every `observation.images.*.topic` | `CompressedImage` | `VideoRecorder`, persistent from REFRESH_TOPICS |
| SUB | every camera_info topic | `CameraInfo` | `CameraInfoSnapshot`, one message per episode |
| CLI | `/rosbag_recorder/send_command` | `rosbag_recorder/SendCommand` | PREPARE / START / STOP / STOP_AND_DELETE / FINISH |

C++ `service_bag_recorder`: SRV `/rosbag_recorder/send_command`, PUB
`/rosbag_recorder/monitor` (`RecordingMonitor`), SUB every topic from
`schema.get_mcap_record_topics` (state + tactile + action + extras − camera_info).

### A.3 Topic → writer routing (ffw_sg2_rev1)

- `observation.images.*` (4 `CompressedImage` topics) → `VideoRecorder` → `videos/<cam>.mp4` + `_timestamps.parquet`
- camera_info extras (4) → `CameraInfoSnapshot` → `camera_info/<cam>.yaml`
- `/joint_states`, `/odom`, 4× `/leader/*/joint_trajectory`, `/cmd_vel`, `/tf` → C++ recorder → `<n>.mcap`
- `/leader/joystick_controller/tact_trigger` → orchestrator only (not recorded)

### A.4 bt_node (`orchestrator_bt_node`, ffw_sg2_rev1 only)

SRV `/bt/set_running` (SetBool), `/bt/load_and_run` (LoadAndRunTree), `/bt/nodes/catalog`
(GetNodeCatalog); PUB `/bt/status`, `/bt/active_nodes` (String); CLI `/task/command`
(cleanup STOP_INFERENCE). Actions: `SendCommand` (CLI `/task/command`, SUB
`/task/inference_status`), `JointControl` (SUB state JointState topics, PUB action
JointTrajectory topics), `Rotate` (PUB `/cmd_vel`, SUB `/odom`), `Wait`.
`launch/bt_node.launch.py` flattens the robot YAML into `joint_topic_list` /
`joint_order.leader_<modality>` params these actions read.

### A.5 UI ROS surface (via rosbridge :7090)

SUB: `/task/inference_status`, `/training/status`, `/heartbeat`, `/data/recording/status`,
`/data/status`, `/huggingface/status`, `/task/action_event`, `/rosbag_recorder/monitor`.
SRV CALL: all orchestrator services above plus `/data/hub`, `/data/edit`, `/bt/nodes/catalog`.
HTTP: `/api/*` → supervisor_api :7100, `/data-api/*` → video_file_server :7082.
