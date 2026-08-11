from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import threading

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))
sys.path.insert(0, str(_REPO_ROOT / "orchestrator"))
import cyclo_data  # noqa: E402
import cyclo_data.converter  # noqa: E402
import cyclo_data.hub  # noqa: E402


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        return
    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        sys.modules.setdefault(parent, ModuleType(parent))
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


_stub_module("huggingface_hub", HfApi=_Dummy)
_stub_module("interfaces.msg", RecordingStatus=_Dummy)
_stub_module("cyclo_data.converter.orchestrator", DataConverter=_Dummy)
_stub_module(
    "cyclo_data.hub.progress_tracker",
    HuggingFaceLogCapture=_Dummy,
    HuggingFaceProgressTqdm=_Dummy,
)
_stub_module("psutil", cpu_percent=lambda interval=None: 0.0)

from cyclo_data.recorder.session_manager import DataManager  # noqa: E402
import cyclo_data.recorder.session_manager as session_manager_module  # noqa: E402


class _VanishingPath:
    def is_file(self):
        return True

    def stat(self):
        raise FileNotFoundError("removed during scan")


def _make_manager(root: Path, *, subtask_total: int = 2) -> DataManager:
    manager = DataManager.__new__(DataManager)
    manager._save_rosbag_path = str(root)
    manager._segmented_storage_mode = True
    manager._physical_segment_total = subtask_total
    manager._subtask_mode = subtask_total > 1
    manager._main_task_instruction = "main instruction"
    manager._task_instruction_source = "user"
    manager._subtask_instructions = [
        f"subtask {idx}" for idx in range(subtask_total)
    ]
    manager._task_info = SimpleNamespace(
        task_num="1234",
        task_name="archive test",
        task_type="record",
        policy_type="",
    )
    manager._task_type = "record"
    manager._collection_id = ""
    manager._robot_type = "test_robot"
    manager._state_lock = threading.Lock()
    manager._saved_subtasks_cache = {}
    return manager


def test_inference_save_repo_name_uses_stable_collection_without_mutation(tmp_path):
    task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
        policy_type="act",
    )
    collection_id = "ACT_dataset_20260810T010203_000004Z_abcd1234"

    repo_name = DataManager._make_save_repo_name(
        tmp_path,
        task_info,
        collection_id=collection_id,
    )

    assert repo_name == f"{collection_id}_MCAP"
    assert task_info.task_num == ""
    assert task_info.task_name == ""


def test_inference_collection_rejects_path_traversal(tmp_path):
    task_info = SimpleNamespace(task_type="inference", policy_type="act")

    with pytest.raises(ValueError, match="Invalid inference collection_id"):
        DataManager._make_save_repo_name(
            tmp_path,
            task_info,
            collection_id="../outside",
        )


@pytest.mark.parametrize(
    "task_num,task_name",
    [
        ("../outside", "pick"),
        ("1", "/outside"),
        ("1", ".."),
        ("1", "folder\\child"),
        ("1", "bad\x00name"),
    ],
)
def test_record_repo_name_rejects_unsafe_path_components(
    tmp_path,
    task_num,
    task_name,
):
    task_info = SimpleNamespace(
        task_num=task_num,
        task_name=task_name,
        task_type="record",
    )

    with pytest.raises(ValueError, match="Invalid recording"):
        DataManager._make_save_repo_name(tmp_path, task_info)


def test_inference_manager_rejects_symlink_collection_root(tmp_path):
    collection_id = "ACT_dataset_symlink"
    inference_root = tmp_path / "inference"
    inference_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (inference_root / f"{collection_id}_MCAP").symlink_to(
        outside,
        target_is_directory=True,
    )
    task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
        policy_type="act",
        task_instruction=[],
        subtask_instruction=[],
        include_robotis_license=False,
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        DataManager(
            tmp_path,
            "ffw_sg2_rev1",
            task_info,
            collection_id=collection_id,
        )


def test_inference_manager_uses_nested_workspace_path(tmp_path):
    task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
        policy_type="act",
        task_instruction=[],
        subtask_instruction=[],
        include_robotis_license=False,
    )
    collection_id = "ACT_dataset_20260810T010203_000004Z_abcd1234"

    manager = DataManager(
        tmp_path,
        "ffw_sg2_rev1",
        task_info,
        collection_id=collection_id,
    )

    assert Path(manager._save_rosbag_path) == (
        tmp_path / "inference" / f"{collection_id}_MCAP"
    )
    assert manager._main_task_instruction == "ACT_dataset"
    assert manager._task_instruction_source == "fallback"


def test_inference_segment_metadata_uses_same_time_for_outcome(tmp_path):
    policy_path = tmp_path / "checkpoint" / "checkpoints" / "080000" / "pretrained_model"
    policy_path.mkdir(parents=True)
    config_bytes = b'{"type":"act","chunk_size":30}'
    (policy_path / "config.json").write_bytes(config_bytes)
    (policy_path / "model.safetensors").write_bytes(b"weights")
    task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
        service_type="lerobot",
        policy_type="act",
        policy_path=str(policy_path),
        control_hz=15,
        inference_hz=15,
        task_instruction=[],
        subtask_instruction=[],
        include_robotis_license=False,
    )
    collection_id = "ACT_dataset_20260810T010203_000004Z_abcd1234"
    manager = DataManager(
        tmp_path,
        "ffw_sg2_rev1",
        task_info,
        collection_id=collection_id,
    )
    manager.start_recording()
    episode_dir = Path(manager.get_save_rosbag_path())
    episode_dir.mkdir(parents=True, exist_ok=True)

    manager.save_robotis_metadata(
        episode_outcome={
            "schema_version": 1,
            "status": "success",
            "success": True,
            "source": "operator_ui",
        }
    )

    info = json.loads((episode_dir / "episode_info.json").read_text())
    assert info["task_instruction"] == "ACT_dataset"
    assert info["task_instruction_source"] == "fallback"
    assert info["collection_id"] == collection_id
    assert info["outcome"]["status"] == "success"
    assert info["outcome"]["annotated_at"] == info["timestamp"]

    provenance = info["policy_provenance"]
    assert provenance["schema_version"] == 1
    assert provenance["service_type"] == "lerobot"
    assert provenance["policy_type"] == "act"
    assert provenance["policy_path"] == str(policy_path)
    assert provenance["checkpoint_id"] == "080000"
    assert provenance["control_hz"] == 15
    assert provenance["inference_hz"] == 15
    assert provenance["config_path"] == "config.json"
    assert provenance["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert provenance["artifacts"] == [
        {"name": "config.json", "size_bytes": len(config_bytes)},
        {"name": "model.safetensors", "size_bytes": 7},
    ]
    canonical_manifest = json.dumps(
        provenance["artifacts"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert provenance["artifact_manifest_sha256"] == hashlib.sha256(
        canonical_manifest
    ).hexdigest()

    contract = info["rl_episode_contract"]
    assert contract["schema_version"] == 1
    assert contract["collection_id"] == collection_id
    assert contract["session_id"] == collection_id
    assert contract["cameras"] == {
        "names": ["cam_left_head", "cam_left_wrist", "cam_right_wrist"],
        "image_shape": [3, 256, 256],
    }
    expected_names = [
        "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
        "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
        "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
        "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
        "head_joint1", "head_joint2", "lift_joint", "linear_x", "linear_y",
        "angular_z",
    ]
    assert contract["state"] == {
        "dim": 22,
        "names": expected_names,
        "topics": ["/joint_states", "/odom"],
    }
    assert contract["action"] == {
        "dim": 22,
        "names": expected_names,
        "topics": [
            "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            "/leader/joystick_controller_left/joint_trajectory",
            "/leader/joystick_controller_right/joint_trajectory",
            "/cmd_vel",
        ],
        "hz": 15,
    }
    assert contract["chunk"] == {
        "size": 30,
        "action_dim": 22,
        "topic": "/inference/action_chunk",
        "ack_topic": "/inference/action_step_ack",
    }
    assert contract["terminal_reward"] == {
        "semantics": "binary_terminal",
        "success": 1.0,
        "failure": 0.0,
        "intermediate": 0.0,
    }


def test_non_inference_metadata_does_not_add_rl_snapshots(tmp_path):
    task_info = SimpleNamespace(
        task_num="1",
        task_name="teleop",
        task_type="record",
        policy_type="",
        policy_path="/must/not/be/scanned",
        task_instruction=["pick"],
        subtask_instruction=[],
        include_robotis_license=False,
    )
    manager = DataManager(tmp_path, "ffw_sg2_rev1", task_info)
    manager.start_recording()
    episode_dir = Path(manager.get_save_rosbag_path())
    episode_dir.mkdir(parents=True, exist_ok=True)

    manager.save_robotis_metadata()

    info = json.loads((episode_dir / "episode_info.json").read_text())
    assert "policy_provenance" not in info
    assert "rl_episode_contract" not in info


def _write_segment(
    root: Path,
    *,
    full_idx: int,
    subtask_idx: int,
    subtask_total: int,
    with_video: bool = True,
    subtask_instruction: str | None = None,
    extra_info: dict | None = None,
) -> Path:
    segment = root / str(full_idx) / "segments" / str(subtask_idx)
    segment.mkdir(parents=True, exist_ok=True)
    (segment / f"segment_{subtask_idx}.mcap").write_bytes(
        f"mcap-{subtask_idx}".encode()
    )
    start_ns = 1_000_000_000 + subtask_idx * 100_000_000
    duration_ns = 50_000_000
    metadata = {
        "rosbag2_bagfile_information": {
            "version": 9,
            "storage_identifier": "mcap",
            "duration": {"nanoseconds": duration_ns},
            "starting_time": {"nanoseconds_since_epoch": start_ns},
            "message_count": 3,
            "topics_with_message_count": [
                {
                    "topic_metadata": {
                        "name": "/joint_states",
                        "type": "sensor_msgs/msg/JointState",
                        "serialization_format": "cdr",
                        "offered_qos_profiles": "",
                    },
                    "message_count": 3,
                }
            ],
            "compression_format": "",
            "compression_mode": "",
            "relative_file_paths": [f"segment_{subtask_idx}.mcap"],
            "files": [
                {
                    "path": f"segment_{subtask_idx}.mcap",
                    "starting_time": {"nanoseconds_since_epoch": start_ns},
                    "duration": {"nanoseconds": duration_ns},
                    "message_count": 3,
                }
            ],
            "custom_data": None,
            "ros_distro": "jazzy",
        }
    }
    (segment / "metadata.yaml").write_text(
        yaml.safe_dump(metadata, sort_keys=False),
        encoding="utf-8",
    )
    info = {
        "recording_mode": "subtask" if subtask_total > 1 else "single_segment",
        "full_episode_index": full_idx,
        "subtask_index": subtask_idx,
        "subtask_total": subtask_total,
        "episode_index": subtask_idx,
        "subtask_instruction": subtask_instruction or f"subtask {subtask_idx}",
    }
    if with_video:
        videos = segment / "videos"
        videos.mkdir()
        (videos / "cam0.mp4").write_bytes(b"raw-mjpeg")
        (videos / "cam0_timestamps.parquet").write_bytes(b"timestamps")
        info["video_stats"] = {"cam0": {"frames_written": 1}}
    if extra_info:
        info.update(extra_info)
    (segment / "episode_info.json").write_text(json.dumps(info, indent=2))
    return segment


def test_file_size_if_present_ignores_concurrent_removal():
    assert DataManager._file_size_if_present(_VanishingPath()) == 0


def test_archive_moves_segmented_files_and_marks_pending(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    first = _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    second = _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)

    out = manager._archive_full_episode(0)

    assert out == root / "0"
    assert (out / "0_0.mcap").read_bytes() == b"mcap-0"
    assert (out / "0_1.mcap").read_bytes() == b"mcap-1"
    assert not (first / "segment_0.mcap").exists()
    assert not (second / "segment_1.mcap").exists()
    assert not (out / "segments").exists()
    assert (out / "videos" / "0_0" / "cam0.mp4").read_bytes() == b"raw-mjpeg"
    assert (out / "videos" / "0_0" / "cam0_timestamps.parquet").read_bytes() == (
        b"timestamps"
    )

    info = json.loads((out / "episode_info.json").read_text())
    assert info["transcoding_status"] == "pending"
    assert info["video_segments"] == [
        {
            "mcap": "0_0.mcap",
            "video_dir": "videos/0_0",
            "cameras": ["cam0"],
            "raw_cameras": [],
        },
        {
            "mcap": "0_1.mcap",
            "video_dir": "videos/0_1",
            "cameras": ["cam0"],
            "raw_cameras": [],
        },
    ]


def test_archive_marks_episode_without_videos_not_required(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=1)
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
    )

    out = manager._archive_full_episode(0)

    assert out == root / "0"
    assert not (out / "segments").exists()
    info = json.loads((out / "episode_info.json").read_text())
    assert info["transcoding_status"] == "not_required"
    assert info["video_segments"] == []


def test_archive_preserves_inference_outcome_and_provenance(tmp_path):
    root = tmp_path / "inference" / "ACT_dataset_session_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._task_type = "inference"
    manager._collection_id = "ACT_dataset_session"
    manager._main_task_instruction = "ACT_dataset"
    manager._task_instruction_source = "fallback"
    manager._task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
        policy_type="act",
    )
    outcome = {
        "schema_version": 1,
        "status": "success",
        "success": True,
        "source": "operator_ui",
        "annotated_at": "2026-08-10T01:02:03Z",
    }
    policy_provenance = {
        "schema_version": 1,
        "service_type": "lerobot",
        "policy_type": "act",
        "policy_path": "/workspace/checkpoint/act/checkpoints/080000/pretrained_model",
        "checkpoint_id": "080000",
        "control_hz": 15,
        "inference_hz": 15,
        "config_path": "config.json",
        "config_sha256": "config-digest",
        "artifact_manifest_sha256": "manifest-digest",
        "artifacts": [{"name": "config.json", "size_bytes": 123}],
    }
    rl_episode_contract = {
        "schema_version": 1,
        "collection_id": "ACT_dataset_session",
        "session_id": "ACT_dataset_session",
        "cameras": {
            "names": ["cam_left_head", "cam_left_wrist", "cam_right_wrist"],
            "image_shape": [3, 256, 256],
        },
        "state": {"dim": 22, "names": ["state"] * 22, "topics": ["/state"]},
        "action": {
            "dim": 22,
            "names": ["action"] * 22,
            "topics": ["/action"],
            "hz": 15,
        },
        "chunk": {
            "size": 30,
            "action_dim": 22,
            "topic": "/inference/action_chunk",
            "ack_topic": "/inference/action_step_ack",
        },
        "terminal_reward": {
            "semantics": "binary_terminal",
            "success": 1.0,
            "failure": 0.0,
            "intermediate": 0.0,
        },
    }
    segment_info = {
        "task_instruction": "ACT_dataset",
        "task_instruction_source": "fallback",
        "task_type": "inference",
        "policy_type": "act",
        "collection_id": "ACT_dataset_session",
        "policy_provenance": policy_provenance,
        "rl_episode_contract": rl_episode_contract,
        "outcome": outcome,
    }
    for subtask_idx in range(2):
        _write_segment(
            root,
            full_idx=0,
            subtask_idx=subtask_idx,
            subtask_total=2,
            with_video=False,
            extra_info=segment_info,
        )

    out = manager._archive_full_episode(0)
    summary = json.loads((out / "episode_info.json").read_text())

    assert summary["outcome"] == outcome
    assert summary["collection_id"] == "ACT_dataset_session"
    assert summary["task_instruction"] == "ACT_dataset"
    assert summary["task_instruction_source"] == "fallback"
    assert summary["policy_provenance"] == policy_provenance
    assert summary["rl_episode_contract"] == rl_episode_contract


def test_archive_write_failure_preserves_segments_and_retries(monkeypatch, tmp_path):
    root = tmp_path / "inference" / "ACT_dataset_session_MCAP"
    manager = _make_manager(root, subtask_total=1)
    outcome = {
        "schema_version": 1,
        "status": "failure",
        "success": False,
        "source": "operator_ui",
        "annotated_at": "2026-08-10T01:02:03Z",
    }
    segment = _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
        extra_info={"outcome": outcome},
    )
    original_atomic_write = session_manager_module._atomic_write_json
    attempts = 0

    def fail_once(path, obj, indent=2):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("simulated metadata fsync failure")
        return original_atomic_write(path, obj, indent=indent)

    monkeypatch.setattr(session_manager_module, "_atomic_write_json", fail_once)

    with pytest.raises(OSError, match="simulated metadata fsync failure"):
        manager._archive_full_episode(0)

    assert (segment / "episode_info.json").exists()
    assert json.loads((segment / "episode_info.json").read_text())["outcome"] == outcome

    out = manager._archive_full_episode(0)
    assert not (out / "segments").exists()
    assert json.loads((out / "episode_info.json").read_text())["outcome"] == outcome


def test_archive_video_partial_move_is_retry_safe(monkeypatch, tmp_path):
    root = tmp_path / "inference" / "ACT_dataset_session_MCAP"
    manager = _make_manager(root, subtask_total=1)
    segment = _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=True,
    )
    original_move = DataManager._move_file
    failed = False

    def fail_first_sidecar(src, dst):
        nonlocal failed
        if not failed and Path(src).name.endswith("_timestamps.parquet"):
            failed = True
            raise OSError("simulated sidecar move failure")
        return original_move(src, dst)

    monkeypatch.setattr(
        DataManager,
        "_move_file",
        staticmethod(fail_first_sidecar),
    )
    with pytest.raises(RuntimeError, match="sidecar move failure"):
        manager._archive_full_episode(0)

    out = root / "0"
    assert (out / "videos" / "0_0" / "cam0.mp4").exists()
    assert (segment / "videos" / "cam0_timestamps.parquet").exists()
    assert (out / "segments").exists()

    monkeypatch.setattr(DataManager, "_move_file", staticmethod(original_move))
    out = manager._archive_full_episode(0)

    assert (out / "videos" / "0_0" / "cam0.mp4").exists()
    assert (out / "videos" / "0_0" / "cam0_timestamps.parquet").exists()
    assert not (out / "segments").exists()


@pytest.mark.parametrize("with_metadata_identity", [True, False])
def test_archive_split_mcap_partial_move_is_retry_safe(
    monkeypatch,
    tmp_path,
    with_metadata_identity,
):
    root = tmp_path / "inference" / "ACT_dataset_session_MCAP"
    manager = _make_manager(root, subtask_total=1)
    segment = _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
    )
    (segment / "segment_0.mcap").unlink()
    first = segment / "split_0.mcap"
    second = segment / "split_1.mcap"
    first.write_bytes(b"first-split")
    second.write_bytes(b"second-split")
    metadata_path = segment / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    bag_info = metadata["rosbag2_bagfile_information"]
    file_template = dict(bag_info["files"][0])
    bag_info["relative_file_paths"] = (
        ["split_0.mcap", "split_1.mcap"]
        if with_metadata_identity else []
    )
    bag_info["files"] = [
        {
            **file_template,
            "path": "split_0.mcap" if with_metadata_identity else "",
        },
        {
            **file_template,
            "path": "split_1.mcap" if with_metadata_identity else "",
        },
    ]
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False))

    original_move = DataManager._move_file
    failed = False

    def fail_second_split(src, dst):
        nonlocal failed
        if not failed and Path(src).name == "split_1.mcap":
            failed = True
            raise OSError("simulated second split move failure")
        return original_move(src, dst)

    monkeypatch.setattr(
        DataManager,
        "_move_file",
        staticmethod(fail_second_split),
    )
    with pytest.raises(OSError, match="second split move failure"):
        manager._archive_full_episode(0)

    out = root / "0"
    assert (out / "0_0_0.mcap").read_bytes() == b"first-split"
    assert second.read_bytes() == b"second-split"
    assert (out / "segments").exists()

    monkeypatch.setattr(DataManager, "_move_file", staticmethod(original_move))
    manager._archive_full_episode(0)

    assert (out / "0_0_0.mcap").read_bytes() == b"first-split"
    assert (out / "0_0_1.mcap").read_bytes() == b"second-split"
    assert not (out / "segments").exists()


def test_manager_construction_recovers_complete_crash_segments(tmp_path):
    collection_id = "ACT_dataset_crash_recovery"
    root = tmp_path / "inference" / f"{collection_id}_MCAP"
    outcome = {
        "schema_version": 1,
        "status": "success",
        "success": True,
        "source": "operator_ui",
        "annotated_at": "2026-08-10T01:02:03Z",
    }
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
        extra_info={
            "task_instruction": "ACT_dataset",
            "task_instruction_source": "fallback",
            "task_type": "inference",
            "policy_type": "act",
            "collection_id": collection_id,
            "robot_type": "ffw_sg2_rev1",
            "outcome": outcome,
        },
    )
    task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
        policy_type="act",
        task_instruction=[],
        subtask_instruction=[],
        include_robotis_license=False,
    )

    manager = DataManager(
        tmp_path,
        "ffw_sg2_rev1",
        task_info,
        collection_id=collection_id,
    )

    assert manager._recovered_episode_dirs == [root / "0"]
    assert not (root / "0" / "segments").exists()
    summary = json.loads((root / "0" / "episode_info.json").read_text())
    assert summary["outcome"] == outcome
    assert manager._current_full_episode_index == 1


def test_archive_preserves_pending_raw_spool(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=1)
    segment = _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=1,
        with_video=False,
    )
    videos = segment / "videos"
    videos.mkdir()
    (videos / "cam0.mjpeg.tmp").write_bytes(b"raw-spool")
    (videos / "cam0_timestamps.parquet").write_bytes(b"timestamps")
    (videos / "cam0_recorder_stats.json").write_text(
        json.dumps({"frames_written": 1, "remux_status": "pending"}),
        encoding="utf-8",
    )
    info_path = segment / "episode_info.json"
    info = json.loads(info_path.read_text())
    info["video_stats"] = {"cam0": {"frames_written": 1, "remux_status": "pending"}}
    info["transcoding_status"] = "pending"
    info["video_remux_status"] = "pending"
    info_path.write_text(json.dumps(info, indent=2))

    out = manager._archive_full_episode(0)

    assert not (out / "segments").exists()
    archived_video_dir = out / "videos" / "0_0"
    assert (archived_video_dir / "cam0.mjpeg.tmp").read_bytes() == b"raw-spool"
    assert (archived_video_dir / "cam0_timestamps.parquet").read_bytes() == (
        b"timestamps"
    )
    assert (archived_video_dir / "cam0_recorder_stats.json").exists()
    summary = json.loads((out / "episode_info.json").read_text())
    assert summary["transcoding_status"] == "pending"
    assert summary["video_remux_status"] == "pending"


def test_discard_current_full_episode_removes_all_saved_subtasks(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._current_full_episode_index = 0
    manager._current_subtask_index = 1
    manager._current_scenario_number = 1
    manager._record_episode_count = 2
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)

    deleted = manager.discard_current_full_episode()

    assert deleted == 2
    assert not (root / "0").exists()
    assert manager._current_subtask_index == 0
    assert manager._current_scenario_number == 0


def test_discard_full_episode_deletes_requested_episode_without_cursor_drift(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._current_full_episode_index = 1
    manager._current_subtask_index = 1
    manager._current_scenario_number = 1
    manager._record_episode_count = 3
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)
    _write_segment(root, full_idx=1, subtask_idx=0, subtask_total=2)

    deleted = manager.discard_full_episode(0)

    assert deleted == 2
    assert not (root / "0").exists()
    assert (root / "1").exists()
    assert manager._current_full_episode_index == 1
    assert manager._current_subtask_index == 1
    assert manager._current_scenario_number == 1


def test_discard_recording_can_reset_active_episode_subtask_cursor(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._segmented_storage_mode = True
    manager._status = "recording"
    manager._start_time_s = 123.0
    manager._record_episode_count = 2
    manager._current_subtask_index = 2
    manager._current_scenario_number = 2

    manager.discard_recording(reset_subtask_index=True)

    assert manager._status == "idle"
    assert manager._record_episode_count == 2
    assert manager._current_subtask_index == 0
    assert manager._current_scenario_number == 0


def test_missing_subtasks_reports_gap_in_saved_segments(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._current_full_episode_index = 0
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=3)
    _write_segment(root, full_idx=0, subtask_idx=2, subtask_total=3)

    assert manager.saved_subtask_indices_for_full_episode() == {0, 2}
    assert manager.missing_subtasks_for_full_episode() == [1]


def test_active_segment_directory_without_episode_info_is_not_saved(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._current_full_episode_index = 0
    _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=3)
    active_segment = root / "0" / "segments" / "1"
    active_segment.mkdir(parents=True)
    (active_segment / "segment_1.mcap").write_bytes(b"recording")

    assert manager.saved_subtask_indices_for_full_episode() == {0}
    assert manager.missing_subtasks_for_full_episode() == [1, 2]


def test_saved_subtask_indices_uses_cache_when_segments_dir_missing(monkeypatch, tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=3)
    manager._current_full_episode_index = 4
    manager._saved_subtasks_cache = {4: {0, 2}}
    monkeypatch.setattr(
        manager,
        "_episode_dirs_for_full_subtask",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("status lookup must not scan the whole task root")
        ),
    )

    assert manager.saved_subtask_indices_for_full_episode() == {0, 2}
    assert manager.missing_subtasks_for_full_episode() == [1]


def test_episode_dirs_for_full_subtask_uses_nested_segments_without_root_scan(
    monkeypatch,
    tmp_path,
):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    first = _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=2)
    second = _write_segment(root, full_idx=0, subtask_idx=1, subtask_total=2)
    monkeypatch.setattr(
        manager,
        "_iter_subtask_episode_dirs",
        lambda: (_ for _ in ()).throw(
            AssertionError("nested layout should not fall back to root scan")
        ),
    )

    assert manager._episode_dirs_for_full_subtask(0) == [first, second]
    assert manager._episode_dirs_for_full_subtask(0, 1) == [second]
    assert manager._saved_subtasks_cache[0] == {0, 1}


def test_update_task_info_license_toggle_does_not_rescan(monkeypatch, tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    manager._status = "idle"
    manager._current_full_episode_index = 3
    manager._current_subtask_index = 1
    manager._current_scenario_number = 1
    manager.current_instruction = "old instruction"
    manager._include_robotis_license = False
    monkeypatch.setattr(
        manager,
        "_validate_existing_segment_count",
        lambda: (_ for _ in ()).throw(
            AssertionError("same layout refresh must not rescan validation")
        ),
    )
    monkeypatch.setattr(
        manager,
        "_find_next_subtask_position",
        lambda: (_ for _ in ()).throw(
            AssertionError("same layout refresh must not recompute cursor")
        ),
    )
    task_info = SimpleNamespace(
        task_instruction=["updated instruction"],
        subtask_instruction=["subtask 0", "subtask 1"],
        include_robotis_license=True,
    )

    manager.update_task_info(task_info)

    assert manager._include_robotis_license is True
    assert manager._current_full_episode_index == 3
    assert manager._current_subtask_index == 1
    assert manager.current_instruction == "updated instruction"


def test_full_episode_archive_errors_report_corrupt_saved_segment(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=1)
    manager._current_full_episode_index = 0
    segment = _write_segment(root, full_idx=0, subtask_idx=0, subtask_total=1)
    (segment / "metadata.yaml").unlink()
    for mcap in segment.glob("*.mcap"):
        mcap.unlink()

    assert manager.full_episode_archive_errors() == [
        "subtask 0: missing metadata.yaml",
        "subtask 0: missing .mcap file",
    ]


def test_archive_writes_korean_subtask_instruction_as_utf8(tmp_path):
    root = tmp_path / "Task_1234_archive_MCAP"
    manager = _make_manager(root, subtask_total=2)
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=0,
        subtask_total=2,
        with_video=False,
        subtask_instruction="화장품 집기",
    )
    _write_segment(
        root,
        full_idx=0,
        subtask_idx=1,
        subtask_total=2,
        with_video=False,
        subtask_instruction="정리하기",
    )

    out = manager._archive_full_episode(0)

    raw = (out / "episode_info.json").read_text(encoding="utf-8")
    assert "화장품 집기" in raw
    assert "\\ud654" not in raw
