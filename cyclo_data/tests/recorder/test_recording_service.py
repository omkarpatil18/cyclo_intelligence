from __future__ import annotations

from pathlib import Path
from threading import RLock
from types import ModuleType, SimpleNamespace
import json
import sys

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "cyclo_data"))
sys.path.insert(0, str(_REPO_ROOT / "orchestrator"))
sys.path.insert(0, str(_REPO_ROOT / "shared"))

import cyclo_data  # noqa: E402
import cyclo_data.recorder  # noqa: E402
import cyclo_data.services  # noqa: E402


def _stub_module(name: str, **attrs) -> None:
    if name in sys.modules:
        module = sys.modules[name]
        for key, value in attrs.items():
            setattr(module, key, value)
        return
    parts = name.split(".")
    for idx in range(1, len(parts)):
        parent = ".".join(parts[:idx])
        sys.modules.setdefault(parent, ModuleType(parent))
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


class _RecordingStatus:
    READY = 0
    RECORDING = 1
    SAVING = 2


class _DataOperationStatus:
    IDLE = 0
    RUNNING = 1
    COMPLETED = 2
    FAILED = 3
    CANCELLED = 4


class _RecordingCommand:
    class Request:
        START = 0
        STOP = 1
        PAUSE = 2
        RESUME = 3
        FINISH = 4
        MOVE_TO_NEXT = 5
        RERECORD = 6
        SKIP_TASK = 7
        CANCEL = 8
        REFRESH_TOPICS = 9
        START_SEGMENT = 10
        STOP_SEGMENT = 11
        DISCARD_SEGMENT = 12
        FINISH_EPISODE = 13
        DISCARD_EPISODE = 14
        SET_TASK_INFO = 15
        CANCEL_SEGMENT = 16
        EPISODE_OUTCOME_UNSPECIFIED = 0
        EPISODE_OUTCOME_SUCCESS = 1
        EPISODE_OUTCOME_FAILURE = 2


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


_stub_module(
    "interfaces.msg",
    DataOperationStatus=_DataOperationStatus,
    RecordingStatus=_RecordingStatus,
)
_stub_module("interfaces.srv", RecordingCommand=_RecordingCommand)
_stub_module("cyclo_data.recorder.camera_info_snapshot", CameraInfoSnapshot=_Dummy)
_stub_module("cyclo_data.recorder.rosbag_control", RosbagControl=_Dummy)
_stub_module("cyclo_data.recorder.transcoder", TranscodeWorker=_Dummy)
_stub_module("cyclo_data.recorder.video_recorder", VideoRecorder=_Dummy)
_stub_module("huggingface_hub", HfApi=_Dummy)
_stub_module("cyclo_data.converter.orchestrator", DataConverter=_Dummy)
_stub_module(
    "cyclo_data.hub.progress_tracker",
    HuggingFaceLogCapture=_Dummy,
    HuggingFaceProgressTqdm=_Dummy,
)
_stub_module("psutil", cpu_percent=lambda interval=None: 0.0)

import cyclo_data.services.recording_service as recording_service_module  # noqa: E402
from cyclo_data.services.recording_service import RecordingService  # noqa: E402

for _module_name in (
    "cyclo_data.recorder.camera_info_snapshot",
    "cyclo_data.recorder.rosbag_control",
    "cyclo_data.recorder.transcoder",
    "cyclo_data.recorder.video_recorder",
):
    sys.modules.pop(_module_name, None)
    _parent_name, _attr_name = _module_name.rsplit(".", 1)
    _parent = sys.modules.get(_parent_name)
    if _parent is not None and hasattr(_parent, _attr_name):
        delattr(_parent, _attr_name)


def _request(
    segment_index=0,
    tags=None,
    task_type='',
    collection_id='',
    episode_outcome=0,
    **attrs,
):
    task_info = attrs.pop('task_info', None) or SimpleNamespace(
        task_num='',
        task_name='',
        task_type=task_type,
        policy_type='',
        tags=tags or [],
    )
    return SimpleNamespace(
        segment_index=segment_index,
        task_info=task_info,
        collection_id=collection_id,
        episode_outcome=episode_outcome,
        **attrs,
    )


def test_discard_episode_segment_index_zero_keeps_legacy_cursor_behavior():
    assert RecordingService._extract_full_episode_index(_request(0)) is None


def test_discard_episode_segment_index_encodes_full_episode_index_plus_one():
    assert RecordingService._extract_full_episode_index(_request(1)) == 0
    assert RecordingService._extract_full_episode_index(_request(8)) == 7


def test_discard_episode_accepts_transitional_explicit_target_fields():
    req = _request(0, has_full_episode_index=True, full_episode_index=7)
    assert RecordingService._extract_full_episode_index(req) == 7


def test_discard_episode_accepts_transitional_target_tag():
    req = _request(0, tags=["recording_full_episode_index:7"])
    assert RecordingService._extract_full_episode_index(req) == 7


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warn(self, message):
        self.warnings.append(message)

    def warning(self, message):
        self.warnings.append(message)

    def info(self, message):
        pass

    def error(self, message):
        self.errors.append(message)


def _service_with_logger():
    logger = _Logger()
    service = RecordingService.__new__(RecordingService)
    service._node = SimpleNamespace(get_logger=lambda: logger)
    service._data_manager = None
    return service, logger


def test_ensure_data_manager_reuses_same_task_without_candidate_scan(
    monkeypatch,
    tmp_path,
):
    service, _ = _service_with_logger()
    service._session_lock = RLock()
    service.DEFAULT_SAVE_ROOT_PATH = tmp_path
    updates = []
    existing = SimpleNamespace(
        _save_repo_name="Task_42_pick_MCAP",
        _save_path=tmp_path / "Task_42_pick_MCAP",
        _task_type="",
        _collection_id='',
        _robot_type="ffw_sg2_rev1",
        is_recording=lambda: False,
        update_task_info=lambda task_info: updates.append(task_info),
    )
    service._data_manager = existing

    class ExplodingDataManager:
        @classmethod
        def _make_save_repo_name(
            cls, save_root_path, task_info, collection_id=''
        ):
            assert save_root_path == tmp_path
            return "Task_42_pick_MCAP"

        def __init__(self, *args, **kwargs):
            raise AssertionError(
                "same task must not construct a scan-heavy manager"
            )

    monkeypatch.setattr(
        recording_service_module,
        "DataManager",
        ExplodingDataManager,
    )
    task_info = SimpleNamespace(task_num="42", task_name="pick")

    assert service._ensure_data_manager(task_info, "ffw_sg2_rev1") is existing
    assert updates == [task_info]


def test_ensure_data_manager_reuses_same_inference_collection(
    monkeypatch,
    tmp_path,
):
    service, _ = _service_with_logger()
    service._session_lock = RLock()
    service.DEFAULT_SAVE_ROOT_PATH = tmp_path
    collection_id = "ACT_dataset_20260810T010203_000004Z_abcd1234"
    updates = []
    existing = SimpleNamespace(
        _save_repo_name=f"{collection_id}_MCAP",
        _save_path=tmp_path / "inference" / f"{collection_id}_MCAP",
        _task_type="inference",
        _collection_id=collection_id,
        _robot_type="ffw_sg2_rev1",
        is_recording=lambda: False,
        update_task_info=lambda task_info: updates.append(task_info),
    )
    service._data_manager = existing

    class ExplodingDataManager:
        @classmethod
        def _make_save_repo_name(
            cls, save_root_path, task_info, collection_id=''
        ):
            return f"{collection_id}_MCAP"

        def __init__(self, *args, **kwargs):
            raise AssertionError("same collection must reuse its manager")

    monkeypatch.setattr(
        recording_service_module,
        "DataManager",
        ExplodingDataManager,
    )
    task_info = SimpleNamespace(
        task_num="",
        task_name="",
        task_type="inference",
    )

    assert service._ensure_data_manager(
        task_info,
        "ffw_sg2_rev1",
        collection_id=collection_id,
    ) is existing
    assert updates == [task_info]


def test_ensure_data_manager_does_not_reuse_cross_type_basename_collision(
    monkeypatch,
    tmp_path,
):
    service, _ = _service_with_logger()
    service._session_lock = RLock()
    service.DEFAULT_SAVE_ROOT_PATH = tmp_path
    collision_id = "Task_1_name"
    existing = SimpleNamespace(
        _save_repo_name=f"{collision_id}_MCAP",
        _save_path=tmp_path / "inference" / f"{collision_id}_MCAP",
        _task_type="inference",
        _collection_id=collision_id,
        _robot_type="ffw_sg2_rev1",
        is_recording=lambda: False,
        update_task_info=lambda _task_info: (_ for _ in ()).throw(
            AssertionError("cross-type manager must not be reused")
        ),
    )
    service._data_manager = existing

    class FakeDataManager:
        @classmethod
        def _make_save_repo_name(
            cls, save_root_path, task_info, collection_id=''
        ):
            if getattr(task_info, "task_type", "") == "inference":
                return f"{collection_id}_MCAP"
            return f"Task_{task_info.task_num}_{task_info.task_name}_MCAP"

        def __init__(self, save_root_path, robot_type, task_info, collection_id=''):
            self._save_repo_name = self._make_save_repo_name(
                save_root_path, task_info, collection_id
            )
            self._task_type = getattr(task_info, "task_type", "")
            self._collection_id = collection_id if self._task_type == "inference" else ""
            root = Path(save_root_path)
            if self._task_type == "inference":
                root = root / "inference"
            self._save_path = root / self._save_repo_name
            self._robot_type = robot_type
            self._recovered_episode_dirs = []

    monkeypatch.setattr(
        recording_service_module,
        "DataManager",
        FakeDataManager,
    )
    task_info = SimpleNamespace(
        task_num="1",
        task_name="name",
        task_type="record",
    )

    manager = service._ensure_data_manager(task_info, "ffw_sg2_rev1")

    assert manager is not existing
    assert manager._task_type == "record"
    assert manager._save_path == tmp_path / "Task_1_name_MCAP"


def test_episode_outcome_contract_rejects_invalid_or_non_inference_labels():
    service, _ = _service_with_logger()
    response = SimpleNamespace(success=True, message="")

    assert service._validate_episode_outcome_request(
        _request(
            command=_RecordingCommand.Request.STOP,
            episode_outcome=99,
            task_type="inference",
        ),
        response,
        "STOP",
    ) is False
    assert response.message == "Invalid episode_outcome: 99"

    response = SimpleNamespace(success=True, message="")
    assert service._validate_episode_outcome_request(
        _request(
            command=_RecordingCommand.Request.STOP,
            episode_outcome=_RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS,
            task_type="record",
        ),
        response,
        "STOP",
    ) is False
    assert "only valid for inference" in response.message


def test_episode_outcome_contract_accepts_inference_stop_label():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _task_type="inference",
        _collection_id="ACT_dataset_session",
        is_recording=lambda: True,
    )
    response = SimpleNamespace(success=True, message="")

    accepted = service._validate_episode_outcome_request(
        _request(
            command=_RecordingCommand.Request.STOP,
            episode_outcome=_RecordingCommand.Request.EPISODE_OUTCOME_FAILURE,
            task_type="inference",
        ),
        response,
        "STOP",
    )

    assert accepted is True
    assert service._episode_outcome_metadata(
        _request(
            task_type="inference",
            episode_outcome=_RecordingCommand.Request.EPISODE_OUTCOME_FAILURE,
        )
    ) == {
        "schema_version": 1,
        "status": "failure",
        "success": False,
        "source": "operator_ui",
    }


def test_collection_contract_rejects_stale_stop_but_allows_fresh_start():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _task_type="inference",
        _collection_id="ACT_dataset_old",
        is_recording=lambda: False,
    )

    response = SimpleNamespace(success=True, message="")
    stale_stop = service._validate_collection_request(
        _request(
            command=_RecordingCommand.Request.STOP,
            task_type="inference",
            collection_id="ACT_dataset_new",
        ),
        response,
        "STOP",
    )
    assert stale_stop is False
    assert "stale inference collection" in response.message

    response = SimpleNamespace(success=True, message="")
    fresh_start = service._validate_collection_request(
        _request(
            command=_RecordingCommand.Request.START,
            task_type="inference",
            collection_id="ACT_dataset_new",
        ),
        response,
        "START",
    )
    assert fresh_start is True


def test_collection_contract_rejects_noncanonical_whitespace():
    service, _ = _service_with_logger()
    response = SimpleNamespace(success=True, message="")

    accepted = service._validate_collection_request(
        _request(
            command=_RecordingCommand.Request.START,
            task_type="inference",
            collection_id=" ACT_dataset_session ",
        ),
        response,
        "START",
    )

    assert accepted is False
    assert "Invalid inference collection_id" in response.message


def test_collection_contract_rejects_task_type_mismatch_both_directions():
    service, _ = _service_with_logger()
    response = SimpleNamespace(success=True, message="")
    service._data_manager = SimpleNamespace(
        _task_type="inference",
        _collection_id="ACT_dataset_session",
        is_recording=lambda: True,
    )
    assert service._validate_collection_request(
        _request(command=_RecordingCommand.Request.STOP, task_type="record"),
        response,
        "STOP",
    ) is False

    response = SimpleNamespace(success=True, message="")
    service._data_manager = SimpleNamespace(
        _task_type="record",
        _collection_id="",
        is_recording=lambda: True,
    )
    assert service._validate_collection_request(
        _request(
            command=_RecordingCommand.Request.STOP,
            task_type="inference",
            collection_id="ACT_dataset_session",
        ),
        response,
        "STOP",
    ) is False
    assert "does not match active record manager" in response.message


def test_inference_label_cannot_be_written_into_record_manager():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _task_type="record",
        is_recording=lambda: True,
    )
    response = SimpleNamespace(success=True, message="")

    accepted = service._validate_episode_outcome_request(
        _request(
            command=_RecordingCommand.Request.STOP,
            task_type="inference",
            episode_outcome=_RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS,
        ),
        response,
        "STOP",
    )

    assert accepted is False
    assert "requires an active inference manager" in response.message
    assert service._episode_outcome_metadata(
        _request(
            task_type="inference",
            episode_outcome=_RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS,
        )
    ) is None


def test_labeled_stop_without_active_recording_fails():
    service, _ = _service_with_logger()
    response = SimpleNamespace(success=True, message="")

    result = service._do_stop_and_save(
        _request(
            command=_RecordingCommand.Request.STOP,
            task_type="inference",
            episode_outcome=_RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS,
        ),
        response,
        "STOP",
        event="finish",
    )

    assert result.success is False
    assert result.message == "STOP: no active recording session"


def test_validate_active_segment_rejects_stale_segment_request():
    service, logger = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        get_current_subtask_index=lambda: 1,
    )
    response = SimpleNamespace(success=True, message="")

    ok = service._validate_active_segment(
        _request(segment_index=2),
        response,
        "STOP_SEGMENT",
    )

    assert ok is False
    assert response.success is False
    assert response.message == "STOP_SEGMENT: active subtask is 1, but request targeted 2"
    assert logger.warnings == [response.message]


def test_validate_active_segment_accepts_current_segment_request():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        get_current_subtask_index=lambda: 1,
    )
    response = SimpleNamespace(success=True, message="")

    ok = service._validate_active_segment(
        _request(segment_index=1),
        response,
        "STOP_SEGMENT",
    )

    assert ok is True
    assert response.success is True


def test_start_segment_rejects_when_recording_is_already_active():
    service, logger = _service_with_logger()
    service._finish_episode_in_progress = lambda: False
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    data_manager = SimpleNamespace(
        is_recording=lambda: True,
        set_current_subtask_index=lambda index: (_ for _ in ()).throw(
            AssertionError("must not change subtask while recording")
        ),
    )
    service._ensure_data_manager = (
        lambda task_info, robot_type, collection_id='': data_manager
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(
        segment_index=2,
        command=_RecordingCommand.Request.START_SEGMENT,
        robot_type="ffw_sg2_rev1",
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert response.message == "START blocked: recording already active"
    assert logger.warnings == [response.message]


def test_start_retries_failed_archive_before_reusing_episode_slot():
    service, _ = _service_with_logger()
    manager = SimpleNamespace()
    service._data_manager = manager
    service._finish_episode_error = "summary fsync failed"
    service._finish_episode_in_progress = lambda: False
    retries = []
    service._start_finish_episode_thread = lambda value: retries.append(value) or True
    response = SimpleNamespace(success=True, message="")

    result = service._do_start(
        _request(
            command=_RecordingCommand.Request.START,
            robot_type="ffw_sg2_rev1",
        ),
        response,
    )

    assert result.success is False
    assert result.message == (
        "START blocked: retrying failed episode archive "
        "(summary fsync failed)"
    )
    assert retries == [manager]


def test_set_task_info_cannot_replace_manager_during_archive():
    service, _ = _service_with_logger()
    service._finish_episode_in_progress = lambda: True
    service._ensure_data_manager = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("manager must not be replaced during archive")
    )
    response = SimpleNamespace(success=True, message="")

    result = service._do_set_task_info(
        _request(
            command=_RecordingCommand.Request.SET_TASK_INFO,
            robot_type="ffw_sg2_rev1",
        ),
        response,
    )

    assert result.success is False
    assert result.message == "SET_TASK_INFO blocked: episode archive still running"


def test_failed_start_cleanup_preserves_preexisting_episode(tmp_path):
    service, _ = _service_with_logger()
    episode_dir = tmp_path / "0" / "segments" / "0"
    episode_dir.mkdir(parents=True)
    preserved = episode_dir / "episode_info.json"
    preserved.write_text('{"outcome":{"status":"success"}}')
    service._video_recorder = None
    service._camera_info = None
    service._rosbag = SimpleNamespace()

    service._cleanup_failed_start(
        episode_dir=episode_dir,
        data_manager=None,
        rosbag_started=False,
        episode_preexisting=True,
    )

    assert preserved.exists()


def test_workspace_scan_reopens_orphaned_inference_collection(
    monkeypatch,
    tmp_path,
):
    collection_id = "ACT_dataset_20260810T010203_000004Z_abcd1234"
    segment_dir = (
        tmp_path / "inference" / f"{collection_id}_MCAP"
        / "0" / "segments" / "0"
    )
    segment_dir.mkdir(parents=True)
    info = {
        "collection_id": collection_id,
        "task_type": "inference",
        "policy_type": "act",
        "task_instruction": "ACT_dataset",
        "subtask_instructions": [],
        "robot_type": "ffw_sg2_rev1",
    }
    (segment_dir / "episode_info.json").write_text(json.dumps(info))
    recovered_dir = segment_dir.parents[1]
    constructed = []

    class FakeDataManager:
        @staticmethod
        def _read_episode_info(path):
            return json.loads((Path(path) / "episode_info.json").read_text())

        def __init__(self, **kwargs):
            constructed.append(kwargs)
            self._recovered_episode_dirs = [recovered_dir]

    monkeypatch.setattr(recording_service_module, "DataManager", FakeDataManager)
    service, _ = _service_with_logger()

    recovered = service._recover_pending_inference_archives(tmp_path)

    assert recovered == [recovered_dir]
    assert constructed[0]["collection_id"] == collection_id
    assert constructed[0]["task_info"].policy_type == "act"
    assert constructed[0]["robot_type"] == "ffw_sg2_rev1"


def test_start_segment_rejects_request_that_skips_next_missing_subtask():
    service, logger = _service_with_logger()
    service._finish_episode_in_progress = lambda: False
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [1, 2],
        set_current_subtask_index=lambda index: (_ for _ in ()).throw(
            AssertionError("must not jump over a missing subtask")
        ),
    )
    service._ensure_data_manager = (
        lambda task_info, robot_type, collection_id='': data_manager
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(
        segment_index=2,
        command=_RecordingCommand.Request.START_SEGMENT,
        robot_type="ffw_sg2_rev1",
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert response.message == (
        "START_SEGMENT: next available subtask is 1, but request targeted 2"
    )
    assert logger.warnings == [response.message]


def test_start_segment_rejects_when_current_episode_is_already_complete():
    service, logger = _service_with_logger()
    service._finish_episode_in_progress = lambda: False
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [],
        set_current_subtask_index=lambda index: (_ for _ in ()).throw(
            AssertionError("must not restart a complete episode")
        ),
    )
    service._ensure_data_manager = (
        lambda task_info, robot_type, collection_id='': data_manager
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(
        segment_index=1,
        command=_RecordingCommand.Request.START_SEGMENT,
        robot_type="ffw_sg2_rev1",
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert response.message == (
        "START_SEGMENT: current episode already has all subtasks; "
        "finish or discard episode before starting again"
    )
    assert logger.warnings == [response.message]


def test_finish_episode_rejects_missing_subtasks_before_archive_thread():
    service, logger = _service_with_logger()
    service._data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [1],
    )
    service._start_finish_episode_thread = lambda data_manager: (_ for _ in ()).throw(
        AssertionError("archive thread must not start with missing subtasks")
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(command=_RecordingCommand.Request.FINISH_EPISODE)

    result = service._do_finish_episode(request, response)

    assert result is response
    assert response.success is False
    assert response.message == "FINISH_EPISODE: missing subtask(s) [1]"
    assert logger.warnings == [response.message]


def test_stop_segment_rejects_when_no_active_recording():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(is_recording=lambda: False)
    response = SimpleNamespace(success=True, message="")
    request = _request(command=_RecordingCommand.Request.STOP_SEGMENT)

    result = service._do_stop_and_save(
        request,
        response,
        "STOP_SEGMENT",
        event="finish",
    )

    assert result is response
    assert response.success is False
    assert response.message == "STOP_SEGMENT: no active recording"


def test_stop_segment_saves_metadata_even_without_urdf_path(tmp_path):
    service, _ = _service_with_logger()
    episode_dir = tmp_path / "0" / "segments" / "0"
    metadata_calls = []
    stopped = []
    events = []

    data_manager = SimpleNamespace(
        _record_episode_count=0,
        _segmented_storage_mode=True,
        is_recording=lambda: True,
        get_current_subtask_index=lambda: 0,
        get_status=lambda: "recording",
        get_save_rosbag_path=lambda: str(episode_dir),
        save_robotis_metadata=lambda **kwargs: metadata_calls.append(kwargs),
        stop_recording=lambda **kwargs: stopped.append(kwargs),
    )
    service._data_manager = data_manager
    service._rosbag = SimpleNamespace(
        stop_rosbag=lambda: None,
        publish_action_event=lambda event: events.append(event),
    )
    service._video_recorder = None
    service._camera_info = None
    service._last_camera_rotations = {"cam0": 0}
    service._last_image_topics = {"cam0": "/image"}
    service._last_camera_info_topics = {"cam0": "/camera_info"}
    service._publish_umbrella_status = lambda *args, **kwargs: None
    response = SimpleNamespace(success=True, message="")
    request = _request(
        command=_RecordingCommand.Request.STOP_SEGMENT,
        segment_index=0,
        urdf_path="",
    )

    result = service._do_stop_and_save(
        request,
        response,
        "STOP_SEGMENT",
        event="finish",
    )

    assert result is response
    assert response.success is True
    assert response.message == "Subtask saved"
    assert len(metadata_calls) == 1
    assert metadata_calls[0]["urdf_path"] == ""
    assert metadata_calls[0]["camera_rotations"] == {"cam0": 0}
    assert stopped == [{"finish_full_episode": False}]
    assert events == ["finish"]


def test_metadata_write_failure_does_not_advance_episode(tmp_path):
    service, _ = _service_with_logger()
    episode_dir = tmp_path / "0" / "segments" / "0"
    stopped = []
    data_manager = SimpleNamespace(
        _record_episode_count=0,
        _segmented_storage_mode=True,
        is_recording=lambda: True,
        get_status=lambda: "recording",
        get_save_rosbag_path=lambda: str(episode_dir),
        save_robotis_metadata=lambda **kwargs: (_ for _ in ()).throw(
            OSError("metadata disk error")
        ),
        stop_recording=lambda **kwargs: stopped.append(kwargs),
    )
    service._data_manager = data_manager
    service._rosbag = SimpleNamespace(stop_rosbag=lambda: None)
    service._video_recorder = None
    service._camera_info = None
    service._last_camera_rotations = {}
    service._last_image_topics = {}
    service._last_camera_info_topics = {}

    with pytest.raises(OSError, match="metadata disk error"):
        service._do_stop_and_save(
            _request(
                command=_RecordingCommand.Request.STOP,
                task_type="inference",
                episode_outcome=(
                    _RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS
                ),
            ),
            SimpleNamespace(success=True, message=""),
            "STOP",
            event="finish",
        )

    assert stopped == []


def test_start_segment_rolls_back_rosbag_when_writer_start_fails(tmp_path):
    service, logger = _service_with_logger()
    episode_dir = tmp_path / "0" / "segments" / "0"
    rosbag_calls = []
    stopped_writers = []
    started = []

    class FailingVideoRecorder:
        def start_episode(self, _episode_dir):
            raise RuntimeError("camera writer boom")

        def stop_episode(self):
            stopped_writers.append(True)
            return {}

    data_manager = SimpleNamespace(
        _segmented_storage_mode=True,
        is_recording=lambda: False,
        missing_subtasks_for_full_episode=lambda: [0],
        set_current_subtask_index=lambda index: None,
        get_save_rosbag_path=lambda allow_idle=False: str(episode_dir),
        start_recording=lambda: started.append(True),
    )
    service._finish_episode_in_progress = lambda: False
    service._ensure_data_manager = (
        lambda task_info, robot_type, collection_id='': data_manager
    )
    service._ensure_video_pipeline = lambda robot_type: None
    service._last_prepared_topics = ()
    service._video_recorder = FailingVideoRecorder()
    service._camera_info = None
    service._publish_umbrella_status = lambda *args, **kwargs: None

    def start_rosbag(rosbag_uri):
        rosbag_calls.append(("start", rosbag_uri))
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "partial.mcap").write_bytes(b"partial")

    service._rosbag = SimpleNamespace(
        is_available=lambda: True,
        start_rosbag=start_rosbag,
        stop_and_delete_rosbag=lambda: rosbag_calls.append(("stop_delete", None)),
    )
    response = SimpleNamespace(success=True, message="")
    request = _request(
        command=_RecordingCommand.Request.START_SEGMENT,
        segment_index=0,
        robot_type="ffw_sg2_rev1",
        topics=[],
    )

    result = service._do_start(request, response)

    assert result is response
    assert response.success is False
    assert "camera writer boom" in response.message
    assert rosbag_calls == [
        ("start", str(episode_dir)),
        ("stop_delete", None),
    ]
    assert stopped_writers == [True]
    assert started == []
    assert not episode_dir.exists()
    assert logger.errors == [response.message]


def test_refresh_topics_caches_robot_type_for_idle_status():
    service, _ = _service_with_logger()
    prepared_topics = []
    video_robot_types = []
    published = []

    service._session_lock = RLock()
    service._robot_type = ''
    service._data_manager = None
    service._rosbag = SimpleNamespace(is_available=lambda: True)
    service._prepare_rosbag_topics = lambda topics: prepared_topics.extend(topics)
    service._ensure_video_pipeline = lambda robot_type: video_robot_types.append(robot_type)
    service._recording_status_pub = SimpleNamespace(
        publish=lambda status: published.append(status)
    )
    service._video_recorder = None
    service._cpu_checker = SimpleNamespace(get_cpu_usage=lambda: 0.0)
    response = SimpleNamespace(success=False, message='')
    request = _request(
        robot_type='ffw_sg2_rev2',
        topics=['/joint_states'],
    )

    result = service._do_refresh_topics(request, response)
    service._publish_recording_status()

    assert result is response
    assert response.success is True
    assert service._robot_type == 'ffw_sg2_rev2'
    assert prepared_topics == ['/joint_states']
    assert video_robot_types == ['ffw_sg2_rev2']
    assert published[-1].robot_type == 'ffw_sg2_rev2'


def test_prepare_topics_caches_only_after_rosbag_accepts_inventory():
    service, _ = _service_with_logger()
    service._last_prepared_topics = ('/old_topic',)
    attempts = []

    def fail_prepare(*, topics):
        attempts.append(list(topics))
        raise RuntimeError('prepare rejected')

    service._rosbag = SimpleNamespace(prepare_rosbag=fail_prepare)

    with pytest.raises(RuntimeError, match='prepare rejected'):
        service._prepare_rosbag_topics(['/joint_states', '/tf'])

    assert service._last_prepared_topics == ('/old_topic',)

    service._rosbag = SimpleNamespace(
        prepare_rosbag=lambda *, topics: attempts.append(list(topics)),
    )
    service._prepare_rosbag_topics(['/joint_states', '/tf'])

    assert attempts == [
        ['/joint_states', '/tf'],
        ['/joint_states', '/tf'],
    ]
    assert service._last_prepared_topics == ('/joint_states', '/tf')


def test_recording_status_prefers_current_service_robot_type_over_old_manager():
    service, _ = _service_with_logger()
    published = []

    service._session_lock = RLock()
    service._robot_type = 'ffw_sg2_rev2'
    service._data_manager = SimpleNamespace(
        get_current_record_status=lambda: SimpleNamespace(
            robot_type='ffw_sg2_rev1',
            record_phase=_RecordingStatus.READY,
        )
    )
    service._video_recorder = None
    service._recording_status_pub = SimpleNamespace(
        publish=lambda status: published.append(status)
    )

    service._publish_recording_status()

    assert published[-1].robot_type == 'ffw_sg2_rev2'


def test_cancel_segment_rejects_when_no_active_recording():
    service, _ = _service_with_logger()
    service._data_manager = SimpleNamespace(is_recording=lambda: False)
    service._publish_umbrella_status = lambda *args, **kwargs: None
    response = SimpleNamespace(success=True, message="")
    request = _request(command=_RecordingCommand.Request.CANCEL_SEGMENT)

    result = service._do_cancel(request, response)

    assert result is response
    assert response.success is False
    assert response.message == "CANCEL_SEGMENT: no active recording"
