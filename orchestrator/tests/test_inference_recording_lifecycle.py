import threading
from types import SimpleNamespace
from unittest.mock import Mock

from interfaces.msg import InferenceStatus, TaskInfo
from interfaces.srv import RecordingCommand, SendCommand

from orchestrator.internal.communication.communicator import Communicator
from orchestrator.orchestrator_node import OrchestratorNode


def _task_info(*, policy_type='act', service_type='lerobot'):
    info = TaskInfo()
    info.task_type = 'inference'
    info.policy_type = policy_type
    info.service_type = service_type
    info.record_inference_mode = True
    return info


def _node_for_lifecycle():
    node = object.__new__(OrchestratorNode)
    node._state_lock = threading.Lock()
    node._recording_command_lock = threading.Lock()
    node._inference_lifecycle_lock = threading.Lock()
    node._inference_record_collection_id = ''
    node._inference_session_closing = False
    node._inference_reconfiguration_in_progress = False
    node.container_service_client = None
    node.on_inference = False
    node.on_recording = False
    node.start_recording_time = 0.0
    node.params = {}
    node.communicator = None
    node.robot_type = 'ffw_sg2_rev1'
    node._last_ui_task_info = None
    node._prepared_inference_task_info = None
    node._loaded_inference_policy_path = ''
    node._loaded_inference_publish_to_robot = False
    node._loaded_inference_acceleration_mode = 'pytorch'
    node._loaded_inference_acceleration_engine_path = ''
    node._loaded_inference_action_request_mode = 'async'
    node.get_logger = Mock(return_value=Mock())
    return node


def test_communicator_exposes_separate_normal_and_inference_inventories():
    communicator = object.__new__(Communicator)
    communicator._mcap_topics = ['/joint_states', '/tf']
    communicator._inference_mcap_topics = [
        '/joint_states',
        '/tf',
        '/inference/action_chunk',
        '/inference/action_step_ack',
    ]

    assert communicator.get_mcap_topics() == ['/joint_states', '/tf']
    assert communicator.get_mcap_topics(inference=True) == [
        '/joint_states',
        '/tf',
        '/inference/action_chunk',
        '/inference/action_step_ack',
    ]


def test_activation_reuses_collection_and_teardown_clears_session(monkeypatch):
    node = _node_for_lifecycle()
    client = Mock()
    client._cancelled = threading.Event()
    node.container_service_client = client
    monkeypatch.setattr(
        'orchestrator.orchestrator_node.threading.Thread',
        lambda **_kwargs: SimpleNamespace(start=lambda: None),
    )

    assert node._activate_inference_session(client, _task_info()) is True
    collection_id = node._inference_record_collection_id
    assert collection_id.startswith('ACT_dataset_')
    assert node._activate_inference_session(client, _task_info()) is True
    assert node._inference_record_collection_id == collection_id

    assert node._teardown_inference_client(expected_client=client) is True

    assert node.container_service_client is None
    assert node.on_inference is False
    assert node._inference_record_collection_id == ''


def test_stale_failure_cannot_detach_or_publish_ready_for_new_client():
    node = _node_for_lifecycle()
    current_client = Mock()
    stale_client = Mock()
    node.container_service_client = current_client
    node.on_inference = True
    node._inference_record_collection_id = 'ACT_dataset_current'

    detached = node._teardown_inference_client(expected_client=stale_client)

    assert detached is False
    assert node.container_service_client is current_client
    assert node.on_inference is True
    assert node._inference_record_collection_id == 'ACT_dataset_current'


def test_closing_finish_serializes_and_rejects_late_joystick_start():
    node = _node_for_lifecycle()
    client = Mock()
    node.container_service_client = client
    node.on_inference = True
    node._inference_record_collection_id = 'ACT_dataset_active'
    task_info = _task_info()

    finish_entered = threading.Event()
    release_finish = threading.Event()
    calls = []

    def send_recording_command(**kwargs):
        if kwargs['command'] == RecordingCommand.Request.FINISH:
            finish_entered.set()
            assert release_finish.wait(timeout=2.0)
        calls.append(kwargs)
        accepted = not (
            kwargs['command'] == RecordingCommand.Request.START
            and not kwargs['collection_id']
        )
        return SimpleNamespace(
            success=True,
            response=SimpleNamespace(success=accepted),
            message='',
        )

    node._cyclo_data = SimpleNamespace(
        send_recording_command=send_recording_command,
    )
    finish_thread = threading.Thread(
        target=node._forward_recording,
        kwargs={
            'command': RecordingCommand.Request.FINISH,
            'task_info': task_info,
            'close_inference_session': True,
        },
    )
    start_thread = threading.Thread(
        target=node._forward_recording,
        kwargs={
            'command': RecordingCommand.Request.START,
            'task_info': task_info,
        },
    )

    finish_thread.start()
    assert finish_entered.wait(timeout=2.0)
    start_thread.start()
    release_finish.set()
    finish_thread.join(timeout=2.0)
    start_thread.join(timeout=2.0)

    assert not finish_thread.is_alive()
    assert not start_thread.is_alive()
    assert [call['command'] for call in calls] == [
        RecordingCommand.Request.FINISH,
        RecordingCommand.Request.START,
    ]
    assert calls[0]['collection_id'] == 'ACT_dataset_active'
    assert calls[1]['collection_id'] == ''
    assert node.on_recording is False


def test_delayed_joystick_command_cannot_target_new_collection():
    node = _node_for_lifecycle()
    node.container_service_client = Mock()
    node.on_inference = True
    node._inference_record_collection_id = 'ACT_dataset_new'
    calls = []

    def send_recording_command(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            response=SimpleNamespace(success=False),
            message='',
        )

    node._cyclo_data = SimpleNamespace(
        send_recording_command=send_recording_command,
    )

    node._forward_recording(
        RecordingCommand.Request.STOP,
        task_info=_task_info(),
        expected_collection_id='ACT_dataset_old',
    )

    assert calls[0]['collection_id'] == ''
    assert node.on_recording is False


def test_repeated_inference_right_trigger_cannot_save_unlabeled():
    node = _node_for_lifecycle()
    node._prepared_inference_task_info = _task_info()
    node._forward_recording = Mock()

    node._toggle_inference_trigger_recording(
        is_recording=True,
        expected_collection_id='ACT_dataset_active',
    )

    node._forward_recording.assert_not_called()
    node.get_logger.return_value.warning.assert_called_once_with(
        'Inference recording is already active; use Success or Fail '
        'in RL Data Collect to save it'
    )


def test_inference_clear_cannot_race_active_recording_into_unlabeled_save():
    node = _node_for_lifecycle()
    node.container_service_client = Mock()
    node.on_inference = True
    node.on_recording = True
    node._inference_record_collection_id = 'ACT_dataset_active'
    node._cyclo_data = Mock()

    result = node._forward_recording(
        RecordingCommand.Request.FINISH,
        task_info=_task_info(),
        close_inference_session=True,
    )

    assert result.success is False
    assert result.response is None
    assert result.message == (
        'Active inference recording must be saved with Success or Fail '
        'before clearing inference'
    )
    assert node._inference_session_closing is False
    assert node.on_recording is True
    node._cyclo_data.send_recording_command.assert_not_called()


def test_inference_reconfiguration_rejects_active_recording_atomically():
    node = _node_for_lifecycle()
    node.on_recording = True

    assert node._begin_inference_reconfiguration() is False
    assert node._inference_reconfiguration_in_progress is False

    node.on_recording = False
    node._inference_reconfiguration_in_progress = True
    assert node._begin_inference_reconfiguration() is False


def test_async_start_holds_reconfiguration_until_worker_finishes(monkeypatch):
    node = _node_for_lifecycle()
    node._client_cb_group = object()
    node.init_robot_control_parameters_from_user_task = Mock()
    node._publish_inference_phase = Mock()
    deferred_targets = []

    class FakeContainerServiceClient:
        CMD_LOAD = 'load'
        CMD_START = 'start'
        CMD_STOP = 'stop'
        CMD_UNLOAD = 'unload'

        def __init__(self, *, service_prefix, **_kwargs):
            self._service_prefix = service_prefix

        def connect(self):
            return None

        def inference_command(self, command, **_kwargs):
            if command == self.CMD_LOAD:
                return SimpleNamespace(
                    success=True,
                    message='loaded',
                    data={'action_keys': []},
                )
            if command == self.CMD_START:
                return SimpleNamespace(
                    success=True,
                    message='started',
                    data={},
                )
            raise AssertionError(f'unexpected command: {command}')

    class DeferredThread:
        def __init__(self, *, target, **_kwargs):
            self._target = target

        def start(self):
            deferred_targets.append(self._target)

    monkeypatch.setattr(
        'orchestrator.orchestrator_node.ContainerServiceClient',
        FakeContainerServiceClient,
    )
    monkeypatch.setattr(
        'orchestrator.orchestrator_node.threading.Thread',
        DeferredThread,
    )

    task_info = _task_info()
    task_info.policy_path = '/workspace/checkpoint/act'
    response = SimpleNamespace(success=False, message='')
    request = SimpleNamespace(
        command=SendCommand.Request.START_INFERENCE,
        task_info=task_info,
    )

    result = node.user_interaction_callback(request, response)

    assert result.success is True
    assert len(deferred_targets) == 1
    assert node._inference_reconfiguration_in_progress is True
    assert node._begin_inference_reconfiguration() is False

    deferred_targets[0]()

    assert node._inference_reconfiguration_in_progress is False
    assert node.on_inference is True


def test_inference_record_start_is_rejected_during_reconfiguration():
    node = _node_for_lifecycle()
    node.container_service_client = Mock()
    node.on_inference = True
    node._inference_record_collection_id = 'ACT_dataset_active'
    node._inference_reconfiguration_in_progress = True
    calls = []

    def send_recording_command(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            success=True,
            response=SimpleNamespace(success=False),
            message='',
        )

    node._cyclo_data = SimpleNamespace(
        send_recording_command=send_recording_command,
    )

    node._forward_recording(
        RecordingCommand.Request.START,
        task_info=_task_info(),
        include_topics=True,
    )

    assert calls[0]['collection_id'] == ''
    assert node.on_recording is False


def test_forward_recording_selects_inference_topic_inventory_from_task_info():
    node = _node_for_lifecycle()
    node.communicator = Mock()
    node.communicator.get_mcap_topics.side_effect = (
        lambda inference=False: (
            ['/joint_states', '/inference/action_chunk',
             '/inference/action_step_ack']
            if inference else ['/joint_states']
        )
    )
    calls = []
    node._cyclo_data = SimpleNamespace(
        send_recording_command=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                success=True,
                response=SimpleNamespace(success=True),
                message='',
            )
        ),
    )

    node._forward_recording(
        RecordingCommand.Request.SET_TASK_INFO,
        task_info=_task_info(),
        include_topics=True,
    )

    node.communicator.get_mcap_topics.assert_called_once_with(inference=True)
    assert calls[0]['topics'] == [
        '/joint_states',
        '/inference/action_chunk',
        '/inference/action_step_ack',
    ]


def test_forward_recording_keeps_policy_topics_out_of_normal_recording():
    node = _node_for_lifecycle()
    node.communicator = Mock()
    node.communicator.get_mcap_topics.side_effect = (
        lambda inference=False: (
            ['/joint_states', '/inference/action_chunk']
            if inference else ['/joint_states']
        )
    )
    calls = []
    node._cyclo_data = SimpleNamespace(
        send_recording_command=lambda **kwargs: (
            calls.append(kwargs)
            or SimpleNamespace(
                success=True,
                response=SimpleNamespace(success=True),
                message='',
            )
        ),
    )
    task_info = TaskInfo()
    task_info.task_type = 'record'

    node._forward_recording(
        RecordingCommand.Request.SET_TASK_INFO,
        task_info=task_info,
        include_topics=True,
    )

    node.communicator.get_mcap_topics.assert_called_once_with(inference=False)
    assert calls[0]['topics'] == ['/joint_states']


def test_stale_phase_is_not_published_after_teardown():
    node = _node_for_lifecycle()
    current_client = Mock()
    stale_client = Mock()
    node.container_service_client = current_client
    node.on_inference = True
    node._inference_record_collection_id = 'ACT_dataset_active'
    node._publish_inference_phase = Mock()

    published = node._publish_inference_phase_if_current(
        stale_client,
        InferenceStatus.INFERENCING,
    )

    assert published is False
    node._publish_inference_phase.assert_not_called()


def test_async_activation_cannot_reopen_a_closing_session():
    node = _node_for_lifecycle()
    client = Mock()
    node.container_service_client = client
    node._inference_record_collection_id = 'ACT_dataset_active'
    node._inference_session_closing = True

    activated = node._activate_inference_session(client, _task_info())

    assert activated is False
    assert node.on_inference is False
    assert node._inference_session_closing is True
    assert node._inference_record_collection_id == 'ACT_dataset_active'


def test_failed_record_only_close_does_not_poison_next_inference():
    node = _node_for_lifecycle()
    node._inference_session_closing = True

    node._restore_inference_session_after_failed_close()

    assert node._inference_session_closing is False
