from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from interfaces.srv import RecordingCommand
from orchestrator.internal.communication.cyclo_data_client import CycloDataClient
from orchestrator.internal.communication.inference_collection import (
    make_inference_collection_id,
    resolve_inference_policy_type,
)
from orchestrator.internal.communication.recording_outcome import (
    forward_inference_record_stop,
)
import pytest


def _client_with_captured_call():
    client = CycloDataClient.__new__(CycloDataClient)
    client._recording = object()
    client._call = Mock(return_value=object())
    return client


def _captured_request(client):
    return client._call.call_args.args[1]


def test_recording_command_defaults_outcome_to_unspecified():
    client = _client_with_captured_call()

    client.send_recording_command(command=RecordingCommand.Request.START)

    assert _captured_request(client).episode_outcome == (
        RecordingCommand.Request.EPISODE_OUTCOME_UNSPECIFIED
    )
    assert _captured_request(client).collection_id == ''


def test_recording_command_forwards_collection_id():
    client = _client_with_captured_call()

    client.send_recording_command(
        command=RecordingCommand.Request.START,
        collection_id='ACT_dataset_20260810T010203_000004Z_abcd1234',
    )

    assert _captured_request(client).collection_id == (
        'ACT_dataset_20260810T010203_000004Z_abcd1234'
    )


def test_act_collection_id_is_utc_and_filesystem_safe():
    collection_id = make_inference_collection_id(
        'act',
        now=datetime(2026, 8, 10, 1, 2, 3, 4, tzinfo=timezone.utc),
        nonce='abcd1234',
    )

    assert collection_id == (
        'ACT_dataset_20260810T010203_000004Z_abcd1234'
    )


def test_legacy_lerobot_policy_resolves_from_checkpoint_config(tmp_path):
    (tmp_path / 'config.json').write_text('{"type": "smolvla"}')

    assert resolve_inference_policy_type(
        'lerobot',
        service_type='lerobot',
        policy_path=str(tmp_path),
    ) == 'smolvla'


def test_legacy_backend_defaults_are_concrete_policy_families():
    assert resolve_inference_policy_type(
        '', service_type='lerobot'
    ) == 'act'
    assert resolve_inference_policy_type(
        '', service_type='groot'
    ) == 'n17'


def test_recording_command_forwards_success_outcome():
    client = _client_with_captured_call()

    client.send_recording_command(
        command=RecordingCommand.Request.STOP,
        episode_outcome=RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS,
    )

    assert _captured_request(client).episode_outcome == (
        RecordingCommand.Request.EPISODE_OUTCOME_SUCCESS
    )


def test_recording_command_forwards_failure_outcome():
    client = _client_with_captured_call()

    client.send_recording_command(
        command=RecordingCommand.Request.STOP,
        episode_outcome=RecordingCommand.Request.EPISODE_OUTCOME_FAILURE,
    )

    assert _captured_request(client).episode_outcome == (
        RecordingCommand.Request.EPISODE_OUTCOME_FAILURE
    )


def test_inference_record_stop_forwards_request_outcome():
    expected = object()
    forward_recording = Mock(return_value=expected)
    task_info = object()
    request = SimpleNamespace(
        task_info=task_info,
        episode_outcome=RecordingCommand.Request.EPISODE_OUTCOME_FAILURE,
    )

    result = forward_inference_record_stop(request, forward_recording)

    assert result is expected
    forward_recording.assert_called_once_with(
        RecordingCommand.Request.STOP,
        task_info=task_info,
        episode_outcome=RecordingCommand.Request.EPISODE_OUTCOME_FAILURE,
    )


def test_inference_record_stop_rejects_invalid_outcome():
    forward_recording = Mock()
    request = SimpleNamespace(task_info=object(), episode_outcome=99)

    with pytest.raises(ValueError, match='Invalid episode outcome: 99'):
        forward_inference_record_stop(request, forward_recording)

    forward_recording.assert_not_called()


def test_inference_record_stop_rejects_unlabeled_outcome():
    forward_recording = Mock()
    request = SimpleNamespace(
        task_info=object(),
        episode_outcome=(
            RecordingCommand.Request.EPISODE_OUTCOME_UNSPECIFIED
        ),
    )

    with pytest.raises(
        ValueError,
        match='Inference recording outcome must be Success or Fail',
    ):
        forward_inference_record_stop(request, forward_recording)

    forward_recording.assert_not_called()
