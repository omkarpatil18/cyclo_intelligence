# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Admission-gate tests with an in-memory MCAP reader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from cyclo_data.validation.inference_episode import (
    ValidationContract,
    _expected_rl_contract,
    validate_inference_episode,
)
from cyclo_data.validation.scripts.validate_inference_episode import main


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


class FakeFrameTimestamps:
    def __init__(self, stamps):
        self.frame_index = list(range(len(stamps)))
        self.header_stamp_ns = [round(value * 1e9) for value in stamps]
        self.recv_ns = [round(value * 1e9) for value in stamps]


class FakeBagReader:
    def __init__(self, topic_types, messages):
        self._topic_types = topic_types
        self._messages = messages
        self.closed = False

    def open(self):
        return True

    def get_topic_types(self):
        return dict(self._topic_types)

    def read_messages(self, topic_filter=None):
        allowed = set(topic_filter or self._messages)
        records = []
        for topic, stream in self._messages.items():
            if topic in allowed:
                records.extend((topic, msg, stamp) for msg, stamp in stream)
        yield from sorted(records, key=lambda item: item[2])

    def close(self):
        self.closed = True


def _state_message(contract):
    joint_names = contract.state[0].names
    return _ns(
        name=list(joint_names),
        position=[0.01 * index for index in range(len(joint_names))],
    )


def _odom_message():
    return _ns(
        twist=_ns(
            twist=_ns(
                linear=_ns(x=0.0, y=0.0, z=0.0),
                angular=_ns(x=0.0, y=0.0, z=0.0),
            )
        )
    )


def _action_message(spec, values):
    if spec.msg_type == 'geometry_msgs/msg/Twist':
        return _ns(
            linear=_ns(x=values[0], y=values[1], z=0.0),
            angular=_ns(x=0.0, y=0.0, z=values[2]),
        )
    return _ns(
        joint_names=list(spec.names),
        points=[_ns(positions=list(values))],
    )


def _write_episode(tmp_path, contract):
    root = tmp_path / 'collection_MCAP' / '0'
    video_dir = root / 'videos' / '0_0'
    checkpoint = tmp_path / 'checkpoint' / 'checkpoints' / '080000' / 'pretrained_model'
    video_dir.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    (root / '0_0.mcap').write_bytes(b'synthetic-mcap')
    config_bytes = b'{"type":"act","chunk_size":30}'
    (checkpoint / 'config.json').write_bytes(config_bytes)
    (checkpoint / 'model.safetensors').write_bytes(b'weights')

    start = 1000.0
    state_stamps = [start + index / 15 for index in range(32)]
    frame_data = {}
    frame_counts = {}
    for camera in contract.cameras:
        mp4 = video_dir / f'{camera}.mp4'
        sidecar = video_dir / f'{camera}_timestamps.parquet'
        stats_path = video_dir / f'{camera}_recorder_stats.json'
        mp4.write_bytes(b'fake-mp4')
        sidecar.write_bytes(b'fake-parquet')
        count = len(state_stamps)
        frame_data[sidecar] = FakeFrameTimestamps(state_stamps)
        frame_counts[mp4] = count
        stats_path.write_text(json.dumps({
            'frames_received': count,
            'frames_written': count,
            'frames_metadata_written': count,
            'frames_remuxed': count,
            'frames_dropped_invalid': 0,
            'frames_dropped_queue': 0,
            'raw_write_error': None,
            'metadata_error': None,
            'remux_error': None,
            'remux_status': 'done',
        }))

    artifacts = [
        {'name': 'config.json', 'size_bytes': len(config_bytes)},
        {'name': 'model.safetensors', 'size_bytes': len(b'weights')},
    ]
    manifest = json.dumps(
        artifacts, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode()
    collection_id = 'ACT_dataset_test_collection'
    episode = {
        'format_version': 'robotis_v2',
        'task_type': 'inference',
        'policy_type': 'act',
        'robot_type': contract.robot_type,
        'collection_id': collection_id,
        'transcoding_status': 'done',
        'video_remux_status': 'done',
        'transcoding_cameras_failed': {},
        'outcome': {
            'schema_version': 1,
            'status': 'success',
            'success': True,
            'source': 'operator_ui',
            'annotated_at': '2026-08-11T00:00:00Z',
        },
        'policy_provenance': {
            'schema_version': 1,
            'service_type': 'lerobot',
            'policy_type': 'act',
            'policy_path': str(checkpoint),
            'checkpoint_id': '080000',
            'control_hz': 15,
            'inference_hz': 15,
            'config_path': 'config.json',
            'config_sha256': hashlib.sha256(config_bytes).hexdigest(),
            'artifact_manifest_sha256': hashlib.sha256(manifest).hexdigest(),
            'artifacts': artifacts,
        },
        'rl_episode_contract': _expected_rl_contract(collection_id, contract),
        'video_segments': [{
            'mcap': '0_0.mcap',
            'video_dir': 'videos/0_0',
            'cameras': list(contract.cameras),
            'raw_cameras': list(contract.cameras),
        }],
    }
    (root / 'episode_info.json').write_text(json.dumps(episode, indent=2))

    rows = [
        [(row * contract.action_dim + column) / 10000.0
         for column in range(contract.action_dim)]
        for row in range(contract.chunk_size)
    ]
    chunk = _ns(
        session_id=101,
        seq_id=7,
        chunk_size=contract.chunk_size,
        action_dim=contract.action_dim,
        data=[value for row in rows for value in row],
    )
    messages = {
        topic: [] for topic in contract.required_topics
    }
    messages[contract.chunk_topic] = [(chunk, start)]
    for index, stamp in enumerate(state_stamps):
        messages[contract.state[0].topic].append((_state_message(contract), stamp))
        messages[contract.state[1].topic].append((_odom_message(), stamp))
    action_offsets = []
    offset = 0
    for spec in contract.action:
        action_offsets.append((spec, offset))
        offset += len(spec.names)
    for action_index, row in enumerate(rows):
        stamp = start + (action_index + 1) / 15
        for spec, offset in action_offsets:
            values = row[offset:offset + len(spec.names)]
            messages[spec.topic].append((_action_message(spec, values), stamp))
        ack_fields = {
            'session_id': 101,
            'seq_id': 7,
            'action_index': action_index,
            'executed_steps': action_index + 1,
            'chunk_size': contract.chunk_size,
            'executed_action': list(row),
            'timestamp': stamp,
        }
        messages[contract.ack_topic].append((_ns(status=0, **ack_fields), stamp))
        if action_index == contract.chunk_size - 1:
            messages[contract.ack_topic].append((_ns(status=1, **ack_fields), stamp))

    topic_types = dict(contract.required_topics)
    topic_counts = [
        {
            'topic_metadata': {
                'name': topic,
                'type': topic_type,
                'serialization_format': 'cdr',
                'offered_qos_profiles': [],
                'type_description_hash': '',
            },
            'message_count': len(messages[topic]),
        }
        for topic, topic_type in topic_types.items()
    ]
    metadata = {
        'rosbag2_bagfile_information': {
            'version': 9,
            'storage_identifier': 'mcap',
            'relative_file_paths': ['0_0.mcap'],
            'message_count': sum(len(stream) for stream in messages.values()),
            'topics_with_message_count': topic_counts,
        }
    }
    (root / 'metadata.yaml').write_text(yaml.safe_dump(metadata, sort_keys=False))

    reader = FakeBagReader(topic_types, messages)
    dependencies = {
        'bag_reader_factory': lambda _path: reader,
        'frame_timestamp_loader': lambda path, _camera: frame_data[path],
        'video_probe': lambda path: (256, 256, frame_counts[path]),
    }
    return root, episode, messages, dependencies, reader


def _codes(report):
    return {issue.code for issue in report.errors}


def test_contract_uses_showroom_sg2_order():
    contract = ValidationContract.from_robot_config()

    assert contract.cameras == (
        'cam_left_head', 'cam_left_wrist', 'cam_right_wrist',
    )
    assert contract.state_dim == 22
    assert len(contract.action_names) == 22
    assert contract.action_names[-3:] == ('linear_x', 'linear_y', 'angular_z')
    assert contract.chunk_size == 30
    assert contract.target_hz == 15


def test_complete_episode_passes(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, _episode, _messages, dependencies, reader = _write_episode(
        tmp_path, contract
    )

    report = validate_inference_episode(root, contract, **dependencies)

    assert report.valid, report.to_dict()
    assert report.metrics['chunks'] == 1
    assert report.metrics['executed_actions'] == 30
    assert report.metrics['terminal_reward'] == 1.0
    assert reader.closed


def test_missing_policy_chunk_and_ack_fail_closed(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, episode, messages, dependencies, _reader = _write_episode(tmp_path, contract)
    episode.pop('policy_provenance')
    episode.pop('rl_episode_contract')
    (root / 'episode_info.json').write_text(json.dumps(episode))
    messages[contract.chunk_topic].clear()
    messages[contract.ack_topic].clear()

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert {'provenance.missing', 'contract.missing', 'chunk.missing'} <= _codes(report)


def test_ack_action_mismatch_is_rejected(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, _episode, messages, dependencies, _reader = _write_episode(tmp_path, contract)
    executed_ack = messages[contract.ack_topic][0][0]
    executed_ack.executed_action[0] += 1.0

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert 'ack.action_mismatch' in _codes(report)


def test_partial_chunk_and_cancel_are_rejected(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, _episode, messages, dependencies, _reader = _write_episode(tmp_path, contract)
    messages[contract.ack_topic] = messages[contract.ack_topic][10:]
    cancel = _ns(
        session_id=101,
        seq_id=7,
        action_index=9,
        executed_steps=10,
        chunk_size=30,
        status=2,
        executed_action=[0.0] * 22,
        timestamp=1001.0,
    )
    messages[contract.ack_topic].append((cancel, 1001.0))

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert {'ack.partial_chunk', 'ack.cancelled'} <= _codes(report)


def test_seven_hz_sensor_stream_is_rejected(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, _episode, messages, dependencies, _reader = _write_episode(tmp_path, contract)
    for spec in contract.state:
        messages[spec.topic] = [
            (msg, 1000.0 + index / 7)
            for index, (msg, _stamp) in enumerate(messages[spec.topic])
        ]

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert 'timing.rate_mismatch' in _codes(report)


def test_video_dimension_mismatch_is_rejected(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, _episode, _messages, dependencies, _reader = _write_episode(
        tmp_path, contract
    )
    dependencies['video_probe'] = lambda _path: (640, 480, 32)

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert 'camera.dimensions' in _codes(report)


def test_changed_or_missing_checkpoint_is_rejected(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, episode, _messages, dependencies, _reader = _write_episode(
        tmp_path, contract
    )
    policy_path = Path(episode['policy_provenance']['policy_path'])
    (policy_path / 'model.safetensors').write_bytes(b'different-size')

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert 'provenance.artifact_size_mismatch' in _codes(report)


def test_malformed_camera_stats_fail_without_internal_error(tmp_path):
    contract = ValidationContract.from_robot_config()
    root, _episode, _messages, dependencies, _reader = _write_episode(
        tmp_path, contract
    )
    stats = next(root.glob('videos/0_0/*_recorder_stats.json'))
    payload = json.loads(stats.read_text())
    payload['frames_dropped_queue'] = 'not-an-int'
    stats.write_text(json.dumps(payload))

    report = validate_inference_episode(root, contract, **dependencies)

    assert not report.valid
    assert 'camera.stats_value' in _codes(report)


def test_cli_missing_episode_returns_invalid_json(tmp_path, capsys):
    code = main([str(tmp_path / 'missing'), '--json'])

    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload['valid'] is False
    assert payload['errors'][0]['code'] == 'path.not_directory'
