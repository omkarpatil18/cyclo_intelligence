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

"""Fail-closed validator for raw ACT inference/RL episodes.

The validator runs before an episode is admitted to an offline or online RL
replay buffer.  It deliberately rejects legacy recordings that do not contain
atomic ``ActionChunk`` commands, per-step execution acknowledgements, policy
provenance, or the declared ACT data contract.  Rejecting an episode never
modifies it.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import yaml


_CHUNK_TYPE = 'interfaces/msg/ActionChunk'
_ACK_TYPE = 'interfaces/msg/ActionStepAck'
_SHA256_RE = re.compile(r'^[0-9a-f]{64}$')
_MISSING = object()


@dataclass(frozen=True)
class TopicContract:
    """One ordered state/action slice from the shared robot config."""

    name: str
    topic: str
    msg_type: str
    names: tuple[str, ...]


@dataclass(frozen=True)
class ValidationContract:
    """Expected SG2 ACT recording layout."""

    robot_type: str
    cameras: tuple[str, ...]
    image_shape: tuple[int, int, int]
    state: tuple[TopicContract, ...]
    action: tuple[TopicContract, ...]
    chunk_topic: str = '/inference/action_chunk'
    ack_topic: str = '/inference/action_step_ack'
    chunk_size: int = 30
    action_dim: int = 22
    target_hz: float = 15.0
    rate_relative_tolerance: float = 0.20
    max_gap_factor: float = 3.0
    max_alignment_skew_sec: float = 0.10
    max_camera_skew_sec: float = 0.05
    action_atol: float = 1e-8

    @property
    def state_dim(self) -> int:
        return sum(len(spec.names) for spec in self.state)

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(name for spec in self.action for name in spec.names)

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(name for spec in self.state for name in spec.names)

    @property
    def required_topics(self) -> dict[str, str]:
        result = {spec.topic: spec.msg_type for spec in self.state}
        result.update({spec.topic: spec.msg_type for spec in self.action})
        result[self.chunk_topic] = _CHUNK_TYPE
        result[self.ack_topic] = _ACK_TYPE
        return result

    @classmethod
    def from_robot_config(
        cls,
        robot_type: str = 'ffw_sg2_rev1',
        explicit_path: Optional[Path | str] = None,
    ) -> 'ValidationContract':
        """Build the contract from shared YAML insertion order."""
        config_path = _find_robot_config(robot_type, explicit_path)
        with config_path.open('r', encoding='utf-8') as stream:
            raw = yaml.safe_load(stream) or {}
        try:
            section = raw['orchestrator']['ros__parameters'][robot_type]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f'orchestrator.ros__parameters.{robot_type} missing in '
                f'{config_path}'
            ) from exc

        observation = section.get('observation') or {}
        images = observation.get('images') or {}
        state_cfg = observation.get('state') or {}
        action_cfg = section.get('action') or {}

        def topic_specs(groups: Mapping[str, Any], default_type: str):
            specs: list[TopicContract] = []
            for group_name, config in groups.items():
                if not isinstance(config, Mapping) or not config.get('topic'):
                    continue
                specs.append(TopicContract(
                    name=str(group_name),
                    topic=str(config['topic']),
                    msg_type=str(config.get('msg_type') or default_type),
                    names=tuple(str(name) for name in config.get('joint_names') or []),
                ))
            return tuple(specs)

        cameras = tuple(str(name) for name in images)
        state = topic_specs(state_cfg, 'sensor_msgs/msg/JointState')
        action = topic_specs(action_cfg, 'trajectory_msgs/msg/JointTrajectory')
        contract = cls(
            robot_type=robot_type,
            cameras=cameras,
            image_shape=(3, 256, 256),
            state=state,
            action=action,
        )
        if len(contract.cameras) != 3:
            raise ValueError(
                f'{robot_type} ACT contract requires 3 cameras, got '
                f'{len(contract.cameras)}'
            )
        if contract.state_dim != 22 or len(contract.action_names) != 22:
            raise ValueError(
                f'{robot_type} ACT contract requires state/action 22D, got '
                f'{contract.state_dim}/{len(contract.action_names)}'
            )
        return contract


@dataclass(frozen=True)
class ValidationIssue:
    """Machine-readable validation finding."""

    severity: str
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Validation result; valid means that no error was observed."""

    episode_path: str
    contract_schema_version: int = 1
    issues: list[ValidationIssue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == 'error' for issue in self.issues)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == 'error']

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == 'warning']

    def error(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(ValidationIssue('error', code, message, context))

    def warning(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(ValidationIssue('warning', code, message, context))

    def to_dict(self) -> dict[str, Any]:
        return {
            'valid': self.valid,
            'episode_path': self.episode_path,
            'contract_schema_version': self.contract_schema_version,
            'errors': [asdict(issue) for issue in self.errors],
            'warnings': [asdict(issue) for issue in self.warnings],
            'metrics': self.metrics,
        }


def _find_robot_config(
    robot_type: str,
    explicit_path: Optional[Path | str],
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    for variable in ('ORCHESTRATOR_CONFIG_PATH', 'ROBOT_CLIENT_CONFIG_DIR'):
        value = os.environ.get(variable)
        if value:
            candidates.append(Path(value) / f'{robot_type}_config.yaml')
    candidates.append(Path('/orchestrator_config') / f'{robot_type}_config.yaml')
    repo_root = Path(__file__).resolve().parents[3]
    candidates.extend((
        repo_root / 'shared' / 'shared' / 'robot_configs'
        / f'{robot_type}_config.yaml',
        repo_root / 'shared' / 'robot_configs' / f'{robot_type}_config.yaml',
    ))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f'Robot config for {robot_type!r} not found: '
        f'{[str(path) for path in candidates]}'
    )


def _field(value: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        result = value.get(name, _MISSING)
    else:
        result = getattr(value, name, _MISSING)
    if result is _MISSING:
        if default is _MISSING:
            raise AttributeError(f'{type(value).__name__} has no field {name!r}')
        return default
    return result


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _finite_vector(value: Any) -> Optional[list[float]]:
    try:
        result = [float(item) for item in _sequence(value)]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result


def _close_vectors(left: Sequence[float], right: Sequence[float], atol: float) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)
        for a, b in zip(left, right)
    )


def _safe_relative(root: Path, raw_path: Any) -> Optional[Path]:
    try:
        value = str(raw_path or '')
        candidate = Path(value)
        if not value or candidate.is_absolute() or '..' in candidate.parts:
            return None
        resolved = (root / candidate).resolve()
        resolved.relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _strict_json(path: Path, report: ValidationReport, code: str) -> Optional[Any]:
    if not path.is_file():
        report.error(f'{code}.missing', f'Missing {path.name}', path=str(path))
        return None
    try:
        with path.open('r', encoding='utf-8') as stream:
            return json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        report.error(f'{code}.invalid', f'Cannot parse {path.name}: {exc}')
        return None


def _strict_yaml(path: Path, report: ValidationReport, code: str) -> Optional[Any]:
    if not path.is_file():
        report.error(f'{code}.missing', f'Missing {path.name}', path=str(path))
        return None
    try:
        with path.open('r', encoding='utf-8') as stream:
            return yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        report.error(f'{code}.invalid', f'Cannot parse {path.name}: {exc}')
        return None


def _default_bag_reader(path: Path):
    from cyclo_data.reader.bag_reader import BagReader
    return BagReader(path)


def _default_frame_loader(path: Path, camera: str):
    from cyclo_data.reader.frame_timestamps import load_frame_timestamps
    return load_frame_timestamps(path, camera)


def _default_video_probe(path: Path) -> tuple[int, int, int]:
    from cyclo_data.recorder.transcoder import _mp4_dimensions, _mp4_frame_count
    width, height = _mp4_dimensions(path)
    return width, height, _mp4_frame_count(path)


def _topic_count_map(metadata: Mapping[str, Any]) -> dict[str, tuple[str, int]]:
    try:
        entries = metadata['rosbag2_bagfile_information']['topics_with_message_count']
    except (KeyError, TypeError):
        return {}
    result: dict[str, tuple[str, int]] = {}
    for entry in entries or []:
        try:
            topic_meta = entry['topic_metadata']
            result[str(topic_meta['name'])] = (
                str(topic_meta['type']),
                int(entry['message_count']),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result


def _validate_episode_info(
    episode: Mapping[str, Any],
    contract: ValidationContract,
    report: ValidationReport,
) -> None:
    expected_scalars = {
        'format_version': 'robotis_v2',
        'task_type': 'inference',
        'policy_type': 'act',
        'robot_type': contract.robot_type,
        'transcoding_status': 'done',
        'video_remux_status': 'done',
    }
    for name, expected in expected_scalars.items():
        actual = episode.get(name)
        if actual != expected:
            report.error(
                f'episode.{name}',
                f'{name} must be {expected!r}, got {actual!r}',
            )
    collection_id = episode.get('collection_id')
    if not isinstance(collection_id, str) or not collection_id.strip():
        report.error('episode.collection_id', 'collection_id must be non-empty')
    failed_cameras = episode.get('transcoding_cameras_failed') or {}
    if failed_cameras:
        report.error(
            'camera.transcoding_failed',
            'One or more camera transcodes failed',
            cameras=failed_cameras,
        )

    outcome = episode.get('outcome')
    if not isinstance(outcome, Mapping):
        report.error('outcome.missing', 'Terminal success/failure label is missing')
    else:
        status = outcome.get('status')
        success = outcome.get('success')
        valid = (
            outcome.get('schema_version') == 1
            and status in {'success', 'failure'}
            and isinstance(success, bool)
            and success == (status == 'success')
            and isinstance(outcome.get('annotated_at'), str)
            and bool(outcome.get('annotated_at'))
        )
        if not valid:
            report.error('outcome.invalid', 'Outcome must be a terminal v1 label')
        else:
            report.metrics['outcome'] = status
            report.metrics['terminal_reward'] = 1.0 if success else 0.0


def _validate_provenance(
    episode: Mapping[str, Any],
    report: ValidationReport,
) -> None:
    provenance = episode.get('policy_provenance')
    if not isinstance(provenance, Mapping):
        report.error('provenance.missing', 'policy_provenance is missing')
        return
    required_text = (
        'service_type', 'policy_type', 'policy_path', 'checkpoint_id',
        'config_path', 'config_sha256', 'artifact_manifest_sha256',
    )
    if provenance.get('schema_version') != 1:
        report.error('provenance.schema_version', 'Unsupported provenance schema')
    for key in required_text:
        if not isinstance(provenance.get(key), str) or not provenance.get(key):
            report.error('provenance.field', f'{key} must be a non-empty string')
    if provenance.get('policy_type') != 'act':
        report.error('provenance.policy_type', 'Provenance policy_type must be act')
    for key in ('config_sha256', 'artifact_manifest_sha256'):
        digest = provenance.get(key)
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            report.error('provenance.digest', f'{key} is not a SHA-256 digest')

    artifacts = provenance.get('artifacts')
    canonical: list[dict[str, Any]] = []
    if not isinstance(artifacts, list) or not artifacts:
        report.error('provenance.artifacts', 'Artifact manifest must be non-empty')
        return
    seen: set[str] = set()
    for entry in artifacts:
        if not isinstance(entry, Mapping):
            report.error('provenance.artifact', 'Artifact entry must be an object')
            continue
        name = entry.get('name')
        size = entry.get('size_bytes')
        path = Path(str(name or ''))
        if (
            not isinstance(name, str) or not name or path.is_absolute()
            or '..' in path.parts or name in seen
        ):
            report.error('provenance.artifact_name', 'Unsafe/duplicate artifact name')
            continue
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            report.error('provenance.artifact_size', f'Invalid size for {name!r}')
            continue
        seen.add(name)
        canonical.append({'name': name, 'size_bytes': size})
    if canonical != sorted(canonical, key=lambda item: item['name']):
        report.error('provenance.artifact_order', 'Artifacts must be path-sorted')
    manifest = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    digest = hashlib.sha256(manifest).hexdigest()
    if provenance.get('artifact_manifest_sha256') != digest:
        report.error('provenance.manifest_hash', 'Artifact manifest SHA-256 mismatch')

    config_name = provenance.get('config_path')
    config_entry = next(
        (entry for entry in canonical if entry['name'] == config_name),
        None,
    )
    if config_entry is None:
        report.error('provenance.config_artifact', 'config_path is absent from artifacts')
    try:
        policy_path = Path(str(provenance.get('policy_path') or ''))
        policy_exists = policy_path.is_dir() and not policy_path.is_symlink()
    except (OSError, ValueError):
        policy_path = Path('.')
        policy_exists = False
    if not policy_exists:
        report.error(
            'provenance.policy_path_missing',
            'Recorded policy directory is missing or is a symlink',
        )
        return
    for entry in canonical:
        artifact = _safe_relative(policy_path, entry['name'])
        if artifact is None or not artifact.is_file() or artifact.is_symlink():
            report.error(
                'provenance.artifact_missing',
                f"Recorded artifact is missing: {entry['name']}",
            )
            continue
        try:
            size = int(artifact.stat().st_size)
        except OSError as exc:
            report.error(
                'provenance.artifact_read',
                f"Cannot stat artifact {entry['name']}: {exc}",
            )
            continue
        if size != entry['size_bytes']:
            report.error(
                'provenance.artifact_size_mismatch',
                f"Artifact size changed: {entry['name']}",
            )
    if isinstance(config_name, str):
        config_file = _safe_relative(policy_path, config_name)
        if config_file is None or not config_file.is_file():
            report.error('provenance.config_missing', 'Recorded policy config is missing')
        else:
            try:
                actual = hashlib.sha256(config_file.read_bytes()).hexdigest()
            except OSError as exc:
                report.error('provenance.config_read', f'Cannot read policy config: {exc}')
            else:
                if actual != provenance.get('config_sha256'):
                    report.error('provenance.config_hash', 'Policy config SHA-256 mismatch')


def _expected_rl_contract(
    collection_id: str,
    contract: ValidationContract,
) -> dict[str, Any]:
    return {
        'schema_version': 1,
        'collection_id': collection_id,
        'session_id': collection_id,
        'cameras': {
            'names': list(contract.cameras),
            'image_shape': list(contract.image_shape),
        },
        'state': {
            'dim': contract.state_dim,
            'names': list(contract.state_names),
            'topics': [spec.topic for spec in contract.state],
        },
        'action': {
            'dim': contract.action_dim,
            'names': list(contract.action_names),
            'topics': [spec.topic for spec in contract.action],
            'hz': int(contract.target_hz),
        },
        'chunk': {
            'size': contract.chunk_size,
            'action_dim': contract.action_dim,
            'topic': contract.chunk_topic,
            'ack_topic': contract.ack_topic,
        },
        'terminal_reward': {
            'semantics': 'binary_terminal',
            'success': 1.0,
            'failure': 0.0,
            'intermediate': 0.0,
        },
    }


def _validate_rl_contract(
    episode: Mapping[str, Any],
    contract: ValidationContract,
    report: ValidationReport,
) -> None:
    recorded = episode.get('rl_episode_contract')
    if not isinstance(recorded, Mapping):
        report.error('contract.missing', 'rl_episode_contract is missing')
        return
    expected = _expected_rl_contract(str(episode.get('collection_id') or ''), contract)
    if dict(recorded) != expected:
        report.error(
            'contract.mismatch',
            'Recorded RL episode contract differs from the SG2 ACT contract',
            expected=expected,
            actual=dict(recorded),
        )


def _validate_timing(
    label: str,
    stamps: Sequence[float],
    contract: ValidationContract,
    report: ValidationReport,
    *,
    code_prefix: str = 'timing',
) -> None:
    metric_key = label.strip('/').replace('/', '.') or 'root'
    report.metrics.setdefault('rates_hz', {})[metric_key] = None
    if len(stamps) < 3:
        report.error(
            f'{code_prefix}.samples',
            f'{label} needs at least 3 samples',
            samples=len(stamps),
        )
        return
    deltas = [float(right) - float(left) for left, right in zip(stamps, stamps[1:])]
    if any(not math.isfinite(delta) or delta <= 0.0 for delta in deltas):
        report.error(
            f'{code_prefix}.non_monotonic',
            f'{label} timestamps are not strictly increasing',
        )
        return
    duration = float(stamps[-1]) - float(stamps[0])
    rate = (len(stamps) - 1) / duration
    report.metrics['rates_hz'][metric_key] = round(rate, 6)
    tolerance = contract.target_hz * contract.rate_relative_tolerance
    if abs(rate - contract.target_hz) > tolerance:
        report.error(
            f'{code_prefix}.rate_mismatch',
            f'{label} rate is {rate:.3f} Hz, expected {contract.target_hz:.3f} Hz',
            observed_hz=rate,
            expected_hz=contract.target_hz,
            tolerance_hz=tolerance,
        )
    max_gap = max(deltas)
    allowed_gap = contract.max_gap_factor / contract.target_hz
    if max_gap > allowed_gap:
        report.error(
            f'{code_prefix}.gap',
            f'{label} has a {max_gap:.4f}s gap (limit {allowed_gap:.4f}s)',
            max_gap_sec=max_gap,
        )


def _frame_values(frame_data: Any) -> tuple[list[int], list[int], list[int]]:
    indices = [int(value) for value in _sequence(_field(frame_data, 'frame_index'))]
    headers = [int(value) for value in _sequence(_field(frame_data, 'header_stamp_ns'))]
    receives = [int(value) for value in _sequence(_field(frame_data, 'recv_ns'))]
    return indices, headers, receives


def _camera_stamp_seconds(headers: list[int], receives: list[int]) -> list[float]:
    source = headers if headers and any(value > 0 for value in headers) else receives
    return [value / 1e9 for value in source]


def _nearest_distance(sorted_stamps: Sequence[float], value: float) -> float:
    if not sorted_stamps:
        return math.inf
    index = bisect_left(sorted_stamps, value)
    distances: list[float] = []
    if index < len(sorted_stamps):
        distances.append(abs(sorted_stamps[index] - value))
    if index > 0:
        distances.append(abs(sorted_stamps[index - 1] - value))
    return min(distances)


def _has_next(sorted_stamps: Sequence[float], value: float, limit: float) -> bool:
    index = bisect_left(sorted_stamps, value)
    return index < len(sorted_stamps) and sorted_stamps[index] - value <= limit


def _validate_cameras(
    root: Path,
    episode: Mapping[str, Any],
    contract: ValidationContract,
    report: ValidationReport,
    frame_loader: Callable[[Path, str], Any],
    video_probe: Optional[Callable[[Path], tuple[int, int, int]]],
) -> dict[str, list[float]]:
    segments = episode.get('video_segments')
    if not isinstance(segments, list) or not segments:
        report.error('camera.segments', 'video_segments must be non-empty')
        return {}
    camera_stamps: dict[str, list[float]] = {name: [] for name in contract.cameras}
    for segment_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping):
            report.error('camera.segment', 'Video segment must be an object')
            continue
        cameras = segment.get('cameras')
        if list(cameras or []) != list(contract.cameras):
            report.error(
                'camera.set_mismatch',
                'Video segment camera order/set differs from contract',
                segment=segment_index,
                expected=list(contract.cameras),
                actual=list(cameras or []),
            )
        video_dir = _safe_relative(root, segment.get('video_dir'))
        if video_dir is None or not video_dir.is_dir():
            report.error(
                'camera.video_dir',
                'Unsafe or missing video_dir',
                segment=segment_index,
            )
            continue
        for camera in contract.cameras:
            mp4 = video_dir / f'{camera}.mp4'
            sidecar = video_dir / f'{camera}_timestamps.parquet'
            stats_path = video_dir / f'{camera}_recorder_stats.json'
            for kind, path in (('mp4', mp4), ('sidecar', sidecar), ('stats', stats_path)):
                if not path.is_file() or path.stat().st_size <= 0:
                    report.error(
                        f'camera.{kind}_missing',
                        f'Missing/empty {kind} for {camera}',
                        path=str(path),
                    )
            if not sidecar.is_file():
                continue
            try:
                frame_data = frame_loader(sidecar, camera)
                indices, headers, receives = _frame_values(frame_data)
            except Exception as exc:  # noqa: BLE001 - data boundary
                report.error(
                    'camera.sidecar_invalid',
                    f'Cannot read timestamps for {camera}: {exc}',
                )
                continue
            count = len(indices)
            if count < 3 or len(headers) != count or len(receives) != count:
                report.error(
                    'camera.sidecar_shape',
                    f'Invalid timestamp columns for {camera}',
                    rows=count,
                )
                continue
            if indices != list(range(count)):
                report.error(
                    'camera.frame_index',
                    f'{camera} frame_index must be contiguous from zero',
                )
            stamps = _camera_stamp_seconds(headers, receives)
            _validate_timing(
                f'camera:{camera}', stamps, contract, report,
                code_prefix='camera',
            )
            if camera_stamps[camera] and stamps[0] <= camera_stamps[camera][-1]:
                report.error(
                    'camera.segment_time',
                    f'{camera} segment timestamps overlap or regress',
                )
            camera_stamps[camera].extend(stamps)

            stats = _strict_json(stats_path, report, 'camera.stats')
            if isinstance(stats, Mapping):
                expected_counts = (
                    'frames_received', 'frames_written',
                    'frames_metadata_written', 'frames_remuxed',
                )
                for key in expected_counts:
                    try:
                        actual_count = int(stats.get(key))
                    except (TypeError, ValueError):
                        actual_count = -1
                    if actual_count != count:
                        report.error(
                            'camera.frame_count',
                            f'{camera} {key}={actual_count}, sidecar={count}',
                        )
                for key, value in stats.items():
                    if key.startswith('frames_dropped_'):
                        try:
                            dropped = int(value or 0)
                        except (TypeError, ValueError, OverflowError):
                            report.error(
                                'camera.stats_value',
                                f'{camera} {key} is not an integer',
                            )
                        else:
                            if dropped != 0:
                                report.error(
                                    'camera.dropped_frames',
                                    f'{camera} {key}={value}',
                                )
                    if key.endswith('_error') and value not in (None, ''):
                        report.error(
                            'camera.recorder_error',
                            f'{camera} {key}={value!r}',
                        )
                if stats.get('remux_status') != 'done':
                    report.error(
                        'camera.remux_status',
                        f'{camera} remux_status must be done',
                    )
            if video_probe is not None and mp4.is_file():
                try:
                    width, height, video_frames = video_probe(mp4)
                except Exception as exc:  # noqa: BLE001 - external media boundary
                    report.error('camera.probe', f'Cannot probe {camera}: {exc}')
                else:
                    expected_height, expected_width = contract.image_shape[1:]
                    if (width, height) != (expected_width, expected_height):
                        report.error(
                            'camera.dimensions',
                            f'{camera} is {width}x{height}, expected '
                            f'{expected_width}x{expected_height}',
                        )
                    if int(video_frames) != count:
                        report.error(
                            'camera.video_frame_count',
                            f'{camera} MP4={video_frames}, sidecar={count}',
                        )

    reference = camera_stamps.get(contract.cameras[0], []) if contract.cameras else []
    for camera in contract.cameras[1:]:
        stamps = camera_stamps.get(camera, [])
        if reference and stamps:
            max_skew = max(_nearest_distance(stamps, stamp) for stamp in reference)
            report.metrics.setdefault('camera_max_skew_sec', {})[camera] = round(
                max_skew, 6
            )
            if max_skew > contract.max_camera_skew_sec:
                report.error(
                    'camera.alignment',
                    f'{camera} maximum inter-camera skew is {max_skew:.4f}s',
                )
    report.metrics['camera_frames'] = {
        camera: len(stamps) for camera, stamps in camera_stamps.items()
    }
    return camera_stamps


def _extract_joint_state(msg: Any, expected: Sequence[str]) -> Optional[list[float]]:
    names = [str(value) for value in _sequence(_field(msg, 'name', []))]
    positions = _finite_vector(_field(msg, 'position', []))
    if positions is None or names != list(expected) or len(positions) != len(expected):
        return None
    return positions


def _extract_odom(msg: Any) -> Optional[list[float]]:
    try:
        twist = _field(_field(msg, 'twist'), 'twist')
        linear = _field(twist, 'linear')
        angular = _field(twist, 'angular')
        values = [
            _field(linear, 'x'), _field(linear, 'y'), _field(angular, 'z'),
        ]
    except AttributeError:
        return None
    return _finite_vector(values)


def _extract_action(msg: Any, spec: TopicContract) -> Optional[list[float]]:
    if spec.msg_type == 'geometry_msgs/msg/Twist':
        try:
            linear = _field(msg, 'linear')
            angular = _field(msg, 'angular')
            values = [
                _field(linear, 'x'), _field(linear, 'y'), _field(angular, 'z'),
            ]
        except AttributeError:
            return None
        return _finite_vector(values)
    names = [str(value) for value in _sequence(_field(msg, 'joint_names', []))]
    points = _sequence(_field(msg, 'points', []))
    if names != list(spec.names) or len(points) != 1:
        return None
    return _finite_vector(_field(points[0], 'positions', []))


def _nearest_message(
    messages: Sequence[tuple[Any, float]],
    stamp: float,
) -> tuple[Optional[Any], float]:
    if not messages:
        return None, math.inf
    stamps = [item[1] for item in messages]
    index = bisect_left(stamps, stamp)
    candidates: list[tuple[Any, float]] = []
    if index < len(messages):
        candidates.append(messages[index])
    if index > 0:
        candidates.append(messages[index - 1])
    msg, msg_stamp = min(candidates, key=lambda item: abs(item[1] - stamp))
    return msg, abs(msg_stamp - stamp)


def _validate_topic_streams(
    messages: Mapping[str, list[tuple[Any, float]]],
    contract: ValidationContract,
    report: ValidationReport,
) -> None:
    for spec in (*contract.state, *contract.action):
        stream = messages.get(spec.topic, [])
        stamps = [stamp for _msg, stamp in stream]
        _validate_timing(spec.topic, stamps, contract, report)
        for index, (msg, _stamp) in enumerate(stream):
            if spec in contract.state:
                if spec.msg_type == 'sensor_msgs/msg/JointState':
                    vector = _extract_joint_state(msg, spec.names)
                elif spec.msg_type == 'nav_msgs/msg/Odometry':
                    vector = _extract_odom(msg)
                else:
                    vector = []
            else:
                vector = _extract_action(msg, spec)
            if vector is None or len(vector) != len(spec.names):
                report.error(
                    'topic.vector_contract',
                    f'{spec.topic} sample {index} violates ordered '
                    f'{len(spec.names)}D contract',
                )
                break


def _validate_chunks(
    messages: Mapping[str, list[tuple[Any, float]]],
    contract: ValidationContract,
    report: ValidationReport,
) -> list[tuple[float, list[float]]]:
    chunk_stream = messages.get(contract.chunk_topic, [])
    ack_stream = messages.get(contract.ack_topic, [])
    chunk_log_stamps = [stamp for _msg, stamp in chunk_stream]
    if any(
        right <= left
        for left, right in zip(chunk_log_stamps, chunk_log_stamps[1:])
    ):
        report.error(
            'chunk.timestamp_order',
            'ActionChunk log timestamps are not strictly increasing',
        )
    ack_log_stamps = [stamp for _msg, stamp in ack_stream]
    if any(
        right < left
        for left, right in zip(ack_log_stamps, ack_log_stamps[1:])
    ):
        report.error(
            'ack.timestamp_order',
            'ActionStepAck log timestamps regress',
        )
    chunks: dict[tuple[int, int], tuple[list[list[float]], float]] = {}
    ordered_keys: list[tuple[int, int]] = []
    for msg, stamp in chunk_stream:
        try:
            key = (int(_field(msg, 'session_id')), int(_field(msg, 'seq_id')))
            chunk_size = int(_field(msg, 'chunk_size'))
            action_dim = int(_field(msg, 'action_dim'))
            data = _finite_vector(_field(msg, 'data'))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            report.error('chunk.decode', f'Cannot decode ActionChunk: {exc}')
            continue
        if chunk_size == 0 and action_dim == 0:
            report.error('chunk.cancelled', f'Cancellation chunk recorded for {key}')
            continue
        if (
            chunk_size != contract.chunk_size
            or action_dim != contract.action_dim
            or data is None
            or len(data) != contract.chunk_size * contract.action_dim
        ):
            report.error(
                'chunk.shape',
                f'Chunk {key} must be {contract.chunk_size}x{contract.action_dim}',
                chunk_size=chunk_size,
                action_dim=action_dim,
                data_size=None if data is None else len(data),
            )
            continue
        if key in chunks:
            report.error('chunk.duplicate', f'Duplicate chunk {key}')
            continue
        rows = [
            data[index * action_dim:(index + 1) * action_dim]
            for index in range(chunk_size)
        ]
        chunks[key] = (rows, stamp)
        ordered_keys.append(key)

    if not chunks:
        report.error('chunk.missing', 'No complete ActionChunk was recorded')
    sessions = {key[0] for key in ordered_keys}
    if len(sessions) > 1:
        report.error('chunk.session_switch', 'Multiple ActionChunk sessions in episode')
    for previous, current in zip(ordered_keys, ordered_keys[1:]):
        if current[0] == previous[0] and current[1] != previous[1] + 1:
            report.error(
                'chunk.seq_gap',
                f'Chunk sequence gap: {previous} -> {current}',
            )

    ack_by_key: dict[tuple[int, int], list[tuple[Any, float]]] = {}
    for msg, stamp in ack_stream:
        try:
            key = (int(_field(msg, 'session_id')), int(_field(msg, 'seq_id')))
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            report.error('ack.decode', f'Cannot decode ActionStepAck: {exc}')
            continue
        ack_by_key.setdefault(key, []).append((msg, stamp))
    for key in ack_by_key:
        if key not in chunks:
            report.error('ack.orphan', f'ACK has no recorded source chunk: {key}')

    executed_timeline: list[tuple[float, list[float]]] = []
    completed_stamp: dict[tuple[int, int], float] = {}
    for key in ordered_keys:
        if key not in chunks:
            continue
        rows, chunk_stamp = chunks[key]
        executed: dict[int, tuple[Any, float]] = {}
        completed: list[tuple[Any, float]] = []
        for ack, ack_stamp in ack_by_key.get(key, []):
            try:
                status = int(_field(ack, 'status'))
                action_index = int(_field(ack, 'action_index'))
                executed_steps = int(_field(ack, 'executed_steps'))
                chunk_size = int(_field(ack, 'chunk_size'))
                action = _finite_vector(_field(ack, 'executed_action'))
                source_timestamp = float(_field(ack, 'timestamp'))
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                report.error('ack.decode', f'Cannot decode ACK for {key}: {exc}')
                continue
            if not math.isfinite(source_timestamp):
                report.error('ack.timestamp', f'ACK timestamp is non-finite for {key}')
            if status == 2:
                report.error('ack.cancelled', f'Chunk {key} was cancelled')
                continue
            if status not in (0, 1):
                report.error('ack.status', f'Unknown ACK status {status} for {key}')
                continue
            if (
                action_index < 0 or action_index >= contract.chunk_size
                or chunk_size != contract.chunk_size
                or executed_steps != action_index + 1
                or action is None or len(action) != contract.action_dim
            ):
                report.error('ack.contract', f'Malformed ACK for {key}/{action_index}')
                continue
            if not _close_vectors(action, rows[action_index], contract.action_atol):
                report.error(
                    'ack.action_mismatch',
                    f'Executed action differs from chunk row {key}/{action_index}',
                )
            if ack_stamp < chunk_stamp:
                report.error('ack.causality', f'ACK precedes chunk publication: {key}')
            if status == 0:
                if action_index in executed:
                    report.error('ack.duplicate', f'Duplicate EXECUTED {key}/{action_index}')
                else:
                    executed[action_index] = (ack, ack_stamp)
                    executed_timeline.append((ack_stamp, rows[action_index]))
            else:
                completed.append((ack, ack_stamp))
        missing = sorted(set(range(contract.chunk_size)) - set(executed))
        if missing:
            report.error(
                'ack.partial_chunk',
                f'Chunk {key} is missing EXECUTED acknowledgements',
                missing_indices=missing,
            )
        if len(completed) != 1:
            report.error(
                'ack.completed_count',
                f'Chunk {key} needs exactly one COMPLETED ACK, got {len(completed)}',
            )
        else:
            completed_ack, done_stamp = completed[0]
            try:
                done_index = int(_field(completed_ack, 'action_index'))
                done_steps = int(_field(completed_ack, 'executed_steps'))
            except (AttributeError, TypeError, ValueError):
                done_index, done_steps = -1, -1
            if done_index != contract.chunk_size - 1 or done_steps != contract.chunk_size:
                report.error('ack.completed_contract', f'Malformed COMPLETED ACK for {key}')
            completed_stamp[key] = done_stamp

    for previous, current in zip(ordered_keys, ordered_keys[1:]):
        if previous in completed_stamp and current in chunks:
            if chunks[current][1] < completed_stamp[previous]:
                report.error(
                    'chunk.overlap',
                    f'Chunk {current} published before {previous} completed',
                )
    executed_timeline.sort(key=lambda item: item[0])
    _validate_timing(
        'executed_ack',
        [stamp for stamp, _row in executed_timeline],
        contract,
        report,
        code_prefix='ack',
    )
    report.metrics['chunks'] = len(chunks)
    report.metrics['executed_actions'] = len(executed_timeline)
    return executed_timeline


def _validate_alignment(
    messages: Mapping[str, list[tuple[Any, float]]],
    camera_stamps: Mapping[str, list[float]],
    executed: Sequence[tuple[float, list[float]]],
    contract: ValidationContract,
    report: ValidationReport,
) -> None:
    if not executed:
        return
    state_stamps = {
        spec.topic: [stamp for _msg, stamp in messages.get(spec.topic, [])]
        for spec in contract.state
    }
    for ack_stamp, full_action in executed:
        offset = 0
        for spec in contract.action:
            expected = full_action[offset:offset + len(spec.names)]
            offset += len(spec.names)
            msg, skew = _nearest_message(messages.get(spec.topic, []), ack_stamp)
            if msg is None or skew > contract.max_alignment_skew_sec:
                report.error(
                    'alignment.action_missing',
                    f'No aligned command on {spec.topic} for ACK at {ack_stamp:.6f}',
                )
                continue
            actual = _extract_action(msg, spec)
            if actual is None or not _close_vectors(actual, expected, contract.action_atol):
                report.error(
                    'alignment.action_mismatch',
                    f'{spec.topic} command differs from acknowledged chunk action',
                )
        for topic, stamps in state_stamps.items():
            if _nearest_distance(stamps, ack_stamp) > contract.max_alignment_skew_sec:
                report.error(
                    'alignment.state_skew',
                    f'No aligned state sample on {topic}',
                )
            if not _has_next(stamps, ack_stamp, contract.max_gap_factor / contract.target_hz):
                report.error(
                    'alignment.next_state_missing',
                    f'No causal next state sample on {topic}',
                )
        for camera, stamps in camera_stamps.items():
            if _nearest_distance(stamps, ack_stamp) > contract.max_alignment_skew_sec:
                report.error(
                    'alignment.camera_skew',
                    f'No aligned frame for {camera}',
                )
            if not _has_next(stamps, ack_stamp, contract.max_gap_factor / contract.target_hz):
                report.error(
                    'alignment.next_frame_missing',
                    f'No causal next frame for {camera}',
                )


def validate_inference_episode(
    episode_path: Path | str,
    contract: Optional[ValidationContract] = None,
    *,
    bag_reader_factory: Optional[Callable[[Path], Any]] = None,
    frame_timestamp_loader: Optional[Callable[[Path, str], Any]] = None,
    video_probe: Optional[Callable[[Path], tuple[int, int, int]]] = None,
    probe_video: bool = True,
) -> ValidationReport:
    """Validate one archived inference episode without modifying it.

    Dependencies are injectable so unit tests can exercise the complete gate
    without ROS, MCAP, PyArrow, or ffprobe.  Production defaults are imported
    lazily from ``cyclo_data``.
    """
    root = Path(episode_path).resolve()
    report = ValidationReport(str(root))
    if not root.is_dir():
        report.error('path.not_directory', 'Episode path is not a directory')
        return report
    if contract is None:
        try:
            contract = ValidationContract.from_robot_config()
        except Exception as exc:  # noqa: BLE001 - configuration boundary
            report.error('contract.config', f'Cannot load robot contract: {exc}')
            return report

    episode = _strict_json(root / 'episode_info.json', report, 'episode_info')
    metadata = _strict_yaml(root / 'metadata.yaml', report, 'metadata')
    if not isinstance(episode, Mapping):
        return report
    _validate_episode_info(episode, contract, report)
    _validate_provenance(episode, report)
    _validate_rl_contract(episode, contract, report)

    frame_loader = frame_timestamp_loader or _default_frame_loader
    probe = (video_probe or _default_video_probe) if probe_video else None
    camera_stamps = _validate_cameras(
        root, episode, contract, report, frame_loader, probe
    )
    if not isinstance(metadata, Mapping):
        return report
    bag_info = metadata.get('rosbag2_bagfile_information')
    if not isinstance(bag_info, Mapping):
        report.error('metadata.bag_info', 'rosbag2_bagfile_information is missing')
        return report
    if bag_info.get('storage_identifier') != 'mcap':
        report.error('metadata.storage', 'storage_identifier must be mcap')
    relative_files = bag_info.get('relative_file_paths')
    if not isinstance(relative_files, list) or not relative_files:
        report.error('metadata.mcap_files', 'No MCAP files declared in metadata')
    else:
        for relative in relative_files:
            mcap = _safe_relative(root, relative)
            if mcap is None or mcap.suffix != '.mcap' or not mcap.is_file():
                report.error('metadata.mcap_missing', f'Unsafe/missing MCAP: {relative!r}')
            elif mcap.stat().st_size <= 0:
                report.error('metadata.mcap_empty', f'Empty MCAP: {relative!r}')

    declared_topics = _topic_count_map(metadata)
    for topic, expected_type in contract.required_topics.items():
        declared = declared_topics.get(topic)
        if declared is None:
            report.error('metadata.topic_missing', f'Metadata omits {topic}')
        elif declared[0] != expected_type:
            report.error(
                'metadata.topic_type',
                f'{topic} type is {declared[0]!r}, expected {expected_type!r}',
            )
        elif declared[1] <= 0:
            report.error('metadata.topic_empty', f'{topic} has no messages')

    factory = bag_reader_factory or _default_bag_reader
    try:
        reader = factory(root)
        opened = bool(reader.open())
    except Exception as exc:  # noqa: BLE001 - MCAP boundary
        report.error('bag.open', f'Cannot open MCAP: {exc}')
        return report
    if not opened:
        report.error('bag.open', 'BagReader could not open MCAP')
        return report
    try:
        topic_types = reader.get_topic_types()
        for topic, expected_type in contract.required_topics.items():
            actual = topic_types.get(topic)
            if actual is None:
                report.error('bag.topic_missing', f'MCAP omits {topic}')
            elif actual != expected_type:
                report.error(
                    'bag.topic_type',
                    f'{topic} type is {actual!r}, expected {expected_type!r}',
                )
        messages: dict[str, list[tuple[Any, float]]] = {
            topic: [] for topic in contract.required_topics
        }
        for topic, msg, timestamp in reader.read_messages(
            list(contract.required_topics)
        ):
            if topic not in messages:
                continue
            try:
                stamp = float(timestamp)
            except (TypeError, ValueError, OverflowError):
                report.error('bag.timestamp', f'Non-numeric timestamp on {topic}')
                continue
            if not math.isfinite(stamp):
                report.error('bag.timestamp', f'Non-finite timestamp on {topic}')
                continue
            messages[topic].append((msg, stamp))
    except Exception as exc:  # noqa: BLE001 - decoder/data boundary
        report.error('bag.decode', f'Cannot decode MCAP messages: {exc}')
        return report
    finally:
        close = getattr(reader, 'close', None)
        if callable(close):
            close()

    _validate_topic_streams(messages, contract, report)
    executed = _validate_chunks(messages, contract, report)
    _validate_alignment(messages, camera_stamps, executed, contract, report)
    report.metrics['topic_message_counts'] = {
        topic: len(stream) for topic, stream in messages.items()
    }
    report.metrics['error_count'] = len(report.errors)
    report.metrics['warning_count'] = len(report.warnings)
    return report
