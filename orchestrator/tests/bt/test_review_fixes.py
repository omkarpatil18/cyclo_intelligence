#!/usr/bin/env python3
#
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

import sys
import types

import pytest


class _QoSProfile:
    def __init__(self, *args, **kwargs):
        pass


class _ReliabilityPolicy:
    RELIABLE = object()


def _install_ros_stubs():
    rclpy_mod = types.ModuleType('rclpy')
    qos_mod = types.ModuleType('rclpy.qos')
    qos_mod.QoSProfile = _QoSProfile
    qos_mod.ReliabilityPolicy = _ReliabilityPolicy

    sensor_msgs_mod = types.ModuleType('sensor_msgs')
    sensor_msgs_msg_mod = types.ModuleType('sensor_msgs.msg')
    sensor_msgs_msg_mod.JointState = object

    trajectory_msgs_mod = types.ModuleType('trajectory_msgs')
    trajectory_msgs_msg_mod = types.ModuleType('trajectory_msgs.msg')
    trajectory_msgs_msg_mod.JointTrajectory = object
    trajectory_msgs_msg_mod.JointTrajectoryPoint = object

    geometry_msgs_mod = types.ModuleType('geometry_msgs')
    geometry_msgs_msg_mod = types.ModuleType('geometry_msgs.msg')
    geometry_msgs_msg_mod.Twist = object

    nav_msgs_mod = types.ModuleType('nav_msgs')
    nav_msgs_msg_mod = types.ModuleType('nav_msgs.msg')
    nav_msgs_msg_mod.Odometry = object

    interfaces_mod = types.ModuleType('interfaces')
    interfaces_msg_mod = types.ModuleType('interfaces.msg')
    interfaces_srv_mod = types.ModuleType('interfaces.srv')

    class _InferenceStatus:
        READY = 0
        INFERENCING = 2
        PAUSED = 3

    class _TaskInfo:
        inference_mode = ''
        action_request_mode = ''
        acceleration_mode = ''
        acceleration_engine_path = ''

    class _SendCommandRequest:
        START_INFERENCE = 1
        STOP_INFERENCE = 2
        RESUME_INFERENCE = 3
        FINISH = 4

    class _SendCommand:
        Request = _SendCommandRequest

    interfaces_msg_mod.InferenceStatus = _InferenceStatus
    interfaces_msg_mod.TaskInfo = _TaskInfo
    interfaces_srv_mod.SendCommand = _SendCommand

    sys.modules.setdefault('rclpy', rclpy_mod)
    sys.modules.setdefault('rclpy.qos', qos_mod)
    sys.modules.setdefault('sensor_msgs', sensor_msgs_mod)
    sys.modules.setdefault('sensor_msgs.msg', sensor_msgs_msg_mod)
    sys.modules.setdefault('trajectory_msgs', trajectory_msgs_mod)
    sys.modules.setdefault('trajectory_msgs.msg', trajectory_msgs_msg_mod)
    sys.modules.setdefault('geometry_msgs', geometry_msgs_mod)
    sys.modules.setdefault('geometry_msgs.msg', geometry_msgs_msg_mod)
    sys.modules.setdefault('nav_msgs', nav_msgs_mod)
    sys.modules.setdefault('nav_msgs.msg', nav_msgs_msg_mod)
    sys.modules.setdefault('interfaces', interfaces_mod)
    sys.modules.setdefault('interfaces.msg', interfaces_msg_mod)
    sys.modules.setdefault('interfaces.srv', interfaces_srv_mod)


_install_ros_stubs()

from orchestrator.bt.actions.joint_control import _coerce_positions  # noqa: E402
from orchestrator.bt.actions.send_command import SendCommand  # noqa: E402
from orchestrator.bt.actions.arm_state_gate import ArmStateGate  # noqa: E402
from orchestrator.bt.bt_core import NodeStatus  # noqa: E402
from orchestrator.bt.node_registry import _annotation_to_port_type  # noqa: E402
from orchestrator.bt.node_registry import build_registry  # noqa: E402
from orchestrator.bt.node_registry import catalog_payload  # noqa: E402


class _DummyNode:
    def create_client(self, *args, **kwargs):
        return object()

    def create_subscription(self, *args, **kwargs):
        return object()

    def get_logger(self):
        class _Logger:
            def info(self, *args, **kwargs):
                pass

            def warn(self, *args, **kwargs):
                pass

            def error(self, *args, **kwargs):
                pass

        return _Logger()


def _make_wait_action(**overrides):
    kwargs = {
        'node': _DummyNode(),
        'left_target_joints': 'arm_l_joint1, arm_l_joint2',
        'left_target_positions': '0.1, -0.2',
        'right_target_joints': 'arm_r_joint1, arm_r_joint2',
        'right_target_positions': '0.3, -0.4',
    }
    kwargs.update(overrides)
    return ArmStateGate(**kwargs)


def _joint_state(
    left=(0.1, -0.2),
    right=(0.3, -0.4),
    gripper_l=0.0,
    gripper_r=0.0,
):
    return types.SimpleNamespace(
        name=[
            'arm_l_joint1',
            'arm_l_joint2',
            'arm_r_joint1',
            'arm_r_joint2',
            'gripper_l_joint1',
            'gripper_r_joint1',
        ],
        position=[
            left[0],
            left[1],
            right[0],
            right[1],
            gripper_l,
            gripper_r,
        ],
    )


def test_coerce_positions_accepts_comma_separated_strings():
    assert _coerce_positions('0.0, 1.5, -2') == [0.0, 1.5, -2.0]
    assert _coerce_positions('') == []


def test_stringified_annotations_map_to_port_types():
    assert _annotation_to_port_type('bool', None) == 'bool'
    assert _annotation_to_port_type('list[float]', None) == 'number'
    assert _annotation_to_port_type('Optional[int]', None) == 'number'
    assert _annotation_to_port_type('str', None) == 'string'


def test_resume_send_command_legacy_simulation_ignores_mode():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'ResumeInference',
        {'command': 'RESUME', 'inference_mode': 'simulation'},
    )

    assert action.inference_mode == ''


def test_load_send_command_sets_acceleration_mode():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'LoadInference',
        {
            'command': 'LOAD',
            'model': 'groot:n17',
            'acceleration_mode': 'tensorrt',
            'acceleration_engine_path': 'custom.trt',
        },
    )
    task_info = action._build_task_info()

    assert action.acceleration_mode == 'tensorrt_dit'
    assert task_info.acceleration_mode == 'tensorrt_dit'
    assert task_info.acceleration_engine_path == 'custom.trt'


def test_load_send_command_sets_action_request_mode():
    context = types.SimpleNamespace(node=_DummyNode())

    action = SendCommand.from_xml_params(
        context,
        'LoadInference',
        {
            'command': 'LOAD',
            'action_request_mode': 'sync',
        },
    )
    task_info = action._build_task_info()

    assert action.action_request_mode == 'sync'
    assert task_info.action_request_mode == 'sync'


def test_arm_state_gate_succeeds_without_gripper_detection():
    action = _make_wait_action()
    action._joint_state_callback(_joint_state())

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_left_gripper_requires_closed_then_opened():
    action = _make_wait_action(detect_left_gripper=True)
    action._joint_state_callback(_joint_state(gripper_l=0.0))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=1.0))
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.0))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_right_gripper_detection():
    action = _make_wait_action(detect_right_gripper=True)
    action._joint_state_callback(_joint_state(gripper_r=1.0))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_r=0.0))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_both_grippers_must_open():
    action = _make_wait_action(
        detect_left_gripper=True,
        detect_right_gripper=True,
    )
    action._joint_state_callback(_joint_state(gripper_l=1.0, gripper_r=1.0))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.0, gripper_r=1.0))
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.0, gripper_r=0.0))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_uses_custom_gripper_thresholds():
    action = _make_wait_action(
        detect_left_gripper=True,
        gripper_closed_value=0.8,
        gripper_open_value=0.2,
        gripper_threshold=0.02,
    )
    action._joint_state_callback(_joint_state(gripper_l=0.79))

    assert action.tick() == NodeStatus.RUNNING
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.23))
    assert action.tick() == NodeStatus.RUNNING

    action._joint_state_callback(_joint_state(gripper_l=0.21))
    assert action.tick() == NodeStatus.SUCCESS


def test_arm_state_gate_rejects_mismatched_targets():
    with pytest.raises(ValueError):
        _make_wait_action(
            left_target_joints='arm_l_joint1, arm_l_joint2',
            left_target_positions='0.1',
        )


def test_arm_state_gate_timeout_fails_when_unmet():
    action = _make_wait_action(timeout_sec=0.001)

    assert action.tick() == NodeStatus.RUNNING
    action._start_time -= 1.0
    assert action.tick() == NodeStatus.FAILURE


def test_arm_state_gate_catalog_entry_has_expected_ports():
    registry = build_registry()
    catalog = {entry['tag']: entry for entry in catalog_payload(registry)}
    ports = {
        port['name']: port
        for port in catalog['ArmStateGate']['ports']
    }

    assert catalog['ArmStateGate']['category'] == 'action'
    assert ports['left_target_joints']['type'] == 'string'
    assert ports['left_target_positions']['type'] == 'string'
    assert ports['joint_threshold']['type'] == 'number'
    assert ports['detect_left_gripper']['type'] == 'bool'
    assert ports['detect_right_gripper']['type'] == 'bool'
    assert ports['gripper_threshold']['type'] == 'number'
    assert ports['timeout_sec']['type'] == 'number'
