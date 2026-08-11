from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_rosbag_control(monkeypatch):
    class SendCommand:
        class Request:
            PREPARE = 1
            START = 2
            STOP = 3
            STOP_AND_DELETE = 4
            FINISH = 5

    class String:
        def __init__(self):
            self.data = ''

    modules = {
        'rclpy': ModuleType('rclpy'),
        'rclpy.callback_groups': ModuleType('rclpy.callback_groups'),
        'rclpy.node': ModuleType('rclpy.node'),
        'rosbag_recorder': ModuleType('rosbag_recorder'),
        'rosbag_recorder.srv': ModuleType('rosbag_recorder.srv'),
        'std_msgs': ModuleType('std_msgs'),
        'std_msgs.msg': ModuleType('std_msgs.msg'),
    }
    modules['rclpy.callback_groups'].CallbackGroup = object
    modules['rclpy.callback_groups'].ReentrantCallbackGroup = object
    modules['rclpy.node'].Node = object
    modules['rosbag_recorder.srv'].SendCommand = SendCommand
    modules['std_msgs.msg'].String = String
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    path = REPO_ROOT / 'cyclo_data/cyclo_data/recorder/rosbag_control.py'
    spec = spec_from_file_location('rosbag_control_sync_test', path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_waits_for_rosbag_response(monkeypatch):
    module = _load_rosbag_control(monkeypatch)
    control = object.__new__(module.RosbagControl)
    control._send_rosbag_command = Mock()

    control.prepare_rosbag(['/joint_states', '/inference/action_chunk'], 1.25)

    control._send_rosbag_command.assert_called_once_with(
        command=module.SendCommand.Request.PREPARE,
        topics=['/joint_states', '/inference/action_chunk'],
        wait_for_response=True,
        timeout_sec=1.25,
    )
