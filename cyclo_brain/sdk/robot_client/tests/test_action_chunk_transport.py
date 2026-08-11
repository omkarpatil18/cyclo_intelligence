#!/usr/bin/env python3

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:  # The transport-only test does not exercise image decoding.
    sys.modules["cv2"] = SimpleNamespace()


SDK_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
ZENOH_SDK_ROOT = REPO_ROOT / "cyclo_brain" / "sdk" / "zenoh_ros2_sdk"
ROBOT_CONFIG_ROOT = REPO_ROOT / "shared" / "shared" / "robot_configs"
for path in (SDK_ROOT, ZENOH_SDK_ROOT, ROBOT_CONFIG_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import robot_client.robot_client as robot_client_module  # noqa: E402
from robot_client.messages import (  # noqa: E402
    ACTION_CHUNK_DEF,
    ACTION_CHUNK_TOPIC,
    ACTION_STEP_ACK_DEF,
    ACTION_STEP_ACK_TOPIC,
)


class FakeEndpoint:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.messages = []
        self.closed = False

    def publish(self, **kwargs) -> None:
        self.messages.append(kwargs)

    def close(self) -> None:
        self.closed = True


def _bare_client():
    client = robot_client_module.RobotClient.__new__(robot_client_module.RobotClient)
    client._lock = threading.Lock()
    client._router_ip = "127.0.0.1"
    client._router_port = 7447
    client._domain_id = 0
    client._subscribers = []
    client._command_publishers = {}
    client._preview_publisher = None
    client._action_chunk_publisher = None
    client._latest_action_step_ack = None
    client._closed = False
    return client


class ActionChunkTransportTests(unittest.TestCase):
    def test_transport_uses_atomic_chunk_and_ack_contracts(self) -> None:
        client = _bare_client()
        with (
            patch.object(robot_client_module, "ROS2Publisher", FakeEndpoint),
            patch.object(robot_client_module, "ROS2Subscriber", FakeEndpoint),
        ):
            client._init_chunk_transport()

        publisher = client._action_chunk_publisher
        subscriber = client._subscribers[-1]
        self.assertEqual(publisher.kwargs["topic"], ACTION_CHUNK_TOPIC)
        self.assertEqual(publisher.kwargs["msg_definition"], ACTION_CHUNK_DEF)
        self.assertEqual(subscriber.kwargs["topic"], ACTION_STEP_ACK_TOPIC)
        self.assertEqual(subscriber.kwargs["msg_definition"], ACTION_STEP_ACK_DEF)

    def test_publish_chunk_is_row_major_and_cancel_is_zero_sized(self) -> None:
        client = _bare_client()
        publisher = FakeEndpoint()
        client._action_chunk_publisher = publisher
        chunk = np.arange(12, dtype=np.float64).reshape(3, 4)

        client.publish_action_chunk(chunk, session_id=8, seq_id=2)
        message = publisher.messages[-1]
        self.assertEqual(message["session_id"], 8)
        self.assertEqual(message["seq_id"], 2)
        self.assertEqual(message["chunk_size"], 3)
        self.assertEqual(message["action_dim"], 4)
        self.assertIsInstance(message["data"], np.ndarray)
        self.assertTrue(message["data"].flags.c_contiguous)
        np.testing.assert_array_equal(message["data"], np.arange(12, dtype=np.float64))

        client.cancel_action_chunk(8, 2)
        self.assertEqual(publisher.messages[-1]["chunk_size"], 0)
        self.assertEqual(publisher.messages[-1]["action_dim"], 0)
        self.assertIsInstance(publisher.messages[-1]["data"], np.ndarray)
        self.assertEqual(publisher.messages[-1]["data"].dtype, np.float64)
        self.assertEqual(publisher.messages[-1]["data"].size, 0)

        with self.assertRaises(ValueError):
            client.publish_action_chunk(np.zeros(4), 8, 3)
        with self.assertRaises(ValueError):
            client.publish_action_chunk(np.asarray([[np.nan]]), 8, 3)

    def test_ack_snapshot_is_thread_safe_and_defensive(self) -> None:
        client = _bare_client()
        client._update_action_step_ack(
            SimpleNamespace(
                session_id=9,
                seq_id=3,
                action_index=1,
                executed_steps=2,
                chunk_size=30,
                status=0,
                executed_action=[1.0, 2.0],
                timestamp=12.5,
            )
        )

        first = client.get_latest_action_step_ack()
        first["executed_action"][0] = 99.0
        second = client.get_latest_action_step_ack()
        self.assertEqual(second["session_id"], 9)
        np.testing.assert_allclose(second["executed_action"], [1.0, 2.0])

    def test_close_closes_chunk_endpoints(self) -> None:
        client = _bare_client()
        subscriber = FakeEndpoint()
        publisher = FakeEndpoint()
        client._subscribers.append(subscriber)
        client._action_chunk_publisher = publisher

        client.close()

        self.assertTrue(subscriber.closed)
        self.assertTrue(publisher.closed)


if __name__ == "__main__":
    unittest.main()
