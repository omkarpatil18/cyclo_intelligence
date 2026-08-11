#!/usr/bin/env python3

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

robot_client_stub = types.ModuleType("robot_client")
robot_client_stub.RobotClient = object
sys.modules.setdefault("robot_client", robot_client_stub)

from main_runtime.control_loop import (  # noqa: E402
    ACTION_EXECUTION_MODE_CHUNK_ACK,
    ACTION_EXECUTION_MODE_ROW,
    ACTION_STEP_ACK_CANCELLED,
    ACTION_STEP_ACK_COMPLETED,
    ACTION_STEP_ACK_EXECUTED,
    ControlLoop,
    normalize_action_execution_mode,
)
from action_chunk_processing import ActionChunkProcessor  # noqa: E402


class FakeProcessor:
    output_hz = 100.0

    def __init__(self, actions=None, buffer_size=100) -> None:
        self._actions = list(actions or [])
        self.buffer_size = buffer_size
        self.clear_count = 0
        self.pushed_chunks = []
        self.scheduled_delays = []
        self.align_flags = []

    def pop_action(self):
        if self._actions:
            return self._actions.pop(0)
        return None

    def clear(self) -> None:
        self.clear_count += 1
        self._actions.clear()
        self.buffer_size = 0

    def push_actions(self, chunk, scheduled_start_delay_s=None, align=True):
        data = np.asarray(chunk, dtype=np.float64)
        self.pushed_chunks.append(data.copy())
        self.scheduled_delays.append(scheduled_start_delay_s)
        self.align_flags.append(bool(align))
        self.buffer_size += len(data)
        return len(data)


class FakeRobot:
    def __init__(self) -> None:
        self.commands = []
        self.previews = []
        self.idles = []
        self.action_chunks = []
        self.cancelled_action_chunks = []
        self.latest_action_step_ack = None
        self.action_keys = ["arm"]

    def publish_action(self, action, action_keys) -> None:
        self.commands.append((np.asarray(action).copy(), list(action_keys)))

    def publish_action_preview(self, action, action_keys) -> None:
        self.previews.append((np.asarray(action).copy(), list(action_keys)))

    def publish_idle_action(self, action_keys) -> None:
        self.idles.append(list(action_keys))

    def publish_action_chunk(self, chunk, *, session_id, seq_id) -> None:
        self.action_chunks.append(
            (
                np.asarray(chunk, dtype=np.float64).copy(),
                int(session_id),
                int(seq_id),
            )
        )

    def get_latest_action_step_ack(self):
        return self.latest_action_step_ack

    def cancel_action_chunk(self, session_id, seq_id) -> None:
        self.cancelled_action_chunks.append((int(session_id), int(seq_id)))

    def close(self) -> None:
        pass


class FakeRequester:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = []

    def get_action(self, task_instruction):
        self.calls.append(task_instruction)
        return self.response


class AlwaysAliveRequest:
    """Prevent a test tick from launching an unrelated refill thread."""

    @staticmethod
    def is_alive() -> bool:
        return True


def make_action_step_ack(
    *,
    session_id,
    seq_id,
    action_index,
    executed_steps,
    chunk_size,
    status,
    timestamp=1.0,
):
    return SimpleNamespace(
        session_id=session_id,
        seq_id=seq_id,
        action_index=action_index,
        executed_steps=executed_steps,
        chunk_size=chunk_size,
        status=status,
        executed_action=[],
        timestamp=timestamp,
    )


class ControlLoopSafetyTests(unittest.TestCase):
    def _make_loop(self, processor: FakeProcessor, robot: FakeRobot) -> ControlLoop:
        loop = ControlLoop(requester=object())
        loop._running = True
        loop._robot = robot
        loop._processor = processor
        loop._action_keys = ["arm"]
        return loop

    def _make_chunk_ack_loop(self, chunk_size=2, action_dim=2):
        action_list = np.arange(
            chunk_size * action_dim,
            dtype=np.float64,
        ).tolist()
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=chunk_size,
            action_dim=action_dim,
            action_list=action_list,
        )
        requester = FakeRequester(response)
        processor = FakeProcessor(buffer_size=0)
        robot = FakeRobot()
        loop = ControlLoop(
            requester=requester,
            target_chunk_size=chunk_size,
            action_execution_mode=ACTION_EXECUTION_MODE_CHUNK_ACK,
        )
        loop._running = True
        loop._robot = robot
        loop._processor = processor
        loop._publish_to_robot = True
        loop._action_execution_mode = ACTION_EXECUTION_MODE_CHUNK_ACK
        loop._chunk_session_id = 101
        return loop, requester, processor, robot

    def _publish_chunk(self, loop: ControlLoop) -> None:
        loop._request_and_buffer("pick", loop._generation, "sync")

    def test_dry_run_publishes_preview_without_robot_command(self) -> None:
        action = np.asarray([0.1, 0.2], dtype=np.float64)
        processor = FakeProcessor(actions=[action])
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)

        loop.set_publish_to_robot(False)
        loop.tick()

        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(len(robot.previews), 1)
        np.testing.assert_allclose(robot.previews[0][0], action)

    def test_robot_mode_publishes_preview_and_robot_command(self) -> None:
        action = np.asarray([0.3, 0.4], dtype=np.float64)
        processor = FakeProcessor(actions=[action])
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True

        loop.tick()

        self.assertEqual(len(robot.commands), 1)
        self.assertEqual(len(robot.previews), 1)
        np.testing.assert_allclose(robot.commands[0][0], action)
        np.testing.assert_allclose(robot.previews[0][0], action)

    def test_row_mode_streams_one_unmodified_action_per_tick(self) -> None:
        chunk_size = 30
        action_dim = 22
        chunk = np.arange(chunk_size * action_dim, dtype=np.float64).reshape(
            chunk_size,
            action_dim,
        )
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=chunk_size,
            action_dim=action_dim,
            action_list=chunk.reshape(-1).tolist(),
        )
        requester = FakeRequester(response)
        processor = ActionChunkProcessor(
            inference_hz=15.0,
            control_hz=15.0,
            postprocess=False,
            target_chunk_size=chunk_size,
        )
        robot = FakeRobot()
        loop = ControlLoop(
            requester=requester,
            inference_hz=15.0,
            control_hz=15.0,
            target_chunk_size=chunk_size,
            postprocess_actions=False,
            action_request_mode="sync",
            action_execution_mode=ACTION_EXECUTION_MODE_ROW,
        )
        loop._running = True
        loop._robot = robot
        loop._processor = processor
        loop._publish_to_robot = True
        loop._action_keys = ["arm"]

        loop._request_and_buffer("pick", loop._generation, "sync")
        loop._request_thread = AlwaysAliveRequest()

        self.assertEqual(processor.buffer_size, chunk_size)
        self.assertAlmostEqual(loop._tick_period(), 1.0 / 15.0)
        self.assertEqual(robot.action_chunks, [])

        for expected_count in range(1, chunk_size + 1):
            loop.tick()
            self.assertEqual(len(robot.commands), expected_count)
            self.assertEqual(processor.buffer_size, chunk_size - expected_count)

        self.assertEqual(len(robot.previews), chunk_size)
        np.testing.assert_array_equal(
            np.stack([action for action, _keys in robot.commands]),
            chunk,
        )
        loop.tick()
        self.assertEqual(len(robot.commands), chunk_size)
        self.assertEqual(robot.idles, [["arm"]])

    def test_robot_publish_error_does_not_crash_tick(self) -> None:
        class FailingRobot(FakeRobot):
            def publish_action(self, action, action_keys) -> None:
                raise RuntimeError("publish failed")

        processor = FakeProcessor(actions=[np.asarray([0.5], dtype=np.float64)])
        robot = FailingRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True

        loop.tick()

        self.assertEqual(len(robot.previews), 1)

    def test_robot_mode_publishes_idle_when_action_buffer_is_empty(self) -> None:
        processor = FakeProcessor(actions=[], buffer_size=100)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = True
        loop._action_keys = ["mobile"]

        loop.tick()

        self.assertEqual(robot.idles, [["mobile"]])
        self.assertEqual(len(robot.commands), 0)
        self.assertEqual(len(robot.previews), 0)

    def test_dry_run_does_not_publish_idle_when_action_buffer_is_empty(self) -> None:
        processor = FakeProcessor(actions=[], buffer_size=100)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._publish_to_robot = False
        loop._action_keys = ["mobile"]

        loop.tick()

        self.assertEqual(robot.idles, [])

    def test_mode_change_clears_buffer(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)

        loop.set_publish_to_robot(True)

        self.assertEqual(processor.clear_count, 1)

    def test_pause_clears_buffer(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)

        loop.pause()

        self.assertEqual(processor.clear_count, 1)

    def test_refill_threshold_includes_observed_request_latency(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._refill_margin_s = 0.25
        loop._request_latency_ema_s = 0.25

        self.assertEqual(loop._refill_threshold(processor), 50)

    def test_initial_latency_sample_is_ignored_for_warmup(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._latency_warmup_remaining = 1

        loop._record_request_latency(5.0)
        self.assertIsNone(loop._request_latency_ema_s)

        loop._record_request_latency(0.25)
        self.assertEqual(loop._request_latency_ema_s, 0.25)

    def test_refill_latency_outlier_is_ignored(self) -> None:
        processor = FakeProcessor()
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._latency_warmup_remaining = 0
        loop._max_refill_latency_s = 1.0

        loop._record_request_latency(0.2)
        loop._record_request_latency(5.0)

        self.assertEqual(loop._request_latency_ema_s, 0.2)

    def test_async_mode_requests_before_buffer_is_empty(self) -> None:
        processor = FakeProcessor(buffer_size=10)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._action_request_mode = "async"
        loop._refill_margin_s = 0.2
        loop._request_latency_ema_s = None

        self.assertTrue(loop._should_request_actions(processor))

        processor.buffer_size = 30
        self.assertFalse(loop._should_request_actions(processor))

    def test_sync_mode_waits_until_buffer_is_empty(self) -> None:
        processor = FakeProcessor(buffer_size=1)
        robot = FakeRobot()
        loop = self._make_loop(processor, robot)
        loop._action_request_mode = "sync"

        self.assertFalse(loop._should_request_actions(processor))

        processor.buffer_size = 0
        self.assertTrue(loop._should_request_actions(processor))

    def test_sync_mode_buffers_chunk_without_scheduled_skip(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2, 0.3, 0.4],
        )
        processor = FakeProcessor(buffer_size=0)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._robot = FakeRobot()
        loop._processor = processor

        loop._request_and_buffer("pick", loop._generation, "sync")

        self.assertEqual(len(processor.pushed_chunks), 1)
        self.assertIsNone(processor.scheduled_delays[-1])
        self.assertEqual(processor.align_flags[-1], False)

    def test_async_mode_buffers_chunk_with_latency_and_buffer_delay(self) -> None:
        response = SimpleNamespace(
            success=True,
            message="ok",
            chunk_size=2,
            action_dim=2,
            action_list=[0.1, 0.2, 0.3, 0.4],
        )
        processor = FakeProcessor(buffer_size=50)
        loop = ControlLoop(requester=FakeRequester(response))
        loop._running = True
        loop._robot = FakeRobot()
        loop._processor = processor

        loop._request_and_buffer("pick", loop._generation, "async")

        self.assertEqual(len(processor.pushed_chunks), 1)
        self.assertIsNotNone(processor.scheduled_delays[-1])
        self.assertGreaterEqual(processor.scheduled_delays[-1], 0.5)
        self.assertEqual(processor.align_flags[-1], True)

    def test_normalize_action_execution_mode_accepts_chunk_ack(self) -> None:
        self.assertEqual(
            normalize_action_execution_mode(" CHUNK_ACK "),
            ACTION_EXECUTION_MODE_CHUNK_ACK,
        )
        self.assertEqual(
            normalize_action_execution_mode("unsupported"),
            ACTION_EXECUTION_MODE_ROW,
        )
        self.assertEqual(
            normalize_action_execution_mode(None),
            ACTION_EXECUTION_MODE_ROW,
        )

    def test_chunk_ack_publishes_full_chunk_once_without_processor_push(self) -> None:
        loop, requester, processor, robot = self._make_chunk_ack_loop()

        self._publish_chunk(loop)
        self._publish_chunk(loop)

        self.assertEqual(requester.calls, ["pick", "pick"])
        self.assertEqual(len(robot.action_chunks), 1)
        chunk, session_id, seq_id = robot.action_chunks[0]
        np.testing.assert_array_equal(
            chunk,
            np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
        )
        self.assertEqual((session_id, seq_id), (101, 0))
        self.assertEqual(processor.pushed_chunks, [])
        self.assertEqual(loop._awaiting_chunk_seq_id, 0)
        self.assertEqual(loop._next_chunk_seq_id, 1)

    def test_chunk_ack_executed_ack_keeps_next_request_gated(self) -> None:
        loop, _requester, processor, robot = self._make_chunk_ack_loop()
        self._publish_chunk(loop)
        robot.latest_action_step_ack = make_action_step_ack(
            session_id=101,
            seq_id=0,
            action_index=0,
            executed_steps=1,
            chunk_size=2,
            status=ACTION_STEP_ACK_EXECUTED,
        )

        loop._consume_action_step_ack_locked(robot)

        self.assertEqual(loop._acked_chunk_steps, 1)
        self.assertEqual(loop._awaiting_chunk_seq_id, 0)
        self.assertFalse(loop._should_request_actions(processor))

    def test_chunk_ack_ignores_wrong_ids_and_malformed_completed_ack(self) -> None:
        loop, _requester, processor, robot = self._make_chunk_ack_loop()
        self._publish_chunk(loop)
        ignored_acks = [
            make_action_step_ack(
                session_id=999,
                seq_id=0,
                action_index=1,
                executed_steps=2,
                chunk_size=2,
                status=ACTION_STEP_ACK_COMPLETED,
                timestamp=1.0,
            ),
            make_action_step_ack(
                session_id=101,
                seq_id=999,
                action_index=1,
                executed_steps=2,
                chunk_size=2,
                status=ACTION_STEP_ACK_COMPLETED,
                timestamp=2.0,
            ),
            make_action_step_ack(
                session_id=101,
                seq_id=0,
                action_index=0,
                executed_steps=1,
                chunk_size=2,
                status=ACTION_STEP_ACK_COMPLETED,
                timestamp=3.0,
            ),
            make_action_step_ack(
                session_id=101,
                seq_id=0,
                action_index=1,
                executed_steps=2,
                chunk_size=3,
                status=ACTION_STEP_ACK_COMPLETED,
                timestamp=4.0,
            ),
        ]

        for ack in ignored_acks:
            robot.latest_action_step_ack = ack
            loop._consume_action_step_ack_locked(robot)
            self.assertEqual(loop._awaiting_chunk_seq_id, 0)
            self.assertFalse(loop._should_request_actions(processor))

    def test_chunk_ack_matching_completed_ack_releases_next_request(self) -> None:
        loop, _requester, processor, robot = self._make_chunk_ack_loop()
        self._publish_chunk(loop)
        robot.latest_action_step_ack = make_action_step_ack(
            session_id=101,
            seq_id=0,
            action_index=1,
            executed_steps=2,
            chunk_size=2,
            status=ACTION_STEP_ACK_COMPLETED,
        )

        loop._consume_action_step_ack_locked(robot)

        self.assertIsNone(loop._awaiting_chunk_seq_id)
        self.assertTrue(loop._should_request_actions(processor))

    def test_chunk_ack_matching_cancelled_ack_releases_next_request(self) -> None:
        loop, _requester, processor, robot = self._make_chunk_ack_loop()
        self._publish_chunk(loop)
        robot.latest_action_step_ack = make_action_step_ack(
            session_id=101,
            seq_id=0,
            action_index=0,
            executed_steps=1,
            chunk_size=2,
            status=ACTION_STEP_ACK_CANCELLED,
        )

        loop._consume_action_step_ack_locked(robot)

        self.assertIsNone(loop._awaiting_chunk_seq_id)
        self.assertTrue(loop._should_request_actions(processor))

    def test_pause_cancels_pending_chunk_and_clears_local_state(self) -> None:
        loop, _requester, processor, robot = self._make_chunk_ack_loop()
        self._publish_chunk(loop)

        loop.pause()

        self.assertEqual(robot.cancelled_action_chunks, [(101, 0)])
        self.assertIsNone(loop._awaiting_chunk_seq_id)
        self.assertFalse(loop._running)
        self.assertEqual(processor.clear_count, 1)


if __name__ == "__main__":
    unittest.main()
