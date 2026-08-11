#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""Robot-facing control loop owned by the Main process."""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np


_parents = Path(__file__).resolve().parents
_default_acp = str(_parents[4] / "sdk" / "action_chunk_processing") if len(_parents) > 4 else ""
_ACTION_CHUNK_PATH = os.environ.get("ACTION_CHUNK_PROCESSING_SDK_PATH", _default_acp)
if os.path.exists(_ACTION_CHUNK_PATH) and _ACTION_CHUNK_PATH not in sys.path:
    sys.path.insert(0, _ACTION_CHUNK_PATH)

_default_rc = str(_parents[4] / "sdk" / "robot_client") if len(_parents) > 4 else ""
_ROBOT_CLIENT_PATH = os.environ.get("ROBOT_CLIENT_SDK_PATH", _default_rc)
if os.path.exists(_ROBOT_CLIENT_PATH) and _ROBOT_CLIENT_PATH not in sys.path:
    sys.path.insert(0, _ROBOT_CLIENT_PATH)

from action_chunk_processing import ActionChunkProcessor  # noqa: E402
from robot_client import RobotClient  # noqa: E402


try:  # pragma: no cover - SDK exists only in runtime container here.
    from zenoh_ros2_sdk import get_logger
except Exception:  # pragma: no cover
    import logging

    def get_logger(name: str):
        return logging.getLogger(name)


logger = get_logger("main_runtime.control_loop")

ACTION_REQUEST_MODE_ASYNC = "async"
ACTION_REQUEST_MODE_SYNC = "sync"
ACTION_REQUEST_MODES = {ACTION_REQUEST_MODE_ASYNC, ACTION_REQUEST_MODE_SYNC}
ACTION_EXECUTION_MODE_ROW = "row"
ACTION_EXECUTION_MODE_CHUNK_ACK = "chunk_ack"
ACTION_EXECUTION_MODES = {
    ACTION_EXECUTION_MODE_ROW,
    ACTION_EXECUTION_MODE_CHUNK_ACK,
}
ACTION_STEP_ACK_EXECUTED = 0
ACTION_STEP_ACK_COMPLETED = 1
ACTION_STEP_ACK_CANCELLED = 2


def normalize_action_request_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == ACTION_REQUEST_MODE_SYNC:
        return ACTION_REQUEST_MODE_SYNC
    return ACTION_REQUEST_MODE_ASYNC


def normalize_action_execution_mode(value: object) -> str:
    mode = str(value or "").strip().lower()
    if mode == ACTION_EXECUTION_MODE_CHUNK_ACK:
        return ACTION_EXECUTION_MODE_CHUNK_ACK
    return ACTION_EXECUTION_MODE_ROW


class ControlLoop:
    """Ticks RobotClient command publishing and refills action buffers."""

    def __init__(
        self,
        requester,
        inference_hz: float = 15.0,
        control_hz: float = 100.0,
        chunk_align_window_s: float = 0.3,
        target_chunk_size: Optional[int] = None,
        postprocess_actions: bool = True,
        alignment_mode: str = "l2",
        refill_margin_s: float = 0.2,
        latency_warmup_samples: int = 1,
        max_refill_latency_s: Optional[float] = 2.0,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
        action_execution_mode: str = ACTION_EXECUTION_MODE_ROW,
        action_chunk_ack_timeout_s: float = 30.0,
    ) -> None:
        self._requester = requester
        self._inference_hz = float(inference_hz)
        self._control_hz = float(control_hz)
        self._chunk_align_window_s = float(chunk_align_window_s)
        self._target_chunk_size = target_chunk_size
        self._postprocess_actions = bool(postprocess_actions)
        self._alignment_mode = alignment_mode
        self._refill_margin_s = float(refill_margin_s)
        self._request_latency_ema_s: Optional[float] = None
        self._request_latency_alpha = 0.2
        self._latency_warmup_samples = max(0, int(latency_warmup_samples))
        self._latency_warmup_remaining = self._latency_warmup_samples
        self._max_refill_latency_s = (
            None
            if max_refill_latency_s is None or max_refill_latency_s <= 0.0
            else float(max_refill_latency_s)
        )
        self._default_action_request_mode = normalize_action_request_mode(
            action_request_mode
        )
        self._action_request_mode = self._default_action_request_mode
        self._default_action_execution_mode = normalize_action_execution_mode(
            action_execution_mode
        )
        self._action_execution_mode = self._default_action_execution_mode
        self._action_chunk_ack_timeout_s = max(
            0.0,
            float(action_chunk_ack_timeout_s),
        )

        self._lock = threading.RLock()
        self._robot: Optional[RobotClient] = None
        self._processor: Optional[ActionChunkProcessor] = None
        self._task_instruction = ""
        self._action_keys: list[str] = []
        self._publish_to_robot = False
        self._running = False
        self._generation = 0
        self._shutdown = threading.Event()
        self._request_thread: Optional[threading.Thread] = None
        self._thread: Optional[threading.Thread] = None
        self._chunk_session_id = 0
        self._next_chunk_seq_id = 0
        self._awaiting_chunk_seq_id: Optional[int] = None
        self._awaiting_chunk_size = 0
        self._awaiting_chunk_started_at: Optional[float] = None
        self._acked_chunk_steps = 0
        self._last_action_step_ack_token: Optional[tuple] = None
        self._action_chunk_timeout_warned = False

    def configure(
        self,
        robot_type: str,
        task_instruction: str = "",
        action_keys: Optional[list[str]] = None,
        publish_to_robot: bool = False,
        action_request_mode: Optional[str] = None,
        action_execution_mode: Optional[str] = None,
    ) -> None:
        with self._lock:
            self.deconfigure()
            self._action_request_mode = normalize_action_request_mode(
                action_request_mode
                if action_request_mode is not None
                else self._default_action_request_mode
            )
            self._action_execution_mode = normalize_action_execution_mode(
                action_execution_mode
                if action_execution_mode is not None
                else self._default_action_execution_mode
            )
            self._robot = RobotClient(
                robot_type,
                enable_command_publishers=True,
                enable_preview_publisher=True,
                enable_chunk_transport=(
                    self._action_execution_mode == ACTION_EXECUTION_MODE_CHUNK_ACK
                ),
            )
            self._processor = ActionChunkProcessor(
                inference_hz=self._inference_hz,
                control_hz=self._control_hz,
                chunk_align_window_s=self._chunk_align_window_s,
                postprocess=self._postprocess_actions,
                target_chunk_size=self._target_chunk_size,
                alignment_mode=self._alignment_mode,
            )
            self._task_instruction = task_instruction or ""
            self._action_keys = list(action_keys or self._robot.action_keys)
            self._publish_to_robot = bool(publish_to_robot)
            self._chunk_session_id = time.time_ns()
            self._next_chunk_seq_id = 0
            self._clear_awaiting_chunk_locked()
            self._reset_request_latency_locked()
            self._generation += 1
            logger.info(
                "configured RobotClient command path for %s "
                "(publish_to_robot=%s action_request_mode=%s "
                "action_execution_mode=%s session_id=%d)",
                robot_type,
                self._publish_to_robot,
                self._action_request_mode,
                self._action_execution_mode,
                self._chunk_session_id,
            )

    def deconfigure(self) -> None:
        with self._lock:
            self._cancel_awaiting_chunk_locked()
            self._running = False
            self._task_instruction = ""
            self._action_keys = []
            self._publish_to_robot = False
            self._action_request_mode = self._default_action_request_mode
            self._action_execution_mode = self._default_action_execution_mode
            self._processor = None
            self._generation += 1
            if self._robot is not None:
                self._robot.close()
                self._robot = None
            self._chunk_session_id = 0
            self._next_chunk_seq_id = 0
            self._clear_awaiting_chunk_locked()
            self._reset_request_latency_locked()

    def start(self, publish_to_robot: Optional[bool] = None) -> None:
        with self._lock:
            if publish_to_robot is not None:
                self._set_publish_to_robot_locked(bool(publish_to_robot))
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._cancel_awaiting_chunk_locked()
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1

    def stop(self) -> None:
        with self._lock:
            self._cancel_awaiting_chunk_locked()
            self._running = False
            if self._processor is not None:
                self._processor.clear()
            self._generation += 1

    def set_publish_to_robot(self, publish_to_robot: bool) -> None:
        with self._lock:
            self._set_publish_to_robot_locked(bool(publish_to_robot))

    def _set_publish_to_robot_locked(self, publish_to_robot: bool) -> None:
        if self._publish_to_robot == publish_to_robot:
            return
        if self._publish_to_robot and not publish_to_robot:
            self._cancel_awaiting_chunk_locked()
        self._publish_to_robot = publish_to_robot
        if self._processor is not None:
            self._processor.clear()
        self._generation += 1

    def set_task_instruction(self, task_instruction: str) -> None:
        with self._lock:
            self._task_instruction = task_instruction or ""

    def run_background(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self) -> None:
        next_t = time.monotonic()
        while not self._shutdown.is_set():
            period = self._tick_period()
            self.tick()
            next_t += period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.deconfigure()

    def tick(self) -> None:
        with self._lock:
            if not self._running or self._robot is None or self._processor is None:
                return
            robot = self._robot
            processor = self._processor
            task_instruction = self._task_instruction
            action_keys = list(self._action_keys)
            generation = self._generation
            publish_to_robot = self._publish_to_robot
            action_request_mode = self._action_request_mode

            if self._chunk_ack_active_locked():
                self._consume_action_step_ack_locked(robot)
                self._warn_if_action_chunk_timed_out_locked()
            else:
                action = processor.pop_action()
                if action is not None:
                    preview = getattr(robot, "publish_action_preview", None)
                    if callable(preview):
                        try:
                            preview(action, action_keys)
                        except Exception as e:
                            logger.warning("failed to publish action preview: %s", e)
                    if publish_to_robot:
                        try:
                            robot.publish_action(action, action_keys)
                        except Exception as e:
                            logger.error("failed to publish robot action: %s", e)
                elif publish_to_robot:
                    idle = getattr(robot, "publish_idle_action", None)
                    if callable(idle):
                        try:
                            idle(action_keys)
                        except Exception as e:
                            logger.error("failed to publish idle robot action: %s", e)

            should_request = self._should_request_actions(processor)

        if should_request:
            self._request_thread = threading.Thread(
                target=self._request_and_buffer,
                args=(task_instruction, generation, action_request_mode),
                daemon=True,
            )
            self._request_thread.start()

    def _request_and_buffer(
        self,
        task_instruction: str,
        generation: int,
        action_request_mode: str = ACTION_REQUEST_MODE_ASYNC,
    ) -> None:
        action_request_mode = normalize_action_request_mode(action_request_mode)
        started_at = time.monotonic()
        try:
            response = self._requester.get_action(task_instruction)
        except Exception as e:
            latency_s = time.monotonic() - started_at
            self._record_request_latency(latency_s)
            logger.warning("get_action raised: %s", e)
            return
        latency_s = time.monotonic() - started_at
        self._record_request_latency(latency_s)
        if not response.success:
            logger.warning("get_action failed: %s", response.message)
            return
        if response.chunk_size <= 0 or response.action_dim <= 0:
            logger.warning("get_action returned empty action list")
            return
        data = np.asarray(response.action_list, dtype=np.float64)
        if data.size != response.chunk_size * response.action_dim:
            logger.warning(
                "action list size mismatch: %d != %d * %d",
                data.size,
                response.chunk_size,
                response.action_dim,
            )
            return
        chunk = data.reshape(response.chunk_size, response.action_dim)
        with self._lock:
            if (
                generation == self._generation
                and self._running
                and self._processor is not None
                and self._robot is not None
            ):
                if self._chunk_ack_active_locked():
                    if self._awaiting_chunk_seq_id is not None:
                        logger.warning(
                            "discarding action response while chunk %d still awaits ACK",
                            self._awaiting_chunk_seq_id,
                        )
                        return
                    if (
                        self._target_chunk_size is not None
                        and response.chunk_size != self._target_chunk_size
                    ):
                        logger.warning(
                            "chunk_ack requires exactly %d actions, got %d",
                            self._target_chunk_size,
                            response.chunk_size,
                        )
                        return
                    seq_id = self._next_chunk_seq_id
                    self._next_chunk_seq_id += 1
                    self._awaiting_chunk_seq_id = seq_id
                    self._awaiting_chunk_size = int(response.chunk_size)
                    self._awaiting_chunk_started_at = time.monotonic()
                    self._acked_chunk_steps = 0
                    self._last_action_step_ack_token = None
                    self._action_chunk_timeout_warned = False
                    try:
                        self._robot.publish_action_chunk(
                            chunk,
                            session_id=self._chunk_session_id,
                            seq_id=seq_id,
                        )
                    except Exception as e:
                        logger.error("failed to publish action chunk: %s", e)
                        self._clear_awaiting_chunk_locked()
                        return
                    logger.info(
                        "published action chunk: session=%d seq=%d shape=%dx%d; "
                        "waiting for environment completion ACK",
                        self._chunk_session_id,
                        seq_id,
                        response.chunk_size,
                        response.action_dim,
                    )
                    return

                buffer_delay_s = self._processor.buffer_size / max(
                    1.0,
                    self._processor.output_hz,
                )
                scheduled_start_delay_s = (
                    None
                    if action_request_mode == ACTION_REQUEST_MODE_SYNC
                    else latency_s + buffer_delay_s
                )
                produced = self._processor.push_actions(
                    chunk,
                    scheduled_start_delay_s=scheduled_start_delay_s,
                    align=action_request_mode != ACTION_REQUEST_MODE_SYNC,
                )
                scheduled_start_text = (
                    "none"
                    if scheduled_start_delay_s is None
                    else f"{scheduled_start_delay_s:.3f}s"
                )
                logger.debug(
                    "buffered action chunk: source=%d produced=%d "
                    "mode=%s latency=%.3fs buffer_delay=%.3fs "
                    "scheduled_start=%s",
                    response.chunk_size,
                    produced,
                    action_request_mode,
                    latency_s,
                    buffer_delay_s,
                    scheduled_start_text,
                )

    def _should_request_actions(self, processor: ActionChunkProcessor) -> bool:
        if self._request_thread is not None and self._request_thread.is_alive():
            return False
        if self._chunk_ack_active_locked():
            return self._awaiting_chunk_seq_id is None
        if self._action_request_mode == ACTION_REQUEST_MODE_SYNC:
            return processor.buffer_size <= 0
        return processor.buffer_size < self._refill_threshold(processor)

    def _chunk_ack_active_locked(self) -> bool:
        return (
            self._action_execution_mode == ACTION_EXECUTION_MODE_CHUNK_ACK
            and self._publish_to_robot
        )

    def _consume_action_step_ack_locked(self, robot: RobotClient) -> None:
        getter = getattr(robot, "get_latest_action_step_ack", None)
        if not callable(getter):
            return
        try:
            ack = getter()
        except Exception as e:
            logger.warning("failed to read action step ACK: %s", e)
            return
        if ack is None:
            return

        def value(name: str, default=0):
            if isinstance(ack, dict):
                return ack.get(name, default)
            return getattr(ack, name, default)

        token = (
            int(value("session_id")),
            int(value("seq_id")),
            int(value("action_index", -1)),
            int(value("executed_steps")),
            int(value("status", -1)),
            float(value("timestamp", 0.0)),
        )
        if token == self._last_action_step_ack_token:
            return
        self._last_action_step_ack_token = token
        if self._awaiting_chunk_seq_id is None:
            return
        if (
            token[0] != self._chunk_session_id
            or token[1] != self._awaiting_chunk_seq_id
        ):
            return

        action_index = token[2]
        executed_steps = token[3]
        status = token[4]
        ack_chunk_size = int(value("chunk_size"))
        if status == ACTION_STEP_ACK_EXECUTED:
            if (
                0 <= action_index < self._awaiting_chunk_size
                and executed_steps == action_index + 1
            ):
                self._acked_chunk_steps = max(
                    self._acked_chunk_steps,
                    executed_steps,
                )
            return
        if status == ACTION_STEP_ACK_COMPLETED:
            if (
                ack_chunk_size != self._awaiting_chunk_size
                or executed_steps != self._awaiting_chunk_size
                or action_index != self._awaiting_chunk_size - 1
            ):
                logger.warning(
                    "ignoring malformed completed ACK for seq=%d: "
                    "index=%d executed=%d chunk_size=%d expected=%d",
                    self._awaiting_chunk_seq_id,
                    action_index,
                    executed_steps,
                    ack_chunk_size,
                    self._awaiting_chunk_size,
                )
                return
            logger.info(
                "environment completed action chunk: session=%d seq=%d steps=%d",
                self._chunk_session_id,
                self._awaiting_chunk_seq_id,
                executed_steps,
            )
            self._clear_awaiting_chunk_locked()
            return
        if status == ACTION_STEP_ACK_CANCELLED:
            logger.info(
                "environment cancelled action chunk: session=%d seq=%d "
                "executed=%d/%d",
                self._chunk_session_id,
                self._awaiting_chunk_seq_id,
                executed_steps,
                self._awaiting_chunk_size,
            )
            self._clear_awaiting_chunk_locked()

    def _warn_if_action_chunk_timed_out_locked(self) -> None:
        if (
            self._awaiting_chunk_seq_id is None
            or self._awaiting_chunk_started_at is None
            or self._action_chunk_ack_timeout_s <= 0.0
            or self._action_chunk_timeout_warned
        ):
            return
        elapsed_s = time.monotonic() - self._awaiting_chunk_started_at
        if elapsed_s < self._action_chunk_ack_timeout_s:
            return
        self._action_chunk_timeout_warned = True
        logger.warning(
            "action chunk ACK timeout: session=%d seq=%d acked=%d/%d "
            "elapsed=%.1fs; not republishing to avoid duplicate execution",
            self._chunk_session_id,
            self._awaiting_chunk_seq_id,
            self._acked_chunk_steps,
            self._awaiting_chunk_size,
            elapsed_s,
        )

    def _cancel_awaiting_chunk_locked(self) -> None:
        if self._awaiting_chunk_seq_id is None:
            return
        if self._robot is not None:
            cancel = getattr(self._robot, "cancel_action_chunk", None)
            if callable(cancel):
                try:
                    cancel(self._chunk_session_id, self._awaiting_chunk_seq_id)
                except Exception as e:
                    logger.warning("failed to cancel external action chunk: %s", e)
        self._clear_awaiting_chunk_locked()

    def _clear_awaiting_chunk_locked(self) -> None:
        self._awaiting_chunk_seq_id = None
        self._awaiting_chunk_size = 0
        self._awaiting_chunk_started_at = None
        self._acked_chunk_steps = 0
        self._last_action_step_ack_token = None
        self._action_chunk_timeout_warned = False

    def _refill_threshold(self, processor: ActionChunkProcessor) -> int:
        threshold_s = max(0.0, self._refill_margin_s)
        if self._request_latency_ema_s is not None:
            threshold_s += max(0.0, self._request_latency_ema_s)
        return max(1, int(math.ceil(threshold_s * processor.output_hz)))

    def _record_request_latency(self, latency_s: float) -> None:
        latency_s = max(0.0, float(latency_s))
        with self._lock:
            if self._latency_warmup_remaining > 0:
                self._latency_warmup_remaining -= 1
                return
            if (
                self._max_refill_latency_s is not None
                and latency_s > self._max_refill_latency_s
            ):
                logger.debug(
                    "ignoring GET_ACTION latency sample %.3fs above %.3fs",
                    latency_s,
                    self._max_refill_latency_s,
                )
                return
            if self._request_latency_ema_s is None:
                self._request_latency_ema_s = latency_s
            else:
                alpha = self._request_latency_alpha
                self._request_latency_ema_s = (
                    alpha * latency_s
                    + (1.0 - alpha) * self._request_latency_ema_s
                )

    def _reset_request_latency_locked(self) -> None:
        self._request_latency_ema_s = None
        self._latency_warmup_remaining = self._latency_warmup_samples

    def _tick_period(self) -> float:
        with self._lock:
            if self._processor is None:
                hz = self._control_hz
            else:
                hz = self._processor.output_hz
        return 1.0 / max(1.0, hz)
