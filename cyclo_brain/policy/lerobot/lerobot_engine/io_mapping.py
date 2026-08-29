#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LeRobot engine I/O mapping helpers (IoMappingMixin).

Extracted from ``engine.py`` to keep the core ``LeRobotEngine`` class
focused on the ``InferenceEngine`` API. Mixed into the engine via
multiple inheritance; bind-mounted into the policy container as part
of the ``/app/lerobot_engine/`` package.

Owns:
- ``_init_robot``: create RobotClient + resolve camera / state mappings.
- ``_teardown_robot``: release the RobotClient.
- ``_policy_image_keys``: read the policy's expected image input keys.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Dict, Iterable, List

from .constants import IMAGE_KEY_PREFIX as _IMAGE_KEY_PREFIX

from robot_client import RobotClient


logger = logging.getLogger("lerobot_engine")


_CAMERA_SEMANTIC_RE = re.compile(
    r"^cam_(?P<a>left|right|head|wrist)_(?P<b>left|right|head|wrist)$"
)


class IoMappingMixin:
    """Robot wiring — camera / state modality resolution and teardown."""

    def _init_robot(self, robot_type: str) -> None:
        """Create RobotClient + resolve camera / state mappings."""
        self._robot = RobotClient(robot_type)

        # Cameras: only those that match a policy input key
        # ``observation.images.<cam>``. Cameras advertised by the robot
        # but not consumed by the policy are silently ignored — same
        # behavior as GR00TInference.
        policy_image_keys = self._policy_image_keys()
        active = self._resolve_camera_mappings(
            self._robot.camera_names,
            policy_image_keys,
        )
        if not active and policy_image_keys:
            raise RuntimeError(
                "No cameras match the policy's expected input keys: "
                f"policy needs {sorted(policy_image_keys)}, robot has "
                f"{self._robot.camera_names}"
            )
        self._cameras = active

        # State modalities: sorted follower joint groups. We follow the
        # same convention groot uses (sorted modality names map to the
        # training-time concat order). Synthetic per-modality views (with
        # ``parent``) win over their leaf physical group; otherwise the
        # leaf group is used directly.
        groups = self._robot._config.get("joint_groups", {})
        parents = {cfg.get("parent") for cfg in groups.values() if cfg.get("parent")}
        modality_groups = []
        for name, cfg in groups.items():
            if cfg.get("role") != "follower" or not name.startswith("follower_"):
                continue
            if cfg.get("parent"):
                modality_groups.append(name)
            elif name not in parents:
                modality_groups.append(name)
        modalities = sorted(name[len("follower_"):] for name in modality_groups)
        if not modalities:
            raise RuntimeError(
                f"No follower joint groups in robot_type={robot_type}"
            )

        # Mobile is sourced from sensors["odom"] in the new schema —
        # bridge it into observation.state alongside the joint states so
        # policies trained on the legacy physical_ai_server pipeline
        # (with mobile as a 3-vector modality) still see it.
        sensors = self._robot._config.get("sensors", {})
        self._has_mobile_state = "odom" in sensors
        if self._has_mobile_state:
            modalities = sorted(set(modalities) | {"mobile"})

        self._state_modalities = modalities
        self._action_keys = self._policy_action_keys(modalities)

        # Block until at least one frame from each sensor lands. 10 s is
        # generous — typical hardware comes up in <2 s.
        self._robot.wait_for_ready(timeout=10.0)
        logger.info(
            "Robot ready: cameras=%s state_modalities=%s",
            list(self._cameras.keys()),
            self._state_modalities,
        )

    def _policy_action_keys(self, modalities: List[str]) -> List[str]:
        """Leading action groups whose widths add up to the policy's action dim.

        A policy trained on a subset of the robot's action topics (e.g. arms
        only) must not command the remaining groups; ``publish_action`` slices
        the flat action vector by these keys in order, so the prefix must sum
        exactly to what the policy emits.
        """
        try:
            action_dim = int(self._policy.config.output_features["action"].shape[0])
        except Exception:
            return list(modalities)
        keys: List[str] = []
        total = 0
        for key in modalities:
            if total >= action_dim:
                break
            keys.append(key)
            total += self._action_width(key)
        if total != action_dim:
            raise RuntimeError(
                f"Policy action dim {action_dim} does not match robot action "
                f"groups {modalities} (matched {keys} = {total})"
            )
        if keys != list(modalities):
            logger.info("Policy commands %s only (action dim %d)", keys, action_dim)
        return keys

    def _action_width(self, action_key: str) -> int:
        """Width ``RobotClient.publish_action`` consumes for one action key."""
        cfg = self._robot._action_groups.get(self._robot._resolve_action_key(action_key))
        if cfg is None:
            return 0
        if cfg.get("msg_type") == "geometry_msgs/msg/Twist":
            return 3
        return len(cfg.get("joint_names", []))

    def _teardown_robot(self) -> None:
        if self._robot is not None:
            try:
                self._robot.close()
            except Exception:
                pass
            self._robot = None

    def _policy_image_keys(self) -> set:
        """Image keys the engine must feed, expressed as camera names.

        A checkpoint fine-tuned with ``--rename_map`` keeps the base model's
        slot names (e.g. ``observation.images.camera1``) in ``input_features``
        and stores the camera-name -> slot-name map in its saved preprocessor.
        The preprocessor applies that rename at inference, so the engine must
        resolve robot cameras against the map's *source* keys.
        """
        try:
            features = getattr(self._policy.config, "input_features", {}) or {}
            keys = {k for k in features.keys() if k.startswith(_IMAGE_KEY_PREFIX)}
        except Exception:
            return set()
        slot_to_camera = {v: k for k, v in self._preprocessor_rename_map().items()}
        return {slot_to_camera.get(k, k) for k in keys}

    def _preprocessor_rename_map(self) -> Dict[str, str]:
        """Return the saved ``rename_observations_processor`` map, if any."""
        preprocessor = getattr(self, "_preprocessor", None)
        for step in getattr(preprocessor, "steps", None) or []:
            rename_map = getattr(step, "rename_map", None)
            if isinstance(rename_map, Mapping):
                return dict(rename_map)
        return {}

    @classmethod
    def _resolve_camera_mappings(
        cls,
        robot_camera_names: Iterable[str],
        policy_image_keys: set,
    ) -> Dict[str, str]:
        """Map RobotClient camera names to policy image feature keys.

        The canonical Cyclo camera names are ``cam_<side>_<part>`` such as
        ``cam_left_head``. Some runtime configs or checkpoints may expose
        ``rgb.`` prefixes. Older single-head checkpoints may use
        ``cam_head`` for the left head camera. Exact matches remain preferred.
        """
        camera_names = list(robot_camera_names)
        if not policy_image_keys:
            return {cam: f"{_IMAGE_KEY_PREFIX}{cam}" for cam in camera_names}

        active: Dict[str, str] = {}
        used_policy_keys = set()
        for cam in camera_names:
            exact = f"{_IMAGE_KEY_PREFIX}{cam}"
            candidates = cls._camera_policy_key_candidates(cam)
            matches = sorted(policy_image_keys & candidates)
            if not matches:
                continue

            if exact in matches:
                chosen = exact
            elif len(matches) == 1:
                chosen = matches[0]
            else:
                raise RuntimeError(
                    f"Ambiguous camera mapping for {cam}: matches {matches}"
                )

            if chosen in used_policy_keys:
                raise RuntimeError(
                    f"Policy camera key {chosen} matched multiple robot cameras"
                )
            active[cam] = chosen
            used_policy_keys.add(chosen)

        missing = sorted(policy_image_keys - used_policy_keys)
        if missing:
            raise RuntimeError(
                "Missing camera mappings for policy input keys: "
                f"{missing}; robot has {camera_names}; matched {active}"
            )
        return active

    @staticmethod
    def _camera_policy_key_candidates(camera_name: str) -> set:
        aliases = {camera_name}
        parts = camera_name.split(".")
        suffix = parts[-1]
        prefixes = parts[:-1]
        aliases.add(suffix)

        semantic_names = {suffix}
        if suffix == "cam_left_head":
            semantic_names.add("cam_head")
            # AI Worker datasets record the left head camera as the scene view.
            semantic_names.add("scene")

        match = _CAMERA_SEMANTIC_RE.match(suffix)
        if match:
            first = match.group("a")
            second = match.group("b")
            side = first if first in {"left", "right"} else second
            part = first if first in {"head", "wrist"} else second
            if side in {"left", "right"} and part in {"head", "wrist"}:
                semantic_names.add(f"cam_{side}_{part}")
                semantic_names.add(f"cam_{part}_{side}")
                # Datasets often drop the cam_ prefix, e.g. wrist_left.
                semantic_names.add(f"{part}_{side}")
                semantic_names.add(f"{side}_{part}")

        for name in semantic_names:
            aliases.add(name)
            aliases.add(f"rgb.{name}")
            if prefixes:
                aliases.add(".".join([*prefixes, name]))

        return {f"{_IMAGE_KEY_PREFIX}{alias}" for alias in aliases}
