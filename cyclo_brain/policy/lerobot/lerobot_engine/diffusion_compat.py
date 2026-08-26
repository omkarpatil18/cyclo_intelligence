#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
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

"""Diffusion Policy compatibility helpers for the stateless engine.

Upstream ``DiffusionPolicy`` differs from ACT-style policies in two ways
that matter for this runtime:

1. ``DiffusionConfig.validate_features`` refuses a policy whose only
   input is ``observation.state`` (it insists on an image or an
   ``observation.environment_state``), and ``DiffusionModel.compute_loss``
   asserts the same on the training batch. Proprioception-only policies
   are a legitimate use case here, so :func:`allow_state_only_diffusion`
   relaxes that single check (everything else in the validator still
   runs) and, for state-only configs, feeds ``compute_loss`` a zero-width
   ``observation.environment_state`` placeholder — the model only reads
   that key when the config declares an environment-state feature, so
   the placeholder never reaches the network. It is applied at load time
   in ``loading.py`` and by training scripts.

2. ``DiffusionPolicy.predict_action_chunk`` — the offline (empty-queue)
   path the engine uses — expects observations with an explicit
   ``n_obs_steps`` time axis, e.g. ``observation.state`` of shape
   ``(B, n_obs_steps, D)``. The engine builds one observation per
   request, ``(B, D)``. :func:`expand_obs_time_dim` inserts that axis,
   repeating the current observation ``n_obs_steps`` times — the same
   thing upstream ``select_action`` does on its first call.

The module deliberately depends only on ``torch`` and (lazily) on
``lerobot`` so it can be imported by training tooling and unit tests
without the rest of the engine / RobotClient stack.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import torch

logger = logging.getLogger("lerobot_engine")

DIFFUSION_POLICY_TYPE = "diffusion"
_STATE_ONLY_ERROR_PREFIX = "You must provide at least one image"
_PATCH_FLAG = "_cyclo_allows_state_only"


def is_diffusion_config(config: Any) -> bool:
    """True when ``config`` is an upstream ``DiffusionConfig``."""
    return getattr(config, "type", None) == DIFFUSION_POLICY_TYPE


def allow_state_only_diffusion() -> bool:
    """Let ``DiffusionConfig`` accept ``observation.state``-only inputs.

    Idempotent. Returns True when the patch is active, False when
    upstream lerobot (or its diffusion policy) is not importable.
    """
    try:
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    except Exception as exc:  # pragma: no cover - lerobot missing
        logger.debug("DiffusionConfig unavailable, skipping patch: %s", exc)
        return False

    if getattr(DiffusionConfig.validate_features, _PATCH_FLAG, False):
        return True

    _patch_compute_loss()
    original = DiffusionConfig.validate_features

    def validate_features(self) -> None:  # type: ignore[no-untyped-def]
        try:
            original(self)
        except ValueError as exc:
            state_only = (
                str(exc).startswith(_STATE_ONLY_ERROR_PREFIX)
                and self.robot_state_feature is not None
            )
            if not state_only:
                raise
            logger.info(
                "Diffusion policy conditions on observation.state only "
                "(no image / environment_state inputs)"
            )

    setattr(validate_features, _PATCH_FLAG, True)
    DiffusionConfig.validate_features = validate_features
    return True


def _patch_compute_loss() -> None:
    """Satisfy ``compute_loss``'s image-or-env-state assertion for state-only configs."""
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionModel
    from lerobot.utils.constants import OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

    if getattr(DiffusionModel.compute_loss, _PATCH_FLAG, False):
        return
    original = DiffusionModel.compute_loss

    def compute_loss(self, batch):  # type: ignore[no-untyped-def]
        state_only = (
            not self.config.image_features
            and self.config.env_state_feature is None
            and OBS_IMAGES not in batch
            and OBS_ENV_STATE not in batch
        )
        if state_only:
            batch = dict(batch)
            state = batch[OBS_STATE]
            batch[OBS_ENV_STATE] = state.new_zeros(*state.shape[:-1], 0)
        return original(self, batch)

    setattr(compute_loss, _PATCH_FLAG, True)
    DiffusionModel.compute_loss = compute_loss


def expand_obs_time_dim(
    batch: Dict[str, Any], n_obs_steps: int
) -> Dict[str, Any]:
    """Insert the ``n_obs_steps`` axis expected by ``predict_action_chunk``.

    ``observation.state`` / ``observation.environment_state`` tensors of
    shape ``(B, D)`` become ``(B, n_obs_steps, D)``; image tensors
    ``observation.images.*`` of shape ``(B, C, H, W)`` become
    ``(B, n_obs_steps, C, H, W)``. Tensors that already carry the time
    axis, and non-observation keys, are passed through untouched.
    """
    n_obs_steps = max(1, int(n_obs_steps))
    out: Dict[str, Any] = dict(batch)
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor) or not key.startswith("observation."):
            continue
        is_image = key.startswith("observation.images.")
        batched_ndim = 4 if is_image else 2
        if value.dim() != batched_ndim:
            continue
        expanded = value.unsqueeze(1)
        if n_obs_steps > 1:
            expanded = expanded.expand(
                value.shape[0], n_obs_steps, *value.shape[1:]
            ).contiguous()
        out[key] = expanded
    return out
