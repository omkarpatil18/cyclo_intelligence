#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0

"""LeRobot prediction helpers."""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
import torch

import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla  # noise_idx global


logger = logging.getLogger("lerobot_engine")

# Lottery-ticket noise selection (https://arxiv.org/abs/2603.15757), switchable at
# runtime without restarting the engine. The file is re-read only when its mtime changes.
#   file missing        -> modeling_smolvla.noise_idx is left untouched (source default governs)
#   empty / "none"      -> random noise every call (default SmolVLA behaviour)
#   integer k           -> fixed noise ticket k (seeded with k, shared across batch)
# Host path: docker/workspace/noise_idx.txt  ->  e.g.  echo 2 > docker/workspace/noise_idx.txt
# (Same file as the GR00T engine; only affects SmolVLA policies.)
NOISE_IDX_FILE = os.environ.get("LEROBOT_NOISE_IDX_FILE", "/workspace/noise_idx.txt")


class PredictionMixin:
    """Policy input batch -> action chunk."""

    def _sync_noise_idx(self) -> None:
        """Apply NOISE_IDX_FILE to modeling_smolvla.noise_idx if the file changed (~2us via stat)."""
        try:
            mtime = os.stat(NOISE_IDX_FILE).st_mtime_ns
        except FileNotFoundError:
            return  # no override file: leave modeling_smolvla.noise_idx as-is
        if mtime == getattr(self, "_noise_idx_mtime", None):
            return
        self._noise_idx_mtime = mtime

        try:
            with open(NOISE_IDX_FILE) as f:
                text = f.read().strip()
            value = None if (not text or text.lower() == "none") else int(text)
        except (OSError, ValueError) as e:
            logger.warning(
                "Ignoring %s (%s); keeping noise_idx=%s",
                NOISE_IDX_FILE,
                e,
                modeling_smolvla.noise_idx,
            )
            return

        if value != modeling_smolvla.noise_idx:
            modeling_smolvla.noise_idx = value
            logger.info("SmolVLA noise_idx set to %s (from %s)", value, NOISE_IDX_FILE)

    def _predict_chunk(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return a chunk tensor of shape (1, T, A)."""
        assert self._policy is not None
        # Lottery-ticket noise sweep: uncomment to follow /workspace/noise_idx.txt live.
        # self._sync_noise_idx()
        try:
            action = self._policy.predict_action_chunk(batch)
            if action.dim() == 2:
                action = action.unsqueeze(1)
            return action
        except (NotImplementedError, AttributeError):
            logger.debug(
                "predict_action_chunk unavailable; falling back to select_action"
            )
            action = self._policy.select_action(batch)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            return action.unsqueeze(1)

    @staticmethod
    def _to_numpy_chunk(action: torch.Tensor) -> np.ndarray:
        """(B, T, A) or (B, A) tensor -> (T, A) float64 numpy."""
        chunk = action.detach().cpu()
        if chunk.dim() == 3:
            chunk = chunk[0]
        elif chunk.dim() == 2:
            pass
        elif chunk.dim() == 1:
            chunk = chunk.unsqueeze(0)
        else:
            raise ValueError(
                f"Unexpected action tensor shape: {tuple(chunk.shape)}"
            )
        return chunk.to(torch.float64).numpy()
