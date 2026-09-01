#!/usr/bin/env python3
"""Compositional inference for two (or more) SmolVLA flow-matching policies.

At every Euler denoising step each policy predicts a velocity field v_i for the
shared noisy action chunk; the chunk is updated with the weighted combination
    v = sum_i w_i * v_i          (cf. comp_fsl CompositionalSampler, which sums
                                  weighted per-model means the same way)
    x <- x + dt * v

Because each policy normalizes actions with its own MEAN_STD stats, the shared
state x is kept in PHYSICAL action space; per policy we convert
    x_norm_i = (x_phys - mean_i) / std_i        (model input)
    v_phys_i = v_norm_i * std_i                 (model output)
which is exact for the affine normalizer and reduces to plain averaging when
the stats coincide. Padded action dims (beyond the real action dim) use
mean=0/std=1.

A composite "model dir" is a directory containing ``composite.json``::

    {"models": ["<dirA>", "<dirB>"], "weights": [0.5, 0.5]}

Model entries may be absolute paths or names resolved against the composite
dir's parent. The FIRST model is primary: its preprocessor drives image
resize hints, its postprocessor un-normalizes the final chunk, and its config
is what the engine reports.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import lerobot.policies.smolvla.modeling_smolvla as modeling_smolvla

logger = logging.getLogger("lerobot_engine")


def zero_state_flag(model_dir: str) -> bool:
    """True when this checkpoint was trained with zeroed proprioception and
    must be fed zeros for observation.state at inference. Detected via a
    ``noproprio.flag`` marker file or a ``noproprio`` dir-name suffix."""
    d = Path(model_dir)
    return (d / "noproprio.flag").exists() or "noproprio" in d.name.lower()


def zero_state(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Replace state tensors with zeros (post-preprocessing)."""
    for k, v in list(batch.items()):
        if k.endswith("observation.state") and torch.is_tensor(v):
            batch[k] = torch.zeros_like(v)
    return batch


class _Captured(Exception):
    """Sentinel used to stop predict_action_chunk after input preparation."""


def _capture_model_inputs(policy, batch) -> Tuple:
    """Run policy.predict_action_chunk just far enough to capture the fully
    prepared (images, img_masks, lang_tokens, lang_masks, state) tensors,
    without running the sampling loop. Keeps us exactly in sync with the
    policy's own queue/prepare logic."""
    captured: List[Tuple] = []
    original = policy.model.sample_actions

    def recorder(images, img_masks, lang_tokens, lang_masks, state, **kw):
        captured.append((images, img_masks, lang_tokens, lang_masks, state))
        raise _Captured

    policy.model.sample_actions = recorder
    try:
        policy.predict_action_chunk(batch)
    except _Captured:
        pass
    finally:
        policy.model.sample_actions = original
    if not captured:
        raise RuntimeError("Failed to capture SmolVLA model inputs")
    return captured[0]


def _action_stats(postprocessor, action_dim_padded: int, real_dim: int, device):
    """(mean, std) tensors of length ``action_dim_padded`` from the policy's
    saved unnormalizer step; padded dims get mean=0/std=1."""
    mean = torch.zeros(action_dim_padded, device=device, dtype=torch.float32)
    std = torch.ones(action_dim_padded, device=device, dtype=torch.float32)
    for step in getattr(postprocessor, "steps", []) or []:
        stats = getattr(step, "stats", None) or {}
        a = stats.get("action") or {}
        m, s = a.get("mean"), a.get("std")
        if m is None or s is None:
            continue
        m = torch.as_tensor(m, device=device, dtype=torch.float32).flatten()
        s = torch.as_tensor(s, device=device, dtype=torch.float32).flatten()
        n = min(real_dim, m.numel())
        mean[:n] = m[:n]
        std[:n] = torch.clamp(s[:n], min=1e-8)
        return mean, std
    logger.warning("Composite: no action MEAN_STD stats found; assuming identity")
    return mean, std


class SmolVLAComposer:
    """Joint flow-matching sampler over N SmolVLA policies."""

    def __init__(
        self,
        policies: List[Any],
        postprocessors: List[Any],
        weights: List[float],
        mode: str = "joint",
    ):
        if mode not in ("joint", "stitch"):
            raise ValueError(f"composite mode must be 'joint' or 'stitch', got {mode!r}")
        self.mode = mode
        if len(policies) < 2:
            raise ValueError("Composite needs at least 2 policies")
        if len(weights) != len(policies):
            raise ValueError("weights must match number of models")
        cfg0 = policies[0].config
        for p in policies[1:]:
            c = p.config
            if (c.chunk_size, c.max_action_dim, c.num_steps) != (
                cfg0.chunk_size, cfg0.max_action_dim, cfg0.num_steps
            ):
                raise ValueError(
                    "Composite policies must share chunk_size/max_action_dim/num_steps"
                )
        self.policies = policies
        # Each entry: a scalar, or a per-action-dim vector (len == real action
        # dim). Per-dim vectors enable "stitching" compositions, e.g. left-arm
        # dims from the left policy and right-arm dims from the right policy.
        self.weights = weights
        self.postprocessors = postprocessors
        self._stats: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._w: Optional[List[torch.Tensor]] = None
        logger.info("SmolVLAComposer: mode=%s, %d policies, weights=%s", mode, len(policies), weights)

    def _ensure_stats(self, device):
        if self._stats is not None:
            return
        cfg0 = self.policies[0].config
        real_dim = cfg0.action_feature.shape[0]
        self._stats = [
            _action_stats(pp, cfg0.max_action_dim, real_dim, device)
            for pp in self.postprocessors
        ]
        # Weight tensors, shape (max_action_dim,), broadcast over (B, T, D).
        # Scalars fill all dims; vectors must have length == real action dim,
        # padded dims get 1/N so the (discarded) padding still denoises sanely.
        n = len(self.policies)
        ws: List[torch.Tensor] = []
        for w in self.weights:
            if isinstance(w, (int, float)):
                vec = torch.full(
                    (cfg0.max_action_dim,), float(w), device=device, dtype=torch.float32
                )
            else:
                v = torch.as_tensor(
                    [float(x) for x in w], device=device, dtype=torch.float32
                ).flatten()
                if v.numel() != real_dim:
                    raise ValueError(
                        f"per-dim weights must have length {real_dim}, got {v.numel()}"
                    )
                vec = torch.full(
                    (cfg0.max_action_dim,), 1.0 / n, device=device, dtype=torch.float32
                )
                vec[:real_dim] = v
            ws.append(vec)
        total = torch.stack(ws).sum(dim=0)[:real_dim]
        if not torch.allclose(total, torch.ones_like(total), atol=1e-3):
            logger.warning(
                "Composite weights do not sum to 1 per dim (min %.3f max %.3f) — using as given",
                float(total.min()), float(total.max()),
            )
        self._w = ws
        # Joint-mode initial prior: per-dim mean/std taken from the policy that
        # OWNS each dim (weights normalized per dim; zero columns fall back to
        # equal shares). Fixes the collapsed/shifted starting noise when the
        # policies' normalizer stats differ.
        wstack = torch.stack(ws)                       # (N, D)
        colsum = wstack.sum(dim=0)
        share = torch.where(
            colsum.abs() > 1e-6,
            wstack / torch.clamp(colsum.abs(), min=1e-6),
            torch.full_like(wstack, 1.0 / n),
        )
        means = torch.stack([m for m, _ in self._stats])   # (N, D)
        stds = torch.stack([sd for _, sd in self._stats])  # (N, D)
        self._init_mean = (share * means).sum(dim=0)
        self._init_std = torch.clamp((share * stds).sum(dim=0), min=1e-8)

    @torch.no_grad()
    def predict_action_chunk(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """batches[i] = observation preprocessed by policy i's own preprocessor.
        Returns actions in the PRIMARY policy's normalized space, sliced to the
        real action dim — i.e. exactly what policy0.predict_action_chunk would
        return, ready for policy0's postprocessor."""
        assert len(batches) == len(self.policies)
        if self.mode == "stitch":
            return self._predict_stitch(batches)
        cfg0 = self.policies[0].config

        # 1) Per-policy input prep + prefix KV cache (vision/language runs once).
        prefixes = []
        device = None
        for policy, batch in zip(self.policies, batches):
            policy.eval()
            images, img_masks, lang_tokens, lang_masks, state = _capture_model_inputs(
                policy, batch
            )
            device = state.device
            model = policy.model
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            att_2d = modeling_smolvla.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=att_2d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )
            prefixes.append((model, prefix_pad_masks, past_key_values))

        self._ensure_stats(device)
        bsize = 1
        shape = (bsize, cfg0.chunk_size, cfg0.max_action_dim)

        # 2) Initial noise (honors the lottery-ticket noise_idx if set),
        #    lifted into physical space via the primary policy's stats.
        if modeling_smolvla.noise_idx is None:
            noise = torch.randn(shape, dtype=torch.float32, device=device)
        else:
            g = torch.Generator(device=device).manual_seed(int(modeling_smolvla.noise_idx))
            noise = torch.randn(shape, dtype=torch.float32, device=device, generator=g)
        x_phys = noise * self._init_std + self._init_mean

        # 3) Shared Euler loop with weighted velocity composition (physical space).
        num_steps = cfg0.num_steps
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(bsize)
            v_phys = torch.zeros_like(x_phys)
            for w_vec, (model, prefix_pad_masks, past_kv), (mean_i, std_i) in zip(
                self._w, prefixes, self._stats
            ):
                # Clamp the per-policy normalized VIEW so cross-dataset mean/std
                # offsets cannot push conditioning absurdly out of distribution.
                x_norm_i = torch.clamp((x_phys - mean_i) / std_i, -5.0, 5.0)
                v_i = model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_kv,
                    x_t=x_norm_i,
                    timestep=t_tensor,
                )
                v_phys = v_phys + w_vec * (v_i * std_i)
            x_phys = x_phys + dt * v_phys

        # 4) Back to primary-normalized space; slice to the real action dim.
        mean0, std0 = self._stats[0]
        x0_norm = (x_phys - mean0) / std0
        real_dim = cfg0.action_feature.shape[0]
        return x0_norm[:, :, :real_dim]

    @torch.no_grad()
    def _predict_stitch(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Sanity-check mode: each policy denoises its OWN trajectory fully in
        its own normalized space (100% in-distribution — the native inference
        path, including the noise_idx ticket), and only the FINISHED chunks are
        combined per-dim in physical space. For 0/1 block weights this is exact
        two-specialist stitching; it gives up the joint co-denoising coupling."""
        device = next(self.policies[0].parameters()).device
        self._ensure_stats(device)
        combined = None
        real_dim = self.policies[0].config.action_feature.shape[0]
        for policy, batch, (mean_i, std_i), w_vec in zip(
            self.policies, batches, self._stats, self._w
        ):
            a = policy.predict_action_chunk(batch)          # (1, T, real) in i-space
            a_phys = a * std_i[:real_dim] + mean_i[:real_dim]
            part = w_vec[:real_dim] * a_phys
            combined = part if combined is None else combined + part
        mean0, std0 = self._stats[0]
        return (combined - mean0[:real_dim]) / std0[:real_dim]


def load_composite_spec(model_path: str) -> Optional[Dict[str, Any]]:
    """Return {'models': [abs paths], 'weights': [floats]} if model_path is a
    composite dir (contains composite.json), else None."""
    root = Path(model_path)
    spec_file = root / "composite.json"
    if not spec_file.exists():
        return None
    spec = json.loads(spec_file.read_text())
    models = spec.get("models") or []
    if len(models) < 2:
        raise ValueError("composite.json needs >= 2 entries in 'models'")
    resolved = []
    for m in models:
        p = Path(m)
        if not p.is_absolute():
            p = root.parent / m
        if not (p / "config.json").exists():
            raise FileNotFoundError(f"composite model dir has no config.json: {p}")
        resolved.append(str(p))
    weights = spec.get("weights") or [1.0 / len(resolved)] * len(resolved)
    if len(weights) != len(resolved):
        raise ValueError("composite.json: len(weights) != len(models)")
    parsed = []
    for w in weights:
        if isinstance(w, (list, tuple)):
            parsed.append([float(x) for x in w])
        else:
            parsed.append(float(w))
    mode = str(spec.get("mode", "joint")).lower()
    return {"models": resolved, "weights": parsed, "mode": mode}
