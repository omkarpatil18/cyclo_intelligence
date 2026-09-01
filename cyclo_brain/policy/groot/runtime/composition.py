#!/usr/bin/env python3
"""Output (stitch) composition for GR00T policies.

A composite "model dir" contains ``composite.json``::

    {"models": ["<dirA>", "<dirB>"], "weights": [[...16 floats...], ...],
     "mode": "stitch"}

Each policy runs its NATIVE inference on the same raw observation
(``Gr00tPolicy.get_action`` returns UNNORMALIZED physical actions via its own
saved processor), and the finished chunks are combined per action dimension:
``out[d] = sum_i w_i[d] * a_i[d]``. Weight vectors follow the flat action
layout (concatenation of the policy's action modality keys, e.g. arm_left(8)
+ arm_right(8)); scalars broadcast over all dims.

Joint (score/velocity) composition is NOT implemented for GR00T yet — it
requires refactoring the flow-matching loop in gr00t_n1d7.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("groot_inference")


def load_composite_spec(model_path: str) -> Optional[Dict[str, Any]]:
    """Return {'models': [abs paths], 'weights': [...], 'mode': str} if
    model_path is a composite dir (contains composite.json), else None."""
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
    parsed: List[Any] = []
    for w in weights:
        if isinstance(w, (list, tuple)):
            parsed.append([float(x) for x in w])
        else:
            parsed.append(float(w))
    mode = str(spec.get("mode", "stitch")).lower()
    return {"models": resolved, "weights": parsed, "mode": mode}


def combine_actions(
    action_dicts: List[Dict[str, np.ndarray]],
    weights: List[Any],
    action_keys: List[str],
) -> Dict[str, np.ndarray]:
    """Per-dim weighted sum of physical action dicts from N policies."""
    keys = [
        k for k in action_keys
        if isinstance(action_dicts[0].get(k), np.ndarray)
    ]
    widths = [int(action_dicts[0][k].shape[-1]) for k in keys]
    total = int(sum(widths))

    vecs = []
    for w in weights:
        if isinstance(w, (int, float)):
            vecs.append(np.full(total, float(w), dtype=np.float32))
        else:
            v = np.asarray(w, dtype=np.float32).flatten()
            if v.size != total:
                raise ValueError(
                    f"per-dim weights must have length {total} "
                    f"(action layout {list(zip(keys, widths))}), got {v.size}"
                )
            vecs.append(v)
    colsum = np.sum(vecs, axis=0)
    if not np.allclose(colsum, 1.0, atol=1e-3):
        logger.warning(
            "Composite weights do not sum to 1 per dim (min %.3f max %.3f) — using as given",
            float(colsum.min()), float(colsum.max()),
        )

    out: Dict[str, np.ndarray] = {}
    off = 0
    for k, wdt in zip(keys, widths):
        acc = None
        for a, v in zip(action_dicts, vecs):
            part = v[off:off + wdt] * a[k]
            acc = part if acc is None else acc + part
        out[k] = acc.astype(action_dicts[0][k].dtype, copy=False)
        off += wdt
    return out


# --------------------------------------------------------------------------- #
# Joint (score / velocity) composition
# --------------------------------------------------------------------------- #
import torch  # noqa: E402

import gr00t.model.gr00t_n1d7.gr00t_n1d7 as _gr00t_n1d7  # noqa: E402  (noise ticket)


class _Captured(Exception):
    """Sentinel: stop Gr00tPolicy.get_action right after input collation."""


def _capture_collated_inputs(policy, observation) -> dict:
    """Run the policy's native preprocessing (processor + collate + dtype cast)
    and capture the kwargs that would be passed to model.get_action."""
    captured = []
    original = policy.model.get_action

    def recorder(*args, **kwargs):
        captured.append((args, kwargs))
        raise _Captured

    policy.model.get_action = recorder
    try:
        policy.get_action(observation)
    except _Captured:
        pass
    finally:
        policy.model.get_action = original
    if not captured:
        raise RuntimeError("failed to capture GR00T model inputs")
    args, kwargs = captured[0]
    # Gr00tPolicy calls self.model.get_action(**collated) where the collator
    # wraps everything as {"inputs": <flat dict>} (see Gr00tN1d7DataCollator:
    # ``return BatchFeature(data={"inputs": batch})``), so the flat model-input
    # dict arrives as the ``inputs`` kwarg. Unwrap exactly the way the call
    # binds to ``Gr00tN1d7.get_action(self, inputs, options=None)``.
    if "inputs" in kwargs:
        return kwargs["inputs"]
    if args:
        return args[0]
    return kwargs


def _action_affine(policy, pad_dim: int):
    """Per-dim (scale, shift) mapping normalized [-1,1] -> physical, from the
    policy's q01/q99 (or min/max) action stats. Padded dims are identity."""
    tag = policy.embodiment_tag.value
    sap = policy.processor.state_action_processor
    groups = sap.modality_configs[tag]["action"].modality_keys
    scale = np.ones(pad_dim, dtype=np.float32)
    shift = np.zeros(pad_dim, dtype=np.float32)
    off = 0
    for g in groups:
        params = sap.norm_params[tag]["action"][g]
        lo = np.asarray(params["min"], dtype=np.float32).flatten()
        hi = np.asarray(params["max"], dtype=np.float32).flatten()
        d = lo.size
        scale[off:off + d] = np.maximum((hi - lo) / 2.0, 1e-8)
        shift[off:off + d] = (hi + lo) / 2.0
        off += d
    return scale, shift, off  # off == real action dim


class GrootScoreComposer:
    """Score (velocity) composition across N Gr00tPolicy instances.

    Shared trajectory kept in PHYSICAL action space; per policy the view is
    converted with its own min-max affine (x_norm = (x_phys - shift)/scale),
    velocities converted back (v_phys = v_norm * scale) and combined per-dim:
    v = sum_i w_i[d] * v_phys_i[d]. Initial noise uses the ownership-weighted
    prior. Views are clamped to +/-3 in normalized space (decode clips to
    [-1,1] at the end regardless).
    """

    VIEW_CLAMP = 3.0

    def __init__(self, policies: List[Any], weights: List[Any]):
        if len(policies) < 2:
            raise ValueError("score composition needs >= 2 policies")
        head0 = policies[0].model.action_head
        for p in policies[1:]:
            h = p.model.action_head
            if (h.action_dim, h.action_horizon, h.num_inference_timesteps) != (
                head0.action_dim, head0.action_horizon, head0.num_inference_timesteps
            ):
                raise ValueError("composite policies must share action head dims")
        self.policies = policies
        self.pad_dim = int(head0.action_dim)
        self.horizon = int(head0.config.action_horizon)
        self.num_steps = int(head0.num_inference_timesteps)

        affines = [_action_affine(p, self.pad_dim) for p in policies]
        real_dim = affines[0][2]
        self.scales = [a[0] for a in affines]
        self.shifts = [a[1] for a in affines]

        n = len(policies)
        vecs = []
        for w in weights:
            if isinstance(w, (int, float)):
                vecs.append(np.full(self.pad_dim, float(w), dtype=np.float32))
            else:
                v = np.asarray(w, dtype=np.float32).flatten()
                if v.size != real_dim:
                    raise ValueError(
                        f"per-dim weights must have length {real_dim}, got {v.size}"
                    )
                vec = np.full(self.pad_dim, 1.0 / n, dtype=np.float32)
                vec[:real_dim] = v
                vecs.append(vec)
        self.w = vecs
        # Ownership-weighted initial prior (per-dim; zero columns -> equal).
        wstack = np.stack(vecs)
        colsum = np.clip(np.abs(wstack).sum(axis=0), 1e-6, None)
        share = np.abs(wstack) / colsum
        self.init_scale = sum(s * sc for s, sc in zip(share, self.scales))
        self.init_shift = sum(s * sh for s, sh in zip(share, self.shifts))
        logger.info(
            "GrootScoreComposer: %d policies, steps=%d, horizon=%d, pad_dim=%d, "
            "real_dim=%d", n, self.num_steps, self.horizon, self.pad_dim, real_dim,
        )

    @torch.no_grad()
    def get_action(self, observation) -> Dict[str, np.ndarray]:
        # 1) Native preprocessing per policy, then conditioning (backbone runs once).
        conds = []
        device = None
        dtype = None
        for policy in self.policies:
            inputs = _capture_collated_inputs(policy, observation)
            model = policy.model
            backbone_inputs, action_inputs = model.prepare_input(inputs)
            backbone_outputs = model.backbone(backbone_inputs)
            feats = model.action_head._encode_features(backbone_outputs, action_inputs)
            device = feats.backbone_features.device
            dtype = feats.backbone_features.dtype
            conds.append(
                (model.action_head, feats, action_inputs.embodiment_id, backbone_outputs)
            )

        # 2) Initial noise (honors the lottery-ticket noise_idx), lifted to
        #    physical space with the ownership-weighted prior.
        shape = (1, self.horizon, self.pad_dim)
        if _gr00t_n1d7.noise_idx is None:
            noise = torch.randn(shape, dtype=torch.float32, device=device)
        else:
            g = torch.Generator(device=device).manual_seed(int(_gr00t_n1d7.noise_idx))
            noise = torch.randn(shape, dtype=torch.float32, device=device, generator=g)
        t_scales = [torch.as_tensor(s, device=device) for s in self.scales]
        t_shifts = [torch.as_tensor(s, device=device) for s in self.shifts]
        t_w = [torch.as_tensor(w, device=device) for w in self.w]
        init_scale = torch.as_tensor(self.init_scale, device=device)
        init_shift = torch.as_tensor(self.init_shift, device=device)
        x_phys = noise * init_scale + init_shift

        # 3) Shared Euler loop, weighted velocity combination in physical space.
        dt = 1.0 / self.num_steps
        for t_index in range(self.num_steps):
            v_phys = torch.zeros_like(x_phys)
            for (head, feats, emb_id, bb_out), sc, sh, wv in zip(
                conds, t_scales, t_shifts, t_w
            ):
                x_view = torch.clamp(
                    (x_phys - sh) / sc, -self.VIEW_CLAMP, self.VIEW_CLAMP
                ).to(dtype)
                v = head.velocity_step(
                    actions=x_view,
                    t_index=t_index,
                    backbone_features=feats.backbone_features,
                    state_features=feats.state_features,
                    embodiment_id=emb_id,
                    backbone_output=bb_out,
                )
                v_phys = v_phys + wv * (v.float() * sc)
            x_phys = x_phys + dt * v_phys

        # 4) Back to the PRIMARY policy's normalized space; decode with its
        #    saved processor (clips to [-1,1], splits per group, unnormalizes).
        x0_norm0 = ((x_phys - t_shifts[0]) / t_scales[0]).float().cpu().numpy()
        p0 = self.policies[0]
        unnorm = p0.processor.decode_action(x0_norm0, p0.embodiment_tag, None)
        return {k: v.astype(np.float32) for k, v in unnorm.items()}
