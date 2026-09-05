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

Joint (score/velocity) composition IS implemented as well: see
``GrootScoreComposer`` below (shared trajectory in the normalized action
space, per-step velocity combination via Gr00tN1d7ActionHead.velocity_step).
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
    and capture the flat model-input dict passed into model.get_action."""
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
    # wraps everything as {"inputs": <flat dict>} — unwrap exactly the way the
    # call binds to ``Gr00tN1d7.get_action(self, inputs, options=None)``.
    if "inputs" in kwargs:
        return kwargs["inputs"]
    if args:
        return args[0]
    return kwargs


class GrootScoreComposer:
    """Score (velocity) composition across N Gr00tPolicy instances.

    NAIVE composition: one shared trajectory in the normalized action space
    (policies are assumed to share normalization — for checkpoints with
    differing stats this is knowingly inconsistent and accepted). Plain
    N(0,1) initial noise (honors the lottery-ticket noise_idx); per step
    v = sum_i w_i[d] * v_i, x <- x + dt * v; the finished chunk is decoded
    with the PRIMARY policy's saved processor.
    """

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

        # Per-dim weight vectors (scalars broadcast; vectors cover the real
        # action dim, padded dims get 1/N).
        tag = policies[0].embodiment_tag.value
        sap = policies[0].processor.state_action_processor
        groups = sap.modality_configs[tag]["action"].modality_keys
        real_dim = int(sum(
            sap.norm_params[tag]["action"][g]["dim"].item() for g in groups
        ))
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
        colsum = np.sum([v[:real_dim] for v in vecs], axis=0)
        if not np.allclose(colsum, 1.0, atol=1e-3):
            logger.warning(
                "Composite weights do not sum to 1 per dim (min %.3f max %.3f) — using as given",
                float(colsum.min()), float(colsum.max()),
            )
        self.w = vecs
        logger.info(
            "GrootScoreComposer (naive): %d policies, steps=%d, horizon=%d, "
            "pad_dim=%d, real_dim=%d", n, self.num_steps, self.horizon,
            self.pad_dim, real_dim,
        )

    @torch.no_grad()
    def get_action(self, observation) -> Dict[str, np.ndarray]:
        # 1) Native preprocessing + conditioning per policy (backbone once each).
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

        # 2) Plain N(0,1) initial noise in the shared normalized space
        #    (same lottery-ticket semantics as single-policy inference).
        # Same dtype conventions as get_action_with_features (backbone dtype).
        shape = (1, self.horizon, self.pad_dim)
        if _gr00t_n1d7.noise_idx is None:
            x_t = torch.randn(shape, dtype=dtype, device=device)
        else:
            g = torch.Generator(device=device).manual_seed(int(_gr00t_n1d7.noise_idx))
            x_t = torch.randn(shape, dtype=dtype, device=device, generator=g)
        t_w = [torch.as_tensor(w, device=device, dtype=dtype) for w in self.w]

        # 3) Shared Euler loop: v = sum_i w_i * v_i.
        dt = 1.0 / self.num_steps
        for t_index in range(self.num_steps):
            v = torch.zeros_like(x_t)
            for (head, feats, emb_id, bb_out), wv in zip(conds, t_w):
                v_i = head.velocity_step(
                    actions=x_t,
                    t_index=t_index,
                    backbone_features=feats.backbone_features,
                    state_features=feats.state_features,
                    embodiment_id=emb_id,
                    backbone_output=bb_out,
                )
                v = v + wv * v_i
            x_t = x_t + dt * v

        # 4) Decode with the PRIMARY policy's saved processor.
        p0 = self.policies[0]
        unnorm = p0.processor.decode_action(
            x_t.float().cpu().numpy(), p0.embodiment_tag, None
        )
        return {k: v.astype(np.float32) for k, v in unnorm.items()}
