#!/usr/bin/env python3
"""Compositional inference for two (or more) LeRobot policies of the same
family — SmolVLA (flow matching) or DiffusionPolicy (DDPM/DDIM).

NAIVE composition in the shared normalized action space (all policies are
assumed to use the same normalizer — e.g. *sharednorm* checkpoints whose
stats were computed across datasets; for policies with differing stats this
is knowingly inconsistent and that is accepted):

  joint  ("score"):  the per-step model outputs (SmolVLA: velocity v_t;
                     diffusion: predicted noise eps) are combined as
                     sum_i w_i * out_i on ONE shared trajectory;
  stitch ("output"): each policy denoises its own trajectory natively and the
                     finished chunks are combined per-dim.

Weights per model: scalar, or per-action-dim vector (len == real action dim);
padded dims (SmolVLA only) get 1/N. A composite "model dir" contains
``composite.json``::

    {"models": ["<dirA>", "<dirB>"], "weights": [...], "mode": "joint"}

The FIRST model is primary: its preprocessor drives image resize hints and
its postprocessor un-normalizes the final chunk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def sample_noise(shape, device, dtype=torch.float32) -> torch.Tensor:
    """Initial noise for the shared trajectory, honoring the noise_idx
    lottery ticket (module global in modeling_smolvla, hot-reloaded from
    /workspace/noise_idx.txt)."""
    if modeling_smolvla.noise_idx is None:
        return torch.randn(shape, dtype=dtype, device=device)
    g = torch.Generator(device=device).manual_seed(int(modeling_smolvla.noise_idx))
    return torch.randn(shape, dtype=dtype, device=device, generator=g)


class _Captured(Exception):
    """Sentinel used to stop predict_action_chunk after input preparation."""


def _capture_via(policy, attr_owner, attr_name, batch, keep):
    """Run policy.predict_action_chunk with attr_owner.attr_name replaced by a
    recorder, so the policy's own preprocessing runs verbatim but sampling is
    skipped. ``keep(*args, **kwargs)`` picks what to capture."""
    captured = []
    original = getattr(attr_owner, attr_name)

    def recorder(*args, **kwargs):
        captured.append(keep(*args, **kwargs))
        raise _Captured

    setattr(attr_owner, attr_name, recorder)
    try:
        policy.predict_action_chunk(batch)
    except _Captured:
        pass
    finally:
        setattr(attr_owner, attr_name, original)
    if not captured:
        raise RuntimeError(f"Failed to capture inputs via {attr_name}")
    return captured[0]


class _ComposerBase:
    """Shared plumbing: weight vectors and the stitch (output-composition)
    baseline. Subclasses implement the family-specific joint sampler."""

    def __init__(
        self,
        policies: List[Any],
        postprocessors: List[Any],  # kept for API compatibility; unused
        weights: List[Any],
        mode: str = "joint",
    ):
        if mode not in ("joint", "stitch"):
            raise ValueError(f"composite mode must be 'joint' or 'stitch', got {mode!r}")
        if len(policies) < 2:
            raise ValueError("Composite needs at least 2 policies")
        if len(weights) != len(policies):
            raise ValueError("weights must match number of models")
        self.mode = mode
        self.policies = policies
        self.weights = weights
        self._w: Optional[List[torch.Tensor]] = None
        self._check_compatible()
        logger.info(
            "%s: mode=%s, %d policies, weights=%s",
            type(self).__name__, mode, len(policies), weights,
        )

    # -- family-specific hooks -------------------------------------------
    def _check_compatible(self) -> None:
        raise NotImplementedError

    def _padded_dim(self) -> int:
        """Width of the sampled trajectory (>= real action dim)."""
        return self._real_dim()

    def _predict_joint(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        raise NotImplementedError

    # -- shared ----------------------------------------------------------
    def _real_dim(self) -> int:
        return self.policies[0].config.action_feature.shape[0]

    def _ensure_weights(self, device):
        if self._w is not None:
            return
        real_dim = self._real_dim()
        padded_dim = self._padded_dim()
        n = len(self.policies)
        ws: List[torch.Tensor] = []
        for w in self.weights:
            if isinstance(w, (int, float)):
                vec = torch.full((padded_dim,), float(w), device=device, dtype=torch.float32)
            else:
                v = torch.as_tensor(
                    [float(x) for x in w], device=device, dtype=torch.float32
                ).flatten()
                if v.numel() != real_dim:
                    raise ValueError(
                        f"per-dim weights must have length {real_dim}, got {v.numel()}"
                    )
                vec = torch.full((padded_dim,), 1.0 / n, device=device, dtype=torch.float32)
                vec[:real_dim] = v
            ws.append(vec)
        total = torch.stack(ws).sum(dim=0)[:real_dim]
        if not torch.allclose(total, torch.ones_like(total), atol=1e-3):
            logger.warning(
                "Composite weights do not sum to 1 per dim (min %.3f max %.3f) — using as given",
                float(total.min()), float(total.max()),
            )
        self._w = ws

    @torch.no_grad()
    def predict_action_chunk(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """batches[i] = observation preprocessed by policy i's own preprocessor.
        Returns actions in the shared normalized space, sliced to the real
        action dim — ready for the primary policy's postprocessor."""
        assert len(batches) == len(self.policies)
        for policy in self.policies:
            policy.eval()
        if self.mode == "stitch":
            return self._predict_stitch(batches)
        return self._predict_joint(batches)

    @torch.no_grad()
    def _predict_stitch(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Output-composition baseline: each policy denoises its OWN trajectory
        natively (incl. the noise_idx ticket); finished chunks are combined
        per-dim in the shared normalized space."""
        device = next(self.policies[0].parameters()).device
        self._ensure_weights(device)
        real_dim = self._real_dim()
        combined = None
        for policy, batch, w_vec in zip(self.policies, batches, self._w):
            a = policy.predict_action_chunk(batch)  # (1, T, real_dim)
            part = w_vec[:real_dim] * a
            combined = part if combined is None else combined + part
        return combined


class SmolVLAComposer(_ComposerBase):
    """Joint flow-matching sampler over N SmolVLA policies: one shared
    trajectory, v = sum_i w_i * v_i at every Euler step."""

    def _check_compatible(self) -> None:
        cfg0 = self.policies[0].config
        for p in self.policies[1:]:
            c = p.config
            if (c.chunk_size, c.max_action_dim, c.num_steps) != (
                cfg0.chunk_size, cfg0.max_action_dim, cfg0.num_steps
            ):
                raise ValueError(
                    "Composite policies must share chunk_size/max_action_dim/num_steps"
                )

    def _padded_dim(self) -> int:
        return self.policies[0].config.max_action_dim

    def _capture_model_inputs(self, policy, batch):
        return _capture_via(
            policy, policy.model, "sample_actions", batch,
            keep=lambda images, img_masks, lang_tokens, lang_masks, state, **kw: (
                images, img_masks, lang_tokens, lang_masks, state
            ),
        )

    @torch.no_grad()
    def _predict_joint(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        cfg0 = self.policies[0].config

        # Per-policy input prep + prefix KV cache (vision/language runs once).
        prefixes = []
        device = None
        for policy, batch in zip(self.policies, batches):
            images, img_masks, lang_tokens, lang_masks, state = self._capture_model_inputs(
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

        self._ensure_weights(device)
        x_t = sample_noise((1, cfg0.chunk_size, cfg0.max_action_dim), device)

        # Shared Euler loop: v = sum_i w_i * v_i, x <- x + dt * v.
        num_steps = cfg0.num_steps
        dt = -1.0 / num_steps
        for step in range(num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device).expand(1)
            v = torch.zeros_like(x_t)
            for (model, prefix_pad_masks, past_kv), w_vec in zip(prefixes, self._w):
                v_i = model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_kv,
                    x_t=x_t,
                    timestep=t_tensor,
                )
                v = v + w_vec * v_i
            x_t = x_t + dt * v

        return x_t[:, :, : self._real_dim()]


class DiffusionComposer(_ComposerBase):
    """Joint DDPM/DDIM sampler over N DiffusionPolicy policies: one shared
    trajectory, eps = sum_i w_i * eps_i at every scheduler step (score
    composition; each policy conditions on its own observation encoding)."""

    def _check_compatible(self) -> None:
        cfg0 = self.policies[0].config
        keys = (
            "horizon", "n_obs_steps", "n_action_steps",
            "noise_scheduler_type", "prediction_type",
            "num_train_timesteps", "num_inference_steps",
        )
        sig0 = tuple(getattr(cfg0, k) for k in keys)
        for p in self.policies[1:]:
            sig = tuple(getattr(p.config, k) for k in keys)
            if sig != sig0 or p.config.action_feature.shape != cfg0.action_feature.shape:
                raise ValueError(
                    f"Composite diffusion policies must share {keys} and action dim"
                )

    def _capture_global_cond(self, policy, batch):
        """The policy's own predict_action_chunk builds the batch (image
        stacking, obs history) and encodes it; we stop right before sampling."""
        return _capture_via(
            policy, policy.diffusion, "conditional_sample", batch,
            keep=lambda batch_size, global_cond=None, **kw: global_cond,
        )

    @torch.no_grad()
    def _predict_joint(self, batches: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        cfg0 = self.policies[0].config
        conds = [
            self._capture_global_cond(policy, batch)
            for policy, batch in zip(self.policies, batches)
        ]

        m0 = self.policies[0].diffusion
        device = conds[0].device
        dtype = next(m0.parameters()).dtype
        self._ensure_weights(device)

        # Shared reverse-diffusion loop, mirroring DiffusionModel.conditional_sample.
        sample = sample_noise(
            (1, cfg0.horizon, cfg0.action_feature.shape[0]), device, dtype
        )
        scheduler = m0.noise_scheduler
        scheduler.set_timesteps(m0.num_inference_steps)
        for t in scheduler.timesteps:
            t_batch = torch.full(sample.shape[:1], t, dtype=torch.long, device=device)
            out = torch.zeros_like(sample)
            for policy, cond, w_vec in zip(self.policies, conds, self._w):
                out = out + w_vec * policy.diffusion.unet(sample, t_batch, global_cond=cond)
            sample = scheduler.step(out, t, sample).prev_sample

        # Extract n_action_steps worth of actions, as generate_actions does.
        start = cfg0.n_obs_steps - 1
        return sample[:, start : start + cfg0.n_action_steps]


def make_composer(policies, postprocessors, weights, mode="joint"):
    """Pick the composer for the policy family of the (homogeneous) set."""
    names = {getattr(p, "name", type(p).__name__) for p in policies}
    if len(names) != 1:
        raise ValueError(f"Composite policies must be one family, got {names}")
    name = names.pop()
    composer = {"smolvla": SmolVLAComposer, "diffusion": DiffusionComposer}.get(name)
    if composer is None:
        raise ValueError(f"No composer implemented for policy family {name!r}")
    return composer(policies, postprocessors, weights, mode=mode)


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
    mode = str(spec.get("mode", "joint")).lower()
    return {"models": resolved, "weights": parsed, "mode": mode}
