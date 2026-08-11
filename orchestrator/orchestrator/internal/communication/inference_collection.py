# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Dependency-light helpers for inference recording collection IDs."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid


_POLICY_DATASET_NAMES = {
    'act': 'ACT_dataset',
    'n17': 'GR00T_dataset',
}

_POLICY_ALIASES = {
    'groot': 'n17',
    'groot:n17': 'n17',
    'n1.7': 'n17',
    'lerobot:act': 'act',
}


def _policy_type_from_config(policy_path: str) -> str:
    """Read a policy family without importing a policy runtime."""
    raw_path = str(policy_path or '').strip()
    if not raw_path:
        return ''
    root = Path(raw_path)
    candidates = (
        root / 'config.json',
        root / 'pretrained_model' / 'config.json',
    )
    for config_path in candidates:
        try:
            with config_path.open('r', encoding='utf-8') as stream:
                config = json.load(stream) or {}
        except (OSError, ValueError, TypeError):
            continue
        for key in ('type', 'policy_type', 'policy_family'):
            value = str(config.get(key) or '').strip().lower()
            if value:
                return _POLICY_ALIASES.get(value, value)
    return ''


def resolve_inference_policy_type(
    policy_type: str,
    *,
    service_type: str = '',
    policy_path: str = '',
) -> str:
    """Resolve legacy backend labels to a concrete policy family.

    The result is recording provenance only; it is never injected into the
    policy's language observation.
    """
    policy = str(policy_type or '').strip().lower()
    if policy and policy != 'lerobot':
        return _POLICY_ALIASES.get(policy, policy)

    configured = _policy_type_from_config(policy_path)
    if configured:
        return configured

    service = str(service_type or '').strip().strip('/').lower()
    if service == 'groot':
        return 'n17'
    if service == 'lerobot' or policy == 'lerobot':
        return 'act'
    return ''


def inference_dataset_name(policy_type: str) -> str:
    """Return a filesystem-safe dataset label for a policy family."""
    policy = str(policy_type or '').strip().lower()
    if policy in _POLICY_DATASET_NAMES:
        return _POLICY_DATASET_NAMES[policy]
    safe_policy = re.sub(r'[^a-z0-9]+', '_', policy).strip('_')
    if safe_policy:
        return f'{safe_policy.upper()}_dataset'
    return 'Inference_dataset'


def make_inference_collection_id(
    policy_type: str,
    *,
    now: datetime | None = None,
    nonce: str | None = None,
) -> str:
    """Create a collision-resistant UTC identifier for one inference run."""
    instant = now or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        instant = instant.replace(tzinfo=timezone.utc)
    instant = instant.astimezone(timezone.utc)
    timestamp = instant.strftime('%Y%m%dT%H%M%S_%fZ')
    suffix = str(nonce or uuid.uuid4().hex[:8]).strip().lower()
    if not re.fullmatch(r'[a-z0-9]{4,32}', suffix):
        raise ValueError(f'Invalid inference collection nonce: {suffix!r}')
    return f'{inference_dataset_name(policy_type)}_{timestamp}_{suffix}'
