#!/usr/bin/env python3
"""Restore pooled group statistics into a v3.0 dataset after conversion.

convert_dataset_v21_to_v30 regenerates meta/stats.json from the local episodes,
which silently replaces the group-pooled observation.state/action statistics with
per-task ones. That would leave every policy in a group on its own transform while
still being labelled shared-norm -- trains and evaluates fine, but is not composable.

This copies the pooled entries back over the regenerated ones, leaving the
converter's other entries (image and index columns) untouched.

Usage: restore_pooled_stats.py V21_SRC V30_DST
"""
import json
import sys
from pathlib import Path

# Only these carry the shared-norm contract; the rest are per-dataset bookkeeping.
POOLED_KEYS = ("observation.state", "action")


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    pooled = json.loads((src / "meta" / "stats.json").read_text())
    stats_path = dst / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text())

    changed = []
    for key in POOLED_KEYS:
        if key not in pooled:
            continue
        before = stats.get(key, {})
        merged = dict(before)
        merged.update(pooled[key])  # pooled fields win; keep extras like "count"
        stats[key] = merged
        if before.get("min") != pooled[key].get("min"):
            changed.append(key)

    stats_path.write_text(json.dumps(stats, indent=4))
    print(f"{dst.name}: restored pooled stats for {changed or 'nothing (already equal)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
