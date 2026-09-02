#!/usr/bin/env python3
"""Build *_lerobot_v21_sharednorm datasets with normalization pooled over a group.

Statistics (mean/std/min/max/q01/q99 for observation.state and action) are computed
over *every frame of every task in the group*, then written identically into each
member dataset. That gives all group members one shared invertible transform, which
is what score-space composition requires.

GR00T's stock normalizer then does q01/q99 min-max to [-1,1] with outlier clipping,
so no code patches are needed -- unlike the no-norm recipe, which had to force
clip_outliers=False to protect raw radians.

Usage: make_sharednorm.py OUT_DIR REF_MODALITY NAME=SRC [NAME=SRC ...]
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Columns that get pooled stats. Matches the reference stats.json key set.
STAT_COLS = ("observation.state", "action", "timestamp")


def load_column(dataset: Path, col: str) -> np.ndarray:
    """Concatenate one column across every episode parquet as (frames, width)."""
    chunks = []
    for f in sorted((dataset / "data").rglob("*.parquet")):
        tbl = pq.read_table(f, columns=[col])
        arr = tbl.column(col).to_pylist()
        a = np.asarray(arr, dtype=np.float64)
        if a.ndim == 1:
            a = a[:, None]
        chunks.append(a)
    if not chunks:
        raise SystemExit(f"no parquet data under {dataset}/data")
    return np.concatenate(chunks, axis=0)


def main() -> int:
    out_dir = Path(sys.argv[1])
    ref_modality = Path(sys.argv[2])
    members = [m.split("=", 1) for m in sys.argv[3:]]

    # --- pool every frame of every member, per column -------------------------
    pooled: dict[str, np.ndarray] = {}
    per_member_frames: dict[str, int] = {}
    for name, src in members:
        for col in STAT_COLS:
            a = load_column(Path(src), col)
            pooled[col] = a if col not in pooled else np.concatenate([pooled[col], a], axis=0)
            if col == "observation.state":
                per_member_frames[name] = a.shape[0]

    stats = {}
    for col in STAT_COLS:
        a = pooled[col]
        stats[col] = {
            "mean": a.mean(axis=0).tolist(),
            "std": a.std(axis=0).tolist(),
            "min": a.min(axis=0).tolist(),
            "max": a.max(axis=0).tolist(),
            "q01": np.percentile(a, 1, axis=0).tolist(),
            "q99": np.percentile(a, 99, axis=0).tolist(),
        }

    total = pooled["observation.state"].shape[0]
    print("=== pooled over group ===")
    for n, f in per_member_frames.items():
        print(f"  {n:32s} {f:6d} frames")
    print(f"  {'TOTAL':32s} {total:6d} frames")
    print()
    for col in STAT_COLS:
        w = len(stats[col]["mean"])
        print(f"  {col:20s} width={w:3d} "
              f"mean[:3]={[round(v,4) for v in stats[col]['mean'][:3]]} "
              f"q01[:3]={[round(v,4) for v in stats[col]['q01'][:3]]} "
              f"q99[:3]={[round(v,4) for v in stats[col]['q99'][:3]]}")
    print()

    # --- write the identical stats into every member -------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src in members:
        dst = out_dir / f"{name}_lerobot_v21_sharednorm"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=True)

        (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))
        if not (dst / "meta" / "modality.json").exists():
            shutil.copy2(ref_modality, dst / "meta" / "modality.json")
        rel = dst / "meta" / "relative_stats.json"
        if not rel.exists():
            rel.write_text("{}")
        print(f"wrote {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
