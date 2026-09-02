#!/usr/bin/env python3
"""Build a camera-subset copy of a shared-norm dataset for LeRobot training.

LeRobot derives its visual inputs from `dataset.meta.camera_keys`, which comes
straight from meta/info.json -- there is no CLI flag to select a camera subset.
So each camera variant needs its own dataset whose feature list contains only the
wanted cameras.

Videos are symlinked rather than copied: the variants are only a different view of
the same footage, and copying would multiply disk for no reason.

meta/stats.json is carried over untouched, so every variant of a group keeps the
group's shared normalization.

Usage: make_camvariant.py SRC DST CAM [CAM ...]
"""
import json
import os
import shutil
import sys
from pathlib import Path

VIDEO_PREFIX = "observation.images.rgb."


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    keep = set(sys.argv[3:])

    info = json.loads((src / "meta" / "info.json").read_text())
    available = {k.removeprefix(VIDEO_PREFIX)
                 for k in info["features"] if k.startswith(VIDEO_PREFIX)}
    missing = keep - available
    if missing:
        raise SystemExit(f"cameras not in {src.name}: {sorted(missing)} "
                         f"(available: {sorted(available)})")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # meta/: copy every file, then rewrite info.json with the reduced feature set.
    shutil.copytree(src / "meta", dst / "meta")
    for extra in ("annotations", "data"):
        if (src / extra).exists():
            shutil.copytree(src / extra, dst / extra, symlinks=True)
    for f in src.glob("*.json"):
        shutil.copy2(f, dst / f.name)

    dropped = sorted(available - keep)
    info["features"] = {
        k: v for k, v in info["features"].items()
        if not k.startswith(VIDEO_PREFIX) or k.removeprefix(VIDEO_PREFIX) in keep
    }
    (dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    # videos/: symlink only the kept camera directories.
    for chunk in sorted((src / "videos").glob("chunk-*")):
        for camdir in sorted(chunk.iterdir()):
            if not camdir.name.startswith(VIDEO_PREFIX):
                continue
            if camdir.name.removeprefix(VIDEO_PREFIX) not in keep:
                continue
            out = dst / "videos" / chunk.name / camdir.name
            out.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(camdir.resolve(), out)

    kept = sorted(k.removeprefix(VIDEO_PREFIX)
                  for k in info["features"] if k.startswith(VIDEO_PREFIX))
    print(f"{dst.name}: kept={kept} dropped={dropped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
