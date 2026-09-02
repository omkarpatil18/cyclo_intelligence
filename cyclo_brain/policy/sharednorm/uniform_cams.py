#!/usr/bin/env python3
"""Re-encode a v3.0 dataset's cameras to one common resolution.

Diffusion Policy's validate_features() compares the dataset's declared image
shapes and aborts if they differ. It runs before any resize, so --policy.resize_shape
cannot rescue mismatched cameras -- the footage itself has to be uniform.

Rewrites every video to SIZExSIZE and updates meta/info.json to match.

Usage: uniform_cams.py DATASET_ROOT SIZE
"""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FFMPEG = "/scratch/opatil3/kfmn/ffmpeg7/bin/ffmpeg"
PREFIX = "observation.images."


def reencode(mp4: Path, size: int) -> tuple[Path, int]:
    tmp = mp4.with_suffix(".tmp.mp4")
    r = subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
         "-vf", f"scale={size}:{size}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-pix_fmt", "yuv420p", str(tmp)],
        capture_output=True, text=True)
    if r.returncode == 0 and tmp.exists():
        tmp.replace(mp4)
    else:
        tmp.unlink(missing_ok=True)
    return mp4, r.returncode


def main() -> int:
    root, size = Path(sys.argv[1]), int(sys.argv[2])
    vids = sorted(root.rglob("*.mp4"))
    if not vids:
        raise SystemExit(f"no videos under {root}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda p: reencode(p, size), vids))
    bad = [str(p) for p, rc in results if rc != 0]

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for key, feat in info["features"].items():
        if not key.startswith(PREFIX):
            continue
        shape = list(feat.get("shape", []))
        if len(shape) == 3:
            feat["shape"] = [shape[0], size, size]
        vi = feat.get("info")
        if isinstance(vi, dict):
            vi["video.height"] = size
            vi["video.width"] = size
    info_path.write_text(json.dumps(info, indent=4))

    print(f"{root.name}: {len(vids)} videos -> {size}x{size}"
          f"{'  FAILED: ' + str(len(bad)) if bad else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
