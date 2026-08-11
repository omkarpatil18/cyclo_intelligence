#!/usr/bin/env python3
# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI for the raw inference episode admission gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cyclo_data.validation.inference_episode import validate_inference_episode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Validate one raw SG2 ACT inference/RL episode.',
    )
    parser.add_argument('episode', type=Path, help='Archived episode directory')
    parser.add_argument(
        '--json', action='store_true', dest='as_json',
        help='Print the complete machine-readable report',
    )
    parser.add_argument(
        '--no-video-probe', action='store_true',
        help='Skip ffprobe dimensions/frame-count checks',
    )
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = validate_inference_episode(
            args.episode,
            probe_video=not args.no_video_probe,
        )
    except Exception as exc:  # noqa: BLE001 - CLI safety boundary
        print(f'Internal validation error: {exc}', file=sys.stderr)
        return 2
    payload = report.to_dict()
    if args.as_json:
        print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    else:
        status = 'VALID' if report.valid else 'INVALID'
        print(
            f'[{status}] {report.episode_path} '
            f'(errors={len(report.errors)}, warnings={len(report.warnings)})'
        )
        for issue in report.issues:
            print(f'  {issue.severity.upper()} {issue.code}: {issue.message}')
        if report.metrics:
            print('  metrics: ' + json.dumps(report.metrics, ensure_ascii=False))
    return 0 if report.valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
