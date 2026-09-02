#!/usr/bin/env bash
# Launch one diffusion run. Kept as a file so nothing has to survive ssh quoting.
#   run_dp.sh TASK VARIANT
# LeRobot defaults are used throughout; only dataset and output path vary.
set -uo pipefail

WS=${GROOT_WS:-/scratch/opatil3/groot_nonorm_ws}
TASK=$1
VARIANT=$2
NAME="${TASK}_${VARIANT}"

export LD_LIBRARY_PATH="$WS/ffmpeg/lib:${LD_LIBRARY_PATH:-}"
export HF_LEROBOT_HOME="$WS/lerobot_home"
export TOKENIZERS_PARALLELISM=false

# Diffusion Policy requires every camera to share one resolution, and this robot's
# do not (head 376x672, wrist 424x240). The wrist-only variant is uniform so it runs
# on stock defaults; the 3-camera variant needs an explicit resize or it aborts with
# "we expect all image shapes to match".
# n_obs_steps=1 is a hard requirement, not a tuning choice: the robot's inference
# stack buffers a single observation frame. LeRobot's default of 2 would train a
# policy that cannot be deployed. This also matches GR00T, whose modality config
# uses delta_indices=[0] (single observation).
EXTRA=(--policy.n_obs_steps=1)

cd "$WS"
exec "$WS/lerobot_env/bin/lerobot-train" \
  --policy.type=diffusion \
  --dataset.repo_id="omkarpatil/${NAME//_/-}" \
  --dataset.root="$WS/workspace/datasets_v30/$NAME" \
  --policy.device=cuda \
  --output_dir="$WS/checkpoints_dp/$NAME" \
  --policy.push_to_hub=false \
  "${EXTRA[@]}"
