#!/usr/bin/env bash
# One shared-norm fine-tune, stock GR00T training (no launch-driver patches).
#
#   train_sharednorm.sh NAME
#
# Differences from the no-norm recipe, all of them required by the shared-norm design:
#   * dataset  = *_lerobot_v21_sharednorm (pooled group stats, identical across members)
#   * modality = ffw_sg2_rev1_arms_sharednorm_config.py (no mean_std_embedding_keys, so
#                GR00T uses its stock q01/q99 min-max + clip_outliers path)
#   * entrypoint = gr00t/experiment/launch_finetune.py directly, NOT
#                  launch_finetune_noclip.py, so clip_outliers stays True and
#                  load_bf16 stays at its stock False.
# Every other training argument is unchanged from the no-norm runs.
set -euo pipefail

WS=${GROOT_WS:-/scratch/opatil3/groot_nonorm_ws}
NAME=$1
STEPS=${STEPS:-20000}

BASE=$(cat "$WS/.base_model_path")
MCFG="$WS/workspace/tools/ffw_sg2_rev1_arms_sharednorm_config.py"

export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$WS/ffmpeg/lib:${LD_LIBRARY_PATH:-}"

cd "$WS/Isaac-GR00T"
exec "$WS/env/bin/python" "$WS/Isaac-GR00T/gr00t/experiment/launch_finetune.py" \
  --base_model_path "$BASE" \
  --dataset_path "$WS/workspace/datasets/${NAME}_lerobot_v21_sharednorm" \
  --embodiment_tag NEW_EMBODIMENT \
  --modality_config_path "$MCFG" \
  --num_gpus 1 \
  --output_dir "$WS/checkpoints/${NAME}_groot_sharednorm" \
  --experiment_name "groot_${NAME}_sharednorm" \
  --save_steps 10000 --save_total_limit 5 --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size 32 --gradient_accumulation_steps 1 \
  --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 \
  --episode_sampling_rate 0.1
