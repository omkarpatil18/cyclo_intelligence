# Shared-norm training

Trains GR00T N1.7 and LeRobot Diffusion Policy so that every policy in a
**composition group** applies the same invertible normalization, which is what
score-space composition requires.

Statistics (`mean/std/min/max/q01/q99`) are pooled over *all frames of all tasks in
a group*, then written identically into each member's dataset. Training itself is
stock — the only thing that differs from a default fine-tune is the statistics file.

Groups used for FFW SG2 Rev1:

| Group | Tasks | Pooled frames |
|---|---|---|
| A | push-tape-left, push-tape-right | 5 768 |
| B | pick-blue-cylinder-left-arm, pick-blue-cylinder-right-arm, blue-cylinder-handover | 11 870 |
| C | move-soft-toy-left, move-soft-toy-right | 5 249 |

Composability holds **within** a group, not across groups, and not across
architectures: GR00T reads `q01`/`q99` (`use_percentiles=True`), Diffusion Policy
reads `min`/`max` (`STATE`/`ACTION` default to `MIN_MAX`). Same file, different
fields. Verify with the stats hash printed by `make_sharednorm.py`.

## Pipeline

```bash
# 1. MCAP -> LeRobot v2.1 at 15 fps (matches the rest of the datasets; the
#    converter defaults to 30, which silently halves the action-chunk horizon)
python cyclo_data/cyclo_data/converter/scripts/convert_rosbag_to_lerobot.py \
    --input-dir raw/<task> --output conv15/<task>_lerobot_v21 \
    --repo-id <user>/<task> --version v2.1 --fps 15 \
    --robot-type ffw_sg2_rev1 \
    --robot-config shared/shared/robot_configs/ffw_sg2_rev1_config.yaml

# 2. pool group statistics into each member (writes *_lerobot_v21_sharednorm)
python make_sharednorm.py <out_dir> <ref_modality.json> \
    task-a=conv15/task-a_lerobot_v21  task-b=conv15/task-b_lerobot_v21

# 3. GR00T
GROOT_WS=... ./train_groot_sharednorm.sh <task>
```

For Diffusion Policy the datasets need extra preparation:

```bash
# camera subset (LeRobot has no camera-selection flag; the feature list in
# meta/info.json is what it reads, so a subset needs its own dataset)
python make_camvariant.py <src> <dst> cam_left_head cam_left_wrist cam_right_wrist

# LeRobot 0.6.1 requires v3.0; GR00T requires v2.1, so diffusion gets its own copies
python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id <id> --root <dst> --push-to-hub=false

# MUST run after conversion: the v2.1->v3.0 converter regenerates stats.json from
# local data, silently replacing pooled group values with per-task ones. Skipping
# this yields policies labelled shared-norm that are not composable.
python restore_pooled_stats.py <v21_src> <v30_dst>

# only if using >1 camera: Diffusion Policy requires a single resolution across
# cameras, and this robot's differ (head 376x672, wrist 424x240)
python uniform_cams.py <v30_dst> 240

GROOT_WS=... ./train_diffusion_sharednorm.sh <task> dp_cam3
```

## n_obs_steps

`train_diffusion_sharednorm.sh` forces `--policy.n_obs_steps=1`. LeRobot defaults to
2, but the cyclo inference engine feeds a single live frame, and
`generate_actions` asserts the observation-step count matches the trained config.
A model trained at 2 cannot be served. See
`lerobot_engine/preprocessing.py::_state_obs_steps`, which adds the matching
observation-step axis for diffusion policies (ACT needs `(B, D)` and would break;
SmolVLA accepts either).
