# ffw_sg2_rev1, arms-only action, stock GR00T normalization for the shared-norm recipe.
#
# Identical to ffw_sg2_rev1_arms_meanstd_config.py except that mean_std_embedding_keys is
# NOT set on state/action. That field selects StateActionProcessor "Strategy 2"
# (normalize_values_meanstd); omitting it falls through to "Strategy 3", the stock
# min-max-to-[-1,1] path, which with Gr00tN1d7Config.use_percentiles=True (default) uses
# q01/q99 as the bounds and clips outliers (clip_outliers defaults True) -- i.e. exactly
# GR00T's pretraining scheme, so no launch-driver patches are required.
#
# The mean/std variant exists only to make normalization the identity when paired with
# identity stats (the no-norm recipe); with pooled group statistics it would silently
# apply mean/std instead of the intended stock transform.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig,
)

_STATE_KEYS = ["arm_left", "arm_right", "head", "lift", "odometry"]
_ACTION_KEYS = ["arm_left", "arm_right"]

config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["cam_left_head", "cam_left_wrist", "cam_right_wrist"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=_STATE_KEYS),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=_ACTION_KEYS,
        action_configs=[
            ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF,
                         format=ActionFormat.DEFAULT)
            for _ in _ACTION_KEYS
        ],
    ),
    "language": ModalityConfig(delta_indices=[0],
                               modality_keys=["annotation.human.primitive_instruction"]),
}

register_modality_config(config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
