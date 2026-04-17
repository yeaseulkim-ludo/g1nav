"""GR00T modality config for the G1Nav locomotion embodiment."""
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

g1nav_config = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["lower_body_joints", "lower_body_joint_vels"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),   # predict 16 steps = 1.6 s at 10 Hz
        modality_keys=["lower_body_joints"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(g1nav_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
