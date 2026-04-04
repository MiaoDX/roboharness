"""Backwards-compatible re-export — use ``roboharness.robots.unitree_g1`` instead."""

from __future__ import annotations

from roboharness.robots.unitree_g1.locomotion import (  # noqa: F401
    ACTION_SCALE,
    ANG_VEL_SCALE,
    CMD_SCALE,
    DOF_POS_SCALE,
    DOF_VEL_SCALE,
    GROOT_BALANCE_FILE,
    GROOT_DEFAULT_ANGLES,
    GROOT_HF_REPO,
    GROOT_WALK_FILE,
    HOLOSOMA_ACTION_SCALE,
    HOLOSOMA_DEFAULT_ANGLES,
    HOLOSOMA_GAIT_PERIOD,
    HOLOSOMA_HF_REPO,
    HOLOSOMA_MODEL_FILE,
    HOLOSOMA_OBS_DIM,
    NUM_BODY_JOINTS,
    NUM_LOWER_BODY_JOINTS,
    OBS_FRAME_DIM,
    OBS_HISTORY_LEN,
    GrootLocomotionController,
    HolosomaLocomotionController,
    get_gravity_orientation,
)

__all__ = [
    "GrootLocomotionController",
    "HolosomaLocomotionController",
    "get_gravity_orientation",
]
