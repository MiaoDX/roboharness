"""Unitree G1 humanoid support — locomotion controllers and constants.

Provides GR00T-based locomotion policies (Balance + Walk) and G1-specific
joint configuration constants. Requires ``pip install roboharness[demo]``
for ONNX runtime and HuggingFace model downloads.

Usage::

    from roboharness.robots.unitree_g1 import GrootLocomotionController

    ctrl = GrootLocomotionController()
    action = ctrl.compute(
        command={"velocity": [0, 0, 0]},
        state={"qpos": data.qpos, "qvel": data.qvel},
    )
"""

from __future__ import annotations

__all__: list[str] = []

# Conditional exports — only available when [demo] extras are installed
try:
    from roboharness.robots.unitree_g1.locomotion import (
        GrootLocomotionController,
        HolosomaLocomotionController,
        MotionClip,
        MotionClipLoader,
        SonicLocomotionController,
        SonicMode,
    )

    __all__ += [
        "GrootLocomotionController",
        "HolosomaLocomotionController",
        "MotionClip",
        "MotionClipLoader",
        "SonicLocomotionController",
        "SonicMode",
    ]
except ImportError:
    pass
