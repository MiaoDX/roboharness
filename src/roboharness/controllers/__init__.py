"""Controller implementations.

Generic controllers (robot-agnostic):
  - ``WbcIkController`` — Differential-IK via Pinocchio + Pink (requires ``[demo]``)

Robot-specific controllers have moved to ``roboharness.robots``:
  - ``roboharness.robots.unitree_g1.GrootLocomotionController`` (requires ``[demo]``)
"""

from __future__ import annotations

__all__: list[str] = []

# Conditional export — only available when [demo] extras are installed
try:
    from roboharness.controllers.wbc_ik import WbcIkController, WbcIkSettings

    __all__ += ["WbcIkController", "WbcIkSettings"]
except ImportError:
    pass
