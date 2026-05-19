"""Backwards-compatible re-export — use ``roboharness.robots.unitree_g1`` instead."""

from __future__ import annotations

from roboharness.robots.unitree_g1 import locomotion as _locomotion

__all__ = list(_locomotion.__all__)

globals().update({name: getattr(_locomotion, name) for name in __all__})
