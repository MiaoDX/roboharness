"""Generic evaluation protocols for roboharness."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import numpy as np


class PolicyAdapter(Protocol):
    """Protocol for a policy callable: maps observation to action."""

    def __call__(self, obs: np.ndarray | dict[str, Any]) -> np.ndarray:
        """Return an action given an observation."""
        ...
