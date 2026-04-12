"""VectorEnvAdapter — adapts a single-instance VectorEnv to standard gym.Env.

LeRobot's ``make_env()`` factory wraps environments in ``SyncVectorEnv`` even
when ``n_envs=1``.  This causes incompatibilities with wrappers (like
``RobotHarnessWrapper``) that expect a standard ``gym.Env`` interface, because
observations gain a batch dimension, ``step()`` returns arrays instead of
scalars, and the ``unwrapped`` chain breaks.

``VectorEnvAdapter`` fixes this by squeezing the batch dimension so downstream
code sees a standard single-env interface.

Usage::

    from lerobot.envs.factory import make_env
    from roboharness.wrappers import VectorEnvAdapter, RobotHarnessWrapper

    vec_env = make_env("lerobot/unitree-g1-mujoco", n_envs=1)
    env = VectorEnvAdapter(vec_env)
    env = RobotHarnessWrapper(env, ...)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


class VectorEnvAdapter(gym.Env):  # type: ignore[type-arg]
    """Adapts a ``VectorEnv`` (``num_envs=1``) to a standard ``gym.Env``.

    Parameters
    ----------
    vec_env:
        A Gymnasium ``VectorEnv`` instance with exactly one sub-environment.
        Typically the return value of LeRobot's ``make_env(..., n_envs=1)``.
    """

    def __init__(self, vec_env: gym.vector.VectorEnv) -> None:  # type: ignore[name-defined]
        num_envs: int = getattr(vec_env, "num_envs", 0)
        if num_envs != 1:
            msg = (
                f"VectorEnvAdapter requires num_envs=1, got {num_envs}. "
                "Pass a VectorEnv wrapping exactly one sub-environment."
            )
            raise ValueError(msg)

        self._vec_env = vec_env

        # Expose single-env spaces (without the batch dimension).
        self.observation_space = vec_env.single_observation_space
        self.action_space = vec_env.single_action_space

        # Preserve env metadata expected by Gymnasium wrappers.
        self.metadata: dict[str, Any] = getattr(vec_env, "metadata", {})
        self.render_mode: str | None = getattr(vec_env, "render_mode", None)

        # Cache a reference to the underlying single env for attribute proxying.
        # SyncVectorEnv stores sub-envs in .envs; AsyncVectorEnv doesn't expose
        # them, so this may be None.
        envs = getattr(vec_env, "envs", None)
        self._single_env: gym.Env | None = envs[0] if envs else None  # type: ignore[index,type-arg]

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self._vec_env.reset(seed=seed, options=options)
        return _squeeze_obs(obs), _squeeze_info(info)

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        # VectorEnv.step() expects shape (num_envs, *action_shape).
        batched = np.expand_dims(np.asarray(action), axis=0)
        obs, rewards, terminateds, truncateds, info = self._vec_env.step(batched)

        return (
            _squeeze_obs(obs),
            float(rewards[0]),
            bool(terminateds[0]),
            bool(truncateds[0]),
            _squeeze_info(info),
        )

    def render(self) -> Any:
        frames: Any = self._vec_env.render()
        if frames is None:
            return None
        # SyncVectorEnv.render() returns a tuple of frames (one per env)
        # or a stacked ndarray with shape (num_envs, H, W, C).
        if isinstance(frames, tuple):
            return frames[0] if frames else None
        if isinstance(frames, list):
            return frames[0] if frames else None
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            return frames[0]
        return frames

    def close(self) -> None:
        self._vec_env.close()

    # ------------------------------------------------------------------
    # Attribute proxying
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> gym.Env:  # type: ignore[type-arg]
        """Return the underlying single env's ``unwrapped`` (bypassing VectorEnv)."""
        if self._single_env is not None:
            return getattr(self._single_env, "unwrapped", self._single_env)
        return self  # type: ignore[return-value]

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying single env.

        This lets callers access env-specific attributes (e.g. ``mj_model``,
        ``render_camera``) that are defined on the sub-env but not on the
        VectorEnv wrapper.
        """
        # Avoid infinite recursion for private / dunder attributes.
        if name.startswith("_"):
            raise AttributeError(name)
        single = self.__dict__.get("_single_env")
        if single is not None:
            return getattr(single, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _squeeze_obs(obs: Any) -> Any:
    """Remove the batch dimension from an observation."""
    if isinstance(obs, dict):
        return {k: v[0] if hasattr(v, "__getitem__") else v for k, v in obs.items()}
    if hasattr(obs, "__getitem__"):
        return obs[0]
    return obs


def _squeeze_info(info: dict[str, Any]) -> dict[str, Any]:
    """Remove the batch dimension from a VectorEnv info dict.

    Gymnasium's ``SyncVectorEnv`` returns info values as lists or arrays
    with length ``num_envs``.  For ``num_envs=1`` we extract the single
    element.
    """
    if not info:
        return {}
    squeezed: dict[str, Any] = {}
    for key, val in info.items():
        if (isinstance(val, np.ndarray) and val.ndim >= 1 and val.shape[0] == 1) or (
            isinstance(val, (list, tuple)) and len(val) == 1
        ):
            squeezed[key] = val[0]
        else:
            squeezed[key] = val
    return squeezed
