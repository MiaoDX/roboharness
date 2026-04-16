"""LeRobot policy loading adapter — normalizes checkpoint inference to obs → action."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LeRobotPolicyAdapter:
    """Adapter that loads a LeRobot policy checkpoint and exposes obs → action."""

    def __init__(self, policy: Any, device: str = "cpu") -> None:
        """Wrap a loaded LeRobot policy.

        Args:
            policy: A LeRobot policy instance with a ``select_action`` method.
            device: Torch device string for inference.
        """
        self.policy = policy
        self.device = device

    def __call__(self, obs: np.ndarray | dict[str, Any]) -> np.ndarray:
        """Run inference and return a numpy action array."""
        import torch

        with torch.inference_mode():
            if isinstance(obs, dict):
                tensor_obs = {k: torch.as_tensor(v, device=self.device) for k, v in obs.items()}
            else:
                tensor_obs = torch.as_tensor(obs, device=self.device)
            action = self.policy.select_action(tensor_obs)
            if hasattr(action, "cpu"):
                action = action.cpu()
            if hasattr(action, "numpy"):
                action = action.numpy()
        return np.asarray(action)


def load_lerobot_policy(checkpoint_path: str | Path, device: str = "cpu") -> LeRobotPolicyAdapter:
    """Load a LeRobot policy from a checkpoint directory.

    Uses LeRobot's own policy-loading utilities so checkpoint formats
    (Diffusion, ACT, TDMPC, etc.) are handled automatically.

    Args:
        checkpoint_path: Path to a LeRobot checkpoint directory.
        device: Torch device for inference (default "cpu").

    Returns:
        A ``LeRobotPolicyAdapter`` wrapping the loaded policy.

    Raises:
        FileNotFoundError: If the checkpoint directory does not exist.
        RuntimeError: If policy loading fails (wrapped with a clear message).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_path}\n"
            "Please provide a valid LeRobot checkpoint path."
        )

    try:
        from lerobot.common.policies.factory import (  # type: ignore[import-not-found]
            make_policy,
        )
        from lerobot.configs.train import TrainConfig  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is required to load policies. Install with: pip install roboharness[lerobot]"
        ) from exc

    try:  # noqa: SIM105
        import torch  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass

    cfg_path = checkpoint_path / "train_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Missing train_config.json in checkpoint directory: {checkpoint_path}"
        )

    try:
        train_cfg = TrainConfig.from_json(cfg_path)
        policy = make_policy(train_cfg, checkpoint_dir=checkpoint_path)
        policy.eval()
        if hasattr(policy, "to"):
            policy = policy.to(device)

        logger.info("Loaded LeRobot policy from %s on %s", checkpoint_path, device)
        return LeRobotPolicyAdapter(policy, device=device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load LeRobot policy from {checkpoint_path}. "
            f"Ensure the checkpoint is compatible with your LeRobot version. "
            f"Original error: {exc}"
        ) from exc
