"""Tests for LeRobot policy adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roboharness.evaluate.lerobot_policy_adapter import (
    LeRobotPolicyAdapter,
    load_lerobot_policy,
)


class FakeTensor:
    """Fake torch tensor for testing without torch installed."""

    def __init__(self, data: Any) -> None:
        self._data = np.asarray(data)

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._data


class FakePolicy:
    """Fake LeRobot policy for testing."""

    def __init__(self) -> None:
        self.eval_called = False
        self.to_device: str | None = None

    def eval(self) -> FakePolicy:
        self.eval_called = True
        return self

    def to(self, device: str) -> FakePolicy:
        self.to_device = device
        return self

    def select_action(self, obs: Any) -> Any:
        if isinstance(obs, dict):
            first = next(iter(obs.values()))
            shape = getattr(first, "shape", (4,))
            return FakeTensor(np.ones((shape[0], 3)))
        return FakeTensor(np.ones(3))


def _mock_torch_module() -> Any:
    """Build a minimal mock torch module."""
    mock_torch = MagicMock()

    def as_tensor(x: Any, device: str | None = None) -> Any:
        if isinstance(x, dict):
            return {k: np.asarray(v) for k, v in x.items()}
        return np.asarray(x)

    mock_torch.as_tensor = as_tensor
    mock_torch.inference_mode = MagicMock()
    mock_torch.inference_mode.return_value.__enter__ = lambda self: self
    mock_torch.inference_mode.return_value.__exit__ = lambda *args: None
    return mock_torch


class TestLeRobotPolicyAdapter:
    @patch.dict(
        "sys.modules",
        {"torch": _mock_torch_module()},
    )
    def test_call_with_numpy_obs(self) -> None:
        policy = FakePolicy()
        adapter = LeRobotPolicyAdapter(policy, device="cpu")

        obs = np.zeros(4, dtype=np.float32)
        action = adapter(obs)

        assert np.array_equal(action, np.ones(3))

    @patch.dict(
        "sys.modules",
        {"torch": _mock_torch_module()},
    )
    def test_call_with_dict_obs(self) -> None:
        policy = FakePolicy()
        adapter = LeRobotPolicyAdapter(policy, device="cpu")

        obs = {"state": np.zeros(4, dtype=np.float32)}
        action = adapter(obs)

        assert np.array_equal(action, np.ones((4, 3)))

    @patch.dict(
        "sys.modules",
        {"torch": _mock_torch_module()},
    )
    def test_call_returns_numpy(self) -> None:
        policy = FakePolicy()
        adapter = LeRobotPolicyAdapter(policy, device="cpu")

        obs = np.zeros(4, dtype=np.float32)
        action = adapter(obs)

        assert isinstance(action, np.ndarray)
        assert np.array_equal(action, np.ones(3))


class TestLoadLeRobotPolicy:
    def test_missing_checkpoint_directory(self) -> None:
        with pytest.raises(FileNotFoundError, match="Checkpoint directory not found"):
            load_lerobot_policy("/definitely/does/not/exist")

    def test_missing_train_config(self, tmp_path: Path) -> None:
        mock_torch = _mock_torch_module()
        with (
            patch.dict(
                "sys.modules",
                {
                    "torch": mock_torch,
                    "lerobot": MagicMock(),
                    "lerobot.common": MagicMock(),
                    "lerobot.common.policies": MagicMock(),
                    "lerobot.common.policies.factory": MagicMock(),
                    "lerobot.configs": MagicMock(),
                    "lerobot.configs.train": MagicMock(),
                },
            ),
            pytest.raises(FileNotFoundError, match=re.escape("Missing train_config.json")),
        ):
            load_lerobot_policy(str(tmp_path))

    def test_happy_path(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "train_config.json"
        cfg_path.write_text('{"policy": {}}')

        fake_policy = FakePolicy()
        mock_train = MagicMock()
        mock_train.from_json.return_value = MagicMock()
        mock_make = MagicMock()
        mock_make.return_value = fake_policy

        with patch.dict(
            "sys.modules",
            {
                "torch": _mock_torch_module(),
                "lerobot": MagicMock(),
                "lerobot.common": MagicMock(),
                "lerobot.common.policies": MagicMock(),
                "lerobot.common.policies.factory": MagicMock(make_policy=mock_make),
                "lerobot.configs": MagicMock(),
                "lerobot.configs.train": MagicMock(TrainConfig=mock_train),
            },
        ):
            adapter = load_lerobot_policy(str(tmp_path), device="cpu")

        assert isinstance(adapter, LeRobotPolicyAdapter)
        assert adapter.policy is fake_policy
        assert fake_policy.eval_called is True
        assert fake_policy.to_device == "cpu"
        mock_make.assert_called_once()

    def test_loading_failure_wrapped(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "train_config.json"
        cfg_path.write_text('{"policy": {}}')

        mock_train = MagicMock()
        mock_train.from_json.return_value = MagicMock()
        mock_make = MagicMock()
        mock_make.side_effect = RuntimeError("corrupted checkpoint")

        with (
            patch.dict(
                "sys.modules",
                {
                    "torch": _mock_torch_module(),
                    "lerobot": MagicMock(),
                    "lerobot.common": MagicMock(),
                    "lerobot.common.policies": MagicMock(),
                    "lerobot.common.policies.factory": MagicMock(make_policy=mock_make),
                    "lerobot.configs": MagicMock(),
                    "lerobot.configs.train": MagicMock(TrainConfig=mock_train),
                },
            ),
            pytest.raises(RuntimeError, match="Failed to load LeRobot policy"),
        ):
            load_lerobot_policy(str(tmp_path))

    def test_import_error_without_lerobot(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "train_config.json"
        cfg_path.write_text('{"policy": {}}')

        with (
            patch.dict("sys.modules", {"lerobot": None}),
            pytest.raises(RuntimeError, match="LeRobot is required"),
        ):
            load_lerobot_policy(str(tmp_path))
