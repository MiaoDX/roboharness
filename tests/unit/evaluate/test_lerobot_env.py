"""Tests for shared LeRobot environment utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from roboharness.evaluate.lerobot_env import (
    _add_mujoco_rendering,
    _patch_config_for_headless,
    _try_lerobot_make_env,
    create_native_env,
)


class FakeSpace:
    def __init__(self, shape: tuple[int, ...] = (4,)) -> None:
        self.shape = shape


class FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.observation_space = FakeSpace((4,))
        self.action_space = FakeSpace((2,))


class TestPatchConfigForHeadless:
    @patch.dict("os.environ", {}, clear=True)
    @patch("huggingface_hub.snapshot_download")
    def test_no_display_patches_config(self, mock_snapshot: Any, tmp_path: Path) -> None:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        config_path = repo_dir / "config.yaml"
        config_path.write_text("ENABLE_ONSCREEN: true\n")
        mock_snapshot.return_value = str(repo_dir)

        _patch_config_for_headless("lerobot/unitree-g1-mujoco")

        text = config_path.read_text()
        assert "ENABLE_ONSCREEN: false" in text
        assert "ENABLE_OFFSCREEN: true" in text

    @patch.dict("os.environ", {"DISPLAY": ":0"}, clear=True)
    def test_with_display_skips_patch(self) -> None:
        with patch("huggingface_hub.snapshot_download") as mock_snapshot:
            _patch_config_for_headless("lerobot/unitree-g1-mujoco")
            mock_snapshot.assert_not_called()

    @patch.dict("os.environ", {}, clear=True)
    @patch("huggingface_hub.snapshot_download")
    def test_snapshot_download_failure_graceful(self, mock_snapshot: Any) -> None:
        mock_snapshot.side_effect = Exception("network error")
        # Should not raise
        _patch_config_for_headless("lerobot/unitree-g1-mujoco")


class TestTryLerobotMakeEnv:
    def test_success(self) -> None:
        mock_module = MagicMock()
        fake_vec = MagicMock()
        fake_vec.observation_space = FakeSpace((4,))
        fake_vec.action_space = FakeSpace((2,))
        fake_vec.num_envs = 1
        mock_module.make_env.return_value = fake_vec

        with patch.dict("sys.modules", {"lerobot.common.envs.factory": mock_module}):
            env = _try_lerobot_make_env("lerobot/unitree-g1-mujoco")
            assert env is not None

    def test_import_error_returns_none(self) -> None:
        with patch.dict("sys.modules", {"lerobot": None}):
            env = _try_lerobot_make_env("lerobot/unitree-g1-mujoco")
            assert env is None

    def test_make_env_failure_returns_none(self) -> None:
        mock_module = MagicMock()
        mock_module.make_env.side_effect = RuntimeError("bad config")

        with patch.dict("sys.modules", {"lerobot.common.envs.factory": mock_module}):
            env = _try_lerobot_make_env("lerobot/unitree-g1-mujoco")
            assert env is None


class TestCreateNativeEnv:
    @patch("roboharness.evaluate.lerobot_env._try_lerobot_make_env")
    @patch("roboharness.evaluate.lerobot_env._fallback_hub_make_env")
    @patch("roboharness.evaluate.lerobot_env._add_mujoco_rendering")
    @patch("roboharness.evaluate.lerobot_env._patch_config_for_headless")
    def test_uses_fallback_when_make_env_fails(
        self,
        _mock_patch: Any,
        mock_add_render: Any,
        mock_fallback: Any,
        mock_try: Any,
    ) -> None:
        mock_try.return_value = None
        fake_env = FakeEnv()
        mock_fallback.return_value = fake_env

        env = create_native_env("lerobot/unitree-g1-mujoco")
        assert env is fake_env
        mock_add_render.assert_called_once_with(fake_env)

    @patch("roboharness.evaluate.lerobot_env._try_lerobot_make_env")
    @patch("roboharness.evaluate.lerobot_env._fallback_hub_make_env")
    @patch("roboharness.evaluate.lerobot_env._add_mujoco_rendering")
    @patch("roboharness.evaluate.lerobot_env._patch_config_for_headless")
    def test_uses_make_env_when_available(
        self,
        _mock_patch: Any,
        mock_add_render: Any,
        mock_fallback: Any,
        mock_try: Any,
    ) -> None:
        fake_env = FakeEnv()
        mock_try.return_value = fake_env

        env = create_native_env("lerobot/unitree-g1-mujoco")
        assert env is fake_env
        mock_fallback.assert_not_called()
        mock_add_render.assert_called_once_with(fake_env)


class TestAddMujocoRendering:
    def test_no_model_data_warns(self, capsys: Any) -> None:
        env = FakeEnv()
        _add_mujoco_rendering(env)
        captured = capsys.readouterr()
        assert "could not find MuJoCo model/data" in captured.out

    def test_adds_render_camera(self, monkeypatch: Any) -> None:
        mujoco = pytest.importorskip("mujoco")
        env = FakeEnv()

        fake_model = MagicMock()
        fake_model.ncam = 1
        cam_mock = MagicMock()
        cam_mock.name = "front"
        fake_model.camera = lambda i: cam_mock

        fake_data = MagicMock()
        fake_data.qpos = np.zeros(10)

        env.mj_model = fake_model
        env.mj_data = fake_data

        fake_renderer = MagicMock()
        fake_renderer.render.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_renderer_cls = MagicMock(return_value=fake_renderer)
        monkeypatch.setattr(mujoco, "Renderer", mock_renderer_cls)

        _add_mujoco_rendering(env)

        assert hasattr(env, "render_camera")
        assert hasattr(env, "cameras")
        frame = env.render_camera("front")
        assert frame.shape == (480, 640, 3)
