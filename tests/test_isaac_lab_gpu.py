"""GPU tests for Isaac Lab compatibility — validates RobotHarnessWrapper with CUDA tensors.

These tests require an NVIDIA GPU with CUDA support. They verify that the wrapper
correctly handles PyTorch tensors living on GPU memory, which is how Isaac Lab
environments actually operate (as opposed to the CPU-tensor mocks in
test_isaac_lab_compat.py).

Key behaviors validated:
  - Wrapper passes through CUDA tensor observations without moving them to CPU
  - Checkpoint capture correctly converts GPU tensors (rewards, render frames)
  - State serialization handles GPU tensor metadata (shape, dtype)
  - _to_float / _to_numpy_rgb work with CUDA tensors
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
gym = pytest.importorskip("gymnasium", reason="gymnasium not installed")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

from gymnasium import spaces  # noqa: E402

from roboharness.wrappers import RobotHarnessWrapper  # noqa: E402
from roboharness.wrappers.gymnasium_wrapper import (  # noqa: E402
    MultiCameraCapability,
    _to_float,
    _to_numpy_rgb,
)

# ---------------------------------------------------------------------------
# Mock envs that return CUDA tensors (mimicking real Isaac Lab behavior)
# ---------------------------------------------------------------------------


class CudaIsaacLabEnv(gym.Env):
    """Mock Isaac Lab env that returns CUDA tensors for obs/rewards.

    Real Isaac Lab environments keep all data on GPU. This mock replicates
    that behavior for CI validation.
    """

    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__()
        self.num_envs = num_envs
        self.render_mode = render_mode
        self._step_count = 0
        self._device = torch.device("cuda:0")

        obs_dim = 12
        act_dim = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._step_count = 0
        obs = torch.zeros(self.num_envs, *self.observation_space.shape, device=self._device)
        return obs, {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self._step_count += 1
        obs = torch.randn(self.num_envs, *self.observation_space.shape, device=self._device)
        reward = torch.tensor([0.42] * self.num_envs, device=self._device)
        terminated = torch.tensor([False] * self.num_envs, device=self._device)
        truncated = torch.tensor([False] * self.num_envs, device=self._device)
        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray:
        return np.zeros((480, 640, 3), dtype=np.uint8)


class CudaIsaacLabDictObsEnv(CudaIsaacLabEnv):
    """Mock Isaac Lab env with dict observation space, all on CUDA."""

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__(num_envs=num_envs, render_mode=render_mode)
        self.observation_space = spaces.Dict(
            {
                "policy": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
            }
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        self._step_count = 0
        obs = {"policy": torch.zeros(self.num_envs, 12, device=self._device)}
        return obs, {}

    def step(self, action: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self._step_count += 1
        obs = {"policy": torch.randn(self.num_envs, 12, device=self._device)}
        reward = torch.tensor([0.42] * self.num_envs, device=self._device)
        terminated = torch.tensor([False] * self.num_envs, device=self._device)
        truncated = torch.tensor([False] * self.num_envs, device=self._device)
        return obs, reward, terminated, truncated, {}


class CudaRenderEnv(CudaIsaacLabEnv):
    """Mock env where render() returns a CUDA tensor (e.g., from a GPU renderer)."""

    def render(self) -> Any:
        # Some GPU renderers return tensors directly
        return torch.zeros(480, 640, 3, dtype=torch.uint8, device=self._device)


class CudaTiledCameraEnv(CudaIsaacLabEnv):
    """Mock Isaac Lab env with TiledCamera returning CUDA tensor RGB data."""

    def __init__(self, num_envs: int = 1, render_mode: str = "rgb_array"):
        super().__init__(num_envs=num_envs, render_mode=render_mode)
        device = self._device

        class _CameraData:
            def __init__(self) -> None:
                # Isaac Lab TiledCamera stores images as GPU tensors
                self.output = {"rgb": torch.zeros(480, 640, 3, dtype=torch.uint8, device=device)}

        class _TiledCamera:
            __name__ = "TiledCamera"

            def __init__(self) -> None:
                self.data = _CameraData()

        class _Scene(dict):
            pass

        self.scene = _Scene({"tiled_camera": _TiledCamera()})


# ---------------------------------------------------------------------------
# Unit tests for helper functions with CUDA tensors
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_to_float_cuda_scalar() -> None:
    """_to_float should extract a Python float from a CUDA scalar tensor."""
    t = torch.tensor(3.14, device="cuda:0")
    assert _to_float(t) == pytest.approx(3.14, abs=1e-5)


@pytest.mark.gpu
def test_to_float_cuda_multi_element() -> None:
    """_to_float should return the mean for multi-element CUDA tensors."""
    t = torch.tensor([1.0, 2.0, 3.0], device="cuda:0")
    assert _to_float(t) == pytest.approx(2.0, abs=1e-5)


@pytest.mark.gpu
def test_to_numpy_rgb_cuda_uint8() -> None:
    """_to_numpy_rgb should move a CUDA uint8 tensor to CPU numpy."""
    t = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8, device="cuda:0")
    result = _to_numpy_rgb(t)
    assert isinstance(result, np.ndarray)
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.uint8


@pytest.mark.gpu
def test_to_numpy_rgb_cuda_float_normalized() -> None:
    """_to_numpy_rgb should scale [0,1] CUDA float tensors to uint8."""
    t = torch.rand(480, 640, 3, device="cuda:0")
    result = _to_numpy_rgb(t)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.max() <= 255


# ---------------------------------------------------------------------------
# Wrapper integration tests with CUDA tensors
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_wrapper_passthrough_cuda_obs(tmp_path) -> None:
    """Wrapper must not move CUDA observations to CPU — they stay on GPU."""
    env = CudaIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 3}],
        output_dir=tmp_path,
    )
    obs, _ = wrapped.reset()
    assert obs.is_cuda, "Reset obs should remain on CUDA"

    for _ in range(3):
        obs, _reward, _, _, _ = wrapped.step(torch.zeros(1, 7, device="cuda:0"))

    assert obs.is_cuda, "Step obs should remain on CUDA"


@pytest.mark.gpu
def test_checkpoint_state_with_cuda_reward(tmp_path) -> None:
    """Checkpoint state.json must correctly serialize CUDA tensor rewards."""
    env = CudaIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 2}],
        output_dir=tmp_path,
        task_name="gpu_test",
    )
    wrapped.reset()
    for _ in range(2):
        _, _, _, _, info = wrapped.step(torch.zeros(1, 7, device="cuda:0"))

    assert "checkpoint" in info
    state_path = tmp_path / "gpu_test" / "trial_001" / "cp1" / "state.json"
    state = json.loads(state_path.read_text())
    assert state["checkpoint"] == "cp1"
    assert state["step"] == 2
    assert state["reward"] == pytest.approx(0.42, abs=1e-5)


@pytest.mark.gpu
def test_checkpoint_state_records_cuda_obs_metadata(tmp_path) -> None:
    """state.json should record obs shape/dtype from CUDA tensor observations."""
    env = CudaIsaacLabEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
        task_name="gpu_meta",
    )
    wrapped.reset()
    _, _, _, _, _info = wrapped.step(torch.zeros(1, 7, device="cuda:0"))

    state_path = tmp_path / "gpu_meta" / "trial_001" / "cp1" / "state.json"
    state = json.loads(state_path.read_text())
    assert state["obs_shape"] == [1, 12]
    assert "float32" in state["obs_dtype"]


@pytest.mark.gpu
def test_wrapper_dict_obs_cuda(tmp_path) -> None:
    """Wrapper should handle dict observations with CUDA tensor values."""
    env = CudaIsaacLabDictObsEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
        task_name="gpu_dict",
    )
    obs, _ = wrapped.reset()
    assert isinstance(obs, dict)
    assert obs["policy"].is_cuda

    obs, _, _, _, info = wrapped.step(torch.zeros(1, 7, device="cuda:0"))
    assert obs["policy"].is_cuda
    assert "checkpoint" in info

    state_path = tmp_path / "gpu_dict" / "trial_001" / "cp1" / "state.json"
    state = json.loads(state_path.read_text())
    assert "obs_keys" in state
    assert "policy" in state["obs_keys"]


@pytest.mark.gpu
def test_wrapper_multi_env_cuda(tmp_path) -> None:
    """Wrapper should work with vectorized CUDA envs (num_envs > 1)."""
    env = CudaIsaacLabEnv(num_envs=4)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
    )
    obs, _ = wrapped.reset()
    assert obs.shape[0] == 4
    assert obs.is_cuda

    obs, _, _, _, info = wrapped.step(torch.zeros(4, 7, device="cuda:0"))
    assert obs.shape[0] == 4
    assert obs.is_cuda
    assert "checkpoint" in info


@pytest.mark.gpu
def test_render_cuda_tensor_captured(tmp_path) -> None:
    """Wrapper should handle render() returning a CUDA tensor."""
    env = CudaRenderEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
        task_name="gpu_render",
    )
    wrapped.reset()
    _, _, _, _, info = wrapped.step(torch.zeros(1, 7, device="cuda:0"))

    assert "checkpoint" in info
    capture_dir = tmp_path / "gpu_render" / "trial_001" / "cp1"
    assert (capture_dir / "default_rgb.png").exists()


@pytest.mark.gpu
def test_isaac_tiled_camera_cuda(tmp_path) -> None:
    """Isaac Lab TiledCamera with CUDA tensor data should be captured."""
    env = CudaTiledCameraEnv(num_envs=1)
    wrapped = RobotHarnessWrapper(
        env,
        cameras=["tiled_camera"],
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
        task_name="gpu_tiled",
    )
    assert wrapped.camera_capability == MultiCameraCapability.ISAAC_TILED

    wrapped.reset()
    _, _, _, _, info = wrapped.step(torch.zeros(1, 7, device="cuda:0"))

    assert "checkpoint" in info
    files = info["checkpoint"]["files"]
    assert "tiled_camera_rgb" in files

    meta_path = tmp_path / "gpu_tiled" / "trial_001" / "cp1" / "metadata.json"
    meta = json.loads(meta_path.read_text())
    assert "tiled_camera" in meta["cameras"]
    assert meta["camera_capability"] == "isaac_tiled"


@pytest.mark.gpu
def test_multi_reward_cuda_mean(tmp_path) -> None:
    """Multi-env CUDA rewards should be averaged in state.json."""
    env = CudaIsaacLabEnv(num_envs=4)
    wrapped = RobotHarnessWrapper(
        env,
        checkpoints=[{"name": "cp1", "step": 1}],
        output_dir=tmp_path,
        task_name="gpu_multi_reward",
    )
    wrapped.reset()
    _, _, _, _, _info = wrapped.step(torch.zeros(4, 7, device="cuda:0"))

    state_path = tmp_path / "gpu_multi_reward" / "trial_001" / "cp1" / "state.json"
    state = json.loads(state_path.read_text())
    # All rewards are 0.42, so mean should be 0.42
    assert state["reward"] == pytest.approx(0.42, abs=1e-5)
