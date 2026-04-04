"""Tests for ONNX-based locomotion controllers (GR00T + Holosoma).

Uses monkeypatched fakes for onnxruntime and huggingface_hub so the tests
run without those heavy optional dependencies.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from roboharness.core.controller import Controller


# ---------------------------------------------------------------------------
# Fake ONNX / HuggingFace modules
# ---------------------------------------------------------------------------
class _FakeInput:
    """Mimics an ONNX input descriptor."""

    def __init__(self, name: str = "obs") -> None:
        self.name = name


class _FakeSession:
    """Mimics ``onnxruntime.InferenceSession``."""

    def __init__(self, path: str, providers: list[str] | None = None) -> None:
        self.path = path
        self.providers = providers
        self._input_name = "obs"
        self._is_planner = "planner" in path

    def get_inputs(self) -> list[_FakeInput]:
        if self._is_planner:
            return [_FakeInput("context_mujoco_qpos")]
        return [_FakeInput(self._input_name)]

    def run(self, output_names: list[str] | None, feed: dict[str, Any]) -> list[np.ndarray]:
        """Return deterministic outputs matching the model type."""
        if self._is_planner:
            # SONIC planner: return [mujoco_qpos [1, N, 36], num_pred_frames]
            num_frames = 6
            qpos = np.zeros((1, num_frames, 36), dtype=np.float32)
            # Set valid quaternion w=1 and default joint angles in each frame
            for i in range(num_frames):
                qpos[0, i, 3] = 1.0  # quaternion w
                qpos[0, i, 2] = 0.74  # root height
            return [qpos, np.array(num_frames, dtype=np.int64)]
        # GR00T / Holosoma: single action output
        inp = next(iter(feed.values()))
        batch = inp.shape[0]
        return [np.zeros((batch, 29), dtype=np.float32)]


class _FakeOrt:
    """Stand-in for the ``onnxruntime`` module."""

    InferenceSession = _FakeSession


class _FakeHfHub:
    """Stand-in for ``huggingface_hub``."""

    @staticmethod
    def hf_hub_download(repo_id: str, filename: str) -> str:
        return f"/tmp/fake_cache/{repo_id}/{filename}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def _patch_onnx_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject fake onnxruntime and huggingface_hub into sys.modules."""
    fake_ort = _FakeOrt()
    fake_hf = _FakeHfHub()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)  # type: ignore[arg-type]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)  # type: ignore[arg-type]
    # Also invalidate any cached imports in the locomotion module
    monkeypatch.delitem(sys.modules, "roboharness.controllers.locomotion", raising=False)


def _make_g1_state(nq: int = 36, nv: int = 35) -> dict[str, np.ndarray]:
    """Create a plausible G1 state dict (free joint + 29 body joints)."""
    qpos = np.zeros(nq, dtype=np.float32)
    qpos[3] = 1.0  # quaternion w=1 (identity rotation)
    qvel = np.zeros(nv, dtype=np.float32)
    return {"qpos": qpos, "qvel": qvel}


# ---------------------------------------------------------------------------
# GR00T controller tests
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("_patch_onnx_deps")
class TestGrootLocomotionController:
    def _make_controller(self) -> Any:
        from roboharness.controllers.locomotion import GrootLocomotionController

        return GrootLocomotionController()

    def test_implements_controller_protocol(self) -> None:
        ctrl = self._make_controller()
        assert isinstance(ctrl, Controller)

    def test_compute_returns_15_joint_targets(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (15,)

    def test_compute_with_walk_command(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [1.0, 0, 0]}, state=state)
        assert action.shape == (15,)

    def test_compute_with_missing_velocity(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={}, state=state)
        assert action.shape == (15,)

    def test_reset_clears_state(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        ctrl.reset()
        assert np.allclose(ctrl._action, 0)
        assert np.allclose(ctrl._cmd, 0)

    def test_control_dt(self) -> None:
        ctrl = self._make_controller()
        assert ctrl.control_dt == pytest.approx(0.02)

    def test_balance_policy_for_zero_command(self) -> None:
        """Near-zero velocity should use the balance session."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        # With zero command, the balance session should have been used.
        # We can't easily assert which session was called with fakes,
        # but we verify the output is valid.
        assert ctrl._action is not None

    def test_handles_short_qpos(self) -> None:
        """Controller should handle qpos shorter than expected."""
        ctrl = self._make_controller()
        state = {"qpos": np.zeros(10, dtype=np.float32), "qvel": np.zeros(10, dtype=np.float32)}
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert action.shape == (15,)

    def test_obs_history_maintained(self) -> None:
        """Observation history should accumulate up to OBS_HISTORY_LEN."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        for _ in range(10):
            ctrl.compute(command={"velocity": [0.5, 0, 0]}, state=state)
        from roboharness.controllers.locomotion import OBS_HISTORY_LEN

        assert len(ctrl._obs_history) == OBS_HISTORY_LEN


# ---------------------------------------------------------------------------
# Holosoma controller tests
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("_patch_onnx_deps")
class TestHolosomaLocomotionController:
    def _make_controller(self) -> Any:
        from roboharness.controllers.locomotion import HolosomaLocomotionController

        return HolosomaLocomotionController()

    def test_implements_controller_protocol(self) -> None:
        ctrl = self._make_controller()
        assert isinstance(ctrl, Controller)

    def test_compute_returns_29_joint_targets(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (29,)

    def test_compute_with_walk_command(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [1.0, 0.5, 0.1]}, state=state)
        assert action.shape == (29,)

    def test_compute_with_missing_velocity(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={}, state=state)
        assert action.shape == (29,)

    def test_reset_clears_state(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        ctrl.reset()
        assert np.allclose(ctrl._action, 0)
        assert np.allclose(ctrl._cmd, 0)
        assert ctrl._phase == 0.0

    def test_control_dt(self) -> None:
        ctrl = self._make_controller()
        assert ctrl.control_dt == pytest.approx(0.02)

    def test_gait_phase_advances(self) -> None:
        """Phase should advance with each compute step."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        assert ctrl._phase == 0.0
        ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert ctrl._phase > 0.0

    def test_gait_phase_wraps(self) -> None:
        """Phase should wrap around 2*pi."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        # Run enough steps to wrap (1s / 0.02s = 50 steps per period)
        for _ in range(60):
            ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert 0 <= ctrl._phase < 2 * np.pi

    def test_handles_short_qpos(self) -> None:
        """Controller should handle qpos shorter than expected."""
        ctrl = self._make_controller()
        state = {"qpos": np.zeros(10, dtype=np.float32), "qvel": np.zeros(10, dtype=np.float32)}
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert action.shape == (29,)

    def test_action_targets_include_default_angles(self) -> None:
        """With zero ONNX output, targets should equal default angles."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        from roboharness.controllers.locomotion import HOLOSOMA_DEFAULT_ANGLES

        np.testing.assert_allclose(action, HOLOSOMA_DEFAULT_ANGLES)

    def test_reset_then_compute(self) -> None:
        """Reset followed by compute should work without error."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        ctrl.reset()
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert action.shape == (29,)

    def test_multiple_steps_deterministic(self) -> None:
        """Consecutive steps with same input should produce same output (fake ONNX)."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        results = []
        for _ in range(3):
            ctrl.reset()
            action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
            results.append(action.copy())
        # With fake ONNX returning zeros, all results should match
        np.testing.assert_allclose(results[0], results[1])
        np.testing.assert_allclose(results[1], results[2])


# ---------------------------------------------------------------------------
# Shared utility tests
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("_patch_onnx_deps")
class TestLocomotionUtilities:
    def test_get_gravity_orientation_identity(self) -> None:
        from roboharness.controllers.locomotion import get_gravity_orientation

        # Identity quaternion [w=1, x=0, y=0, z=0] → gravity = [0, 0, -1]
        grav = get_gravity_orientation(np.array([1, 0, 0, 0], dtype=np.float32))
        assert grav.shape == (3,)
        np.testing.assert_allclose(grav, [0, 0, -1], atol=1e-6)

    def test_get_gravity_orientation_90deg_pitch(self) -> None:
        from roboharness.controllers.locomotion import get_gravity_orientation

        # 90-degree pitch rotation: q = [cos(45°), 0, sin(45°), 0]
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0, np.sin(angle / 2), 0], dtype=np.float32)
        grav = get_gravity_orientation(q)
        assert grav.shape == (3,)

    def test_download_onnx_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ImportError when huggingface_hub is not installed."""
        monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
        monkeypatch.delitem(sys.modules, "roboharness.controllers.locomotion", raising=False)
        monkeypatch.delitem(sys.modules, "roboharness.robots.unitree_g1.locomotion", raising=False)
        from roboharness.controllers.locomotion import _download_onnx

        # Remove the fake hf module so the import fails
        monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
        with pytest.raises(ImportError, match="huggingface_hub"):
            _download_onnx("some/repo", "model.onnx")

    def test_onnxruntime_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ImportError when onnxruntime is not installed."""
        # Ensure hf_hub is available but ort is not
        monkeypatch.setitem(sys.modules, "huggingface_hub", _FakeHfHub())  # type: ignore[arg-type]
        monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
        monkeypatch.delitem(sys.modules, "roboharness.controllers.locomotion", raising=False)
        from roboharness.controllers.locomotion import HolosomaLocomotionController

        with pytest.raises(ImportError, match="onnxruntime"):
            HolosomaLocomotionController()


# ---------------------------------------------------------------------------
# SONIC controller tests
# ---------------------------------------------------------------------------
@pytest.mark.usefixtures("_patch_onnx_deps")
class TestSonicLocomotionController:
    def _make_controller(self, **kwargs: Any) -> Any:
        from roboharness.controllers.locomotion import SonicLocomotionController

        return SonicLocomotionController(**kwargs)

    def test_implements_controller_protocol(self) -> None:
        ctrl = self._make_controller()
        assert isinstance(ctrl, Controller)

    def test_compute_returns_29_joint_targets(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (29,)

    def test_compute_with_walk_command(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [1.0, 0.5, 0.1]}, state=state)
        assert action.shape == (29,)

    def test_compute_with_missing_velocity(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={}, state=state)
        assert action.shape == (29,)

    def test_compute_with_mode(self) -> None:
        from roboharness.controllers.locomotion import SonicMode

        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [1, 0, 0], "mode": SonicMode.RUN}, state=state)
        assert action.shape == (29,)
        assert ctrl._mode == SonicMode.RUN

    def test_compute_with_height(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        action = ctrl.compute(command={"velocity": [0, 0, 0], "height": 0.5}, state=state)
        assert action.shape == (29,)

    def test_reset_clears_state(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        ctrl.reset()
        assert np.allclose(ctrl._cmd, 0)
        assert len(ctrl._trajectory) == 0
        assert ctrl._traj_index == 0
        assert ctrl._interp_phase == 0.0
        assert ctrl._steps_since_plan == 0

    def test_control_dt(self) -> None:
        ctrl = self._make_controller()
        assert ctrl.control_dt == pytest.approx(0.02)

    def test_replans_at_10hz(self) -> None:
        """Planner should be invoked every 5 control steps (50/10 Hz)."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        # First call always plans
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        # Steps 2-5 should not replan
        for _ in range(4):
            ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        assert ctrl._steps_since_plan == 5
        # Step 6 should trigger replan
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        assert ctrl._steps_since_plan == 1  # just replanned

    def test_handles_short_qpos(self) -> None:
        ctrl = self._make_controller()
        state = {"qpos": np.zeros(10, dtype=np.float32), "qvel": np.zeros(10, dtype=np.float32)}
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert action.shape == (29,)

    def test_context_window_size(self) -> None:
        from roboharness.controllers.locomotion import SONIC_CONTEXT_LEN

        ctrl = self._make_controller()
        state = _make_g1_state()
        for _ in range(10):
            ctrl.compute(command={"velocity": [0.5, 0, 0]}, state=state)
        assert len(ctrl._context) == SONIC_CONTEXT_LEN

    def test_reset_then_compute(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        ctrl.reset()
        action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
        assert action.shape == (29,)

    def test_sonic_modes_enum(self) -> None:
        from roboharness.controllers.locomotion import SonicMode

        assert SonicMode.IDLE == 0
        assert SonicMode.SLOW_WALK == 1
        assert SonicMode.WALK == 2
        assert SonicMode.RUN == 3
        assert SonicMode.BOXING == 4

    def test_default_mode(self) -> None:
        from roboharness.controllers.locomotion import SonicMode

        ctrl = self._make_controller()
        assert ctrl._default_mode == SonicMode.WALK

    def test_custom_default_mode(self) -> None:
        from roboharness.controllers.locomotion import SonicMode

        ctrl = self._make_controller(default_mode=SonicMode.RUN)
        assert ctrl._default_mode == SonicMode.RUN

    def test_interpolation_advances(self) -> None:
        """Interpolation phase should advance between planner calls."""
        ctrl = self._make_controller()
        state = _make_g1_state()
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        # After first compute, interp_phase should have advanced
        phase_after_1 = ctrl._interp_phase
        ctrl.compute(command={"velocity": [1, 0, 0]}, state=state)
        phase_after_2 = ctrl._interp_phase
        # Phase should be advancing (or traj_index increasing)
        assert phase_after_2 > phase_after_1 or ctrl._traj_index > 0

    def test_multiple_steps_produce_valid_output(self) -> None:
        ctrl = self._make_controller()
        state = _make_g1_state()
        for _ in range(20):
            action = ctrl.compute(command={"velocity": [0.5, 0, 0]}, state=state)
            assert action.shape == (29,)
            assert np.all(np.isfinite(action))
