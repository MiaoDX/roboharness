"""MuJoCo + Meshcat backend adapter.

This is the reference implementation for the SimulatorBackend protocol.
Requires: pip install robot-harness[mujoco]

Usage:
    from robot_harness.backends.mujoco_meshcat import MuJoCoMeshcatBackend

    backend = MuJoCoMeshcatBackend(
        model_path="robot.xml",
        cameras=["front", "side", "top"],
    )
    harness = Harness(backend, output_dir="./output")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot_harness.core.capture import CameraView


class MuJoCoMeshcatBackend:
    """Backend adapter for MuJoCo physics + Meshcat visualization.

    Implements the SimulatorBackend protocol for the most common
    robotics simulation setup.
    """

    def __init__(
        self,
        model_path: str | Path,
        cameras: list[str] | None = None,
        render_width: int = 640,
        render_height: int = 480,
    ):
        try:
            import mujoco
        except ImportError:
            raise ImportError(
                "MuJoCo is required for this backend. "
                "Install with: pip install robot-harness[mujoco]"
            )

        self._model_path = Path(model_path)
        self._camera_names = cameras or ["front"]
        self._render_width = render_width
        self._render_height = render_height

        # Load MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(
            self._model, height=self._render_height, width=self._render_width
        )

        # Meshcat visualizer (optional, for interactive debugging)
        self._meshcat_vis = None

    def step(self, action: Any) -> dict[str, Any]:
        """Advance simulation by one step."""
        import mujoco

        if action is not None:
            np.copyto(self._data.ctrl, np.asarray(action, dtype=np.float64))

        mujoco.mj_step(self._model, self._data)

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state."""
        return {
            "time": float(self._data.time),
            "qpos": self._data.qpos.copy().tolist(),
            "qvel": self._data.qvel.copy().tolist(),
            "ctrl": self._data.ctrl.copy().tolist(),
        }

    def save_state(self) -> dict[str, Any]:
        """Save full MuJoCo state for later restoration."""
        import mujoco

        state_size = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        state = np.empty(state_size, dtype=np.float64)
        mujoco.mj_getState(self._model, self._data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        return {"mujoco_state": state, "time": float(self._data.time)}

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore MuJoCo state from a previous snapshot."""
        import mujoco

        mujoco.mj_setState(
            self._model,
            self._data,
            state["mujoco_state"],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )

    def capture_camera(self, camera_name: str) -> CameraView:
        """Capture RGB and depth from a named camera."""

        self._renderer.update_scene(self._data, camera=camera_name)

        # RGB
        rgb = self._renderer.render().copy()

        # Depth
        self._renderer.enable_depth_rendering()
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        return CameraView(name=camera_name, rgb=rgb, depth=depth)

    def get_sim_time(self) -> float:
        """Get current simulation time."""
        return float(self._data.time)

    def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state."""
        import mujoco

        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        return self.get_state()

    def setup_meshcat(self) -> None:
        """Initialize Meshcat visualizer for interactive debugging."""
        try:
            import meshcat
        except ImportError:
            raise ImportError(
                "Meshcat is required for visualization. "
                "Install with: pip install robot-harness[mujoco]"
            )
        self._meshcat_vis = meshcat.Visualizer()
