"""MuJoCo backend adapter.

This is the reference implementation for the SimulatorBackend protocol.
Requires: pip install roboharness[demo]

Usage:
    from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend
    from roboharness.backends.visualizer import MeshcatVisualizer

    # Default: off-screen MuJoCo renderer
    backend = MuJoCoMeshcatBackend(model_path="robot.xml", cameras=["front"])

    # With Meshcat interactive viewer
    backend = MuJoCoMeshcatBackend(
        model_path="robot.xml",
        cameras=["front"],
        visualizer="meshcat",
    )

    harness = Harness(backend, output_dir="./output")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from roboharness.backends.visualizer import (
    MeshcatVisualizer,
    MuJoCoNativeVisualizer,
    Visualizer,
)
from roboharness.core.capture import CameraView


class MuJoCoMeshcatBackend:
    """Backend adapter for MuJoCo physics with pluggable visualization.

    Implements the SimulatorBackend protocol. Visualization is delegated
    to a ``Visualizer`` instance, which can be swapped without touching
    the physics code.

    Parameters
    ----------
    model_path : str | Path | None
        Path to an MJCF/URDF file.
    xml_string : str | None
        Inline MJCF XML (alternative to model_path).
    cameras : list[str] | None
        Camera names to use (default: ``["front"]``).
    render_width, render_height : int
        Resolution for off-screen rendering.
    visualizer : Visualizer | str | None
        A ``Visualizer`` instance, or one of the shorthand strings
        ``"native"`` / ``"meshcat"``.  Defaults to ``"native"``
        (MuJoCo off-screen renderer).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        cameras: list[str] | None = None,
        render_width: int = 640,
        render_height: int = 480,
        xml_string: str | None = None,
        visualizer: Visualizer | str | None = None,
    ):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError(
                "MuJoCo is required for this backend. Install with: pip install roboharness[demo]"
            ) from exc

        if xml_string is None and model_path is None:
            raise ValueError("Either model_path or xml_string must be provided.")

        self._camera_names = cameras or ["front"]
        self._render_width = render_width
        self._render_height = render_height

        # Load MuJoCo model
        if xml_string is not None:
            self._model = mujoco.MjModel.from_xml_string(xml_string)
        else:
            self._model_path = Path(model_path)  # type: ignore[arg-type]
            self._model = mujoco.MjModel.from_xml_path(str(self._model_path))
        self._data = mujoco.MjData(self._model)

        # Initialize visualizer
        self._visualizer = self._resolve_visualizer(visualizer)

    # ------------------------------------------------------------------
    # Visualizer resolution
    # ------------------------------------------------------------------

    def _resolve_visualizer(self, viz: Visualizer | str | None) -> Visualizer:
        """Resolve a visualizer from a shorthand string or instance."""
        if viz is None or viz == "native":
            return MuJoCoNativeVisualizer(
                self._model,
                self._data,
                width=self._render_width,
                height=self._render_height,
            )
        if viz == "meshcat":
            return MeshcatVisualizer(
                self._model,
                self._data,
                width=self._render_width,
                height=self._render_height,
            )
        if isinstance(viz, str):
            raise ValueError(
                f"Unknown visualizer shorthand '{viz}'. "
                "Use 'native', 'meshcat', or pass a Visualizer instance."
            )
        # Already a Visualizer instance
        return viz

    @property
    def visualizer(self) -> Visualizer:
        """Return the active visualizer."""
        return self._visualizer

    # ------------------------------------------------------------------
    # SimulatorBackend protocol
    # ------------------------------------------------------------------

    def step(self, action: Any) -> dict[str, Any]:
        """Advance simulation by one step."""
        import mujoco

        if action is not None:
            np.copyto(self._data.ctrl, np.asarray(action, dtype=np.float64))

        mujoco.mj_step(self._model, self._data)
        self._visualizer.sync()

        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state.

        Returns numpy arrays for ``qpos``, ``qvel``, and ``ctrl`` so callers
        can use them directly in numeric computations without conversion.
        JSON serialization is handled by :class:`~roboharness._utils.NumpyEncoder`.
        """
        return {
            "time": float(self._data.time),
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "ctrl": self._data.ctrl.copy(),
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
        self._visualizer.sync()

    def capture_camera(self, camera_name: str) -> CameraView:
        """Capture RGB and depth from a named camera via the active visualizer."""
        return self._visualizer.capture_camera(camera_name)

    def get_sim_time(self) -> float:
        """Get current simulation time."""
        return float(self._data.time)

    def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state."""
        import mujoco

        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._visualizer.sync()
        return self.get_state()
