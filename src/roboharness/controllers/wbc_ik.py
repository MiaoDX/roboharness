"""Whole-Body Control IK solver using Pinocchio + Pink.

A thin wrapper around the Pink differential-IK library that converts
target end-effector poses into joint configurations via iterative QP
solving.  Inspired by GR00T Decoupled WBC's ``BodyIKSolver`` but
depends only on ``pin``, ``pin-pink``, and ``qpsolvers`` — no torch,
no GPU.

Usage:
    from roboharness.controllers.wbc_ik import WbcIkController

    controller = WbcIkController(
        urdf_path="robot.urdf",
        end_effector_frames=["left_hand", "right_hand"],
    )
    action = controller.compute(
        command={"left_hand": target_pose_4x4},
        state={"qpos": current_q},
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WbcIkSettings:
    """Configuration for the IK solver.

    Parameters
    ----------
    dt:
        Integration timestep for each IK iteration.
    num_iterations:
        Number of QP solve iterations per ``compute()`` call.
    position_cost:
        Weight for end-effector position tracking.
    orientation_cost:
        Weight for end-effector orientation tracking.
    posture_cost:
        Weight for regularisation toward the reference posture.
    damping:
        Levenberg-Marquardt damping for the QP.
    solver:
        QP solver backend name (e.g. ``"quadprog"``, ``"osqp"``).
    """

    dt: float = 0.05
    num_iterations: int = 3
    position_cost: float = 8.0
    orientation_cost: float = 2.0
    posture_cost: float = 1e-2
    damping: float = 1e-3
    solver: str = "quadprog"


class WbcIkController:
    """Differential-IK controller using Pinocchio + Pink.

    Implements the :class:`~roboharness.core.controller.Controller` protocol.

    Parameters
    ----------
    urdf_path:
        Path to the robot URDF file.
    end_effector_frames:
        Names of the frames to track (must exist in the URDF).
    mesh_dir:
        Optional directory containing mesh files referenced by the URDF.
    settings:
        IK solver settings.  Uses defaults if not provided.
    reference_configuration:
        Default joint configuration used as posture regularisation
        target.  If *None*, uses the zero configuration.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        end_effector_frames: list[str],
        mesh_dir: str | Path | None = None,
        settings: WbcIkSettings | None = None,
        reference_configuration: np.ndarray | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError:
            raise ImportError(
                "Pinocchio is required for WbcIkController. "
                "Install with: pip install roboharness[wbc]"
            )
        try:
            import pink
        except ImportError:
            raise ImportError(
                "Pink is required for WbcIkController. "
                "Install with: pip install roboharness[wbc]"
            )

        self._pin = pin
        self._pink = pink
        self._settings = settings or WbcIkSettings()
        self._ee_frame_names = list(end_effector_frames)

        # Load robot model
        urdf_path = Path(urdf_path)
        if mesh_dir is not None:
            self._model = pin.buildModelFromUrdf(str(urdf_path))
            self._collision_model = None
            self._visual_model = None
        else:
            self._model = pin.buildModelFromUrdf(str(urdf_path))

        self._data = self._model.createData()
        self._nq = self._model.nq

        # Build Pink Configuration
        self._configuration = pink.Configuration(
            self._model,
            self._data,
            reference_configuration
            if reference_configuration is not None
            else np.zeros(self._nq),
        )

        # Create frame tasks for each end-effector
        self._tasks: dict[str, Any] = {}
        for frame_name in self._ee_frame_names:
            # Verify frame exists
            if not self._model.existFrame(frame_name):
                available = [
                    self._model.frames[i].name for i in range(self._model.nframes)
                ]
                raise ValueError(
                    f"Frame '{frame_name}' not found in URDF. "
                    f"Available frames: {available}"
                )
            task = pink.tasks.FrameTask(
                frame_name,
                position_cost=self._settings.position_cost,
                orientation_cost=self._settings.orientation_cost,
                lm_damping=self._settings.damping,
            )
            self._tasks[frame_name] = task

        # Posture regularisation task
        self._posture_task = pink.tasks.PostureTask(cost=self._settings.posture_cost)

        # Store reference configuration
        self._q_ref = (
            reference_configuration.copy()
            if reference_configuration is not None
            else np.zeros(self._nq)
        )

    @property
    def nq(self) -> int:
        """Number of configuration variables (joint DOFs)."""
        return self._nq

    @property
    def end_effector_frames(self) -> list[str]:
        """List of tracked end-effector frame names."""
        return list(self._ee_frame_names)

    def compute(self, command: dict[str, Any], state: dict[str, Any]) -> Any:
        """Compute joint configuration from target end-effector poses.

        Parameters
        ----------
        command:
            Mapping of frame name → 4×4 homogeneous transform (np.ndarray).
            Only frames listed in ``end_effector_frames`` are accepted.
        state:
            Must contain ``"qpos"`` — the current joint configuration
            as a sequence of floats or np.ndarray.

        Returns
        -------
        np.ndarray
            Target joint configuration (shape ``(nq,)``).
        """
        pin = self._pin

        # Update current configuration from state
        qpos = np.asarray(state["qpos"], dtype=np.float64)
        self._configuration.q = qpos.copy()
        self._configuration.update()

        # Set target for each end-effector task
        for frame_name, task in self._tasks.items():
            if frame_name in command:
                target_matrix = np.asarray(command[frame_name], dtype=np.float64)
                target_se3 = pin.SE3(
                    target_matrix[:3, :3],
                    target_matrix[:3, 3],
                )
                task.set_target(target_se3)

        # Set posture task target
        self._posture_task.set_target(self._q_ref)

        # Collect all tasks
        all_tasks = list(self._tasks.values()) + [self._posture_task]

        # Iterative QP solve
        dt = self._settings.dt
        for _ in range(self._settings.num_iterations):
            velocity = self._pink.solve_ik(
                self._configuration,
                all_tasks,
                dt,
                solver=self._settings.solver,
            )
            self._configuration.integrate_inplace(velocity, dt)

        return self._configuration.q.copy()
