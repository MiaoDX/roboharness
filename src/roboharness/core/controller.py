"""Controller protocol for high-level command → low-level action conversion.

Controllers sit between the Agent and the SimulatorBackend, translating
high-level commands (e.g. target end-effector poses) into low-level
actions (e.g. joint positions or torques) that the backend can execute.

Usage:
    controller = WbcIkController(urdf_path="robot.urdf", ...)
    state = harness.get_state()
    action = controller.compute(
        command={"target_ee_pose": target_pose},
        state=state,
    )
    harness.step(action)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Controller(Protocol):
    """Protocol that controllers must implement.

    A controller converts high-level commands (target poses, velocity
    commands, etc.) into low-level actions (joint angles, torques, etc.)
    that a SimulatorBackend can execute via ``step()``.

    The pipeline is::

        Agent → command → Controller.compute() → action → Backend.step()
    """

    def compute(self, command: dict[str, Any], state: dict[str, Any]) -> Any:
        """Compute low-level action from a high-level command.

        Parameters
        ----------
        command:
            High-level command from the agent. The schema is
            controller-specific. For IK controllers this typically
            contains target end-effector poses.
        state:
            Current simulation state as returned by
            ``SimulatorBackend.get_state()``.

        Returns
        -------
        Any
            Action suitable for ``SimulatorBackend.step(action)``.
        """
        ...
