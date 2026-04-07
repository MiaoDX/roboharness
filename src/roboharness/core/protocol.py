"""Semantic task protocols — define meaningful capture phases for robot tasks.

Instead of capturing at arbitrary simulation step counts, a TaskProtocol defines
semantically meaningful phases (e.g. "pre_grasp", "approach", "grasp", "lift")
that map to the natural structure of a task.

Usage:
    from roboharness.core.protocol import GRASP_PROTOCOL, TaskProtocol, TaskPhase

    # Use a built-in protocol
    harness.load_protocol(GRASP_PROTOCOL)

    # Use a subset of phases
    harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "grasp", "lift"])

    # Define a custom protocol
    my_protocol = TaskProtocol(
        name="assembly",
        description="Multi-step assembly task",
        phases=[
            TaskPhase("pick", "Pick up the part"),
            TaskPhase("align", "Align part with target"),
            TaskPhase("insert", "Insert part into slot"),
        ],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskPhase:
    """A single semantic phase in a task protocol.

    Attributes:
        name: Short identifier (used as checkpoint name).
        description: Human-readable description of what happens in this phase.
        cameras: Camera names to capture at this phase. Defaults to ["front"].
        metadata: Arbitrary metadata for this phase (e.g. expected duration, tolerances).
    """

    name: str
    description: str = ""
    cameras: list[str] = field(default_factory=lambda: ["front"])
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskProtocol:
    """A semantic protocol defining the phases of a robot task.

    A TaskProtocol is a reusable template that describes the natural stages
    of a task type. Loading a protocol into a Harness automatically registers
    the appropriate checkpoints with their camera configurations.

    Attributes:
        name: Protocol identifier (e.g. "grasp", "locomotion").
        description: Human-readable description of the task type.
        phases: Ordered list of semantic phases.
    """

    name: str
    description: str = ""
    phases: list[TaskPhase] = field(default_factory=list)

    def phase_names(self) -> list[str]:
        """Return ordered list of phase names."""
        return [p.name for p in self.phases]

    def get_phase(self, name: str) -> TaskPhase:
        """Look up a phase by name. Raises KeyError if not found."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        raise KeyError(
            f"Phase '{name}' not in protocol '{self.name}'. Available: {self.phase_names()}"
        )

    def select(self, phase_names: list[str]) -> TaskProtocol:
        """Return a new protocol containing only the specified phases (in order).

        Raises KeyError if any name is not in this protocol.
        """
        selected = []
        for name in phase_names:
            selected.append(self.get_phase(name))
        return TaskProtocol(
            name=self.name,
            description=self.description,
            phases=selected,
        )


# ---------------------------------------------------------------------------
# Built-in protocols
# ---------------------------------------------------------------------------

GRASP_PROTOCOL = TaskProtocol(
    name="grasp",
    description="Pick-and-place grasping task",
    phases=[
        TaskPhase("plan", "Plan grasp trajectory and visualize target path"),
        TaskPhase("pre_grasp", "Move to pre-grasp pose above the object"),
        TaskPhase("approach", "Approach the object along the planned path"),
        TaskPhase("grasp", "Close gripper on the object"),
        TaskPhase("lift", "Lift the grasped object"),
        TaskPhase("place", "Place the object at the target location"),
        TaskPhase("home", "Return to home position"),
    ],
)

LOCOMOTION_PROTOCOL = TaskProtocol(
    name="locomotion",
    description="Legged locomotion task",
    phases=[
        TaskPhase("initial", "Initial standing pose"),
        TaskPhase("accelerate", "Accelerate to target velocity"),
        TaskPhase("steady", "Steady-state locomotion"),
        TaskPhase("decelerate", "Decelerate to stop"),
        TaskPhase("terminal", "Final standing pose"),
    ],
)

LOCO_MANIPULATION_PROTOCOL = TaskProtocol(
    name="loco_manipulation",
    description="Mobile manipulation — locomotion combined with grasping",
    phases=[
        # Navigation
        TaskPhase("navigate", "Walk to the target object area"),
        # Manipulation
        TaskPhase("pre_grasp", "Position arm for grasping"),
        TaskPhase("grasp", "Grasp the object"),
        # Transport
        TaskPhase("transport", "Walk while holding the object"),
        # Placement
        TaskPhase("place", "Place the object at the destination"),
        TaskPhase("retreat", "Step back and return to ready pose"),
    ],
)

REACH_PROTOCOL = TaskProtocol(
    name="reach",
    description="End-effector reaching / pointing task",
    phases=[
        TaskPhase("rest", "Initial rest position"),
        TaskPhase("reach", "Reach toward target"),
        TaskPhase("hold", "Hold position at target"),
        TaskPhase("retract", "Retract to rest position"),
    ],
)

DANCE_PROTOCOL = TaskProtocol(
    name="dance",
    description="Rhythmic motion / dance routine",
    phases=[
        TaskPhase("ready", "Initial ready pose"),
        TaskPhase("sequence", "Main dance sequence"),
        TaskPhase("finale", "Final pose"),
    ],
)

# Registry of all built-in protocols for discovery
BUILTIN_PROTOCOLS: dict[str, TaskProtocol] = {
    p.name: p
    for p in [
        GRASP_PROTOCOL,
        LOCOMOTION_PROTOCOL,
        LOCO_MANIPULATION_PROTOCOL,
        REACH_PROTOCOL,
        DANCE_PROTOCOL,
    ]
}
