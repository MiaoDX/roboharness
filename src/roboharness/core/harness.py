"""Core Harness — the main interface for AI Coding Agents."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from roboharness.core.capture import CameraView, CaptureResult
from roboharness.core.checkpoint import Checkpoint, CheckpointStore
from roboharness.core.protocol import TaskProtocol
from roboharness.core.rerun_logger import RerunCaptureLogger


@runtime_checkable
class SimulatorBackend(Protocol):
    """Protocol that simulator adapters must implement.

    Each simulator (MuJoCo, Isaac Lab, ManiSkill, etc.) provides an adapter
    that implements this interface. The Harness orchestrates the loop.
    """

    def step(self, action: Any) -> dict[str, Any]:
        """Advance simulation by one step. Returns state dict."""
        ...

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state (joint angles, poses, contacts, etc.)."""
        ...

    def save_state(self) -> dict[str, Any]:
        """Save full simulation state for later restoration."""
        ...

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore simulation to a previously saved state."""
        ...

    def capture_camera(self, camera_name: str) -> CameraView:
        """Capture RGB (and optionally depth) from a named camera."""
        ...

    def get_sim_time(self) -> float:
        """Get current simulation time in seconds."""
        ...

    def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state. Returns initial observation."""
        ...


class Harness:
    """Visual Testing Harness for AI Coding Agents.

    The Harness manages the simulation loop, checkpoints, and multi-view
    captures. It provides a simple interface for agents to control the
    simulation, inspect visual output, and iterate on their approach.

    Usage:
        backend = MuJoCoBackend(model_path="robot.xml")
        harness = Harness(backend, output_dir="./harness_output")

        harness.add_checkpoint("pre_grasp", cameras=["front", "side", "top"])
        harness.add_checkpoint("contact", cameras=["front", "wrist"])
        harness.add_checkpoint("lift", cameras=["front", "side", "top"])

        harness.reset()
        while not done:
            result = harness.run_to_next_checkpoint(actions)
            # Agent inspects result.views and result.state
            # Agent decides next action
    """

    def __init__(
        self,
        backend: SimulatorBackend,
        output_dir: str | Path = "./harness_output",
        task_name: str = "default",
        enable_rerun: bool = False,
        rerun_app_id: str = "roboharness",
    ):
        self.backend = backend
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self._checkpoints: list[Checkpoint] = []
        self._current_checkpoint_idx: int = 0
        self._step_count: int = 0
        self._trial_count: int = 0
        self._checkpoint_store = CheckpointStore(self.output_dir / "snapshots")
        self._rerun_logger = RerunCaptureLogger(app_id=rerun_app_id) if enable_rerun else None
        self._active_protocol: TaskProtocol | None = None

    # ---- Checkpoint management ----

    def add_checkpoint(
        self,
        name: str,
        cameras: list[str] | None = None,
        trigger_step: int | None = None,
        trigger_condition: str | None = None,
        **metadata: Any,
    ) -> None:
        """Register a checkpoint in the simulation timeline."""
        cp = Checkpoint(
            name=name,
            cameras=cameras or ["front"],
            trigger_step=trigger_step,
            trigger_condition=trigger_condition,
            metadata=metadata,
        )
        self._checkpoints.append(cp)

    def load_protocol(
        self,
        protocol: TaskProtocol,
        phases: list[str] | None = None,
    ) -> None:
        """Load a semantic task protocol, registering its phases as checkpoints.

        This replaces any previously registered checkpoints. Each phase in the
        protocol becomes a checkpoint with the phase's camera configuration
        and metadata.

        Args:
            protocol: The task protocol to load.
            phases: Optional subset of phase names to use. If None, all phases
                are loaded. Order follows the ``phases`` list if provided.
        """
        if phases is not None:
            protocol = protocol.select(phases)

        self._checkpoints.clear()
        self._active_protocol = protocol

        for phase in protocol.phases:
            self.add_checkpoint(
                name=phase.name,
                cameras=phase.cameras,
                **phase.metadata,
            )

    @property
    def active_protocol(self) -> TaskProtocol | None:
        """The currently loaded task protocol, or None."""
        return self._active_protocol

    # ---- Core loop ----

    def reset(self) -> dict[str, Any]:
        """Reset simulation and internal counters."""
        self._step_count = 0
        self._current_checkpoint_idx = 0
        self._trial_count += 1
        if self._rerun_logger is not None:
            self._rerun_logger.configure_trial(self._trial_dir(), self.task_name)
        return self.backend.reset()

    def step(self, action: Any) -> dict[str, Any]:
        """Advance simulation by one step."""
        result = self.backend.step(action)
        self._step_count += 1
        return result

    def run_to_next_checkpoint(self, action_sequence: list[Any]) -> CaptureResult | None:
        """Run simulation until the next checkpoint is reached.

        Executes the given action sequence step-by-step. When a checkpoint
        trigger condition is met (or all actions are exhausted), pauses and
        captures multi-view screenshots.

        Returns CaptureResult at the checkpoint, or None if no more checkpoints.
        """
        if self._current_checkpoint_idx >= len(self._checkpoints):
            return None

        checkpoint = self._checkpoints[self._current_checkpoint_idx]

        # Execute actions
        for action in action_sequence:
            self.step(action)
            if checkpoint.trigger_step is not None and self._step_count >= checkpoint.trigger_step:
                break

        # Capture at checkpoint
        result = self.capture(checkpoint)

        # Save checkpoint state
        sim_state = self.backend.save_state()
        self._checkpoint_store.save(checkpoint.name, sim_state)

        self._current_checkpoint_idx += 1
        return result

    def capture(self, checkpoint: Checkpoint | None = None) -> CaptureResult:
        """Capture multi-view screenshots at current simulation state.

        Can be called manually at any time, or automatically via
        run_to_next_checkpoint().
        """
        cameras = checkpoint.cameras if checkpoint else ["front"]
        cp_name = checkpoint.name if checkpoint else f"step_{self._step_count}"

        views = []
        for cam_name in cameras:
            view = self.backend.capture_camera(cam_name)
            views.append(view)

        state = self.backend.get_state()
        sim_time = self.backend.get_sim_time()

        capture_meta = {"trial": self._trial_count, "task": self.task_name}
        if self._rerun_logger is not None and self._rerun_logger.rrd_path is not None:
            capture_meta["rerun_rrd"] = str(self._rerun_logger.rrd_path)

        result = CaptureResult(
            checkpoint_name=cp_name,
            step=self._step_count,
            sim_time=sim_time,
            views=views,
            state=state,
            metadata=capture_meta,
        )

        # Save to disk
        capture_dir = self._trial_dir() / cp_name
        result.save(capture_dir)
        if self._rerun_logger is not None:
            self._rerun_logger.log_capture(result)

        return result

    def restore_checkpoint(self, name: str) -> None:
        """Restore simulation to a previously saved checkpoint."""
        sim_state = self._checkpoint_store.restore(name)
        self.backend.restore_state(sim_state)

        # Reset checkpoint index to the restored one
        for i, cp in enumerate(self._checkpoints):
            if cp.name == name:
                self._current_checkpoint_idx = i
                break

    def get_state(self) -> dict[str, Any]:
        """Get current simulation state."""
        return self.backend.get_state()

    def list_checkpoints(self) -> list[str]:
        """List all registered checkpoint names."""
        return [cp.name for cp in self._checkpoints]

    def list_saved_checkpoints(self) -> list[str]:
        """List all checkpoints that have saved state snapshots."""
        return self._checkpoint_store.list_checkpoints()

    # ---- Output management ----

    def _trial_dir(self) -> Path:
        """Get output directory for current trial."""
        return self.output_dir / self.task_name / f"trial_{self._trial_count:03d}"
