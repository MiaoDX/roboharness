"""Tests for semantic task protocols."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from roboharness.core.capture import CameraView
from roboharness.core.harness import Harness
from roboharness.core.protocol import (
    BUILTIN_PROTOCOLS,
    DANCE_PROTOCOL,
    GRASP_PROTOCOL,
    LOCO_MANIPULATION_PROTOCOL,
    LOCOMOTION_PROTOCOL,
    TaskPhase,
    TaskProtocol,
)

# ---------------------------------------------------------------------------
# Mock backend (reused from test_harness.py pattern)
# ---------------------------------------------------------------------------


class MockBackend:
    def __init__(self) -> None:
        self._time = 0.0
        self._state: dict[str, Any] = {"qpos": [0.0], "qvel": [0.0]}

    def step(self, action: Any) -> dict[str, Any]:
        self._time += 0.01
        self._state["qpos"] = [self._state["qpos"][0] + 0.1]
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        return {**self._state, "time": self._time}

    def save_state(self) -> dict[str, Any]:
        return {"state": {**self._state}, "time": self._time}

    def restore_state(self, state: dict[str, Any]) -> None:
        self._state = {**state["state"]}
        self._time = state["time"]

    def capture_camera(self, camera_name: str) -> CameraView:
        return CameraView(name=camera_name, rgb=np.zeros((64, 64, 3), dtype=np.uint8))

    def get_sim_time(self) -> float:
        return self._time

    def reset(self) -> dict[str, Any]:
        self._time = 0.0
        self._state = {"qpos": [0.0], "qvel": [0.0]}
        return self.get_state()


# ---------------------------------------------------------------------------
# TaskPhase tests
# ---------------------------------------------------------------------------


class TestTaskPhase:
    def test_basic_creation(self):
        phase = TaskPhase("grasp", "Close gripper on object")
        assert phase.name == "grasp"
        assert phase.description == "Close gripper on object"
        assert phase.cameras == ["front"]
        assert phase.metadata == {}

    def test_custom_cameras(self):
        phase = TaskPhase("lift", "Lift object", cameras=["front", "wrist", "top"])
        assert phase.cameras == ["front", "wrist", "top"]

    def test_metadata(self):
        phase = TaskPhase("approach", "Approach object", metadata={"speed": 0.1})
        assert phase.metadata["speed"] == 0.1

    def test_frozen(self):
        phase = TaskPhase("grasp", "Close gripper")
        with pytest.raises(AttributeError):
            phase.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TaskProtocol tests
# ---------------------------------------------------------------------------


class TestTaskProtocol:
    def test_basic_creation(self):
        protocol = TaskProtocol(
            name="test",
            description="Test protocol",
            phases=[
                TaskPhase("a", "Phase A"),
                TaskPhase("b", "Phase B"),
            ],
        )
        assert protocol.name == "test"
        assert len(protocol.phases) == 2

    def test_phase_names(self):
        protocol = TaskProtocol(
            name="test",
            phases=[TaskPhase("x"), TaskPhase("y"), TaskPhase("z")],
        )
        assert protocol.phase_names() == ["x", "y", "z"]

    def test_get_phase(self):
        phase_b = TaskPhase("b", "Phase B")
        protocol = TaskProtocol(
            name="test",
            phases=[TaskPhase("a", "Phase A"), phase_b],
        )
        assert protocol.get_phase("b") == phase_b

    def test_get_phase_missing_raises(self):
        protocol = TaskProtocol(name="test", phases=[TaskPhase("a")])
        with pytest.raises(KeyError, match="not_here"):
            protocol.get_phase("not_here")

    def test_select_subset(self):
        protocol = TaskProtocol(
            name="full",
            description="Full protocol",
            phases=[
                TaskPhase("a"),
                TaskPhase("b"),
                TaskPhase("c"),
                TaskPhase("d"),
            ],
        )
        subset = protocol.select(["b", "d"])
        assert subset.phase_names() == ["b", "d"]
        assert subset.name == "full"
        assert subset.description == "Full protocol"

    def test_select_preserves_order(self):
        protocol = TaskProtocol(
            name="ordered",
            phases=[TaskPhase("a"), TaskPhase("b"), TaskPhase("c")],
        )
        # Request in different order than defined — result follows request order
        subset = protocol.select(["c", "a"])
        assert subset.phase_names() == ["c", "a"]

    def test_select_missing_raises(self):
        protocol = TaskProtocol(name="test", phases=[TaskPhase("a")])
        with pytest.raises(KeyError, match="missing"):
            protocol.select(["a", "missing"])

    def test_empty_protocol(self):
        protocol = TaskProtocol(name="empty")
        assert protocol.phase_names() == []
        assert protocol.phases == []

    def test_frozen(self):
        protocol = TaskProtocol(name="test")
        with pytest.raises(AttributeError):
            protocol.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Built-in protocols
# ---------------------------------------------------------------------------


class TestBuiltinProtocols:
    def test_grasp_protocol(self):
        assert GRASP_PROTOCOL.name == "grasp"
        names = GRASP_PROTOCOL.phase_names()
        assert "pre_grasp" in names
        assert "grasp" in names
        assert "lift" in names
        assert "plan" in names
        assert "place" in names
        assert "home" in names
        assert len(names) == 7

    def test_locomotion_protocol(self):
        assert LOCOMOTION_PROTOCOL.name == "locomotion"
        names = LOCOMOTION_PROTOCOL.phase_names()
        assert "initial" in names
        assert "steady" in names
        assert "terminal" in names
        assert len(names) == 5

    def test_loco_manipulation_protocol(self):
        assert LOCO_MANIPULATION_PROTOCOL.name == "loco_manipulation"
        names = LOCO_MANIPULATION_PROTOCOL.phase_names()
        assert "navigate" in names
        assert "grasp" in names
        assert "transport" in names
        assert "place" in names
        assert len(names) == 6

    def test_dance_protocol(self):
        assert DANCE_PROTOCOL.name == "dance"
        assert len(DANCE_PROTOCOL.phases) == 3

    def test_builtin_registry(self):
        assert set(BUILTIN_PROTOCOLS.keys()) == {
            "grasp",
            "locomotion",
            "loco_manipulation",
            "dance",
        }
        for name, proto in BUILTIN_PROTOCOLS.items():
            assert proto.name == name
            assert len(proto.phases) > 0

    def test_all_builtin_phases_have_descriptions(self):
        for proto in BUILTIN_PROTOCOLS.values():
            for phase in proto.phases:
                assert phase.description, f"{proto.name}.{phase.name} has no description"


# ---------------------------------------------------------------------------
# Harness integration: load_protocol
# ---------------------------------------------------------------------------


class TestHarnessLoadProtocol:
    def test_load_protocol_adds_checkpoints(self, tmp_path):
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.load_protocol(GRASP_PROTOCOL)
        assert harness.list_checkpoints() == GRASP_PROTOCOL.phase_names()

    def test_load_protocol_with_phase_subset(self, tmp_path):
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.load_protocol(GRASP_PROTOCOL, phases=["pre_grasp", "grasp", "lift"])
        assert harness.list_checkpoints() == ["pre_grasp", "grasp", "lift"]

    def test_load_protocol_respects_cameras(self, tmp_path):
        protocol = TaskProtocol(
            name="cam_test",
            phases=[
                TaskPhase("a", cameras=["front", "wrist"]),
                TaskPhase("b", cameras=["top"]),
            ],
        )
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.load_protocol(protocol)

        # Verify cameras are correctly passed through
        assert harness._checkpoints[0].cameras == ["front", "wrist"]
        assert harness._checkpoints[1].cameras == ["top"]

    def test_load_protocol_sets_active_protocol(self, tmp_path):
        harness = Harness(MockBackend(), output_dir=tmp_path)
        assert harness.active_protocol is None
        harness.load_protocol(GRASP_PROTOCOL)
        assert harness.active_protocol is not None
        assert harness.active_protocol.name == "grasp"

    def test_load_protocol_subset_stores_subset(self, tmp_path):
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.load_protocol(GRASP_PROTOCOL, phases=["grasp", "lift"])
        assert harness.active_protocol is not None
        assert harness.active_protocol.phase_names() == ["grasp", "lift"]

    def test_load_protocol_clears_previous_checkpoints(self, tmp_path):
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.add_checkpoint("old_cp")
        harness.load_protocol(GRASP_PROTOCOL, phases=["grasp"])
        assert harness.list_checkpoints() == ["grasp"]

    def test_load_protocol_end_to_end(self, tmp_path):
        """Full workflow: load protocol, reset, run through phases."""
        protocol = TaskProtocol(
            name="mini",
            phases=[TaskPhase("step_a"), TaskPhase("step_b")],
        )
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.load_protocol(protocol)
        harness.reset()

        result_a = harness.run_to_next_checkpoint([None] * 5)
        assert result_a is not None
        assert result_a.checkpoint_name == "step_a"

        result_b = harness.run_to_next_checkpoint([None] * 5)
        assert result_b is not None
        assert result_b.checkpoint_name == "step_b"

        # No more checkpoints
        assert harness.run_to_next_checkpoint([None]) is None

    def test_load_protocol_phase_metadata_in_checkpoint(self, tmp_path):
        protocol = TaskProtocol(
            name="meta_test",
            phases=[TaskPhase("x", metadata={"timeout": 10})],
        )
        harness = Harness(MockBackend(), output_dir=tmp_path)
        harness.load_protocol(protocol)
        assert harness._checkpoints[0].metadata == {"timeout": 10}
