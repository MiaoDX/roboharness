"""Shared MuJoCo grasp fixture used by the example scripts."""

from __future__ import annotations

import numpy as np

from roboharness.core.protocol import TaskPhase, TaskProtocol

MUJOCO_GRASP_TASK = "mujoco_grasp"
MUJOCO_GRASP_CAMERAS = ["front", "side", "top"]
MUJOCO_GRASP_PHASE_ORDER = ["plan", "pre_grasp", "approach", "grasp", "lift"]
MUJOCO_GRASP_PHASE_LABELS = {
    "plan": "plan_start",
    "pre_grasp": "pre_grasp",
    "approach": "approach",
    "grasp": "contact",
    "lift": "lift",
}
MUJOCO_GRASP_PRIMARY_VIEWS = {
    "plan": ["front", "top"],
    "pre_grasp": ["front", "top"],
    "approach": ["side", "top"],
    "grasp": ["front", "side"],
    "lift": ["front", "side"],
}

GRASP_MJCF = """\
<mujoco model="simple_grasp">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.6 0.8 1.0" rgb2="0.2 0.3 0.5"
             width="256" height="256"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"
             width="256" height="256"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4" reflectance="0.1"/>
    <material name="table_mat" rgba="0.6 0.4 0.2 1"/>
    <material name="cube_mat" rgba="0.9 0.2 0.2 1"/>
    <material name="gripper_mat" rgba="0.3 0.3 0.7 1"/>
  </asset>

  <worldbody>
    <geom type="plane" size="1 1 0.01" material="grid_mat"/>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <light pos="0.5 0.5 1.5" dir="-0.3 -0.3 -1" diffuse="0.4 0.4 0.4"/>

    <camera name="front" pos="0.75 0 0.55" xyaxes="0 1 0 -0.4 0 0.75"/>
    <camera name="side" pos="0 0.75 0.55" xyaxes="-1 0 0 0 -0.4 0.75"/>
    <camera name="top" pos="0 0 1.2" xyaxes="1 0 0 0 1 0"/>

    <body name="table" pos="0 0 0.2">
      <geom type="box" size="0.3 0.3 0.02" material="table_mat"/>
      <geom type="cylinder" size="0.015 0.1" pos=" 0.25  0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos="-0.25  0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos=" 0.25 -0.25 -0.12"/>
      <geom type="cylinder" size="0.015 0.1" pos="-0.25 -0.25 -0.12"/>
    </body>

    <body name="cube" pos="0 0 0.25">
      <joint type="free"/>
      <geom type="box" size="0.025 0.025 0.025" mass="0.02" material="cube_mat"
            friction="2.0 0.1 0.001" condim="4" solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>

    <body name="gripper_base" pos="0 0 0.55">
      <joint name="gripper_z" type="slide" axis="0 0 1" range="-0.35 0.1" damping="50"/>
      <geom type="cylinder" size="0.02 0.03" material="gripper_mat"/>

      <body name="finger_left" pos="0 0.04 -0.06">
        <joint name="finger_left" type="slide" axis="0 1 0" range="-0.02 0.015" damping="0.5"/>
        <geom type="box" size="0.012 0.012 0.04" material="gripper_mat"
              friction="2.0 0.1 0.001" condim="4"/>
      </body>

      <body name="finger_right" pos="0 -0.04 -0.06">
        <joint name="finger_right" type="slide" axis="0 1 0" range="-0.015 0.02" damping="0.5"/>
        <geom type="box" size="0.012 0.012 0.04" material="gripper_mat"
              friction="2.0 0.1 0.001" condim="4"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position name="gripper_z_ctrl" joint="gripper_z" kp="200" ctrlrange="-0.35 0.1"/>
    <position name="finger_left_ctrl" joint="finger_left" kp="100" ctrlrange="-0.02 0.015"/>
    <position name="finger_right_ctrl" joint="finger_right" kp="100" ctrlrange="-0.015 0.02"/>
  </actuator>
</mujoco>
"""


def make_action_sequence(
    target_z: float, finger_left: float, finger_right: float, n_steps: int
) -> list[np.ndarray]:
    """Create a constant control sequence for a scripted phase."""
    action = np.array([target_z, finger_left, finger_right], dtype=np.float64)
    return [action.copy() for _ in range(n_steps)]


def build_grasp_phases() -> dict[str, list[np.ndarray]]:
    """Build the scripted grasp motion, including the initial planning checkpoint."""
    left_open, left_closed = 0.015, -0.02
    right_open, right_closed = -0.015, 0.02

    return {
        "plan": [],
        "pre_grasp": make_action_sequence(
            target_z=0.05,
            finger_left=left_open,
            finger_right=right_open,
            n_steps=500,
        ),
        "approach": make_action_sequence(
            target_z=-0.24,
            finger_left=left_open,
            finger_right=right_open,
            n_steps=500,
        ),
        "grasp": make_action_sequence(
            target_z=-0.24,
            finger_left=left_closed,
            finger_right=right_closed,
            n_steps=800,
        ),
        "lift": make_action_sequence(
            target_z=-0.10,
            finger_left=left_closed,
            finger_right=right_closed,
            n_steps=800,
        ),
    }


def build_grasp_protocol() -> TaskProtocol:
    """Return the shared grasp protocol with explicit multi-camera capture."""
    return TaskProtocol(
        name="grasp",
        description="Pick-and-place grasping task",
        phases=[
            TaskPhase(
                "plan",
                "Plan grasp trajectory and visualize target path",
                cameras=list(MUJOCO_GRASP_CAMERAS),
            ),
            TaskPhase(
                "pre_grasp",
                "Move to pre-grasp pose above the object",
                cameras=list(MUJOCO_GRASP_CAMERAS),
            ),
            TaskPhase(
                "approach",
                "Approach the object along the planned path",
                cameras=list(MUJOCO_GRASP_CAMERAS),
            ),
            TaskPhase(
                "grasp",
                "Close gripper on the object",
                cameras=list(MUJOCO_GRASP_CAMERAS),
            ),
            TaskPhase(
                "lift",
                "Lift the object while preserving the grasp",
                cameras=list(MUJOCO_GRASP_CAMERAS),
            ),
        ],
    )
