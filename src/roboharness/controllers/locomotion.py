"""ONNX-based locomotion controllers for humanoid robots.

Provides RL-trained locomotion policies that output joint position targets
from proprioceptive observations. The ONNX models are downloaded from
HuggingFace on first use and cached locally.

Currently supported controllers:
  - GR00T: Balance + Walk dual-policy from NVlabs GR00T-WholeBodyControl
  - Holosoma: FastSAC single-policy from Amazon (Unitree G1 29-DOF)

Not yet supported:
  - SONIC (nvidia/GEAR-SONIC): encoder/decoder/planner architecture with
    multi-tensor inputs. CPU inference is feasible but requires significant
    work (3 models, 6-11 input tensors, 5 locomotion modes). See issue #86.

These controllers implement the ``Controller`` protocol and can be used
standalone with any MuJoCo model, without DDS or unitree_sdk2py.

Usage:
    from roboharness.controllers.locomotion import GrootLocomotionController
    from roboharness.controllers.locomotion import HolosomaLocomotionController

    # GR00T: lower-body only (15-DOF)
    ctrl = GrootLocomotionController()
    state = {"qpos": data.qpos, "qvel": data.qvel}
    action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
    data.ctrl[:15] = action  # lower body + waist joints

    # Holosoma: full-body (29-DOF)
    ctrl = HolosomaLocomotionController()
    state = {"qpos": data.qpos, "qvel": data.qvel}
    action = ctrl.compute(command={"velocity": [0, 0, 0]}, state=state)
    data.ctrl[:29] = action  # full body joints

Reference implementations:
    huggingface.co/lerobot — src/lerobot/robots/unitree_g1/gr00t_locomotion.py
    huggingface.co/nepyope/holosoma_locomotion — FastSAC G1 29-DOF policy
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# G1 joint configuration (29-DOF body)
# ---------------------------------------------------------------------------
# Joints 0-5: left leg (hip pitch/roll/yaw, knee, ankle pitch/roll)
# Joints 6-11: right leg (same)
# Joints 12-14: waist (yaw, roll, pitch)
# Joints 15-21: left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
# Joints 22-28: right arm (same)

NUM_BODY_JOINTS = 29
NUM_LOWER_BODY_JOINTS = 15  # joints 0-14 (legs + waist), controlled by locomotion

# Default standing angles (radians) — slight knee bend for stability
GROOT_DEFAULT_ANGLES = np.zeros(NUM_BODY_JOINTS, dtype=np.float32)
GROOT_DEFAULT_ANGLES[0] = -0.1  # left hip pitch
GROOT_DEFAULT_ANGLES[6] = -0.1  # right hip pitch
GROOT_DEFAULT_ANGLES[3] = 0.3  # left knee
GROOT_DEFAULT_ANGLES[9] = 0.3  # right knee
GROOT_DEFAULT_ANGLES[4] = -0.2  # left ankle pitch
GROOT_DEFAULT_ANGLES[10] = -0.2  # right ankle pitch

# Scaling constants (from GR00T WBC training)
ACTION_SCALE = 0.25
ANG_VEL_SCALE = 0.25
DOF_POS_SCALE = 1.0
DOF_VEL_SCALE = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)

# Observation frame: 86-dim per timestep, 6-frame history → 516-dim input
OBS_FRAME_DIM = 86
OBS_HISTORY_LEN = 6

# HuggingFace model source
GROOT_HF_REPO = "nepyope/GR00T-WholeBodyControl_g1"
GROOT_BALANCE_FILE = "GR00T-WholeBodyControl-Balance.onnx"
GROOT_WALK_FILE = "GR00T-WholeBodyControl-Walk.onnx"


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Compute gravity direction in body frame from quaternion [w, x, y, z]."""
    qw, qx, qy, qz = quaternion
    grav = np.zeros(3, dtype=np.float32)
    grav[0] = 2 * (-qz * qx + qw * qy)
    grav[1] = -2 * (qz * qy + qw * qx)
    grav[2] = 1 - 2 * (qw * qw + qz * qz)
    return grav


def _download_onnx(repo_id: str, filename: str) -> str:
    """Download an ONNX model from HuggingFace, returning the local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for locomotion controllers. "
            "Install with: pip install roboharness[lerobot]"
        ) from e
    path: str = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


class GrootLocomotionController:
    """GR00T Balance + Walk locomotion controller via ONNX inference.

    Downloads pre-trained RL policies from HuggingFace and runs them at
    50 Hz to produce lower-body joint position targets (joints 0-14).

    The controller automatically switches between Balance (standing) and
    Walk (locomotion) policies based on command magnitude.

    Implements the ``Controller`` protocol::

        action = ctrl.compute(
            command={"velocity": [vx, vy, yaw_rate]},
            state={"qpos": qpos, "qvel": qvel},
        )

    Parameters
    ----------
    repo_id:
        HuggingFace repo with ONNX models.
    default_height:
        Desired base height in meters.
    """

    control_dt: float = 0.02  # 50 Hz

    def __init__(
        self,
        repo_id: str = GROOT_HF_REPO,
        default_height: float = 0.74,
    ):
        self._repo_id = repo_id
        self._default_height = default_height

        # Download and load ONNX models
        self._balance_session = self._load_onnx(GROOT_BALANCE_FILE)
        self._walk_session = self._load_onnx(GROOT_WALK_FILE)

        # Internal state
        self._action = np.zeros(NUM_LOWER_BODY_JOINTS, dtype=np.float32)
        self._obs_history: deque[np.ndarray] = deque(maxlen=OBS_HISTORY_LEN)
        self._cmd = np.zeros(3, dtype=np.float32)

        # Pre-fill history with zeros
        for _ in range(OBS_HISTORY_LEN):
            self._obs_history.append(np.zeros(OBS_FRAME_DIM, dtype=np.float32))

        logger.info("GR00T locomotion controller loaded (Balance + Walk)")

    def _load_onnx(self, filename: str) -> Any:
        """Load an ONNX model into an inference session."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for locomotion controllers. "
                "Install with: pip install onnxruntime"
            ) from e
        path = _download_onnx(self._repo_id, filename)
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def compute(self, command: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
        """Compute lower-body joint targets from velocity command and robot state.

        Parameters
        ----------
        command:
            ``{"velocity": [vx, vy, yaw_rate]}`` — desired base velocity.
            Omit or pass zeros for standing.
        state:
            Must contain:
            - ``"qpos"``: joint positions (at least first 36 elements for
              free joint quaternion + 29 body joints)
            - ``"qvel"``: joint velocities (at least first 35 elements for
              free joint + 29 body joints)

        Returns
        -------
        np.ndarray
            Joint position targets for joints 0-14 (lower body + waist).
        """
        # Parse command
        vel = command.get("velocity", [0.0, 0.0, 0.0])
        self._cmd[:] = vel

        # Parse state — real G1 model has free joint (7 qpos, 6 qvel) then 29+14 joints
        qpos = np.asarray(state["qpos"], dtype=np.float32)
        qvel = np.asarray(state["qvel"], dtype=np.float32)

        # Extract IMU data from free joint
        # qpos[3:7] = quaternion (w, x, y, z) for MuJoCo free joint
        base_quat = qpos[3:7] if len(qpos) > 7 else np.array([1, 0, 0, 0], dtype=np.float32)
        # qvel[3:6] = angular velocity for MuJoCo free joint
        ang_vel = qvel[3:6] if len(qvel) > 6 else np.zeros(3, dtype=np.float32)

        # Body joint positions and velocities (skip free joint)
        qj_offset = 7 if len(qpos) > 7 else 0
        dqj_offset = 6 if len(qvel) > 6 else 0
        qj = qpos[qj_offset : qj_offset + NUM_BODY_JOINTS]
        dqj = qvel[dqj_offset : dqj_offset + NUM_BODY_JOINTS]

        # Pad if needed (model may have fewer joints than expected)
        if len(qj) < NUM_BODY_JOINTS:
            qj = np.pad(qj, (0, NUM_BODY_JOINTS - len(qj)))
        if len(dqj) < NUM_BODY_JOINTS:
            dqj = np.pad(dqj, (0, NUM_BODY_JOINTS - len(dqj)))

        # Build single observation frame (86-dim)
        obs = np.zeros(OBS_FRAME_DIM, dtype=np.float32)
        obs[0:3] = self._cmd * CMD_SCALE
        obs[3] = self._default_height
        obs[4:7] = 0.0  # orientation command (zeros)
        obs[7:10] = ang_vel * ANG_VEL_SCALE
        obs[10:13] = get_gravity_orientation(base_quat)
        obs[13:42] = (qj - GROOT_DEFAULT_ANGLES) * DOF_POS_SCALE
        obs[42:71] = dqj * DOF_VEL_SCALE
        obs[71:86] = self._action  # previous action

        # Append to history
        self._obs_history.append(obs.copy())

        # Stack history: 6 frames x 86 = 516-dim (oldest first)
        obs_stacked = np.concatenate(list(self._obs_history)).reshape(1, -1).astype(np.float32)

        # Select policy: Balance for near-zero commands, Walk otherwise
        cmd_magnitude = float(np.linalg.norm(self._cmd))
        session = self._balance_session if cmd_magnitude < 0.05 else self._walk_session

        # ONNX inference
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: obs_stacked})
        self._action[:] = output[0].flatten()[:NUM_LOWER_BODY_JOINTS]

        # Decode action → joint position targets
        target = GROOT_DEFAULT_ANGLES[:NUM_LOWER_BODY_JOINTS] + self._action * ACTION_SCALE
        return target

    def reset(self) -> None:
        """Reset internal state (call on episode reset)."""
        self._action[:] = 0.0
        self._cmd[:] = 0.0
        self._obs_history.clear()
        for _ in range(OBS_HISTORY_LEN):
            self._obs_history.append(np.zeros(OBS_FRAME_DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# Holosoma — FastSAC single-policy controller (Amazon)
# ---------------------------------------------------------------------------
# HuggingFace model source
HOLOSOMA_HF_REPO = "nepyope/holosoma_locomotion"
HOLOSOMA_MODEL_FILE = "fastsac_g1_29dof.onnx"

# Observation frame: 100-dim single frame (no history stacking)
# [0:29]   last action (unscaled)
# [29:32]  angular velocity (IMU gyro) * ANG_VEL_SCALE
# [32]     yaw velocity command
# [33:35]  linear velocity command (x, y)
# [35:37]  gait phase cosine (2 values)
# [37:66]  joint positions (relative to default) * DOF_POS_SCALE
# [66:95]  joint velocities * DOF_VEL_SCALE
# [95:98]  gravity orientation
# [98:100] gait phase sine (2 values)
HOLOSOMA_OBS_DIM = 100

# Default standing angles for Holosoma G1 29-DOF (same joint layout as GR00T)
HOLOSOMA_DEFAULT_ANGLES = np.zeros(NUM_BODY_JOINTS, dtype=np.float32)
HOLOSOMA_DEFAULT_ANGLES[0] = -0.1  # left hip pitch
HOLOSOMA_DEFAULT_ANGLES[6] = -0.1  # right hip pitch
HOLOSOMA_DEFAULT_ANGLES[3] = 0.3  # left knee
HOLOSOMA_DEFAULT_ANGLES[9] = 0.3  # right knee
HOLOSOMA_DEFAULT_ANGLES[4] = -0.2  # left ankle pitch
HOLOSOMA_DEFAULT_ANGLES[10] = -0.2  # right ankle pitch

# Holosoma action scaling (same as GR00T)
HOLOSOMA_ACTION_SCALE = 0.25

# Gait parameters
HOLOSOMA_GAIT_PERIOD = 1.0  # seconds


class HolosomaLocomotionController:
    """Holosoma FastSAC locomotion controller via ONNX inference.

    Downloads a pre-trained FastSAC policy from HuggingFace and runs it at
    50 Hz to produce full-body joint position targets (all 29 DOF).

    Unlike GR00T which uses dual Balance/Walk policies, Holosoma uses a
    single unified policy with explicit gait phase inputs to handle both
    standing and walking.

    Implements the ``Controller`` protocol::

        action = ctrl.compute(
            command={"velocity": [vx, vy, yaw_rate]},
            state={"qpos": qpos, "qvel": qvel},
        )

    Parameters
    ----------
    repo_id:
        HuggingFace repo with ONNX model.
    """

    control_dt: float = 0.02  # 50 Hz

    def __init__(self, repo_id: str = HOLOSOMA_HF_REPO):
        self._repo_id = repo_id

        # Download and load ONNX model
        self._session = self._load_onnx(HOLOSOMA_MODEL_FILE)

        # Internal state
        self._action = np.zeros(NUM_BODY_JOINTS, dtype=np.float32)
        self._cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw_rate]
        self._phase = 0.0  # gait phase in [0, 2*pi)
        self._step_count = 0

        logger.info("Holosoma locomotion controller loaded (FastSAC G1 29-DOF)")

    def _load_onnx(self, filename: str) -> Any:
        """Load an ONNX model into an inference session."""
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for locomotion controllers. "
                "Install with: pip install onnxruntime"
            ) from e
        path = _download_onnx(self._repo_id, filename)
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    def compute(self, command: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
        """Compute full-body joint targets from velocity command and robot state.

        Parameters
        ----------
        command:
            ``{"velocity": [vx, vy, yaw_rate]}`` — desired base velocity.
            Omit or pass zeros for standing.
        state:
            Must contain:
            - ``"qpos"``: joint positions (at least first 36 elements for
              free joint quaternion + 29 body joints)
            - ``"qvel"``: joint velocities (at least first 35 elements for
              free joint + 29 body joints)

        Returns
        -------
        np.ndarray
            Joint position targets for all 29 body joints.
        """
        # Parse command
        vel = command.get("velocity", [0.0, 0.0, 0.0])
        self._cmd[:] = vel

        # Parse state
        qpos = np.asarray(state["qpos"], dtype=np.float32)
        qvel = np.asarray(state["qvel"], dtype=np.float32)

        # Extract IMU data from free joint
        base_quat = qpos[3:7] if len(qpos) > 7 else np.array([1, 0, 0, 0], dtype=np.float32)
        ang_vel = qvel[3:6] if len(qvel) > 6 else np.zeros(3, dtype=np.float32)

        # Body joint positions and velocities (skip free joint)
        qj_offset = 7 if len(qpos) > 7 else 0
        dqj_offset = 6 if len(qvel) > 6 else 0
        qj = qpos[qj_offset : qj_offset + NUM_BODY_JOINTS]
        dqj = qvel[dqj_offset : dqj_offset + NUM_BODY_JOINTS]

        # Pad if needed
        if len(qj) < NUM_BODY_JOINTS:
            qj = np.pad(qj, (0, NUM_BODY_JOINTS - len(qj)))
        if len(dqj) < NUM_BODY_JOINTS:
            dqj = np.pad(dqj, (0, NUM_BODY_JOINTS - len(dqj)))

        # Update gait phase
        self._phase += 2 * np.pi * self.control_dt / HOLOSOMA_GAIT_PERIOD
        self._phase %= 2 * np.pi
        # Two-leg phase: left leg at phase, right leg at phase + pi
        gait_cos = np.array([np.cos(self._phase), np.cos(self._phase + np.pi)], dtype=np.float32)
        gait_sin = np.array([np.sin(self._phase), np.sin(self._phase + np.pi)], dtype=np.float32)

        # Build 100-dim observation
        obs = np.zeros(HOLOSOMA_OBS_DIM, dtype=np.float32)
        obs[0:29] = self._action  # last action (unscaled)
        obs[29:32] = ang_vel * ANG_VEL_SCALE
        obs[32] = self._cmd[2]  # yaw velocity command
        obs[33:35] = self._cmd[:2]  # linear velocity command (x, y)
        obs[35:37] = gait_cos
        obs[37:66] = (qj - HOLOSOMA_DEFAULT_ANGLES) * DOF_POS_SCALE
        obs[66:95] = dqj * DOF_VEL_SCALE
        obs[95:98] = get_gravity_orientation(base_quat)
        obs[98:100] = gait_sin

        # ONNX inference
        obs_input = obs.reshape(1, -1)
        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: obs_input})
        self._action[:] = output[0].flatten()[:NUM_BODY_JOINTS]

        # Decode action → joint position targets
        target = HOLOSOMA_DEFAULT_ANGLES + self._action * HOLOSOMA_ACTION_SCALE
        return target

    def reset(self) -> None:
        """Reset internal state (call on episode reset)."""
        self._action[:] = 0.0
        self._cmd[:] = 0.0
        self._phase = 0.0
        self._step_count = 0
