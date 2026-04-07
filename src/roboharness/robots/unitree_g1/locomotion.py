"""ONNX-based locomotion controllers for humanoid robots.

Provides RL-trained locomotion policies that output joint position targets
from proprioceptive observations. The ONNX models are downloaded from
HuggingFace on first use and cached locally.

Currently supported controllers:
  - GR00T: Balance + Walk dual-policy from NVlabs GR00T-WholeBodyControl
  - Holosoma: FastSAC single-policy from Amazon (Unitree G1 29-DOF)

  - SONIC: Kinematic planner from NVIDIA GEAR-SONIC. Converts velocity
    commands into full-body joint targets via ``planner_sonic.onnx``.
    Supports multiple locomotion modes (idle, walk, run, boxing, etc.).

These controllers implement the ``Controller`` protocol and can be used
standalone with any MuJoCo model, without DDS or unitree_sdk2py.

Usage:
    from roboharness.robots.unitree_g1 import GrootLocomotionController
    from roboharness.robots.unitree_g1 import HolosomaLocomotionController

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

import dataclasses
import enum
import logging
import pathlib
from collections import deque
from typing import Any, ClassVar

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

# Default standing angles (radians) — slight knee bend for stability.
# All three controllers (GR00T, Holosoma, SONIC) use the same base pose for
# the Unitree G1, so we define it once and copy.
_G1_DEFAULT_STANDING = np.zeros(NUM_BODY_JOINTS, dtype=np.float32)
_G1_DEFAULT_STANDING[0] = -0.1  # left hip pitch
_G1_DEFAULT_STANDING[6] = -0.1  # right hip pitch
_G1_DEFAULT_STANDING[3] = 0.3  # left knee
_G1_DEFAULT_STANDING[9] = 0.3  # right knee
_G1_DEFAULT_STANDING[4] = -0.2  # left ankle pitch
_G1_DEFAULT_STANDING[10] = -0.2  # right ankle pitch

GROOT_DEFAULT_ANGLES = _G1_DEFAULT_STANDING.copy()

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
            "Install with: pip install roboharness[demo]"
        ) from e
    path: str = hf_hub_download(repo_id=repo_id, filename=filename)
    return path


def _load_onnx_session(repo_id: str, filename: str) -> Any:
    """Download and load an ONNX model into an inference session."""
    try:
        import onnxruntime as ort
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for locomotion controllers. "
            "Install with: pip install onnxruntime"
        ) from e
    path = _download_onnx(repo_id, filename)
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def _parse_imu_and_joints(
    state: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract IMU data and body joint pos/vel from simulator state.

    Returns ``(base_quat, ang_vel, joint_pos, joint_vel)`` with all arrays
    padded to ``NUM_BODY_JOINTS`` length.
    """
    qpos = np.asarray(state["qpos"], dtype=np.float32)
    qvel = np.asarray(state["qvel"], dtype=np.float32)

    # Free joint: qpos[3:7] = quaternion, qvel[3:6] = angular velocity
    base_quat = qpos[3:7] if len(qpos) > 7 else np.array([1, 0, 0, 0], dtype=np.float32)
    ang_vel = qvel[3:6] if len(qvel) > 6 else np.zeros(3, dtype=np.float32)

    # Body joints (skip free joint)
    qj_offset = 7 if len(qpos) > 7 else 0
    dqj_offset = 6 if len(qvel) > 6 else 0
    qj = qpos[qj_offset : qj_offset + NUM_BODY_JOINTS]
    dqj = qvel[dqj_offset : dqj_offset + NUM_BODY_JOINTS]

    if len(qj) < NUM_BODY_JOINTS:
        qj = np.pad(qj, (0, NUM_BODY_JOINTS - len(qj)))
    if len(dqj) < NUM_BODY_JOINTS:
        dqj = np.pad(dqj, (0, NUM_BODY_JOINTS - len(dqj)))

    return base_quat, ang_vel, qj, dqj


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
        self._balance_session = _load_onnx_session(repo_id, GROOT_BALANCE_FILE)
        self._walk_session = _load_onnx_session(repo_id, GROOT_WALK_FILE)

        # Internal state
        self._action = np.zeros(NUM_LOWER_BODY_JOINTS, dtype=np.float32)
        self._obs_history: deque[np.ndarray] = deque(maxlen=OBS_HISTORY_LEN)
        self._cmd = np.zeros(3, dtype=np.float32)

        # Pre-fill history with zeros
        for _ in range(OBS_HISTORY_LEN):
            self._obs_history.append(np.zeros(OBS_FRAME_DIM, dtype=np.float32))

        logger.info("GR00T locomotion controller loaded (Balance + Walk)")

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

        # Parse state
        base_quat, ang_vel, qj, dqj = _parse_imu_and_joints(state)

        # Build single observation frame (86-dim)
        obs = np.zeros(OBS_FRAME_DIM, dtype=np.float32)
        obs[0:3] = self._cmd * CMD_SCALE
        obs[3] = self._default_height
        # obs[4:7] is already zero (orientation command)
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

HOLOSOMA_DEFAULT_ANGLES = _G1_DEFAULT_STANDING.copy()

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
        self._session = _load_onnx_session(repo_id, HOLOSOMA_MODEL_FILE)

        # Internal state
        self._action = np.zeros(NUM_BODY_JOINTS, dtype=np.float32)
        self._cmd = np.zeros(3, dtype=np.float32)  # [vx, vy, yaw_rate]
        self._phase = 0.0  # gait phase in [0, 2*pi)

        logger.info("Holosoma locomotion controller loaded (FastSAC G1 29-DOF)")

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
        base_quat, ang_vel, qj, dqj = _parse_imu_and_joints(state)

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


# ---------------------------------------------------------------------------
# SONIC — Kinematic planner controller (NVIDIA GEAR-SONIC)
# ---------------------------------------------------------------------------
# HuggingFace model source
SONIC_HF_REPO = "nvidia/GEAR-SONIC"
SONIC_PLANNER_FILE = "planner_sonic.onnx"

# Planner qpos frame layout (36-dim per frame):
#   [0:3]   root position (x, y, z)
#   [3:7]   root quaternion (w, x, y, z)
#   [7:36]  29 body joint angles (radians)
SONIC_QPOS_DIM = 36
SONIC_CONTEXT_LEN = 4  # 4-frame context window

# Planner runs at 10 Hz, outputs poses at 30 Hz, control loop at 50 Hz
SONIC_PLANNER_DT = 0.1  # 10 Hz planner cycle
SONIC_OUTPUT_RATE = 30  # Hz — model output rate
SONIC_CONTROL_RATE = 50  # Hz — control loop rate

SONIC_DEFAULT_ANGLES = _G1_DEFAULT_STANDING.copy()

# Default pelvis height (meters)
SONIC_DEFAULT_HEIGHT = 0.74

# Default number of allowed prediction tokens
SONIC_DEFAULT_NUM_TOKENS = 11

# Phase 2: Encoder+Decoder model files
SONIC_ENCODER_FILE = "encoder_sonic.onnx"
SONIC_DECODER_FILE = "decoder_sonic.onnx"

# Encoder input: 29 joint pos + 29 joint vel + 1 root height + 6D rotation = 65
SONIC_ENCODER_INPUT_DIM = 65
# Encoder output / decoder latent input
SONIC_LATENT_DIM = 64
# Decoder robot state input: 29 joint pos + 29 joint vel = 58
SONIC_ROBOT_STATE_DIM = 58
# Decoder output: 29-DOF joint targets
SONIC_DECODER_OUTPUT_DIM = NUM_BODY_JOINTS


class SonicMode(enum.IntEnum):
    """Locomotion modes supported by the SONIC planner.

    The planner accepts an integer ``mode`` input that selects the movement
    style.  Phase 1 exposes the five most common modes; the full set (27 in
    SONIC V2) can be added later.
    """

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    BOXING = 4


@dataclasses.dataclass
class MotionClip:
    """A motion capture clip containing joint and root data at fixed FPS.

    Each field has ``num_frames`` rows sampled at ``fps`` Hz (default 50 Hz).
    """

    joint_positions: np.ndarray  # (N, 29) — joint angles in radians
    joint_velocities: np.ndarray  # (N, 29) — joint angular velocities
    root_height: np.ndarray  # (N,) — pelvis height in meters
    root_rotation_6d: np.ndarray  # (N, 6) — 6D rotation representation
    fps: float = 50.0
    name: str = ""

    @property
    def num_frames(self) -> int:
        return int(self.joint_positions.shape[0])

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.num_frames / self.fps

    def reference_frame(self, index: int) -> np.ndarray:
        """Build a 65-dim encoder input vector for the given frame index.

        Layout: [29 joint_pos, 29 joint_vel, 1 root_height, 6 root_rotation_6d].
        Clamps ``index`` to valid range.
        """
        i = max(0, min(index, self.num_frames - 1))
        return np.concatenate(
            [
                self.joint_positions[i],
                self.joint_velocities[i],
                self.root_height[i : i + 1],
                self.root_rotation_6d[i],
            ]
        ).astype(np.float32)


class MotionClipLoader:
    """Load :class:`MotionClip` instances from CSV directories.

    Expected directory layout::

        clip_dir/
            joint_positions.csv    # (N, 29)
            joint_velocities.csv   # (N, 29)
            root_height.csv        # (N, 1)
            root_rotation_6d.csv   # (N, 6)

    All CSV files are comma-delimited with no header row.
    """

    _REQUIRED_FILES: ClassVar[list[str]] = [
        "joint_positions.csv",
        "joint_velocities.csv",
        "root_height.csv",
        "root_rotation_6d.csv",
    ]

    @classmethod
    def load(cls, directory: str | pathlib.Path, fps: float = 50.0) -> MotionClip:
        """Load a motion clip from *directory*.

        Parameters
        ----------
        directory:
            Path to a directory containing the four required CSV files.
        fps:
            Sampling rate of the data (default 50 Hz).

        Raises
        ------
        FileNotFoundError
            If any required CSV file is missing.
        """
        d = pathlib.Path(directory)
        for fname in cls._REQUIRED_FILES:
            if not (d / fname).exists():
                raise FileNotFoundError(f"Missing required file: {d / fname}")

        joint_pos = np.loadtxt(d / "joint_positions.csv", delimiter=",", dtype=np.float32)
        joint_vel = np.loadtxt(d / "joint_velocities.csv", delimiter=",", dtype=np.float32)
        root_h = np.loadtxt(d / "root_height.csv", delimiter=",", dtype=np.float32)
        root_rot = np.loadtxt(d / "root_rotation_6d.csv", delimiter=",", dtype=np.float32)

        # Ensure 2D shape for single-column root_height
        if root_h.ndim == 2:
            root_h = root_h[:, 0]

        return MotionClip(
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            root_height=root_h,
            root_rotation_6d=root_rot,
            fps=fps,
            name=d.name,
        )


class SonicLocomotionController:
    """SONIC locomotion controller with planner and encoder+decoder pipelines.

    Downloads ONNX models from HuggingFace and supports two modes:

    **Planner mode** (Phase 1): Uses ``planner_sonic.onnx`` at 10 Hz to produce
    full-body pose trajectories from velocity commands. Between planner calls the
    controller interpolates the 30 Hz output to 50 Hz for smooth control.

    **Tracking mode** (Phase 2): Uses ``encoder_sonic.onnx`` and
    ``decoder_sonic.onnx`` to track reference motion clips. The encoder converts
    a 65-dim motion reference into a 64-dim latent token, and the decoder maps
    the latent plus the current 58-dim robot state into 29-DOF joint targets.

    Implements the ``Controller`` protocol::

        # Planner mode
        action = ctrl.compute(
            command={"velocity": [vx, vy, yaw_rate], "mode": SonicMode.WALK},
            state={"qpos": qpos, "qvel": qvel},
        )

        # Tracking mode
        ctrl.set_tracking_clip(clip)
        action = ctrl.compute(
            command={"tracking": True},
            state={"qpos": qpos, "qvel": qvel},
        )

    Parameters
    ----------
    repo_id:
        HuggingFace repo with the ONNX models.
    default_height:
        Desired pelvis height in meters.
    default_mode:
        Locomotion mode when not specified in the command.
    """

    control_dt: float = 1.0 / SONIC_CONTROL_RATE  # 50 Hz (0.02 s)

    def __init__(
        self,
        repo_id: str = SONIC_HF_REPO,
        default_height: float = SONIC_DEFAULT_HEIGHT,
        default_mode: SonicMode = SonicMode.WALK,
    ):
        self._repo_id = repo_id
        self._default_height = default_height
        self._default_mode = default_mode

        # Download and load ONNX models
        self._planner_session = _load_onnx_session(repo_id, SONIC_PLANNER_FILE)
        self._encoder_session = _load_onnx_session(repo_id, SONIC_ENCODER_FILE)
        self._decoder_session = _load_onnx_session(repo_id, SONIC_DECODER_FILE)

        # Context window: last 4 full qpos frames (36-dim each)
        self._context: deque[np.ndarray] = deque(maxlen=SONIC_CONTEXT_LEN)

        # Predicted trajectory from last planner call (30 Hz frames)
        self._trajectory: list[np.ndarray] = []
        self._traj_index: int = 0

        # Interpolation state for 30→50 Hz resampling
        self._interp_phase: float = 0.0

        # Steps since last planner invocation (at 50 Hz)
        self._steps_since_plan: int = 0
        self._plan_interval: int = int(SONIC_PLANNER_DT * SONIC_CONTROL_RATE)  # 5 steps

        # Current command state
        self._cmd = np.zeros(3, dtype=np.float32)
        self._mode = default_mode

        # Phase 2: tracking state
        self._tracking_clip: MotionClip | None = None
        self._tracking_frame_index: int = 0

        # Initialise context with default standing pose
        standing = self._make_standing_qpos()
        for _ in range(SONIC_CONTEXT_LEN):
            self._context.append(standing.copy())

        logger.info("SONIC locomotion controller loaded (planner + encoder/decoder)")

    @staticmethod
    def _make_standing_qpos() -> np.ndarray:
        """Build a default standing qpos frame (36-dim)."""
        qpos = np.zeros(SONIC_QPOS_DIM, dtype=np.float32)
        qpos[3] = 1.0  # quaternion w = 1 (identity)
        qpos[2] = SONIC_DEFAULT_HEIGHT  # root z = pelvis height
        qpos[7:36] = SONIC_DEFAULT_ANGLES
        return qpos

    def _run_planner(self, height: float) -> None:
        """Invoke the planner ONNX model and store the predicted trajectory."""
        # Build context tensor [1, 4, 36]
        context = np.stack(list(self._context), axis=0).reshape(1, SONIC_CONTEXT_LEN, -1)
        context = context.astype(np.float32)

        # Movement direction from velocity command (x, y forward; z = 0)
        cmd_norm = float(np.linalg.norm(self._cmd[:2]))
        if cmd_norm > 1e-6:
            move_dir = np.array(
                [self._cmd[0] / cmd_norm, self._cmd[1] / cmd_norm, 0.0], dtype=np.float32
            )
        else:
            move_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Facing direction: rotate by yaw_rate (small angle approx for single step)
        yaw = self._cmd[2]
        facing_dir = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)

        # Target velocity — use command magnitude, ≤0 means use mode default
        target_vel = np.array([cmd_norm], dtype=np.float32)

        feed: dict[str, np.ndarray] = {
            "context_mujoco_qpos": context,
            "target_vel": target_vel,
            "mode": np.array([int(self._mode)], dtype=np.int64),
            "movement_direction": move_dir.reshape(1, 3),
            "facing_direction": facing_dir.reshape(1, 3),
            "height": np.array([height], dtype=np.float32),
            "random_seed": np.array([0], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.ones((1, SONIC_DEFAULT_NUM_TOKENS), dtype=np.int64),
        }

        outputs = self._planner_session.run(None, feed)
        # outputs[0] = mujoco_qpos [1, N, 36], outputs[1] = num_pred_frames
        pred_qpos = outputs[0][0]  # [N, 36]
        num_frames = int(outputs[1]) if np.ndim(outputs[1]) == 0 else int(outputs[1].flat[0])
        num_frames = max(1, min(num_frames, len(pred_qpos)))

        self._trajectory = [pred_qpos[i] for i in range(num_frames)]
        self._traj_index = 0
        self._interp_phase = 0.0

    def set_tracking_clip(self, clip: MotionClip) -> None:
        """Set a motion clip for tracking mode.

        Parameters
        ----------
        clip:
            The :class:`MotionClip` to track. Call :meth:`compute` with
            ``command={"tracking": True}`` to use it.
        """
        self._tracking_clip = clip
        self._tracking_frame_index = 0

    def clear_tracking_clip(self) -> None:
        """Remove the current tracking clip and reset tracking state."""
        self._tracking_clip = None
        self._tracking_frame_index = 0

    def _run_tracking(self, state: dict[str, Any]) -> np.ndarray:
        """Run the encoder+decoder pipeline for one tracking step.

        Encodes the current motion reference frame into a latent token,
        then decodes it with the current robot state to produce joint targets.
        """
        if self._tracking_clip is None:
            raise RuntimeError("No tracking clip set — call set_tracking_clip() first")

        # Get 65-dim reference from the current clip frame
        ref = self._tracking_clip.reference_frame(self._tracking_frame_index)
        ref_input = ref.reshape(1, -1).astype(np.float32)

        # Encoder: 65-dim → 64-dim latent
        latent = self._encoder_session.run(None, {"motion_ref": ref_input})[0]

        # Build 58-dim robot state: 29 joint pos + 29 joint vel
        _base_quat, _ang_vel, qj, dqj = _parse_imu_and_joints(state)

        robot_state = np.concatenate([qj, dqj]).reshape(1, -1).astype(np.float32)

        # Decoder: (64-dim latent, 58-dim state) → 29-DOF targets
        targets = self._decoder_session.run(None, {"latent": latent, "robot_state": robot_state})[0]

        # Advance tracking frame (at control rate relative to clip fps)
        step_frames = self._tracking_clip.fps / SONIC_CONTROL_RATE
        self._tracking_frame_index = min(
            int(self._tracking_frame_index + step_frames),
            self._tracking_clip.num_frames - 1,
        )

        result: np.ndarray = targets.flatten()[:NUM_BODY_JOINTS]
        return result

    def compute(self, command: dict[str, Any], state: dict[str, Any]) -> np.ndarray:
        """Compute full-body joint targets from velocity command and robot state.

        Parameters
        ----------
        command:
            For planner mode: ``{"velocity": [vx, vy, yaw_rate]}``.
            Optional ``"mode"`` key selects locomotion style (``SonicMode``).
            Optional ``"height"`` overrides pelvis height.

            For tracking mode: ``{"tracking": True}``. Requires a clip set
            via :meth:`set_tracking_clip`.
        state:
            Must contain:
            - ``"qpos"``: joint positions (at least first 36 elements for
              free joint quaternion + 29 body joints)
            - ``"qvel"``: joint velocities (for tracking mode)

        Returns
        -------
        np.ndarray
            Joint position targets for all 29 body joints.
        """
        # Phase 2: tracking mode — use encoder+decoder if a clip is active
        if command.get("tracking") and self._tracking_clip is not None:
            return self._run_tracking(state)

        # Parse command (planner mode)
        vel = command.get("velocity", [0.0, 0.0, 0.0])
        self._cmd[:] = vel
        self._mode = SonicMode(command.get("mode", self._default_mode))
        height = float(command.get("height", self._default_height))

        # Parse state — extract full qpos for context
        qpos = np.asarray(state["qpos"], dtype=np.float32)
        # Build 36-dim qpos frame for context
        if len(qpos) >= SONIC_QPOS_DIM:
            context_frame = qpos[:SONIC_QPOS_DIM].copy()
        else:
            context_frame = np.zeros(SONIC_QPOS_DIM, dtype=np.float32)
            context_frame[: len(qpos)] = qpos
            context_frame[3] = 1.0  # ensure valid quaternion

        # Update context window
        self._context.append(context_frame)

        # Re-plan at 10 Hz (every plan_interval control steps)
        if self._steps_since_plan >= self._plan_interval or not self._trajectory:
            self._run_planner(height)
            self._steps_since_plan = 0
        self._steps_since_plan += 1

        # Interpolate 30 Hz trajectory → 50 Hz control
        # Ratio: 30/50 = 0.6 model frames per control step
        interp_step = SONIC_OUTPUT_RATE / SONIC_CONTROL_RATE  # 0.6

        if len(self._trajectory) < 2:
            # Single frame or empty — just return it directly
            frame = self._trajectory[0] if self._trajectory else self._make_standing_qpos()
            result: np.ndarray = frame[7:36].copy()
            return result

        # Compute interpolated frame
        idx = min(self._traj_index, len(self._trajectory) - 2)
        alpha = self._interp_phase
        frame_a = self._trajectory[idx]
        frame_b = self._trajectory[min(idx + 1, len(self._trajectory) - 1)]
        interpolated = frame_a + alpha * (frame_b - frame_a)

        # Advance interpolation phase
        self._interp_phase += interp_step
        while self._interp_phase >= 1.0 and self._traj_index < len(self._trajectory) - 2:
            self._interp_phase -= 1.0
            self._traj_index += 1

        # Return only the 29 joint angles (skip root pos + quaternion)
        joints: np.ndarray = interpolated[7:36].copy()
        return joints

    def reset(self) -> None:
        """Reset internal state (call on episode reset).

        Resets planner and tracking state. The tracking clip reference is
        preserved — call :meth:`clear_tracking_clip` to remove it.
        """
        self._cmd[:] = 0.0
        self._mode = self._default_mode
        self._trajectory.clear()
        self._traj_index = 0
        self._interp_phase = 0.0
        self._steps_since_plan = 0
        self._tracking_frame_index = 0
        self._context.clear()
        standing = self._make_standing_qpos()
        for _ in range(SONIC_CONTEXT_LEN):
            self._context.append(standing.copy())
