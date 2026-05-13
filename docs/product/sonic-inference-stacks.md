# SONIC Inference Stacks

This note separates the SONIC inference paths that are easy to blur together when
reading the repo.

## Overview

| Stack | Runtime | Primary inputs | Primary outputs | What roboharness does |
|---|---|---|---|---|
| Upstream GEAR-SONIC deploy stack | C++ + TensorRT | `planner_sonic.onnx` or `model_encoder.onnx` + `model_decoder.onnx` | Low-level motor targets on the robot | Not shipped here; this is NVIDIA's reference deployment path |
| Roboharness planner path | Python + ONNX Runtime CPU | Planner context (`context_mujoco_qpos`) + command inputs such as velocity, mode, facing/movement directions, and height | Future MuJoCo `qpos` trajectory from `planner_sonic.onnx`, then 29 MuJoCo-order joint targets | Builds planner inputs, interpolates 30 Hz planner output to 50 Hz control, returns joint targets |
| Roboharness tracking path | Python + ONNX Runtime CPU | Encoder `obs_dict` (1762D) built from a motion clip window, decoder `obs_dict` (994D) built from token state + robot history | 29 normalized SONIC actions from `model_decoder.onnx`, decoded to MuJoCo-order joint targets | Packs clip/state observations, converts MuJoCo order to IsaacLab order for model inputs, decodes policy output back to MuJoCo joint targets |

## Planner vs Encoder+Decoder

`planner_sonic.onnx` and `model_encoder.onnx`/`model_decoder.onnx` are not interchangeable.

- Planner mode is command-conditioned locomotion.
  It consumes recent 36D MuJoCo `qpos` context and high-level locomotion inputs, then predicts a future full-body trajectory.
- Tracking mode is motion-conditioned whole-body control.
  It consumes a future motion window plus recent robot histories and returns a 29D action vector that must still be decoded with SONIC default angles and action scales.

In roboharness, tracking uses the published model contracts:

- Encoder input: one `obs_dict` tensor with shape `(1, 1762)`
- Encoder output: `encoded_tokens` with shape `(1, 64)`
- Decoder input: one `obs_dict` tensor with shape `(1, 994)`
- Decoder output: `action` with shape `(1, 29)`

## Joint Order And Clip Format

Roboharness clip CSVs are stored in MuJoCo order because that matches the simulator-facing API in this repo:

- `joint_positions.csv`: `(N, 29)` MuJoCo order
- `joint_velocities.csv`: `(N, 29)` MuJoCo order
- `root_height.csv`: `(N, 1)`
- `root_rotation_6d.csv`: `(N, 6)` row-wise flatten of the first two columns of the root rotation matrix

The SONIC policy observations use IsaacLab joint ordering internally. The controller converts MuJoCo-order clip/state data to IsaacLab order before ONNX inference, then converts the 29D decoder action back into MuJoCo-order joint targets for the caller.

## Validation Policy

For supported SONIC paths in this repo, real-model execution is the default validation standard.

- Mocked ONNX sessions remain in `tests/test_locomotion.py` for narrow unit checks such as packing, ordering, and output decoding.
- Real-model validation lives in `tests/test_sonic_real_models.py` and runs both planner mode and tracking mode against the published HuggingFace ONNX files.
- `roboharness[dev]` includes `huggingface_hub` and `onnxruntime`, so `pytest -q` exercises the real SONIC Python path in the normal repo test workflow.

This repo does not claim that the Python path is the same implementation as NVIDIA's C++/TensorRT deploy stack. It claims that the Python path it ships is explicit, separate, and validated against the real models.
