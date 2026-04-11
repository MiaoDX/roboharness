# ROS Discourse Post: Visual Testing Harness for AI Coding Agents

_Context: Issue [miaodx/roboharness#149](https://github.com/MiaoDX/roboharness/issues/149) — post on ROS Discourse for community feedback._

**Post in:** https://discourse.ros.org/ → **General** category

**Title:** `Visual testing harness for AI coding agents in robot simulation — looking for feedback`

---

## Full Post Content

Hi ROS community,

**A question for those using Claude Code, Codex, or other AI coding agents for robotics work:** how do you debug simulation behavior when the agent is writing the control code?

My specific problem: when I used Claude Code to write MuJoCo control scripts, it could read error logs and joint angles, but it couldn't _see_ what the robot was actually doing. The agent would iterate on code that looked plausible but produced obviously wrong behavior — wrong grasp orientations, unstable footing, arm trajectories that clipped through geometry. These failures are trivial for a human to spot in a viewer, but invisible to a text-only agent.

---

### What I built: roboharness

[roboharness](https://github.com/MiaoDX/roboharness) is a visual testing harness that pauses simulation at named checkpoints and captures multi-view screenshots alongside structured JSON state. The agent reads these files directly — no separate VLM inference step needed.

**MuJoCo grasp demo** (front view — Plan → Pregrasp → Approach → Close → Lift → Holding):

![MuJoCo grasp front view](https://raw.githubusercontent.com/MiaoDX/roboharness/main/assets/X32_Y28_Z13_front_view.gif)

**[→ Live Report (auto-generated from CI on every push)](https://miaodx.com/roboharness/grasp/)**

The report shows 6 checkpoints from pre-grasp to object-in-hand. CI regenerates it via GitHub Actions on every push using `MUJOCO_GL=osmesa` for headless rendering.

The core pattern is two lines to wrap any Gymnasium environment:

```python
from roboharness.wrappers import RobotHarnessWrapper

env = RobotHarnessWrapper(env,
    checkpoints=[{"name": "pre_grasp", "step": 50}, {"name": "lift", "step": 120}],
    output_dir="./harness_output",
)
```

At each checkpoint, roboharness saves:
- PNG screenshots from all configured cameras (front, side, wrist, top-down)
- `state.json` — joint positions, velocities, ctrl commands
- `metadata.json` — sim_time, step index, camera list

The AI agent reads these files and reasons about what to change next.

Or with the lower-level API:

```python
from roboharness import Harness
from roboharness.backends.mujoco_meshcat import MuJoCoMeshcatBackend

backend = MuJoCoMeshcatBackend(model_path="robot.xml", cameras=["front", "side"])
harness = Harness(backend, output_dir="./output")

harness.add_checkpoint("pre_grasp")
harness.add_checkpoint("lift")
result = harness.run_to_next_checkpoint(actions)
# result.views → multi-view screenshots, result.state → joint angles + velocities
```

---

### Current status

| Simulator | Status |
|---|---|
| MuJoCo (native backend) | ✅ Implemented — headless via `MUJOCO_GL=osmesa` or `egl` |
| Gymnasium wrapper | ✅ Works with Isaac Lab, ManiSkill, LeRobot, etc. |
| LeRobot (`make_env()` factory) | ✅ Implemented |
| **Gazebo / ROS2** | 📋 Planned — not yet implemented |

---

### Questions for this community

The Gazebo integration is the obvious next step for ROS users, and I'd genuinely value input before committing to an approach:

1. **Capture method:** For screenshot capture from Gazebo, which do you prefer in practice — subscribing to `/camera/image_raw` via a ROS2 node, or using Gazebo's native snapshot API (where available)? Or is there a third approach that's more CI-friendly?

2. **State source:** Is TF2 the right source for robot state (end-effector poses, joint frames) in ROS2 context? Would you also expect `/joint_states` to be exposed, or is TF sufficient?

3. **CI renderer:** If you run headless Gazebo in CI today, which renderer are you using — `--headless-rendering` with WebGL, OSMesa, or something else? Osmesa works reliably for MuJoCo; curious whether the same holds for Gazebo.

4. **Use case fit:** Does AI-agent-driven robot simulation development cause debugging pain in your workflow? Or is this solving a problem you don't actually have? Honest answer appreciated — I'd rather know the use case is wrong than optimize for the wrong thing.

---

GitHub: https://github.com/MiaoDX/roboharness  
MIT License, Python 3.10+, numpy-only core (MuJoCo/Meshcat optional)

Happy to answer technical questions. Looking for feedback, not promoting — I'm genuinely trying to understand whether this is useful to ROS developers and what the right Gazebo integration path looks like.

---

## Posting Notes

- Post in **General** category on https://discourse.ros.org/
- ROS Discourse account required (login with GitHub)
- Add tags: `simulation`, `testing`, `ai-agents` (if tag system allows)
- The GIF is embedded via raw GitHub URL — it will render inline on Discourse
- Tone check: the four questions at the end must feel like real questions, not softening before a pitch. Edit them if they read as rhetorical
- After posting, note the Discourse thread URL in a comment on issue #149 so the response thread is tracked

---

## Related issues

- **#152** — Gazebo/ROS2 showcase (the implementation that this post is seeking feedback for)
- **#150** — ros2_mcp collaboration discussion (complementary tool, different angle)
- **#146** — awesome-ros2 listing (deferred — requires forking external repo)
