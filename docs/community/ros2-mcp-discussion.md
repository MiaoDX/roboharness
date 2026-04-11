# ros2_mcp Collaboration Discussion: Prepared Content

_Context: Issue [miaodx/roboharness#150](https://github.com/MiaoDX/roboharness/issues/150) — open a collaboration discussion on the ros2_mcp repository._

**Post at:** https://github.com/LCAS/ros2_mcp — open a **Discussion** (or Issue if Discussions are disabled)

**Title:** `Integration idea: roboharness for checkpoint-based visual testing alongside ros2_mcp real-time interaction`

---

## Full Discussion Content

Hi ros2_mcp team,

I maintain [roboharness](https://github.com/MiaoDX/roboharness) — a visual testing harness for AI coding agents in robot simulation. I wanted to open a discussion about a potential complementary integration, since the two tools appear to be solving different but related problems in the "AI agent + robotics" space.

### What ros2_mcp does (as I understand it)

ros2_mcp exposes ROS2 topic interaction — subscribe, publish, read joint states, call services — over the MCP protocol. This lets AI coding agents **actively control** a running ROS2 system in real-time.

### What roboharness MCP does

roboharness exposes **checkpoint-based visual testing** over MCP. Rather than real-time interaction, it pauses the simulation at named moments to capture multi-view screenshots and structured state, then lets the agent evaluate whether the robot behaved correctly:

```
capture_checkpoint  → multi-view PNGs + state.json at a named moment
evaluate_constraints → pass/fail verdict with per-metric details
compare_baselines   → regression detection against historical runs
evaluate_batch_trials → aggregate CI-ready success rate across N trials
```

The output is designed for AI agents to read: PNG screenshots the agent can see, JSON state it can parse.

### Why these seem complementary, not competing

ros2_mcp is about **interaction**: the agent drives the robot, reads sensor data, reacts to events in real-time.

roboharness is about **evaluation**: the agent pauses, looks at what happened, checks constraints, and decides whether the outcome was correct.

A combined workflow might look like this:

```
1. [ros2_mcp] Subscribe to /joint_states and /cmd_vel
2. [ros2_mcp] Publish nav goal → robot starts moving
3. [roboharness] capture_checkpoint("pre_approach") → save front + top-down PNGs + pose
4. [ros2_mcp] Monitor /odom until robot reaches waypoint
5. [roboharness] capture_checkpoint("at_waypoint") → verify pose constraints
6. [roboharness] evaluate_constraints → did joint limits hold? Was the final pose within 5cm of goal?
7. [roboharness] compare_baselines → is this run better or worse than the last 5?
```

The agent uses ros2_mcp to operate the robot, and roboharness to judge whether the operation succeeded.

### Integration question

A few things I'd find useful to understand from your side:

1. **Simulation vs. real robot:** Does ros2_mcp work exclusively with real ROS2 systems, or does it also work with Gazebo / Isaac Sim publishing topics? Roboharness currently targets simulation (MuJoCo, Isaac Lab, Gymnasium-based envs); the overlap is probably Gazebo + Isaac Sim.

2. **State capture overlap:** ros2_mcp presumably exposes `/joint_states` and TF data. Roboharness also captures joint state at checkpoints. Is there a clean way to let roboharness read from ros2_mcp's ROS2 connection rather than maintaining a separate one? Or does it make more sense to keep them independent?

3. **Checkpoint trigger from ROS events:** Ideally, `capture_checkpoint` would be triggered by a ROS event (e.g., "capture when the gripper closes" = subscribe to a topic and fire on state change). Is that a pattern you'd consider exposing in ros2_mcp?

I'm not proposing a merge — both tools have different scopes. But if there's appetite for a combined example (e.g., "use ros2_mcp to drive a Gazebo arm + roboharness to evaluate grasp quality"), I'd be happy to contribute one.

GitHub: https://github.com/MiaoDX/roboharness  
roboharness MCP server docs: https://github.com/MiaoDX/roboharness/tree/main/src/roboharness/mcp  
MIT License, Python 3.10+

---

## Posting Notes

- **Where:** https://github.com/LCAS/ros2_mcp — use the Discussions tab if available; otherwise open an Issue
- **Tone check:** The three questions at the bottom should read as genuine technical questions, not rhetorical ones. Edit if they don't.
- **Goal:** Establish a connection and get feedback on whether a combined workflow example is useful — not request any code changes on their side
- **Follow-up:** After posting, add the discussion/issue URL as a comment on [issue #150](https://github.com/MiaoDX/roboharness/issues/150) so the response thread is tracked

---

## Related issues

- **#149** — ROS Discourse post (broader ROS community outreach, different angle)
- **#152** — Gazebo/ROS2 showcase (the implementation that would make this integration concrete)
- **#146** — awesome-ros2 listing (deferred — requires external repo PR)
