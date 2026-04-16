<!-- /autoplan restore point: /home/mi/.gstack/projects/MiaoDX-roboharness/dev-0416-autoplan-restore-20260416-132559.md -->

# Plan: Real LeRobot Evaluation Plugin Integration

## Problem

The `roboharness[lerobot]` optional dependency and `lerobot_plugin.py` skeleton exist, but they do not actually integrate with real LeRobot. `examples/lerobot_eval_harness.py` uses a random CartPole policy, and `lerobot_plugin.py` is a generic Gymnasium evaluator. A LeRobot user cannot run one command to get visual regression testing in CI.

## Goal

Make the LeRobot plugin load and evaluate real LeRobot policies from checkpoints, producing checkpoint screenshots + structured JSON + CI pass/fail gates.

## Premises

1. LeRobot users want headless visual evaluation in CI (P1 — the cited issues are closed, but the underlying need for screenshots + reproducibility + CI gates persists)
2. Using LeRobot's `make_env()` + policy loading utilities with roboharness's existing `evaluate_policy()` loop is cleaner than wrapping `eval.py` (P2)
3. The plugin should work with standard LeRobot checkpoint formats (Diffusion, ACT, TDMPC) without per-policy-type code (P3)
4. A smoke-testable example with a small pre-trained checkpoint is essential for DX (P4)

## Scope — IN

1. **Generic `PolicyAdapter` Protocol** (`src/roboharness/evaluate/__init__.py` or new `protocol.py`)
   - Define `PolicyAdapter = Callable[[np.ndarray | dict[str, Any]], np.ndarray]`

2. **LeRobot policy loading adapter** (`src/roboharness/evaluate/lerobot_policy_adapter.py`)
   - Load policy from LeRobot checkpoint directory
   - Normalize policy inference to `obs → action` callable implementing `PolicyAdapter`
   - Handle CPU inference via `map_location="cpu"`
   - Wrap loading errors with clear, actionable exceptions

3. **Shared LeRobot env utilities** (`src/roboharness/evaluate/lerobot_env.py`)
   - Extract `create_native_env()`, `_patch_config_for_headless()`, `_try_lerobot_make_env()`, `_fallback_hub_make_env()`, `_add_mujoco_rendering()` from `examples/lerobot_g1_native.py`
   - Make env creation reusable across examples

4. **Integrate with LeRobot eval flow** (`src/roboharness/evaluate/lerobot_plugin.py`)
   - Keep the existing generic `evaluate_policy()`, `LeRobotEvalConfig`, and `LeRobotEvalReport` intact (public API)
   - Add `evaluate_lerobot_policy(checkpoint_path, repo_id, ...)` as a new LeRobot-specific entry point that delegates to `evaluate_policy()`
   - Add lightweight action shape validation before `env.step()`
   - Insert visual checkpoints at configured steps during episode rollouts
   - Produce `LeRobotEvalReport` with per-episode success, rewards, checkpoint paths

5. **Update example** (`examples/lerobot_eval_harness.py`)
   - Accept `--checkpoint-path` and `--repo-id` args
   - Validate checkpoint path early; exit 1 with a clear message if invalid
   - Fall back to CartPole random only when no checkpoint provided
   - Print one-command copy-paste usage for a real LeRobot checkpoint

6. **Tests**
   - `tests/test_lerobot_policy_adapter.py` — mock LeRobot policy loading: happy path, CPU inference, missing checkpoint, loading failure wrapping
   - `tests/test_lerobot_env.py` — mock env creation: `make_env()` success, ImportError fallback, exception fallback, headless patching, MuJoCo rendering injection
   - `tests/test_lerobot_eval_plugin.py` — existing tests + `evaluate_lerobot_policy()` integration + action shape validation
   - `tests/test_lerobot_eval_harness.py` — example CLI paths: CartPole fallback, invalid checkpoint, threshold pass/fail

7. **Docs update**
   - `README.md` section: "LeRobot Evaluation in CI"
   - One-command example in `docs/roadmap-2026-q2.md` status

8. **Dependencies**
   - Add `torch` to `roboharness[lerobot]` optional dependencies in `pyproject.toml`

## Scope — NOT IN

- Training policies (roboharness is a test harness, not a trainer)
- Multi-GPU distributed eval
- Real-time teleoperation or remote control
- VLM judge integration (deferred to later roadmap item D)
- Replacing the generic `evaluate_policy()` public API with a LeRobot-specific one
- Routing through `RobotHarnessWrapper` as the primary eval path (kept as a future option)

## Existing code to leverage

- `src/roboharness/evaluate/lerobot_plugin.py` — config, report, threshold checks
- `src/roboharness/_utils.py` — `save_image` for checkpoint screenshots
- `tests/test_lerobot_eval_plugin.py` — fake env test patterns
- `examples/lerobot_g1_native.py` — proven native LeRobot integration via `make_env()` + `RobotHarnessWrapper`
- `tests/test_lerobot_native_compat.py` and `tests/test_lerobot_g1_compat.py` — test coverage for the native wrapper path
- `src/roboharness/wrappers/vector_env_adapter.py` — `SyncVectorEnv` normalization

## Implementation approach

**Approach A: Use `make_env()` + existing `evaluate_policy()` loop (chosen)**
- Pros: Reuses roboharness's battle-tested rollout logic, avoids coupling to LeRobot's unstable `eval.py`, checkpoint injection is trivial
- Cons: Need to handle LeRobot env creation and policy loading ourselves
- Effort: ~1 session

**Approach B: Wrap LeRobot `eval.py` (rejected)**
- Pros: Uses LeRobot's battle-tested rollout logic, handles different policy types automatically
- Cons: Deep coupling to unstable internal API, monkey-patching required for checkpoint screenshots
- Rejected: violates P2

## Files to modify / create

- CREATE: `src/roboharness/evaluate/lerobot_policy_adapter.py`
- CREATE: `src/roboharness/evaluate/lerobot_env.py`
- EDIT: `src/roboharness/evaluate/lerobot_plugin.py`
- EDIT: `examples/lerobot_eval_harness.py`
- EDIT: `examples/lerobot_g1_native.py`
- EDIT: `tests/test_lerobot_eval_plugin.py`
- CREATE: `tests/test_lerobot_policy_adapter.py`
- CREATE: `tests/test_lerobot_env.py`
- CREATE: `tests/test_lerobot_eval_harness.py`
- EDIT: `README.md`
- EDIT: `docs/roadmap-2026-q2.md`
- EDIT: `pyproject.toml`

## Data flow

```
LeRobot checkpoint directory
        │
        ▼
┌─────────────────────────┐
│  LeRobotPolicyAdapter   │  ← loads policy, normalizes to obs → action
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│    create_native_env    │  ← make_env() + VectorEnvAdapter + rendering
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│    evaluate_policy()    │  ← existing rollout loop with checkpoint capture
└─────────────────────────┘
        │
        ├── checkpoint screenshots → output_dir/episode_*/step_*/
        │
        ▼
┌─────────────────────────┐
│    LeRobotEvalReport    │  ← JSON with per-episode stats
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  check_eval_threshold   │  ← CI pass/fail gate
└─────────────────────────┘
```

## Test plan

1. `tests/test_lerobot_policy_adapter.py` — unit tests with mocked LeRobot policy and torch (no `lerobot` import needed)
2. `tests/test_lerobot_env.py` — unit tests with mocked `make_env()`, config patching, and MuJoCo rendering (no `lerobot` or `mujoco` import needed)
3. `tests/test_lerobot_eval_plugin.py` — extend existing tests with `evaluate_lerobot_policy()` integration and action shape validation
4. `tests/test_lerobot_eval_harness.py` — test example CLI paths by importing as a module with mocked argv/policy/env
5. Integration test: run `examples/lerobot_eval_harness.py --env CartPole-v1` (no checkpoint)
6. If a small LeRobot checkpoint is available locally: manual smoke test with real policy
7. CI: ensure existing tests pass + new tests added; coverage must not drop below 90%

## Risks

- LeRobot API changes between versions (mitigation: pin min version in `[lerobot]` extra)
- Checkpoint formats vary across policy types (mitigation: use LeRobot's own loading utilities)
- Large checkpoint downloads in CI (mitigation: tests use mocks; example is optional; existing CI caches `~/.cache/huggingface/hub`)

---

## Phase 1: CEO Review

### What this plan is NOT

- It is not a training framework (correctly excluded).
- It is not a new abstraction layer for all simulators (the plan stays within `evaluate/`).
- It is not a guaranteed wedge: if LeRobot ships native screenshot + JSON export, the parity features here become commodity.

### What already exists

- `roboharness[lerobot]` optional dependency and generic `lerobot_plugin.py` skeleton.
- `examples/lerobot_g1_native.py` — proven native LeRobot integration via `make_env()` + `RobotHarnessWrapper`.
- `tests/test_lerobot_native_compat.py` and `tests/test_lerobot_g1_compat.py` — test coverage for the native wrapper path.
- README already advertises native LeRobot support.

### Dream-state delta

A LeRobot user can run one command to evaluate a real checkpoint in CI and get visual checkpoints + structured JSON + pass/fail gates. Today, the plugin only supports generic Gymnasium envs with a random CartPole fallback.

### CEO consensus table

| Question | Claude CEO | Codex CEO | Consensus |
|----------|------------|-----------|-----------|
| Is this the right problem? | Yes, but frame as generic `PolicyAdapter` protocol | Yes, but productize existing native integration first | **Agree on direction; disagree on tactical surface** |
| Are premises valid? | P2/P3 fragile; eval.py is unstable | P1 is stale (issues closed); P2 is optimistic | **P1 and P2 are the weakest premises** |
| Scope / 6-month risk | Well-bounded; risk is LeRobot copies it | One-session estimate is fantasy; CI story is theatre | **Scope is safe, execution estimate is optimistic** |
| Biggest missed alternative | Generic protocol instead of LeRobot-specific plugin | Upstream recipe/showcase instead of new plugin surface | **Both favor reuse over new plugin-specific API** |
| Verdict | Proceed with reframing | Proceed with tightened scope + adoption loop | **Proceed, but do not treat premises as settled** |

### Premise gate

1. **P1 — LeRobot users want headless visual evaluation in CI**  
   *Status:* PARTIALLY VALID. The cited issues (#538, #2375) are closed, but the underlying need (screenshots + reproducibility + CI gates) persists in the community. The plan should not lean on stale issues as primary validation.

2. **P2 — Using LeRobot's `make_env()` + existing `evaluate_policy()` is better than wrapping `eval.py`**  
   *Status:* VALID. Avoids coupling to an unstable internal API and reuses the battle-tested roboharness rollout loop.

3. **P3 — Standard checkpoint formats without per-policy-type code**  
   *Status:* VALID. Using LeRobot's own loading utilities is the right mitigation.

4. **P4 — Smoke-testable example with a small pre-trained checkpoint is essential for DX**  
   *Status:* VALID. The current CartPole fallback undermines the product story; a real checkpoint example is required.

**CEO verdict:** Proceed to implementation with the reframed approach.

---

## Phase 2: Engineering Review

### Architecture decisions

- **Do NOT wrap LeRobot `eval.py`.** Use `make_env()` + existing `evaluate_policy()` loop instead.
- **Add a generic `PolicyAdapter` Protocol.** `LeRobotPolicyAdapter` implements it; future framework integrations can reuse the protocol.
- **Extract shared env utilities.** `lerobot_env.py` holds `create_native_env()` and related helpers, reused by both `lerobot_eval_harness.py` and `lerobot_g1_native.py`.
- **Keep generic API intact.** `evaluate_policy()`, `LeRobotEvalConfig`, and `LeRobotEvalReport` remain public. `evaluate_lerobot_policy()` is an additive LeRobot-specific entry point.
- **Broaden policy interface.** Accept `np.ndarray | dict[str, Any]` so LeRobot policies with dict observations work without preprocessing hacks.

### Code quality decisions

- Rename planned `lerobot_loader.py` → `lerobot_policy_adapter.py` for clarity.
- Add explicit checkpoint path validation in the example (exit 1 with clear message if invalid).
- Add lightweight action shape validation in `evaluate_policy()` before `env.step()`.
- Wrap checkpoint loading errors with clear, actionable exceptions.

### Test decisions

- Add `tests/test_lerobot_policy_adapter.py` with mocked loading branches.
- Add `tests/test_lerobot_env.py` with mocked `make_env()` and rendering branches.
- Add `tests/test_lerobot_eval_harness.py` for example CLI paths.
- Extend `tests/test_lerobot_eval_plugin.py` with integration and action-shape tests.

### Performance decisions

- Existing CI already caches HuggingFace models (`~/.cache/huggingface/hub`) in the `lerobot-g1-native-example` job.
- No additional `local_files_only` flag needed for this iteration.

### Dependency decisions

- Add `torch` to `roboharness[lerobot]` optional dependencies in `pyproject.toml`.

### Outside voice (Codex) — key additional concerns

- **Artifact schema mismatch:** `RobotHarnessWrapper` produces richer `state.json` + `metadata.json` + images than `lerobot_plugin.py`. *Resolution:* Keep `evaluate_policy()` as the primary path for now; consider unifying schemas in a future refactor.
- **CI gate under-specified:** No seed control or init-state policy. *Resolution:* Document that v1 is deterministic env evaluation; seeding and benchmark fixtures are future work.
- **Test plan too weak:** CartPole-only integration test is "fake coverage." *Resolution:* Approved 3 new test modules to cover all branches with mocks.
- **Policy interface too narrow:** `np.ndarray`-only doesn't handle dict observations. *Resolution:* Broadened to `np.ndarray | dict[str, Any]`.
- **Backward compatibility risk:** Public API exports would break if made LeRobot-specific. *Resolution:* Generic API kept intact; LeRobot path is additive.

### Failure modes

| Codepath | Realistic failure | Test? | Error handling? | User sees? |
|----------|-------------------|-------|-----------------|------------|
| `lerobot_policy_adapter.py` — checkpoint version mismatch | Loading raises exception | Yes | Yes (wrapped) | Clear error message |
| `lerobot_policy_adapter.py` — wrong action shape | Policy outputs wrong shape | Yes | Yes (validation) | Clear error before step |
| `lerobot_env.py` — `make_env()` fails, fallback fails | Both paths raise | Yes | Partial | LeRobot/MuJoCo error |
| `lerobot_env.py` — headless patching fails | `snapshot_download` or YAML error | Yes | Yes (early return) | Silent skip, then possible failure |
| `examples/lerobot_eval_harness.py` — invalid checkpoint path | Directory missing | Yes | Yes (validation + exit 1) | Clear error message |

**Critical gaps flagged:** 0

### Parallelization

Sequential implementation recommended. Policy adapter and env utilities are conceptually parallelizable, but both touch `evaluate/`, creating merge-conflict risk. Given the small scope, sequential is simpler.

| Step | Modules | Depends on |
|------|---------|------------|
| Policy adapter | `evaluate/lerobot_policy_adapter.py`, tests | — |
| Env utilities | `evaluate/lerobot_env.py`, `examples/lerobot_g1_native.py`, tests | — |
| Plugin integration | `evaluate/lerobot_plugin.py`, `examples/lerobot_eval_harness.py`, tests | Policy adapter, Env utilities |
| Docs + deps | `README.md`, `docs/roadmap-2026-q2.md`, `pyproject.toml` | Plugin integration |

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| CEO Review | `/plan-ceo-review` | Scope & strategy | 1 | — | 4 premises evaluated, proceed with reframing |
| Codex Review | `/codex review` | Independent 2nd opinion | 1 | issues_found | Artifact schema mismatch, backward compat, dict obs, test gaps |
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | issues_open | 13 issues found, 0 unresolved, 0 critical gaps |
| Design Review | `/plan-design-review` | UI/UX gaps | 0 | — | — |
| DX Review | `/plan-devex-review` | Developer experience gaps | 0 | — | — |

- **UNRESOLVED:** 0
- **VERDICT:** ENG REVIEW COMPLETE — all issues resolved in conversation. Plan updated and ready for implementation.
