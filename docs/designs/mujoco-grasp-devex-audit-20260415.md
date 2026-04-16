# MuJoCo Grasp DX Audit

Date: 2026-04-15

Reviewed surfaces:
- Public landing page: `https://miaodx.com/roboharness/`
- Public report page: `https://miaodx.com/roboharness/grasp/`
- Local quick-start flow from `README.md`
- Local generated report from `examples/mujoco_grasp.py --report`
- Contributor and environment docs under `README.md`, `CONTRIBUTING.md`, and `docs/development-workflow.md`

Review branch:
- `codex/design-review-mujoco-20260415-docs`

## Outcome

This is a real developer product, not demo theater. The example can produce a useful
artifact quickly, the generated report is actionable, and the repo has enough
contributor structure to feel maintained.

But the first-five-minutes story still has two sharp edges:

1. The public landing page is a showcase, not a getting-started surface. It points
   developers at reports, not at the first command to run.
2. The documented headless MuJoCo path is brittle. In this environment, the
   README-recommended `MUJOCO_GL=osmesa` path failed immediately with a low-level
   OpenGL traceback, while `MUJOCO_GL=egl` succeeded end-to-end.

That means the product is good, but the golden path still leaks implementation detail.

## Tested Commands

Successful help path:

```bash
python examples/mujoco_grasp.py --help
```

Broken docs-first headless path in this environment:

```bash
MUJOCO_GL=osmesa python examples/mujoco_grasp.py --output-dir /tmp/roboharness-devex-review/grasp --report
```

Observed result:
- failed in `0.18s`
- traceback ended in `AttributeError: 'NoneType' object has no attribute 'glGetError'`

Successful alternate headless path in this environment:

```bash
MUJOCO_GL=egl python examples/mujoco_grasp.py --output-dir /tmp/roboharness-devex-review/grasp-egl --report
```

Observed result:
- completed in `1.26s`
- produced `report.html`, `autonomous_report.json`, `alarms.json`, and `phase_manifest.json`

## Getting Started Audit

GETTING STARTED AUDIT
=====================
Step 1: Open the public landing page. Time: ~30s. Friction: medium.
Evidence: live homepage screenshot and text extraction.

Step 2: Discover how to actually run the demo. Time: ~30-60s. Friction: medium.
Evidence: the live landing page links to reports, the repo, and build logs, but does
not expose an install CTA or copy-paste quick start.

Step 3: Find the quick-start command in `README.md`. Time: ~30s. Friction: low.
Evidence: `README.md` provides a concrete `pip install roboharness[demo]` plus
`python examples/mujoco_grasp.py --report`.

Step 4: Run the documented headless command. Time: `0.18s`. Friction: high.
Evidence: the `MUJOCO_GL=osmesa` path failed immediately with a raw PyOpenGL traceback.

Step 5: Retry with `MUJOCO_GL=egl`. Time: `1.26s`. Friction: medium.
Evidence: the alternate backend generated the full artifact pack successfully.

TOTAL:
- 5 steps
- first attempted feedback in `0.18s`
- first successful local artifact in `1.26s` after backend correction
- clean-install time not measured because demo dependencies were already installed

## Key Findings

### 1. The public homepage does not close the loop to hello world

Impact: High

What happened:
- The public landing page is strong as a report gallery.
- It is weak as a docs entrypoint.
- A new developer sees demo tiles, repo link, and build logs, but no install CTA, no
  quick-start snippet, and no docs search.

Why it matters:
- The first page should tell a developer what to do next.
- Right now it mostly tells them what already shipped.

Fix:
- Add one visible getting-started block to the public landing page:
  `pip install roboharness[demo]`
  `MUJOCO_GL=egl python examples/mujoco_grasp.py --report`
- Link directly to the README quick start or a dedicated docs page.
- Add a simple `Getting Started` CTA beside the report gallery.

### 2. The README headless MuJoCo recommendation is brittle

Impact: High

What happened:
- `README.md` tells headless Linux and CI users to prefix the command with
  `MUJOCO_GL=osmesa`.
- In this environment that exact path failed immediately.
- `MUJOCO_GL=egl` worked without code changes.

Why it matters:
- The golden path should not require developers to know MuJoCo backend trivia before
  they see value.
- This is the kind of failure that burns trust in minute one.

Fix:
- Update the quick-start note to document `egl` and `osmesa` as an environment matrix
  instead of a single global recommendation.
- Prefer the backend that matches current supported environments, or auto-detect when
  possible.
- When backend initialization fails, catch the low-level exception and print
  `problem + cause + fix`, not only the raw traceback.

### 3. The GPU workflow doc includes an invalid extra name

Impact: High

What happened:
- `docs/development-workflow.md` tells developers to install
  `.[demo,dev,wbc,unitree]`.
- `pyproject.toml` defines `demo`, `dev`, `lerobot`, and `wbc`, but not `unitree`.

Why it matters:
- This is a straight paper cut in the setup path.
- Bad setup commands cost more trust than missing commands because developers assume the
  docs are authoritative.

Fix:
- Replace the invalid extra with the actual supported combination.
- Keep one source of truth for install combinations and reuse it across docs.

### 4. The generated report is useful, but the public artifact is heavy

Impact: Medium

What happened:
- The live homepage downloaded quickly.
- The public MuJoCo grasp report was about `1.1 MB` and took `27.88s` to fully
  download from this environment.
- The local report rendered correctly over localhost and stayed responsive at `375px`,
  but the public page was unstable enough that the headless browser helper repeatedly
  dropped the session.

Why it matters:
- Self-contained HTML is portable and good for CI artifacts.
- It also increases download cost and makes generic tooling struggle.

Fix:
- Keep the self-contained report for portability.
- Consider a lighter summary mode for the first viewport, with deferred loading or
  smaller payloads for heavier media.
- Preserve the current `Current vs Baseline` summary contract while making the first
  artifact cheaper to fetch.

### 5. Error quality is mixed

Impact: Medium

What happened:
- CLI flag errors are crisp and immediate via argparse.
- Runtime backend errors are raw tracebacks with no friendly explanation.
- The public 404 page is the stock GitHub Pages screen, not a product-specific recovery
  page.

Why it matters:
- Good DX is not just happy-path speed. It is recovery speed.
- Developers need to know what broke, why, and what to try next.

Fix:
- Wrap backend/bootstrap failures with actionable copy.
- Add a project-specific 404 page that links back to the demo index and repo.

### 6. Community and measurement are present, but light

Impact: Low

What happened:
- The repo has issue templates and an active issue tracker.
- GitHub Discussions are disabled.
- The public site did not expose search, analytics, or feedback affordances in this
  audit.

Why it matters:
- This is enough for an early technical project.
- It is not enough if the goal is broader developer adoption and fast DX feedback loops.

Fix:
- Keep the issue templates.
- Add a lightweight docs feedback path or contact link.
- Consider enabling Discussions once community traffic justifies it.

## DX Live Audit Scorecard

+====================================================================+
|              DX LIVE AUDIT - SCORECARD                             |
+====================================================================+
| Dimension            | Score | Evidence                          | Method   |
|----------------------|-------|-----------------------------------|----------|
| Getting Started      | 6/10  | homepage + README + local runs    | TESTED   |
| API/CLI/SDK          | 7/10  | `--help`, runtime command surface | PARTIAL  |
| Error Messages       | 5/10  | argparse, traceback, public 404   | PARTIAL  |
| Documentation        | 6/10  | README, homepage, docs structure  | TESTED   |
| Upgrade Path         | 6/10  | changelog + deprecation signals   | INFERRED |
| Dev Environment      | 5/10  | setup docs + extras mismatch      | INFERRED |
| Community            | 6/10  | issue templates + issue tracker   | PARTIAL  |
| DX Measurement       | 4/10  | issue templates, no docs loop     | INFERRED |
+--------------------------------------------------------------------+
| TTHW (measured)      | <1 min to local artifact after backend fix  | TESTED   |
| Overall DX           | 6/10                                        |          |
+====================================================================+

Notes:
- No formal `plan-devex-review` log entry was present during this audit, so there is
  no structured boomerang table.
- Informally, the shipped experience under-delivers the approved phase-2 DX intent in
  two places: the public getting-started surface and backend-specific recovery guidance.

## Evidence Highlights

Live homepage:
- The page is polished and findable as a report gallery.
- It does not expose install or getting-started copy.

Local successful report:
- The generated summary does the core job:
  `PASS/no failed phase`, `Current vs Baseline`, and `Agent Next Action` are all in the
  first screen.
- Mobile overflow on the local report is fixed; `scrollWidth == innerWidth == 375`.

Console / browser behavior:
- The local report emitted repeated WebGL performance warnings tied to `ReadPixels`
  during scene rendering.
- The public `grasp/` page was materially heavier than the homepage and less reliable
  in headless browsing.

## Upgrade Path Audit

What is good:
- `CHANGELOG.md` is current and user-facing.
- The latest entry clearly names the MuJoCo grasp report fixes and added regression
  tests.

What is missing:
- No migration guide surfaced in the top-level contributor flow.
- No documented deprecation policy in the main onboarding path.
- No codemod or upgrade helper surface, which is fine today but will matter once the
  public API broadens.

## Developer Environment Audit

What is good:
- `CONTRIBUTING.md` gives a compact dev setup and the correct default verification loop.
- `docs/development-workflow.md` explains the CPU/GPU split clearly and makes the
  constraints explicit.

What is weak:
- The invalid `unitree` extra in the GPU setup example is a real setup trap.
- The main quick-start flow still leaks backend selection detail into onboarding.

## Community & Ecosystem Audit

What is good:
- Bug and feature issue templates exist.
- Recent open issues show the repo is active.

What is weak:
- No Discussions surface.
- No project-specific community hub from the public landing page.

## DX Measurement Audit

What exists:
- GitHub issue templates for bugs and feature requests.
- CI-generated public reports on every push to main.

What does not yet exist:
- Docs search on the public landing page.
- Public docs feedback widget or explicit feedback CTA.
- Visible docs analytics or onboarding instrumentation in the audited surfaces.

## Next Steps

1. Fix the golden path first.
   - Update `README.md` and the public landing page so the first visible command matches
     what actually works in supported headless environments.

2. Fix the broken environment doc command.
   - Remove or replace the invalid `unitree` extra in
     `docs/development-workflow.md`.

3. Improve runtime recovery messages.
   - Turn backend-init failures into actionable guidance instead of a raw traceback.

4. Re-run `/devex-review` after those fixes.
   - This is a clean boomerang candidate because the sharp edges are now concrete and
     measurable.
