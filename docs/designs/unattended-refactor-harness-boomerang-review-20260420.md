# Unattended Refactor Harness Boomerang Review

Date: 2026-04-20

Reviewed surfaces:
- `README.md` front door and quick-start copy
- installed package metadata from `pyproject.toml` and the local wheel install
- `python examples/demos/mujoco/grasp.py --help`
- deterministic fail/pass `report.html` artifacts generated from the fixture-backed
  MuJoCo wedge helpers and rendered through gstack browse at desktop and mobile widths

## Outcome

The shipped approval report held its product contract.

The first screen still reads like an approval queue instead of a dashboard:
`Run Decision -> Approval Queue -> Hard Metric Results -> Current vs Baseline ->
Agent Next Action -> Phase Timeline`. The fail artifact kept both manifest-selected
evidence cards in order, the pass artifact suppressed unchanged cases cleanly, and
the mobile render stayed within the viewport with no horizontal overflow.

The one material boomerang finding was on the front door. The README told users to
`pip install roboharness[demo]` and then run `python examples/demos/mujoco/grasp.py --report`,
but the published distribution installs only the `roboharness` CLI and does not ship
the repo `examples/` directory. That made the top-level quick start source-only while
reading like a package-install path.

## Verified Positives

- The fail artifact rendered two evidence cards in manifest order at `390px` width.
- Mobile layout stayed bounded with `scrollWidth == innerWidth == 390`.
- The evidence surface still uses semantic `figure` / `figcaption` markup.
- Evidence alt text still includes phase, view, and current/baseline role.
- Pass-state copy, artifact-pack metadata, and baseline-authority copy stayed aligned.

## Finding 1. README front door conflated package install and repo demo

Impact: High

Observed:
- The top README quick start paired `pip install roboharness[demo]` with
  `python examples/demos/mujoco/grasp.py --report`.
- The installed distribution exposes one console script:
  `roboharness=roboharness.cli:main`.
- The installed distribution contains no `examples/` files.

Why it matters:
- The first copy-paste path should fail only for real environment reasons, not because
  the docs hid a required repo checkout.
- This is a minute-one trust break for the unattended-refactor story the wedge is
  supposed to sell.

Fix:
- Split the front door into `Package-First Integration` and `Repo Demo`.
- Make the repo-demo path explicitly clone-based and editable-install based.
- Point package users at the real entry path: the installed CLI plus wrapper / Harness
  integration.

## Follow-On Recommendation

`TODO 5` is now satisfied by this boomerang pass.

The next load-bearing move is still `TODO 6`: build the seeded evaluator corpus so the
approval queue can earn trust instead of only claiming it.
