# Post-hoc semantic snapshot evidence is first-class

Status: accepted

Roboharness will support post-hoc semantic snapshot evidence as a first-class
artifact path alongside the existing `SimulatorBackend.step()` harness loop. A
downstream project may run its own robot runtime, simulator launch topology,
hardware-live surface, safety checks, and controller transport, then hand
Roboharness an ordered semantic snapshot bundle. Roboharness owns the reusable
artifact language around that handoff: semantic snapshot bundles, renderer
reports, autonomous evidence reports, proof-pack assembly inputs, baseline
comparison surfaces, and bounded visual review preparation.

The `SimulatorBackend` protocol remains the simple package-first path for
step-oriented simulators. It is not the only way to use Roboharness. This keeps
hardware-style and deploy-live flows from pretending to be action-sequence
loops just to fit the original core API.

Project-specific execution stays outside Roboharness. The downstream project
owns task planning, runtime startup, robot model loading, safety evidence,
control backend selection, renderer implementation, and any domain-specific
semantic checks. Roboharness may validate and aggregate the artifacts those
systems emit, but it must not absorb the project runtime.

The first dogfood target is GR00T whole-body control. That repo already
demonstrates a useful flow: run a representative manipulation surface, record
semantic phase snapshots, replay those snapshots through Meshcat or MuJoCo,
then aggregate snapshot metrics, renderer trustworthiness, verdict reasons, and
failure taxonomy into an autonomous report. If Roboharness cannot express that
flow naturally, the Roboharness abstraction is incomplete.

Consequences:

- Add a public `roboharness.evidence` package for artifact models and JSON
  helpers.
- Keep `roboharness.core.Harness` and `SimulatorBackend` intact as one ingestion
  path rather than broadening them into downstream runtime orchestration.
- Keep `roboharness.approval` focused on paired evidence, visual review records,
  and approval summaries.
- Let project `HarnessContract` workflows refer to evidence expectations over
  time, but do not turn the contract into a renderer or runtime configuration
  system.
- Validate GR00T-style snapshot bundles and renderer reports through fixtures
  before migrating the GR00T runner itself.

Rejected alternatives:

- A GR00T-only adapter layer in the downstream repo. This would prove only that
  another wrapper can be written, not that Roboharness has the right reusable
  evidence abstraction.
- Forcing deploy-live and hardware-live evidence through `SimulatorBackend`.
  That would couple Roboharness to action-loop assumptions that are false for
  many real robot validation surfaces.
- Moving robot runtime/session ownership into Roboharness. That would expand
  scope beyond the evidence and approval layer and make each downstream robot
  integration harder to maintain.
