# TODOS

This file captures deferred work from approved planning and review artifacts.

## 1. Add temporal evidence when still images are ambiguous

- What: Add a short clip, scrubber, or temporal overlay for failure cases where paired
  still images are suggestive but not decisive.
- Why: The approved phase-2 plan explicitly keeps still-image evidence as the first
  wedge, but some motion-rooted failures will remain under-explained.
- Pros: Improves trust for trajectory and timing regressions, reduces false confidence
  from a single frame, and strengthens the agent rerun loop.
- Cons: Adds asset weight, UI complexity, and new decisions about encoding, playback,
  and report portability.
- Context: Deferred by the approved review in
  `docs/designs/mujoco-alarmed-grasp-loop-phase-2-plan-reviewed.md`. The current plan
  already introduces an explicit `ambiguous still-image evidence` state so this work
  has a clear trigger.
- Depends on / blocked by: Phase 2 implementation landing first, plus at least one real
  ambiguous failure observed in MuJoCo or a second stack.

## 2. Add click-to-zoom or lightbox support for evidence cards

- What: Let users enlarge the new phase-2 comparison cards without dropping into the
  deeper checkpoint gallery.
- Why: The approved plan keeps evidence cards passive to stay wedge-tight, but detailed
  image inspection may still benefit from a lighter-weight zoom path.
- Pros: Faster visual inspection, better use of the new evidence cards, less context
  switching between summary and gallery.
- Cons: Broadens the UI surface, adds focus/keyboard/a11y work, and duplicates some of
  the value of the existing checkpoint gallery.
- Context: Explicitly deferred in the approved design review because the deeper gallery
  already exists and the phase-2 bottleneck is proof ordering, not interaction chrome.
- Depends on / blocked by: Phase 2 evidence cards shipping first and proving useful
  enough to justify a richer interaction.

## 3. Sync README and docs to the new evidence contract

- What: Update the public docs so the MuJoCo grasp example explains the paired-evidence
  summary, explicit evidence states, and the intended `failed phase -> proof -> rerun`
  workflow.
- Why: The approved DX review intentionally kept docs sync out of the implementation
  phase, but the artifact contract should become discoverable once it ships.
- Pros: Better onboarding, less guesswork for new users, and alignment between the
  generated report and the documented workflow.
- Cons: More scope in the follow-up pass and a risk of documenting behavior before the
  implementation stabilizes.
- Context: Deferred in the approved DX review in favor of keeping phase 2 focused on
  the artifact pack itself.
- Depends on / blocked by: Phase 2 implementation merging first. Best follow-up path is
  a dedicated docs sync pass, such as `/document-release`.

## 4. Extract a shared evidence contract only after a second stack needs it

- What: Promote the phase-local evidence-pair resolver into a shared abstraction only
  when another task or simulator genuinely needs the same contract.
- Why: The approved CEO and engineering reviews both flagged premature extraction as
  strategy debt.
- Pros: Avoids calcifying a one-off example seam, keeps the first implementation
  explicit, and forces the abstraction to be justified by a second concrete use case.
- Cons: Leaves some local duplication in place for now and delays platform leverage.
- Context: The approved plan locks phase 2 to example-local code and states the
  extraction trigger explicitly.
- Depends on / blocked by: A second task or stack proving it needs the same
  `manifest-selected paired evidence` contract.

## 5. Run post-implementation boomerang reviews

- What: Run visual and developer-experience follow-up reviews after phase 2 is
  implemented.
- Why: The approved plan now makes concrete claims about first-viewport proof, headless
  usability, and time-to-understanding. Those claims should be checked against reality.
- Pros: Confirms the plan survived implementation, catches visual drift, and validates
  whether the live loop is actually easier to use.
- Cons: More review time after the code lands.
- Context: Recommended by the approved design and DX sections as the right way to
  validate the artifact after shipping, not before.
- Depends on / blocked by: Phase 2 implementation landing. Suggested follow-ups are
  `/design-review` and `/devex-review`.
