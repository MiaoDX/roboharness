# Roboharness Context

Roboharness provides the evidence language for robot simulation runs that are evaluated by AI coding agents and, when necessary, humans.

## Language

**Agent Visual Review**:
Visual evaluation performed by a multimodal coding agent against a bounded proof pack and review manifest. It is not a separate VLM service or standalone robot-vision model.
_Avoid_: Model Judge, external VLM judge, standalone VLM evaluator

**Proof Pack**:
A compact set of visual evidence, metrics, baseline comparisons, and decision metadata prepared for agent visual review or human escalation.
_Avoid_: Screenshot folder, dashboard dump, artifact pile

**Visual Reviewer Invocation**:
An isolated agent visual review call that reads a proof pack and returns a structured verdict. It may run through a subagent, CLI trigger, CI job, or MCP tool, but its context is bounded to the evidence and review contract.
_Avoid_: Separate VLM, same-session self-review, screenshot glance

**Visual Review Dimension**:
A named aspect of robot behavior that the visual reviewer evaluates separately before an overall verdict is aggregated. Examples include robot posture, hand pose, object contact, trajectory naturalness, and task success.
_Avoid_: Overall impression, vibes, single visual score

**Motion Window Evidence**:
A short, ordered slice of visual and state evidence around a behavior that cannot be judged from one still frame. It is smaller than a full episode recording and is required for dimensions such as trajectory naturalness or late sharp motion.
_Avoid_: Trajectory recording, full video, checkpoint gallery

**Hard Metric Gate**:
A deterministic numeric or boolean condition that can accept or reject a run without visual interpretation. Hard metric gates provide the safety floor for visual judgment.
_Avoid_: Heuristic, visual score, model opinion

**Metric-First Review**:
The review strategy that uses hard metric gates wherever reliable metrics are available and reserves agent visual review for necessary judgments that are hard to metricize.
_Avoid_: Visual-first review, model-as-default-evaluator

**Metricization Rationale**:
The per-dimension explanation for why a visual review dimension is not fully covered by metrics, or which metrics provide its fallback safety floor.
_Avoid_: Unexplained visual dimension, model convenience, visual-only by default

**Metricization Gate**:
The prepare-time validation that rejects visual review dimensions without a metric fallback or an explicit reason they cannot be reliably metricized.
_Avoid_: Warning-only metric rationale, implicit visual fallback

**Human Escalation Boundary**:
A decision class reserved for human review because automatic acceptance would be too costly or semantically unsafe. The boundary should be explicit and as small as possible.
_Avoid_: Manual review, human in the loop

**Baseline Blessing**:
Human approval that makes a proposed new baseline authoritative for future comparisons.
_Avoid_: Auto-update, replace baseline, approve screenshot

## Flagged Ambiguities

**Model Judge**:
This term is ambiguous. In this project, prefer **Agent Visual Review** when the evaluator is a coding agent's built-in multimodal model; use "external VLM judge" only for a separate model or service.

**Visual Harness**:
The overall roboharness system for producing visual and metric evidence from robot simulation runs. Do not use this term for the narrower agent visual review step.

**First Visual Review Slice**:
The first agent visual review slice covers static pose-only dimensions. Temporal dimensions such as trajectory naturalness, late sharp motion, and contact sequence must return insufficient evidence until motion window evidence exists.

**Static Pose Visual Dimensions**:
The first visual review slice evaluates robot posture, hand pose, object-relative position, obvious collision or penetration, and task-success visual agreement. These dimensions do not include trajectory naturalness, smoothness, contact sequence, or overall aesthetic quality.

**Visual Veto Policy**:
Agent visual review can veto or escalate a metric-passing run, but it cannot rescue a hard metric failure. Required visual dimensions must pass before a final automatic pass is allowed.
_Avoid_: Visual override, model rescue, metric bypass

**Dimension Verdict**:
The structured result for one visual review dimension. Valid first-slice values are PASS, FAIL, INSUFFICIENT_EVIDENCE, NEEDS_HUMAN, and NOT_APPLICABLE.
_Avoid_: Ambiguous, maybe, looks okay

**Visual Confidence**:
A coarse confidence value attached to each visual dimension verdict. Valid values are high, medium, and low; numeric confidence is not used.
_Avoid_: Probability score, calibrated likelihood, confidence percentage

**NEEDS_HUMAN**:
A verdict meaning the model should not make the final decision even if it can inspect the evidence. Use it for baseline blessing, intended migration confirmation, view conflicts, low-confidence high-risk judgments, or other human escalation boundaries.
_Avoid_: Ambiguous, failed, blocked

**Human Escalation Reason**:
A fixed taxonomy value explaining why a visual review requires human decision. First-slice values include missing required evidence, view conflict, low-confidence high-risk judgment, baseline blessing required, migration intent confirmation required, unsupported temporal dimension, and current-only review cannot auto-pass.
_Avoid_: Free-text escalation reason, vague ambiguity

**Visual Review Record**:
The structured JSON output from a visual reviewer invocation. It contains an overall visual verdict, dimension verdicts, referenced evidence paths, confidence, and short rationales.
_Avoid_: Free-form review, chat answer, image caption

**REVIEW_INVALID**:
An approval-level verdict meaning the visual review record could not be trusted or ingested. Use it for schema-invalid review output, illegal enum values, evidence references outside the manifest, case mismatch, or schema-version mismatch.
_Avoid_: CONTRACT_INVALID, metric failure, ambiguous

**Visual Review Manifest**:
The minimal structured input given to a visual reviewer invocation. It lists case intent, required visual dimensions, evidence paths, metric summary, and review policy without exposing the full HTML report or implementation narrative.
_Avoid_: Full report HTML, implementation diff, prompt transcript

**Manifest Evidence Boundary**:
The rule that a visual review record may reference only evidence paths declared in the visual review manifest. Missing desired evidence must be reported as insufficient evidence rather than discovered ad hoc from the artifact directory.
_Avoid_: Free evidence browsing, reviewer-selected screenshots, unbounded artifact access

**Two-Step Visual Review**:
The first implementation boundary for agent visual review. Roboharness prepares the visual review manifest, prompt, schema, and selected evidence; an agent invocation writes the visual review record; roboharness then ingests that record for aggregation.
_Avoid_: Built-in model call, provider-specific review loop, all-in-one auto judge

**Visual Review Summary**:
The approval-report summary of a visual review record. It references the manifest and record paths, reports the overall visual verdict, and lists only blocking dimensions or human escalation reasons.
_Avoid_: Full visual review record inside approval report, duplicated reviewer rationale

**Paired Visual Review**:
The default visual review mode where the reviewer compares current evidence against baseline evidence under the task intent. Regression review requires paired evidence; migration review uses the old baseline for contrast but still cannot bless a new baseline automatically.
_Avoid_: Current-only visual approval, isolated screenshot judgment

**Current-Only Visual Check**:
A limited visual review mode used when baseline evidence is unavailable. It may find obvious visual failures, but it cannot produce final automatic pass for approval.
_Avoid_: Regression review, automatic pass, baseline-free approval

**Manifest-Bound Review**:
A visual reviewer invocation that answers only the dimensions declared in the visual review manifest. It must not invent task criteria, infer unseen motion, or use implementation intent to fill evidence gaps.
_Avoid_: Open-ended success judgment, free-form task review, general visual critique

**Dual First Visual Review Integration**:
The first visual review implementation supports both the deterministic MuJoCo grasp wedge and G1-style humanoid proof surfaces. MuJoCo is the canonical paired approval path; G1 starts as a static-pose visual review path with stricter human escalation unless paired baseline evidence and safe dimensions are available.
_Avoid_: MuJoCo-only v1, humanoid auto-approval, one-size-fits-all approval

**G1 First Visual Review Target**:
The first humanoid target for visual review is G1 WBC reach. G1 locomotion and SONIC-style motion tracking remain outside automatic visual pass until motion window evidence exists.
_Avoid_: Locomotion visual pass from stills, SONIC auto-approval from checkpoints

## Example Dialogue

Developer: "The hard metric gates pass, but the final grasp pose changed."

Domain Expert: "Put the changed pose in the proof pack and ask for agent visual review against the baseline."

Developer: "If the agent says the posture is intended, can it bless the new baseline?"

Domain Expert: "No. That crosses the human escalation boundary; baseline blessing stays explicit."

Developer: "Should the visual reviewer return one pass/fail?"

Domain Expert: "No. It should evaluate visual review dimensions separately, then aggregate the overall verdict."
