# Academic Conference Opportunities — 2026

_Research compiled: April 2026. Revisit for ICRA 2027 in Q4 2026; revisit for NeurIPS 2026 workshops in September 2026._

---

## Summary

| Venue | Dates | Deadline | Action | Status |
|-------|-------|----------|--------|--------|
| ICRA 2026 | Jun 1–5, Vienna | **Passed** | Plan ICRA 2027 instead | ❌ Missed |
| CoRL 2026 | Nov 9–12, Austin TX | Abstract May 25 / Paper May 28 | System paper possible | ⚡ Actionable |
| NeurIPS 2026 (main) | Dec 6–12 | May 6 AOE | **Passed** | ❌ Missed |
| NeurIPS 2026 (workshops) | Dec 6–12 | ~September (TBA) | Monitor, submit demo/position paper | ⏳ Watch |
| Google DeepMind Accelerator | Cohort Jun 2026 | Closed Mar 25 | Monitor for next cohort | ❌ Missed, next TBA |
| GSoC 2027 | Summer 2027 | Org apps ~Jan–Feb 2027 | Prepare materials Q4 2026 | ⏳ Plan |

---

## ICRA 2026 — Missed, Plan for 2027

**Dates:** June 1–5, 2026 (workshops: June 1 and June 5), VIECON, Vienna, Austria  
**Status:** Workshop notifications already sent. All workshop submission deadlines have passed.

### Relevant workshops (for reference and ICRA 2027 planning)

The following workshops at ICRA 2026 are closely aligned with roboharness:

| Workshop | Day | Relevance |
|----------|-----|-----------|
| Synthetic Data for Robot Learning | Mon Jun 1 | Simulation-based data, visual validation |
| Generative Digital Twins for Sim2Real and Real2Sim Transfer | Mon Jun 1 | Sim-to-real validation gap — exactly what roboharness addresses |
| Reinforcement Learning in the Era of Imitation Learning | Mon Jun 1 | Evaluation methodology for RL policies |
| From Data to Decisions: VLA Pipelines for Real Robots | Fri Jun 5 | VLA evaluation, tool-chain gap |
| Beyond Teleoperation: Learning from Diverse Human and Simulation Data | Fri Jun 5 | Sim-based learning evaluation |

**ICRA 2027 action plan (Q4 2026):**

- Identify call for workshop proposals (typically released ~October, deadline ~December)
- Submit a **workshop proposal** for "Visual Evaluation Frameworks for Robot Learning Agents"
  - Angle: AI coding agents need visual feedback loops; existing CI for robotics is blind
  - Roboharness positions as a reference implementation of the harness engineering approach
- Alternatively, submit a **demo paper** to an aligned workshop (Sim2Real or RL evaluation)

---

## CoRL 2026 — Actionable (Abstract Due May 25)

**Dates:** November 9–12, 2026, Austin, Texas, USA  
**Abstract deadline:** May 25, 2026  
**Full paper deadline:** May 28, 2026  
**Submission:** https://openreview.net/group?id=robot-learning.org/CoRL/2026/Conference

### Fit assessment

CoRL accepts papers on robot learning with real-robot experiments or strong sim-to-real evidence. A roboharness **system paper** is plausible if:

1. Experimental validation with a real robot (or convincing sim-to-real transfer evidence) is available
2. The paper demonstrates improvement in an agent's iteration speed or debugging quality
3. User study or quantitative comparison with baseline (no visual feedback) exists

**Current gap:** Roboharness lacks the real-robot experiments CoRL requires. Purely sim-based results are unlikely to clear review bar without sim-to-real justification.

### Recommended paper angle (if experiments are available)

**Title:** "Harness Engineering for Robot Learning: Visual Checkpoint Testing Accelerates AI Agent Iteration"

**Key claims:**
- AI coding agents editing robot control policies benefit from visual feedback at checkpoint intervals
- Checkpoint screenshots + structured JSON reduce debugging loop time vs. log-only iteration
- Integration is three lines of code (drop-in `RobotHarnessWrapper`)
- Demonstrated on: MuJoCo Grasp, G1 WBC Reach, G1 LeRobot locomotion

**Decision checklist (before submitting):**
- [ ] Can we generate quantitative data comparing agent iteration speed with/without roboharness?
- [ ] Do we have real-robot or strong sim-to-real experiment?
- [ ] Is the paper 8 pages achievable given current result set?

**If yes to all → submit.** Deadline May 25 for abstract.

---

## NeurIPS 2026 — Main Track Missed; Watch for Workshops

**Dates:** December 6–12, 2026  
**Main track full paper deadline:** May 6, 2026 AOE — **passed**  
**Workshop CFPs:** Typically announced September 2026 (check neurips.cc/Conferences/2026)

### Workshop strategy

NeurIPS workshops are a better fit for roboharness than the main track because:
- Lower bar for tool/framework papers without extensive real-robot experiments
- Position papers and demos are common
- Robotics × ML × evaluation methodology is a natural workshop fit

**Candidate workshop topics to watch for in September 2026:**
- "Robot Learning" or "Embodied AI" workshop
- "Evaluation and Benchmarking in Robotics" (recurring topic)
- "Machine Learning for Robotics" or "Foundation Models for Robotics"
- "AI for Scientific Discovery" (sim-based agent testing fits here)

**When workshops are announced:**
1. Identify 2–3 workshops with deadlines in October–November 2026
2. Submit a **4-page demo/position paper** describing roboharness + results from showcase demos
3. Angle: "Harness engineering as the missing CI layer for robot learning"

---

## Google DeepMind Accelerator: Robotics

**Status:** First cohort applications closed March 25, 2026. Cohort kicks off June 2026 in London.

**Program details:**
- 10–15 early-stage robotics startups (European HQ, VC-backed, 5+ technical team members)
- Equity-free, 12–15 weeks
- Up to $350K Google Cloud credits via Google for Startups
- Target sectors: logistics, manufacturing, health/life sciences, HRI, education, navigation

**Current eligibility gap:**
- roboharness is an open-source project, not a VC-backed startup — does not meet current eligibility criteria
- The program targets commercial startups, not OSS/academic tools

**Action:** Monitor for future cohorts or a research track. Check https://deepmind.google/models/gemini-robotics/accelerator/ in Q3/Q4 2026 for next cohort announcement.

---

## GSoC 2027 — Prepare Materials Q4 2026

**Program:** Google Summer of Code — open-source organization sponsoring student contributors  
**Org application window:** Typically January 22 – February 6 (based on 2026 timeline; 2027 dates not yet published)  
**Accepted orgs announced:** ~February 21 (based on 2026)

### Eligibility checklist

- [x] Open-source project (MIT license)
- [x] Active GitHub repository with meaningful commit history
- [x] Clear contributing guide (CONTRIBUTING.md exists)
- [ ] Prior GSoC participation (not required for first-time orgs, but helpful)
- [ ] Minimum 2 mentors willing to commit time (need to identify)

### What to prepare (Q4 2026, target November–December)

See `docs/academic/gsoc-2027-application.md` for the full draft application.

**Key sections needed for org application:**
1. Organization name and description (2–3 sentences)
2. Why Google should accept roboharness as a GSoC org
3. Project ideas list (3–5 concrete ideas with scope, skills required, difficulty)
4. Mentor roster (name, GitHub handle, availability)
5. Communication channels (GitHub Discussions, Discord, etc.)

**Project idea candidates:**
- **Gazebo/ROS2 backend** — implement `SimulatorBackend` for ROS-based Gazebo simulation
- **VLM evaluation judge** — integrate open-source 4B VLM for autonomous checkpoint scoring
- **Isaac Lab checkpoint integration** — full round-trip with NVIDIA Isaac Lab, visual reports
- **Web dashboard** — real-time Rerun/Meshcat-based live monitoring dashboard
- **Pinocchio/MuJoCo benchmark suite** — standardized evaluation harness for WBC controllers

**Timeline:**
- Q3 2026 (Jul–Sep): Build community presence (ROS Discourse, HuggingFace, Discord)
- Q4 2026 (Oct–Dec): Draft project ideas, identify mentors, prepare application
- Jan 2027: Submit org application when window opens
- Feb 2027: If accepted, publish project ideas and start engaging potential contributors

---

## References

- ICRA 2026: https://2026.ieee-icra.org/
- CoRL 2026: https://www.corl.org/
- NeurIPS 2026: https://neurips.cc/Conferences/2026
- Google DeepMind Accelerator: https://deepmind.google/models/gemini-robotics/accelerator/
- GSoC timeline (2026 reference): https://developers.google.com/open-source/gsoc/timeline
- Issue: miaodx/roboharness#157
