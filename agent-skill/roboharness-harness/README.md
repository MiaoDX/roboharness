<!-- Generated from contract.py by roboharness.contract. Do not edit. -->
# Roboharness Project Harness Skill

Source of truth: `contract.py`.

Generated artifacts:

- `SKILL.md` - agent-facing workflow guidance
- `contract.snapshot.json` - normalized machine snapshot
- `schemas/` - generated JSON schemas for snapshots and manifests
- `scope-brief-template.md` - template for proposing contract changes
- `stubs/run-validation.py` - optional validation-command runner
- `.generated-manifest.json` - drift-check manifest

Regenerate:

```bash
roboharness contract generate contract.py --output-dir .
```

Check drift:

```bash
roboharness contract check contract.py --output-dir .
```
