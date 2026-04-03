"""Load and save constraint definitions from YAML or JSON files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from roboharness.evaluate.assertions import MetricAssertion
from roboharness.evaluate.result import Operator, Severity

# Optional YAML support — fall back to JSON-only if PyYAML is not installed.
try:
    import yaml

    _HAS_YAML = True
except ImportError:  # pragma: no cover
    _HAS_YAML = False


def _parse_assertion(raw: dict[str, Any]) -> MetricAssertion:
    """Parse a single assertion definition dict into a MetricAssertion."""
    operator = Operator(raw["operator"])
    threshold: float | tuple[float, float]
    if operator == Operator.IN_RANGE:
        th = raw["threshold"]
        threshold = (float(th[0]), float(th[1]))
    else:
        threshold = float(raw["threshold"])

    return MetricAssertion(
        metric=raw["metric"],
        operator=operator,
        threshold=threshold,
        severity=Severity(raw.get("severity", "major")),
        phase=raw.get("phase", "*"),
    )


def load_constraints(path: Path) -> list[MetricAssertion]:
    """Load constraint assertions from a YAML or JSON file.

    The file should contain a top-level ``constraints`` key with a list of
    assertion definitions.  Each definition requires ``metric``, ``operator``,
    and ``threshold``; ``severity`` defaults to ``"major"`` and ``phase``
    defaults to ``"*"`` (trial-level).
    """
    text = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            msg = "PyYAML is required to load .yaml constraint files: pip install pyyaml"
            raise ImportError(msg)
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    raw_list = data.get("constraints", [])
    return [_parse_assertion(raw) for raw in raw_list]


def save_constraints(assertions: list[MetricAssertion], path: Path) -> None:
    """Save constraint assertions to a JSON file."""
    data = {
        "constraints": [
            {
                "metric": a.metric,
                "operator": a.operator.value,
                "threshold": list(a.threshold) if isinstance(a.threshold, tuple) else a.threshold,
                "severity": a.severity.value,
                "phase": a.phase,
            }
            for a in assertions
        ]
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
