from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11
    import tomli as tomllib


def _matrix_ids() -> set[str]:
    data = tomllib.loads(Path("constraints/demo_matrix.toml").read_text())
    return {demo["id"] for demo in data["demo"]}


def test_demo_matrix_includes_expected_ids() -> None:
    expected = {
        "grasp",
        "g1-reach",
        "g1-loco",
        "g1-native-groot",
        "g1-native-sonic",
        "sonic-planner",
        "sonic",
    }
    assert _matrix_ids() == expected


def test_hf_manual_fetch_keeps_sonic_planner_in_sync() -> None:
    workflow = Path(".github/workflows/hf-space.yml").read_text()
    assert "_site/sonic-planner" in workflow
    assert "$BASE/sonic-planner/" in workflow
