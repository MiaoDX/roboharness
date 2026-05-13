from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from examples import mujoco_grasp
from examples._mujoco_grasp_wedge import ErrorEnvelope


def test_main_fails_fast_without_display_or_gl_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class UnexpectedBackendInit:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("backend should not be constructed")

    monkeypatch.setattr(mujoco_grasp, "MuJoCoMeshcatBackend", UnexpectedBackendInit)
    monkeypatch.setattr(sys, "argv", ["mujoco_grasp.py", "--output-dir", str(tmp_path)])
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        mujoco_grasp.main()

    message = str(excinfo.value)
    assert "MuJoCo renderer failed to start." in message
    assert "no display server was detected and MUJOCO_GL is unset" in message
    assert "Detected MUJOCO_GL=unset, DISPLAY=unset." in message
    assert "MUJOCO_GL=osmesa" in message
    assert "MUJOCO_GL=egl" in message


def test_main_surfaces_friendly_rendering_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingBackend:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("an OpenGL platform library has not been loaded into this process")

    monkeypatch.setattr(mujoco_grasp, "MuJoCoMeshcatBackend", FailingBackend)
    monkeypatch.setattr(sys, "argv", ["mujoco_grasp.py", "--output-dir", str(tmp_path)])
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        mujoco_grasp.main()

    message = str(excinfo.value)
    assert "MuJoCo renderer failed to start." in message
    assert "Cause: an OpenGL platform library has not been loaded into this process" in message
    assert "Detected MUJOCO_GL=egl, DISPLAY=unset." in message
    assert "MUJOCO_GL=osmesa" in message
    assert "MUJOCO_GL=egl" in message


def test_main_preserves_non_rendering_backend_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FailingBackend:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(mujoco_grasp, "MuJoCoMeshcatBackend", FailingBackend)
    monkeypatch.setattr(sys, "argv", ["mujoco_grasp.py", "--output-dir", str(tmp_path)])
    monkeypatch.setenv("MUJOCO_GL", "egl")

    with pytest.raises(RuntimeError, match="boom"):
        mujoco_grasp.main()


def test_main_writes_contract_compile_diagnostic_and_exits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, str | None] = {}

    def fake_compile_contract(
        *,
        baseline_source: str,
        contract_path: str | None = None,
        contract_preset: str = "",
        contract_prompt: str | None = None,
    ) -> dict[str, object]:
        captured["baseline_source"] = baseline_source
        captured["contract_path"] = contract_path
        captured["contract_preset"] = contract_preset
        captured["contract_prompt"] = contract_prompt
        raise mujoco_grasp.ContractCompileError(
            ErrorEnvelope(
                problem="Contract blocked.",
                cause="contract prompt could not be grounded to a reviewed preset.",
                fix="Use --contract-preset or an explicit JSON contract.",
                docs_url="docs/designs/unattended-refactor-harness-v1.md",
                recoverable=True,
                next_action="Fix contract",
            )
        )

    monkeypatch.setattr(mujoco_grasp, "compile_contract", fake_compile_contract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mujoco_grasp.py",
            "--output-dir",
            str(tmp_path),
            "--baseline-report",
            "fixtures/baseline.json",
            "--contract-preset",
            "mujoco_migration_guarded_v1",
            "--contract-prompt",
            "treat this as migration mode and require manual blessing",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        mujoco_grasp.main()

    assert excinfo.value.code == 2
    assert captured == {
        "baseline_source": "fixtures/baseline.json",
        "contract_path": None,
        "contract_preset": "mujoco_migration_guarded_v1",
        "contract_prompt": "treat this as migration mode and require manual blessing",
    }

    diagnostics = json.loads((tmp_path / "contract_compile_error.json").read_text())
    assert diagnostics == {
        "schema_version": 1,
        "error": {
            "problem": "Contract blocked.",
            "cause": "contract prompt could not be grounded to a reviewed preset.",
            "fix": "Use --contract-preset or an explicit JSON contract.",
            "docs_url": "docs/designs/unattended-refactor-harness-v1.md",
            "recoverable": True,
            "next_action": "Fix contract",
        },
    }
