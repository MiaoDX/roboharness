"""Tests for roboharness.mcp.server — MCP server wrapper.

The ``mcp`` SDK is an optional dependency.  These tests inject lightweight
stand-ins for ``mcp.server.Server`` and ``mcp.types`` so the dispatch
logic in ``create_server`` can be validated without installing the SDK.
"""

from __future__ import annotations

import asyncio
import json
import sys
import types as stdlib_types
from typing import Any

import pytest

from roboharness.core.harness import Harness

from .conftest import MockBackend

# ── Lightweight MCP SDK stand-ins ───────────────────────────────────────


class _FakeServer:
    """Captures handler functions registered via ``@server.list_tools()`` etc."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._list_handler: Any = None
        self._call_handler: Any = None

    def list_tools(self):
        def _deco(fn: Any) -> Any:
            self._list_handler = fn
            return fn

        return _deco

    def call_tool(self):
        def _deco(fn: Any) -> Any:
            self._call_handler = fn
            return fn

        return _deco


class _FakeTool:
    def __init__(self, *, name: str, description: str, inputSchema: dict) -> None:
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


class _FakeTextContent:
    def __init__(self, *, type: str, text: str) -> None:
        self.type = type
        self.text = text


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def _patch_mcp():
    """Install fake ``mcp`` modules so ``server.py`` can be imported."""
    fake_mcp = stdlib_types.ModuleType("mcp")
    fake_server_mod = stdlib_types.ModuleType("mcp.server")
    fake_server_mod.Server = _FakeServer  # type: ignore[attr-defined]
    fake_types_mod = stdlib_types.ModuleType("mcp.types")
    fake_types_mod.TextContent = _FakeTextContent  # type: ignore[attr-defined]
    fake_types_mod.Tool = _FakeTool  # type: ignore[attr-defined]

    originals = {
        k: sys.modules.get(k) for k in ("mcp", "mcp.server", "mcp.types", "roboharness.mcp.server")
    }

    sys.modules["mcp"] = fake_mcp
    sys.modules["mcp.server"] = fake_server_mod
    sys.modules["mcp.types"] = fake_types_mod
    # Remove cached import so the re-import picks up fakes
    sys.modules.pop("roboharness.mcp.server", None)

    yield

    for key, val in originals.items():
        if val is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = val


@pytest.fixture()
def harness(tmp_path):
    h = Harness(MockBackend(), output_dir=tmp_path, task_name="mcp_test")
    h.reset()
    return h


@pytest.fixture()
def server(_patch_mcp, harness):
    """Create an MCP server wired to a mock harness."""
    from roboharness.mcp.server import create_server

    return create_server(harness)


def _run(coro: Any) -> Any:
    """Run an async handler synchronously."""
    return asyncio.run(coro)


# ── create_server wiring ────────────────────────────────────────────────


def test_create_server_returns_named_server(server):
    assert server.name == "roboharness"


def test_create_server_custom_name(_patch_mcp, harness):
    from roboharness.mcp.server import create_server

    s = create_server(harness, server_name="my-harness")
    assert s.name == "my-harness"


def test_create_server_registers_both_handlers(server):
    assert server._list_handler is not None
    assert server._call_handler is not None


# ── list_tools ──────────────────────────────────────────────────────────


def test_list_tools_returns_three_tools(server):
    tools = _run(server._list_handler())
    assert len(tools) == 3


def test_list_tools_has_expected_names(server):
    tools = _run(server._list_handler())
    names = {t.name for t in tools}
    assert names == {"capture_checkpoint", "evaluate_constraints", "compare_baselines"}


def test_list_tools_each_has_description_and_schema(server):
    tools = _run(server._list_handler())
    for tool in tools:
        assert tool.description
        assert isinstance(tool.inputSchema, dict)


# ── call_tool dispatch ──────────────────────────────────────────────────


def test_call_capture_checkpoint(server):
    result = _run(server._call_handler("capture_checkpoint", {}))
    assert len(result) == 1
    payload = json.loads(result[0].text)
    assert payload["checkpoint_name"] == "step_0"
    assert payload["step"] == 0
    assert len(payload["views"]) == 1
    assert payload["views"][0]["name"] == "front"


def test_call_evaluate_constraints_pass(server):
    result = _run(
        server._call_handler(
            "evaluate_constraints",
            {
                "report": {"summary_metrics": {"grip_error": 3.0}},
                "assertions": [{"metric": "grip_error", "operator": "lt", "threshold": 5.0}],
            },
        )
    )
    payload = json.loads(result[0].text)
    assert payload["verdict"] == "pass"


def test_call_compare_baselines_no_history(server):
    result = _run(
        server._call_handler(
            "compare_baselines",
            {"task": "grasp", "current_rate": 0.8},
        )
    )
    payload = json.loads(result[0].text)
    assert payload["regressed"] is False
    assert payload["previous_rate"] is None


def test_call_unknown_tool_returns_error(server):
    result = _run(server._call_handler("nonexistent_tool", {}))
    assert len(result) == 1
    assert "Unknown tool: nonexistent_tool" in result[0].text
