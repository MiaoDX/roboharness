"""Thin MCP server that exposes :class:`HarnessTools` over the MCP protocol.

Requires the ``mcp`` package (``pip install mcp``).  Import this module only
when you intend to run the server; the rest of ``roboharness.mcp`` works
without the SDK.

Usage::

    from roboharness import Harness
    from roboharness.mcp.server import create_server

    harness = Harness(backend, output_dir="./output")
    server = create_server(harness)
    server.run()                 # blocks, listens on stdio
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from roboharness.core.harness import Harness
from roboharness.mcp.tools import TOOL_SCHEMAS, HarnessTools


def create_server(
    harness: Harness,
    history_dir: str | None = None,
    server_name: str = "roboharness",
) -> Server:
    """Create an MCP :class:`Server` wired to *harness*.

    Parameters
    ----------
    harness:
        A fully-initialised :class:`Harness` with a simulator backend.
    history_dir:
        Optional path for baseline history storage.
    server_name:
        Human-readable server name advertised to clients.
    """
    tools = HarnessTools(harness, history_dir=history_dir)
    server = Server(server_name)

    dispatch: dict[str, Any] = {
        "capture_checkpoint": tools.capture_checkpoint,
        "evaluate_constraints": tools.evaluate_constraints,
        "compare_baselines": tools.compare_baselines,
    }

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        return [
            Tool(
                name=schema["name"],
                description=schema["description"],
                inputSchema=schema["inputSchema"],
            )
            for schema in TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = dispatch.get(name)
        if handler is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        result = handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    return server
