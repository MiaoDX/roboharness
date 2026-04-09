"""MCP (Model Context Protocol) server interface for roboharness.

Exposes roboharness functionality as MCP tools so AI agents can interact
with running simulations via the standard MCP protocol.

Tools provided:
  - ``capture_checkpoint`` -- capture multi-view screenshots and state
  - ``evaluate_constraints`` -- run constraint evaluator on a report
  - ``compare_baselines`` -- compare current run against history
"""

from __future__ import annotations

from roboharness.mcp.tools import HarnessTools

__all__ = ["HarnessTools"]
