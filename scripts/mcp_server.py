#!/usr/bin/env python3
"""MCP server exposing local Codex session memory as a tool.

Any MCP-compatible harness (Codex CLI, OpenCode, Goose, Claude Code, ...)
can call `search_codex_memory` to retrieve relevant past sessions for a
new task, instead of starting from scratch.

Run directly (stdio transport):
    .venv/bin/python scripts/mcp_server.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import search as search_mod  # noqa: E402

from mcp.server import MCPServer

mcp = MCPServer("codex-memory")


@mcp.tool()
def search_codex_memory(query: str, top_k: int = 5) -> str:
    """Search past local Codex coding-agent sessions for ones relevant to
    a task, so you can reuse prior approaches, decisions, and known gotchas
    instead of starting from scratch.

    Args:
        query: description of the current task, in the same language you'd
            naturally describe it (English or Chinese both work).
        top_k: how many past sessions to return, most relevant first.
    """
    try:
        results = search_mod.search(query, top_k)
    except FileNotFoundError as e:
        return f"Index not built yet: {e}"

    if not results:
        return "No relevant past sessions found."

    lines = []
    for rank, m in enumerate(results, 1):
        s = m.get("summary") or {}
        lines.append(
            f"#{rank} (score={m['score']:.3f}) {m.get('thread_name') or '(无标题)'}\n"
            f"  project: {m.get('cwd')}\n"
            f"  task: {s.get('task', '')}\n"
            f"  approach: {s.get('approach', '')}\n"
            f"  outcome: {s.get('outcome', '')}\n"
            f"  gotchas: {s.get('gotchas', '')}\n"
            f"  source: {m.get('file')}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    mcp.run()
