#!/usr/bin/env python3
"""MCP server exposing local Codex session memory as a tool.

Any MCP-compatible harness (Codex CLI, OpenCode, Goose, Claude Code, ...)
can call `search_codex_memory` to retrieve relevant past sessions for a
new task, instead of starting from scratch. Two granularities:
  - overview (default): short LLM-summarized session, cheap to scan
  - detailed: the actual turn(s) -- real commands run, real output

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
def search_codex_memory(query: str, top_k: int = 5, detailed: bool = False) -> str:
    """Search past local Codex coding-agent sessions for ones relevant to
    a task, so you can reuse prior approaches, decisions, and known gotchas
    instead of starting from scratch.

    Args:
        query: description of the current task, in the same language you'd
            naturally describe it (English or Chinese both work).
        top_k: how many results to return, most relevant first.
        detailed: false (default) returns short session-level overviews
            (task/approach/outcome/gotchas) -- good for a first pass, "did
            I do something like this before". true returns individual
            turns with the actual user request, the agent's reply, and
            every tool call's real command + output -- use this when you
            need the specifics (exact command, exact fix, exact error) to
            replicate or learn from, not just a summary.
    """
    try:
        if detailed:
            results = search_mod.search_detailed(query, top_k)
        else:
            results = search_mod.search(query, top_k)
    except FileNotFoundError as e:
        return f"Index not built yet: {e}"

    if not results:
        return "No relevant past sessions/turns found."

    if detailed:
        lines = []
        for rank, m in enumerate(results, 1):
            tool_lines = "\n".join(
                f"  [{tc['name']}] {tc['input'][:400]}"
                + (f"\n    -> {tc['output'][:300]}" if tc.get("output") else "")
                for tc in m.get("tool_calls", [])[:15]
            )
            lines.append(
                f"#{rank} (score={m['score']:.3f}) {m.get('thread_name') or '(无标题)'} "
                f"(turn {m.get('turn_index')})\n"
                f"  project: {m.get('cwd')}\n"
                f"  user: {m.get('user_message', '')[:500]}\n"
                f"  agent: {m.get('assistant_text', '')[:500]}\n"
                f"{tool_lines}\n"
                f"  source: {m.get('file')}"
            )
        return "\n\n".join(lines)

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
            f"  tool_sequence: {m.get('tool_sequence', '')}\n"
            f"  source: {m.get('file')}"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    mcp.run()
