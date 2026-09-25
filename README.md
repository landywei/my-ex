# my-ex

Turn your local [Codex](https://openai.com/codex) session history into a searchable memory that a new coding agent can retrieve from — so past work (how a bug was actually fixed, decisions made, project-specific gotchas) isn't lost when you stop using Codex.

Fully local: your session transcripts never leave your machine. This repo only contains the pipeline code; your extracted data and embeddings stay in `data/`, which is gitignored.

## How it works

Two retrieval tiers, built from the same raw source but at different granularity:

**Session-level (coarse, cheap to scan)**
1. `extract_sessions.py` — parse Codex's local rollout files (`~/.codex/{sessions,archived_sessions}/**/*.jsonl`) into one record per session: first real user request, final agent reply, tool-call names used, project path. Boilerplate (injected environment context, AGENTS.md instructions, plugin suggestions) is stripped so this reflects the actual ask.
2. `summarize_sessions.py` — compress each session into a short note (task / approach / outcome / gotchas) via a local LLM (qwen2.5:7b-instruct).
3. `build_index.py` — embed each summary (plus its collapsed tool-call sequence, e.g. `exec_command → apply_patch → exec_command`) with a local embedding model (bge-m3) into `data/index/{vectors.npy,meta.jsonl}`.

**Turn-level (detailed, close to verbatim)**
1. `extract_turns.py` — walk every raw response item and split each session into turns (one user message + everything the agent did in response: its replies, and every tool call with its actual command/input and the output that came back). Internal Codex sub-conversations (its own approval/risk-assessment loop) are filtered out.
2. `build_turn_index.py` — embed each turn directly (no LLM compression) into `data/index/{turn_vectors.npy,turn_meta.jsonl}`.

**Retrieve** — `scripts/search.py` (CLI, `--detailed` flag) or `scripts/mcp_server.py` (MCP server, `detailed` param) queries either tier. Any MCP-compatible coding agent (Codex CLI, OpenCode, Goose, Claude Code, ...) can call this to pull relevant past sessions — or the exact commands/fixes from a past turn — into context for a new task.

Neither tier holds the full raw transcript verbatim (that stays in the original `~/.codex` rollout files, one `file` pointer away in every result) — the session tier is an LLM paraphrase, the turn tier is capped-but-largely-verbatim (tool call input/output capped at a few thousand characters each so a handful of huge outputs don't blow up storage).

## Status

Early / personal project, in active use. Both tiers verified end-to-end on ~500 real sessions / ~7,500 turns.

## Prerequisites

- Python 3.10+ (the MCP SDK needs it; a venv is recommended — see below)
- [Ollama](https://ollama.com), running locally, with two models pulled:
  ```bash
  ollama pull qwen2.5:7b-instruct   # summarization
  ollama pull bge-m3                # embeddings
  ```

## Setup

```bash
python3.1x -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Usage

```bash
# session-level (coarse)
./.venv/bin/python scripts/extract_sessions.py     # ~/.codex -> data/sessions_extracted.jsonl
./.venv/bin/python scripts/summarize_sessions.py   # -> data/sessions_summarized.jsonl (resumable)
./.venv/bin/python scripts/build_index.py          # -> data/index/{vectors.npy,meta.jsonl}

# turn-level (detailed)
./.venv/bin/python scripts/extract_turns.py        # ~/.codex -> data/turns_extracted.jsonl
./.venv/bin/python scripts/build_turn_index.py     # -> data/index/{turn_vectors.npy,turn_meta.jsonl}

# query either tier
./.venv/bin/python scripts/search.py "your task description here"
./.venv/bin/python scripts/search.py --detailed "the exact command I used for X"
```

### As an MCP server

```bash
./.venv/bin/python scripts/mcp_server.py
```

Exposes one tool, `search_codex_memory(query, top_k, detailed)`. Point any MCP-compatible
harness at this command (stdio transport) to give it retrieval access to your
Codex history. Example (Claude Code `.mcp.json` / similar config shape used by
most MCP clients):

```json
{
  "mcpServers": {
    "my-ex": {
      "command": "/absolute/path/to/my-ex/.venv/bin/python",
      "args": ["/absolute/path/to/my-ex/scripts/mcp_server.py"]
    }
  }
}
```

Or, using the Claude Code CLI directly:

```bash
claude mcp add --scope user my-ex -- /absolute/path/to/my-ex/.venv/bin/python /absolute/path/to/my-ex/scripts/mcp_server.py
```

## Privacy

Nothing in `data/` is committed (see `.gitignore`). Session transcripts can contain private code and business content — do not commit extracted data, summaries, or embedding indexes to this repo.

## License

MIT
