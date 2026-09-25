# codex-memory

Turn your local [Codex](https://openai.com/codex) session history into a searchable memory that a new coding agent can retrieve from — so past work (how a bug was actually fixed, decisions made, project-specific gotchas) isn't lost when you stop using Codex.

Fully local: your session transcripts never leave your machine. This repo only contains the pipeline code; your extracted data and embeddings stay in `data/`, which is gitignored.

## How it works

1. **Extract** — parse Codex's local rollout files (`~/.codex/{sessions,archived_sessions}/**/*.jsonl`) into structured per-session records (task, outcome, tool calls, project path).
2. **Summarize** — compress each session into a short, retrieval-friendly note (task / approach / outcome / gotchas) using a local model.
3. **Index** — embed the summaries with a local embedding model and store them in a local vector index.
4. **Retrieve** — via `scripts/search.py` (CLI) or `scripts/mcp_server.py` (MCP server), so any MCP-compatible coding agent (Codex CLI, OpenCode, Goose, Claude Code, ...) can pull relevant past sessions into context for a new task.

## Status

Early / personal project. All scripts are written; extraction is verified end-to-end on real data. Summarize/index are pending a first full run.

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
./.venv/bin/python scripts/extract_sessions.py     # ~/.codex -> data/sessions_extracted.jsonl
./.venv/bin/python scripts/summarize_sessions.py   # -> data/sessions_summarized.jsonl (resumable)
./.venv/bin/python scripts/build_index.py          # -> data/index/{vectors.npy,meta.jsonl}
./.venv/bin/python scripts/search.py "your task description here"
```

### As an MCP server

```bash
./.venv/bin/python scripts/mcp_server.py
```

Exposes one tool, `search_codex_memory(query, top_k)`. Point any MCP-compatible
harness at this command (stdio transport) to give it retrieval access to your
Codex history. Example (Claude Code `.mcp.json` / similar config shape used by
most MCP clients):

```json
{
  "mcpServers": {
    "codex-memory": {
      "command": "/absolute/path/to/codex-memory/.venv/bin/python",
      "args": ["/absolute/path/to/codex-memory/scripts/mcp_server.py"]
    }
  }
}
```

## Privacy

Nothing in `data/` is committed (see `.gitignore`). Session transcripts can contain private code and business content — do not commit extracted data, summaries, or embedding indexes to this repo.

## License

MIT
