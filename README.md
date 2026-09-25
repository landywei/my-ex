# codex-memory

Turn your local [Codex](https://openai.com/codex) session history into a searchable memory that a new coding agent can retrieve from — so past work (how a bug was actually fixed, decisions made, project-specific gotchas) isn't lost when you stop using Codex.

Fully local: your session transcripts never leave your machine. This repo only contains the pipeline code; your extracted data and embeddings stay in `data/`, which is gitignored.

## How it works

1. **Extract** — parse Codex's local rollout files (`~/.codex/{sessions,archived_sessions}/**/*.jsonl`) into structured per-session records (task, outcome, tool calls, project path).
2. **Summarize** — compress each session into a short, retrieval-friendly note (task / approach / outcome / gotchas) using a local model.
3. **Index** — embed the summaries with a local embedding model and store them in a local vector index.
4. **Retrieve** — given a new task, query the index for relevant past sessions and inject them as context for your agent/harness of choice.

## Status

Early / personal project. All four scripts are written; extraction is verified end-to-end on real data. Summarize/index are pending a first full run.

## Prerequisites

- Python 3.9+ with `numpy`
- [Ollama](https://ollama.com), running locally, with two models pulled:
  ```bash
  ollama pull qwen2.5:7b-instruct   # summarization
  ollama pull bge-m3                # embeddings
  ```

## Usage

```bash
python3 scripts/extract_sessions.py     # ~/.codex -> data/sessions_extracted.jsonl
python3 scripts/summarize_sessions.py   # -> data/sessions_summarized.jsonl (resumable)
python3 scripts/build_index.py          # -> data/index/{vectors.npy,meta.jsonl}
python3 scripts/search.py "your task description here"
```

## Privacy

Nothing in `data/` is committed (see `.gitignore`). Session transcripts can contain private code and business content — do not commit extracted data, summaries, or embedding indexes to this repo.

## License

MIT
