# codex-memory

Turn your local [Codex](https://openai.com/codex) session history into a searchable memory that a new coding agent can retrieve from — so past work (how a bug was actually fixed, decisions made, project-specific gotchas) isn't lost when you stop using Codex.

Fully local: your session transcripts never leave your machine. This repo only contains the pipeline code; your extracted data and embeddings stay in `data/`, which is gitignored.

## How it works

1. **Extract** — parse Codex's local rollout files (`~/.codex/{sessions,archived_sessions}/**/*.jsonl`) into structured per-session records (task, outcome, tool calls, project path).
2. **Summarize** — compress each session into a short, retrieval-friendly note (task / approach / outcome / gotchas) using a local model.
3. **Index** — embed the summaries with a local embedding model and store them in a local vector index.
4. **Retrieve** — given a new task, query the index for relevant past sessions and inject them as context for your agent/harness of choice.

## Status

Early / personal project. Step 1 (extract) is working. Steps 2-4 are in progress.

## Usage

```bash
python3 scripts/extract_sessions.py
```

Reads from `~/.codex` and writes `data/sessions_extracted.jsonl`.

## Privacy

Nothing in `data/` is committed (see `.gitignore`). Session transcripts can contain private code and business content — do not commit extracted data, summaries, or embedding indexes to this repo.

## License

MIT
