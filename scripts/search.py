#!/usr/bin/env python3
"""Query the local session index: given a task description, return the
most relevant past Codex sessions.

Usage:
    python3 scripts/search.py "how did I fix the theme color precedence bug"
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "bge-m3"

ROOT = Path(__file__).parent.parent
VEC_PATH = ROOT / "data" / "index" / "vectors.npy"
META_PATH = ROOT / "data" / "index" / "meta.jsonl"
TURN_VEC_PATH = ROOT / "data" / "index" / "turn_vectors.npy"
TURN_META_PATH = ROOT / "data" / "index" / "turn_meta.jsonl"


def embed(text: str) -> np.ndarray:
    body = json.dumps({"model": MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return np.array(data["embedding"], dtype=np.float32)


def cosine_sim(query_vec, mat):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    m = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return m @ q


def _search_index(query: str, top_k: int, vec_path: Path, meta_path: Path) -> list:
    if not vec_path.exists():
        raise FileNotFoundError(f"missing {vec_path}, build the index first")

    mat = np.load(vec_path)
    meta = [json.loads(l) for l in meta_path.open()]

    qvec = embed(query)
    sims = cosine_sim(qvec, mat)
    order = np.argsort(-sims)[:top_k]

    results = []
    for idx in order:
        m = dict(meta[idx])
        m["score"] = float(sims[idx])
        results.append(m)
    return results


def search(query: str, top_k: int = 5) -> list:
    """Return the top_k most relevant past *sessions* for `query`: coarse,
    LLM-summarized overviews (task/approach/outcome/gotchas). Good for
    "what did I do about X" / "which session handled Y"."""
    return _search_index(query, top_k, VEC_PATH, META_PATH)


def search_detailed(query: str, top_k: int = 5) -> list:
    """Return the top_k most relevant past *turns* for `query`: one user
    request plus the agent's actual replies and tool calls (command run,
    output returned), largely verbatim. Good for "what exact command did
    I run" / "what was the actual fix"."""
    return _search_index(query, top_k, TURN_VEC_PATH, TURN_META_PATH)


def _print_results(results):
    for rank, m in enumerate(results, 1):
        s = m.get("summary") or {}
        print(f"\n#{rank}  score={m['score']:.3f}  {m.get('thread_name') or '(无标题)'}")
        print(f"    project: {m.get('cwd')}")
        print(f"    task:    {s.get('task', '')}")
        print(f"    outcome: {s.get('outcome', '')}")
        if s.get("gotchas") and s.get("gotchas") != "无":
            print(f"    gotchas: {s.get('gotchas')}")
        if m.get("tool_sequence"):
            print(f"    tools:   {m.get('tool_sequence')}")
        print(f"    file:    {m.get('file')}")


def _print_detailed_results(results):
    for rank, m in enumerate(results, 1):
        print(f"\n#{rank}  score={m['score']:.3f}  {m.get('thread_name') or '(无标题)'}  (turn {m.get('turn_index')})")
        print(f"    project: {m.get('cwd')}")
        if m.get("user_message"):
            print(f"    user:    {m['user_message'][:300]}")
        if m.get("assistant_text"):
            print(f"    agent:   {m['assistant_text'][:300]}")
        for tc in m.get("tool_calls", [])[:10]:
            print(f"    [{tc['name']}] {tc['input'][:200]}")
            if tc.get("output"):
                print(f"      -> {tc['output'][:200]}")
        print(f"    file:    {m.get('file')}")


def main():
    args = sys.argv[1:]
    detailed = "--detailed" in args
    args = [a for a in args if a != "--detailed"]
    if not args:
        print("usage: search.py [--detailed] <query text> [top_k]", file=sys.stderr)
        sys.exit(1)
    query = args[0]
    top_k = int(args[1]) if len(args) > 1 else 5

    try:
        if detailed:
            results = search_detailed(query, top_k)
        else:
            results = search(query, top_k)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if detailed:
        _print_detailed_results(results)
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
