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


def search(query: str, top_k: int = 5) -> list:
    """Return the top_k most relevant past sessions for `query`, each as a
    dict with score, thread_name, cwd, summary (task/approach/outcome/
    gotchas) and the original rollout file path."""
    if not VEC_PATH.exists():
        raise FileNotFoundError(f"missing {VEC_PATH}, run build_index.py first")

    mat = np.load(VEC_PATH)
    meta = [json.loads(l) for l in META_PATH.open()]

    qvec = embed(query)
    sims = cosine_sim(qvec, mat)
    order = np.argsort(-sims)[:top_k]

    results = []
    for idx in order:
        m = dict(meta[idx])
        m["score"] = float(sims[idx])
        results.append(m)
    return results


def _print_results(results):
    for rank, m in enumerate(results, 1):
        s = m.get("summary") or {}
        print(f"\n#{rank}  score={m['score']:.3f}  {m.get('thread_name') or '(无标题)'}")
        print(f"    project: {m.get('cwd')}")
        print(f"    task:    {s.get('task', '')}")
        print(f"    outcome: {s.get('outcome', '')}")
        if s.get("gotchas") and s.get("gotchas") != "无":
            print(f"    gotchas: {s.get('gotchas')}")
        print(f"    file:    {m.get('file')}")


def main():
    if len(sys.argv) < 2:
        print("usage: search.py <query text> [top_k]", file=sys.stderr)
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    try:
        results = search(query, top_k)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    _print_results(results)


if __name__ == "__main__":
    main()
