#!/usr/bin/env python3
"""Embed summarized sessions with a local Ollama embedding model and
store them as a flat local vector index (numpy matrix + metadata jsonl).

Brute-force cosine similarity is used at search time -- fine up to a
few tens of thousands of sessions, no ANN index needed.
"""
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "bge-m3"

ROOT = Path(__file__).parent.parent
IN_PATH = ROOT / "data" / "sessions_summarized.jsonl"
EXTRACTED_PATH = ROOT / "data" / "sessions_extracted.jsonl"
INDEX_DIR = ROOT / "data" / "index"
VEC_PATH = INDEX_DIR / "vectors.npy"
META_PATH = INDEX_DIR / "meta.jsonl"


def load_tool_sequences() -> dict:
    """session_id -> compact tool-call sequence, consecutive repeats
    collapsed (e.g. many exec_command calls in a row -> one)."""
    seqs = {}
    if not EXTRACTED_PATH.exists():
        return seqs
    for line in EXTRACTED_PATH.open():
        rec = json.loads(line)
        names = rec.get("tool_call_names") or []
        collapsed = []
        for n in names:
            if not collapsed or collapsed[-1] != n:
                collapsed.append(n)
        seqs[rec["session_id"]] = " → ".join(collapsed[:40])
    return seqs


def embed(text: str, timeout=60) -> list:
    body = json.dumps({"model": MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["embedding"]


def summary_to_text(rec, tool_seq: str = "") -> str:
    s = rec.get("summary") or {}
    parts = [
        rec.get("thread_name") or "",
        f"任务: {s.get('task', '')}",
        f"做法: {s.get('approach', '')}",
        f"结果: {s.get('outcome', '')}",
        f"坑点: {s.get('gotchas', '')}",
        f"工具调用顺序: {tool_seq}" if tool_seq else "",
    ]
    return "\n".join(p for p in parts if p)


def main():
    if not IN_PATH.exists():
        print(f"missing {IN_PATH}, run summarize_sessions.py first", file=sys.stderr)
        sys.exit(1)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in IN_PATH.open()]
    tool_seqs = load_tool_sequences()
    print(f"embedding {len(records)} summaries", file=sys.stderr)

    vectors = []
    meta = []
    for i, rec in enumerate(records):
        tool_seq = tool_seqs.get(rec["session_id"], "")
        text = summary_to_text(rec, tool_seq)
        if not text.strip():
            continue
        try:
            vec = embed(text)
        except Exception as e:
            print(f"ERROR embedding {rec['session_id']}: {e}", file=sys.stderr)
            continue
        vectors.append(vec)
        meta.append({
            "session_id": rec["session_id"],
            "thread_name": rec.get("thread_name"),
            "cwd": rec.get("cwd"),
            "created_at": rec.get("created_at"),
            "file": rec.get("file"),
            "summary": rec.get("summary"),
            "tool_sequence": tool_seq,
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(records)}", file=sys.stderr)

    mat = np.array(vectors, dtype=np.float32)
    np.save(VEC_PATH, mat)
    with META_PATH.open("w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"wrote {mat.shape} vectors -> {VEC_PATH}", file=sys.stderr)
    print(f"wrote {len(meta)} meta records -> {META_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
