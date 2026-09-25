#!/usr/bin/env python3
"""Embed every turn from turns_extracted.jsonl into a detailed, turn-level
vector index. This is the fine-grained counterpart to build_index.py: that
one embeds a short LLM-written summary per *session*; this one embeds the
actual (capped but largely verbatim) content of every *turn*, including
concrete tool-call commands and their output.

Search this index when you need the specifics -- the exact commands used,
the exact error message, the exact patch -- not just "what was this
session about".
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "bge-m3"
# bge-m3's context window is 8192 tokens, but Chinese-heavy text can run
# close to 2 tokens/char, so keep well under that; embed() also retries
# shorter on overflow for the rare long-tail turn.
EMBED_TEXT_CAP = 3000  # chars fed to the embedding model per turn

ROOT = Path(__file__).parent.parent
IN_PATH = ROOT / "data" / "turns_extracted.jsonl"
INDEX_DIR = ROOT / "data" / "index"
VEC_PATH = INDEX_DIR / "turn_vectors.npy"
META_PATH = INDEX_DIR / "turn_meta.jsonl"


def embed(text: str, timeout=60) -> list:
    for attempt_text in (text, text[:1500], text[:600]):
        body = json.dumps({
            "model": MODEL,
            "prompt": attempt_text,
            "options": {"num_ctx": 8192},
        }).encode("utf-8")
        req = urllib.request.Request(
            OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data["embedding"]
        except urllib.error.HTTPError as e:
            if e.code == 500 and attempt_text is not text[:600]:
                continue  # likely context-length overflow, retry shorter
            raise
    raise RuntimeError("embedding failed even at minimal length")


def turn_to_embed_text(rec) -> str:
    parts = [rec.get("thread_name") or ""]
    if rec.get("user_message"):
        parts.append(f"用户: {rec['user_message']}")
    if rec.get("assistant_text"):
        parts.append(f"agent: {rec['assistant_text']}")
    for tc in rec.get("tool_calls", []):
        parts.append(f"[{tc['name']}] {tc['input'][:300]}")
        if tc.get("output"):
            parts.append(f"  -> {tc['output'][:200]}")
    text = "\n".join(p for p in parts if p)
    return text[:EMBED_TEXT_CAP]


def main():
    if not IN_PATH.exists():
        print(f"missing {IN_PATH}, run extract_turns.py first", file=sys.stderr)
        sys.exit(1)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in IN_PATH.open()]
    print(f"embedding {len(records)} turns", file=sys.stderr)

    vectors = []
    meta = []
    for i, rec in enumerate(records):
        text = turn_to_embed_text(rec)
        if not text.strip():
            continue
        try:
            vec = embed(text)
        except Exception as e:
            print(f"ERROR embedding {rec.get('session_id')}#{rec.get('turn_index')}: {e}", file=sys.stderr)
            continue
        vectors.append(vec)
        meta.append(rec)  # keep the full turn record for detailed retrieval
        if (i + 1) % 200 == 0:
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
