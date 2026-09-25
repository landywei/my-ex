#!/usr/bin/env python3
"""Extract full-detail per-turn records from local Codex rollout files.

Unlike extract_sessions.py (which keeps only the first/last message per
session for a cheap coarse summary), this walks every response_item and
splits each session into turns: one user message plus everything the
agent did in response (its replies and every tool call, with the actual
command/input and its output). This is the raw material for a detailed,
turn-level retrieval index -- not just "what was this session about" but
"what exactly did I run, and what came back".

Writes data/turns_extracted.jsonl, one record per turn.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_sessions import (  # noqa: E402
    find_rollout_files,
    load_thread_names,
    extract_text,
    clean_task_text,
    _INNER_TOOL_RE,
)

ROOT = Path(__file__).parent.parent
OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "turns_extracted.jsonl"

INPUT_CAP = 4000    # per tool call, chars
OUTPUT_CAP = 3000   # per tool call, chars

# Internal Codex sub-conversations (e.g. its own approval/risk-assessment
# loop) aren't the user's actual work -- skip turns that are clearly that.
_NOISE_PREFIXES = (
    "The following is the Codex agent history",
)


def extract_output_text(payload) -> str:
    out = payload.get("output")
    if isinstance(out, str):
        return out
    if isinstance(out, list):
        parts = [b.get("text", "") for b in out if isinstance(b, dict)]
        return "\n".join(p for p in parts if p)
    return ""


def parse_session_turns(path: Path, thread_name: str):
    session_id = None
    cwd = None

    turns = []
    call_index = {}  # call_id -> (turn_idx, tool_call_idx)

    def new_turn():
        turns.append({
            "user_message": "",
            "assistant_messages": [],
            "tool_calls": [],  # {name, input, output}
            "_noise": False,
        })

    with path.open() as f:
        for ordinal, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = rec.get("type")
            payload = rec.get("payload", {})

            if typ == "session_meta":
                session_id = payload.get("session_id") or payload.get("id")
                cwd = payload.get("cwd")
                continue

            if typ != "response_item":
                continue

            item_type = payload.get("type")
            role = payload.get("role")

            if item_type == "message" and role == "user":
                text = clean_task_text(extract_text(payload.get("content")))
                if not text:
                    continue
                new_turn()
                turns[-1]["user_message"] = text
                turns[-1]["_noise"] = text.startswith(_NOISE_PREFIXES)

            elif item_type == "message" and role == "assistant":
                text = extract_text(payload.get("content"))
                if text and turns:
                    turns[-1]["assistant_messages"].append(text)

            elif item_type in ("function_call", "tool_call", "custom_tool_call"):
                if not turns:
                    new_turn()
                name = payload.get("name", "")
                args = payload.get("arguments") or payload.get("input") or ""
                inner = _INNER_TOOL_RE.findall(str(args))
                if inner:
                    name = inner[0]
                call_id = payload.get("call_id") or payload.get("id")
                tc = {"name": name, "input": str(args)[:INPUT_CAP], "output": ""}
                turns[-1]["tool_calls"].append(tc)
                if call_id:
                    call_index[call_id] = (len(turns) - 1, len(turns[-1]["tool_calls"]) - 1)

            elif item_type in ("function_call_output", "custom_tool_call_output"):
                call_id = payload.get("call_id")
                if call_id in call_index:
                    t_idx, c_idx = call_index[call_id]
                    turns[t_idx]["tool_calls"][c_idx]["output"] = extract_output_text(payload)[:OUTPUT_CAP]

    out = []
    for i, t in enumerate(turns):
        if t["_noise"]:
            continue
        if not t["user_message"] and not t["assistant_messages"] and not t["tool_calls"]:
            continue
        out.append({
            "session_id": session_id,
            "turn_index": i,
            "thread_name": thread_name,
            "cwd": cwd,
            "file": str(path),
            "user_message": t["user_message"][:6000],
            "assistant_text": "\n".join(t["assistant_messages"])[:6000],
            "tool_calls": t["tool_calls"],
        })
    return out


def main():
    files = find_rollout_files()
    print(f"Found {len(files)} rollout files", file=sys.stderr)
    thread_names = load_thread_names()

    n_turns, n_err = 0, 0
    with OUT_PATH.open("w") as out:
        for i, path in enumerate(files):
            try:
                # thread_name lookup needs the session_id, which we only
                # know after parsing; do a cheap pre-pass via the id in
                # the filename isn't reliable, so just parse once and
                # patch thread_name after the fact.
                turns = parse_session_turns(path, "")
                if turns:
                    name = thread_names.get(turns[0]["session_id"], "")
                    if not name and turns[0]["user_message"]:
                        name = turns[0]["user_message"].splitlines()[0][:60]
                    for t in turns:
                        t["thread_name"] = name
                        out.write(json.dumps(t, ensure_ascii=False) + "\n")
                        n_turns += 1
            except Exception as e:
                n_err += 1
                print(f"ERROR parsing {path}: {e}", file=sys.stderr)
            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{len(files)} files, {n_turns} turns so far", file=sys.stderr)

    print(f"Done. turns={n_turns} err={n_err} -> {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
