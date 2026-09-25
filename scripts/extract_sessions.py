#!/usr/bin/env python3
"""Extract structured records from local Codex rollout session files.

Reads ~/.codex/{archived_sessions,sessions}/**/*.jsonl and writes one
compact JSON record per session to an output .jsonl file, for later
summarization / embedding.
"""
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

CODEX_HOME = Path.home() / ".codex"
OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / "data" / "sessions_extracted.jsonl"

# Codex injects boilerplate wrapper blocks into the first user message
# (environment context, AGENTS.md instructions, plugin suggestions, ...).
# Strip these so downstream summarization/search sees the actual request.
NOISE_TAGS = [
    "environment_context",
    "recommended_plugins",
    "INSTRUCTIONS",
    "in-app-browser-context",
    "response-annotations",
    "turn_aborted",
    "subagents",
]
_NOISE_RE = re.compile(
    r"<(" + "|".join(re.escape(t) for t in NOISE_TAGS) + r")\b[^>]*>.*?</\1>",
    re.DOTALL,
)
_HEADING_RE = re.compile(r"^#+\s*AGENTS\.md instructions\s*$", re.MULTILINE)


def clean_task_text(text: str) -> str:
    if not text:
        return text
    text = _NOISE_RE.sub("", text)
    text = _HEADING_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def find_rollout_files():
    seen = {}
    for base in [CODEX_HOME / "archived_sessions", CODEX_HOME / "sessions"]:
        if not base.exists():
            continue
        for p in base.rglob("*.jsonl"):
            # dedupe by session id embedded in filename
            seen[p.name] = p
    return list(seen.values())

def extract_text(content):
    """response_item content can be a string or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                t = block.get("text") or block.get("input") or ""
                if isinstance(t, str) and t:
                    parts.append(t)
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)

def parse_session(path: Path):
    session_id = None
    cwd = None
    thread_source = None
    created_at = None
    model = None
    user_messages = []
    assistant_messages = []
    tool_calls = []  # (name, brief)
    files_touched = set()

    with path.open() as f:
        for line in f:
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
                thread_source = payload.get("thread_source")
                created_at = payload.get("timestamp")
                model = payload.get("model_provider")

            elif typ == "response_item":
                item_type = payload.get("type")
                role = payload.get("role")
                if item_type == "message" and role == "user":
                    text = extract_text(payload.get("content"))
                    if text:
                        user_messages.append(text)
                elif item_type == "message" and role == "assistant":
                    text = extract_text(payload.get("content"))
                    if text:
                        assistant_messages.append(text)
                elif item_type in ("function_call", "tool_call"):
                    name = payload.get("name", "")
                    args = payload.get("arguments") or payload.get("input") or ""
                    brief = str(args)[:200]
                    tool_calls.append((name, brief))
                    if name in ("apply_patch", "edit_file", "write_file"):
                        # try to pull a path out of the args
                        for token in str(args).split():
                            if "/" in token and len(token) < 200:
                                files_touched.add(token.strip("\"',:"))

    # The first user message is often a pure boilerplate turn (environment
    # context, AGENTS.md instructions) injected before the user's actual
    # request. Use the first message that has real content after cleaning.
    cleaned_first_msg = ""
    for um in user_messages:
        cleaned = clean_task_text(um)
        if len(cleaned) > 5:
            cleaned_first_msg = cleaned
            break

    return {
        "session_id": session_id,
        "file": str(path),
        "cwd": cwd,
        "thread_source": thread_source,
        "created_at": created_at,
        "model_provider": model,
        "n_user_msgs": len(user_messages),
        "n_assistant_msgs": len(assistant_messages),
        "n_tool_calls": len(tool_calls),
        "first_user_message": cleaned_first_msg[:4000],
        "last_assistant_message": assistant_messages[-1][:4000] if assistant_messages else "",
        "tool_call_names": [t[0] for t in tool_calls],
        "files_touched_sample": list(files_touched)[:20],
    }

def load_thread_names():
    idx_path = CODEX_HOME / "session_index.jsonl"
    names = {}
    if idx_path.exists():
        with idx_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    names[rec["id"]] = rec.get("thread_name")
                except (json.JSONDecodeError, KeyError):
                    continue
    return names

def main():
    files = find_rollout_files()
    print(f"Found {len(files)} rollout files", file=sys.stderr)
    thread_names = load_thread_names()

    n_ok, n_err = 0, 0
    with OUT_PATH.open("w") as out:
        for i, path in enumerate(files):
            try:
                rec = parse_session(path)
                name = thread_names.get(rec["session_id"], "")
                if not name and rec["first_user_message"]:
                    name = rec["first_user_message"].splitlines()[0][:60]
                rec["thread_name"] = name
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_ok += 1
            except Exception as e:
                n_err += 1
                print(f"ERROR parsing {path}: {e}", file=sys.stderr)
            if (i + 1) % 50 == 0:
                print(f"  processed {i+1}/{len(files)}", file=sys.stderr)

    print(f"Done. ok={n_ok} err={n_err} -> {OUT_PATH}", file=sys.stderr)

if __name__ == "__main__":
    main()
