#!/usr/bin/env python3
"""Summarize extracted Codex sessions into short retrieval-friendly notes.

Reads data/sessions_extracted.jsonl (from extract_sessions.py), calls a
local Ollama model per session, and writes data/sessions_summarized.jsonl.
Resumable: sessions already summarized (by session_id) are skipped.
"""
import json
import sys
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:7b-instruct"

ROOT = Path(__file__).parent.parent
IN_PATH = ROOT / "data" / "sessions_extracted.jsonl"
OUT_PATH = ROOT / "data" / "sessions_summarized.jsonl"

PROMPT_TEMPLATE = """你在阅读一段本地 Codex coding agent 的历史会话记录，需要把它压缩成简短的检索用笔记，方便以后按任务语义搜索到这次会话。

项目路径: {cwd}
会话标题: {thread_name}
用户的初始请求（可能包含系统指令噪音，忽略无关部分）:
{first_user_message}

最终结果（agent 的最后一条回复）:
{last_assistant_message}

用到的工具调用次数统计: {tool_summary}
涉及的文件（部分）: {files}

请用中文输出一个 JSON 对象，字段如下，每个字段 1-2 句话，简洁：
- "task": 用户实际想做什么（去掉系统指令噪音，说人话）
- "approach": agent 实际是怎么做的（关键步骤或方法，不是逐字复述工具调用）
- "outcome": 结果如何，成功/失败/部分完成
- "gotchas": 有没有踩坑、返工、特殊约束或需要注意的点；没有就写"无"

只输出 JSON，不要其他文字。"""


def summarize_tool_calls(names):
    from collections import Counter
    c = Counter(names)
    top = c.most_common(8)
    return ", ".join(f"{n}x{cnt}" for n, cnt in top)


def call_ollama(prompt: str, timeout=120) -> dict:
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return json.loads(data["response"])


def load_done_ids():
    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["session_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main():
    if not IN_PATH.exists():
        print(f"missing {IN_PATH}, run extract_sessions.py first", file=sys.stderr)
        sys.exit(1)

    done_ids = load_done_ids()
    print(f"{len(done_ids)} sessions already summarized, resuming", file=sys.stderr)

    records = [json.loads(l) for l in IN_PATH.open()]
    todo = [r for r in records if r["session_id"] not in done_ids]
    print(f"{len(todo)} sessions to summarize", file=sys.stderr)

    with OUT_PATH.open("a") as out:
        for i, rec in enumerate(todo):
            if not rec.get("first_user_message") and not rec.get("last_assistant_message"):
                continue  # empty/degenerate session, skip
            prompt = PROMPT_TEMPLATE.format(
                cwd=rec.get("cwd") or "unknown",
                thread_name=rec.get("thread_name") or "(无标题)",
                first_user_message=(rec.get("first_user_message") or "")[:3000],
                last_assistant_message=(rec.get("last_assistant_message") or "")[:3000],
                tool_summary=summarize_tool_calls(rec.get("tool_call_names", [])),
                files=", ".join(rec.get("files_touched_sample", [])[:10]) or "无",
            )
            try:
                summary = call_ollama(prompt)
            except Exception as e:
                print(f"ERROR summarizing {rec['session_id']}: {e}", file=sys.stderr)
                continue

            out_rec = {
                "session_id": rec["session_id"],
                "thread_name": rec.get("thread_name"),
                "cwd": rec.get("cwd"),
                "created_at": rec.get("created_at"),
                "file": rec.get("file"),
                "n_tool_calls": rec.get("n_tool_calls"),
                "summary": summary,
            }
            out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            out.flush()

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(todo)} summarized", file=sys.stderr)

    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()
