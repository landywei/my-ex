# my-ex

[English](README.md) | **简体中文**

把你本地的 [Codex](https://openai.com/codex) 会话历史变成一个可检索的记忆库，供新的 coding agent 调用——这样即使不再用 Codex 了，之前的工作成果（bug 到底是怎么修的、做过什么决定、踩过什么坑）也不会白白丢掉。

完全本地：你的会话记录不会离开这台机器。这个仓库只包含 pipeline 代码；你提取出来的数据和 embedding 都留在 `data/` 目录下，已经被 `.gitignore` 排除。

## 工作原理

两层检索，都从同一份原始数据派生，但粒度不同：

**Session 层（粗粒度，扫一眼就行）**
1. `extract_sessions.py` —— 解析 Codex 本地的 rollout 文件（`~/.codex/{sessions,archived_sessions}/**/*.jsonl`），每个会话提取成一条记录：第一条真实的用户请求、最后一条 agent 回复、用到的工具调用名称、项目路径。会先把噪音（自动注入的环境上下文、AGENTS.md 指令、插件推荐）清理掉，保证这条记录反映的是真实诉求。
2. `summarize_sessions.py` —— 用本地 LLM（qwen2.5:7b-instruct）把每个会话压成一段简短笔记（任务 / 做法 / 结果 / 坑点）。
3. `build_index.py` —— 把摘要（加上折叠后的工具调用序列，比如 `exec_command → apply_patch → exec_command`）用本地 embedding 模型（bge-m3）编码，存进 `data/index/{vectors.npy,meta.jsonl}`。

**Turn 层（细粒度，接近原文）**
1. `extract_turns.py` —— 遍历每个原始 response item，把每个会话切成一个个 turn（一条用户消息 + agent 对此做出的所有响应：它的回复，以及每一次工具调用的具体命令/输入和返回的输出）。Codex 自己内部的审批/风险评估子对话会被过滤掉。
2. `build_turn_index.py` —— 直接对每个 turn 做 embedding（不经过 LLM 压缩），存进 `data/index/{turn_vectors.npy,turn_meta.jsonl}`。

**检索** —— 通过 `scripts/search.py`（命令行，`--detailed` 参数）或 `scripts/mcp_server.py`（MCP server，`detailed` 参数）查询任意一层。任何支持 MCP 的 coding agent（Codex CLI、OpenCode、Goose、Claude Code……）都能调用它，把相关的历史会话——或者某次 turn 里具体的命令/修复方法——拉进当前任务的上下文里。

两层都不存完整逐字原文（那个始终留在原始的 `~/.codex` rollout 文件里，每条检索结果都带一个 `file` 路径指回去）——session 层是 LLM 转述，turn 层基本是原文但做了截断（每次工具调用的输入/输出各截断到几千字符，避免个别巨大输出把存储撑爆）。

## 现状

早期个人项目，本人在日常使用中。两层都已经在约 500 个真实会话 / 约 7500 个 turn 上完整跑通验证过。

## 前置依赖

- Python 3.10+（MCP SDK 需要；建议用虚拟环境，见下文）
- [Ollama](https://ollama.com)，本地运行，需要拉两个模型：
  ```bash
  ollama pull qwen2.5:7b-instruct   # 摘要用
  ollama pull bge-m3                # embedding 用
  ```

## 安装

```bash
python3.1x -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## 使用

```bash
# session 层（粗粒度）
./.venv/bin/python scripts/extract_sessions.py     # ~/.codex -> data/sessions_extracted.jsonl
./.venv/bin/python scripts/summarize_sessions.py   # -> data/sessions_summarized.jsonl（可断点续跑）
./.venv/bin/python scripts/build_index.py          # -> data/index/{vectors.npy,meta.jsonl}

# turn 层（细粒度）
./.venv/bin/python scripts/extract_turns.py        # ~/.codex -> data/turns_extracted.jsonl
./.venv/bin/python scripts/build_turn_index.py     # -> data/index/{turn_vectors.npy,turn_meta.jsonl}

# 查询任意一层
./.venv/bin/python scripts/search.py "你的任务描述"
./.venv/bin/python scripts/search.py --detailed "我当时具体是怎么处理 X 的"
```

### 作为 MCP server

```bash
./.venv/bin/python scripts/mcp_server.py
```

暴露一个工具，`search_codex_memory(query, top_k, detailed)`。让任何支持 MCP 的
harness 指向这个命令（stdio transport），就能给它接入你的 Codex 历史检索能力。
示例（Claude Code 的 `.mcp.json`，大多数 MCP 客户端配置格式类似）：

```json
{
  "mcpServers": {
    "my-ex": {
      "command": "/absolute/path/to/my-ex/.venv/bin/python",
      "args": ["/absolute/path/to/my-ex/scripts/mcp_server.py"]
    }
  }
}
```

或者直接用 Claude Code 的 CLI：

```bash
claude mcp add --scope user my-ex -- /absolute/path/to/my-ex/.venv/bin/python /absolute/path/to/my-ex/scripts/mcp_server.py
```

## 隐私

`data/` 目录下的任何内容都不会被提交（见 `.gitignore`）。会话记录里可能包含私有代码和业务内容——不要把提取出来的数据、摘要或 embedding 索引提交进这个仓库。

## License

MIT
