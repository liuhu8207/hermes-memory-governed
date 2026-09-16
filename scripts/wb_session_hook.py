#!/usr/bin/env python
"""WorkBuddy ``SessionStart`` hook — inject the governed store's L1 rules.

Why this file exists
--------------------
WorkBuddy already has its own memory (``~/.workbuddy/MEMORY.md``, the
per-workspace ``.workbuddy/memory/``, plus server-side profile). None of that is
the governed store, so rules and profile written into HGM never reached a
WorkBuddy session. WorkBuddy hooks close that gap: for ``SessionStart`` and
``UserPromptSubmit`` the hook's stdout is appended to the conversation context.

Contract
--------
* JSON arrives on stdin (``session_id`` / ``cwd`` / ``source``).
* JSON goes out on stdout: ``hookSpecificOutput.additionalContext``.
* Windows runs hooks under **Git Bash**, not cmd/PowerShell.

Design rules (deliberate, do not "simplify" away)
-------------------------------------------------
1. **Never block the session.** Any failure — missing CLI, bad stdin, timeout,
   unparseable output — emits an empty but *successful* response. A memory
   system that can brick the editor is worse than one that is briefly silent.
2. **Bounded injection.** L1 is small today (~2.3k chars) but it is hand-edited
   and will grow. The cap keeps a runaway file from eating the context window;
   truncation is stated in the output, never silent.
3. **State the agent identity.** The CLI cannot always infer its caller, so the
   hook declares it rather than letting writes land as ``external``.
4. **Force UTF-8 on stdout.** The payload is Chinese and Windows Python picks
   its stdout encoding from the locale, not from the data. Without this the
   injected context silently becomes mojibake under a non-UTF-8 console.
5. **stderr is opt-in.** Progress is useful when debugging and dangerous when
   the host coalesces streams — it would corrupt the JSON. Set
   ``HGM_HOOK_DEBUG=1`` to see it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Do this before anything can write, or the first line may already be mis-encoded.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

REPO = Path(__file__).resolve().parent.parent
CLI = REPO / "memory_cli.py"

#: Declared explicitly. Env detection also works here, but relying on it would
#: make the hook's output depend on how WorkBuddy happens to launch it.
AGENT = "workbuddy"

#: The CLI self-relocates to the project venv when the invoked interpreter has
#: no LanceDB, so the extra model load is the only cost. Measured ~1.8s total.
TIMEOUT_SECONDS = 25

#: Ceiling for the whole injected block. Measured L1 is ~2.3k chars; the rest is
#: headroom for the capability hint.
MAX_CHARS = 6000

HEADER = "## 治理记忆（HGM）· 会话启动注入"

CAPABILITY = """
### 这个存储里还有什么

L1 规则之外，还有三层**没有**自动注入，需要时再取，避免每次都占上下文：

| 层 | 内容 | 怎么取 |
|---|---|---|
| L2 | 向量事实层（结构化事实，带 agent 署名） | `recall "<查询>"` |
| L3 | 对话归档（历史会话原文） | `recall "<查询>"` 一并返回 |
| KB | Obsidian 知识库笔记 | `kb-search "<查询>"` |

写入通道（都会过质量门，被拒时 JSON 里带 `reason`，退出码 1）：

```
python "{cli}" recall "<查询>"
python "{cli}" remember "<具体事实>"      # 写 L2，需过门禁
python "{cli}" kb-add "<标题>" "<正文>"    # 写 KB 笔记
```

**边界**：L1 是只读的——它每轮作为规则注入，开放写入等于让 agent 改自己的行为准则。
要提升某条事实的地位，走 `memory_promote.py` 或人工编辑。

**注意**：本存储与 WorkBuddy 自带记忆（`~/.workbuddy/MEMORY.md`、
`.workbuddy/memory/`）**互不感知**。它们各自独立，不要假设一处写入会在另一处可见。
"""


def read_payload() -> dict:
    """Hook payload from stdin. Missing or malformed input is not fatal."""
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def fetch_l1() -> dict:
    """Ask the CLI for the hand-written layers.

    Returns ``{}`` on any problem — the caller degrades to the capability hint
    alone rather than injecting half a rulebook.
    """
    env = dict(os.environ)
    env["HGM_AGENT"] = AGENT
    # Measured failure: with PYTHONIOENCODING=gbk inherited, the CLI emitted GBK
    # bytes while this side decoded UTF-8, json.loads failed, and the hook
    # silently degraded to the capability hint (3089 chars -> 853). The child's
    # encoding must be pinned by us, not inherited from whatever launched the
    # hook.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(CLI), "l1"],
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            env=env,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}
    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _clean(text) -> str:
    """Drop template placeholders so empty sections don't waste context.

    L1 files ship with HTML-comment placeholders. A section that is *only* a
    placeholder carries no information, and injecting it just teaches the model
    to ignore the block.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped.startswith("#") and "直接编辑此文件" in stripped:
            continue
        lines.append(line)
    body = "\n".join(lines).strip()
    # A heading with no content under it is noise.
    meaningful = [ln for ln in body.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    return body if meaningful else ""


def build_context(l1: dict) -> str:
    """Assemble the injected block, capped at :data:`MAX_CHARS`."""
    parts = [HEADER, ""]
    parts.append(f"你已接入共享记忆存储 `hermes-memory-governed`（agent 身份：`{AGENT}`）。")
    parts.append("")

    sections = (
        ("手写规则（L1 · 必须遵守）", l1.get("memory_rules_md")),
        ("用户画像（L4）", l1.get("persona_md")),
        ("用户自述（L1）", l1.get("user_profile_md")),
    )
    for title, body in sections:
        cleaned = _clean(body)
        if cleaned:
            parts.append(f"### {title}")
            parts.append("")
            parts.append(cleaned)
            parts.append("")

    parts.append(CAPABILITY.format(cli=CLI.as_posix()))

    text = "\n".join(parts).strip()
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip() + (
            f"\n\n> ⚠️ 注入内容已截断（上限 {MAX_CHARS} 字符）。"
            "如需完整内容请直接运行上面的 `l1` 命令。"
        )
    return text


def main() -> int:
    payload = read_payload()
    l1 = fetch_l1()
    context = build_context(l1)

    out = {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        },
    }
    # Diagnostic trail, not part of the context. Opt-in: if the host coalesces
    # stderr into stdout this line would corrupt the JSON payload, and losing a
    # whole session's memory injection to a log line is a bad trade.
    if os.environ.get("HGM_HOOK_DEBUG") == "1":
        sys.stderr.write(
            f"[hgm-hook] 已注入 {len(context)} 字符 "
            f"(L1 {'读取成功' if l1 else '读取失败，降级为能力提示'}; "
            f"session={payload.get('session_id', '?')})\n"
        )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
