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
import re
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

#: Injected only when the payload carries a working directory. This block is
#: the whole point of project attribution: without it an agent has no way to
#: learn that the facts it produces are invisible to every other agent, and it
#: will keep writing into its own local memory by default.
PROJECT_HINT = """
### 当前项目

工作目录：`{cwd}`
项目标识：`{project}`

**跨 agent 约定**。这个存储是和 Hermes Gateway、DSH 等其他 agent 共用的。
同一份代码库，你可能换一个 agent 再来——但只写在你自己会话里的东西，
别的 agent 看不见。项目里发生的事，写进来才算共享：

```
python "{cli}" recall "<查询>" --project {project}   # 本项目 + 全局事实，屏蔽其他项目
python "{cli}" remember "<具体事实>"                  # 不传 --project 时自动归属本项目
python "{cli}" remember "<通用事实>" --project ""     # 显式标为全局（主机、拓扑、凭据位置等）
```

判断标准很简单：**换一台机器、换一个项目还成立**的，就是全局事实；
**只有这个代码库才成立**的，就归属本项目。标错了不会丢数据，
但会让它在别的项目里查不到。
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


#: MSYS hands out ``/d/repos/x``, which a native ``python.exe`` cannot resolve.
#: The host exports that form, so it must be normalised before use.
_MSYS_DRIVE = re.compile(r"^/([A-Za-z])/(.*)$")


def to_windows_path(raw: str) -> str:
    """``/d/foo`` -> ``D:/foo``. Anything else is returned unchanged."""
    m = _MSYS_DRIVE.match(str(raw or "").strip())
    if m:
        return f"{m.group(1).upper()}:/{m.group(2)}"
    return str(raw or "").strip()


def resolve_cwd(payload: dict) -> tuple:
    """Find the session's working directory **and say where it came from**.

    Measured 2026-09-17: the SessionStart payload does not reliably carry
    ``cwd``. The IDE hook reference documents it, the CLI reference does not
    (it lists ``permission_mode`` in its place), and a field report from an
    instrumented desktop session lists only
    ``session_id / source / transcript_path``.

    Betting on a single field is the failure this store exists to eliminate:
    when the bet loses, the project block below vanishes **silently**, and
    that block is the only thing telling an agent that the facts it writes are
    invisible to every other agent.

    Order: the payload, then the variables the host documents, then the
    process cwd (the hook reference states hooks run inside it). The source
    travels with the value — a fallback that quietly named the wrong project
    would be worse than no project block at all.
    """
    cwd = str(payload.get("cwd") or "").strip()
    if cwd:
        return cwd, "payload.cwd"
    for var in ("CODEBUDDY_PROJECT_DIR", "CLAUDE_PROJECT_DIR"):
        value = os.environ.get(var, "").strip()
        if value:
            return to_windows_path(value), var
    try:
        return os.getcwd(), "os.getcwd()"
    except OSError:
        return "", ""


def resolve_project(cwd: str) -> str:
    """Project label for ``cwd``, reusing the CLI's own definition.

    Importing the CLI instead of re-implementing the walk keeps the two from
    drifting apart: a hook that disagreed with ``remember --project`` about what
    "the current project" means would silently partition the store. The import
    is cheap — ``memory_cli`` loads LanceDB and the embedding backend lazily, so
    this costs a path operation and nothing else.

    Returns ``""`` on any problem, and the caller then omits the block rather
    than injecting a guess.
    """
    if not cwd:
        return ""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import memory_cli  # noqa: PLC0415 — deliberate; see docstring

        return memory_cli.infer_project(cwd)
    except Exception:  # noqa: BLE001 — decoration only, never fatal
        return ""


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


def build_context(l1: dict, cwd: str = "") -> str:
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

    # Placed after the rules deliberately: the cap truncates from the bottom,
    # and "where am I" matters less than "how must I behave".
    project = resolve_project(cwd)
    if project:
        parts.append(PROJECT_HINT.format(cwd=cwd, project=project,
                                         cli=CLI.as_posix()))

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
    # Which project this session belongs to drives both the project block and
    # the write-attribution advice. The lookup is deliberately redundant —
    # see :func:`resolve_cwd` for why one field is not enough.
    cwd, cwd_source = resolve_cwd(payload)
    context = build_context(l1, cwd)

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
            f"cwd 来源={cwd_source or '未取到'}; "
            f"session={payload.get('session_id', '?')})\n"
        )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
