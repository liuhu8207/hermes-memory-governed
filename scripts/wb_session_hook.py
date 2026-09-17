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
import time
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
| L2 | 向量事实层（结构化事实，带 agent 署名） | `hgm_recall` |
| L3 | 对话归档（历史会话原文） | `hgm_recall` 一并返回 |
| KB | Obsidian 知识库笔记 | `hgm_kb_search` |

**优先用 MCP 工具**——`hgm_recall` / `hgm_remember` / `hgm_kb_search` /
`hgm_kb_add` / `hgm_agents`。它们是常驻进程，一次召回约 **0.3s**；
而下面的 CLI 每次要 **2.8s**（要重新起解释器并加载索引）。

> 工具不在你的工具列表里，才改用 CLI。以前这里只写了 CLI 命令，结果
> agent 明明有工具却全部走 CLI——**指引写什么，它就用什么**。

写入（都会过质量门，被拒时会说明原因）：

```
hgm_remember "<具体事实>"          # 写 L2；不传 project 时按工作目录归属
hgm_kb_add "<标题>" "<正文>"        # 写 KB 笔记

# 仅当没有 MCP 工具时：
python "{cli}" remember "<具体事实>"
python "{cli}" kb-add "<标题>" "<正文>"
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
hgm_recall "<查询>" project="{project}"   # 本项目 + 全局事实，屏蔽其他项目
hgm_remember "<具体事实>"                  # 不传 project 时自动归属本项目
hgm_remember "<通用事实>" project=""       # 显式标为全局（主机、拓扑、凭据位置等）
```

没有 MCP 工具时，等价命令是 `python "{cli}" recall|remember ...`。

判断标准很简单：**换一台机器、换一个项目还成立**的，就是全局事实；
**只有这个代码库才成立**的，就归属本项目。标错了不会丢数据，
但会让它在别的项目里查不到。
"""


#: When to go and look, as opposed to what is there to look at. The capability
#: hint below already says *what* exists and *how* to fetch it; measured
#: experience is that this is not the part agents get wrong — they know the
#: commands exist and still answer from memory. Naming the situations is what
#: turns a tool into a habit.
#:
#: Deliberately short, and deliberately paired with a "do NOT look" line: a
#: block that only ever says "look things up" trains the model to spend context
#: on general-knowledge questions this store cannot answer.
WHEN_TO_LOOK = """
### 什么时候该先查再答

命中任一条，先 `hgm_recall` 再回答，别凭印象：

| 触发情形 | 例子 |
|---|---|
| 涉及本机或内网设施 | 哪台机器、什么端口、服务跑在哪 |
| 凭据与访问方式 | 密码存哪、怎么连上、免密怎么配的 |
| 动手改配置之前 | 重启服务、改代理、动网关或路由 |
| 「上次／之前／为什么」 | 上次那个问题怎么修的、当初为什么这么定 |
| 你正要写一条新事实 | 先查有无重复或与既有事实冲突 |

**反过来**：纯通用知识（算法、语言语法、公开概念、常识）**不要查** —— 这个
存储里只有你这套环境的事实，通用问题查了也只会得到无关内容。
"""


#: Sections of ``persona.md`` that are a *derived dump* rather than a persona.
#: The L4 generator appends the whole of L2 to the profile — ``## Known Facts``
#: is literally described as a dump in ``_sync.py``. Injecting it defeats three
#: things at once, so it is stripped before injection:
#:
#: 1. the tool surface — measured 2026-09-17, 20 of 23 L2 facts were already in
#:    this injection, so an agent had no reason ever to call ``hgm_recall``;
#: 2. project isolation — the dump is not filtered by ``project``, so a fact
#:    scoped to one project reached every session, which is exactly what the
#:    scope was built to prevent;
#: 3. roughly 1KB of duplication in every session's context.
#:
#: The *persona* part (who the user is, what they prefer) is left alone; only
#: the generated listings are cut.
_DERIVED_PERSONA_SECTIONS = ("## Knowledge Areas", "## Known Facts", "## Stats")


def _persona_only(text) -> str:
    """Drop the generated fact listing from ``persona.md``."""
    if not isinstance(text, str) or not text.strip():
        return ""
    cut = len(text)
    for header in _DERIVED_PERSONA_SECTIONS:
        at = text.find(header)
        if at != -1:
            cut = min(cut, at)
    return text[:cut].strip()


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


#: How stale the offline snapshot may be before a session start refreshes it.
#: ``remember`` refreshes the file itself (see ``memory_cli.cmd_snapshot``), so
#: this normally finds a fresh one and does nothing; it is the safety net for
#: facts the plugin wrote directly, which never pass through the CLI.
SNAPSHOT_MAX_AGE_SECONDS = 3600

#: Separate from the L1 timeout: a refresh is a convenience, and it must not be
#: able to hold a session open for as long as a rulebook fetch may.
SNAPSHOT_TIMEOUT_SECONDS = 20


def snapshot_file():
    """Where the CLI writes the snapshot. Asked of the CLI, not re-derived."""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import memory_cli  # noqa: PLC0415 — deliberate; see resolve_project

        return Path(memory_cli.snapshot_path())
    except Exception:  # noqa: BLE001 — no path means no refresh
        return None


def ensure_snapshot() -> str:
    """Refresh the offline snapshot when missing or stale; return a status word.

    The snapshot exists so the per-prompt hook can answer "does the store
    already say something about this?" without spawning an interpreter and
    loading LanceDB on every message. Something has to keep it current, and the
    two cheap moments are: right after a write (the CLI does that, LanceDB being
    already open) and session start (here).

    Never raises. A convenience that can fail a session is not a convenience.
    """
    path = snapshot_file()
    if path is None:
        return "path unknown"
    age = None
    try:
        age = time.time() - path.stat().st_mtime
        if age <= SNAPSHOT_MAX_AGE_SECONDS:
            return f"fresh ({age / 60:.0f}m)"
    except OSError:
        pass                                  # missing — refresh it below

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    try:
        proc = subprocess.run([sys.executable, str(CLI), "snapshot"],
                              capture_output=True, env=env,
                              timeout=SNAPSHOT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        return f"refresh failed: {type(exc).__name__}"
    if proc.returncode != 0:
        return f"refresh rc={proc.returncode}"
    return "refreshed" if age is not None else "created"


def build_context(l1: dict, cwd: str = "") -> str:
    """Assemble the injected block, capped at :data:`MAX_CHARS`."""
    parts = [HEADER, ""]
    parts.append(f"你已接入共享记忆存储 `hermes-memory-governed`（agent 身份：`{AGENT}`）。")
    parts.append("")

    sections = (
        ("手写规则（L1 · 必须遵守）", l1.get("memory_rules_md")),
        # persona.md carries a generated dump of every L2 fact; see
        # _DERIVED_PERSONA_SECTIONS for why that must not be injected.
        ("用户画像（L4）", _persona_only(l1.get("persona_md"))),
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

    # Ahead of the capability block: if the cap ever bites, the situations are
    # worth more than the command syntax, which the agent can rediscover from
    # `--help` while a missed lookup leaves it confidently wrong.
    parts.append(WHEN_TO_LOOK.strip())

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
    # Keep the per-prompt hook's data current. Cheap in the common case (a
    # stat), and it runs once per session rather than once per message.
    snap = ensure_snapshot()
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
            f"cwd 来源={cwd_source or '未取到'}; 快照={snap}; "
            f"session={payload.get('session_id', '?')})\n"
        )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
