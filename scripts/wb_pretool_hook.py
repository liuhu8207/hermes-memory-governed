#!/usr/bin/env python
"""WorkBuddy ``PreToolUse`` hook — keep agents out of the hand-written layers.

Why this exists
---------------
The governed store's contract says L1 (``memory/MEMORY.md``, ``USER.md``) and L4
(``persona.md``) are not agent-writable: L1 is injected as rules into **every**
session, so an agent that can write it can rewrite its own operating
instructions — and every other agent's.

Measured 2026-09-17, that contract had no teeth. Asked to "记一下：以后重启
Gateway 必须用 schtasks", the agent did not call ``hgm_remember`` at all. It read
its way to ``~/AppData/Local/hermes/memory/MEMORY.md`` and **edited the file** —
``Edit`` twice, zero memory-tool calls — and explained itself: "这条规则要同时
管住我，和你的 Hermes / DSH". The reasoning was sound; the store was simply not
in a position to stop it, because the file is on disk and the agent has an editor.
The L1 rule grew from 1374 to 2241 bytes.

So the rule is enforced where it can actually be enforced: at the tool call.
``PreToolUse`` can answer ``permissionDecision: deny``, which the host honours.

What is blocked, and what is deliberately not
---------------------------------------------
* ``Edit`` / ``Write`` targeting a protected file — denied outright, and the
  path is matched the way the filesystem reads it, not the way it is typed:
  ``memory.md`` and ``MEMORY.md.`` are the same file as ``MEMORY.md`` here, so
  they are denied too (:func:`_resolve_key`).
* ``Bash`` mentioning a protected file **and** a write indicator — denied. This
  one is a heuristic, and a narrow one: both conditions must hold, so ``cat``,
  ``grep``, ``ls`` and every other read still pass. A command that writes to
  those files through something this does not recognise will get through; the
  goal is to close the obvious door and to make the intent unmistakable, not to
  claim a sandbox.
* Everything else — allowed. A guard that blocks legitimate work is worse than
  the hole it closes, because it gets switched off.

Contract
--------
* JSON in on stdin, JSON out on stdout, one line, stdout only.
* **Never blocks by accident.** Any parse failure, unknown tool, or internal
  error answers ``allow``. The failure mode of a broken guard must be "the rule
  is unenforced", never "the session cannot edit anything".
* Every decision involving a protected path is logged with its reason, because a
  guard whose refusals are invisible cannot be tuned.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

#: Files an agent must not write. L1 is the rules layer (injected everywhere);
#: persona is the generated profile. All are human-owned by contract.
PROTECTED_NAMES = ("MEMORY.md", "USER.md", "persona.md", "persona_meta.json")

#: Tools whose whole purpose is writing a file — matched on the path alone.
#:
#: Two naming styles, both accepted on purpose. The hook reference states that
#: the ``matcher`` accepts either style, but that the ``tool_name`` the *script*
#: receives depends on the run environment: the CLI uses ``Write`` / ``Edit`` /
#: ``Bash`` while the IDE uses ``write_to_file`` / ``replace_in_file`` /
#: ``execute_command``. Measured 2026-09-17 in this environment, the names are
#: CLI-style (Bash 1307, Edit 370, Write 108 across one evening's sessions) — so
#: the guard did work. It would have failed *silently* under the other style,
#: which is the failure this project keeps meeting: not a crash, just a rule that
#: quietly stops being enforced. Accepting both costs one tuple.
FILE_WRITE_TOOLS = (
    "Edit", "Write", "MultiEdit", "NotebookEdit",                 # CLI style
    "replace_in_file", "write_to_file", "multi_replace_in_file",  # IDE style
    "insert_content", "apply_diff", "create_file",
)

#: The shell tool, likewise under both names.
SHELL_TOOLS = ("Bash", "execute_command")

#: Tokens that make a shell command a write rather than a read. Narrow on
#: purpose: a false positive here blocks real work.
WRITE_TOKENS = (">", ">>", "tee ", "sed -i", "Set-Content", "Add-Content",
                "Out-File", "open(", "write_text", "writeFile", "cp ", "mv ",
                "rm ", "del ", "truncate", "printf ", "echo ")

REASON = (
    "refused: {name} 属于人工手写层（L1 规则 / L4 人格），agent 不可写 —— "
    "它每轮作为规则注入到所有 agent，改它等于改行为准则。"
    "要记事实请用 `hgm_remember`（写入 L2，会被其他 agent 检索到）；"
    "要立规则请告诉用户，由用户决定是否提升进 L1。"
)

MAX_LOG_LINES = 500


def memory_dir() -> Path:
    home = os.environ.get("HERMES_HOME") or str(
        Path.home() / "AppData" / "Local" / "hermes")
    return Path(home) / "memory"


def log_path() -> Path:
    return memory_dir() / "guard_log.txt"


def log(event: str, detail: str) -> None:
    """Append one line. A log must never be able to break a decision."""
    if os.environ.get("HGM_GUARD_LOG") == "0":
        return
    try:
        from datetime import datetime

        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        lines.append(f"{datetime.now().isoformat(timespec='seconds')}\t{event}\t{detail}")
        if len(lines) > MAX_LOG_LINES:
            lines = lines[-MAX_LOG_LINES:]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 — diagnostics are not worth a wrong decision
        pass


#: Keys that carry the file a write tool is aiming at. Restricting the check to
#: these is what keeps the guard from blocking a *document about* L1: matching
#: every string in the tool input would refuse any edit whose new text mentions
#: ``MEMORY.md`` — which is most writing about this memory system, including the
#: notes explaining this guard.
PATH_KEYS = ("file_path", "filePath", "path", "notebook_path", "target_file",
             "file", "target")


def _strings(value, out: list) -> None:
    """Every string anywhere in the tool input, keys included."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for k, v in value.items():
            out.append(str(k))
            _strings(v, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _strings(item, out)


def _target_paths(tool_input: dict) -> list:
    """The paths a write tool is aiming at, in preference order."""
    out = []
    if isinstance(tool_input, dict):
        for key in PATH_KEYS:
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
    if out:
        return out
    # Unknown shape: fall back to strings that look like a path — a separator
    # somewhere before the file name, so prose that merely names the file is not
    # mistaken for a target.
    every = []
    _strings(tool_input, every)
    return [s for s in every if "/" in s or "\\" in s]


#: A trailing dot-run on a path segment. Windows strips those before opening the
#: file, so ``MEMORY.md.`` and ``MEMORY.md`` are the same file on this box.
_TRAILING_DOTS = re.compile(r"\.+(?=/|$)")


def _resolve_key(text: str) -> str:
    """Normalise a path the way the *filesystem* will before matching it.

    Matching the literal text is not enough, because the guard cannot be
    bypassed by *anything* an agent might plausibly type — including the same
    file spelled differently. Measured 2026-09-17 on this machine, two Windows
    behaviours each turned a denial into an allow::

        .../hermes/memory/MEMORY.md    → deny     (correct)
        .../hermes/memory/memory.md    → allow    ✗ case-insensitive FS
        .../hermes/memory/Memory.MD    → allow    ✗ case-insensitive FS
        .../hermes/memory/MEMORY.md.   → allow    ✗ trailing dot stripped
        .../hermes/memory/USER.md      → deny     (correct)

    ``normcase`` supplies case-insensitivity where the platform defines it and
    is a no-op elsewhere; ``re.IGNORECASE`` in :func:`protected_hit` then keeps
    the rule true on both platforms rather than only on Windows. Separators are
    re-normalised *after* ``normcase`` because on Windows it rewrites them.
    """
    flat = os.path.normcase(text).replace("\\", "/").strip().strip("\"'")
    return _TRAILING_DOTS.sub("", flat)


def protected_hit(text: str) -> str:
    """The protected file name this text names as a *path*, or ``""``.

    Both separators are accepted because Windows callers produce both, and the
    comparison is spelling-insensitive because the filesystem is — see
    :func:`_resolve_key`.
    """
    if not text:
        return ""
    flat = _resolve_key(text)
    for name in PROTECTED_NAMES:
        if re.search(r"(?:^|[/\s=])" + re.escape(name) + r"(?:$|[\"'\s,)])",
                     flat, re.IGNORECASE):
            return name
    return ""


def read_payload() -> dict:
    try:
        raw = sys.stdin.read()
    except Exception:  # noqa: BLE001
        return {}
    try:
        data = json.loads(raw) if raw.strip() else {}
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def decide(payload: dict) -> tuple:
    """Return ``(decision, reason, tool, hit)`` — ``decision`` is allow or deny."""
    tool = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input") or {}

    if tool in FILE_WRITE_TOOLS:
        for candidate in _target_paths(tool_input):
            hit = protected_hit(candidate)
            if hit:
                return "deny", REASON.format(name=hit), tool, hit
        return "allow", "", tool, ""

    if tool in SHELL_TOOLS:
        strings = []
        _strings(tool_input, strings)
        # Both conditions are required: naming the file is not enough (the agent
        # may be reading it), and a write token alone is not enough (it may be
        # writing something else).
        named = next((s for s in strings if protected_hit(s)), "")
        writes = any(tok in s for s in strings for tok in WRITE_TOKENS)
        if named and writes:
            hit = protected_hit(named)
            return "deny", REASON.format(name=hit), tool, hit
        return "allow", "", tool, ""

    return "allow", "", tool, ""


def main() -> int:
    payload = read_payload()
    decision, reason, tool, hit = decide(payload)
    noted = bool(hit)
    if noted:
        log(decision, f"tool={tool} file={hit} session={payload.get('session_id', '?')}")

    out = {"continue": True}
    if decision == "deny":
        out["hookSpecificOutput"] = {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    # Allowed calls get no hookSpecificOutput at all: saying "allow" explicitly
    # would override a decision the user's own settings may have made.
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 — a guard must never break a session
        log("internal_error", f"{type(exc).__name__}: {exc}")
        print(json.dumps({"continue": True}, ensure_ascii=False))
        sys.exit(0)
