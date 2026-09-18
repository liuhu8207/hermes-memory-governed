#!/usr/bin/env python
"""HGM as an MCP server — the governed store as a *documented* tool surface.

Why this exists
---------------
Reaching the store from WorkBuddy currently goes through a ``UserPromptSubmit``
hook, which means reverse-engineering an undocumented payload. Measured
2026-09-17: a 6-character message arrived as ``prompt`` of length 9, a 21-
character one as length 28, and other runs carried 991; the session transcript
shows the host wraps a turn as ~2300 characters of reminders ending in
``<user_query>…</user_query>``. Four rounds of instrumentation did not explain
the short forms, and brute-forcing the payload's digest found nothing. The
fragility is not in the hook code — it is in depending on a payload nobody
documents.

MCP is the opposite: both sides agree on the schema, and *we* define the
arguments. The comparison that prompted this is TencentCloud/Octop, which never
needs a hook at all because it owns its own agent runtime — and which reaches
other agents over ACP (its built-in runners include ``codebuddy --acp``).

    For a one-shot injection the hook is fine (SessionStart works, and its
    payload is simple). For a per-turn question the tool wins.

Contract
--------
* JSON-RPC 2.0 over stdio, one message per line, stdout only — nothing else may
  be printed there or the stream is corrupted. Diagnostics go to stderr.
* **Never crash.** A tool that raises returns ``isError: true`` with the reason;
  a malformed frame gets a JSON-RPC error. The host is a long-lived process and
  a dead server looks to the user like "memory stopped working".
* Every call is appended to ``$HERMES_HOME/memory/mcp_log.txt``, sizes and
  outcomes only — the lesson from the hook is that an invisible surface is an
  undebuggable one.
* Identity comes from ``HGM_AGENT`` (set per client in its MCP config), which
  is the documented chain's second step — see ``memory_cli.resolve_agent``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

for _stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "hgm-governed-memory"

MAX_LOG_LINES = 500

#: ``tools/list`` is logged once per process; see the branch in :func:`handle`.
_LOGGED_TOOLS_LIST = False


# -- logging ----------------------------------------------------------------
def log_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / "AppData" / "Local" / "hermes")
    return Path(home) / "memory" / "mcp_log.txt"


def log(event: str, detail: str = "") -> None:
    """Append one line. A log must never be able to break a call."""
    if os.environ.get("HGM_MCP_LOG") == "0":
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
    except Exception:  # noqa: BLE001 — diagnostics are not worth a broken call
        pass


# -- tool definitions -------------------------------------------------------
def _schema(properties: dict, required: list, **extra) -> dict:
    out = {"type": "object", "properties": properties, "required": required}
    out.update(extra)
    return out


TOOLS = [
    {
        "name": "hgm_recall",
        "description": (
            "Search the shared governed memory (HGM): L2 vector facts written by any "
            "agent, L3 conversation archive, and the Obsidian knowledge base. Use this "
            "before answering anything about this machine, its services, credentials, "
            "paths, or decisions made earlier — and before deciding something was never "
            "decided. Facts carry the agent that wrote them."),
        "inputSchema": _schema(
            {
                "query": {"type": "string", "description": "What to look for, in natural language."},
                "top_k": {"type": "integer", "description": "Max hits per layer (default 5)."},
                "project": {"type": "string",
                            "description": "Narrow L2 to this project plus global facts. "
                                           "Omit to search every project."},
            },
            ["query"],
        ),
    },
    {
        "name": "hgm_remember",
        "description": (
            "Write a durable fact into the shared store so other agents can find it. "
            "Admission is gated: state something concrete and self-contained, naming "
            "hosts, ports, paths, versions or identifiers. Vague wishes are refused, and "
            "the refusal says why. Prefer one precise fact over a paragraph."),
        "inputSchema": _schema(
            {
                "fact": {"type": "string", "description": "The fact, as one concrete statement."},
                "project": {"type": "string",
                            "description": "Project to attribute it to. Omit to infer from the "
                                           "working directory; pass \"\" for a fact that holds "
                                           "everywhere (hosts, topology, credential locations)."},
                "category": {"type": "string", "description": "Free-form label (default: other)."},
                "dry_run": {"type": "boolean",
                            "description": "Run the gate and report the verdict without writing."},
            },
            ["fact"],
        ),
    },
    {
        "name": "hgm_kb_search",
        "description": ("Search the Obsidian knowledge base — long-form notes, meeting "
                        "records and operating docs, as opposed to the structured facts "
                        "hgm_recall returns."),
        "inputSchema": _schema(
            {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "description": "Max notes (default 5)."},
            },
            ["query"],
        ),
    },
    {
        "name": "hgm_kb_add",
        "description": ("Write a governed note into the knowledge base. For durable "
                        "write-ups rather than single facts; use hgm_remember for those."),
        "inputSchema": _schema(
            {
                "title": {"type": "string"},
                "body": {"type": "string"},
                "section": {"type": "string", "description": "Vault subdirectory (default notes)."},
            },
            ["title", "body"],
        ),
    },
    {
        "name": "hgm_transcribe",
        "description": (
            "Transcribe a local audio file (a meeting recording or voice memo) to "
            "text. Long files are split automatically, so an hour-long recording "
            "works, and formats ffmpeg can decode (.amr/.silk included) are "
            "normalised first. Returns the raw transcript and stores nothing — "
            "summarise it yourself and persist the distilled notes with "
            "hgm_kb_add."),
        "inputSchema": _schema(
            {
                "path": {"type": "string",
                         "description": "Absolute path to the audio file."},
                "chunk_minutes": {"type": "number",
                                  "description": "Split audio longer than this many "
                                                 "minutes (default: config "
                                                 "asr.chunk_minutes)."},
            },
            ["path"],
        ),
    },
    {
        "name": "hgm_agents",
        "description": ("Who has written what: fact and note counts per agent, including how "
                        "many entries carry no attribution. Use it to tell whether a fact "
                        "came from another agent or was never claimed by anyone."),
        "inputSchema": _schema({}, []),
    },
]


# -- tool implementations ---------------------------------------------------
def _cli():
    import memory_cli  # noqa: PLC0415 — lazy, so --help paths stay dependency-free

    return memory_cli


def _agent() -> str:
    return _cli().resolve_agent("")


def tool_recall(args: dict) -> str:
    cli = _cli()
    query = _ensure_utf8(str(args.get("query") or "").strip())
    if not query:
        raise ValueError("query is required")
    top_k = int(args.get("top_k") or 5)
    project = cli.normalize_project(args.get("project") or "")

    errors: list = []
    l2 = cli.recall_l2(query, top_k, errors, project=project) if project else \
        cli.recall_l2(query, top_k, errors)
    kb_refused: list = []
    kb = cli.cmd_kb_search(cli.load_config(), query, top_k, "", kb_refused)

    lines = []
    hits = l2.get("hits") or []
    ranking = l2.get("ranking") or {}
    lines.append(f"L2 facts: {len(hits)} hit(s)  [{ranking.get('basis', '?')}]")
    for h in hits:
        who = h.get("agent") or "unattributed"
        lines.append(f"  - ({h.get('score'):.4f}) {h.get('content')}   <{who}>")
    if not hits:
        # Stated, never implied: "nothing matched" and "could not read" must not
        # look the same — the store exists to remove exactly this ambiguity.
        note = ranking.get("note") or "no fact above the floor"
        lines.append(f"  ({note})")
    sem = ranking.get("semantic")
    if isinstance(sem, dict) and sem.get("available") is False:
        lines.append(f"  ! semantic channel unavailable: {sem.get('reason')}")

    lines.append(f"KB notes: {len(kb)} hit(s)")
    for note in kb:
        lines.append(f"  - {note.get('title')}  ({note.get('path')})")
    if errors:
        lines.append("errors: " + "; ".join(str(e) for e in errors))
    if project:
        lines.append(f"project scope: {project}")
    return "\n".join(lines)


def _ensure_utf8(text: str) -> str:
    """Delegate to the plugin's single implementation; see ``_text.py``.

    There were three copies of this logic and none of them guarded the path
    that actually reaches L2. This one is now a call, not a copy.
    """
    return _cli().plugin_module("_text").sanitize_utf8(text)


def tool_remember(args: dict) -> str:
    cli = _cli()
    fact = _ensure_utf8(str(args.get("fact") or "").strip())
    if not fact:
        raise ValueError("fact is required")
    raw_project = args.get("project", None)
    project = None if raw_project is None else cli.normalize_project(raw_project)
    result = cli.cmd_remember(cli.load_config(), fact,
                              _agent(), args.get("category") or "other",
                              bool(args.get("dry_run")), project)
    return json.dumps(result, ensure_ascii=False, indent=2)


def tool_kb_search(args: dict) -> str:
    cli = _cli()
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("query is required")
    refused: list = []
    hits = cli.cmd_kb_search(cli.load_config(), query,
                             int(args.get("top_k") or 5), "", refused)
    lines = [f"{len(hits)} note(s)"]
    for note in hits:
        lines.append(f"  - {note.get('title')}  ({note.get('path')})  "
                     f"{note.get('excerpt') or ''}".rstrip())
    if not hits:
        lines.append("  (nothing matched)")
    if refused:
        lines.append("refused: " + "; ".join(str(r) for r in refused))
    return "\n".join(lines)


def tool_kb_add(args: dict) -> str:
    cli = _cli()
    title = str(args.get("title") or "").strip()
    body = str(args.get("body") or "")
    if not title or not body:
        raise ValueError("title and body are required")
    result = cli.cmd_kb_add(cli.load_config(), title, body,
                            args.get("section") or "notes", [], [], None, _agent())
    return json.dumps(result, ensure_ascii=False, indent=2)


def tool_transcribe(args: dict) -> str:
    cli = _cli()
    path = str(args.get("path") or "").strip()
    if not path:
        raise ValueError("path is required")
    result = cli.cmd_transcribe(cli.load_config(), path, None,
                                args.get("chunk_minutes"))
    return json.dumps(result, ensure_ascii=False, indent=2)


def tool_agents(_args: dict) -> str:
    return json.dumps(_cli().cmd_agents(_cli().load_config()),
                      ensure_ascii=False, indent=2)


HANDLERS = {
    "hgm_recall": tool_recall,
    "hgm_remember": tool_remember,
    "hgm_kb_search": tool_kb_search,
    "hgm_kb_add": tool_kb_add,
    "hgm_transcribe": tool_transcribe,
    "hgm_agents": tool_agents,
}


# -- JSON-RPC ---------------------------------------------------------------
def _result(msg_id, payload) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": payload}


def _error(msg_id, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict:
    """One request in, one response out. Returns None for notifications."""
    if not isinstance(msg, dict):
        return _error(None, -32600, "invalid request")
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        params = msg.get("params") or {}
        client = params.get("clientInfo") or {}
        # Logged because the previous host launch was *invisible*: the server
        # started, was terminated without going through the read loop, and left
        # nothing behind to say whether the handshake had completed. An MCP
        # server that only logs tool calls cannot answer "did the host even get
        # the tool list?" — the same blind spot the WorkBuddy hook had.
        log("initialize", f"protocol={params.get('protocolVersion')} "
                          f"client={client.get('name')}/{client.get('version')}")
        return _result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
        })
    if method in ("notifications/initialized", "initialized"):
        return None                      # notification: no reply, by protocol
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        # Logged once per process, not once per call. A host re-lists the tools
        # on every turn, and with a 500-line cap that flood **evicted every
        # `call` record** — the log looked healthy while the answer to "did any
        # agent use a tool today?" had already been pushed out of it. The single
        # line still proves the host read the tool list.
        global _LOGGED_TOOLS_LIST
        if not _LOGGED_TOOLS_LIST:
            _LOGGED_TOOLS_LIST = True
            log("tools/list", f"count={len(TOOLS)} (logged once per process)")
        return _result(msg_id, {"tools": TOOLS})
    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        handler = HANDLERS.get(name)
        if handler is None:
            return _error(msg_id, -32602, f"unknown tool: {name!r}")
        try:
            text = handler(args)
            log("call", f"{name} ok out={len(text)}B agent={_agent()}")
            return _result(msg_id, {"content": [{"type": "text", "text": text}]})
        except Exception as exc:  # noqa: BLE001 — reported, not fatal
            detail = f"{type(exc).__name__}: {exc}"
            log("call", f"{name} FAILED {detail}")
            # isError, not a JSON-RPC error: the call was well formed, the work
            # failed, and the caller should read the reason rather than retry a
            # protocol-level mistake.
            return _result(msg_id, {
                "content": [{"type": "text", "text": f"failed: {detail}"}],
                "isError": True,
            })
    return _error(msg_id, -32601, f"method not found: {method!r}")


def ensure_usable_interpreter() -> None:
    """Re-exec under an interpreter that can see the store.

    This server imports ``memory_cli`` and calls its functions **in-process**, so
    it needs LanceDB in *this* interpreter. That is a stronger requirement than
    the CLI's, and the CLI's own relocation cannot be reused directly:
    ``memory_cli.bootstrap_interpreter`` re-execs ``memory_cli.py``, not us.

    Its two helpers are reused instead — ``_missing_modules`` and
    ``_candidate_interpreters`` — so "which interpreter can do this" keeps a
    single definition rather than gaining a second one here.

    ``subprocess`` with inherited stdio, not ``os.execve``: measured 2026-09-17,
    an ``execve`` into a different interpreter's ``python.exe`` **segfaulted**
    (exit 139, nothing on stderr) before the handshake — a silent death, the
    worst possible outcome for a surface the host expects to keep answering.
    Handing the child our own stdin/stdout instead means the host sees one
    continuous session, and it is the same approach ``memory_cli``'s own
    bootstrap has used in production. A flag in the environment stops a failed
    attempt from looping; if every candidate fails we carry on and let tool
    calls report the missing layer, which is visible — unlike silently
    answering "nothing remembered".
    """
    try:
        import memory_cli  # noqa: PLC0415
        # Asked in this order on purpose: when nothing is missing (the normal
        # case) the candidate list is never even built.
        missing = memory_cli._missing_modules()
        if not missing:
            return
        candidates = memory_cli._candidate_interpreters()
    except Exception:  # noqa: BLE001 — no CLI import means nothing to relocate to
        return
    if os.environ.get("HGM_MCP_BOOTSTRAPPED") == "1":
        return
    try:
        current = Path(sys.executable).resolve()
    except OSError:
        current = None
    log("bootstrap", f"missing={missing} from={current} "
                     f"venv={'yes' if sys.prefix != sys.base_prefix else 'no'} "
                     f"PYTHONHOME={'set' if os.environ.get('PYTHONHOME') else 'unset'} "
                     f"PYTHONPATH={'set' if os.environ.get('PYTHONPATH') else 'unset'} "
                     f"base={sys.base_prefix}")
    script = str(Path(__file__).resolve())
    for cand in candidates:
        target = Path(cand)
        try:
            if not target.is_file() or target.resolve() == current:
                continue
        except OSError:
            continue
        env = dict(os.environ)
        env["HGM_MCP_BOOTSTRAPPED"] = "1"
        try:
            # No capture_output: the child inherits our stdin/stdout/stderr, so
            # it talks to the host directly and this process only forwards the
            # exit code.
            proc = subprocess.run([str(target), script, *sys.argv[1:]], env=env)
        except OSError:
            continue
        sys.exit(proc.returncode)
    sys.stderr.write(f"[hgm-mcp] warning: no interpreter with {missing}; "
                     f"store calls will fail visibly\n")


def main() -> int:
    ensure_usable_interpreter()
    log("start", f"agent={_agent()} repo={REPO}")
    # stderr is free for diagnostics here — it is stdout that must stay pure.
    # A host that captures stderr gets the startup record; the file log keeps it
    # across restarts.
    sys.stderr.write(f"[hgm-mcp] started agent={_agent()} protocol={PROTOCOL_VERSION}\n")
    sys.stderr.flush()
    first_frame = True
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if first_frame:
            first_frame = False
            # Framing is the one mismatch that makes a host and a server talk
            # past each other: MCP stdio is newline-delimited JSON, but some
            # clients send LSP-style ``Content-Length`` headers instead. Both
            # look like "a line arrived", and the disagreement surfaces only as
            # the host quietly giving up. So the shape of the first frame — its
            # length, whether it starts with ``{``, whether it carries a header
            # — is recorded once. No content: at handshake time it is protocol.
            log("first_frame", f"len={len(line)} json={line.startswith('{')} "
                               f"content_length_header="
                               f"{line.lower().startswith('content-length')}")
        try:
            msg = json.loads(line)
        except (ValueError, TypeError):
            # Unparseable frame: answer with an error rather than dying, so one
            # bad byte cannot take the surface down for the rest of the session.
            print(json.dumps(_error(None, -32700, "parse error")), flush=True)
            log("parse_error", f"len={len(line)}")
            continue
        try:
            response = handle(msg)
        except Exception as exc:  # noqa: BLE001 — the loop must survive anything
            response = _error(msg.get("id") if isinstance(msg, dict) else None,
                              -32603, f"internal error: {type(exc).__name__}: {exc}")
            log("internal_error", traceback.format_exc()[-300:])
            sys.stderr.write(f"[hgm-mcp] internal error: {type(exc).__name__}: {exc}\n")
            sys.stderr.flush()
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)
    log("stop", "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
