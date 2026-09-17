#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Governed Memory CLI — the shared-memory contract for ANY agent.

Every command reads or writes the SAME on-disk store that Hermes uses, so any
agent (DeepSeek Harness, AutoClaw, WorkBuddy, a shell script, a cron job) sees
the same L1 rules, L4 persona, L2 facts, L3 conversation archive, and Obsidian
knowledge base.

This CLI is intentionally dependency-light: it reads L1/L4 as plain Markdown,
searches L3 via SQLite FTS5 (read-only, WAL-safe), keyword-scans L2 (LanceDB,
no embeddings needed), and reads/writes the Obsidian vault as Markdown. It
re-executes itself under the project venv when the current interpreter lacks
LanceDB, so a caller never has to know which Python to use.

Identity
--------
Writes are attributed to an agent. Resolution order:

    1. ``--agent <name>``         explicit, always wins
    2. ``$HGM_AGENT``             set once in the agent's shell profile
    3. auto-detection             from well-known agent env vars
    4. ``external``               the honest default: unknown caller

The detected name lands in the L2 ``agent`` column and in the vault note's
``source`` field, so "who wrote this" is always answerable.

Write channels
--------------
* ``kb-add``   — vault note. Governance: rejects secrets; confidence < 0.7
                 lands in ``inbox/`` for review. **Not** gated by
                 ``external_write_verdict``: a vault note is a document, not a
                 durable fact, and it is reviewed by a human rather than
                 admitted by a gate.
* ``remember`` — L2 durable fact. Governance: ``external_write_verdict``, the
                 gate for writes arriving from an *external* agent — which is
                 what every caller of this CLI is. It is stricter than a bare
                 word-list, because an external writer has no dialogue context
                 to vouch for it, and deliberately **looser** than the
                 bridge/L1 gate, because a wrong L2 row costs one misleading
                 hint while a wrong L1 row becomes a standing rule injected
                 every turn. It is therefore not "the same rules Hermes applies
                 to its own extractions".

Refusals are one shape across every command: ``{"ok": false, "error":
"refused: <cause>", "reason": "<cause>"}`` — both keys, so a caller does not
have to know which command it invoked in order to read its own failure.

L1 (``MEMORY.md`` / ``USER.md``) is **human-authored** and is deliberately NOT
writable from this CLI. No agent may rewrite the standing rules.

Usage (any interpreter will do — the CLI fixes itself):

    python memory_cli.py recall <query> [--top-k N]
    python memory_cli.py kb-search <query> [--top-k N] [--section S]
    python memory_cli.py kb-get <title>
    python memory_cli.py kb-add <title> <body> [--section S] [--tags a b] [--concepts x y] [--confidence 0.8]
    python memory_cli.py remember <fact> [--agent NAME] [--category C] [--dry-run]
    python memory_cli.py l1                          # MEMORY.md + USER.md + persona.md
    python memory_cli.py agents                      # who has written what
    python memory_cli.py health

Paths/config come from HERMES_HOME/governed_memory.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_HERMES_HOME = r"D:/repos\hermes-home"
VAULT_SUBDIRS = ["inbox", "notes", "projects", "areas", "resources", "archive"]

#: Root-level vault files that are navigation scaffolds, not notes. Only
#: consulted when scanning the whole vault (see :func:`iter_notes`).
_VAULT_SCAFFOLD = {"index", "readme", "home"}

#: Agent name written when the caller cannot be identified. Never guess a
#: specific agent's name — mis-attribution is worse than an honest "unknown".
DEFAULT_AGENT = "external"

#: Env var read for the agent identity when ``--agent`` is absent.
AGENT_ENV_VAR = "HGM_AGENT"

#: Env var naming an interpreter to use for the self re-exec.
PYTHON_ENV_VAR = "HGM_PYTHON"

_BOOTSTRAP_FLAG = "HGM_BOOTSTRAPPED"

#: Set on the re-exec'd process so ``runtime`` can still say which interpreter
#: the caller actually invoked. Without it a bootstrapped run looks identical to
#: a native one and "why is my python reporting a venv path" has no answer.
_ORIGINAL_PYTHON_FLAG = "HGM_ORIGINAL_PYTHON"

#: Modules that make the full store reachable. Without them L2 is invisible and
#: the CLI would silently under-report — which is exactly the bug this
#: bootstrap exists to prevent.
_REQUIRED_MODULES = ("lancedb", "pyarrow")

#: Well-known markers each agent sets in its own environment. Only *specific*
#: markers count: inferring from something generic like HERMES_HOME would label
#: every caller "hermes", which is the same class of bug as hardcoding "dsh".
#: NOTE (2026-09-16, measured): these names must be *observed*, not guessed. A
#: first pass used plausible-sounding names for WorkBuddy
#: (``WORKBUDDY_SESSION`` / ``WORKBUDDY_HOME`` / ``CODEBUDDY_SESSION``) — none of
#: which exist. Detection therefore silently fell through to ``external`` and
#: every WorkBuddy write would have been misattributed. The WorkBuddy entries
#: below were read off a live environment with ``env``; the rest still need the
#: same treatment before they can be trusted.
_AGENT_ENV_MARKERS = (
    ("dsh", ("DSH_WORKSPACE", "DSH_PYTHON", "DSH_SESSION")),
    ("autoclaw", ("AUTOCLAW_HOME", "AUTOCLAW_AGENT", "AUTOCLAW_WORKSPACE")),
    ("workbuddy", ("WORKBUDDY_APP_NAME", "WORKBUDDY_CONFIG_DIR",
                   "WORKBUDDY_USER_DATA_DIR", "CODEBUDDY_SESSION_ID")),
    ("mimocode", ("MIMOCODE_HOME", "MIMO_WORKSPACE")),
    ("opencode", ("OPENCODE_HOME", "OPENCODE_SESSION")),
)

_AGENT_NAME_RE = re.compile(r"[^A-Za-z0-9_.\-]")
_AGENT_NAME_MAX = 32

#: Project identifiers are *labels*, not security boundaries, so unlike agent
#: names they keep CJK — this machine's checkouts live under
#: ``D:/repos\项目\...`` and stripping the ideographs would collapse
#: several distinct projects onto the same empty string. Only path separators,
#: whitespace and shell-hostile characters are removed.
_PROJECT_NAME_RE = re.compile(r"[^0-9A-Za-z_.\-\u4e00-\u9fff]")
_PROJECT_NAME_MAX = 64

#: Entries that mark a project root when walking up from a cwd. ``.git`` covers
#: real repositories; the others cover working trees that are not repos yet.
_PROJECT_MARKERS = (".git", ".workbuddy", "pyproject.toml", "package.json")

#: Names that may legitimately appear as a note's writer. Used only to
#: interpret the legacy ``source`` field — see :func:`note_agent`.
_KNOWN_AGENTS = frozenset({
    "hermes", "dsh", "autoclaw", "workbuddy", "mimocode", "opencode", "external",
})

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{15,}"),
    re.compile(r"(?i)(api[_-]?key|app[_-]?secret|token|password|passwd|pwd|secret)"
               r"\s*[:=]\s*([^\s,;，；]+)"),
]
_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')



# -- runtime bootstrap ------------------------------------------------------
def _missing_modules() -> list:
    """Required modules that this interpreter cannot import."""
    import importlib.util
    return [m for m in _REQUIRED_MODULES if importlib.util.find_spec(m) is None]


def _candidate_interpreters() -> list:
    """Interpreters to try, most specific first."""
    out = []
    env_py = os.environ.get(PYTHON_ENV_VAR)
    if env_py:
        out.append(env_py)
    repo = Path(__file__).resolve().parent
    for rel in ((".venv", "Scripts", "python.exe"), (".venv", "Scripts", "python"),
                (".venv", "bin", "python"), ("venv", "Scripts", "python.exe"),
                ("venv", "bin", "python")):
        out.append(str(repo.joinpath(*rel)))
    return out


def bootstrap_interpreter() -> None:
    """Re-exec under an interpreter that can see the whole store.

    Why this exists (measured 2026-09-16): the DSH plugin shells out with
    ``PYTHON = 'python'``, and on this machine the bare ``python`` on PATH is a
    bare runtime with no LanceDB. ``recall_l2`` swallowed the ImportError and
    returned ``[]``, so DSH silently saw an empty fact layer forever — the
    failure was invisible from both ends. Rather than force every agent to know
    which interpreter to use, the CLI relocates itself.

    The re-exec is marked with an env flag so a broken venv cannot cause an
    infinite loop; when no candidate works the CLI proceeds and reports the
    missing layer explicitly instead of pretending it is empty.
    """
    if os.environ.get(_BOOTSTRAP_FLAG) == "1":
        return
    if not _missing_modules():
        return

    try:
        current = Path(sys.executable).resolve()
    except OSError:
        current = None

    for cand in _candidate_interpreters():
        try:
            cp = Path(cand).resolve()
        except OSError:
            continue
        if not cp.is_file() or cp == current:
            continue
        env = dict(os.environ)
        env[_BOOTSTRAP_FLAG] = "1"
        env.setdefault(_ORIGINAL_PYTHON_FLAG, sys.executable)
        try:
            proc = subprocess.run(
                [str(cp), str(Path(__file__).resolve()), *sys.argv[1:]],
                env=env,
            )
        except OSError:
            continue
        sys.exit(proc.returncode)


def runtime_report() -> dict:
    """Describe whether this interpreter can reach every layer.

    Reported *after* any bootstrap, so the answer means "can this environment
    use the full store" rather than "what was missing a moment ago". The
    pre-bootstrap interpreter is kept as ``requested_interpreter`` so the
    relocation stays visible instead of looking like a mystery venv path.
    """
    missing = _missing_modules()
    out = {
        "interpreter": sys.executable,
        "missing_modules": missing,
        "full_store_visible": not missing,
    }
    original = os.environ.get(_ORIGINAL_PYTHON_FLAG, "")
    if original:
        out["requested_interpreter"] = original
        out["note"] = ("auto-relocated: the invoked interpreter has no LanceDB, "
                       "so the CLI re-ran itself under the project venv")
    return out


# -- agent identity ---------------------------------------------------------
def normalize_agent(raw) -> str:
    """Reduce an arbitrary label to a safe, stable agent name."""
    name = _AGENT_NAME_RE.sub("", str(raw or "").strip().lower())
    return name[:_AGENT_NAME_MAX]


def _detect_agent() -> str:
    """Best-effort identification from well-known per-agent env markers."""
    env = os.environ
    for name, markers in _AGENT_ENV_MARKERS:
        if any(env.get(m) for m in markers):
            return name
    return DEFAULT_AGENT


def resolve_agent(explicit: str = "") -> str:
    """Resolve the writing agent: explicit > ``$HGM_AGENT`` > detection.

    Falls back to :data:`DEFAULT_AGENT` rather than guessing a real agent's
    name: a wrong attribution quietly corrupts provenance for everyone.
    """
    for cand in (explicit, os.environ.get(AGENT_ENV_VAR, "")):
        name = normalize_agent(cand)
        if name:
            return name
    return _detect_agent()


def note_agent(meta: dict) -> str:
    """Resolve who wrote a vault note, or ``""`` when that cannot be known.

    ``agent`` is the explicit field written by current versions. ``source``
    PREDATES it and is not a writer field: the existing vault uses it for
    provenance type — ``l3``, ``wiki-backup``, a bare session id. Trusting it
    blindly would invent agents that never existed, so it is honoured only when
    it names a known agent (which is how the one hand-written ``source: dsh``
    note is recognised). Everything else reports as unattributed, honestly.
    """
    explicit = normalize_agent(meta.get("agent"))
    if explicit:
        return explicit
    legacy = normalize_agent(meta.get("source"))
    return legacy if legacy in _KNOWN_AGENTS else ""


# -- plugin access (lazy, cached) -------------------------------------------
_PLUGIN_CACHE: dict = {}


def plugin_module(name: str):
    """Import ``plugin.memory_governed.<name>`` lazily.

    Lazy because the read-only commands (``l1``, ``kb-*``) must keep working on
    an interpreter that has nothing installed — that is the whole point of the
    dependency-light design. Cached because a CLI run touches several modules.
    """
    if name in _PLUGIN_CACHE:
        return _PLUGIN_CACHE[name]
    repo = Path(__file__).resolve().parent
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import importlib
    mod = importlib.import_module(f"plugin.memory_governed.{name}")
    _PLUGIN_CACHE[name] = mod
    return mod


# -- project attribution ----------------------------------------------------
def normalize_project(raw) -> str:
    """Fold a directory name into a stable project label.

    Unlike :func:`normalize_agent` this preserves CJK and does **not**
    lowercase: the label is shown back to the user, so mangling its case makes
    the report harder to read for no benefit.
    """
    return _PROJECT_NAME_RE.sub("", str(raw or "").strip())[:_PROJECT_NAME_MAX]


def infer_project(cwd: str = "") -> str:
    """Best-effort project label for a working directory.

    Walks up from ``cwd`` to the first directory carrying a project marker, so
    a hook invoked from ``<repo>/scripts`` still reports ``<repo>`` rather than
    ``scripts``. Returns ``""`` (global) when no marker is found.

    The leaf directory name used to be the fallback, and it was actively wrong:
    standing in ``~/.workbuddy`` produced the project ``.workbuddy`` — the very
    thing the home-directory guard above exists to prevent — and standing in
    ``~`` produced a project named after the user. A fact written from anywhere
    under ``~`` was therefore filed under a pseudo-project and became invisible
    the moment anyone scoped a query to the real one. No project marker means
    "no project", and that has to be the answer even though it is less
    decorative than a name.

    Never raises. A caller is only decorating its output; a malformed cwd must
    not be allowed to take down the call.
    """
    try:
        start = Path(cwd).expanduser() if cwd else Path(os.getcwd())
        if start.is_file():
            start = start.parent
        if not start.exists():
            # A working directory that does not exist is not a project. Falling
            # back to its leaf name here would attribute facts to a place that
            # is not there, so answer honestly with nothing instead.
            return ""
        # Stop at the home directory rather than walking past it. ``~/.workbuddy``
        # is WorkBuddy's *user-level* config, not a project marker — letting it
        # count would attribute every path under ``~`` (including every temp
        # directory on the machine) to a project named after the user.
        home = Path.home()
        for cand in (start, *start.parents):
            if cand == home:
                break
            try:
                if any((cand / m).exists() for m in _PROJECT_MARKERS):
                    return normalize_project(cand.name)
            except OSError:
                continue
        # No marker anywhere up to the home directory: this is not a project.
        # Returning the leaf name here would invent one — see the docstring.
        return ""
    except Exception:  # noqa: BLE001 — decoration must never be fatal
        return ""


# -- config -----------------------------------------------------------------
def hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or DEFAULT_HERMES_HOME


def load_config() -> dict:
    cfg_path = Path(hermes_home()) / "governed_memory.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def wiki_dir(config: dict) -> Path:
    """Resolve the vault root: configured value first, then HERMES_HOME.

    The fallback used to be a hardcoded ``D:\\Sync\\...\\wiki`` — one developer's
    machine. That made ``HERMES_HOME`` a lie: set it to a sandbox and ``health``
    still reported the real vault's notes, because every read path resolved
    through here. Any write-path test or rehearsal therefore touched the real
    vault no matter what HOME it pointed at, and the only way to get isolation
    was to hand-write a ``governed_memory.json``.

    A default derived from HERMES_HOME is the honest answer: an unconfigured
    home owns its own (probably empty) vault instead of borrowing another
    machine's. Deployments that configure ``wiki_dir`` — including the real one,
    ``C:/Users/example/wiki`` — are unaffected; the explicit value still wins.
    """
    configured = str(config.get("wiki_dir") or "").strip()
    if configured:
        return Path(configured)
    return Path(hermes_home()) / "wiki"


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace").strip()


# -- L1 / L4 ----------------------------------------------------------------
def _refusal(reason: str, **extra) -> dict:
    """Build one refusal payload — both keys, always, with the same cause.

    The CLI used to answer refusals in two shapes: ``{"ok": false, "error":
    "refused: ..."}`` from the kb commands and ``{"ok": false, "reason": ...}``
    from ``remember``/``recall``. A caller therefore had to know which command
    it had invoked before it could read its own failure, and any shared handler
    had to try both keys. Both are now always present:

    * ``error`` — ``"refused: <cause>"``, the shape the kb commands already
      returned and that existing callers match on;
    * ``reason`` — the bare cause, the shape ``remember`` already returned.

    A cause that already carries the ``refused: `` prefix is not double-prefixed,
    so ``VaultPathEscape`` messages (which are built with it) pass through
    unchanged.
    """
    cause = str(reason)
    if cause.startswith("refused: "):
        cause = cause[len("refused: "):]
    out = {"ok": False, "error": f"refused: {cause}", "reason": cause}
    out.update(extra)
    return out


def cmd_l1():
    h = hermes_home()
    return {
        "memory_rules_md": read_text(f"{h}/memory/MEMORY.md"),
        "user_profile_md": read_text(f"{h}/memory/USER.md"),
        "persona_md": read_text(f"{h}/memory/persona.md"),
        "note": "Standing human-authored rules/profile (L1/L4). Always honour them.",
    }


# -- L3: SQLite FTS5 (read-only, WAL-safe) ----------------------------------
def open_l3_ro():
    db = Path(hermes_home()) / "memory" / "l3" / "l3.db"
    if not db.exists():
        return None
    uri = "file:" + str(db).replace("\\", "/") + "?mode=ro&immutable=1"
    import sqlite3
    return sqlite3.connect(uri, uri=True)


def search_l3(query: str, top_k: int, errors: list = None) -> list:
    conn = open_l3_ro()
    if conn is None:
        if errors is not None:
            errors.append("l3: archive database not found or unreadable")
        return []
    try:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
        if not tokens:
            return []
        results = []
        # 1) ranked FTS5 (helps English term/prefix matching)
        try:
            fts = " ".join('"%s"' % t for t in tokens)
            rows = conn.execute(
                "select rowid from messages_fts where messages_fts match ? limit ?",
                (fts, max(top_k * 2, 10)),
            ).fetchall()
            seen = set()
            for (rid,) in rows:
                if rid in seen:
                    continue
                seen.add(rid)
                row = conn.execute(
                    "select role, content, timestamp from messages where id=?",
                    (rid,),
                ).fetchone()
                if row:
                    results.append({"layer": "l3", "role": row[0],
                                    "content": (row[1] or "")[:600],
                                    "timestamp": row[2], "score": 1.0})
                if len(results) >= top_k:
                    return results
        except Exception:
            pass
        # 2) LIKE fallback/complement (reliable, esp. CJK)
        if len(results) < top_k:
            where = " OR ".join("content LIKE ?" for t in tokens)
            like = [f"%{t}%" for t in tokens]
            rows = conn.execute(
                f"select role, content, timestamp from messages where {where} "
                "order by id desc limit ?",
                [*like, top_k],
            ).fetchall()
            for role, content, ts in rows:
                results.append({"layer": "l3", "role": role, "content": (content or "")[:600],
                                "timestamp": ts, "score": 0.8})
        return results
    finally:
        try:
            conn.close()
        except Exception:
            pass


# -- L2: LanceDB keyword scan (no embeddings) -------------------------------
#: How many matched query terms count as "fully matched".
#:
#: The defect was a *ceiling*, not only a threshold. The previous score was
#: ``matched / len(terms)``, which cannot exceed ``1/len(terms)``: every
#: two-term query topped out at 0.5, so no threshold above 0.5 could ever admit
#: one. Measured on the real store, applying ``recall.l2_min_score`` (0.76 — a
#: *cosine* calibration) emptied the layer for 15 of 15 natural-language
#: queries. Saturating the denominator encodes "three matched terms is as much
#: evidence as a keyword channel has", which makes the score independent of how
#: long the question happens to be.
_L2_TERM_SATURATION = 3

#: Floor for the CLI's lexical L2 channel, on that coverage scale.
#:
#: NOT ``recall.l2_min_score``. That one is calibrated on cosine similarity
#: (relevant 8/8 at 0.8458~0.9095, irrelevant 8/8 at 0.6755~0.7515); a coverage
#: fraction lives on an entirely different distribution, and reusing the value
#: was the defect. Operators can override this via
#: ``recall.l2_lexical_min_score`` in ``governed_memory.json``.
#:
#: Calibrated 2026-09-17 against the project's real store (21 facts) with the
#: same 8-relevant / 8-irrelevant protocol as ``tmp/calibrate_l2.py``:
#:
#:     relevant   top1 in [0.667, 1.000]   8/8 kept
#:     irrelevant top1 in [0.000, 0.333]   0/8 leaked
#:
#: The usable window is therefore (1/3, 1/2]: one matched term must keep a
#: two-term query (0.5) but must not keep a three-term one (0.333), and 0.4 is
#: a round value inside it with margin on both sides. For queries of three
#: terms or more the score is quantised at 1/3, so nothing can land between
#: 0.333 and 0.667 and the exact choice inside the window is not knife-edge.
DEFAULT_L2_LEXICAL_FLOOR = 0.4

#: Floor for the CLI's **semantic** L2 channel, on the cosine scale.
#:
#: A DIFFERENT quantity from :data:`DEFAULT_L2_LEXICAL_FLOOR` and a different
#: key (``recall.l2_semantic_min_score``). It must never be read from
#: ``recall.l2_min_score``: that value belongs to the in-plugin channel, and
#: borrowing a threshold calibrated for one quantity to cut another is exactly
#: the defect that already emptied the lexical channel once.
#:
#: Calibrated 2026-09-17 against the real store (21 facts; siliconflow
#: ``BAAI/bge-m3``, 1024-d, cosine) with 10 relevant + 8 irrelevant queries:
#:
#:     relevant   top1 in [0.7843, 0.9640]   10/10 kept
#:     irrelevant top1 in [0.6693, 0.7586]    0/8 leaked
#:
#: The usable window is (0.7586, 0.7843]; 0.77 sits inside it with margin on
#: both sides — 0.011 below the relevant minimum, 0.014 above the irrelevant
#: maximum. The plugin's own cosine floor happens to be 0.76, also inside the
#: window: the two measure the same thing, but they are deliberately separate
#: keys so retuning one cannot silently move the other.
DEFAULT_L2_SEMANTIC_FLOOR = 0.77

#: Candidates fetched per requested hit when a project scope cannot be pushed
#: into LanceDB as a prefilter. Without it another project's rows can fill the
#: whole shortlist and the scoped rows never surface at all.
_L2_SEMANTIC_OVERFETCH = 5

#: Stated on every CLI L2 answer. The two channels both land on [0, 1] but
#: measure different things — one measures meaning, the other measures wording —
#: and each is cut by its own floor. Leaving that unsaid would let a caller
#: treat a lexical 0.8 as a semantic 0.8, which is the same class of error as
#: the hardcoded 0.9 this replaced.
_L2_RANKING_NOTE = (
    "L2 reached through the CLI is keyword-scored (no embedding model is used "
    "on this path). score = matched query terms over min(terms, 3), taking the "
    "better of a phrase match and a word match, on [0,1]. That is a lexical "
    "measure, not a cosine score, and it is cut by its own floor "
    "(recall.l2_lexical_min_score) rather than recall.l2_min_score — the "
    "channels are not interchangeable. score_basis and floor_source say which "
    "one produced these hits.")

#: The counterpart note for the vector channel. Kept as a separate string
#: rather than merged into one paragraph: a caller that reads the note has to be
#: able to tell which channel actually produced the hits it is looking at.
_L2_SEMANTIC_NOTE = (
    "L2 reached through the CLI is vector-scored: the query is embedded by the "
    "same backend the plugin uses and matched by LanceDB cosine distance, "
    "mapped to [0,1] as 1 - d/2. That is a semantic measure, not the lexical "
    "coverage fraction, and it is cut by its own floor "
    "(recall.l2_semantic_min_score) — never by recall.l2_min_score, and never "
    "by the lexical floor. The 'semantic' block in the ranking says whether "
    "this channel was available; a lexical answer is a degradation, not a "
    "choice.")

#: Terms that carry no retrieval signal in either language. Deliberately short
#: and generic: a longer list tuned against the queries it is measured on would
#: be fitting the calibration set rather than the language.
_L2_STOP_TERMS = frozenset("""
的 了 是 在 我 你 他 她 它 们 有 和 与 就 不 都 也 很 到 说 要 去 会 着 好 这 那 哪 吗 呢 吧 啊 呀 么 之 其
什么 怎么 怎样 如何 为什么 哪个 哪些 可以 能 能够 需要 应该 想 让 帮 请 一下 一个 一些 这个 那个
我们 你们 他们 现在 已经 还是 或者 并且 如果 就是 还有 以及 等等 把 被 给 对 从 向 为 为了 于 以 及 或 但 而 则
所 得 地 过 来 时 时候 里 外 中 后 前 多 少 大 小 太 真 好像 可能 大概 一直 才 只 更 最 非常 比较 有点 有些
哪里 为啥 多少 能不能 还要 同时 别的 地方 出来 起来 下去 知道 觉得 感觉 帮忙 麻烦
the a an of to in for on is are was were be been with and or not it its this that at by from as
""".split())

#: ASCII word or a run of CJK. Keeping the two apart matters: the legacy
#: pattern ``[\w\u4e00-\u9fff]+`` glued ``本地IP通过`` into one unusable term.
_L2_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")
_L2_ASCII_RE = re.compile(r"[A-Za-z0-9_]+")
#: The legacy tokenizer, kept only to give phrase matching its units.
_L2_PHRASE_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def l2_query_terms(query: str) -> list:
    """Distinct informative terms of a query: ASCII words + CJK bigrams.

    Bigrams rather than single characters, because one CJK character is far too
    common to be evidence — measured, unigram matching scored
    ``Excel 怎么做数据透视表`` as high as the genuinely relevant
    ``SecretStore 跑在哪台机器上``. And bigrams rather than whole runs, because
    a run only ever matches in one word order: ``数据同步`` could never find
    ``同步数据``, which is half of why the channel went empty.
    """
    out = []
    for token in _L2_TOKEN_RE.findall(str(query or "")):
        if _L2_ASCII_RE.fullmatch(token):
            candidates = [token.lower()]
        elif len(token) > 1:
            candidates = [token[i:i + 2] for i in range(len(token) - 1)]
        else:
            candidates = [token]
        for cand in candidates:
            if cand not in _L2_STOP_TERMS and cand not in out:
                out.append(cand)
    return out


def l2_doc_terms(content: str) -> set:
    """The term set of a stored fact — the same tokenizer as the query side."""
    return set(l2_query_terms(content))


def l2_phrase_terms(query: str) -> list:
    """Contiguous query phrases, MATCHED BY SUBSTRING.

    The second matcher of the dis_max in :func:`l2_lexical_score`. An exact CJK
    phrase is the highest-precision evidence available without a dictionary,
    but as the *only* matcher it is brittle in word order — and that, combined
    with a floor it could never reach, is what emptied the layer.
    """
    out = []
    for token in _L2_PHRASE_RE.findall(str(query or "")):
        token = token.lower()
        if token not in _L2_STOP_TERMS and token not in out:
            out.append(token)
    return out


def _saturated_rate(matched: int, total: int) -> float:
    """``matched`` over a denominator that stops growing at the saturation point."""
    if total <= 0:
        return 0.0
    return min(1.0, matched / max(1, min(total, _L2_TERM_SATURATION)))


def l2_lexical_score(query: str, content: str) -> float:
    """Relevance of ``content`` to ``query`` on [0, 1], from lexical evidence.

    ``dis_max`` over two matchers — exact phrase, and term overlap — because
    neither alone is sufficient:

    * phrase-only cannot match ``数据同步`` to ``同步数据``;
    * term-only scores an unrelated query as high as a relevant one whenever a
      *common* word happens to be rare inside a small store (measured:
      ``Excel 怎么做数据透视表`` tied with ``SecretStore 跑在哪台机器上``,
      because ``数据`` occurs in only 1 of the 21 facts).

    Taking the better of the two keeps the phrase matcher's precision and the
    term matcher's recall. The result is a *lexical* relevance: the same [0,1]
    spirit as the in-plugin cosine score, but not the same quantity, and it is
    thresholded as such.
    """
    low = str(content or "").lower()
    if not low:
        return 0.0
    term_score = 0.0
    terms = l2_query_terms(query)
    if terms:
        doc_terms = l2_doc_terms(str(content))
        term_score = _saturated_rate(sum(1 for t in terms if t in doc_terms),
                                     len(terms))
    phrase_score = 0.0
    phrases = l2_phrase_terms(query)
    if phrases:
        phrase_score = _saturated_rate(sum(1 for p in phrases if p in low),
                                       len(phrases))
    return max(term_score, phrase_score)


def _resolve_named_floor(key: str, explicit=None, default: float = 0.0) -> tuple:
    """Resolve one ``recall.<key>`` threshold as ``(floor, source)``.

    Shared by the lexical and the semantic channel so they can never drift
    apart in *how* a threshold is resolved — only in which key each reads.

    ``source`` names where the number came from, so the answer can distinguish
    "the operator tuned this" from "the built-in calibration" instead of
    implying a gate that was configured when it was not. Order: explicit
    argument, plugin ``recall.<key>``, the same key in ``governed_memory.json``,
    then ``default``.
    """
    def _coerce(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    if explicit is not None:
        value = _coerce(explicit)
        if value is not None:
            return value, "explicit"
    try:
        value = _coerce(getattr(plugin_config().recall, key, None))
        if value is not None:
            return value, "config"
    except Exception:  # noqa: BLE001 — a missing config must not blind recall
        pass
    try:
        value = _coerce((load_config().get("recall") or {}).get(key))
        if value is not None:
            return value, "config-file"
    except Exception:  # noqa: BLE001
        pass
    return default, "default"


def _resolve_l2_floor(explicit=None) -> tuple:
    """Resolve the CLI's **lexical** L2 floor as ``(floor, source)``.

    Deliberately never falls back to ``recall.l2_min_score``: that value is
    calibrated on cosine similarity, and reusing it here is precisely the
    defect this function exists to prevent. See
    :data:`DEFAULT_L2_LEXICAL_FLOOR`.
    """
    return _resolve_named_floor("l2_lexical_min_score", explicit,
                                DEFAULT_L2_LEXICAL_FLOOR)


def _resolve_l2_semantic_floor(explicit=None) -> tuple:
    """Resolve the CLI's **semantic** L2 floor as ``(floor, source)``.

    Its own key, ``recall.l2_semantic_min_score`` — never ``l2_min_score``.
    That one is the *in-plugin* channel's threshold; sharing it would couple
    two channels that are meant to be tunable independently and would hide
    which of them an operator just changed.

    The default happens to sit near the plugin's because both measure bge-m3
    cosine similarity (see :data:`DEFAULT_L2_SEMANTIC_FLOOR`) — the same
    quantity, so similar numbers are expected — but they are read from separate
    keys on purpose, and neither is ever compared against a lexical score.
    """
    return _resolve_named_floor("l2_semantic_min_score", explicit,
                                DEFAULT_L2_SEMANTIC_FLOOR)


def _l2_ranking(ranked: bool, floor: float, source: str,
                filtered: int = 0, truncated: int = 0,
                note: str = None, basis: str = "lexical-dis-max",
                semantic: dict = None) -> dict:
    """Metadata describing *how* the L2 hits were produced.

    Reported separately from the hits because a score without its basis is
    unfalsifiable: a caller cannot tell a filtered list from an unfiltered one,
    nor a lexical 0.8 from a semantic 0.8, unless the answer says so. The
    ``filtered_out`` count is load-bearing — it is what made this defect
    visible in QA (`filtered_out: 3` next to `hits: []`), so it is never
    dropped, even when it is zero.

    ``basis`` names the channel that produced these hits. ``semantic`` describes
    the vector channel whenever it was attempted: whether it was available, why
    not when it was not, and the floor it would have used. It is emitted even
    on a lexical answer, because a lexical answer is then a *degradation* and
    the caller is entitled to know that — otherwise "I searched semantically and
    found nothing" and "I could not embed at all" look identical.
    """
    out = {
        "ranked": ranked,
        "basis": basis,
        "floor": floor,
        "floor_source": source,
        "filtered_out": filtered,
        "truncated": truncated,
        "note": ((_L2_RANKING_NOTE if basis == "lexical-dis-max"
                  else _L2_SEMANTIC_NOTE) if note is None else note),
    }
    if semantic is not None:
        out["semantic"] = semantic
    return out


def _l2_semantic_hits(query: str, top_k: int, project: str = "",
                      errors: list = None) -> dict:
    """Vector L2 hits for ``query``: embed it, then search by cosine distance.

    Reuses the plugin rather than reimplementing any part of it. The embedding
    backend (:class:`EmbeddingService`), the distance→score mapping
    (:func:`_recall.distance_to_score`) and the project predicate
    (:func:`_recall.l2_project_where` / ``l2_project_allows``) are the very
    objects the in-plugin channel uses. A second copy of any of them is exactly
    how two entry points end up giving two answers to one question.

    Returns a verdict dict and never raises:
    ``{"available", "hits", "floor", "floor_source", "filtered_out",
    "truncated", "reason", "backend"}``.

    ``available`` is False when the query could not be embedded or the store
    could not be searched; ``reason`` then says why. The caller is expected to
    fall back to the lexical channel **and say so** — silently returning an
    empty L2 would read to an agent as "nothing has been remembered", which is
    the worst answer this module can give.
    """
    floor, floor_source = _resolve_l2_semantic_floor(None)
    out = {"available": False, "hits": [], "floor": floor,
           "floor_source": floor_source, "filtered_out": 0, "truncated": 0,
           "reason": "", "backend": ""}

    def _unavailable(reason: str) -> dict:
        out["reason"] = reason
        if errors is not None:
            errors.append(f"l2 semantic: {reason}")
        return out

    try:
        cfg = plugin_config()
        embedding = plugin_module("_embedding").EmbeddingService.get(cfg)
    except Exception as e:  # noqa: BLE001 — a bare interpreter must still answer
        return _unavailable("semantic unavailable: cannot load the embedding "
                            f"backend ({type(e).__name__}: {e})")
    if not embedding.available:
        return _unavailable(
            "semantic unavailable: no embedding backend able to embed this "
            f"query ({embedding.last_error or 'backend reported unavailable'})")
    try:
        vector = embedding.embed_one(query)
    except Exception as e:  # noqa: BLE001
        return _unavailable("semantic unavailable: embedding failed "
                            f"({type(e).__name__}: {e})")
    if vector is None:
        return _unavailable(
            "semantic unavailable: the embedding backend returned no vector "
            f"({embedding.last_error or 'no detail given'})")

    try:
        import lancedb

        recall_mod = plugin_module("_recall")
        # Checked before connecting: lancedb.connect() would otherwise *create*
        # the directory, and a read must not have that side effect.
        if not Path(str(cfg.l2_db_path)).exists():
            return _unavailable("semantic unavailable: l2 directory missing")
        db = lancedb.connect(str(cfg.l2_db_path))
        names = [t.name if hasattr(t, "name") else str(t)
                 for t in db.list_tables().tables]
        if "memories" not in names:
            return _unavailable("semantic unavailable: no 'memories' table")
        table = db.open_table("memories")
        if not any(f.name == "vector" for f in table.schema):
            return _unavailable("semantic unavailable: 'memories' has no "
                                "vector column, so no row can be matched by "
                                "meaning")
        limit = max(int(top_k or 0), 1)
        search = table.search(vector).metric("cosine")
        prefiltered = False
        if project:
            try:
                search = search.where(recall_mod.l2_project_where(project),
                                      prefilter=True)
                prefiltered = True
            except Exception:  # noqa: BLE001 — older LanceDB has no prefilter
                limit = limit * _L2_SEMANTIC_OVERFETCH
        hits, filtered = [], 0
        for r in search.limit(limit).to_list():
            if project and not prefiltered \
                    and not recall_mod.l2_project_allows(project, r.get("project")):
                continue
            score = recall_mod.distance_to_score(r.get("_distance"))
            if score < floor:
                filtered += 1
                continue
            hit = {"layer": "l2", "content": str(r.get("content", ""))[:600],
                   "score": round(score, 4),
                   "score_basis": "semantic-cosine"}
            if r.get("agent"):
                hit["agent"] = r.get("agent")
            if r.get("project"):
                hit["project"] = str(r.get("project"))
            hits.append(hit)
        hits.sort(key=lambda h: h["score"], reverse=True)
        kept = hits[:max(int(top_k or 0), 0)]
        out.update({"available": True, "hits": kept, "filtered_out": filtered,
                    "truncated": len(hits) - len(kept),
                    "backend": getattr(embedding, "backend_name", "") or ""})
        return out
    except Exception as e:  # noqa: BLE001
        return _unavailable("semantic unavailable: vector search failed "
                            f"({type(e).__name__}: {e})")


def recall_l2(query: str, top_k: int, errors: list = None,
              project: str = "", l2_floor: float = None,
              lexical_only: bool = False) -> dict:
    """L2 recall through the CLI, ranked and thresholded.

    Two channels, one answer. The **semantic** channel embeds the query with the
    backend the plugin uses and searches by cosine; it is tried first, and when
    it works it *replaces* the lexical one rather than being merged into it —
    the two scores are different quantities, and ranking them in one list would
    be the same dimension error as thresholding one with the other's floor. The
    **lexical** channel is the fallback, used when ``lexical_only`` is set or
    when no embedding backend can serve the query.

    Returns ``{"hits": [...], "ranking": {...}}``. ``ranking["basis"]`` says
    which channel produced the hits and ``ranking["semantic"]`` says whether the
    vector channel was available and, if not, why — so a lexical answer is never
    mistaken for a semantic one, and a degradation never looks like a choice.


    Two things the previous version got wrong, both of which made the shared
    entry point the weakest implementation of the same query:

    * it emitted a hardcoded ``score`` of 0.9, so no threshold could ever
      reject anything coming through the CLI;
    * it stopped at the first ``top_k`` rows in *write order*, so the cut was
      arbitrary — the facts that happened to be written first won, and a
      caller had no way to know anything had been dropped.

    And a third, introduced while fixing those two and caught by QA: the score
    was a bare coverage fraction cut by ``recall.l2_min_score``, a threshold
    calibrated on *cosine similarity*. Two independent errors — the wrong
    dimension, and a ceiling of ``1/len(terms)`` that no multi-word query could
    clear — left the layer empty for 15 of 15 natural-language queries. See
    :data:`DEFAULT_L2_LEXICAL_FLOOR` and :func:`l2_lexical_score`.

    ``errors`` is an optional sink. L2 used to fail silently — an ImportError
    from a bare interpreter produced the same ``[]`` as a genuine no-match, and
    a caller could not tell "nothing remembered" from "cannot read memory".
    Every failure path now records why.

    ``project`` narrows the result to one project *plus* the global facts that
    belong to no project in particular.

    ``l2_floor`` overrides the configured *lexical* threshold; ``None`` resolves
    it through :func:`_resolve_l2_floor`. It has no effect on the semantic
    channel, which resolves its own floor through
    :func:`_resolve_l2_semantic_floor`.

    ``lexical_only`` skips the vector channel entirely — for A/B comparison, for
    offline use, and for callers that want the wording-level answer.
    """
    floor, floor_source = _resolve_l2_floor(l2_floor)

    # Tried first, and when it works it is the answer. Not merged: see the
    # docstring. When it cannot run, `semantic` keeps the reason and is attached
    # to the lexical ranking below, so the fallback is visible rather than
    # looking like a deliberate keyword search.
    semantic = None
    if not lexical_only:
        semantic = _l2_semantic_hits(query, top_k, project, errors)
        if semantic["available"]:
            return {"hits": semantic["hits"],
                    "ranking": _l2_ranking(
                        True, semantic["floor"], semantic["floor_source"],
                        filtered=semantic["filtered_out"],
                        truncated=semantic["truncated"],
                        basis="semantic-cosine", semantic=semantic)}

    def _fail(msg: str) -> dict:
        if errors is not None:
            errors.append(f"l2: {msg}")
        # "ranked: false" here means exactly that: nothing was ranked because
        # nothing could be read. Saying otherwise would let a caller read an
        # empty L2 as "searched and found nothing relevant".
        return {"hits": [],
                "ranking": _l2_ranking(False, floor, floor_source,
                                       note="L2 could not be read, so no hits "
                                            "were ranked or filtered; see the "
                                            "'degraded' entry for the reason.",
                                       semantic=semantic)}

    try:
        import lancedb
    except ImportError as e:
        return _fail(f"lancedb not importable ({e}); run under the project venv")

    try:
        l2 = Path(hermes_home()) / "memory" / "l2"
        if not l2.exists():
            return _fail(f"l2 directory missing: {l2}")
        db = lancedb.connect(str(l2))
        names = getattr(db, "list_tables", None)
        nlist = names() if callable(names) else db.table_names()
        if not isinstance(nlist, (list, tuple)):
            nlist = getattr(nlist, "tables", []) or [nlist]
        tables = [t.name if hasattr(t, "name") else str(t) for t in nlist]
        if "memories" not in tables:
            return _fail("no 'memories' table")
        table = db.open_table("memories")
        arr = table.to_arrow()
        if "content" not in arr.column_names:
            return _fail("'memories' table has no content column")
        contents = arr["content"].to_pylist()
        agents = (arr["agent"].to_pylist() if "agent" in arr.column_names
                  else [None] * len(contents))
        projects = (arr["project"].to_pylist() if "project" in arr.column_names
                    else [None] * len(contents))
        if not (l2_query_terms(query) or l2_phrase_terms(query)):
            return {"hits": [],
                    "ranking": _l2_ranking(True, floor, floor_source,
                                           semantic=semantic)}
        hits, filtered = [], 0
        for c, ag, pj in zip(contents, agents, projects):
            c = str(c or "")
            if not c:
                continue
            pj = str(pj) if pj else ""
            # A project-scoped query still sees global facts (project IS NULL):
            # "SecretStore runs on the NAS" is true wherever you ask from. What
            # it must not see is *another* project's facts.
            if project and pj and pj != project:
                continue
            score = l2_lexical_score(query, c)
            if score <= 0:
                continue
            if score < floor:
                filtered += 1
                continue
            hit = {"layer": "l2", "content": c[:600],
                   "score": round(score, 4),
                   "score_basis": "lexical-dis-max"}
            if ag:
                hit["agent"] = ag
            if pj:
                hit["project"] = pj
            hits.append(hit)
        # Ranked, then cut — the old code took the first top_k in write order
        # and never said it had dropped anything.
        hits.sort(key=lambda h: h["score"], reverse=True)
        kept = hits[:max(int(top_k or 0), 0)]
        return {"hits": kept,
                "ranking": _l2_ranking(True, floor, floor_source, filtered,
                                       len(hits) - len(kept),
                                       semantic=semantic)}
    except Exception as e:  # noqa: BLE001
        return _fail(str(e)[:200])



# -- L2 governed write ------------------------------------------------------
def plugin_config():
    """Load the same config object the plugin uses (single source of truth)."""
    return plugin_module("_config").load_governed_config(hermes_home())


def _l2_schema(dim: int):
    """The one L2 schema, used when a cold start has to create the table.

    Provenance fields come from ``_sync.l2_provenance_fields()`` — the same
    source the plugin's own create-table path uses — so a table built here is
    column-for-column the table the plugin would have built. Hand-writing this
    list is the incident that already shipped: ``project`` was added to a
    backfill list but not to the schema, so old tables gained the column and
    new ones never had it, and nothing failed until an external agent wrote.
    """
    import pyarrow as pa

    return pa.schema([
        pa.field("content", pa.string()),
        pa.field("category", pa.string()),
        pa.field("source", pa.string()),
        pa.field("timestamp", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), int(dim))),
        *plugin_module("_sync").l2_provenance_fields(),
    ])


#: Table creation is not atomic in LanceDB, so a second process can see the
#: name before the dataset is readable. Five passes at 0.15s and growing is
#: ~2.2s worst case — comfortably inside the winner's create time, and far below
#: the point where waiting longer stops being better than failing.
_L2_OPEN_RETRIES = 5
_L2_OPEN_BACKOFF = 0.15


def open_l2_table(cfg, create_dim: int = 0, created: list = None,
                  errors: list = None):
    """Open the L2 table, backfilling the sharing columns on a legacy table.

    The column list is *not* written here. It comes from
    ``_sync.L2_PROVENANCE_COLUMNS`` — the same constant that builds the
    create-table schema and the plugin's own backfill. This is the exact shape
    of an incident that already shipped: ``project`` was added to a backfill
    list but not to the schema, so old tables gained the column while new ones
    never had it, and every test stayed green because they all ran against a
    table that already had it. One definition, three consumers.

    ``create_dim`` lets a *write* create the table when it does not exist yet,
    which is the only way a brand new HERMES_HOME can ever be written to: the
    directory gets made by ``lancedb.connect`` but the table does not, so the
    first ``remember`` on a fresh store used to fail with ``l2_table_missing``
    and no external agent could bootstrap its own memory.

    The dimension must be passed in rather than read from config: the caller
    has just embedded the text and knows the real width, whereas
    ``config.vector.dim`` is the *local* model default (512) and would not match
    an API backend (1024) — a mismatch that only shows up later, as rows that
    cannot be searched.

    ``create_dim <= 0`` (the default) keeps this read-only: ``cmd_agents`` must
    not conjure a table just because it was asked for a report. ``created`` is
    an optional sink that receives ``True`` when a table had to be built, so
    the caller can say so instead of letting a cold start look routine.
    """
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(cfg.l2_db_path)
    table = None
    last_error = None

    # ``lancedb`` creates a table in more than one step: the directory shows up
    # before ``_versions`` does. So a second process can list "memories", decide
    # the table exists, and then fail to open it — measured 2026-09-17, four
    # agents writing at once on a fresh HERMES_HOME, two of them dying with:
    #
    #     RuntimeError: Table 'memories' exists but could not be loaded
    #     (it may be corrupt or incomplete): ... memories.lance was not found:
    #     Not found: .../memories.lance/_versions
    #
    # They printed a traceback and no JSON at all, which breaks the contract
    # that a caller gets a structured answer it can act on. Both halves are
    # fixed here: wait out the other process instead of losing the race, and
    # report exhaustion through ``errors`` rather than by raising.
    for attempt in range(_L2_OPEN_RETRIES):
        try:
            table_names = _l2_table_names(db)
            if "memories" in table_names:
                table = db.open_table("memories")
            elif create_dim > 0:
                try:
                    table = db.create_table("memories", schema=_l2_schema(create_dim))
                    if created is not None:
                        created.append(True)
                except Exception as exc:      # noqa: BLE001 — retried below
                    # Lost the create race. The winner needs a moment to finish;
                    # the next pass will find a complete table and open it.
                    last_error = exc
                    time.sleep(_L2_OPEN_BACKOFF * (attempt + 1))
                    continue
            else:
                return None                  # read-only call, no table yet: fine
            break
        except Exception as exc:             # noqa: BLE001 — retried below
            last_error = exc
            time.sleep(_L2_OPEN_BACKOFF * (attempt + 1))

    if table is None:
        # Say *why* rather than reporting the generic "missing": a table that
        # exists but cannot be loaded is a different problem with a different
        # fix, and conflating them sends the reader looking in the wrong place.
        if errors is not None and last_error is not None:
            errors.append(f"l2_open_failed after {_L2_OPEN_RETRIES} attempts: "
                          f"{type(last_error).__name__}: {last_error}")
        return None

    # lancedb takes a pa.Field here (new column, null-filled), NOT a
    # {name: pa.array} mapping — the array form raises TypeError, and when that
    # happens inside a best-effort block the migration fails invisibly.
    existing = {f.name for f in table.schema}
    # `role` was missing here while cmd_remember writes role="agent" on every
    # row, so `table.add()` raised "field 'role' does not exist" on any table
    # predating the column — every external agent's `remember` failed at once.
    for col, typ in plugin_module("_sync").L2_PROVENANCE_COLUMNS:
        if col not in existing:
            table.add_columns(pa.field(col, getattr(pa, typ)()))
    return table


def _l2_table_names(db) -> list:
    """Table names for a LanceDB connection, across client versions."""
    lister = getattr(db, "list_tables", None)
    listing = lister() if callable(lister) else db.table_names()
    if not isinstance(listing, (list, tuple)):
        listing = getattr(listing, "tables", []) or [listing]
    return [t.name if hasattr(t, "name") else str(t) for t in listing]


def _l2_existing_contents(table) -> set:
    """All ``content`` values currently in L2.

    Pulls only the content column: materialising the 1024-dim vector column for
    every row just to check for duplicates is needlessly expensive.
    """
    try:
        arr = table.to_arrow()
    except Exception:  # noqa: BLE001 — dedupe is best-effort, never fatal
        return set()
    if "content" not in arr.column_names:
        return set()
    return {c for c in arr["content"].to_pylist() if c}


def cmd_remember(config: dict, text: str, agent: str,
                 category: str = "other", dry_run: bool = False,
                 project: str = None) -> dict:
    """Admit ``text`` into L2 as a durable fact attributed to ``agent``.

    The gate is ``external_write_verdict`` — the one for writes arriving from an
    *external* agent, which is what every caller of this CLI is. It is stricter
    than a bare word-list, because an external writer has no dialogue context to
    vouch for it, and deliberately looser than ``screen_bridge_content``, whose
    candidates are promoted into L1 where a wrong row becomes a standing rule
    injected every turn; a wrong L2 row costs one misleading hint instead.

    It is therefore *not* "the same rules Hermes applies to its own
    extractions", which is what this docstring used to claim — and the two CLI
    write channels are not gated alike either (``kb-add`` is not admitted here
    at all): one uniform external gate for L2, and a human review queue for the
    vault.

    A rejection always names its cause, in ``error`` and in ``reason`` alike;
    nothing is dropped silently.

    ``project`` controls attribution, and the three values are deliberately
    distinct:

    * ``None`` — the argument was not supplied; infer from the current working
      directory. This is the common case, and it exists because an agent
      working inside a checkout should not have to remember to say where it is.
    * ``""`` — explicitly global. Use it for infrastructure knowledge that
      holds wherever you happen to be standing.
    * anything else — that project, folded by :func:`normalize_project`.
    """
    sync = plugin_module("_sync")
    cleaned = " ".join(str(text or "").split())
    admitted, reason = sync.external_write_verdict(cleaned)
    resolved_project = (infer_project() if project is None
                        else normalize_project(project))

    base = {"agent": agent, "category": category, "admitted": admitted,
            "project": resolved_project}
    if not admitted:
        base.update(_refusal(
            reason,
            hint=("rewrite it as a concrete statement (name the host, path, "
                  "version or constraint), or use kb-add to file it as a note")))
        return base

    if dry_run:
        base.update({"ok": True, "dry_run": True,
                     "content": cleaned[:600], "would_write": "l2"})
        return base

    cfg = plugin_config()
    embedding = plugin_module("_embedding").EmbeddingService.get(cfg)
    if not embedding.available:
        # Refusing loudly beats writing a row nothing can ever recall.
        base.update(_refusal(
            "embedding_unavailable",
            detail=embedding.last_error or "embedding backend reported unavailable",
            hint="use kb-add instead — notes do not need an embedding to be found"))
        return base

    vector = embedding.embed_one(cleaned)
    if vector is None:
        base.update(_refusal("embedding_failed", detail=embedding.last_error))
        return base

    # A fresh HERMES_HOME has no 'memories' table yet. Opening read-only here
    # is what made the CLI unable to bootstrap a new store: the very first
    # `remember` failed with l2_table_missing. The dimension comes from the
    # vector we just produced, so a cold-started table matches the backend that
    # will be searched through it.
    cold_created: list = []
    open_errors: list = []
    try:
        table = open_l2_table(cfg, create_dim=len(vector), created=cold_created,
                              errors=open_errors)
    except Exception as exc:  # noqa: BLE001 — the caller gets JSON, not a traceback
        table = None
        open_errors.append(f"{type(exc).__name__}: {exc}")
    if table is None:
        # "could not be opened" and "there is no table" are different problems
        # with different fixes. Collapsing them sends the reader after the wrong
        # cause — and during a create race the table does exist, half-made.
        if open_errors:
            base.update(_refusal("l2_open_failed", detail=str(open_errors[0])[:300]))
        else:
            base.update(_refusal("l2_table_missing",
                                 detail="no 'memories' table in " + str(cfg.l2_db_path)))
        return base
    if cold_created:
        # Said rather than implied: creating the store is an event, and a caller
        # that bootstrapped a home by accident deserves to see that it did.
        base["cold_start"] = True
        # And so is the width it was created at, which is the part that costs
        # later. The vector column's dimension is fixed by this write and
        # cannot be widened in place: a home that cold-starts on the 512-d
        # local model and is afterwards pointed at a 1024-d API backend starts
        # failing every write with a dimension mismatch, whose cause is a
        # config change and whose symptom is weeks later and somewhere else.
        # One sentence, at the only moment the width was still a choice —
        # nothing here refuses or rewrites anything.
        base["vector_dim_note"] = (
            f"L2 table created at {len(vector)} dims from backend "
            f"'{embedding.backend_name}'. The width is fixed at creation: a "
            f"later backend that embeds at a different width needs this table "
            f"rebuilt, or writes fail with a dimension mismatch."
        )

    if cleaned in _l2_existing_contents(table):
        base.update({"ok": True, "duplicate": True, "content": cleaned[:600]})
        return base

    dim = len(vector)
    row = {
        "content": cleaned,
        "category": category or "other",
        "source": "external-write",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "vector": vector,
        "source_rowid": None,
        "role": "agent",
        "agent": agent,
        # NULL rather than "" when global. LanceDB keeps the two distinct, and
        # "belongs to no project" is a real fact rather than a missing string —
        # recall_l2 relies on the difference to decide what a project-scoped
        # query is allowed to see.
        "project": resolved_project or None,
    }
    try:
        table.add([row])
    except Exception as e:  # noqa: BLE001 — surface the failure, never mask it
        base.update(_refusal("write_failed", detail=str(e)[:300],
                             vector_dim=dim))
        return base

    base.update({"ok": True, "content": cleaned[:600], "vector_dim": dim,
                 "embedding_backend": embedding.backend_name})

    # Refresh the offline snapshot while LanceDB is already open — see
    # :func:`cmd_snapshot`. Doing it here rather than at session start turns a
    # second ~2s process+import into a table read, and it makes the snapshot
    # track writes through the only entry point agents are supposed to use.
    #
    # A failure is *reported*, not swallowed: the write itself succeeded, and a
    # caller whose next session silently matched against a stale file deserves
    # to have been told at the moment it went stale.
    snap = cmd_snapshot(config)
    if not snap.get("ok"):
        base["snapshot_error"] = snap.get("error")
    return base


#: Bumped when the snapshot's shape changes, so a stale file written by an older
#: CLI is recognisable instead of being parsed as if it were current.
L2_SNAPSHOT_SCHEMA = 1

#: Ceiling on how many facts a snapshot carries. A snapshot exists so an agent
#: can ask "does the store already say something about this?" **without**
#: spawning an interpreter, importing LanceDB or calling an embedding backend;
#: past this size the file stops being cheap to read in-process. Measured
#: 2026-09-17: the real store held 21 facts / 1015 characters of content, a
#: ~3.5KB file. The limit is not a guess about today's data.
L2_SNAPSHOT_MAX_FACTS = 2000


def snapshot_path() -> str:
    """Where the offline snapshot lives. Beside the store it mirrors."""
    return os.path.join(hermes_home(), "memory", "l2_snapshot.json")


def cmd_snapshot(config: dict, out_path: str = "", max_facts: int = 0) -> dict:
    """Write every L2 fact to a JSON snapshot and report what was written.

    Why this exists
    ---------------
    Measured 2026-09-17, on the real store, per ``recall`` invocation:

        sh + python start .............. 0.71s
        recall --lexical-only .......... 2.16s
        recall (semantic) .............. 2.83s

    The bulk is interpreter start plus LanceDB load and a full table scan; the
    embedding round trip is only ~0.67s of it. That is acceptable once per
    session and unacceptable once per prompt — so an agent had no affordable way
    to ask the store a question *before* answering, which is exactly when the
    answer would have been useful.

    The store is small enough to answer from a file: 21 facts, 1015 characters
    of content, ~3.5KB of JSON. This command writes that file; the caller then
    matches in-process with no spawn, no LanceDB and no network.

    ``truncated`` is reported rather than implied — the same rule the recall
    path follows for withheld rows. A snapshot that silently dropped facts would
    make the offline matcher confidently blind, and a confident miss is the
    failure this store exists to eliminate.

    Reads only. ``create_dim`` keeps :func:`open_l2_table` in its read-only
    default, so asking for a snapshot can never conjure a table.
    """
    limit = int(max_facts) if max_facts else L2_SNAPSHOT_MAX_FACTS
    path = out_path or snapshot_path()

    # ``open_l2_table`` takes the plugin's config *object* (it needs
    # ``l2_db_path``), not the CLI's plain dict — ``cmd_agents`` resolves it the
    # same way. Passing the dict raises AttributeError at connect time.
    table = open_l2_table(plugin_config(), create_dim=0)
    if table is None:
        return {"ok": False, "error": "l2 table missing", "path": path,
                "count": 0, "facts": []}

    try:
        rows = table.to_arrow().to_pylist()
    except Exception as exc:  # noqa: BLE001 — reported, never swallowed
        return {"ok": False, "error": f"cannot read L2: {exc}", "path": path,
                "count": 0, "facts": []}

    wanted = ("content", "category", "agent", "project", "timestamp", "source", "role")
    facts = []
    for row in rows:
        content = str(row.get("content") or "").strip()
        if not content:
            continue        # a fact with no text is not a fact
        facts.append({k: row.get(k) for k in wanted if k in row})

    # Newest first, so a truncated snapshot keeps the most recent facts rather
    # than an arbitrary write-order prefix.
    facts.sort(key=lambda f: f.get("timestamp") or 0, reverse=True)
    truncated = len(facts) > limit
    if truncated:
        facts = facts[:limit]

    from datetime import datetime, timezone  # noqa: PLC0415 — one caller

    payload = {
        "schema": L2_SNAPSHOT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "home": hermes_home(),
        "count": len(facts),
        "truncated": truncated,
        "max_facts": limit,
        "facts": facts,
    }

    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as exc:
        return {"ok": False, "error": f"cannot write snapshot: {exc}",
                "path": path, "count": 0, "facts": []}

    return {
        "ok": True,
        "path": path,
        "count": len(facts),
        "truncated": truncated,
        "bytes": os.path.getsize(path),
        "generated_at": payload["generated_at"],
    }


def cmd_agents(config: dict) -> dict:
    """Report which agents have written what — the provenance view.

    Answers the question sharing makes urgent and that a single-agent store
    never had to: *who put this here?* Null agents are reported separately
    rather than bucketed into a fake name.
    """
    out = {"agents": {}, "unattributed": 0, "runtime": runtime_report()}

    # L2 facts
    try:
        cfg = plugin_config()
        table = open_l2_table(cfg)
        if table is not None:
            arr = table.to_arrow()
            if "agent" in arr.column_names:
                for a in arr["agent"].to_pylist():
                    a = normalize_agent(a)
                    if not a:
                        out["unattributed"] += 1
                        continue
                    out["agents"].setdefault(a, {"l2_facts": 0, "kb_notes": 0})
                    out["agents"][a]["l2_facts"] += 1
    except Exception as e:  # noqa: BLE001 — provenance report must not crash
        out["l2_error"] = str(e)[:200]

    # vault notes
    try:
        for p in iter_notes(config):
            meta, _ = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
            a = note_agent(meta)
            if not a:
                out["unattributed"] += 1
                continue
            out["agents"].setdefault(a, {"l2_facts": 0, "kb_notes": 0})
            out["agents"][a]["kb_notes"] += 1
    except Exception as e:  # noqa: BLE001
        out["kb_error"] = str(e)[:200]

    out["total_agents"] = len(out["agents"])
    return out


# -- vault path safety -----------------------------------------------------
class VaultPathEscape(ValueError):
    """A requested vault path would land outside the vault.

    Raised rather than silently rewritten. A caller that asks for
    ``--section ../memory`` must be told it was refused, not handed a note
    filed somewhere else: "my write succeeded" turning into "my write
    vanished" is the worst outcome this module can produce, and it is exactly
    what normalising the path quietly would cause.
    """


def _is_link(path: Path) -> bool:
    """True for a symlink **or** a Windows junction.

    Junctions are the easy one to miss: ``os.path.islink`` reports ``False``
    for them because they are a different reparse-tag, yet they redirect just
    as effectively. On this machine both ``hermes-home`` and ``wiki`` are
    junctions, so a check that only saw symlinks would step straight over the
    construct it exists to catch.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        return False
    isjunction = getattr(os.path, "isjunction", None)
    if callable(isjunction):
        try:
            return bool(isjunction(str(path)))
        except OSError:
            return False
    return False


def _is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or sits below it."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _refuse_escaping_links(root: Path, root_real: Path, candidate: Path) -> None:
    """Reject a candidate that leaves the vault through a link.

    Checked *before* the resolved comparison on purpose. ``resolve()`` follows
    links, so by the time the resolved path looks wrong all that is left to
    report is a location the caller never typed; naming the link says what
    actually happened. A link that stays inside the vault is allowed — it is
    the direction, not the construct, that matters.
    """
    try:
        rel = candidate.relative_to(root)
    except ValueError:  # pragma: no cover — guarded by the caller
        return
    cur = root
    for part in rel.parts:
        if part in ("", ".", ".."):
            continue
        cur = cur / part
        if not _is_link(cur):
            continue
        target = Path(os.path.realpath(str(cur)))
        if not _is_within(target, root_real):
            raise VaultPathEscape(
                f"refused: '{cur}' is a link to '{target}', outside the vault "
                f"'{root_real}'")


def safe_vault_path(vault, *parts: str) -> Path:
    """Join ``parts`` under ``vault``, refusing anything that leaves it.

    Three checks, in this order — each one catches something the next would get
    wrong:

    1. **lexical containment** on the *normalised* path. Both ``../`` and an
       absolute ``--section`` collapse here, before a filesystem call can be
       misled by them.
    2. **link containment** for every component below the vault (see
       :func:`_refuse_escaping_links`). A symlink or junction inside the vault
       pointing outside it is an escape even though the path text looks
       innocent.
    3. **resolved containment** as the backstop: ``realpath`` of both sides, so
       a link, a stray ``..`` or a case difference cannot slip past 1 and 2.

    Args:
        vault: Vault root. May itself be a junction — both sides of every
            comparison are resolved through the same rule, so that stays
            inside.
        *parts: Path components to append (``section``, then the filename).

    Returns:
        The normalised absolute path, guaranteed to be inside the vault.

    Raises:
        VaultPathEscape: With an actionable message. Never rewrites the path —
            a refused write is recoverable, a silently relocated one is not.
    """
    root = Path(os.path.abspath(str(vault)))
    candidate = Path(os.path.normpath(str(root.joinpath(*(str(p) for p in parts)))))
    if not _is_within(candidate, root):
        raise VaultPathEscape(
            f"refused: '{candidate}' is outside the vault '{root}'")
    root_real = Path(os.path.realpath(str(root)))
    _refuse_escaping_links(root, root_real, candidate)
    resolved = Path(os.path.realpath(str(candidate)))
    if not _is_within(resolved, root_real):
        raise VaultPathEscape(
            f"refused: '{candidate}' resolves to '{resolved}', which is outside "
            f"the vault '{root_real}'")
    return candidate


#: Win32 device names. ``CON.md`` is rejected by the filesystem even though it
#: looks like an ordinary filename, so it has to be rejected here first —
#: otherwise the caller gets an OSError traceback instead of a verdict, and on
#: the read side ``path.exists()`` can resolve to the console device itself.
_WIN32_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul", "clock$"}
    | {f"com{i}" for i in range(1, 10)}
    | {f"lpt{i}" for i in range(1, 10)})

#: NTFS caps a single path component at 255 UTF-16 code units. Rejected here so
#: both the read and the write path refuse an over-long section with a verdict:
#: unhandled, the write path died with an ``OSError`` and the read path returned
#: an empty result — the same input producing two different symptoms, one of
#: them silent.
_MAX_SECTION_PART = 255


def split_vault_section(section: str) -> list:
    """Split a ``--section`` value into validated path components.

    The **single** validator for a section name, called by both paths —
    ``kb-add`` (write) and ``iter_notes`` (read). Keeping one function is not
    tidiness: when the read path validated less than the write path, every
    malformed section the writer refused with a verdict was accepted by the
    reader and returned as an empty result, so "your section was rejected" and
    "no notes matched" looked identical to the caller.

    Containment (:func:`safe_vault_path`) stops a section *leaving* the vault;
    this stops it being a name the filesystem cannot accept, or one the vault
    walk will not read back. That is a different failure with a different
    symptom, and it was the one still escaping: on NTFS ``notes:evil``
    addresses an alternate data stream of ``notes`` — inside the vault
    lexically, so containment passes it — and ``mkdir`` then raised an uncaught
    ``NotADirectoryError``. No JSON, a bare traceback, and a caller that cannot
    tell "refused" from "the tool is broken".

    Raises:
        VaultPathEscape: on a ``:`` (ADS) or any other Win32-forbidden
            character, a ``..`` component, a whitespace-only component, a
            hidden (dot) directory — which ``iter_notes`` would skip, making the
            write unreadable — a component longer than the filesystem allows, a
            trailing space/dot, a reserved device name, or a section that is
            empty once split.
    """
    raw = str(section or "").replace("\\", "/")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    if not parts:
        raise VaultPathEscape(
            "refused: section must name a directory inside the vault")
    for part in parts:
        if part == "..":
            raise VaultPathEscape(
                f"refused: '..' is not allowed in a section ('{section}')")
        if not part.strip():
            raise VaultPathEscape(
                f"refused: section must name a directory inside the vault "
                f"('{section}')")
        if part.startswith("."):
            # Aligned with iter_notes(), which skips dot-directories. A write
            # into one used to be accepted and then never read back: the note
            # "succeeded" into a place the vault itself walks past.
            raise VaultPathEscape(
                f"refused: '{part}' is a hidden directory; the vault walk "
                f"skips dot-directories, so a note filed here could never be "
                f"read back")
        if len(part) > _MAX_SECTION_PART:
            raise VaultPathEscape(
                f"refused: section component is longer than "
                f"{_MAX_SECTION_PART} characters ('{part[:32]}...')")
        bad = _ILLEGAL_FS.search(part)
        if bad:
            raise VaultPathEscape(
                f"refused: {bad.group(0)!r} is not allowed in a section "
                f"('{section}')")
        if part.rstrip(" .") != part:
            raise VaultPathEscape(
                f"refused: a section name may not end with a space or a dot "
                f"('{part}')")
        if part.split(".")[0].lower() in _WIN32_RESERVED_STEMS:
            raise VaultPathEscape(
                f"refused: '{part}' is a reserved device name")
    return parts


# -- KB vault ---------------------------------------------------------------
def iter_notes(config: dict, subdirs: list = None, errors: list = None) -> list:
    """Every note in the vault, or every note under the named sections.

    Scans the **whole vault** rather than a fixed section list. ``kb-add`` takes
    an arbitrary ``--section`` and creates that directory on demand, so a fixed
    list meant a note filed under any other heading was written successfully and
    then never found again — invisible to ``kb-search``, ``kb-get`` and the
    ``agents`` provenance report alike. A write that cannot be read back is
    worse than a rejected write, because nothing signals the loss.

    Hidden directories are skipped: ``.obsidian`` / ``.trash`` hold tooling, not
    notes. Root-level scaffold files (``index.md``) are not notes either.

    Every note is re-checked for containment on its **real** path, because the
    walk follows links and can leave the vault (see the loop below).

    ``errors`` is an optional sink: a ``--section`` that is malformed or points
    outside the vault is skipped and named there, because returning "no such
    notes" for a query that was refused reads as an empty vault rather than a
    rejected request. Sections are judged by :func:`split_vault_section` — the
    same validator the write path uses — so the read and write paths can never
    disagree about what is a valid section. Paths that leave the vault mid-walk
    are named there too.
    """
    vault = wiki_dir(config)
    roots = [vault] if subdirs is None else []
    for s in (subdirs or []):
        try:
            # The SAME validator the write path uses, deliberately: a section
            # kb-add refuses with a verdict must not be quietly accepted here
            # and returned as "no notes matched".
            roots.append(safe_vault_path(vault, *split_vault_section(s)))
        except VaultPathEscape as e:
            if errors is not None:
                errors.append(str(e))
    out = []
    # Judged against the *real* vault, computed once: the vault itself is a
    # junction in production, and both sides of every comparison must resolve
    # through the same rule for that to stay "inside".
    vault_real = Path(os.path.realpath(str(vault)))
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            # rglob FOLLOWS links, so the walk cannot be trusted to have stayed
            # inside the vault: a junction or symlink under it — and the
            # production ``wiki`` is itself a junction, so this is the normal
            # case, not an exotic one — steps out, and a chain steps out
            # further. Every hit is re-judged on its real path.
            #
            # The old code did the opposite: `except ValueError: out.append(p)`
            # *kept* the out-of-vault files, so kb-search and recall returned
            # the contents of whatever the link pointed at. Refusals are
            # recorded rather than dropped silently — an empty result and a
            # rejected walk look identical to a caller otherwise.
            real = Path(os.path.realpath(str(p)))
            if not _is_within(real, vault_real):
                if errors is not None:
                    errors.append(
                        f"refused: '{p}' resolves to '{real}', outside the "
                        f"vault '{vault_real}'")
                continue
            try:
                rel = p.relative_to(vault)
            except ValueError:
                # Reachable only via a link that stays inside the vault; the
                # visible identity is then the real one.
                rel = real.relative_to(vault_real)
            if any(part.startswith(".") for part in rel.parts[:-1]):
                continue
            if len(rel.parts) == 1 and rel.stem.lower() in _VAULT_SCAFFOLD:
                continue
            out.append(p)
    return out


def parse_frontmatter(text: str):
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return {}, text
    raw = "\n".join(lines[1:close])
    body = "\n".join(lines[close + 1:]).lstrip("\n")
    data = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                data[k] = json.loads(v)
            except Exception:
                data[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        else:
            data[k] = v.strip("'\"")
    return data, body


def slugify(title: str) -> str:
    name = _ILLEGAL_FS.sub(" ", str(title or "")).strip()
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name[:120].strip(" .") or "untitled")


def dump_frontmatter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            items = [json.dumps(str(x), ensure_ascii=False) for x in v if str(x).strip()]
            lines.append(f"{k}: [{', '.join(items)}]")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {json.dumps(str(v), ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def cmd_kb_search(config: dict, query: str, top_k: int, section: str,
                  errors: list = None) -> list:
    # No section -> search the whole vault, not a fixed list of sections.
    subdirs = [section] if section else None
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
    results = []
    for p in iter_notes(config, subdirs, errors):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        title = meta.get("title") or p.stem
        hay = f"{title} {body}".lower()
        score = sum(1 for t in tokens if t.lower() in hay)
        if score > 0:
            snippet = re.sub(r"\s+", " ", body[:200])
            results.append({"title": title, "path": str(p.relative_to(wiki_dir(config))),
                            "section": p.parent.name, "score": score, "snippet": snippet})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def cmd_kb_get(config: dict, title: str):
    """Read one note by title.

    A miss answers with ``ok: false``. It is **not** built by :func:`_refusal`
    and carries no ``refused: `` prefix, because nothing was refused — but the
    caller is an agent, and an answer without ``ok`` was being read as a
    delivered note: "note not found" and "here is your note" looked the same to
    anything keying on the presence of the payload.
    """
    target = slugify(title).lower()
    for p in iter_notes(config):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        if (meta.get("title") or p.stem).lower() == target or p.stem.lower() == target:
            return {"ok": True, "title": meta.get("title") or p.stem,
                    "path": str(p), "meta": meta, "body": body}
    return {"ok": False, "error": f"note not found: {title}"}


def cmd_kb_add(config: dict, title: str, body: str, section: str,
               tags: list, concepts: list, confidence: float,
               agent: str = DEFAULT_AGENT, overwrite: bool = False):
    """Write a governed note into the vault, attributed to ``agent``.

    Sharing a vault between agents turns the old "silently overwrite the file
    of the same title" behaviour into a data-loss bug: two agents that both
    decide to note "部署注意事项" would erase each other in turn. So an existing
    note owned by a *different* agent is refused unless ``overwrite`` is set —
    the same title from the *same* agent is still a normal update.
    """
    for pat in SECRET_PATTERNS:
        if pat.search(body):
            return _refusal("body looks like it contains a secret")

    threshold = float(config.get("kb", {}).get("confidence_threshold", 0.7))
    if confidence is not None:
        section = "notes" if confidence >= threshold else "inbox"
    else:
        section = section or "notes"

    vault = wiki_dir(config)
    # Containment is decided BEFORE the directory is created: mkdir on an
    # escaping path would happily build the target outside the vault, and the
    # note would then "succeed" into a place nothing reads.
    try:
        subdir = safe_vault_path(vault, *split_vault_section(section))
        filename = slugify(title) + ".md"
        path = safe_vault_path(subdir, filename)
    except VaultPathEscape as e:
        return _refusal(str(e),
                        hint=("--section must name a directory inside the "
                              "vault; use kb-add with a plain section name "
                              "such as 'notes'"))

    if filename.split(".")[0].lower() in _WIN32_RESERVED_STEMS:
        # Refused rather than renamed: silently writing "_CON.md" would make
        # the note unfindable by the title the caller asked for.
        return _refusal(f"'{filename}' is a reserved device name",
                        hint="choose a different title")
    if subdir.exists() and not subdir.is_dir():
        # e.g. --section "notes/seed.md", where a note already owns that name.
        return _refusal(f"'{subdir}' exists and is not a directory",
                        hint="pick a different section name")

    try:
        subdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # Last line of defence for anything the checks above did not anticipate
        # (a read-only vault, a name only the driver objects to). A refusal the
        # caller can read beats a traceback it cannot act on.
        return _refusal((f"cannot create section '{section}': "
                         f"{type(e).__name__}: {e}")[:300],
                        hint="pick a different section name")

    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    existed = path.exists()
    created = now
    if existed:
        try:
            existing_meta, _ = parse_frontmatter(
                path.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            existing_meta = {}
        owner = note_agent(existing_meta)
        if owner and owner != agent and not overwrite:
            return _refusal(
                f"note already exists and is owned by '{owner}'",
                path=str(path),
                hint="choose a different title, or pass --overwrite to take it over")
        created = existing_meta.get("created") or now

    # Auto-link concepts, once per concept, without duplicating an existing line.
    missing = [c for c in (concepts or [])
               if c and f"[[{c}]]" not in body]
    if missing:
        body = body.rstrip() + "\n\nRelated: " + " ".join(f"[[{c}]]" for c in missing)

    # `agent` is the writer; `source` describes how the note got here. Keeping
    # them separate is what makes the legacy vault values ("l3", "wiki-backup")
    # interpretable instead of being mistaken for authors.
    meta = {"title": title.strip(), "type": "note", "tags": tags,
            "concepts": concepts, "agent": agent, "source": "agent-write",
            "created": created, "updated": now}
    text = dump_frontmatter(meta) + "\n\n" + body.rstrip() + "\n"

    # Atomic write: one agent must never observe a half-written note. Failures
    # come back as a verdict rather than a traceback — the caller is an agent,
    # and "the tool crashed" is not something it can act on.
    try:
        fd, tmp = tempfile.mkstemp(dir=str(subdir), suffix=".tmp")
    except OSError as e:
        return _refusal((f"cannot write into '{subdir}': "
                         f"{type(e).__name__}: {e}")[:300])
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError as e:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return _refusal((f"cannot write '{path}': "
                         f"{type(e).__name__}: {e}")[:300])
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"ok": True, "path": str(path), "section": section,
            "title": title.strip(), "agent": agent, "updated": existed}



# -- health ----------------------------------------------------------------
#: How far the CLI's semantic floor may sit from the plugin's cosine floor
#: before ``health`` says so.
#:
#: They measure the *same* quantity (bge-m3 cosine, mapped 1 - d/2), so a gap
#: wider than this means the two entry points are cutting a different amount —
#: "two entry points, two answers" in its quietest form, where nothing fails and
#: the same question just gets different hits depending on who asked.
_L2_THRESHOLD_DRIFT_TOLERANCE = 0.05


def _l2_threshold_drift() -> dict:
    """Describe a gap between the CLI's semantic floor and the plugin's.

    Returns ``{}`` when there is nothing to report — including when either side
    is simply at its default, because an unset value is not a disagreement, it
    is an absence.

    Deliberately does not correct or fall back. The two are separate keys on
    purpose: silently aligning them would hide which one an operator meant to
    change, and that is the same loss of signal as never comparing them. This
    only makes the gap readable to whoever reads ``health``.
    """
    try:
        semantic_floor, semantic_source = _resolve_l2_semantic_floor(None)
    except Exception:  # noqa: BLE001 — health must not fail on a dirty config
        return {}

    plugin_value, plugin_source = None, ""
    try:
        raw = float(getattr(plugin_config().recall, "l2_min_score", 0.0) or 0.0)
        if raw > 0.0:
            plugin_value, plugin_source = raw, "recall.l2_min_score"
    except (TypeError, ValueError):
        plugin_value = None
    except Exception:  # noqa: BLE001
        plugin_value = None
    if plugin_value is None:
        try:
            raw = float((load_config().get("recall") or {}).get("l2_min_score") or 0.0)
            if raw > 0.0:
                plugin_value, plugin_source = raw, "recall.l2_min_score (file)"
        except Exception:  # noqa: BLE001
            return {}

    # An unset CLI floor is not a disagreement with the plugin — it just means
    # nobody tuned this side, and warning about it would be noise.
    if plugin_value is None or semantic_source == "default":
        return {}
    delta = abs(float(semantic_floor) - plugin_value)
    if delta <= _L2_THRESHOLD_DRIFT_TOLERANCE:
        return {}
    return {
        "warning": (f"the CLI's semantic floor and the plugin's cosine floor "
                    f"differ by {delta:.3f} (tolerance "
                    f"{_L2_THRESHOLD_DRIFT_TOLERANCE}). Both cut bge-m3 cosine "
                    f"similarity, so the same query is being thresholded "
                    f"differently depending on the entry point. Not corrected: "
                    f"they are separate keys on purpose — change the one you "
                    f"meant."),
        "cli_l2_semantic_min_score": {"value": float(semantic_floor),
                                      "source": semantic_source},
        "plugin_l2_min_score": {"value": plugin_value,
                                "source": plugin_source},
        "delta": round(delta, 4),
    }


def cmd_health(config: dict):
    h = hermes_home()
    l2 = Path(h) / "memory" / "l2"
    l3 = Path(h) / "memory" / "l3" / "l3.db"
    l1 = cmd_l1()
    count_notes = len(iter_notes(config))
    out = {
        "hermes_home": h,
        "l1_memory_md": bool(l1["memory_rules_md"]),
        "l1_user_md": bool(l1["user_profile_md"]),
        "l4_persona_md": bool(l1["persona_md"]),
        "l2_dir_exists": l2.exists(),
        "l3_db_exists": l3.exists(),
        "wiki_dir": str(wiki_dir(config)),
        "vault_note_count": count_notes,
        "ok": bool(l1["memory_rules_md"] or l1["user_profile_md"]),
    }
    # Emitted only when there is something to say: a health line that is always
    # present is a health line nobody reads.
    drift = _l2_threshold_drift()
    if drift:
        out["l2_threshold_drift"] = drift
    return out


#: Commands whose answer depends on L2, and therefore on LanceDB being
#: importable. Only these pay the cost of a possible interpreter re-exec; the
#: read-only Markdown commands stay fast on a bare interpreter.
_L2_COMMANDS = {"recall", "remember", "agents", "health", "runtime", "snapshot"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Governed memory CLI — the shared store, for any agent")
    sub = parser.add_subparsers(dest="cmd")

    # Shared across every subcommand so `--agent` can be given anywhere.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--agent", default="",
        help=f"Agent name to attribute this call to "
             f"(default: ${AGENT_ENV_VAR} or auto-detection, else '{DEFAULT_AGENT}')")

    p = sub.add_parser("recall", parents=[common], help="Search L1+L2+L3+L4 + KB")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--project", default="",
                   help="Narrow L2 to this project plus global facts "
                        "(default: no narrowing — search every project)")
    p.add_argument("--lexical-only", action="store_true",
                   help="Skip the vector channel and answer from keywords "
                        "alone (A/B comparison, offline use). The ranking "
                        "says which channel was used either way.")

    ks = sub.add_parser("kb-search", parents=[common],
                        help="Search the Obsidian knowledge base")
    ks.add_argument("query")
    ks.add_argument("--top-k", type=int, default=10)
    ks.add_argument("--section", default="")

    kg = sub.add_parser("kb-get", parents=[common], help="Read a full note by title")
    kg.add_argument("title")

    ka = sub.add_parser("kb-add", parents=[common],
                        help="Write a durable note into the vault (governed)")
    ka.add_argument("title")
    ka.add_argument("body")
    ka.add_argument("--section", default="notes")
    ka.add_argument("--tags", nargs="*", default=[])
    ka.add_argument("--concepts", nargs="*", default=[])
    ka.add_argument("--confidence", type=float, default=None)
    ka.add_argument("--overwrite", action="store_true",
                    help="Take over a note currently owned by another agent")

    rm = sub.add_parser("remember", parents=[common],
                        help="Admit a durable fact into L2 (shared admission gate)")
    rm.add_argument("fact")
    rm.add_argument("--category", default="other")
    rm.add_argument("--dry-run", action="store_true",
                    help="Run the gate and report the verdict without writing")
    rm.add_argument("--project", default=None,
                    help="Project to attribute the fact to. Omit to infer from "
                         "the current directory, or pass '' for a fact that "
                         "holds everywhere")

    sub.add_parser("l1", parents=[common], help="Print the standing L1 rules + L4 persona")
    sub.add_parser("agents", parents=[common], help="Who has written what (provenance)")
    sn = sub.add_parser(
        "snapshot", parents=[common],
        help="Dump every L2 fact to a JSON file, so a caller can match offline")
    sn.add_argument("--out", default="",
                    help="Where to write it "
                         "(default: $HERMES_HOME/memory/l2_snapshot.json)")
    sn.add_argument("--max-facts", type=int, default=0,
                    help=f"Keep at most this many facts, newest first "
                         f"(default: {L2_SNAPSHOT_MAX_FACTS})")
    sub.add_parser("health", parents=[common], help="Memory health report")
    sub.add_parser("runtime", parents=[common],
                   help="Which interpreter this ran under and what it can see")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.cmd:
        parser.print_help()
        return 2

    if args.cmd in _L2_COMMANDS:
        bootstrap_interpreter()

    config = load_config()
    agent = resolve_agent(getattr(args, "agent", ""))

    if args.cmd == "l1":
        print(json.dumps(cmd_l1(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "runtime":
        print(json.dumps(runtime_report(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "health":
        out = cmd_health(config)
        out["runtime"] = runtime_report()
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "agents":
        print(json.dumps(cmd_agents(config), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "snapshot":
        out = cmd_snapshot(config, getattr(args, "out", ""),
                           getattr(args, "max_facts", 0))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if out.get("ok") else 1

    if args.cmd == "recall":
        degraded: list = []
        scoped = normalize_project(getattr(args, "project", "") or "")
        l2 = recall_l2(args.query, args.top_k, degraded, project=scoped,
                       lexical_only=getattr(args, "lexical_only", False))
        kb_refused: list = []
        out = {
            "query": args.query,
            "l1": cmd_l1(),
            "l2": l2["hits"],
            # How the L2 hits were produced. Emitted unconditionally: a score
            # the caller cannot attribute to a channel is a score they will
            # over-trust.
            "l2_ranking": l2["ranking"],
            "l3": search_l3(args.query, args.top_k, degraded),
            "kb": cmd_kb_search(config, args.query, args.top_k, "", kb_refused),
        }
        if kb_refused:
            out["kb_refused"] = kb_refused
        # Absence of a layer is stated, never implied: "no memory matched" and
        # "memory could not be read" must not look the same to the caller.
        if degraded:
            out["degraded"] = degraded
        # Same principle for the filter: a caller that scoped the query must be
        # able to confirm it, and one that did not must not be left wondering
        # whether another project's facts were silently withheld.
        if scoped:
            out["project"] = scoped
    elif args.cmd == "kb-search":
        refused: list = []
        results = cmd_kb_search(config, args.query, args.top_k, args.section,
                                refused)
        out = {"results": results}
        # An empty list is what "no notes matched" looks like; a refused
        # --section must not borrow that shape.
        if refused:
            out["refused"] = refused
    elif args.cmd == "kb-get":
        out = cmd_kb_get(config, args.title)
    elif args.cmd == "kb-add":
        out = cmd_kb_add(config, args.title, args.body, args.section,
                         args.tags, args.concepts, args.confidence,
                         agent=agent, overwrite=args.overwrite)
    elif args.cmd == "remember":
        out = cmd_remember(config, args.fact, agent,
                           category=args.category, dry_run=args.dry_run,
                           project=args.project)
    else:
        parser.print_help()
        return 2

    print(json.dumps(out, ensure_ascii=False, indent=2))
    # A refused write is a failure the caller must act on, not a soft result.
    if isinstance(out, dict) and out.get("ok") is False:
        return 1
    # A refused *read* is the same kind of failure. Without this, a rejected
    # --section exited 0 and a script keying on rc read it as "empty vault".
    if args.cmd == "kb-search" and isinstance(out, dict) and out.get("refused"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
