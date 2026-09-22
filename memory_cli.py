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
import random
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

#: Floor for the CLI's **semantic** L2 channel, on the **score** scale
#: (``distance_to_score`` / ``1 - d/2``), *not* raw cosine similarity.
#:
#: Unit trap (kept explicit after a near-miss): raw cosine ``cos`` and this
#: score are related by ``score = (1 + cos) / 2``. A floor of 0.77 on the
#: score scale corresponds to raw cosine 0.54 — comparing 0.77 against raw
#: cos would be off by a factor that empties or floods the channel. Every
#: calibration number below was measured through ``distance_to_score``.
#:
#: A DIFFERENT quantity from :data:`DEFAULT_L2_LEXICAL_FLOOR` and a different
#: key (``recall.l2_semantic_min_score``). It must never be read from
#: ``recall.l2_min_score``: that value belongs to the in-plugin channel, and
#: borrowing a threshold calibrated for one quantity to cut another is exactly
#: the defect that already emptied the lexical channel once.
#:
#: Calibrated 2026-09-17 against a 21-fact store (siliconflow ``BAAI/bge-m3``,
#: 1024-d) with 10 relevant + 8 irrelevant queries — on the score scale:
#:
#:     relevant   top1 in [0.7843, 0.9640]   10/10 kept
#:     irrelevant top1 in [0.6693, 0.7586]    0/8 leaked
#:
#: The usable window is (0.7586, 0.7843]; 0.77 sits inside it with margin on
#: both sides — 0.014 below the relevant minimum, 0.011 above the irrelevant
#: maximum. (An earlier revision had the two numbers swapped; read as-is it
#: understates one margin and overstates the other.)
#:
#: 2026-09-22 复核（2058 行含 rebuild 噪声时）：垃圾查询 top1 可达 0.7896，
#: 与相关命中 0.781~0.804 区间重叠 ⇒ 地板落在噪声带内部，``filtered_out`` 恒为 0。
#:
#: 2026-09-22 二次复核（独立构造查询集，每条真相关查询的正确行都指名在库里，
#: 并用 ``pre_restore`` 备份复现了 33 行基线、逐位吻合）：
#:
#:     33 行精选集            噪声上沿 0.7686   真相关下沿 0.7427   重叠 0.0259
#:     1632 行（含 rebuild 碎片） 噪声上沿 0.8914   真相关下沿 0.7506   重叠 0.1408
#:
#: ⇒ **在任何规模的语料上都不存在能同时挡住全部噪声、又保住全部相关命中的
#: 绝对阈值**。唯一能挡住全部噪声的窗口 (0.8914, 0.9116] 只保住 1/16 条相关
#: 查询，等于把召回砍死。
#:
#: **因此不要改本常量的数值。** 这不是"选错数"，是"绝对地板"这个机制在碎片
#: 语料上不成立。可用性来自**精选**（把碎片归纳成事实），不来自阈值 —— 33 行
#: 精选集是唯一让正确行排到 #1 的配置。要改请改机制（相对判据：top1 与次名的
#: gap、命中比例），并重新测量。
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
        search = table.search(vector, vector_column_name="vector").metric("cosine")
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
        _touch_l2_usage([h.get("content", "") for h in kept])
        out.update({"available": True, "hits": kept, "filtered_out": filtered,
                    "truncated": len(hits) - len(kept),
                    "backend": getattr(embedding, "backend_name", "") or ""})
        return out
    except Exception as e:  # noqa: BLE001
        return _unavailable("semantic unavailable: vector search failed "
                            f"({type(e).__name__}: {e})")


def _touch_l2_usage(contents: list) -> None:
    """召回命中即计数（容量淘汰的受害者选择依据，Phase 3）。

    仅在 ``sync.l2_max_items > 0``（容量开启）时落表 —— 默认关闭 = 读路径
    零副作用，与 ``kb.affinity_enabled`` 同一约定。失败静默：统计绝不影响
    召回。语义/词法两条通道的 kept 结果都过这里。
    """
    try:
        cfg = plugin_config()
        if int(getattr(cfg.sync, "l2_max_items", 0) or 0) <= 0:
            return
        life = plugin_module("_lifecycle")
        mem_dir = life.resolve_memory_dir(cfg)
        if mem_dir is not None:
            life.usage_touch(mem_dir, [str(c or "") for c in contents])
    except Exception:  # noqa: BLE001 — 统计失败绝不影响召回
        pass


def _l2_enforce_capacity(table, cfg, agent: str, cap: int) -> list:
    """同 agent 行数将超 ``cap`` 时，把**最少使用**的事实降级归档。

    受害者排序：``last_used`` 升序（无记录 = 0，从未被召回的最先走）；
    平手按 ``timestamp`` 升序（写入越早越先）。归档**先于**删除 —— 归档
    失败顶多是下一次重试时重复一行归档记录，而删除先于归档失败就是数据
    丢失；两种失败模式里可接受的是前者。

    不写墓碑：容量淘汰的事实没有做错什么，重新写入是合法操作（与 retract
    的 supersede 语义刻意区分）。

    Returns:
        被归档的内容前缀列表（空 = 不需要淘汰）。
    """
    life = plugin_module("_lifecycle")
    mem_dir = life.resolve_memory_dir(cfg)
    if mem_dir is None:
        return []
    arr = table.to_arrow()
    names = arr.column_names
    if "content" not in names:
        return []
    contents = arr["content"].to_pylist()
    agents = arr["agent"].to_pylist() if "agent" in names else [None] * len(contents)
    stamps = (arr["timestamp"].to_pylist() if "timestamp" in names
              else [None] * len(contents))
    mine = [(c, s) for c, a, s in zip(contents, agents, stamps)
            if c and str(a or "") == agent]
    over = len(mine) - cap + 1  # 插入后总数必须 ≤ cap
    if over <= 0:
        return []
    usage = life.usage_last_used(mem_dir)

    def _victim_key(item):
        content, stamp = item
        return (usage.get(life.content_fp(content), 0.0), str(stamp or ""))

    victims = sorted(mine, key=_victim_key)[:over]
    # 删除前先备份整表；失败就放弃本轮淘汰（降级可以晚一轮，数据不能丢）。
    if life.backup_table(cfg.l2_db_path, tag="capacity",
                         memory_dir=mem_dir) is None:
        logger.error("容量淘汰跳过：无法在删除前备份 L2 表")
        return []
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    demoted = []
    for content, stamp in victims:
        life.archive_append(mem_dir, {
            "state": "archived", "reason": "capacity-demoted",
            "ts": now, "content": content, "agent": agent,
            "timestamp": stamp, "cap": cap})
        table.delete("content = " + life.sql_quote(content))
        demoted.append(content[:200])
    return demoted


def cmd_retract(config: dict, match: str, reason: str = "",
                agent: str = DEFAULT_AGENT, dry_run: bool = False) -> dict:
    """撤回 L2 事实（supersede）：移入归档 + 写墓碑，防止被重新抽取复活。

    ``match`` 是内容子串（大小写不敏感）；命中多条时全部处理。流程：

    1. 只读打开表（不创建）；无表 → 明确 refusal；
    2. ``--dry-run`` 列出命中不动数据（批量删除前必须先看 scope）；
    3. 每条命中：归档 JSONL（``state=superseded`` + reason）→ 写内容指纹
       墓碑 → 从表中删除。**归档先于删除**（同容量淘汰：宁可重复归档，
       不可丢失）；
    4. 刷新离线快照，让 ``snapshot`` 与表保持一致。

    Returns:
        ``{ok, retracted, tombstones, contents, errors, dry_run?}``。
    """
    match = (match or "").strip()
    if not match:
        return _refusal("empty match",
                        hint="pass a substring of the fact to retract")

    cfg = plugin_config()
    life = plugin_module("_lifecycle")
    mem_dir = life.resolve_memory_dir(cfg)
    if mem_dir is None:
        return _refusal("l2_db_path not configured",
                        detail="cannot locate the memory dir for archive/tombstones")

    open_errors: list = []
    try:
        table = open_l2_table(cfg, errors=open_errors)
    except Exception as exc:  # noqa: BLE001 — 调用方拿 JSON，不拿堆栈
        table = None
        open_errors.append(f"{type(exc).__name__}: {exc}")
    if table is None:
        detail = str(open_errors[0])[:300] if open_errors else "no 'memories' table"
        return _refusal("l2_open_failed", detail=detail)

    try:
        arr = table.to_arrow()
        names = arr.column_names
        contents = arr["content"].to_pylist() if "content" in names else []
        agents = arr["agent"].to_pylist() if "agent" in names else [None] * len(contents)
        stamps = (arr["timestamp"].to_pylist() if "timestamp" in names
                  else [None] * len(contents))
        cats = arr["category"].to_pylist() if "category" in names else [None] * len(contents)
        projects = arr["project"].to_pylist() if "project" in names else [None] * len(contents)
    except Exception as e:  # noqa: BLE001
        return _refusal("l2_read_failed", detail=str(e)[:300])

    needle = match.lower()
    matched = [(c, a, s, cat, pj) for c, a, s, cat, pj
               in zip(contents, agents, stamps, cats, projects)
               if c and needle in str(c).lower()]

    base = {"match": match, "retracted": 0, "tombstones": 0,
            "contents": [str(c)[:200] for c, *_ in matched],
            "errors": []}
    if not matched:
        base.update({"ok": True, "note": "no fact matched"})
        return base
    if dry_run:
        base.update({"ok": True, "dry_run": True, "matched": len(matched)})
        return base

    # 删除前先备份整表。retract 是**不可逆**的语义删除，而 ``match`` 是子串
    # 匹配 —— 很容易一次命中一片；拿不到备份就拒绝动手，且**不碰任何数据**。
    backup = life.backup_table(cfg.l2_db_path, tag="retract", memory_dir=mem_dir)
    if backup is None:
        base.update({"ok": False, "backup": None,
                     "error": "refused: cannot back up the L2 table before "
                              "deleting — nothing was touched"})
        return base
    base["backup"] = str(backup)

    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    seen = set()
    for content, src_agent, stamp, cat, pj in matched:
        if content in seen:
            continue
        seen.add(content)
        life.archive_append(mem_dir, {
            "state": "superseded", "reason": reason or "user-retracted",
            "retracted_by": agent, "ts": now, "content": content,
            "agent": src_agent, "category": cat, "project": pj,
            "timestamp": stamp})
        life.add_tombstone(mem_dir, life.content_fp(content), {
            "content_prefix": content[:80],
            "reason": reason or "user-retracted",
            "retracted_by": agent, "ts": now})
        try:
            table.delete("content = " + life.sql_quote(content))
            base["retracted"] += 1
            base["tombstones"] += 1
        except Exception as e:  # noqa: BLE001 — 已归档 + 已立碑，行还在：如实上报
            base["errors"].append(f"delete failed for {content[:60]}: {e}")

    snap = cmd_snapshot(config)
    if not snap.get("ok"):
        base["snapshot_error"] = snap.get("error")
    base["ok"] = not base["errors"]
    return base


# -- 巩固（consolidate）与评测（eval），Phase 4 / WeKnora 借鉴 ---------------

#: 巩固候选门槛：Jaccard（bigram）≥ 0.55 **且** 余弦 ≥ 0.86。
#:
#: 双条件缺一不可（WeKnora 数值）：词法像 + 向量像才可能是同一件事的两次
#: 表述；只靠词法会把「同话题的不同事实」、只靠向量会把「换了说法的另一件事」
#: 误合并。合并是一次有损操作，误合并比漏合并昂贵得多。
CONS_JACCARD_MIN = 0.55
CONS_COSINE_MIN = 0.86
#: 单簇规模上限。没有它，链式（传递）相似会滚成一个大簇，而 LLM 合并是**有损**
#: 操作 —— 把一堆只是彼此相邻、却互不相干的事实并成一句，比漏合并贵得多。
CONS_MAX_CLUSTER = 5

#: 自评 QA 集的样本上限（snapshot 可能上千条，评测抽样固定 20 条保廉价）。
EVAL_MAX_QUERIES = 20


def _cons_bigrams(text: str) -> set:
    # 归一化复用查重那一份（同一规则），别再抄第四份 —— 漂移会让"相似度"
    # 与"指纹/查重"对同一段文本给出不同答案。
    t = _normalize_for_dedup(text)
    return {t[i:i + 2] for i in range(max(0, len(t) - 1))}


def _cons_jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cons_cosine(va, vb) -> float:
    try:
        fax = [float(x) for x in (va or [])]
        fby = [float(y) for y in (vb or [])]
        dot = sum(x * y for x, y in zip(fax, fby))
        na = sum(x * x for x in fax) ** 0.5
        nb = sum(y * y for y in fby) ** 0.5
    except (TypeError, ValueError):
        return 0.0
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _cons_clusters(rows: list) -> list:
    """同类 + 双门槛的合并候选聚簇（**代表制**，不是传递闭包）。

    为什么不用并查集：传递闭包下 A~B、B~C 会把 {A,B,C} 并成一簇，哪怕 A 与 C
    毫不相干 —— 链式扩张没有上界，而 LLM 合并是**有损**的，误合并比漏合并贵。
    这里改为：按出现顺序取未分配的行当**簇代表**，只吸收与**代表本身**同时过
    双门槛的行，且单簇不超过 ``CONS_MAX_CLUSTER``。

    Args:
        rows: ``[{content, category, vector}]``。

    Returns:
        下标组列表，每组 ≥ 2 行（单行不成簇）。
    """
    n = len(rows)
    grams = [_cons_bigrams(str(r.get("content") or "")) for r in rows]
    cats = [str(r.get("category") or "") for r in rows]
    assigned = [False] * n
    clusters: list = []
    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if len(group) >= CONS_MAX_CLUSTER:
                break
            if cats[i] != cats[j]:
                continue
            if _cons_jaccard(grams[i], grams[j]) < CONS_JACCARD_MIN:
                continue
            if _cons_cosine(rows[i].get("vector"),
                            rows[j].get("vector")) < CONS_COSINE_MIN:
                continue
            group.append(j)
        if len(group) > 1:
            for k in group:
                assigned[k] = True
            clusters.append(sorted(group))
    return clusters


def cmd_consolidate(config: dict, apply: bool = False,
                    agent: str = DEFAULT_AGENT) -> dict:
    """巩固：近重复事实聚簇 → LLM 合并（WeKnora consolidation 的 CLI 落地）。

    候选 = 同类（category）+ Jaccard≥0.55 + 余弦≥0.86（见 ``CONS_*``）。
    默认 dry-run 只报候选 —— 批量操作先报 scope。

    ``--apply`` 时每个簇：LLM 合并（temp=0；**空输出 = 模型拒绝合并**，
    原条目原样保留；LLM 不可用 → 该簇 skipped）→ 原条目归档
    （state=superseded, reason=consolidated）+ 内容指纹墓碑（防旧表述
    被重新抽取复活）→ 插入合并后的单条。任何一步失败都保留原条目并如实
    记录 —— 合并宁可不发生，不可发生一半。

    ``demoted``/``expired`` 不在本命令里：降级归容量（``l2_max_items``）、
    过期归时间衰减（L3），各管各的管道，不在此重复报告。
    """
    cfg = plugin_config()
    life = plugin_module("_lifecycle")
    mem_dir = life.resolve_memory_dir(cfg)
    if mem_dir is None:
        return _refusal("l2_db_path not configured",
                        detail="cannot locate the memory dir")
    open_errors: list = []
    try:
        table = open_l2_table(cfg, errors=open_errors)
    except Exception as exc:  # noqa: BLE001
        table = None
        open_errors.append(f"{type(exc).__name__}: {exc}")
    if table is None:
        detail = str(open_errors[0])[:300] if open_errors else "no 'memories' table"
        return _refusal("l2_open_failed", detail=detail)
    try:
        rows = table.to_arrow().to_pylist()
    except Exception as e:  # noqa: BLE001
        return _refusal("l2_read_failed", detail=str(e)[:300])

    clusters = _cons_clusters(rows)
    base = {"ok": True, "applied": False, "clusters": len(clusters),
            "candidates": [[str(rows[i].get("content") or "")[:80]
                            for i in c] for c in clusters],
            "merged": [], "skipped": [], "errors": []}
    # 候选整份落文件：几十上百个簇没法靠 stdout 的 80 字前缀人工审，而
    # ``--apply`` 是**有损**操作，审必须在动手之前（沿用 ``cmd_snapshot`` →
    # ``l2_snapshot.json`` 的先例：要人工看的东西就落成文件）。
    if clusters:
        try:
            cand = mem_dir / ("consolidate_"
                              + time.strftime("%Y%m%d_%H%M%S") + ".json")
            cand.write_text(json.dumps({
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "thresholds": {"jaccard_min": CONS_JACCARD_MIN,
                               "cosine_min": CONS_COSINE_MIN,
                               "max_cluster": CONS_MAX_CLUSTER},
                "clusters": [
                    {"size": len(c),
                     "members": [{"index": int(i),
                                  "category": rows[i].get("category"),
                                  "agent": rows[i].get("agent"),
                                  "content": str(rows[i].get("content") or "")}
                                 for i in c]}
                    for c in clusters],
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            base["candidates_file"] = str(cand)
        except OSError as e:
            base["errors"].append(f"candidates file not written: {e}")
    if not apply:
        return base
    if not clusters:
        base["note"] = "no merge candidates"
        return base

    # --apply：先验前置条件，缺了就整个合并阶段不动任何数据（abort，
    # 与 WeKnora「model unavailable = abort round」同一语义）。
    embedding = plugin_module("_embedding").EmbeddingService.get(cfg)
    if not embedding.available:
        base.update(_refusal(
            "embedding_unavailable",
            detail=embedding.last_error or "cannot embed the merged fact",
            hint="nothing was changed; fix the embedding backend and retry"))
        base["applied"] = False
        return base
    # 删除前先备份整表；失败即中止整个合并阶段（原条目一条不动）。
    backup = life.backup_table(cfg.l2_db_path, tag="consolidate",
                               memory_dir=mem_dir)
    if backup is None:
        base.update(_refusal(
            "backup_failed",
            detail="cannot back up the L2 table before deleting",
            hint="nothing was changed; free some space and retry"))
        base["applied"] = False
        return base
    base["backup"] = str(backup)
    llm = plugin_module("_llm")
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

    for cidx, members in enumerate(clusters):
        items = [rows[i] for i in members]
        prompt_lines = "\n".join(
            f"{n + 1}. {str(it.get('content') or '')}"
            for n, it in enumerate(items))
        resp = llm.chat_completion(
            [
                {"role": "system", "content": (
                    "Merge near-duplicate facts into ONE fact sentence that "
                    "keeps every unique detail. Output ONLY the merged fact "
                    "(plain text, no list, no quotes). If they should NOT be "
                    "merged, output NOTHING at all.")},
                {"role": "user", "content": f"Facts:\n{prompt_lines}"},
            ],
            cfg, temperature=0.0, max_tokens=300, timeout=60.0)
        if resp is None:
            base["skipped"].append({"cluster": cidx, "reason": "llm_unavailable"})
            continue
        merged_text = str(resp).strip().strip("`\"'")
        if not merged_text:
            # 空输出 = 模型拒绝合并，不是错误：原条目保留，原因说出来。
            base["skipped"].append({"cluster": cidx, "reason": "declined"})
            continue
        vec = embedding.embed_one(merged_text)
        if vec is None:
            base["skipped"].append({"cluster": cidx, "reason": "embedding_failed"})
            continue
        originals = [str(it.get("content") or "") for it in items]
        merged_fp = life.content_fp(merged_text)
        try:
            # 归档 + 墓碑先于删除（宁可重复归档，不可丢失）。
            for it in items:
                content = str(it.get("content") or "")
                life.archive_append(mem_dir, {
                    "state": "superseded", "reason": "consolidated",
                    "ts": now, "content": content,
                    "agent": it.get("agent"), "category": it.get("category"),
                    "timestamp": it.get("timestamp"), "merged_into": merged_fp})
                life.add_tombstone(mem_dir, life.content_fp(content), {
                    "content_prefix": content[:80],
                    "reason": "consolidated", "retracted_by": agent, "ts": now})
            for content in originals:
                table.delete("content = " + life.sql_quote(content))
            table.add([{
                "content": merged_text,
                "category": str(items[0].get("category") or "other"),
                "source": "consolidate", "timestamp": now, "vector": vec,
                "source_rowid": None, "role": "agent", "agent": agent,
                "project": items[0].get("project"),
            }])
            base["merged"].append({"into": merged_text[:200],
                                   "from": [o[:80] for o in originals]})
        except Exception as e:  # noqa: BLE001 — 部分失败如实上报
            base["errors"].append(f"cluster {cidx}: {str(e)[:160]}")

    if base["merged"]:
        snap = cmd_snapshot(config)
        if not snap.get("ok"):
            base["snapshot_error"] = snap.get("error")
    base["applied"] = True
    base["ok"] = not base["errors"]
    return base


def _eval_hit(got: str, expect: str) -> bool:
    """命中判定：归一化后互相包含即算 —— 评测要抓的是「找没找到」，
    不是措辞是否逐字相同。"""
    g = _normalize_for_dedup(got)
    e = _normalize_for_dedup(expect)
    if not g or not e:
        return False
    return g in e or e in g


def cmd_eval(config: dict, qa_file: str = "", top_k: int = 5,
             min_hit: float = 0.0, seed: int = 0,
             engine_kind: str = "cli") -> dict:
    """检索评测：QA 集跑 ``recall_l2``，出 hit@k（Phase 4 验收门）。

    两种集：

    * 默认 ``self-snapshot`` —— 从 L2 快照抽最多 ``EVAL_MAX_QUERIES`` 条
      事实，问句 = 事实本身，期望 = 找回自己。这不是智力题，是**通道健康
      检查**：任何 P0（分数换算错误、门槛错层、通道二选一退化）都会让它
      掉下去 —— 每个阶段开关切换前后各跑一次，就是本方案的验收门。
    * ``--qa file.json`` —— ``[{"q": ..., "expect": ...}, ...]``。

    Returns:
        ``{ok, basis, evaluated, hit_at_k, top_k, misses, degraded?}``。
    """
    if qa_file:
        try:
            raw = json.loads(Path(qa_file).read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return _refusal(f"qa file unreadable: {e}")
        if not isinstance(raw, list):
            return _refusal("qa file must be a JSON list of {q, expect}")
        qa = [{"q": str(it.get("q") or ""), "expect": str(it.get("expect") or "")}
              for it in raw if isinstance(it, dict)]
        basis = "file"
    else:
        # cmd_snapshot 的返回值只带计数（facts 落在文件里）—— 先刷新快照，
        # 再从快照文件读事实。两条路径（写文件 / 返回计数）本就分工如此。
        snap = cmd_snapshot(config)
        if not snap.get("ok"):
            return _refusal(f"snapshot unavailable: {snap.get('error')}")
        try:
            data = json.loads(Path(snapshot_path()).read_text(
                encoding="utf-8"))
        except (OSError, ValueError) as e:
            return _refusal(f"snapshot unreadable: {e}")
        facts = [str(f.get("content") or "")
                 for f in (data.get("facts") or [])
                 if isinstance(f, dict) and str(f.get("content") or "").strip()]
        if not facts:
            return {"ok": True, "basis": "self-snapshot", "evaluated": 0,
                    "hit_at_k": None, "top_k": top_k, "misses": [],
                    "note": "no facts to evaluate"}
        # **随机（固定种子）抽样**，不是等距切片：快照是"最新优先且可能被截断"
        # 的，等距取 20 条永远落在同一批上、且永远取不到最旧/最衰减的事实 ——
        # 那样的门既不可复现也不代表全库。固定种子 = 可复现的随机。
        pool = sorted(set(facts))
        samples = (random.Random(seed).sample(pool, EVAL_MAX_QUERIES)
                   if len(pool) > EVAL_MAX_QUERIES else pool)
        sampled_from = len(pool)
        snapshot_truncated = bool(data.get("truncated"))
        qa = [{"q": s, "expect": s} for s in samples]
        basis = "self-snapshot"
    qa = [item for item in qa if item["q"]]
    if not qa:
        return _refusal("no usable queries (all q fields empty)")

    hits = 0
    misses = []
    degraded: list = []
    # 引擎选择：默认 ``cli`` 是兼容路径（**看不见 ``recall.l2_fusion``**），
    # ``--engine plugin`` 才走与实际召回同源的引擎（融合命中按 native_score
    # 过门槛）。无论用哪条，报告里都写明 —— 上个版本的门对本次交付全盲、
    # 却自称是它的验收门，就是因为"用哪条路"从来没被写出来过。
    engine = None
    engine_name = "cli-recall_l2"
    if str(engine_kind).lower() == "plugin":
        try:
            engine = plugin_module("_recall").RecallEngine(plugin_config())
            engine_name = "plugin-recall"
        except Exception as e:  # noqa: BLE001 — 要报出来，不能静默换引擎
            return _refusal(f"plugin engine unavailable: {type(e).__name__}: {e}",
                            hint="omit --engine plugin to use the CLI path")

    for item in qa:
        if engine is not None:
            got = [str(getattr(r, "content", "") or "")
                   for r in engine.admitted_l2(item["q"], top_k)]
        else:
            result = recall_l2(item["q"], top_k, degraded)
            got = [str(h.get("content") or "") for h in result.get("hits", [])]
        if any(_eval_hit(g, item["expect"]) for g in got):
            hits += 1
        else:
            misses.append({"q": item["q"][:80],
                           "expect": item["expect"][:80]})
    out = {"ok": True, "basis": basis, "evaluated": len(qa),
           "hit_at_k": round(hits / len(qa), 4) if qa else None,
           "top_k": top_k, "hits": hits, "misses": misses,
           "engine": engine_name}
    if basis == "self-snapshot":
        out["sampled_from"] = sampled_from
        out["snapshot_truncated"] = snapshot_truncated
    if degraded:
        out["degraded"] = degraded
    # 门槛判定：没有它，hit@k 只是个数字 —— 实测它在 floor∈[0.0,0.99] 全程恒为
    # 1.0，只有 1.01 才掉下去，等于"永远通过"的橡皮章。有了下限，
    # ``ok:false`` 会让 CLI 以非零退出，才配叫验收门。
    if min_hit > 0.0:
        out["min_hit_at_k"] = min_hit
        if out["hit_at_k"] is None or out["hit_at_k"] < min_hit:
            out["ok"] = False
            out["error"] = (f"hit@{top_k} {out['hit_at_k']} below the required "
                            f"{min_hit}（engine={engine_name}）")
    return out


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
        _touch_l2_usage([h.get("content", "") for h in kept])
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
    # Surrogate hygiene comes from the plugin's single implementation rather
    # than a copy of it: LanceDB stores Arrow strings, which demand strict
    # UTF-8, so a lone surrogate makes the row unwritable. See _text.py.
    cleaned = plugin_module("_text").sanitize_utf8(cleaned)
    resolved_project = (infer_project() if project is None
                        else normalize_project(project))

    base = {"agent": agent, "category": category, "project": resolved_project}

    # 凭据 fail-closed（净化漏斗第 2 步）：与 kb-add 同一闸门。凭据**拒绝**
    # 而不是脱敏后写入 —— 脱敏后的句子通常已无信息量，而凭据策略要求它
    # 根本不落盘。
    for pat in SECRET_PATTERNS:
        if pat.search(cleaned):
            base.update(_refusal(
                "fact looks like it contains a secret",
                hint=("strip the credential and keep only the non-secret part, "
                      "e.g. 'SecretStore holds the TLS material'")))
            return base

    cfg = plugin_config()

    # 墓碑门禁（防复活，净化漏斗第 3 步）：被 retract 撤回过的事实，重新
    # 抽取/写入一律拒绝并点名最初的 reason —— 「删了又长回来」到此为止。
    life = plugin_module("_lifecycle")
    mem_dir = life.resolve_memory_dir(cfg)
    if mem_dir is not None:
        tomb = life.load_tombstones(mem_dir).get(life.content_fp(cleaned))
        if tomb is not None:
            base.update(_refusal(
                "tombstoned",
                detail=(f"retracted at {tomb.get('ts', '?')} "
                        f"(reason: {tomb.get('reason', 'n/a')})"),
                hint=("this fact was explicitly retracted; write a NEW fact "
                      "if the world has since changed")))
            return base

    admitted, reason = sync.external_write_verdict(cleaned)
    base["admitted"] = admitted
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

    existing = _l2_existing_contents(table)
    if cleaned in existing:
        base.update({"ok": True, "duplicate": True, "content": cleaned[:600]})
        return base
    # 长事实的包含性查重（净化漏斗第 4 步）：短句互相包含是常态，只有超过
    # 与 KB 查重同一门槛的长句才做包含判断，避免琐碎误杀。
    if len(cleaned) >= _DEDUP_MIN_CHARS:
        for other in existing:
            if len(other) < _DEDUP_MIN_CHARS:
                continue
            if cleaned in other or other in cleaned:
                base.update({"ok": True, "duplicate": True,
                             "content": cleaned[:600],
                             "duplicate_of": other[:300]})
                return base

    # 容量淘汰（净化漏斗第 5 步，``sync.l2_max_items > 0`` 时）：写入前把
    # **最少使用**的同 agent 事实降级归档（不写墓碑 —— 被挤掉的可以再写回
    # 来）。淘汰失败不阻断写入：容量是治理，不是准入闸门。
    cap = int(getattr(cfg.sync, "l2_max_items", 0) or 0)
    if cap > 0:
        try:
            demoted = _l2_enforce_capacity(table, cfg, agent, cap)
            if demoted:
                base["demoted"] = demoted
        except Exception as e:  # noqa: BLE001
            base["demote_error"] = str(e)[:200]

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
    # An absolute section is refused, not quietly made relative.
    #
    # Splitting "/a/b" on "/" and dropping the empty first component turned it
    # into "a/b" — inside the vault, so containment passed and the note was
    # written to a directory the caller never named. Exactly what
    # :func:`safe_vault_path` says must not happen: a refused write is
    # recoverable, a silently relocated one is not.
    #
    # This went unnoticed because on Windows the mistake is caught by accident —
    # the drive letter's colon trips the alternate-data-stream rule below — so
    # only a POSIX runner (where "/tmp/..." has no colon) exposed it.
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise VaultPathEscape(
            f"refused: '{section}' is an absolute path; --section must name a "
            f"directory inside the vault")
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


#: 与 ``plugin._kb._status_for_section`` 同一规则的 CLI 拷贝（CLI 不能依赖
#: 插件加载，插件也不能反向 import CLI）。``tests/test_kb_governance.py``
#: 钉住两侧逐 section 一致 —— 状态字段的「一个存储一个答案」。
def _status_for_section(section: str) -> str:
    if section in ("inbox", "knowledge"):
        return "draft"
    if section == "archive":
        return "archived"
    return "curated"


#: 包含性查重的最小正文长度（归一化后）。与 ``plugin._kb._DEDUP_MIN_CHARS``
#: 同值；低于它的正文（"好的""收到"）互相包含是常态而不是重复。
_DEDUP_MIN_CHARS = 120


def _normalize_for_dedup(text: str) -> str:
    """查重用的正文归一化：小写 + 折叠全部空白（与插件侧同实现）。"""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


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

    # 包含性查重（跨标题、跨区）：新正文与既有笔记互相包含且都不琐碎 →
    # 拒绝并给出既有路径，而不是静默新建第二份。同一篇（同路径）不算重复。
    # 语义与插件侧 KnowledgeBase.add 一致（tests/test_kb_governance.py 钉住）。
    norm_new = _normalize_for_dedup(body)
    if len(norm_new) >= _DEDUP_MIN_CHARS:
        for other in iter_notes(config):
            if other == path:
                continue
            try:
                other_meta, other_body = parse_frontmatter(
                    other.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
            norm_old = _normalize_for_dedup(other_body)
            if len(norm_old) < _DEDUP_MIN_CHARS:
                continue
            if norm_new in norm_old or norm_old in norm_new:
                return _refusal(
                    "body duplicates existing note "
                    f"'{other_meta.get('title') or other.stem}'",
                    path=str(other),
                    hint=("update the existing note instead (same title), "
                          "or rewrite to add new information"))

    # Auto-link concepts, once per concept, without duplicating an existing line.
    missing = [c for c in (concepts or [])
               if c and f"[[{c}]]" not in body]
    if missing:
        body = body.rstrip() + "\n\nRelated: " + " ".join(f"[[{c}]]" for c in missing)

    # `agent` is the writer; `source` describes how the note got here. Keeping
    # them separate is what makes the legacy vault values ("l3", "wiki-backup")
    # interpretable instead of being mistaken for authors.
    meta = {"title": title.strip(), "type": "note",
            "status": _status_for_section(section), "tags": tags,
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


def cmd_kb_govern(config: dict, apply: bool = False) -> dict:
    """KB 治理巡检：晋升建议 + 自动归档（双库分工的淘汰闭环）。

    默认 dry-run —— 批量移动笔记之前必须先看到 scope（Runtime Safety：
    batch 操作先报数量）。``--apply`` 才落盘。

    配置从 CLI 的原始 dict 桥接到插件 dataclass（同一份
    ``governed_memory.json``，两个入口不能给出两种答案）。失败回 JSON
    verdict 而不是 traceback：调用方是 agent/脚本，裸堆栈不是它能处理的。
    """
    try:
        cfg_mod = plugin_module("_config")
        kb_mod = plugin_module("_kb")
        cfg = cfg_mod.GovernedMemoryConfig()
        cfg_mod._apply_dict_to_config(cfg, config or {})
        # ``l2_db_path`` 必须与插件同口径（``load_governed_config`` 会把它派生成
        # ``$HERMES_HOME/memory/l2``，KB 使用度库就落在它旁边）。手工构造 dataclass
        # 时该键为空 ⇒ ``Path("").parent`` == ``"."`` ⇒ usage 库解析到**进程 CWD**：
        # CLI 因此永远读到空表（``promote_hits`` 永不触发），还会在仓库根留下
        # 一个 0 行的 kb_usage.db。只在**为空**时补默认 —— 调用方（含沙箱测试）
        # 显式给的路径仍然优先。
        if not str(getattr(cfg, "l2_db_path", "") or "").strip():
            cfg.l2_db_path = str(Path(hermes_home()) / "memory" / "l2")
        return kb_mod.KnowledgeBase(cfg).governance(apply=bool(apply))
    except Exception as e:  # noqa: BLE001 — CLI 必须回 verdict，不能抛栈
        return _refusal(f"kb-govern failed: {type(e).__name__}: {e}",
                        hint="check the vault path (wiki_dir) and that the "
                             "plugin imports cleanly (runtime command shows "
                             "the environment)")


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
    # L2 生命周期计数（Phase 3）：active/superseded/archived 三个状态各自
    # 有多少必须一眼可见 —— 「被撤回的会不会复活」「容量淘汰了多少」不该
    # 靠翻文件回答。文件计数离线可得；active 需要读表，失败时如实标 unknown。
    lifecycle: dict = {}
    try:
        life = plugin_module("_lifecycle")
        lifecycle = life.lifecycle_counts(Path(h) / "memory")
        lifecycle["active"] = "unknown"
        try:
            cfgp = plugin_config()
            lifecycle["max_items"] = int(getattr(cfgp.sync, "l2_max_items", 0) or 0)
            tbl = open_l2_table(cfgp)
            if tbl is not None:
                lifecycle["active"] = int(tbl.to_arrow().num_rows)
        except Exception:  # noqa: BLE001 — 表读不到就保持 unknown，不编数字
            pass
    except Exception as e:  # noqa: BLE001
        lifecycle = {"error": f"{type(e).__name__}: {e}"[:200]}
    out = {
        "hermes_home": h,
        "l1_memory_md": bool(l1["memory_rules_md"]),
        "l1_user_md": bool(l1["user_profile_md"]),
        "l4_persona_md": bool(l1["persona_md"]),
        "l2_dir_exists": l2.exists(),
        "l3_db_exists": l3.exists(),
        "l2_lifecycle": lifecycle,
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


# -- audio transcription ----------------------------------------------------
def cmd_transcribe(config: dict, path: str, timeout: "float | None" = None,
                   chunk_minutes: "float | None" = None) -> dict:
    """Transcribe a local audio file to text (long recordings split automatically).

    Delegates to ``_ingest.transcribe_audio_auto`` through the plugin's own
    dataclass config, so the CLI and the plugin resolve the same ASR settings.
    ``timeout`` and ``chunk_minutes`` override ``config.asr`` only when given.
    The result is returned unchanged — including a partial/failed envelope, so a
    caller can still see the text that *was* obtained.
    """
    cfg = plugin_config()
    if timeout is not None:
        cfg.asr.timeout_seconds = timeout
    if chunk_minutes is not None:
        cfg.asr.chunk_minutes = chunk_minutes
    return plugin_module("_ingest").transcribe_audio_auto(path, cfg)


#: Commands whose answer depends on L2, and therefore on LanceDB being
#: importable. Only these pay the cost of a possible interpreter re-exec; the
#: read-only Markdown commands stay fast on a bare interpreter.
_L2_COMMANDS = {"recall", "remember", "agents", "health", "runtime", "snapshot",
                "kb-govern", "retract", "eval"}


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
    common.add_argument(
        "--envelope", action="store_true",
        help="Wrap stdout as {ok, data, meta} — the machine contract for "
             "agent/script callers (exit codes unchanged)")

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

    kgo = sub.add_parser("kb-govern", parents=[common],
                         help="KB 治理巡检：晋升建议 + 自动归档（默认 dry-run）")
    kgo.add_argument("--apply", action="store_true",
                     help="Actually flag promote_suggest / move to archive "
                          "(default: report only — batch moves must show "
                          "scope first)")
    kgo.add_argument("--yes", action="store_true",
                     help="Confirm the destructive --apply (exit 10 without it)")

    tr = sub.add_parser("transcribe", parents=[common],
                        help="Transcribe a local audio file to text "
                             "(long recordings are split automatically)")
    tr.add_argument("path")
    tr.add_argument("--timeout", type=float, default=None,
                    help="Per-request ASR timeout in seconds "
                         "(default: config asr.timeout_seconds, else 600)")
    tr.add_argument("--chunk-minutes", type=float, default=None,
                    help="Split audio longer than this many minutes "
                         "(default: config asr.chunk_minutes, else 10)")

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

    rt = sub.add_parser("retract", parents=[common],
                        help="Retract L2 facts (supersede): archive + tombstone")
    rt.add_argument("match", help="Substring of the fact(s) to retract "
                                  "(case-insensitive)")
    rt.add_argument("--reason", default="",
                    help="Why it is retracted (recorded in the archive)")
    rt.add_argument("--dry-run", action="store_true",
                    help="List matches without archiving or deleting anything")
    rt.add_argument("--yes", action="store_true",
                    help="Confirm the destructive retract (exit 10 without it)")

    ev = sub.add_parser("eval", parents=[common],
                        help="Retrieval eval: hit@k over a QA set "
                             "(default: self-QA from the L2 snapshot)")
    ev.add_argument("--qa", default="",
                    help='Path to a JSON QA file [{"q": ..., "expect": ...}]')
    ev.add_argument("--top-k", type=int, default=5)
    ev.add_argument("--min-hit", type=float, default=0.0,
                    help="Fail (ok:false, non-zero exit) when hit@k is below "
                         "this. Without a floor the number can never fail.")
    ev.add_argument("--seed", type=int, default=0,
                    help="Sampling seed for the self-QA draw (reproducible)")
    ev.add_argument("--engine", default="cli", choices=("cli", "plugin"),
                    help="Which retrieval path to measure. 'cli' = the CLI's "
                         "own recall_l2 (compatibility path; it does NOT follow "
                         "recall.l2_fusion). 'plugin' = the plugin engine the "
                         "agent actually runs, which does.")

    cz = sub.add_parser("consolidate", parents=[common],
                        help="Merge near-duplicate L2 facts (default dry-run)")
    cz.add_argument("--apply", action="store_true",
                    help="Actually merge clusters: archive originals "
                         "(reason=consolidated) + tombstone + insert merged fact")
    cz.add_argument("--yes", action="store_true",
                    help="Confirm the destructive --apply (exit 10 without it)")

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


def _emit(args, out, agent: str) -> str:
    """序列化一条命令的输出。

    默认 = 历史形状（各命令自己的 JSON）。``--envelope`` 时包一层
    ``{ok, data, meta}`` —— 机器契约：``ok`` 由 ``data`` 的 ok/refused
    推导，``meta`` 带 cmd/agent/ts。退出码语义不变，信封只是形状。
    """
    payload = out
    if getattr(args, "envelope", False):
        ok = not (isinstance(out, dict)
                  and (out.get("ok") is False or out.get("refused")))
        payload = {
            "ok": ok,
            "data": out,
            "meta": {"cmd": getattr(args, "cmd", ""), "agent": agent,
                     "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())},
        }
    return json.dumps(payload, ensure_ascii=False, indent=2)


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

    # 破坏性操作的确认闸（exit 10 = awaiting --yes，WeKnora 退出码矩阵）：
    # retract 的真正执行与 kb-govern --apply 的批量移动，都必须显式确认 ——
    # --dry-run 预览过的 scope 与实际执行之间，需要一个不会滑过去的动作。
    destructive = ((args.cmd == "retract" and not args.dry_run)
                   or (args.cmd == "kb-govern" and getattr(args, "apply", False))
                   or (args.cmd == "consolidate"
                       and getattr(args, "apply", False)))
    if destructive and not getattr(args, "yes", False):
        out = {"ok": False,
               "error": "refusing a destructive action without --yes",
               "hint": "preview with --dry-run, then re-run with --yes"}
        print(_emit(args, out, agent))
        return 10

    if args.cmd == "l1":
        print(_emit(args, cmd_l1(), agent))
        return 0
    if args.cmd == "runtime":
        print(_emit(args, runtime_report(), agent))
        return 0
    if args.cmd == "health":
        out = cmd_health(config)
        out["runtime"] = runtime_report()
        print(_emit(args, out, agent))
        return 0
    if args.cmd == "agents":
        print(_emit(args, cmd_agents(config), agent))
        return 0
    if args.cmd == "snapshot":
        out = cmd_snapshot(config, getattr(args, "out", ""),
                           getattr(args, "max_facts", 0))
        print(_emit(args, out, agent))
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
    elif args.cmd == "kb-govern":
        out = cmd_kb_govern(config, apply=args.apply)
    elif args.cmd == "remember":
        out = cmd_remember(config, args.fact, agent,
                           category=args.category, dry_run=args.dry_run,
                           project=args.project)
    elif args.cmd == "retract":
        out = cmd_retract(config, args.match, reason=args.reason,
                          agent=agent, dry_run=args.dry_run)
    elif args.cmd == "consolidate":
        out = cmd_consolidate(config, apply=args.apply, agent=agent)
    elif args.cmd == "eval":
        out = cmd_eval(config, args.qa, args.top_k,
                       min_hit=args.min_hit, seed=args.seed,
                       engine_kind=args.engine)
    elif args.cmd == "transcribe":
        out = cmd_transcribe(config, args.path,
                             getattr(args, "timeout", None),
                             getattr(args, "chunk_minutes", None))
    else:
        parser.print_help()
        return 2

    print(_emit(args, out, agent))
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
