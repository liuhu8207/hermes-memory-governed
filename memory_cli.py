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
                 lands in ``inbox/`` for review.
* ``remember`` — L2 durable fact. Governance: the shared admission gate
                 (``external_write_verdict``), the same one Hermes applies to
                 its own extractions. Rejections always carry a reason.

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

DEFAULT_HERMES_HOME = r"D:\repos\Drive\项目\github\hermes-home"
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
#: ``D:\repos\Drive\项目\...`` and stripping the ideographs would collapse
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
    bare runtime with no LanceDB. ``search_l2`` swallowed the ImportError and
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
#: Stated on every CLI L2 answer. The keyword channel and the in-plugin vector
#: channel both land on [0, 1] and both are cut by ``recall.l2_min_score``, but
#: they measure different things — one measures meaning, this one measures
#: wording. Leaving that unsaid would let a caller treat a lexical hit as a
#: semantic one, which is the same class of error as the hardcoded 0.9 it
#: replaces.
_L2_RANKING_NOTE = (
    "L2 reached through the CLI is keyword-scored (no embedding model is "
    "loaded here): score = share of the query's distinct terms present in the "
    "fact, on [0,1]. The range matches the in-plugin cosine score so "
    "recall.l2_min_score means the same thing, but the measures are not "
    "interchangeable — this channel is ranked and thresholded, not "
    "vector-ranked.")


def l2_lexical_score(tokens, content: str) -> float:
    """Share of the query's distinct terms that ``content`` contains, on [0, 1].

    Why a real score instead of the hardcoded ``0.9`` this replaced: the score
    is the *only* input the ``recall.l2_min_score`` floor has, so a constant
    turns the floor into a no-op — 0.9 clears every threshold anyone would
    configure, and the CLI would keep presenting unranked, unfiltered keyword
    hits as if they had passed the same gate the plugin applies.

    Why [0, 1]: that is where the in-plugin L2 score lives
    (``_recall.distance_to_score`` maps a cosine distance onto it), so one
    configured floor means "relevant enough" on both channels.
    """
    if not tokens:
        return 0.0
    low = str(content or "").lower()
    covered = sum(1 for t in tokens if t in low)
    return covered / len(tokens)


def _resolve_l2_floor(explicit=None) -> tuple:
    """Resolve the L2 score floor as ``(floor, resolved)``.

    ``resolved`` is ``False`` when the plugin's threshold could not be read, so
    the answer can say "unfiltered" instead of implying a gate that never ran.
    """
    if explicit is not None:
        try:
            return float(explicit), True
        except (TypeError, ValueError):
            return 0.0, False
    try:
        recall = plugin_module("_recall")
        return (recall.layer_score_floor("l2", getattr(plugin_config(), "recall", None)),
                True)
    except Exception:  # noqa: BLE001 — a missing threshold must not blind recall
        return 0.0, False


def _l2_ranking(ranked: bool, floor: float, resolved: bool,
                filtered: int = 0, truncated: int = 0,
                note: str = None) -> dict:
    """Metadata describing *how* the L2 hits were produced.

    Reported separately from the hits because a score without its basis is
    unfalsifiable: a caller cannot tell a filtered list from an unfiltered one,
    nor a lexical 0.8 from a semantic 0.8, unless the answer says so.
    """
    return {
        "ranked": ranked,
        "basis": "keyword-coverage",
        "floor": floor,
        "floor_resolved": resolved,
        "filtered_out": filtered,
        "truncated": truncated,
        "note": _L2_RANKING_NOTE if note is None else note,
    }


def search_l2(query: str, top_k: int, errors: list = None,
              project: str = "", l2_floor: float = None) -> list:
    """Keyword-scan L2 without an embedding model. See :func:`recall_l2`."""
    return recall_l2(query, top_k, errors, project, l2_floor)["hits"]


def recall_l2(query: str, top_k: int, errors: list = None,
              project: str = "", l2_floor: float = None) -> dict:
    """Keyword-scan L2 without an embedding model, ranked and thresholded.

    Returns ``{"hits": [...], "ranking": {...}}``. Hits are ordered by
    relevance and every one of them is at or above the L2 floor.

    Two things the previous version got wrong, both of which made the shared
    entry point the weakest implementation of the same query:

    * it emitted a hardcoded ``score`` of 0.9, so the calibrated L2 threshold
      (0.76 — measured: relevant 8/8 at 0.8458~0.9095, irrelevant 8/8 at
      0.6755~0.7515) could never reject anything coming through the CLI;
    * it stopped at the first ``top_k`` rows in *write order*, so the cut was
      arbitrary — the facts that happened to be written first won, and a
      caller had no way to know anything had been dropped.

    ``errors`` is an optional sink. L2 used to fail silently — an ImportError
    from a bare interpreter produced the same ``[]`` as a genuine no-match, and
    a caller could not tell "nothing remembered" from "cannot read memory".
    Every failure path now records why.

    ``project`` narrows the result to one project *plus* the global facts that
    belong to no project in particular.

    ``l2_floor`` overrides the configured threshold; ``None`` reads
    ``recall.l2_min_score`` through :func:`_recall.layer_score_floor`, the same
    function the in-plugin path uses.
    """
    floor, resolved = _resolve_l2_floor(l2_floor)

    def _fail(msg: str) -> dict:
        if errors is not None:
            errors.append(f"l2: {msg}")
        # "ranked: false" here means exactly that: nothing was ranked because
        # nothing could be read. Saying otherwise would let a caller read an
        # empty L2 as "searched and found nothing relevant".
        return {"hits": [],
                "ranking": _l2_ranking(False, floor, resolved,
                                       note="L2 could not be read, so no hits "
                                            "were ranked or filtered; see the "
                                            "'degraded' entry for the reason.")}

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
        # Distinct, lowercased: repeating a term in the query must not inflate
        # the coverage denominator, and "the THE The" is one term, not three.
        tokens = list(dict.fromkeys(
            t.lower() for t in re.findall(r"[\w\u4e00-\u9fff]+", query or "")))
        if not tokens:
            return {"hits": [], "ranking": _l2_ranking(True, floor, resolved)}
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
            score = l2_lexical_score(tokens, c)
            if score <= 0:
                continue
            if score < floor:
                filtered += 1
                continue
            hit = {"layer": "l2", "content": c[:600],
                   "score": round(score, 4),
                   "score_basis": "keyword-coverage"}
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
                "ranking": _l2_ranking(True, floor, resolved, filtered,
                                       len(hits) - len(kept))}
    except Exception as e:  # noqa: BLE001
        return _fail(str(e)[:200])



# -- L2 governed write ------------------------------------------------------
def plugin_config():
    """Load the same config object the plugin uses (single source of truth)."""
    return plugin_module("_config").load_governed_config(hermes_home())


def open_l2_table(cfg):
    """Open the L2 table, backfilling the sharing columns on a legacy table.

    The column list is *not* written here. It comes from
    ``_sync.L2_PROVENANCE_COLUMNS`` — the same constant that builds the
    create-table schema and the plugin's own backfill. This is the exact shape
    of an incident that already shipped: ``project`` was added to a backfill
    list but not to the schema, so old tables gained the column while new ones
    never had it, and every test stayed green because they all ran against a
    table that already had it. One definition, three consumers.
    """
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(cfg.l2_db_path)
    names = getattr(db, "list_tables", None)
    listing = names() if callable(names) else db.table_names()
    if not isinstance(listing, (list, tuple)):
        listing = getattr(listing, "tables", []) or [listing]
    table_names = [t.name if hasattr(t, "name") else str(t) for t in listing]
    if "memories" not in table_names:
        return None
    table = db.open_table("memories")
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

    The gate is the shared one (``external_write_verdict``) — the same rules
    Hermes applies to its own extractions, so an agent cannot write something
    Hermes itself would have rejected. A rejection always names its reason;
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
        base.update({
            "ok": False,
            "reason": reason,
            "hint": ("rewrite it as a concrete statement (name the host, path, "
                     "version or constraint), or use kb-add to file it as a note"),
        })
        return base

    if dry_run:
        base.update({"ok": True, "dry_run": True,
                     "content": cleaned[:600], "would_write": "l2"})
        return base

    cfg = plugin_config()
    embedding = plugin_module("_embedding").EmbeddingService.get(cfg)
    if not embedding.available:
        # Refusing loudly beats writing a row nothing can ever recall.
        base.update({
            "ok": False,
            "reason": "embedding_unavailable",
            "detail": embedding.last_error or "embedding backend reported unavailable",
            "hint": "use kb-add instead — notes do not need an embedding to be found",
        })
        return base

    vector = embedding.embed_one(cleaned)
    if vector is None:
        base.update({"ok": False, "reason": "embedding_failed",
                     "detail": embedding.last_error})
        return base

    table = open_l2_table(cfg)
    if table is None:
        base.update({"ok": False, "reason": "l2_table_missing",
                     "detail": "no 'memories' table in " + str(cfg.l2_db_path)})
        return base

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
        # search_l2 relies on the difference to decide what a project-scoped
        # query is allowed to see.
        "project": resolved_project or None,
    }
    try:
        table.add([row])
    except Exception as e:  # noqa: BLE001 — surface the failure, never mask it
        base.update({"ok": False, "reason": "write_failed", "detail": str(e)[:300],
                     "vector_dim": dim})
        return base

    base.update({"ok": True, "content": cleaned[:600], "vector_dim": dim,
                 "embedding_backend": embedding.backend_name})
    return base


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

    ``errors`` is an optional sink: a ``--section`` that points outside the
    vault is skipped and named there, because returning "no such notes" for a
    query that was refused reads as an empty vault rather than a rejected
    request.
    """
    vault = wiki_dir(config)
    roots = [vault] if subdirs is None else []
    for s in (subdirs or []):
        try:
            roots.append(safe_vault_path(vault, s))
        except VaultPathEscape as e:
            if errors is not None:
                errors.append(str(e))
    out = []
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.md")):
            if not p.is_file():
                continue
            try:
                rel = p.relative_to(vault)
            except ValueError:
                out.append(p)
                continue
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
    target = slugify(title).lower()
    for p in iter_notes(config):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        if (meta.get("title") or p.stem).lower() == target or p.stem.lower() == target:
            return {"title": meta.get("title") or p.stem, "path": str(p),
                    "meta": meta, "body": body}
    return None


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
            return {"ok": False, "error": "refused: body looks like it contains a secret"}

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
        subdir = safe_vault_path(vault, section)
        path = safe_vault_path(subdir, slugify(title) + ".md")
    except VaultPathEscape as e:
        return {"ok": False, "error": str(e),
                "hint": ("--section must name a directory inside the vault; "
                         "use kb-add with a plain section name such as 'notes'")}

    subdir.mkdir(parents=True, exist_ok=True)

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
            return {
                "ok": False,
                "error": f"note already exists and is owned by '{owner}'",
                "path": str(path),
                "hint": "choose a different title, or pass --overwrite to take it over",
            }
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

    # Atomic write: one agent must never observe a half-written note.
    fd, tmp = tempfile.mkstemp(dir=str(subdir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"ok": True, "path": str(path), "section": section,
            "title": title.strip(), "agent": agent, "updated": existed}



# -- health ----------------------------------------------------------------
def cmd_health(config: dict):
    h = hermes_home()
    l2 = Path(h) / "memory" / "l2"
    l3 = Path(h) / "memory" / "l3" / "l3.db"
    l1 = cmd_l1()
    count_notes = len(iter_notes(config))
    return {
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


#: Commands whose answer depends on L2, and therefore on LanceDB being
#: importable. Only these pay the cost of a possible interpreter re-exec; the
#: read-only Markdown commands stay fast on a bare interpreter.
_L2_COMMANDS = {"recall", "remember", "agents", "health", "runtime"}


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

    if args.cmd == "recall":
        degraded: list = []
        scoped = normalize_project(getattr(args, "project", "") or "")
        l2 = recall_l2(args.query, args.top_k, degraded, project=scoped)
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
        out = cmd_kb_get(config, args.title) or {"error": f"note not found: {args.title}"}
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
