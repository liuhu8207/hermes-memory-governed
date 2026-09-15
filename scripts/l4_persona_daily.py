#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate L4 persona.md from L1 notes, L2 memories, and L3 archive.

Reads L1 MEMORY.md + USER.md, L2 LanceDB facts, L3 SQLite conversations,
and generates a compact persona.json + persona.md suitable for system prompt injection.

L2 is read through **pyarrow** (a hard LanceDB dependency) rather than pandas:
the minimal deployment runtime (``hermes-agent/venv``) ships ``lancedb`` /
``pyarrow`` / ``numpy`` but **not** pandas, so the previous ``table.to_pandas()``
call raised ``ModuleNotFoundError`` that was silently swallowed and left L2
facts permanently empty (the P1 under-count defect).
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("l4_persona")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
PERSONA_FILE = env_path("PERSONA_PATH", MEMORY_DIR / "persona.md")
PERSONA_META = env_path("PERSONA_META_PATH", MEMORY_DIR / "persona_meta.json")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")

SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9_-]{15,}", "[REDACTED]"),
    (r"gh[pousr]_[A-Za-z0-9_]{15,}", "[REDACTED]"),
    (r"(?i)(api[_-]?key|app[_-]?secret|token|password)\s*[:=]\s*[^\s,;]+", r"\1=[REDACTED]"),
]

#: Columns surfaced to the persona.  ``vector`` (a 1024-dim embedding) is
#: deliberately excluded so the read does not materialise every embedding.
_L2_COLUMNS: tuple[str, ...] = ("content", "category", "source", "timestamp")


def _env_int(key: str, default: int) -> int:
    """Read a positive int from the environment, falling back on error."""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring invalid %s=%r; using %d", key, raw, default)
        return default


#: Row caps / windows (overridable via env or the function parameters).
#: ``L2_FACTS_LIMIT_DEFAULT`` covers the current L2 scale with head-room;
#: ``L3_RECENT_MAX_DEFAULT`` plus ``L3_RECENT_DAYS_DEFAULT`` bound the L3 read
#: so growth never turns into an unbounded full-table scan.
L2_FACTS_LIMIT_DEFAULT = _env_int("PERSONA_L2_FACTS_LIMIT", 1000)
L3_RECENT_DAYS_DEFAULT = _env_int("PERSONA_L3_RECENT_DAYS", 90)
L3_RECENT_MAX_DEFAULT = _env_int("PERSONA_L3_RECENT_MAX", 2000)

#: How many L2 facts are rendered verbatim into the persona, and the per-fact
#: character cap — keeps the injected persona small even as L2 grows.
PERSONA_MAX_FACTS_DEFAULT = _env_int("PERSONA_MAX_FACTS", 20)
PERSONA_FACT_CHARS_DEFAULT = _env_int("PERSONA_FACT_CHARS", 240)

#: L3 roles that count as a "conversation".  Mirrors the recall-path whitelist
#: (``plugin/memory_governed/_recall.py`` ``L3_RECALL_ROLES``): ``tool`` rows are
#: process/terminal dumps rather than dialogue, so they are excluded from
#: persona synthesis to avoid injecting noise.
L3_RECALL_ROLES: tuple[str, ...] = ("user", "assistant")


def redact(text: str) -> str:
    for pattern, replacement in SECRET_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def load_l1() -> dict:
    """Load L1 handwritten files."""
    result = {"memory": "", "user": ""}

    memory_path = MEMORY_DIR / "MEMORY.md"
    if memory_path.exists():
        result["memory"] = redact(memory_path.read_text(encoding="utf-8", errors="replace"))

    user_path = MEMORY_DIR / "USER.md"
    if user_path.exists():
        result["user"] = redact(user_path.read_text(encoding="utf-8", errors="replace"))

    return result


def _table_records(table: Any, limit: int) -> list[dict]:
    """Read up to *limit* rows from a LanceDB table without requiring pandas.

    Prefers ``table.head(n)`` (a bounded read) followed by a pyarrow column
    projection: pyarrow is a hard LanceDB dependency, whereas pandas is absent
    from the minimal deployment runtime.  pandas is only a last-resort fallback
    for LanceDB builds that do not expose ``head``.
    """
    if limit <= 0:
        return []

    head = getattr(table, "head", None)
    if callable(head):
        arrow = head(limit)
        names = set(getattr(getattr(arrow, "schema", None), "names", []) or [])
        keep = [c for c in _L2_COLUMNS if c in names]
        if keep:
            arrow = arrow.select(keep)
        return arrow.to_pylist()

    # Legacy fallback for LanceDB builds without ``head()`` (needs pandas).
    rows = table.to_pandas().to_dict("records")
    return rows[:limit]


def load_l2_facts(limit: int = L2_FACTS_LIMIT_DEFAULT) -> list[dict]:
    """Load facts from L2 LanceDB (if available).

    Reads via pyarrow so the path keeps working in runtimes that ship
    ``lancedb``/``pyarrow`` but not pandas.  Returns ``[]`` (with a warning,
    never silently) when the store is unavailable or unreadable.
    """
    l2_dir = MEMORY_DIR / "l2"
    if not l2_dir.exists():
        logger.debug("L2 directory missing: %s", l2_dir)
        return []

    try:
        import lancedb
    except ImportError:
        logger.warning("lancedb not installed; skipping L2 facts", exc_info=True)
        return []

    try:
        db = lancedb.connect(str(l2_dir))
        tables = getattr(db.list_tables(), "tables", [])
        if "memories" not in tables:
            logger.warning("L2 table 'memories' not found (have %s)", tables)
            return []
        table = db.open_table("memories")
        return _table_records(table, limit)
    except Exception:
        logger.warning("Failed to read L2 facts from %s", l2_dir, exc_info=True)
        return []


def _l3_connection():
    """Open a read-only L3 connection that has a ``messages`` table, or ``None``.

    All failures are logged (never swallowed) so a broken/unreadable L3 store is
    visible rather than silently yielding empty results.
    """
    if not L3_DB_PATH.exists():
        logger.debug("L3 database missing: %s", L3_DB_PATH)
        return None

    try:
        conn = sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True)
    except Exception:
        logger.warning("Cannot open L3 database %s", L3_DB_PATH, exc_info=True)
        return None

    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
    except Exception:
        conn.close()
        logger.warning("Failed to enumerate L3 tables in %s", L3_DB_PATH, exc_info=True)
        return None

    if "messages" not in tables:
        conn.close()
        logger.warning("L3 table 'messages' not found (have %s)", tables)
        return None

    return conn


def load_l3_recent(
    limit: int = L3_RECENT_MAX_DEFAULT,
    days: int | None = L3_RECENT_DAYS_DEFAULT,
    roles: tuple[str, ...] | None = L3_RECALL_ROLES,
) -> list[str]:
    """Load recent conversation messages from L3.

    ``limit`` caps how many rows are read (guards against unbounded growth),
    ``days`` optionally restricts to messages newer than that many days
    (``None`` disables the time window), and ``roles`` restricts to dialogue
    roles (default: ``user``/``assistant`` — ``tool`` dumps are excluded; pass
    ``roles=None`` to include every role).  L3 stores ``timestamp`` as an epoch
    float, so the window is compared against ``time.time()``.
    """
    conn = _l3_connection()
    if conn is None:
        return []

    try:
        where_parts: list[str] = []
        params: list[Any] = []

        if roles:
            placeholders = ",".join("?" for _ in roles)
            where_parts.append(f"role IN ({placeholders})")
            params.extend(roles)
        if days is not None:
            where_parts.append("timestamp >= ?")
            params.append(time.time() - days * 86400)

        where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        params.append(limit)

        rows = conn.execute(
            f"SELECT content, role FROM messages {where} "
            f"ORDER BY timestamp DESC LIMIT ?",
            tuple(params),
        ).fetchall()

        return [redact(r[0]) for r in rows if r[0]]
    except Exception:
        logger.warning("Failed to read L3 conversations from %s", L3_DB_PATH, exc_info=True)
        return []
    finally:
        conn.close()


def count_l3_messages() -> int:
    """Total messages in the L3 archive (every role, no cap or window)."""
    conn = _l3_connection()
    if conn is None:
        return 0
    try:
        return int(conn.execute("SELECT count(*) FROM messages").fetchone()[0])
    except Exception:
        logger.warning("Failed to count L3 messages in %s", L3_DB_PATH, exc_info=True)
        return 0
    finally:
        conn.close()


def generate_persona(
    l1: dict,
    l2: list,
    l3: list,
    max_facts: int = PERSONA_MAX_FACTS_DEFAULT,
    fact_chars: int = PERSONA_FACT_CHARS_DEFAULT,
    message_count_total: int = 0,
) -> dict:
    """Generate persona dict from L1 + L2 + L3.

    ``conversation_count`` reflects the filtered dialogue messages in *l3*,
    while ``message_count_total`` (when > 0) records the full L3 archive size
    so the raw archive scale is not lost when ``tool`` rows are excluded.
    """
    persona = {
        "language": "Chinese/English",
        "timezone": "Asia/Shanghai",
        "generated_at": datetime.now().isoformat(),
        "source": "hermes-memory-governed",
    }

    # Extract from L1
    if l1.get("user"):
        lines = l1["user"].strip().split("\n")
        persona["user_summary"] = "\n".join(lines[:20])

    if l1.get("memory"):
        lines = l1["memory"].strip().split("\n")
        persona["rules_summary"] = "\n".join(lines[:20])

    # Extract from L2 — category counts plus a bounded, redacted sample of the
    # actual fact text so the persona carries real L2 content, not just an echo
    # of the handwritten L1 files.
    if l2:
        categories: dict[str, int] = {}
        key_facts: list[str] = []
        seen: set[str] = set()
        for fact in l2:
            cat = fact.get("category") or "other"
            categories[cat] = categories.get(cat, 0) + 1

            content = (fact.get("content") or "").strip()
            if not content or content in seen:
                continue
            seen.add(content)
            if len(key_facts) < max_facts:
                key_facts.append(redact(content)[:fact_chars])

        persona["fact_categories"] = categories
        if key_facts:
            persona["key_facts"] = key_facts

    # Extract from L3 — dialogue count (filtered) plus the raw archive size.
    if l3 or message_count_total:
        persona["conversation_count"] = len(l3)
    if message_count_total:
        persona["message_count_total"] = message_count_total

    return persona


def write_persona(persona: dict) -> None:
    """Write persona.json and persona.md."""
    # JSON (structured)
    PERSONA_META.write_text(
        json.dumps(persona, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Markdown (human-readable, for system prompt)
    lines = [
        f"# User Profile",
        f"_Generated: {persona.get('generated_at', '?')}_",
        "",
    ]

    if persona.get("user_summary"):
        lines.extend(["## User", persona["user_summary"], ""])

    if persona.get("rules_summary"):
        lines.extend(["## Rules & Preferences", persona["rules_summary"], ""])

    if persona.get("fact_categories"):
        lines.extend(["## Knowledge Areas"])
        for cat, count in persona["fact_categories"].items():
            lines.append(f"- {cat}: {count} facts")
        lines.append("")

    if persona.get("key_facts"):
        lines.extend(["## Known Facts"])
        for fact in persona["key_facts"]:
            lines.append(f"- {fact}")
        lines.append("")

    conversations = persona.get("conversation_count")
    total_messages = persona.get("message_count_total")
    if conversations is not None or total_messages:
        lines.append("## Stats")
        if conversations is not None:
            lines.append(f"- Conversations archived: {conversations}")
        if total_messages:
            lines.append(f"- Messages archived (all roles): {total_messages}")
        lines.append("")

    PERSONA_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Persona written to %s", PERSONA_FILE)


def main():
    logger.info("Loading L1...")
    l1 = load_l1()

    logger.info("Loading L2 facts...")
    l2 = load_l2_facts()

    logger.info("Loading L3 recent conversations...")
    l3 = load_l3_recent()
    l3_total = count_l3_messages()

    logger.info("Generating persona from L1=%d L2=%d L3=%d (L3 total=%d)",
                len(l1.get("memory", "")) + len(l1.get("user", "")),
                len(l2), len(l3), l3_total)

    persona = generate_persona(l1, l2, l3, message_count_total=l3_total)
    write_persona(persona)

    logger.info("L4 persona generation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
