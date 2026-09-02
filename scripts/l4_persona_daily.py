#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate L4 persona.md from L1 notes, L2 memories, and L3 archive.

Reads L1 MEMORY.md + USER.md, L2 LanceDB facts, L3 SQLite conversations,
and generates a compact persona.json + persona.md suitable for system prompt injection.
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


def load_l2_facts(limit: int = 100) -> list[dict]:
    """Load facts from L2 LanceDB (if available)."""
    try:
        import lancedb
        l2_dir = MEMORY_DIR / "l2"
        if not l2_dir.exists():
            return []
        db = lancedb.connect(str(l2_dir))
        if "memories" not in db.list_tables().tables:
            return []
        table = db.open_table("memories")
        rows = table.to_pandas().to_dict("records") if table.count_rows() > 0 else []
        return rows[:limit]
    except Exception:
        return []


def load_l3_recent(limit: int = 50) -> list[str]:
    """Load recent conversations from L3."""
    if not L3_DB_PATH.exists():
        return []

    try:
        conn = sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        if "messages" not in tables:
            conn.close()
            return []

        rows = conn.execute(
            "SELECT content, role FROM messages ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()

        return [redact(r[0]) for r in rows if r[0]]
    except Exception:
        return []


def generate_persona(l1: dict, l2: list, l3: list) -> dict:
    """Generate persona dict from L1 + L2 + L3."""
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

    # Extract from L2
    if l2:
        categories = {}
        for fact in l2:
            cat = fact.get("category", "other")
            categories.setdefault(cat, []).append(fact.get("content", ""))
        persona["fact_categories"] = {k: len(v) for k, v in categories.items()}

    # Extract from L3
    if l3:
        persona["conversation_count"] = len(l3)

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

    if persona.get("conversation_count"):
        lines.append(f"## Stats")
        lines.append(f"- Conversations archived: {persona['conversation_count']}")
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

    logger.info("Generating persona from L1=%d L2=%d L3=%d",
                len(l1.get("memory", "")) + len(l1.get("user", "")),
                len(l2), len(l3))

    persona = generate_persona(l1, l2, l3)
    write_persona(persona)

    logger.info("L4 persona generation complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
