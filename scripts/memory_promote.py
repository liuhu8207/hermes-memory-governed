#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory promotion: surface recurring L3 content as Bridge candidates.

Inspired by OpenSquilla's memory promotion mechanism. Scans L3 user
messages, groups them by normalized content, and promotes patterns that
recur >= N times into the Bridge review queue (tags: auto-promoted +
review-required — still gated by human review, never auto-landed in L1).

Usage:
    python scripts/memory_promote.py --dry-run
    python scripts/memory_promote.py --min-count 3 --days 90
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("memory_promote")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
BRIDGE_DIR = env_path("BRIDGE_DIR", HERMES_HOME / "cron" / "output" / "scope_recall_bridge")


def normalize(text: str) -> str:
    """Normalize content for grouping: lowercase, collapse whitespace, strip punctuation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[。，！？、：；\"'（）()【】\[\]{}.,!?;:]", "", text)
    return text


def load_user_messages(days: int) -> list[dict[str, Any]]:
    if not L3_DB_PATH.exists():
        logger.warning("L3 database not found: %s", L3_DB_PATH)
        return []
    cutoff = time.time() - days * 86400
    conn = sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True)
    # NOTE: no GROUP BY here — recurrence count IS the promotion signal
    rows = conn.execute(
        """SELECT content, timestamp FROM messages
           WHERE role = 'user' AND content != '' AND timestamp > ?""",
        (cutoff,),
    ).fetchall()
    conn.close()
    return [{"content": r[0], "last_seen": r[1]} for r in rows]


def find_promotable(messages: list[dict[str, Any]], min_count: int, min_len: int) -> list[dict[str, Any]]:
    """Group by normalized content; return patterns recurring >= min_count."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for msg in messages:
        if len(msg["content"]) < min_len:
            continue
        key = normalize(msg["content"])
        if len(key) < 8:  # too generic after normalization
            continue
        groups.setdefault(key, []).append(msg)

    promoted = []
    for key, items in groups.items():
        if len(items) < min_count:
            continue
        representative = items[0]["content"]
        promoted.append({
            "content": representative[:600],
            "target": "memory",
            "memory_type": "memory",
            "tags": ["auto-promoted", "review-required"],
            "source": "hermes-memory-governed",
            "source_path": f"l3:recurred {len(items)}x",
            "metadata": {
                "recurrence_count": len(items),
                "last_seen": datetime.fromtimestamp(items[0]["last_seen"]).isoformat(),
            },
        })
    promoted.sort(key=lambda x: x["metadata"]["recurrence_count"], reverse=True)
    return promoted


def _bridge_exporter():
    """Locate the BridgeExporter (repo dev layout or HERMES_HOME installed layout)."""
    project_root = Path(__file__).resolve().parent.parent
    candidates = []
    if (project_root / "plugin" / "memory_governed").exists():
        candidates.append(str(project_root))  # dev: import plugin.memory_governed._bridge
    installed_parent = HERMES_HOME / "plugins"
    if (installed_parent / "governed").exists():
        candidates.append(str(installed_parent))  # installed: import governed._bridge
    last_error = None
    for cand in candidates:
        if cand in sys.path:
            sys.path.remove(cand)
        sys.path.insert(0, cand)
        try:
            try:
                from plugin.memory_governed._bridge import BridgeExporter  # dev layout
            except ImportError:
                from governed._bridge import BridgeExporter  # installed layout
            return BridgeExporter
        except ImportError as e:
            last_error = e
    raise ModuleNotFoundError(f"BridgeExporter not found: {last_error}")


def export_promoted(promoted: list[dict[str, Any]]) -> dict[str, int]:
    """Append promoted candidates to the Bridge JSONL (dedup + archive intact)."""
    BridgeExporter = _bridge_exporter()
    cfg_dir = BRIDGE_DIR
    exporter = BridgeExporter(type("C", (), {"bridge_dir": str(cfg_dir)}))
    return exporter.export_candidates(promoted)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote recurring L3 patterns to Bridge")
    parser.add_argument("--min-count", type=int, default=3, help="Min occurrences to promote")
    parser.add_argument("--min-len", type=int, default=10, help="Min message length")
    parser.add_argument("--days", type=int, default=90, help="Scan window in days")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing Bridge")
    args = parser.parse_args()

    messages = load_user_messages(args.days)
    logger.info("loaded %d user messages from last %d days", len(messages), args.days)

    promoted = find_promotable(messages, args.min_count, args.min_len)
    if not promoted:
        print("no recurring patterns found — nothing to promote")
        return 0

    print(f"found {len(promoted)} recurring patterns (>= {args.min_count}x):")
    for p in promoted[:20]:
        print(f"  [{p['metadata']['recurrence_count']}x] {p['content'][:80]}")

    if args.dry_run:
        print("(dry-run — nothing written)")
        return 0

    stats = export_promoted(promoted)
    print(f"Bridge: written={stats['written']}, skipped(dup)={stats['skipped']}")
    logger.info("promoted %d patterns to Bridge", stats["written"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
