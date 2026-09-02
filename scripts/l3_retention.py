#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3 retention: prune messages older than N days (messages + messages_fts).

Usage:
    python scripts/l3_retention.py --dry-run --days 180
    python scripts/l3_retention.py --days 180 --yes

Guarantees:
- Both tables are pruned inside ONE transaction: either both are pruned or
  neither is, so the FTS index can never diverge from `messages`.
- A divergence (FTS prune impossible while `messages` was pruned) is reported
  through `_diag.log_data_loss` instead of a silent `logger.warning`.
- Runs against a live database: WAL + a 30s busy timeout, so it waits for the
  provider's write lock instead of failing with "database is locked".
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from typing import Optional

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("l3_retention")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")

try:  # soft dependency: the script must also run outside the repo install
    from plugin.memory_governed._diag import log_data_loss, log_degraded
except Exception:  # noqa: BLE001 - fall back to plain logging
    def log_data_loss(component: str, reason: str, *, detail: str = "",
                      exc: Optional[BaseException] = None) -> None:
        logger.error("DATA LOSS — %s: %s | %s | %s", component, reason, detail, exc)

    def log_degraded(component: str, reason: str, *, detail: str = "",
                     exc: Optional[BaseException] = None) -> None:
        logger.warning("degraded — %s: %s | %s | %s", component, reason, detail, exc)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """True when `name` is a table (or virtual table) in this database."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _count_fts_before(conn: sqlite3.Connection, cutoff: float) -> int:
    """Count FTS rows older than the cutoff (FTS stores the timestamp as text)."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE CAST(timestamp AS REAL) < ?",
            (cutoff,),
        ).fetchone()[0]
    except Exception as e:  # noqa: BLE001 - reported, then treated as unknown
        log_degraded("l3_retention", "fts_count_failed", detail=str(L3_DB_PATH), exc=e)
        return 0


def _open_conn() -> sqlite3.Connection:
    """Open the L3 database in WAL mode with a busy timeout."""
    conn = sqlite3.connect(str(L3_DB_PATH), timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
    except Exception as e:  # noqa: BLE001 - WAL is an optimisation, not a requirement
        log_degraded("l3_retention", "wal_unavailable", exc=e)
    return conn


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune old L3 messages")
    parser.add_argument("--days", type=int, default=180, help="Keep messages from last N days")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--dry-run", action="store_true", help="Count only, no deletion")
    args = parser.parse_args()

    if not L3_DB_PATH.exists():
        logger.warning("L3 database not found: %s", L3_DB_PATH)
        return 0

    cutoff = time.time() - args.days * 86400
    conn = _open_conn()

    try:
        if not _table_exists(conn, "messages"):
            logger.warning("L3 database has no messages table: %s", L3_DB_PATH)
            return 0

        n_msgs = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE timestamp < ?", (cutoff,)
        ).fetchone()[0]

        has_fts = _table_exists(conn, "messages_fts")
        n_fts = _count_fts_before(conn, cutoff) if has_fts else 0

        print(f"Messages older than {args.days} days "
              f"(before cutoff {time.strftime('%Y-%m-%d', time.localtime(cutoff))}):")
        print(f"  messages: {n_msgs}")
        print(f"  messages_fts: {n_fts if has_fts else 'n/a (table missing)'}")

        if args.dry_run:
            return 0

        if n_msgs == 0:
            return 0

        if not args.yes:
            answer = input(f"Delete {n_msgs} old messages? [y/N] ")
            if answer.strip().lower() not in ("y", "yes"):
                logger.info("aborted")
                return 1

        if not has_fts:
            log_degraded(
                "l3_retention",
                "fts_table_missing",
                detail=f"{L3_DB_PATH} — pruning messages only, "
                       "full-text search is not available for this database",
            )

        try:
            conn.execute("BEGIN")
            # FTS first: if it fails, nothing is deleted and the two tables
            # cannot diverge.
            if has_fts:
                conn.execute(
                    "DELETE FROM messages_fts WHERE CAST(timestamp AS REAL) < ?",
                    (cutoff,),
                )
            conn.execute("DELETE FROM messages WHERE timestamp < ?", (cutoff,))
            conn.commit()
            logger.info("pruned %d messages", n_msgs)
        except Exception as e:  # noqa: BLE001
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            log_data_loss(
                "l3_retention",
                "prune_failed",
                detail=f"nothing deleted (transaction rolled back); db={L3_DB_PATH}",
                exc=e,
            )
            return 1

        try:
            conn.execute("PRAGMA optimize")
        except Exception:  # noqa: BLE001 - advisory only
            pass
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
