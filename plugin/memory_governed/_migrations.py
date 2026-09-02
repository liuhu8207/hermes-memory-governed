# -*- coding: utf-8 -*-
"""Schema migrations for the governed memory plugin.

Each migration is (version, name, callable(provider, config)) and must be
IDEMPOTENT (safe to re-run). Applied versions are recorded in
$HERMES_HOME/migrations_state.json so they never run twice.

Historical migrations 001/002 codify changes that were already applied
inline by earlier versions of the code (L3 hash column, L2 source_rowid
column) — they are kept so fresh installs and legacy databases converge
to the same schema through the same mechanism.

Adding a new migration:
    1. Append (version, name, fn) to MIGRATIONS with the next version number.
    2. The fn must be idempotent (check-before-alter).
    3. Call run_pending() from provider.initialize().
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _l3_add_hash_column(provider, config) -> None:
    """Migration 001: L3 messages table needs a hash column for idempotent writes."""
    db_path = Path(config.l3_db_path)
    if not db_path.exists():
        return
    conn = sqlite3.connect(str(db_path))
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "hash" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN hash TEXT")
            conn.commit()
            logger.info("migration 001: added messages.hash column")
    finally:
        conn.close()


def _l2_add_source_rowid_column(provider, config) -> None:
    """Migration 002: L2 memories table needs a source_rowid column for provenance.

    lancedb (and pyarrow) are OPTIONAL dependencies. Their absence is a normal
    deployment mode (L2 disabled, text fallback), so it is logged at INFO —
    only unexpected failures are worth a WARNING.
    """
    l2_dir = Path(config.l2_db_path)
    if not l2_dir.exists():
        return
    try:
        import lancedb

        db = lancedb.connect(str(l2_dir))
        if "memories" not in db.list_tables().tables:
            return
        table = db.open_table("memories")
        names = [f.name for f in table.schema]
        if "source_rowid" in names:
            return
        import pyarrow as pa

        table.add_columns({"source_rowid": pa.array([], type=pa.int64())})
        logger.info("migration 002: added memories.source_rowid column")
    except ImportError as e:
        # Optional dependency missing → expected graceful degradation.
        logger.info("migration 002 skipped (lancedb not installed, L2 disabled): %s", e)
    except Exception as e:
        # Unexpected: the L2 directory exists but could not be migrated.
        logger.warning("migration 002 skipped: %s", e)


def _l3_add_perf_indexes(provider, config) -> None:
    """Migration 003: L3 dedup + retention indexes (idempotent).

    Without these, ``L3Writer.write()``'s dedup lookup
    (``SELECT 1 FROM messages WHERE session_id = ? AND hash = ?``) is a full
    table scan: measured 877 ms for a 50-message turn against 80k rows. With
    ``idx_messages_dedup`` the same write takes 2.6 ms (337x).

    Both indexes are created with ``IF NOT EXISTS`` so this is safe to re-run
    on databases that already have them (e.g. freshly created by L3Writer).
    """
    db_path = Path(config.l3_db_path)
    if not db_path.exists():
        return

    conn = sqlite3.connect(str(db_path))
    try:
        table_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if not table_exists:
            return

        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        if "hash" not in cols:
            # Nothing to index yet — migration 001 adds the column; run again after it.
            logger.info("migration 003 deferred: messages.hash column not present yet")
            return

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_dedup ON messages (session_id, hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (timestamp)"
        )
        conn.commit()
        logger.info("migration 003: ensured idx_messages_dedup / idx_messages_ts")
    finally:
        conn.close()


MIGRATIONS: list[tuple[int, str, Callable[[Any, Any], None]]] = [
    (1, "l3_add_hash_column", _l3_add_hash_column),
    (2, "l2_add_source_rowid_column", _l2_add_source_rowid_column),
    (3, "l3_add_perf_indexes", _l3_add_perf_indexes),
]


class MigrationRunner:
    """Tracks applied migration versions in $HERMES_HOME/migrations_state.json."""

    def __init__(self, hermes_home: Path) -> None:
        self.state_file = Path(hermes_home) / "migrations_state.json"

    def _applied(self) -> set[int]:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                return set(data.get("applied", []))
            except Exception:
                pass
        return set()

    def mark_applied(self, version: int) -> None:
        applied = self._applied()
        applied.add(version)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps({"applied": sorted(applied)}, indent=2), encoding="utf-8"
        )

    def run_pending(self, provider, config) -> list[int]:
        """Run all pending migrations in order. Returns versions applied now."""
        applied = self._applied()
        ran = []
        for version, name, fn in sorted(MIGRATIONS):
            if version in applied:
                continue
            try:
                fn(provider, config)
                self.mark_applied(version)
                ran.append(version)
                logger.info("migration %03d_%s applied", version, name)
            except Exception as e:
                logger.warning("migration %03d_%s failed: %s", version, name, e)
        return ran
