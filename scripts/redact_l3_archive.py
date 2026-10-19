#!/usr/bin/env python3
"""Redact secrets and inline image payloads from the existing L3 archive.

L3 is the raw conversation archive, but it must not remain a second credential
store. New writes are redacted by ``L3Writer``; this script applies the same
function to rows that predate that change.

Usage:
    python scripts/redact_l3_archive.py --dry-run
    python scripts/redact_l3_archive.py --yes
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, env_path

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
BACKUP_DIR = env_path("L3_BACKUP_DIR", MEMORY_DIR / "l3_backups")

_IMAGE_DATA_RE = re.compile(
    r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=\r\n]+",
    re.IGNORECASE,
)


def _content_hash(session_id: str, role: str, content: str) -> str:
    return hashlib.sha256(
        (str(session_id) + "|" + str(role) + "|" + content).encode("utf-8")
    ).hexdigest()


def _backup(db_path: Path) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"l3_pre_redact_{stamp}.db"
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the standalone FTS table with rowid = messages.id.

    The writer inserts the two tables in lockstep, but rebuilding is the only
    way to guarantee that a redacted body and its searchable mirror agree even
    if an older row was inserted or deleted out of band.
    """
    conn.execute("DELETE FROM messages_fts")
    conn.execute(
        "INSERT INTO messages_fts(rowid, content, role, session_id, timestamp) "
        "SELECT id, content, role, session_id, timestamp FROM messages"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Redact L3 archive secrets")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    if not L3_DB_PATH.exists():
        print(f"L3 database not found: {L3_DB_PATH}")
        return 1

    from plugin.memory_governed._bridge import matched_secret_patterns
    from plugin.memory_governed._sync import redact_for_archive

    conn = sqlite3.connect(str(L3_DB_PATH))
    try:
        rows = conn.execute(
            "SELECT id, session_id, role, content FROM messages ORDER BY id"
        ).fetchall()
        changes = []
        image_hits = 0
        pattern_counts: dict[str, int] = {}
        for rowid, session_id, role, content in rows:
            original = content or ""
            redacted = redact_for_archive(original)
            if redacted == original:
                continue
            names = matched_secret_patterns(original)
            for name in names:
                pattern_counts[name] = pattern_counts.get(name, 0) + 1
            if _IMAGE_DATA_RE.search(original):
                image_hits += 1
            changes.append((rowid, session_id, role, original, redacted))

        print(f"L3: {L3_DB_PATH}")
        print(f"rows: {len(rows)}  changed: {len(changes)}")
        print(f"secret patterns: {pattern_counts or '{}'}")
        print(f"inline image payloads: {image_hits}")

        if args.dry_run or not changes:
            return 0

        if not args.yes:
            answer = input(f"Redact {len(changes)} rows? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                print("aborted")
                return 1

        backup = _backup(L3_DB_PATH)
        print(f"backup: {backup}")
        with conn:
            for rowid, session_id, role, _original, redacted in changes:
                conn.execute(
                    "UPDATE messages SET content = ?, hash = ? WHERE id = ?",
                    (redacted, _content_hash(session_id, role, redacted), rowid),
                )
            _rebuild_fts(conn)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        print(f"redacted {len(changes)} rows; FTS rebuilt")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
