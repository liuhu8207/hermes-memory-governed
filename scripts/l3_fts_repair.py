#!/usr/bin/env python3
"""Check and repair the L3 SQLite FTS5 index.

``PRAGMA integrity_check`` can report ``malformed inverted index for FTS5``
while plain SELECTs still work: the shadow index drifted from the rows. The fix
is a full FTS rebuild (``INSERT INTO messages_fts(messages_fts) VALUES('rebuild')``)
after a byte-level backup.

Usage:
    python scripts/l3_fts_repair.py --dry-run
    python scripts/l3_fts_repair.py --yes
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hermes_env import bootstrap, env_path

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
BACKUP_DIR = env_path("L3_BACKUP_DIR", MEMORY_DIR / "l3_backups")


def _integrity(path: Path) -> str:
    conn = sqlite3.connect("file:%s?mode=ro" % path.as_posix(), uri=True)
    try:
        return str(conn.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        conn.close()


def _backup(path: Path, kind: str = "pre_fts_repair") -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / ("l3_%s_%s.db" % (kind, stamp))
    src = sqlite3.connect(str(path))
    dst = sqlite3.connect(str(dest))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair the L3 FTS5 index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    parser.add_argument("--keep-backups", type=int, default=3,
                        help="Keep this many FTS-repair backups (default 3)")
    parser.add_argument("--snapshot", action="store_true",
                        help="Take a known-good post-repair snapshot even when "
                             "the index is already healthy")
    args = parser.parse_args()

    if not L3_DB_PATH.exists():
        print(json.dumps({"ok": False, "error": "L3 database not found",
                          "path": str(L3_DB_PATH)}, indent=2))
        return 1

    before = _integrity(L3_DB_PATH)
    if before == "ok":
        if args.snapshot:
            snapshot = _backup(L3_DB_PATH, kind="post_fts_repair")
            print(json.dumps({"ok": True, "changed": False,
                              "integrity": before,
                              "snapshot": str(snapshot)}, indent=2))
        else:
            print(json.dumps({"ok": True, "changed": False,
                              "integrity": before}, indent=2))
        return 0

    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "changed": False,
                          "integrity": before,
                          "action": "would rebuild messages_fts"}, indent=2))
        return 0

    if not args.yes:
        answer = input("Rebuild the L3 FTS index? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("aborted")
            return 1

    backup = _backup(L3_DB_PATH)
    conn = sqlite3.connect(str(L3_DB_PATH))
    try:
        conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()

    after = _integrity(L3_DB_PATH)
    post_snapshot = (_backup(L3_DB_PATH, kind="post_fts_repair")
                     if after == "ok" else None)
    removed = []
    for pattern in ("l3_pre_fts_repair_*.db", "l3_post_fts_repair_*.db"):
        backups = sorted(BACKUP_DIR.glob(pattern),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[max(1, int(args.keep_backups)):]:
            try:
                old.unlink()
                removed.append(old.name)
            except OSError:  # silent-ok: already-reported — 清理失败不影响修复结果
                pass

    out = {"ok": after == "ok", "changed": True,
           "before": before, "after": after,
           "backup": str(backup),
           "post_repair_snapshot": str(post_snapshot) if post_snapshot else "",
           "pruned": removed}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
