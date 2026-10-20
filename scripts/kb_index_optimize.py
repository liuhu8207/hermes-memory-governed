#!/usr/bin/env python3
"""Compact the KB vector index and drop old versions.

Every upsert/delete in LanceDB leaves a manifest version behind. The real
``kb_notes`` table accumulated **10,911 versions for 52 rows** — unbounded growth
that slows opens and scans. ``Table.optimize(cleanup_older_than=...)`` compacts
the data and removes versions older than the retention window.

Usage:
    python scripts/kb_index_optimize.py --dry-run
    python scripts/kb_index_optimize.py --days 7
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, env_path

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")


def _table_dir() -> Path:
    return MEMORY_DIR / "kb_index" / "kb_notes.lance"


def _stats(directory: Path) -> dict:
    def _count(name: str) -> int:
        root = directory / name
        return sum(1 for _ in root.rglob("*")) if root.is_dir() else 0

    return {"versions": _count("_versions"),
            "transactions": _count("_transactions"),
            "data_files": _count("data")}


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize the KB vector index")
    parser.add_argument("--days", type=int, default=7,
                        help="Keep versions newer than this many days (default 7)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-index-backups", type=int, default=2,
                        help="Keep this many kb_notes.bak_* table backups "
                             "(default 2)")
    args = parser.parse_args()

    table_dir = _table_dir()
    if not table_dir.is_dir():
        print(json.dumps({"ok": False, "error": "kb index table not found",
                          "path": str(table_dir)}, ensure_ascii=False, indent=2))
        return 1

    import lancedb

    before = _stats(table_dir)
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "before": before,
                          "days": args.days}, ensure_ascii=False, indent=2))
        return 0

    db = lancedb.connect(str(table_dir.parent))
    table = db.open_table("kb_notes")
    rows = table.count_rows()
    table.optimize(cleanup_older_than=timedelta(days=max(0, args.days)))
    # schema 迁移留下的整表备份也要有保留期，否则每次迁移多一份副本。
    removed = []
    backups = sorted(table_dir.parent.glob("kb_notes.bak_*.lance"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[max(1, int(args.keep_index_backups)):]:
        try:
            import shutil
            shutil.rmtree(old)
            removed.append(old.name)
        except OSError:  # silent-ok: already-reported — 清理失败不影响优化结果
            pass

    after = _stats(table_dir)
    print(json.dumps({"ok": True, "rows": int(rows), "days": args.days,
                      "before": before, "after": after,
                      "pruned_index_backups": removed},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
