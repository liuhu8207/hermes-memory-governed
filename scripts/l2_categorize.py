#!/usr/bin/env python3
"""Migrate legacy L2 categories to the shared controlled vocabulary.

Default is read-only. ``--apply --yes`` first backs up the whole Lance table,
then updates only rows whose normalized category changes.

Usage:
    python scripts/l2_categorize.py
    python scripts/l2_categorize.py --apply --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, resolve_repo

HERMES_HOME = bootstrap(__file__)
REPO = resolve_repo(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _predicate(row: dict) -> str:
    from plugin.memory_governed._lifecycle import sql_quote

    parts = ["content = " + sql_quote(str(row.get("content") or ""))]
    if row.get("timestamp") is not None:
        parts.append("timestamp = " + sql_quote(str(row["timestamp"])))
    if row.get("agent") is not None:
        parts.append("agent = " + sql_quote(str(row["agent"])))
    return " AND ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate L2 categories")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    from plugin.memory_governed._categories import (
        CONTROLLED_CATEGORIES, classify_fact_category,
        normalize_legacy_category,
    )
    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed._lifecycle import backup_table, resolve_memory_dir

    cfg = load_governed_config(HERMES_HOME)
    memory_dir = resolve_memory_dir(cfg)
    import lancedb
    table = lancedb.connect(str(cfg.l2_db_path)).open_table("memories")
    rows = table.to_arrow().to_pylist()

    changes = []
    for row in rows:
        content = str(row.get("content") or "")
        raw = str(row.get("category") or "")
        current = normalize_legacy_category(raw)
        inferred = classify_fact_category(content)
        target = inferred if inferred != "other" else (current or "other")
        if target not in CONTROLLED_CATEGORIES and target != "other":
            target = "other"
        if target != raw:
            changes.append({"row": row, "from": raw, "to": target,
                            "content": content[:120]})

    out = {"ok": True, "apply": bool(args.apply and not args.dry_run),
           "rows": len(rows), "changes": len(changes),
           "items": [{k: item[k] for k in ("from", "to", "content")}
                     for item in changes[:50]]}

    if args.dry_run or not args.apply:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if not args.yes:
        out.update({"ok": False, "error": "refusing --apply without --yes"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1
    if memory_dir is None:
        out.update({"ok": False, "error": "cannot resolve memory dir"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1
    backup = backup_table(Path(cfg.l2_db_path), tag="categorize",
                          memory_dir=memory_dir)
    if backup is None:
        out.update({"ok": False, "error": "cannot back up L2 table"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    applied = 0
    errors = []
    for item in changes:
        try:
            table.update(where=_predicate(item["row"]),
                         values={"category": item["to"]})
            applied += 1
        except Exception as exc:  # noqa: BLE001  # silent-ok: already-reported — error type is aggregated into the returned errors list
            errors.append(type(exc).__name__)
    out.update({"applied": applied, "errors": errors,
                "remaining": table.count_rows(), "backup": str(backup)})
    out["ok"] = not errors
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
