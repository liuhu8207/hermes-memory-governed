#!/usr/bin/env python3
"""Periodic L2 quality sweep.

Capacity sweeps keep the row count bounded; they do not keep the *content*
clean. This script re-applies the quality classifier to existing rows:

* ``test_residue`` / ``assistant_fragment`` — high precision, safe to retract;
* ``stale_marker`` / ``duplicate`` — reported for human review, never auto-deleted.

Default is a read-only report. ``--apply --yes --kinds test_residue`` is what
the daily cron runs: it removes only unambiguous test residue. Everything else
stays visible for review.

Usage:
    python scripts/l2_quality_sweep.py
    python scripts/l2_quality_sweep.py --apply --yes --kinds test_residue
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, resolve_repo

HERMES_HOME = bootstrap(__file__)
REPO = resolve_repo(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _normalize(text: str) -> str:
    from plugin.memory_governed._text import normalize_for_match

    return normalize_for_match(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep L2 for non-durable facts")
    parser.add_argument("--dry-run", action="store_true",
                        help="Force report-only mode (orchestrator dry run)")
    parser.add_argument("--apply", action="store_true",
                        help="Retract the selected kinds (default: report only)")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm --apply")
    parser.add_argument("--kinds", default="test_residue",
                        help="Comma-separated kinds to retract under --apply "
                             "(default: test_residue)")
    parser.add_argument("--max", type=int, default=50,
                        help="Maximum rows to retract per run")
    args = parser.parse_args()

    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed._lifecycle import (
        archive_append, add_tombstone, backup_table, content_fp,
        resolve_memory_dir, sql_quote,
    )
    from plugin.memory_governed._quality import classify_fact

    cfg = load_governed_config(HERMES_HOME)
    mem_dir = resolve_memory_dir(cfg)
    if mem_dir is None:
        print(json.dumps({"ok": False, "error": "cannot resolve memory dir"},
                         indent=2))
        return 1

    import lancedb

    db = lancedb.connect(str(cfg.l2_db_path))
    table = db.open_table("memories")
    rows = table.to_arrow().to_pylist()

    norm_counts = Counter(_normalize(str(r.get("content") or "")) for r in rows)
    seen: Counter = Counter()
    candidates = []
    for index, row in enumerate(rows):
        content = str(row.get("content") or "")
        verdict = classify_fact(content, role=str(row.get("role") or ""),
                                source=str(row.get("source") or ""))
        if verdict is None:
            norm = _normalize(content)
            if norm_counts[norm] > 1:
                seen[norm] += 1
                if seen[norm] > 1:
                    verdict = {"kind": "duplicate",
                               "reason": "same normalized content appears more than once"}
        if verdict:
            candidates.append({
                "index": index,
                "kind": verdict["kind"],
                "reason": verdict["reason"],
                "agent": row.get("agent"),
                "category": row.get("category"),
                "timestamp": row.get("timestamp"),
                "content": content[:160],
            })

    summary = Counter(c["kind"] for c in candidates)
    out = {"ok": True, "rows": len(rows), "candidates": candidates,
           "summary": dict(summary), "applied": []}

    kinds = {k.strip() for k in args.kinds.split(",") if k.strip()}
    to_apply = [c for c in candidates if c["kind"] in kinds][:max(0, args.max)]

    if args.dry_run or not args.apply or not to_apply:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if not args.yes:
        out.update({"ok": False,
                    "error": "refusing --apply without --yes"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    backup = backup_table(Path(cfg.l2_db_path), tag="quality",
                          memory_dir=mem_dir)
    if backup is None:
        out.update({"ok": False, "error": "cannot back up the L2 table"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    applied = []
    errors = []
    for candidate in to_apply:
        row = rows[candidate["index"]]
        content = str(row.get("content") or "")
        predicate = "content = " + sql_quote(content)
        if row.get("timestamp") is not None:
            predicate += " AND timestamp = " + sql_quote(str(row["timestamp"]))
        if row.get("agent") is not None:
            predicate += " AND agent = " + sql_quote(str(row["agent"]))
        try:
            archive_append(mem_dir, {
                "state": "superseded",
                "reason": "quality-" + candidate["kind"],
                "ts": candidate["timestamp"],
                "content": content,
                "agent": row.get("agent"),
            })
            add_tombstone(mem_dir, content_fp(content), {
                "reason": "quality-" + candidate["kind"],
                "content": content[:200],
            })
            table.delete(predicate)
            applied.append({"index": candidate["index"],
                            "kind": candidate["kind"],
                            "content": content[:120]})
        except Exception as exc:  # silent-ok: already-reported — 聚合进返回的 errors，不中断整轮
            errors.append("%s: %s" % (candidate["index"], type(exc).__name__))

    out.update({"applied": applied, "errors": errors,
                "backup": str(backup),
                "remaining": table.count_rows()})
    out["ok"] = not errors
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
