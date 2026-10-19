#!/usr/bin/env python3
"""Daily global L2 capacity sweep.

The plugin's automatic extraction path intentionally does not evict on every
turn. This script applies the configured ``sync.l2_max_items`` cap once a day:
least-used rows are archived, then removed. A backup is taken before any
deletion.

Usage:
    python scripts/l2_capacity.py --dry-run
    python scripts/l2_capacity.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hermes_env import bootstrap

HERMES_HOME = bootstrap(__file__)


def _repo_root() -> Path:
    configured = os.environ.get("HGM_REPO")
    candidates = [Path(configured)] if configured else []
    candidates.append(SCRIPT_DIR.parent)
    for candidate in candidates:
        if (candidate / "memory_cli.py").exists():
            return candidate
    raise SystemExit(
        "cannot locate the HGM repository; set HGM_REPO to the checkout root"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply the global L2 capacity cap")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cap", type=int, default=None,
                        help="Override sync.l2_max_items for this run")
    args = parser.parse_args()

    repo = _repo_root()
    sys.path.insert(0, str(repo))
    import memory_cli
    from plugin.memory_governed import _lifecycle as life

    cfg = memory_cli.plugin_config()
    cap = args.cap if args.cap is not None else int(
        getattr(cfg.sync, "l2_max_items", 0) or 0
    )
    out = {"ok": True, "cap": cap, "dry_run": bool(args.dry_run)}
    if cap <= 0:
        out.update({"rows": None, "note": "capacity disabled (l2_max_items <= 0)"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    table = memory_cli.open_l2_table(cfg)
    if table is None:
        out.update({"ok": False, "error": "L2 table unavailable"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    rows = table.to_arrow().num_rows
    out["rows"] = int(rows)
    out["over"] = max(0, int(rows) - cap)
    if args.dry_run or out["over"] == 0:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    mem_dir = life.resolve_memory_dir(cfg)
    if mem_dir is None:
        out.update({"ok": False, "error": "cannot resolve memory directory"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1
    demoted = life.enforce_capacity(table, mem_dir, cap, agent=None)
    out["demoted"] = len(demoted)
    out["demoted_prefixes"] = demoted
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
