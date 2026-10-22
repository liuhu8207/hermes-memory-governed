#!/usr/bin/env python3
"""Report near-duplicate L2 facts; merge only when explicitly asked.

The daily maintenance job runs this in report-only mode. ``--apply`` is a
manual, lossy operation: it calls the configured LLM to merge a cluster and
archives/tombstones the originals.

Usage:
    python scripts/memory_consolidate.py --dry-run
    python scripts/memory_consolidate.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hermes_env import bootstrap, resolve_repo

HERMES_HOME = bootstrap(__file__)


def _repo_root() -> Path:
    """Locate the checkout via HGM_REPO / source_repo / script location."""
    return resolve_repo(__file__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Report or merge near-duplicate L2 facts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true",
                        help="Merge clusters (lossy; archives and tombstones originals)")
    args = parser.parse_args()

    repo = _repo_root()
    sys.path.insert(0, str(repo))
    import memory_cli

    out = memory_cli.cmd_consolidate(
        memory_cli.load_config(), apply=bool(args.apply and not args.dry_run)
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
