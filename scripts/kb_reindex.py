#!/usr/bin/env python3
"""Keep the KB vector index in step with the Markdown vault.

External writers (Obsidian, Syncthing, other agents) never touch the index.
This script is the scheduled backstop: it re-embeds only notes whose mtime
changed, or every note with ``--force``.

Usage:
    python scripts/kb_reindex.py --dry-run
    python scripts/kb_reindex.py
    python scripts/kb_reindex.py --force
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
    """Locate the repository without hardcoding a user path."""
    configured = os.environ.get("HGM_REPO")
    candidates = [Path(configured)] if configured else []
    candidates.append(SCRIPT_DIR.parent)
    for candidate in candidates:
        if (candidate / "plugin" / "memory_governed" / "_kb.py").exists():
            return candidate
    raise SystemExit(
        "cannot locate the HGM repository; set HGM_REPO to the checkout root"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync the KB vector index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed every note, not only changed ones")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Allow an empty vault to clear the whole index "
                             "(default: refuse — an empty vault usually means "
                             "a mount or wiki_dir problem)")
    args = parser.parse_args()

    repo = _repo_root()
    sys.path.insert(0, str(repo))
    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed._kb import KnowledgeBase

    kb = KnowledgeBase(load_governed_config(HERMES_HOME))
    paths = kb.scan_vault_paths()

    if args.dry_run:
        status = kb.last_index_status
        print(json.dumps({
            "ok": True,
            "dry_run": True,
            "vault_notes": len(paths),
            "index_status": status,
        }, ensure_ascii=False, indent=2))
        return 0

    # fail-closed：空 vault 几乎总是「挂载没就绪 / wiki_dir 指错了」，
    # 而不是「用户删光了所有笔记」。此时同步会删掉全部索引行，而且旧实现
    # 还报 ok:true / exit 0 —— 一次挂载抖动就被记成一次成功的重建。
    # 确实要清空索引的话，显式加 --allow-empty。
    if not paths and not args.allow_empty:
        print(json.dumps({
            "ok": False,
            "error": "vault is empty; refusing to clear the index",
            "vault_notes": 0,
            "hint": ("check that wiki_dir is mounted/populated; pass "
                     "--allow-empty only if you really want an empty index"),
        }, ensure_ascii=False, indent=2))
        return 1

    result = kb.sync_index(force=args.force)
    result = dict(result, vault_notes=len(paths))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") and not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
