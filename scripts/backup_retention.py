#!/usr/bin/env python3
"""Backup retention with an explicit, auditable standard.

Why this exists: every ``retract`` makes a full L2 table backup. A one-by-one
cleanup of 19 facts grew ``l2_backups`` from 25 to 46 copies (~690 MB). The
backups are real restore points, but they must not grow without bound.

Standard
--------
L2 (table directories under ``memory/l2_backups``):
  * keep the newest 7 by **timestamp in the name** (mtime is unreliable: copied
    directories often inherit an old mtime);
  * keep the newest backup of each of the 3 most recent distinct months;
  * keep named milestones (``cleanup`` / ``planB`` / ``pre_restore``).

L3 (``memory/l3_backups/l3_*.db``): keep the newest 5; ``-wal``/``-shm`` files
are not counted and are removed with their database.

KB index backups (``kb_notes.bak_*.lance`` / ``kb_notes.old*.lance``): keep 2.

Vault backups (``<wiki>/.kb-backups-*``): keep **none** inside the vault. Every
backup is moved to ``memory/vault_backups/`` (outside Obsidian), where the
newest 3 are kept.

Default is a read-only report. ``--apply --yes`` performs deletions/moves.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = HERMES_HOME / "memory"

L2_KEEP_RECENT = 7
L2_KEEP_MONTHS = 3
L2_MILESTONES = ("cleanup", "planb", "pre_restore")
L3_KEEP = 5
KB_KEEP = 2
VAULT_KEEP = 0
VAULT_ARCHIVE_KEEP = 3

_STAMP = re.compile(r"(\d{8})_(\d{6})")
_DATE = re.compile(r"(\d{8})")


def _stamp(path: Path) -> datetime:
    name = path.name
    m = _STAMP.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:  # silent-ok: value-fallback — 名字里的时间戳坏值 → 回退 mtime
            pass
    m = _DATE.search(name)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d")
        except ValueError:  # silent-ok: value-fallback — 名字里的时间戳坏值 → 回退 mtime
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _select_l2(paths: list) -> tuple[set, list]:
    ordered = sorted(paths, key=_stamp, reverse=True)
    keep = set(ordered[:L2_KEEP_RECENT])
    months = {}
    for p in ordered:
        months.setdefault((_stamp(p).year, _stamp(p).month), p)
    for key in sorted(months, reverse=True)[:L2_KEEP_MONTHS]:
        keep.add(months[key])
    for p in ordered:
        if any(m in p.name.lower() for m in L2_MILESTONES):
            keep.add(p)
    return keep, [p for p in ordered if p not in keep]


def _select_recent(paths: list, keep_n: int) -> tuple[set, list]:
    ordered = sorted(paths, key=_stamp, reverse=True)
    keep = set(ordered[:keep_n])
    return keep, [p for p in ordered if p not in keep]


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
        for suffix in ("-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists():
                side.unlink()


def _move_vault_backups(paths: list, archive: Path) -> list:
    archive.mkdir(parents=True, exist_ok=True)
    moved = []
    for p in paths:
        dest = archive / p.name
        if dest.exists():
            dest = archive / (p.name + "." + datetime.now().strftime("%H%M%S"))
        shutil.move(str(p), str(dest))
        moved.append({"from": str(p), "to": str(dest)})
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description="Backup retention")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    l2_dir = MEMORY_DIR / "l2_backups"
    l3_dir = MEMORY_DIR / "l3_backups"
    kb_dir = MEMORY_DIR / "kb_index"
    vault = None
    try:
        from plugin.memory_governed._config import load_governed_config

        vault = Path(load_governed_config(HERMES_HOME).wiki_dir)
    except Exception:  # silent-ok: already-reported — 没有 config 就跳过 vault 部分
        vault = None

    l2_paths = [p for p in l2_dir.iterdir() if p.is_dir()] if l2_dir.is_dir() else []
    l3_paths = [p for p in l3_dir.glob("l3_*.db") if p.is_file()] if l3_dir.is_dir() else []
    kb_paths = ([p for p in kb_dir.glob("kb_notes.bak_*.lance") if p.is_dir()]
                + [p for p in kb_dir.glob("kb_notes.old*.lance") if p.is_dir()]
                if kb_dir.is_dir() else [])
    vault_paths = ([p for p in vault.glob(".kb-backups-*") if p.is_dir()]
                   if vault and vault.is_dir() else [])

    l2_keep, l2_drop = _select_l2(l2_paths)
    l3_keep, l3_drop = _select_recent(l3_paths, L3_KEEP)
    kb_keep, kb_drop = _select_recent(kb_paths, KB_KEEP)
    vault_keep, vault_move = _select_recent(vault_paths, VAULT_KEEP)

    archive = MEMORY_DIR / "vault_backups"
    vault_archive_keep, vault_archive_drop = _select_recent(
        [p for p in archive.iterdir() if p.is_dir()] if archive.is_dir() else [],
        VAULT_ARCHIVE_KEEP)

    plan = {
        "l2": {"keep": [p.name for p in l2_keep], "drop": [p.name for p in l2_drop]},
        "l3": {"keep": [p.name for p in l3_keep], "drop": [p.name for p in l3_drop]},
        "kb_index": {"keep": [p.name for p in kb_keep], "drop": [p.name for p in kb_drop]},
        "vault": {"keep": [p.name for p in vault_keep],
                  "move": [p.name for p in vault_move]},
        "vault_archive": {"keep": [p.name for p in vault_archive_keep],
                          "drop": [p.name for p in vault_archive_drop]},
    }
    reclaim = sum(_size(p) for p in l2_drop + l3_drop + kb_drop + vault_archive_drop)
    out = {"ok": True, "apply": bool(args.apply and not args.dry_run),
           "plan": plan, "reclaim_mb": round(reclaim / (1024 * 1024), 1)}

    if args.dry_run or not args.apply:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0
    if not args.yes:
        out.update({"ok": False, "error": "refusing --apply without --yes"})
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    errors = []
    for p in l2_drop + l3_drop + kb_drop:
        try:
            _remove(p)
        except OSError as exc:  # silent-ok: already-reported — 聚合进 errors
            errors.append("%s: %s" % (p.name, type(exc).__name__))
    try:
        out["moved"] = _move_vault_backups(vault_move, archive)
    except OSError as exc:  # silent-ok: already-reported — 聚合进 errors
        errors.append("vault move: %s" % type(exc).__name__)
    for p in vault_archive_drop:
        try:
            _remove(p)
        except OSError as exc:  # silent-ok: already-reported — 聚合进 errors
            errors.append("%s: %s" % (p.name, type(exc).__name__))

    out["errors"] = errors
    out["ok"] = not errors
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
