#!/usr/bin/env python3
"""Restore backups into an isolated directory and verify them.

A backup that has never been restored is not a backup. This drill copies the
newest (or named) L2 / L3 / vault backups into a temporary tree, opens them the
same way the runtime does, and reports row counts plus integrity checks. The
live store is only read, never written.

Usage:
    python scripts/restore_drill.py --dry-run
    python scripts/restore_drill.py
    python scripts/restore_drill.py --l2-backup memories.lance_capacity_...
    python scripts/restore_drill.py --keep
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# 让 ``plugin.*`` 可导入（读取 wiki_dir、定位 vault 备份）
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, env_path

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")


def _safe_exc(exc: BaseException) -> str:
    """``Type: message`` that never raises, even with a broken ``__str__``."""
    try:
        return "%s: %s" % (type(exc).__name__, exc)
    except Exception:  # silent-ok: already-reported — returned text goes into the report
        return type(exc).__name__


def _latest(paths):
    items = [p for p in paths if p.exists()]
    return max(items, key=lambda p: p.stat().st_mtime) if items else None


def _pick_l2(name: str):
    root = MEMORY_DIR / "l2_backups"
    if not root.is_dir():
        return None
    if name and name != "latest":
        candidate = root / name
        return candidate if candidate.is_dir() else None
    return _latest([p for p in root.iterdir() if p.is_dir()])


def _pick_l3(name: str):
    root = MEMORY_DIR / "l3_backups"
    if not root.is_dir():
        return None
    if name and name != "latest":
        candidate = root / name
        return candidate if candidate.is_file() else None
    # 脱敏前备份和 FTS 修复前备份都是完整 DB 快照，取最新的那一份。
    return _latest(list(root.glob("l3_*.db")))


def _wiki_dir() -> Path:
    try:
        from plugin.memory_governed._config import load_governed_config

        cfg = load_governed_config(HERMES_HOME)
        if str(getattr(cfg, "wiki_dir", "") or "").strip():
            return Path(cfg.wiki_dir)
    except Exception as exc:  # silent-ok: already-reported — 已打印 warning
        print("warning: cannot read governed config: %s" % _safe_exc(exc),
              file=sys.stderr)
    return HERMES_HOME / "wiki"


def _pick_vault(name: str):
    wiki = _wiki_dir()
    if not wiki.is_dir():
        return None
    if name and name != "latest":
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = wiki / name
        return candidate if candidate.is_dir() else None
    return _latest([p for p in wiki.glob(".kb-backups-*") if p.is_dir()])


def _pick_kb_index(name: str):
    root = MEMORY_DIR / "kb_index"
    if not root.is_dir():
        return None
    if name and name != "latest":
        candidate = root / name
        return candidate if candidate.is_dir() else None
    return _latest(list(root.glob("kb_notes.bak_*.lance")))


def _count_l2(path: Path) -> dict:
    import lancedb

    db = lancedb.connect(str(path.parent))
    table = db.open_table("memories")
    rows = table.count_rows()
    cols = [f.name for f in table.schema]
    return {"rows": int(rows), "columns": cols,
            "has_content": "content" in cols, "has_vector": "vector" in cols}


def _count_l3(db_path: Path) -> dict:
    conn = sqlite3.connect("file:%s?mode=ro" % db_path.as_posix(), uri=True)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        try:
            fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        except sqlite3.Error:  # silent-ok: already-reported — fts=None 写进结果，表示旧库没有 FTS 表
            fts = None
        return {"integrity": integrity, "messages": int(messages),
                "fts_rows": fts}
    finally:
        conn.close()


def _count_vault(root: Path) -> dict:
    # 根 index.md 是 MOC 导航，不是笔记，没有 frontmatter 是正常的。
    notes = [p for p in root.rglob("*.md") if p.name != "index.md"]
    malformed = 0
    for p in notes:
        try:
            head = p.read_text(encoding="utf-8", errors="replace").lstrip()
        except OSError:  # silent-ok: already-reported — 计入 malformed
            malformed += 1
            continue
        if not head.startswith("---"):
            malformed += 1
    return {"notes": len(notes), "malformed_frontmatter": malformed}


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore and verify backups")
    parser.add_argument("--l2-backup", default="latest")
    parser.add_argument("--l3-backup", default="latest")
    parser.add_argument("--vault-backup", default="latest")
    parser.add_argument("--kb-index-backup", default="latest")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be restored, touch nothing")
    parser.add_argument("--keep", action="store_true",
                        help="Keep the restored tree and print its path")
    args = parser.parse_args()

    l2 = _pick_l2(args.l2_backup)
    l3 = _pick_l3(args.l3_backup)
    vault = _pick_vault(args.vault_backup)
    kb_index = _pick_kb_index(args.kb_index_backup)

    plan = {
        "l2": str(l2) if l2 else None,
        "l3": str(l3) if l3 else None,
        "vault": str(vault) if vault else None,
        "kb_index": str(kb_index) if kb_index else None,
    }
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "plan": plan},
                         ensure_ascii=False, indent=2))
        return 0

    if not (l2 or l3 or vault):
        print(json.dumps({
            "ok": False,
            "error": "no backups found",
            "searched": {
                "l2": str(MEMORY_DIR / "l2_backups"),
                "l3": str(MEMORY_DIR / "l3_backups"),
                "vault": str(_wiki_dir()),
            },
        }, ensure_ascii=False, indent=2))
        return 1

    started = time.time()
    root = Path(tempfile.mkdtemp(prefix="hgm-restore-drill-"))
    result: dict[str, Any] = {"ok": True, "restored_into": str(root),
                              "sources": plan, "checks": {}}
    try:
        if l2:
            dest = root / "l2" / "memories.lance"
            dest.parent.mkdir(parents=True)
            shutil.copytree(l2, dest)
            result["checks"]["l2"] = _count_l2(dest)

        if l3:
            dest = root / "l3" / "l3.db"
            dest.parent.mkdir(parents=True)
            shutil.copy2(l3, dest)
            result["checks"]["l3"] = _count_l3(dest)

        if vault:
            dest = root / "wiki"
            shutil.copytree(vault, dest)
            result["checks"]["vault"] = _count_vault(dest)

        if kb_index:
            dest = root / "kb_index" / "kb_notes.lance"
            dest.parent.mkdir(parents=True)
            shutil.copytree(kb_index, dest)
            import lancedb

            table = lancedb.connect(str(dest.parent)).open_table("kb_notes")
            cols = [f.name for f in table.schema]
            result["checks"]["kb_index"] = {
                "rows": int(table.count_rows()), "columns": cols,
                "has_mtime": "mtime" in cols}
        else:
            result["checks"]["kb_index"] = {
                "restored": False,
                "note": "no index backup; rebuild from the vault with "
                        "kb_reindex.py --force (the vault is the source of truth)"}

        l2_check = result["checks"].get("l2") or {}
        if l2_check and not (l2_check.get("has_content")
                             and l2_check.get("has_vector")):
            result["ok"] = False
            result["error"] = "restored L2 table is missing required columns"

        l3_check = result["checks"].get("l3") or {}
        if l3_check and l3_check.get("integrity") != "ok":
            # FTS5 的倒排索引可以从 messages 重建；这是**恢复副本**上的修复，
            # 不碰源备份。若重建后仍不 ok，才是真的恢复失败。
            restored_l3 = root / "l3" / "l3.db"
            try:
                conn = sqlite3.connect(str(restored_l3))
                try:
                    conn.execute(
                        "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
                    conn.commit()
                finally:
                    conn.close()
                after = _count_l3(restored_l3)
                l3_check["fts_rebuilt_in_restore"] = True
                l3_check["integrity_after_rebuild"] = after.get("integrity")
                result.setdefault("warnings", []).append(
                    "restored L3 FTS index was malformed and was rebuilt "
                    "(the source backup needs the same repair)")
                if after.get("integrity") != "ok":
                    result["ok"] = False
                    result["error"] = ("restored L3 still fails integrity "
                                       "after FTS rebuild")
            except Exception as e:  # silent-ok: already-reported — 收敛进 result["error"]
                result["ok"] = False
                result["error"] = ("restored L3 FTS repair failed: %s"
                                   % _safe_exc(e))

        vault_check = result["checks"].get("vault") or {}
        if vault_check.get("malformed_frontmatter"):
            result["ok"] = False
            result["error"] = "restored vault has malformed frontmatter"
    except Exception as exc:  # silent-ok: already-reported — 收敛进 result
        result["ok"] = False
        result["error"] = _safe_exc(exc)
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 3)
        if not args.keep:
            shutil.rmtree(root, ignore_errors=True)
            result["restored_into"] = None
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
