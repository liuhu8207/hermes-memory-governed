#!/usr/bin/env python3
"""Compact the KB vector index and drop old versions.

Every upsert/delete in LanceDB leaves a manifest version behind. The real
``kb_notes`` table accumulated **10,911 versions for 52 rows** — unbounded growth
that slows opens and scans. ``Table.optimize(cleanup_older_than=...)`` compacts
the data and removes versions older than the retention window.

Three safety rules (2026-10-20)
-------------------------------
1. **非交互环境必须显式 ``--yes``**。cron / CI 里没有 TTY ⇒ 没人能回答"确定吗"
   ⇒ 静默执行一个**破坏性**操作是最坏的组合：既没人授权，也没人知道它发生过。
   交互环境下会要求确认。
2. **做之前先备份**：``optimize`` 会重写表目录，失败时没有回退点。现在先复制
   一份整表副本，成功后再删掉**我们自己刚建的那份**。
3. **绝不删 schema 迁移留下的 ``kb_notes.bak_<时间戳>``** —— 那是**老索引的唯一
   副本**（旧表 drop 之后只剩它）。旧实现在收尾时会把它们 prune 掉。

Usage:
    python scripts/kb_index_optimize.py --dry-run
    python scripts/kb_index_optimize.py --days 7 --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional
import time
from datetime import timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, env_path, setup_logging

HERMES_HOME = bootstrap(__file__)

#: 读 stdin 前必须先声明编码：宿主给 GBK 时，``input()``/``read()`` 拿到的是
#: 乱码或直接抛异常，而这里的确认提示会被读成"别的答案" ⇒ 用户以为按了 y。
#: 仓库惯例（见 ``wb_prompt_hook.py`` 等）。
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):  # silent-ok: value-fallback — 流不可 reconfigure（如已被读）就沿用默认编码
            pass

logger = setup_logging("kb_index_optimize")

try:  # 软依赖：脚本也要能在仓库安装之外运行
    from plugin.memory_governed._diag import log_degraded
except ImportError:  # silent-ok: value-fallback — 插件不在 sys.path 上 → 退回普通 logging
    def log_degraded(component: str, reason: str, *, detail: str = "",
                     exc: "Optional[BaseException]" = None) -> None:
        logger.warning("degraded — %s: %s | %s | %s",
                       component, reason, detail, exc)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")

#: 我们自己建的 optimize 备份（与 schema 迁移的备份区分开）。
_OPT_BACKUP_INFIX = ".bak_opt_"


def _table_dir() -> Path:
    return MEMORY_DIR / "kb_index" / "kb_notes.lance"


def _stats(directory: Path) -> dict:
    def _count(name: str) -> int:
        root = directory / name
        return sum(1 for _ in root.rglob("*")) if root.is_dir() else 0

    return {"versions": _count("_versions"),
            "transactions": _count("_transactions"),
            "data_files": _count("data")}


def _is_opt_backup(path: Path) -> bool:
    """是不是**我们自己**建的 optimize 备份（schema 迁移的备份不是）。"""
    return _OPT_BACKUP_INFIX in path.name


def _prune_opt_backups(table_dir: Path) -> list:
    """只清理 optimize 自己建的备份（``*.bak_opt_*``）。

    schema 迁移留下的 ``kb_notes.bak_<时间戳>`` 是**老索引的唯一副本**，一律
    保留 —— 曾经在这里被 prune 掉，等于把唯一的回退点删了。
    """
    import shutil
    removed = []
    # 覆盖两类**我们自己**留下的副本：optimize 自己的备份（``bak_opt_*``）与换表
    # 时的旧表副本（``old*``）。迁移留下的 ``kb_notes.bak_<时间戳>`` **不动** ——
    # 那是老索引的唯一副本。
    stale = list(table_dir.parent.glob("kb_notes.bak_opt_*.lance"))
    stale += list(table_dir.parent.glob("kb_notes.old*.lance"))
    for old in stale:
        try:
            shutil.rmtree(old)
            removed.append(old.name)
        except OSError:  # silent-ok: already-reported — 清理失败不影响优化结果
            pass
    return removed


def _backup_table(table_dir: Path) -> Path:
    """优化前留一份整表副本（``optimize`` 会重写表目录，没有回退点）。"""
    import shutil
    dst = table_dir.parent / ("kb_notes%s%s.lance" % (
        _OPT_BACKUP_INFIX, time.strftime("%Y%m%d_%H%M%S")))
    if dst.exists():
        import shutil as _sh
        _sh.rmtree(dst)
    shutil.copytree(table_dir, dst)
    return dst


def _confirm(action: str, *, assume_yes: bool) -> bool:
    """非交互环境必须显式 ``--yes``；交互环境要操作者确认。

    返回 ``False`` = 不执行（调用方应当非零退出）。
    """
    if assume_yes:
        return True
    interactive = False
    try:
        interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError) as e:
        # TTY 探测失败是**子系统级**故障：我们**无法**判断有没有人能确认 ⇒
        # 报出来（不能让它悄悄退化成"假装交互"），并按"非交互"处理 = 拒绝执行。
        log_degraded("kb_index_optimize", "tty_probe_failed", exc=e)
        interactive = False
    if not interactive:
        print(json.dumps({
            "ok": False,
            "error": ("refusing to %s without --yes: this environment is not "
                      "interactive (no TTY), so there is nobody to confirm a "
                      "destructive operation" % action),
            "hint": "re-run with --yes to proceed",
        }, ensure_ascii=False, indent=2))
        return False
    try:
        answer = input("%s? [y/N] " % action)
    except EOFError as e:
        # ``EOFError`` 不属于窄类型（不是"值级回退"），所以**必须上报**：
        # 确认通道被关掉是一个**环境事实**（有人按了 Ctrl-D / stdin 被关），
        # 不是"没给答案"那种值级情况。判未确认（fail-closed）并把这件事
        # 记进计数，别让它悄悄变成"用户没答"。
        log_degraded("kb_index_optimize", "confirmation_input_closed", exc=e)
        print(json.dumps({
            "ok": False,
            "error": ("no confirmation available (stdin closed); treating it as "
                      "NOT confirmed"),
            "hint": "re-run with --yes to proceed",
        }, ensure_ascii=False, indent=2))
        return False
    if answer.strip().lower() not in ("y", "yes"):
        print(json.dumps({"ok": False, "error": "cancelled by operator"},
                         ensure_ascii=False, indent=2))
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize the KB vector index")
    parser.add_argument("--days", type=int, default=7,
                        help="Keep versions newer than this many days (default 7)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm the (destructive) optimization; required "
                             "when there is no TTY")
    parser.add_argument("--keep-index-backups", type=int, default=2,
                        help="Deprecated and ignored: schema-migration "
                             "backups (kb_notes.bak_<timestamp>) are the only "
                             "copy of the pre-migration index and are never "
                             "pruned by this script")
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

    if not _confirm("compact the KB vector index (prune old versions)",
                    assume_yes=args.yes):
        return 2

    db = lancedb.connect(str(table_dir.parent))
    table = db.open_table("kb_notes")
    rows = table.count_rows()
    # 迁移留下的备份：老索引的唯一副本，**只列出来、绝不删**。
    migration_backups = sorted(
        p.name for p in table_dir.parent.glob("kb_notes.bak_*.lance")
        if not _is_opt_backup(p))
    # 先备份再动手：optimize 会重写表目录。
    backup = _backup_table(table_dir)
    try:
        table.optimize(cleanup_older_than=timedelta(days=max(0, args.days)))
    except Exception as e:  # noqa: BLE001 — 失败必须保留备份（那是唯一的回退点）
        # 子系统级故障 ⇒ **必须上报**（不能只打印一行就算了：打印给终端的
        # 没人看得见，计数才看得见）。异常文本交给 log_degraded 安全地格式化
        # （_diag 内部对坏 __str__ 兜底）—— 这里**不急切字符串化** e。
        log_degraded("kb_index_optimize", "optimize_failed", exc=e)
        print(json.dumps({
            "ok": False,
            "error": "optimize failed (%s); the backup was kept" % type(e).__name__,
            "backup_kept": backup.name,
        }, ensure_ascii=False, indent=2))
        return 1

    after = _stats(table_dir)
    print(json.dumps({"ok": True, "rows": int(rows), "days": args.days,
                      "before": before, "after": after,
                      "backup": backup.name,
                      "migration_backups_kept": migration_backups},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
