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
from typing import Optional
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hermes_env import bootstrap, resolve_repo, setup_logging

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

logger = setup_logging("kb_reindex")

try:  # 软依赖：脚本也要能在仓库安装之外运行
    from plugin.memory_governed._diag import log_degraded
except ImportError:  # silent-ok: value-fallback — 插件不在 sys.path 上 → 退回普通 logging
    def log_degraded(component: str, reason: str, *, detail: str = "",
                     exc: "Optional[BaseException]" = None) -> None:
        logger.warning("degraded — %s: %s | %s | %s",
                       component, reason, detail, exc)


def _repo_root() -> Path:
    """Locate the checkout via HGM_REPO / source_repo / script location."""
    return resolve_repo(__file__)


def _confirm(action: str, *, assume_yes: bool) -> bool:
    """非交互环境必须显式 ``--yes``；交互环境要操作者确认。

    只用于**破坏性**动作（清空整个索引）。普通同步是 cron 里的日常兜底
    （``scripts/cron_memory_daily.py`` 会无人值守地调它），而且经过
    fail-closed 改造后它**不再删索引行**，所以不需要也不应该被拦住。
    """
    if assume_yes:
        return True
    interactive = False
    try:
        interactive = bool(sys.stdin.isatty() and sys.stdout.isatty())
    except (AttributeError, ValueError) as e:
        # TTY 探测失败是**子系统级**故障，不是值级回退：我们**无法**判断有没有
        # 人能确认 ⇒ 报出来（不能让它悄悄退化成"假装交互"），并按"非交互"
        # 处理 —— 也就是**拒绝执行**（fail-closed）。
        log_degraded("kb_reindex", "tty_probe_failed", exc=e)
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
        log_degraded("kb_reindex", "confirmation_input_closed", exc=e)
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
    parser = argparse.ArgumentParser(description="Sync the KB vector index")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed every note, not only changed ones")
    parser.add_argument("--allow-empty", action="store_true",
                        help="Allow an empty vault to clear the whole index "
                             "(default: refuse — an empty vault usually means "
                             "a mount or wiki_dir problem)")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm the destructive --allow-empty path; "
                             "required when there is no TTY")
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

    # 清空整个索引是**破坏性**动作：非交互环境（cron）里没人能确认 ⇒ 必须
    # 显式 --yes。普通同步不在此列（无人值守的日常兜底，且已 fail-closed）。
    if args.allow_empty and not _confirm("clear the whole KB index",
                                         assume_yes=args.yes):
        return 2

    # allow_purge=True 是 sync_index 侧"显式确认 vault 就是空的"开关：
    # 删除索引行不可逆，而"vault 读失败"与"vault 真的空"在调用方长得一模一样，
    # 所以 KnowledgeBase.sync_index 默认 fail-closed 拒绝删除。这里把脚本已有的
    # --allow-empty 意图**透传**下去（不传的话 sync_index 会拒绝清空）。
    result = kb.sync_index(force=args.force, allow_purge=args.allow_empty)
    result = dict(result, vault_notes=len(paths))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") and not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
