#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用**当前**质量门回放整张 L2 表，剔除不再符合准入条件的存量条目。

为什么需要它
------------
质量门（``_sync._looks_like_fact`` + ``_fact_signal_score(strong_only=True)``）
只作用于**新写入**。规则收紧之后，历史存量仍是按旧标准进来的，必须回头清一遍，
否则它们会继续参与向量召回、抬高无关查询的地板分。

role 的近似
-----------
L2 表**不存 role**（``_index_l2`` 只落 content/category/source/timestamp/vector）。
回放时统一按 ``user`` 判定 —— 这是**已知的有意偏松**：user 基线 +2，比 assistant
基线 -1 宽松，宁可漏删也不误杀。代价是少量助手叙述可能残留，用 ``--drop-extra``
手工补刀；反过来若按 assistant 判定，会误杀大量真实的用户约束。

用法::

    python scripts/l2_apply_gate.py --dry-run          # 预演，打印保留/剔除清单
    python scripts/l2_apply_gate.py                    # 执行（自动备份）
    python scripts/l2_apply_gate.py --drop-extra "就是你给我的" --drop-extra "阵列卡"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin.memory_governed._sync import (  # noqa: E402
    _MIN_SIGNAL_USER,
    WriteQueue,
    _fact_signal_score,
)


def admits(text: str) -> bool:
    """当前质量门是否会让这条内容进入 L2（按 user 角色判定）。"""
    if not WriteQueue._looks_like_fact(text, "user"):
        return False
    return _fact_signal_score(text, "user", strong_only=True) > _MIN_SIGNAL_USER


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay the L2 quality gate over existing rows.")
    ap.add_argument("--l2-dir", default=str(Path.home() / ".hermes" / "memory" / "l2"))
    ap.add_argument("--table", default="memories")
    ap.add_argument("--drop-extra", action="append", default=[],
                    help="额外剔除的内容子串（人工补刀），可重复")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    db = lancedb.connect(args.l2_dir)
    table = db.open_table(args.table)
    arrow = table.to_arrow()
    contents: List[str] = arrow.column("content").to_pylist()

    keep, drop = [], []
    for i, c in enumerate(contents):
        c = c or ""
        if admits(c) and not any(x in c for x in args.drop_extra):
            keep.append(i)
        else:
            drop.append(i)

    print(f"[l2-gate] {len(contents)} 条 → 保留 {len(keep)} / 剔除 {len(drop)}")
    print("\n--- 保留 ---")
    for i in keep:
        print(f"  + {contents[i][:88].replace(chr(10), ' / ')}")
    print("\n--- 剔除 ---")
    for i in drop:
        print(f"  - {contents[i][:88].replace(chr(10), ' / ')}")

    if args.dry_run:
        print("\n[dry-run] 未做任何修改")
        return 0
    if not drop:
        print("\n没有需要剔除的条目，未做修改")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.table}.bak_gate_{stamp}"
    db.create_table(backup, arrow, mode="overwrite")
    print(f"\n[backup] 已备份 → {backup}")

    kept = arrow.take(keep)
    db.create_table(args.table, kept, mode="overwrite")
    print(f"[done] {args.table} 现有 {db.open_table(args.table).count_rows()} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
