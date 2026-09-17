#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用**当前**质量门回放整张 L2 表，剔除不再符合准入条件的存量条目。

为什么需要它
------------
质量门（``WriteQueue._looks_like_fact`` + ``_sync.dialogue_fact_admits``，
后者与写入路径 ``_extract_atomic_facts`` 逐字同源）
只作用于**新写入**。规则收紧之后，历史存量仍是按旧标准进来的，必须回头清一遍，
否则它们会继续参与向量召回、抬高无关查询的地板分。

role 的真实性
-------------
L2 现在会落真实 ``role``（``_extract_atomic_facts`` 写入），存量行用
``scripts/l2_backfill_role.py`` 从 L3 回填。本脚本按真实角色回放 —— 这很
关键，因为 user 基线 +2、assistant 基线 -1，用错角色的门槛会得出相反结论。

若 role 尚未回填（列缺失 / 为空），本脚本**回退为按 user 判定**并在输出里
明确提示：这是有意的偏松（宁可漏删不误杀），代价是助手叙述可能残留。

用法::

    python scripts/l2_apply_gate.py --dry-run          # 预演，打印保留/剔除清单
    python scripts/l2_apply_gate.py                    # 执行（自动备份）
    python scripts/l2_apply_gate.py --drop-extra "就是你给我的" --drop-extra "阵列卡"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path
from typing import List

import lancedb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plugin.memory_governed._sync import (  # noqa: E402
    WriteQueue,
    dialogue_fact_admits,
    external_write_verdict,
)


def admits(text: str, role: str = "user") -> bool:
    """当前质量门是否会让这条内容进入 L2（按**真实角色**判定）。

    role 影响很大：user 与 assistant 的内容门槛不同（`_MIN_SIGNAL_USER` /
    `_MIN_SIGNAL_ASSISTANT`）。回填 role 之前这里只能一律按 user 走（偏松，
    助手叙述会漏删）。

    ``role == "agent"`` 是外部 agent 经 ``memory_cli.py remember`` 写入的行，
    必须走**同一个** ``external_write_verdict``。否则回放会用对话角色的信号
    门槛去判它们，把靠结构证据进来的行全部误删 —— 例如
    「示例主路由 192.0.2.1 的 SSH 端口是 8022」，信号分 0，但它含 IP /
    专有名词 / 端口，正是该层要留的东西。

    对话角色则走**同一个** ``dialogue_fact_admits`` —— 与 ``_extract_atomic_facts``
    逐字同源。写入与回放必须用同一把尺子，所以尺子只有一处定义。
    """
    if role == "agent":
        return external_write_verdict(text)[0]
    if not WriteQueue._looks_like_fact(text, role):
        return False
    return dialogue_fact_admits(text, role)


def _default_l2_dir() -> str:
    """Locate the L2 directory without hardcoding one machine's layout."""
    home = os.environ.get("HERMES_HOME")
    if not home:
        local = Path(os.environ.get("LOCALAPPDATA", "")) / "hermes"
        home = str(local if local.exists() else Path.home() / ".hermes")
    return str(Path(home) / "memory" / "l2")


def _row_roles(arrow) -> List[str]:
    """取每行的真实 role；缺列/为空时回退 ``user``（旧行为，偏松）。"""
    names = [f.name for f in arrow.schema]
    if "role" not in names:
        print("[l2-gate] L2 无 role 列 —— 全部按 user 判定（偏松）。"
              "可先跑 scripts/l2_backfill_role.py 回填。")
        return ["user"] * arrow.num_rows
    roles = []
    missing = 0
    for v in arrow.column("role").to_pylist():
        if v in ("user", "assistant", "agent"):
            roles.append(str(v))
        else:
            roles.append("user")
            missing += 1
    if missing:
        print(f"[l2-gate] {missing} 行 role 为空 → 按 user 判定（偏松）")
    return roles


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay the L2 quality gate over existing rows.")
    ap.add_argument("--l2-dir", default=_default_l2_dir())
    ap.add_argument("--table", default="memories")
    ap.add_argument("--drop-extra", action="append", default=[],
                    help="额外剔除的内容子串（人工补刀），可重复")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    db = lancedb.connect(args.l2_dir)
    table = db.open_table(args.table)
    arrow = table.to_arrow()
    contents: List[str] = arrow.column("content").to_pylist()
    roles = _row_roles(arrow)

    keep, drop = [], []
    for i, c in enumerate(contents):
        c = c or ""
        if admits(c, roles[i]) and not any(x in c for x in args.drop_extra):
            keep.append(i)
        else:
            drop.append(i)

    print(f"[l2-gate] {len(contents)} 条 → 保留 {len(keep)} / 剔除 {len(drop)}")
    print("\n--- 保留 ---")
    for i in keep:
        print(f"  + [{roles[i]:9s}] {contents[i][:80].replace(chr(10), ' / ')}")
    print("\n--- 剔除 ---")
    for i in drop:
        print(f"  - [{roles[i]:9s}] {contents[i][:80].replace(chr(10), ' / ')}")

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
