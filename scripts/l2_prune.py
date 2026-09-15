#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""按内容精确匹配剔除 L2 里的残留噪声条目（存量清理工具）。

用途
----
L2 事实质量门（``_sync._looks_like_fact``）只拦**新写入**的噪声，管不到
历史上已经进库的条目。本脚本按 ``--drop`` 给出的内容（支持子串匹配）
删除存量噪声，并自动：

1. 删除前把当前表另存为 ``memories.bak_<TS>.lance``（保留 ``.lance`` 后缀，
   否则 ``open_table`` 找不到它）；
2. 用固定大小的 FixedSizeList(1024) 重建 schema —— 直接 ``to_pandas`` 再
   ``create_table`` 会把向量退化成 list，导致后续写入维度不匹配；
3. ``--dry-run`` 预演，打印待删条目与存活条目，不落盘。

用法::

    python scripts/l2_prune.py --l2-dir ~/.hermes/memory/l2 --dry-run
    python scripts/l2_prune.py --drop "还没想好" --drop "- L1手写规则层空的"

注意：这是**不可逆**操作（虽然有备份）。执行前务必先跑 ``--dry-run``。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path
from typing import List

import lancedb
import pyarrow as pa


def _backup(db, table_name: str) -> str:
    tbl = db.open_table(table_name)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{table_name}.bak_{stamp}"
    # 必须保留 .lance 后缀：LanceDB 按目录名发现表
    db.create_table(backup_name, tbl.to_arrow(), mode="overwrite")
    return backup_name


def _prune(l2_dir: str, drops: List[str], dry_run: bool, table: str) -> int:
    db = lancedb.connect(l2_dir)
    tbl = db.open_table(table)
    arrow = tbl.to_arrow()
    contents: List[str] = arrow.column("content").to_pylist()

    keep_idx, drop_idx = [], []
    for i, c in enumerate(contents):
        hit = next((d for d in drops if d in (c or "")), None)
        (drop_idx if hit else keep_idx).append(i)

    print(f"[l2-prune] 当前 {len(contents)} 条 → 待删 {len(drop_idx)} 条，保留 {len(keep_idx)} 条")
    print("\n--- 待删 ---")
    for i in drop_idx:
        print(f"  [{i}] {(contents[i] or '').replace(chr(10), ' / ')[:100]}")
    print("\n--- 保留 ---")
    for i in keep_idx:
        print(f"  [{i}] {(contents[i] or '').replace(chr(10), ' / ')[:100]}")

    if dry_run:
        print("\n[dry-run] 未做任何修改")
        return 0

    if not drop_idx:
        print("\n没有匹配到任何条目，未做修改")
        return 0

    backup_name = _backup(db, table)
    print(f"\n[backup] 已备份 → {backup_name}")

    # 保留原 schema（尤其 vector 的 FixedSizeList），只过滤行
    kept = arrow.take(keep_idx)
    db.create_table(table, kept, mode="overwrite")
    print(f"[done] {table} 现有 {db.open_table(table).count_rows()} 条")
    return 0


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prune residual noise rows from L2.")
    ap.add_argument("--l2-dir", default=str(Path.home() / ".hermes" / "memory" / "l2"))
    ap.add_argument("--table", default="memories")
    ap.add_argument("--drop", action="append", default=[],
                    help="要删除的内容子串，可重复传入")
    ap.add_argument("--drop-file", default="",
                    help="每行一个待删子串的文本文件")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    drops = list(args.drop)
    if args.drop_file:
        drops += [ln.strip() for ln in Path(args.drop_file).read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    if not drops:
        print("错误：至少需要一个 --drop 或 --drop-file", file=sys.stderr)
        return 2
    return _prune(args.l2_dir, drops, args.dry_run, args.table)


if __name__ == "__main__":
    raise SystemExit(main())
