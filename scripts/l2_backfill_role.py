#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""回填 L2 的 ``role`` 列（从 L3 归档经 ``source_rowid`` 反查）。

为什么需要它
------------
``_extract_atomic_facts`` 现在会把真实角色写进 ``role``，但**存量行**是
旧 schema 进来的（没有这一列），``_init_l2`` 只能补一个 NULL 列。

为什么必须在乎 role
-------------------
质量门是**角色相关**的：user 基线 +2、assistant 基线 -1。没有真实 role，
``l2_apply_gate.py`` 只能一律按 user 判定 —— 那是**有意偏松**（宁可漏删
不误杀），代价是助手的过程叙述会残留。回填之后回放才能用回真实规则。

数据来源
--------
L2 ``source_rowid`` → L3 ``messages.rowid``（写入时由 ``_resolve_source_rowid``
建立）。取不到映射的行保持 NULL —— **不做内容模糊匹配**，宁缺毋滥。

用法::

    python scripts/l2_backfill_role.py --dry-run     # 预演，只看能回填多少
    python scripts/l2_backfill_role.py               # 执行（自动备份）
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Optional

import lancedb
import pyarrow as pa

DEFAULT_L2 = Path.home() / "AppData" / "Local" / "hermes" / "memory" / "l2"
DEFAULT_L3 = Path.home() / "AppData" / "Local" / "hermes" / "memory" / "l3" / "l3.db"

#: 这些角色才有资格进 L2（见 _extract_atomic_facts 的 role 过滤）
_L3_ROLES = ("user", "assistant")


def load_l3_roles(l3_path: Path, rowids: List[int]) -> Dict[int, str]:
    """批量取 ``rowid -> role``。查不到的直接缺席。"""
    if not rowids:
        return {}
    out: Dict[int, str] = {}
    conn = sqlite3.connect(str(l3_path))
    try:
        # 分批查询，避免 SQL 变量数量上限
        batch = 400
        for i in range(0, len(rowids), batch):
            chunk = rowids[i:i + batch]
            placeholders = ",".join("?" * len(chunk))
            sql = (f"SELECT rowid, role FROM messages "
                   f"WHERE rowid IN ({placeholders})")
            for rid, role in conn.execute(sql, chunk):
                if role in _L3_ROLES:
                    out[int(rid)] = str(role)
    finally:
        conn.close()
    return out


def load_l3_corpus(l3_path: Path) -> List[tuple]:
    """``[(role, content)]`` —— 供内容反查用（L2 存的是句子，L3 存整条消息）。"""
    conn = sqlite3.connect(str(l3_path))
    try:
        return [
            (str(role), content or "")
            for role, content in conn.execute(
                "SELECT role, content FROM messages WHERE role IN ('user','assistant')"
            )
        ]
    finally:
        conn.close()


def infer_role_by_content(corpus: List[tuple], needle: str) -> Optional[str]:
    """按内容反查角色：**只有在唯一角色命中时才采纳**。

    L2 的 ``content`` 是句子，L3 的 ``content`` 是整条消息，所以前者是后者
    的子串。同一句话若 user 和 assistant 都说过（很常见 —— 助手复述用户的
    约束），角色就有歧义，此时返回 None 保持 NULL：宁可少回填，也不能给
    质量门喂一个错的角色（那会让回放按错规则删数据）。
    """
    needle = (needle or "").strip()
    if len(needle) < 8:  # 太短易误匹配
        return None
    roles = {role for role, content in corpus if needle in content}
    return roles.pop() if len(roles) == 1 else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Backfill L2.role from the L3 archive.")
    ap.add_argument("--l2-dir", default=str(DEFAULT_L2))
    ap.add_argument("--l3-db", default=str(DEFAULT_L3))
    ap.add_argument("--table", default="memories")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    db = lancedb.connect(args.l2_dir)
    table = db.open_table(args.table)
    arrow = table.to_arrow()

    col_names = [f.name for f in arrow.schema]
    if "role" not in col_names:
        print(f"[backfill] L2 无 role 列，先补空列（schema 迁移由 _init_l2 完成）")
        arrow = arrow.append_column("role", pa.array([None] * arrow.num_rows,
                                                     type=pa.string()))

    roles: List[Optional[str]] = list(arrow.column("role").to_pylist())
    has_ref = "source_rowid" in [f.name for f in arrow.schema]
    refs: List[Optional[int]] = (
        list(arrow.column("source_rowid").to_pylist()) if has_ref
        else [None] * arrow.num_rows
    )
    contents: List[str] = list(arrow.column("content").to_pylist())

    need = [int(ref) for r, ref in zip(roles, refs)
            if (r is None or r == "") and ref is not None]
    mapping = load_l3_roles(Path(args.l3_db), need)

    filled_ref = 0
    for i, (cur, ref) in enumerate(zip(roles, refs)):
        if cur:
            continue
        if ref is not None and int(ref) in mapping:
            roles[i] = mapping[int(ref)]
            filled_ref += 1

    # 第二轮：没有 source_rowid 的早期行，按内容在 L3 里反查
    # （唯一角色命中才采纳，见 infer_role_by_content）
    filled_content = 0
    unknown_idx = [i for i, r in enumerate(roles) if not r]
    if unknown_idx:
        corpus = load_l3_corpus(Path(args.l3_db))
        for i in unknown_idx:
            inferred = infer_role_by_content(corpus, str(contents[i] or ""))
            if inferred:
                roles[i] = inferred
                filled_content += 1

    filled = filled_ref + filled_content
    total = arrow.num_rows
    already = sum(1 for r in roles if r)
    print(f"[backfill] 共 {total} 行")
    print(f"  - 已有 role                    : {already - filled}")
    print(f"  - 本次回填（source_rowid → L3）: {filled_ref}")
    print(f"  - 本次回填（内容反查，唯一角色）: {filled_content}")
    print(f"  - 仍未知(NULL)                 : {total - already}")

    if filled:
        print("\n--- 回填明细 ---")
        for i, (r, c) in enumerate(zip(roles, contents)):
            if r and i < len(contents):
                print(f"  [{r:9s}] {str(c)[:76].replace(chr(10), ' / ')}")

    if args.dry_run:
        print("\n[dry-run] 未做任何修改")
        return 0
    if filled == 0:
        print("\n没有可回填的行，未做修改")
        return 0

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{args.table}.bak_role_{stamp}"
    db.create_table(backup, arrow, mode="overwrite")
    print(f"\n[backup] 已备份 → {backup}")

    idx = arrow.schema.get_field_index("role")
    updated = arrow.set_column(idx, "role", pa.array(roles, type=pa.string()))
    db.create_table(args.table, updated, mode="overwrite")
    print(f"[done] {args.table} 现有 {db.open_table(args.table).count_rows()} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
