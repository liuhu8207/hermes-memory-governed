# -*- coding: utf-8 -*-
"""kb_index_optimize 的三条安全契约。

1. **非交互环境（cron / CI）必须显式 ``--yes``** —— 没人能确认，就不该动手。
2. **动手之前先备份** —— ``optimize`` 会重写表目录，失败时要有回退点。
3. **绝不删 schema 迁移留下的 ``kb_notes.bak_<时间戳>``** —— 那是**老索引的
   唯一副本**（旧表 drop 之后只剩它）。旧实现会在收尾时把它们 prune 掉。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "kb_index_optimize.py"


def _load():
    spec = importlib.util.spec_from_file_location("kbo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_index(tmp_path):
    """造一张真实的 3 行表 + 3 个"迁移备份"目录。"""
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")

    memory = tmp_path / "memory"
    index = memory / "kb_index"
    index.mkdir(parents=True)
    db = lancedb.connect(str(index))
    table = db.create_table("kb_notes", schema=pa.schema([
        pa.field("path", pa.string()),
        pa.field("text", pa.string()),
        pa.field("mtime", pa.float64()),
        pa.field("vector", pa.list_(pa.float32(), 2)),
    ]))
    for i in range(3):
        table.add([{"path": "n%d" % i, "text": "t", "mtime": 1.0,
                    "vector": [0.1, 0.2]}])
    for name in ("kb_notes.bak_1.lance", "kb_notes.bak_2.lance",
                 "kb_notes.bak_3.lance"):
        (index / name).mkdir()
    return memory, index, db


def _run(mod, memory, monkeypatch, argv):
    monkeypatch.setattr(mod, "MEMORY_DIR", memory)
    monkeypatch.setattr(mod, "env_path", lambda env, default: default)
    monkeypatch.setattr(sys, "argv", argv)
    return mod.main()


def test_optimize_compacts_and_keeps_rows(tmp_path, monkeypatch):
    memory, index, db = _build_index(tmp_path)
    mod = _load()

    rc = _run(mod, memory, monkeypatch,
              ["kb_index_optimize.py", "--days", "0", "--yes"])

    assert rc == 0
    remaining = db.open_table("kb_notes")
    assert remaining.count_rows() == 3


def test_migration_backups_are_never_pruned(tmp_path, monkeypatch):
    """schema 迁移留下的备份是**老索引的唯一副本**，优化脚本绝不能删。"""
    memory, index, db = _build_index(tmp_path)
    mod = _load()

    rc = _run(mod, memory, monkeypatch,
              ["kb_index_optimize.py", "--days", "0", "--yes",
               "--keep-index-backups", "1"])

    assert rc == 0
    # 只数**迁移**备份（optimize 自己那份副本不算，它另有归属）。
    backups = [p for p in index.glob("kb_notes.bak_*.lance")
               if ".bak_opt_" not in p.name]
    assert len(backups) == 3, "迁移备份一个都不许删"


def test_optimize_backs_up_the_table_before_compacting(tmp_path, monkeypatch):
    """动手之前先备份（optimize 会重写表目录，失败时要有回退点）。"""
    memory, index, db = _build_index(tmp_path)
    mod = _load()

    rc = _run(mod, memory, monkeypatch,
              ["kb_index_optimize.py", "--days", "0", "--yes"])

    assert rc == 0
    opt_backups = list(index.glob("kb_notes.bak_opt_*.lance"))
    assert len(opt_backups) == 1, "optimize 之前必须留一份整表副本"


def test_non_interactive_without_yes_refuses(tmp_path, monkeypatch):
    """没有 TTY 又没有 --yes ⇒ 拒绝执行（既没人授权，也没人知道它发生过）。"""
    memory, index, db = _build_index(tmp_path)
    mod = _load()
    # 非交互：stdin/stdout 都不是 TTY（pytest 捕获输出即如此）。
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

    rc = _run(mod, memory, monkeypatch,
              ["kb_index_optimize.py", "--days", "0"])

    assert rc != 0, "非交互环境没有 --yes 必须拒绝"
    assert not list(index.glob("kb_notes.bak_opt_*.lance")), \
        "被拒绝时不得留下任何备份/优化痕迹"


def test_dry_run_never_needs_confirmation(tmp_path, monkeypatch):
    """--dry-run 是只读的，不该被确认门拦住。"""
    memory, index, db = _build_index(tmp_path)
    mod = _load()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False, raising=False)

    rc = _run(mod, memory, monkeypatch,
              ["kb_index_optimize.py", "--dry-run"])

    assert rc == 0
    assert not list(index.glob("kb_notes.bak_opt_*.lance"))
