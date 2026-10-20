# -*- coding: utf-8 -*-
"""kb_index_optimize keeps the table usable and prunes old table backups."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "kb_index_optimize.py"


def test_optimize_compacts_and_keeps_rows(tmp_path, monkeypatch):
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
    # 两个旧表备份
    for name in ("kb_notes.bak_1.lance", "kb_notes.bak_2.lance",
                 "kb_notes.bak_3.lance"):
        (index / name).mkdir()

    spec = importlib.util.spec_from_file_location("kbo", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "MEMORY_DIR", memory)
    monkeypatch.setattr(mod, "env_path", lambda env, default: default)
    monkeypatch.setattr(sys, "argv", ["kb_index_optimize.py", "--days", "0",
                                      "--keep-index-backups", "1"])

    rc = mod.main()
    assert rc == 0
    remaining = db.open_table("kb_notes")
    assert remaining.count_rows() == 3
    backups = list(index.glob("kb_notes.bak_*.lance"))
    assert len(backups) == 1
