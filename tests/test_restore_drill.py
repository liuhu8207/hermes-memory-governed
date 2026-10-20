# -*- coding: utf-8 -*-
"""restore_drill：把备份恢复到隔离目录并校验。"""

from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "restore_drill.py"


def _load():
    spec = importlib.util.spec_from_file_location("restore_drill_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def drill_env(tmp_path, monkeypatch):
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")

    memory = tmp_path / "memory"
    (memory / "l2_backups").mkdir(parents=True)
    (memory / "l3_backups").mkdir()
    wiki = tmp_path / "wiki"
    backup = wiki / ".kb-backups-test"
    backup.mkdir(parents=True)
    (backup / "a.md").write_text("---\ntitle: A\n---\n\nbody\n", encoding="utf-8")
    (backup / "index.md").write_text("# nav\n", encoding="utf-8")

    # L2 backup: a real Lance table directory named memories.lance_<tag>
    db = lancedb.connect(str(memory / "l2_backups"))
    db.create_table("memories_dummy", schema=pa.schema([
        pa.field("content", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 2)),
    ]))
    (memory / "l2_backups" / "memories_dummy.lance").rename(
        memory / "l2_backups" / "memories.lance_test")

    # L3 backup: valid sqlite with messages + FTS
    l3 = memory / "l3_backups" / "l3_pre_redact_test.db"
    conn = sqlite3.connect(l3)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(content)")
    conn.execute("INSERT INTO messages(content) VALUES ('hello')")
    conn.execute("INSERT INTO messages_fts(rowid,content) VALUES (1,'hello')")
    conn.commit()
    conn.close()

    mod = _load()
    # 演练目录现在落在 HERMES_HOME/restore_drills（不是 %TEMP%）
    monkeypatch.setattr(mod, "HERMES_HOME", tmp_path / "home")
    monkeypatch.setattr(mod, "MEMORY_DIR", memory)
    monkeypatch.setattr(mod, "_wiki_dir", lambda: wiki)
    monkeypatch.setattr(sys, "argv", ["restore_drill.py"])
    return mod


def test_restore_drill_verifies_all_three_stores(drill_env, capsys, tmp_path):
    rc = drill_env.main()
    out = capsys.readouterr().out
    assert rc == 0, out
    assert '"l2"' in out and '"l3"' in out and '"vault"' in out
    assert '"malformed_frontmatter": 0' in out
    # 默认用后即删，不留下明文 L3 副本
    drills = tmp_path / "home" / "restore_drills"
    assert not drills.exists() or not list(drills.iterdir())
