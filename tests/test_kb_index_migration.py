# -*- coding: utf-8 -*-
"""KBIndex 老 schema 迁移：真实 LanceDB，不用假对象。

复评指出 ``test_kb_reindex_guard`` 用 ``_FakeKB``，从没跑过真实
``sync_index`` 与 schema 迁移，于是 ``rename_table`` 在 OSS 上必然失败的
分支没被测出。这里直接建一张无 mtime 的老表，走真实迁移。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KBIndex

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class _FakeService:
    available = True
    dim = 4
    model_name = "fake"

    def embed_one(self, _text):
        return [0.1, 0.2, 0.3, 0.4]


def test_old_schema_is_copied_and_rebuilt(tmp_path, monkeypatch):
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")

    db_path = tmp_path / "kb_index"
    db_path.mkdir()
    db = lancedb.connect(str(db_path))
    db.create_table("kb_notes", schema=pa.schema([
        pa.field("path", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
    ]))
    assert (db_path / "kb_notes.lance").is_dir()

    monkeypatch.setattr(
        "plugin.memory_governed._kb.EmbeddingService.get",
        staticmethod(lambda cfg: _FakeService()))

    cfg = GovernedMemoryConfig()
    cfg.embedding.provider = "fake"
    cfg.embedding.model = "fake"
    idx = KBIndex(cfg, db_path)

    assert idx.available is True, idx.last_error
    backups = list(db_path.glob("kb_notes.bak_*"))
    assert len(backups) == 1, "迁移前必须留下旧表副本"
    assert backups[0].is_dir()

    fresh = lancedb.connect(str(db_path)).open_table("kb_notes")
    assert "mtime" in [f.name for f in fresh.schema]


def test_migration_failure_is_attributed_to_migration(tmp_path, monkeypatch):
    """迁移失败不能被报成 embedding 故障。"""
    lancedb = pytest.importorskip("lancedb")
    pa = pytest.importorskip("pyarrow")

    db_path = tmp_path / "kb_index"
    db_path.mkdir()
    db = lancedb.connect(str(db_path))
    db.create_table("kb_notes", schema=pa.schema([
        pa.field("path", pa.string()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 4)),
    ]))

    monkeypatch.setattr(
        "plugin.memory_governed._kb.EmbeddingService.get",
        staticmethod(lambda cfg: _FakeService()))

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("shutil.copytree", boom)
    idx = KBIndex(GovernedMemoryConfig(), db_path)

    assert idx.available is False
    assert "schema migration failed" in idx.last_error
    # 旧表必须还在：失败时不许 drop
    assert (db_path / "kb_notes.lance").is_dir()
