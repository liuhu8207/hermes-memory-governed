# -*- coding: utf-8 -*-
"""Architecture-level tests (2026-08-28 optimization batch).

Covers the features added in the architecture review round:
- L3 idempotent writes (content_hash dedup)
- L2 source_rowid references + audit drill-down
- Bridge import_approved (review → L1 landing)
- Bridge auto-archiving
- Config type coercion (VectorConfig + numeric fields)
- recall_status cache hit-rate
- scripts: l2_rebuild --dry-run, l3_retention
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    from plugin.memory_governed import GovernedMemoryProvider

    p = GovernedMemoryProvider()
    p.initialize("s1", hermes_home=str(tmp_path / "hermes"))
    yield p, tmp_path / "hermes"
    p.shutdown()


@pytest.fixture
def l3_db(tmp_path):
    """L3 db with two messages (one strong python match, one weak)."""
    db = tmp_path / "l3.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
    )
    now = time.time()
    conn.execute(
        "INSERT INTO messages VALUES (1,'s','user','python python python deep dive',?,NULL,'h1')",
        (now,),
    )
    conn.execute(
        "INSERT INTO messages_fts VALUES ('python python python deep dive','user','s',?)",
        (str(now),),
    )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# L3 idempotent writes
# ---------------------------------------------------------------------------

class TestL3Idempotent:
    def test_duplicate_sync_turn_archives_once(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        from plugin.memory_governed import GovernedMemoryProvider

        p = GovernedMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path / "hermes"))
        msgs = [{"role": "user", "content": "我们项目的部署方案确定了，用 Docker Compose 部署"}]
        p.sync_turn("q", "a", session_id="s-deploy", messages=msgs)
        p.sync_turn("q", "a", session_id="s-deploy", messages=msgs)  # duplicate

        conn = sqlite3.connect(str(tmp_path / "hermes" / "memory" / "l3" / "l3.db"))
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        hashes = conn.execute("SELECT hash FROM messages").fetchall()
        conn.close()
        assert count == 1
        assert all(h[0] for h in hashes)  # hash column filled

    def test_write_returns_rowid_map(self, tmp_path):
        from plugin.memory_governed._sync import L3Writer
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.l3_db_path = str(tmp_path / "l3.db")
        w = L3Writer(cfg)
        w.write([{"role": "user", "content": "hello world"}], session_id="s")
        rowid_map = w.write([{"role": "user", "content": "second message"}], session_id="s")
        w.shutdown()
        assert "second message" in rowid_map
        assert isinstance(rowid_map["second message"], int)


# ---------------------------------------------------------------------------
# L2 source_rowid references
# ---------------------------------------------------------------------------

class TestL2SourceRef:
    def test_schema_has_source_rowid(self, tmp_path, monkeypatch):
        pytest.importorskip("lancedb")
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._config import GovernedMemoryConfig

        # 本测试只验证 schema 列名，不关心 embedding；跳过探测避免触发模型
        # 下载/fallback（拖慢且会覆盖 config.vector.dim）。
        monkeypatch.setattr(WriteQueue, "_init_embedding_service", lambda self: None)

        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "l2")
        q = WriteQueue(cfg)
        q._init_l2()
        assert q._l2_store is not None
        names = [f.name for f in q._l2_store.schema]
        assert "source_rowid" in names
        assert "vector" in names

    def test_config_driven_dim(self, tmp_path, monkeypatch):
        pytest.importorskip("lancedb")
        import pyarrow as pa
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._config import GovernedMemoryConfig

        # 本测试验证 config.vector.dim 驱动建表维度；跳过 embedding 探测，
        # 否则探测会 fallback 并覆盖 config.vector.dim。
        monkeypatch.setattr(WriteQueue, "_init_embedding_service", lambda self: None)

        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "l2")
        cfg.vector.dim = 128
        q = WriteQueue(cfg)
        q._init_l2()
        vfield = next(f for f in q._l2_store.schema if f.name == "vector")
        assert vfield.type.list_size == 128


# ---------------------------------------------------------------------------
# Bridge import_approved + archiving
# ---------------------------------------------------------------------------

class TestBridgeImport:
    def test_approved_lands_in_l1(self, tmp_path):
        from plugin.memory_governed._bridge import BridgeExporter
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.bridge_dir = str(tmp_path / "bridge")
        l1 = tmp_path / "MEMORY.md"
        l1.write_text("# Rules\n", encoding="utf-8")

        be = BridgeExporter(cfg)
        be.export_candidates([
            {"content": "Approved rule: deploy with Docker", "target": "memory", "tags": ["approved"]},
            {"content": "Unreviewed rule: maybe k8s", "target": "memory", "tags": ["review-required"]},
        ])
        report = be.import_approved(l1)
        assert report["imported"] == 1

        content = l1.read_text(encoding="utf-8")
        assert "Approved rule" in content
        assert "maybe k8s" not in content
        assert "bridge:imported" in content  # provenance comment

    def test_import_skips_already_imported(self, tmp_path):
        from plugin.memory_governed._bridge import BridgeExporter
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.bridge_dir = str(tmp_path / "bridge")
        l1 = tmp_path / "MEMORY.md"
        l1.write_text("# Rules\n", encoding="utf-8")

        be = BridgeExporter(cfg)
        be.export_candidates([{"content": "Approved rule: use pytest", "target": "memory", "tags": ["approved"]}])
        r1 = be.import_approved(l1)
        r2 = be.import_approved(l1)  # second pass — already marked imported
        assert r1["imported"] == 1
        assert r2["imported"] == 0
        assert r2["skipped"] >= 1

    def test_auto_archive_at_limit(self, tmp_path):
        from plugin.memory_governed._bridge import BridgeExporter
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.bridge_dir = str(tmp_path / "bridge")
        be = BridgeExporter(cfg)
        be.MAX_JSONL_LINES = 3

        for i in range(5):
            be.export_candidates([
                {"content": f"Unique bridge candidate {i} with enough length", "target": "memory"}
            ])
        archive_dir = tmp_path / "bridge" / "archive"
        assert archive_dir.exists()
        assert len(list(archive_dir.glob("*.jsonl"))) >= 1


# ---------------------------------------------------------------------------
# Config type coercion + VectorConfig
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_numeric_coercion(self, tmp_path):
        from plugin.memory_governed._config import load_governed_config

        cfg_file = tmp_path / "governed_memory.json"
        cfg_file.write_text(json.dumps({"recall": {"l1_budget_tokens": "800"}}), encoding="utf-8")
        cfg = load_governed_config(tmp_path)
        assert cfg.recall.l1_budget_tokens == 800
        assert isinstance(cfg.recall.l1_budget_tokens, int)

    def test_vector_config(self, tmp_path):
        from plugin.memory_governed._config import load_governed_config

        cfg_file = tmp_path / "governed_memory.json"
        cfg_file.write_text(
            json.dumps({"vector": {"model": "bge-small-zh", "dim": 512}}), encoding="utf-8"
        )
        cfg = load_governed_config(tmp_path)
        assert cfg.vector.model == "bge-small-zh"
        assert cfg.vector.dim == 512

    def test_vector_defaults(self):
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        assert cfg.vector.model == "BAAI/bge-small-zh-v1.5"
        assert cfg.vector.dim == 512


# ---------------------------------------------------------------------------
# recall_status hit-rate
# ---------------------------------------------------------------------------

class TestRecallStatus:
    def test_hit_rate(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        from plugin.memory_governed import GovernedMemoryProvider

        p = GovernedMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path / "hermes"))
        p._prefetch_cache["q"] = ("cached text", time.time())
        p.prefetch("q")        # hit
        p.prefetch("q")        # hit
        p.prefetch("other")    # miss
        st = p.recall_status()
        assert st.cache_hits == 2
        assert st.cache_misses == 1
        assert abs(st.cache_hit_rate - 2 / 3) < 1e-6


# ---------------------------------------------------------------------------
# Maintenance scripts
# ---------------------------------------------------------------------------

class TestL2Rebuild:
    def test_dry_run(self, tmp_path, monkeypatch):
        # Build L3 with two extractable messages
        l3_db = tmp_path / "l3.db"
        conn = sqlite3.connect(str(l3_db))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
        )
        now = time.time()
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES ('s','user','The server runs on port 8080 and uses Python',?,'h1')", (now,)
        )
        conn.commit()
        conn.close()

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("L3_DB_PATH", str(l3_db))
        monkeypatch.setenv("L2_DB_PATH", str(tmp_path / "l2"))

        import importlib.util

        old_argv = sys.argv
        sys.argv = ["l2_rebuild.py", "--dry-run"]
        try:
            spec = importlib.util.spec_from_file_location(
                "l2_rebuild", SCRIPTS_DIR / "l2_rebuild.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            assert mod.main() == 0
        finally:
            sys.argv = old_argv


class TestL3Retention:
    def test_prune_removes_old_and_keeps_fresh(self, tmp_path, monkeypatch):
        l3_db = tmp_path / "l3.db"
        conn = sqlite3.connect(str(l3_db))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
        )
        now = time.time()
        old_ts = now - 200 * 86400
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES ('s','user','old message 180 days ago',?,'h1')", (old_ts,)
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES ('s','user','fresh message today',?,'h2')", (now,)
        )
        conn.execute(
            "INSERT INTO messages_fts VALUES ('old message 180 days ago','user','s',?)",
            (str(old_ts),),
        )
        conn.execute(
            "INSERT INTO messages_fts VALUES ('fresh message today','user','s',?)", (str(now),)
        )
        conn.commit()
        conn.close()

        import importlib.util

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("L3_DB_PATH", str(l3_db))
        spec = importlib.util.spec_from_file_location(
            "l3_retention", SCRIPTS_DIR / "l3_retention.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.argv = ["l3_retention.py", "--days", "180", "--yes"]
        assert mod.main() == 0

        conn = sqlite3.connect(str(l3_db))
        rows = [r[0] for r in conn.execute("SELECT content FROM messages").fetchall()]
        fts = [r[0] for r in conn.execute("SELECT content FROM messages_fts").fetchall()]
        conn.close()
        assert rows == ["fresh message today"]
        assert fts == ["fresh message today"]


# ---------------------------------------------------------------------------
# L3 min-max normalization direction
# ---------------------------------------------------------------------------

class TestL3ScoreDirection:
    def test_strong_match_ranks_first(self, tmp_path):
        from plugin.memory_governed._recall import RecallEngine
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.l3_db_path = str(tmp_path / "l3.db")
        conn = sqlite3.connect(str(tmp_path / "l3.db"))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
        )
        now = time.time()
        conn.execute(
            "INSERT INTO messages VALUES (1,'s','user','python python python deep dive',?)", (now,)
        )
        conn.execute(
            "INSERT INTO messages_fts VALUES ('python python python deep dive','user','s',?)",
            (str(now),),
        )
        conn.execute(
            "INSERT INTO messages VALUES (2,'s','user','python mentioned once here',?)", (now,)
        )
        conn.execute(
            "INSERT INTO messages_fts VALUES ('python mentioned once here','user','s',?)",
            (str(now),),
        )
        conn.commit()
        conn.close()

        e = RecallEngine(cfg)
        res = e._search_l3("python")
        assert len(res) >= 2
        assert "deep dive" in res[0].content  # strongest match first
        assert res[0].score >= res[-1].score
