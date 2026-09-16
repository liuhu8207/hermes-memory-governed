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


# ---------------------------------------------------------------------------
# Rebuild must not erase provenance
# ---------------------------------------------------------------------------
# A rebuild is the disaster-recovery path, so it is the last place that may
# quietly destroy attribution. It used to create a 6-column schema with no
# `role` / `agent` / `project`, after dropping the old table without a backup.
# Two consequences, in order:
#   1. every row lost its author;
#   2. `l2_apply_gate._row_roles` then found no `role` column and judged every
#      row as "user" — the strictest ruler — which deletes on the next gate run
#      the external-agent rows that were admitted on structural evidence.
# One rebuild therefore deleted other agents' memory twice over. These tests
# pin the schema, the backup and the carry-over.

_L2_REBUILD_DIM = 8


def _seed_l3(tmp_path, messages):
    """Create an L3 database holding ``messages`` as (role, content) pairs."""
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
    for i, (role, content) in enumerate(messages):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES ('s',?,?,?,?)", (role, content, now + i, f"h{i}")
        )
    conn.commit()
    conn.close()
    return l3_db


def _install_fake_embedding(monkeypatch, dim=_L2_REBUILD_DIM):
    """Make the rebuild's backend resolution succeed offline."""
    from plugin.memory_governed._embedding import EmbeddingService

    # A class body cannot assign `dim = dim`; the class attribute would shadow
    # the parameter. Bind it under another name instead.
    fake_dim = dim

    class _FakeService:
        available = True
        last_error = ""
        backend_name = "fake:test"
        dim = fake_dim

        def embed_batch(self, texts):
            return [[0.1] * fake_dim for _ in texts]

    monkeypatch.setattr(EmbeddingService, "get", staticmethod(lambda cfg: _FakeService()))


def _run_rebuild(tmp_path, monkeypatch, argv=None):
    """Import and run scripts/l2_rebuild.py against the tmp HERMES_HOME."""
    import importlib.util

    old_argv = sys.argv
    sys.argv = ["l2_rebuild.py", *(argv or ["--yes"])]
    try:
        spec = importlib.util.spec_from_file_location(
            "l2_rebuild", SCRIPTS_DIR / "l2_rebuild.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.main()
    finally:
        sys.argv = old_argv


def _prepare_rebuild(tmp_path, monkeypatch):
    """Wire env + fake embedding; return (home, l2_dir)."""
    l3_db = _seed_l3(tmp_path, [("user", "The server runs on port 8080 and uses Python")])
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("L3_DB_PATH", str(l3_db))
    monkeypatch.setenv("L2_DB_PATH", str(tmp_path / "l2"))
    _install_fake_embedding(monkeypatch)
    return tmp_path / "hermes", tmp_path / "l2"


class TestL2RebuildKeepsProvenance:
    def test_rebuilt_table_carries_all_three_attribution_columns(self, tmp_path, monkeypatch):
        """The schema is the bug: a fresh table must not be narrower than an old one."""
        import lancedb

        home, l2_dir = _prepare_rebuild(tmp_path, monkeypatch)
        assert _run_rebuild(tmp_path, monkeypatch) == 0

        table = lancedb.connect(str(l2_dir)).open_table("memories")
        names = [f.name for f in table.schema]
        for col in ("role", "agent", "project"):
            assert col in names, f"rebuilt table is missing '{col}': {names}"

    def test_rebuilt_rows_keep_the_role_they_came_from(self, tmp_path, monkeypatch):
        import lancedb

        _home, l2_dir = _prepare_rebuild(tmp_path, monkeypatch)
        assert _run_rebuild(tmp_path, monkeypatch) == 0

        arrow = lancedb.connect(str(l2_dir)).open_table("memories").to_arrow()
        assert arrow.column("role").to_pylist() == ["user"]
        assert arrow.column("agent").to_pylist() == ["hermes"]

    def test_drop_is_preceded_by_a_backup(self, tmp_path, monkeypatch):
        """No silent drops: the old table must still exist somewhere afterwards."""
        import lancedb

        home, l2_dir = _prepare_rebuild(tmp_path, monkeypatch)
        l2_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(l2_dir))
        db.create_table("memories", data=[{
            "content": "老的一行内容，重建前就存在",
            "category": "other", "source": "test",
            "timestamp": "2026-09-16T00:00:00", "vector": [0.9] * _L2_REBUILD_DIM,
            "source_rowid": 1, "role": "user", "agent": "hermes", "project": None,
        }])

        assert _run_rebuild(tmp_path, monkeypatch) == 0

        # lancedb spells the directory `memories.lance` (or `memories` on older
        # versions), so match either — the backup name keeps the spelling.
        backups = sorted((home / "memory" / "l2_backups").glob("memories*"))
        assert backups, "rebuild dropped the table without leaving a backup"
        assert any(backups[0].iterdir()), f"backup is empty: {backups[0]}"

    def test_external_agent_rows_survive_the_rebuild(self, tmp_path, monkeypatch):
        """Rows with no L3 origin cannot be re-derived, so they must be carried over.

        They are also the rows whose admission rests on structural evidence, so
        losing them is permanent — they would not pass the gate a second time.
        """
        import lancedb

        home, l2_dir = _prepare_rebuild(tmp_path, monkeypatch)
        l2_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(l2_dir))
        db.create_table("memories", data=[{
            "content": "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号",
            "category": "ops", "source": "external-write",
            "timestamp": "2026-09-16T00:00:00", "vector": [0.7] * _L2_REBUILD_DIM,
            "source_rowid": None, "role": "agent", "agent": "dsh", "project": "net",
        }])

        assert _run_rebuild(tmp_path, monkeypatch) == 0

        arrow = lancedb.connect(str(l2_dir)).open_table("memories").to_arrow()
        rows = {c: (r, p) for c, r, p in zip(arrow.column("content").to_pylist(),
                                             arrow.column("role").to_pylist(),
                                             arrow.column("agent").to_pylist())}
        external = "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号"
        assert external in rows, "external agent's fact was dropped by the rebuild"
        assert rows[external] == ("agent", "dsh"), \
            f"carried-over row lost its provenance: {rows[external]}"

    def test_rebuild_aborts_when_it_cannot_back_up(self, tmp_path, monkeypatch, caplog):
        """A backup failure must stop the run, not fall through to the drop."""
        import lancedb

        home, l2_dir = _prepare_rebuild(tmp_path, monkeypatch)
        l2_dir.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(l2_dir))
        db.create_table("memories", data=[{
            "content": "老的一行内容，重建前就存在",
            "category": "other", "source": "test",
            "timestamp": "2026-09-16T00:00:00", "vector": [0.9] * _L2_REBUILD_DIM,
        }])

        import shutil

        real_copytree = shutil.copytree

        def _boom(*args, **kwargs):
            raise OSError("disk on fire")

        monkeypatch.setattr(shutil, "copytree", _boom)
        try:
            assert _run_rebuild(tmp_path, monkeypatch) == 1
        finally:
            monkeypatch.setattr(shutil, "copytree", real_copytree)

        # The old table must still be there, untouched.
        assert lancedb.connect(str(l2_dir)).open_table("memories").count_rows() == 1


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
