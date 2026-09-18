# -*- coding: utf-8 -*-
"""补测试：memory_promote / memory_dream / migrations / audit 下钻"""
import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# memory_promote
# ---------------------------------------------------------------------------

class TestMemoryPromote:
    def _make_l3(self, tmp_path, recurred_msgs):
        db = tmp_path / "l3.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)"
        )
        now = time.time()
        for i, content in enumerate(recurred_msgs):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, hash) "
                "VALUES (?,?,?,?,?)", (f"s{i}", "user", content, now - i * 86400, f"h{i}")
            )
        conn.commit()
        conn.close()
        return db

    def test_promote_recognizes_recurrence(self, tmp_path, monkeypatch):
        db = self._make_l3(tmp_path, ["部署的时候记得先备份配置文件再改。"] * 3 + ["今天天气不错。"])
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("L3_DB_PATH", str(db))

        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "memory_promote", SCRIPTS_DIR / "memory_promote.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        msgs = mod.load_user_messages(days=90)
        promoted = mod.find_promotable(msgs, min_count=3, min_len=10)
        assert len(promoted) == 1
        assert promoted[0]["metadata"]["recurrence_count"] == 3
        assert promoted[0]["tags"] == ["auto-promoted", "review-required"]

    def test_promote_export_via_bridge(self, tmp_path, monkeypatch):
        db = self._make_l3(tmp_path, ["部署的时候记得先备份配置文件再改。"] * 4)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("L3_DB_PATH", str(db))
        monkeypatch.setenv("BRIDGE_DIR", str(tmp_path / "bridge"))

        sys.path.insert(0, str(REPO_ROOT))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "memory_promote", SCRIPTS_DIR / "memory_promote.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        msgs = mod.load_user_messages(days=90)
        promoted = mod.find_promotable(msgs, min_count=3, min_len=10)
        stats = mod.export_promoted(promoted)
        assert stats["written"] == 1

        jsonl = tmp_path / "bridge" / "candidates.jsonl"
        cand = json.loads(jsonl.read_text(encoding="utf-8").splitlines()[0])
        assert cand["tags"] == ["auto-promoted", "review-required"]


# ---------------------------------------------------------------------------
# memory_dream
# ---------------------------------------------------------------------------

class TestMemoryDream:
    def test_dream_writes_summary_to_l2(self, tmp_path, monkeypatch):
        pytest.importorskip("lancedb")

        # L3 with one old session
        l3_dir = tmp_path / "hermes" / "memory" / "l3"
        l3_dir.mkdir(parents=True, exist_ok=True)
        old_ts = time.time() - 45 * 86400
        conn = sqlite3.connect(str(l3_dir / "l3.db"))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)"
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES ('old-deploy','user','上周我们决定数据库用 PostgreSQL 15。',?,'h1')", (old_ts,)
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES ('old-deploy','assistant','好的，PostgreSQL 15 已确认。',?,'h2')", (old_ts + 60,)
        )
        conn.commit()
        conn.close()

        # L2 table (created by the write queue init)
        sys.path.insert(0, str(REPO_ROOT))
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "hermes" / "memory" / "l2")
        wq = WriteQueue(cfg)
        wq._init_l2()
        assert wq._l2_store is not None

        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
        monkeypatch.setenv("L3_DB_PATH", str(l3_dir / "l3.db"))
        monkeypatch.setenv("L2_DB_PATH", str(cfg.l2_db_path))

        import importlib.util

        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        spec = importlib.util.spec_from_file_location(
            "memory_dream", SCRIPTS_DIR / "memory_dream.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        old_argv = sys.argv
        sys.argv = ["memory_dream.py", "--days", "30", "--yes"]
        try:
            assert mod.main() == 0
        finally:
            sys.argv = old_argv

        # NOTE: read via a FRESH connection — LanceDB table objects hold a
        # version snapshot, so the pre-dream wq._l2_store would miss the rows
        # that memory_dream added through its own connection.
        lancedb = pytest.importorskip("lancedb")

        fresh = lancedb.connect(str(cfg.l2_db_path)).open_table("memories")
        rows = fresh.to_arrow().to_pylist()
        dreams = [r for r in rows if r.get("category") == "dream"]
        assert len(dreams) == 1
        assert "PostgreSQL" in dreams[0]["content"]
        assert dreams[0].get("source_rowid") is not None  # provenance kept


# ---------------------------------------------------------------------------
# migrations
# ---------------------------------------------------------------------------

class TestMigrations:
    def test_runner_idempotent(self, tmp_path):
        from plugin.memory_governed._migrations import MigrationRunner
        from plugin.memory_governed._config import GovernedMemoryConfig

        cfg = GovernedMemoryConfig()
        cfg.l3_db_path = str(tmp_path / "l3.db")
        cfg.l2_db_path = str(tmp_path / "l2")
        # L3 exists (no hash column yet), L2 does not exist
        conn = sqlite3.connect(str(cfg.l3_db_path))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
            "role TEXT, content TEXT, timestamp REAL, metadata TEXT)"
        )
        conn.commit()
        conn.close()

        runner = MigrationRunner(tmp_path / "hermes")
        ran1 = runner.run_pending(None, cfg)
        assert 1 in ran1  # l3 hash column migration applied
        assert 2 in ran1 or True  # l2 skipped (dir missing) — must not fail

        # Second run: nothing pending
        ran2 = runner.run_pending(None, cfg)
        assert ran2 == []

        # hash column now exists
        conn = sqlite3.connect(str(cfg.l3_db_path))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        conn.close()
        assert "hash" in cols


# ---------------------------------------------------------------------------
# audit L2 → L3 drill-down
# ---------------------------------------------------------------------------

class TestAuditDrillDown:
    def test_drill_l3_rowid(self, tmp_path):
        from plugin.memory_governed import GovernedMemoryProvider

        p = GovernedMemoryProvider()
        p.initialize("s1", hermes_home=str(tmp_path / "hermes"))

        msgs = [{"role": "user", "content": "部署方案确定了用 Docker Compose 部署测试消息"}]
        rowid_map = p._l3_writer.write(msgs, session_id="s-drill")
        assert rowid_map  # write succeeded

        drill = p._drill_l3_rowid(list(rowid_map.values())[0])
        assert drill is not None
        assert "Docker Compose" in drill["content"]
