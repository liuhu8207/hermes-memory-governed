"""Tests for the governed memory provider."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_hermes(tmp_path):
    """Create a temporary HERMES_HOME with L1-L4 structure."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    # L1 files
    memory_dir = hermes_home / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(
        "# Rules\n- Don't refactor old modules\n- Mobile still uses them", encoding="utf-8"
    )
    (memory_dir / "USER.md").write_text(
        "# User\n- Name: Test User\n- Prefers concise replies", encoding="utf-8"
    )

    # L2 directory
    (memory_dir / "l2").mkdir()

    # L3 directory + SQLite
    l3_dir = memory_dir / "l3"
    l3_dir.mkdir()
    db_path = l3_dir / "l3.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
    )
    now = time.time()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("test-session", "user", "What is Python?", now),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("test-session", "assistant", "Python is a programming language.", now),
    )
    conn.commit()
    # Insert into FTS5 (store timestamp as REAL for consistency)
    conn.execute(
        "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?, ?, ?, ?)",
        ("What is Python?", "user", "test-session", now),
    )
    conn.execute(
        "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?, ?, ?, ?)",
        ("Python is a programming language.", "assistant", "test-session", now),
    )
    conn.commit()
    conn.close()

    # L4 persona
    (memory_dir / "persona.md").write_text(
        "# Profile\n- Language: Chinese/English\n- Timezone: Asia/Shanghai", encoding="utf-8"
    )

    # Bridge directory
    bridge_dir = hermes_home / "cron" / "output" / "scope_recall_bridge"
    bridge_dir.mkdir(parents=True)

    # Config
    config = {
        "scripts_dir": str(hermes_home / "scripts"),
        "wiki_dir": str(tmp_path / "wiki"),
        "l1_memory_path": str(memory_dir / "MEMORY.md"),
        "l1_user_path": str(memory_dir / "USER.md"),
        "l2_db_path": str(memory_dir / "l2"),
        "l3_db_path": str(db_path),
        "l4_persona_path": str(memory_dir / "persona.md"),
        "l4_meta_path": str(memory_dir / "persona_meta.json"),
        "bridge_dir": str(bridge_dir),
    }
    (hermes_home / "governed_memory.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    return hermes_home


@pytest.fixture
def config(tmp_hermes):
    """Load config from temporary HERMES_HOME."""
    from plugin.memory_governed._config import load_governed_config
    return load_governed_config(str(tmp_hermes))


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_loads_from_json(self, config):
        assert config.l1_memory_path.endswith("MEMORY.md")
        assert config.l3_db_path.endswith("l3.db")

    def test_defaults(self, config):
        assert config.recall.l1_budget_tokens == 800
        assert config.recall.l23_budget_tokens == 1200
        assert config.sync.l3_write == "sync"
        assert config.sync.l2_write == "async"

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("GOVERNED_SCRIPTS_DIR", "/custom/scripts")
        from plugin.memory_governed._config import load_governed_config
        cfg = load_governed_config(str(tmp_path))
        assert cfg.scripts_dir == "/custom/scripts"

    def test_bad_json_warning(self, tmp_path):
        """Config loader warns on malformed JSON instead of silently ignoring."""
        import logging
        bad_file = tmp_path / "governed_memory.json"
        bad_file.write_text("{bad json!!!", encoding="utf-8")
        from plugin.memory_governed._config import load_governed_config
        with patch("plugin.memory_governed._config.logger") as mock_log:
            cfg = load_governed_config(str(tmp_path))
            mock_log.warning.assert_called_once()
        # Should still have valid defaults
        assert cfg.l1_memory_path.endswith("MEMORY.md")


# ---------------------------------------------------------------------------
# Recall engine tests
# ---------------------------------------------------------------------------

class TestRecallEngine:
    def test_l1_loads_from_files(self, config):
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        l1 = engine.get_l1()
        assert "Don't refactor old modules" in l1
        assert "Test User" in l1

    def test_l1_cached(self, config):
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        l1_first = engine.get_l1()
        l1_second = engine.get_l1()
        assert l1_first == l1_second
        assert engine._l1_cache is not None

    def test_l4_loads(self, config):
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        l4 = engine.get_l4()
        assert "Chinese/English" in l4

    def test_l3_search(self, config):
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        results = engine._search_l3("Python")
        assert len(results) > 0
        assert any("Python" in r.content for r in results)

    def test_l3_search_cjk(self, config):
        """CJK queries fall back to LIKE search and return results."""
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        # Insert CJK content
        import sqlite3
        conn = sqlite3.connect(config.l3_db_path, uri=False)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("cjk-test", "user", "Python 是最好的编程语言", time.time()),
        )
        conn.commit()
        conn.close()
        # CJK query should use LIKE fallback and still find results
        results = engine._search_l3("\u7f16\u7a0b\u8bed\u8a00")  # 编程语言
        assert len(results) > 0

    def test_l3_like_escape_wildcards(self, config):
        """LIKE wildcards in queries are properly escaped."""
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        # Insert content with special LIKE characters
        import sqlite3
        conn = sqlite3.connect(config.l3_db_path, uri=False)
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            ("escape-test", "user", "Pattern: 100% complete", time.time()),
        )
        conn.commit()
        conn.close()
        # Query containing % should be escaped, not treated as wildcard
        results = engine._search_l3("100%")
        # Should NOT return everything (which would happen if % wasn't escaped)
        for r in results:
            assert "100%" in r.content or "100" in r.content

    def test_format_recall_includes_l1(self, config):
        from plugin.memory_governed._recall import RecallEngine
        engine = RecallEngine(config)
        l1 = engine.get_l1()
        formatted = engine.format_recall([], l1, l1_budget=800, l23_budget=1200)
        assert "[User Rules]" in formatted
        assert "Don't refactor" in formatted

    def test_format_recall_adds_l23(self, config):
        from plugin.memory_governed._recall import RecallEngine, RecallResult
        engine = RecallEngine(config)
        l1 = engine.get_l1()
        results = [
            RecallResult(layer="l2", content="Python is a language", score=0.9),
            RecallResult(layer="l3", content="User asked about Python", score=0.7),
        ]
        formatted = engine.format_recall(results, l1, l1_budget=800, l23_budget=1200)
        assert "[User Rules]" in formatted
        assert "[Fact]" in formatted
        assert "[History]" in formatted

    def test_truncation(self):
        from plugin.memory_governed._recall import RecallEngine
        text = "x" * 5000
        result = RecallEngine._truncate_to_tokens(text, 100)
        assert len(result) < len(text)
        assert "truncated" in result

    def test_temp_pool_per_call(self):
        """Each parallel_recall call creates its own pool — no shared state."""
        from plugin.memory_governed._config import GovernedMemoryConfig
        from plugin.memory_governed._recall import RecallEngine
        cfg = GovernedMemoryConfig()
        engine = RecallEngine(cfg)
        # No shared pool attributes should exist
        assert not hasattr(engine, "_pool")
        assert not hasattr(engine, "_pool_lock")
        # Each call succeeds independently
        r1 = engine.parallel_recall("query-a")
        r2 = engine.parallel_recall("query-b")
        assert isinstance(r1, list)
        assert isinstance(r2, list)

    def test_shutdown_fallback(self):
        """After shutdown, parallel_recall falls back to sequential."""
        from plugin.memory_governed._config import GovernedMemoryConfig
        from plugin.memory_governed._recall import RecallEngine
        cfg = GovernedMemoryConfig()
        engine = RecallEngine(cfg)
        assert engine._shutdown is False
        engine.shutdown()
        assert engine._shutdown is True
        # Should not crash — returns sequential results
        results = engine.parallel_recall("test")
        assert isinstance(results, list)

    def test_concurrent_recalls_no_interference(self):
        """Three concurrent recalls each use their own pool — no queuing."""
        from plugin.memory_governed._config import GovernedMemoryConfig
        from plugin.memory_governed._recall import RecallEngine
        cfg = GovernedMemoryConfig()
        cfg.recall.parallel_timeout_seconds = 2.0
        engine = RecallEngine(cfg)

        results_store = []
        lock = threading.Lock()

        def worker(query, idx):
            t0 = time.time()
            r = engine.parallel_recall(query)
            elapsed = time.time() - t0
            with lock:
                results_store.append((idx, elapsed, len(r)))

        threads = [threading.Thread(target=worker, args=(f"q{i}", i)) for i in range(3)]
        t_start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total = time.time() - t_start

        # All 3 should complete
        assert len(results_store) == 3
        # Total time should be roughly 1x timeout, not 3x (parallel, not serial)
        assert total < 5.0, f"3 concurrent recalls took {total:.1f}s — expected parallel"

    def test_slow_task_respects_timeout(self):
        """A slow L2 search should not block the caller beyond timeout."""
        from plugin.memory_governed._config import GovernedMemoryConfig
        from plugin.memory_governed._recall import RecallEngine
        cfg = GovernedMemoryConfig()
        cfg.recall.parallel_timeout_seconds = 0.5
        engine = RecallEngine(cfg)
        # Inject a slow L2 (2s sleep, well beyond 0.5s timeout)
        original_l2 = engine._search_l2
        engine._search_l2 = lambda q: (time.sleep(2.0) or original_l2(q))

        t0 = time.time()
        results = engine.parallel_recall("test")
        elapsed = time.time() - t0
        assert elapsed < 1.5, f"Blocked for {elapsed:.1f}s — timeout={cfg.recall.parallel_timeout_seconds}s"
        # Should return partial results (L3+L4), not crash
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# Sync engine tests
# ---------------------------------------------------------------------------

class TestSyncEngine:
    def test_l3_write(self, config):
        from plugin.memory_governed._sync import L3Writer
        writer = L3Writer(config)
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        writer.write(messages, session_id="test")

        # Verify written
        conn = sqlite3.connect(config.l3_db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id='test'"
        ).fetchone()[0]
        conn.close()
        assert count == 2

    def test_write_queue_starts_stops(self, config):
        from plugin.memory_governed._sync import WriteQueue
        queue = WriteQueue(config)
        queue.start()
        assert queue._running is True
        queue.stop()

    def test_write_queue_stats_thread_safe(self):
        """WriteQueue stats property acquires lock (no crash under access)."""
        from plugin.memory_governed._config import GovernedMemoryConfig
        from plugin.memory_governed._sync import WriteQueue
        cfg = GovernedMemoryConfig()
        q = WriteQueue(cfg)
        stats = q.stats
        assert "processed" in stats
        assert "errors" in stats

    def test_write_queue_has_embed_model(self):
        """WriteQueue initializes _embed_model to None (was missing before fix)."""
        from plugin.memory_governed._config import GovernedMemoryConfig
        from plugin.memory_governed._sync import WriteQueue
        cfg = GovernedMemoryConfig()
        q = WriteQueue(cfg)
        assert q._embed_model is None
        assert hasattr(q, "_stats_lock")

    def test_fact_extraction(self):
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._config import GovernedMemoryConfig
        config = GovernedMemoryConfig()
        q = WriteQueue(config)
        facts = q._extract_atomic_facts([
            {"role": "user", "content": "The server is running on port 8080 and uses Python."},
            {"role": "assistant", "content": "Got it."},
        ])
        assert isinstance(facts, list)

    def test_categorize(self):
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._config import GovernedMemoryConfig
        config = GovernedMemoryConfig()
        q = WriteQueue(config)
        assert q._categorize("Python is installed") == "tech"
        assert q._categorize("Project deadline is Friday") == "work"
        assert q._categorize("I went for a run in the park") == "life"

    def test_migrate_null_vectors(self, tmp_path):
        """_migrate_null_vectors re-embeds rows with NULL vector column."""
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._config import GovernedMemoryConfig
        # This test verifies the method exists and handles gracefully
        # when lancedb is not installed (no crash)
        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "l2_test")
        q = WriteQueue(cfg)
        # Should not crash even without lancedb/embedding model
        q._migrate_null_vectors()
        # With l2_store=None, it should be a no-op
        assert q._l2_store is None


# ---------------------------------------------------------------------------
# Compress engine tests
# ---------------------------------------------------------------------------

class TestCompressEngine:
    def test_disabled_returns_empty(self):
        from plugin.memory_governed._compress import MermaidCompressor
        from plugin.memory_governed._config import GovernedMemoryConfig
        config = GovernedMemoryConfig()
        config.mermaid_compress.enabled = False
        compressor = MermaidCompressor(config)
        assert compressor.compress([{"role": "user", "content": "test"}]) == ""

    def test_builds_canvas(self):
        from plugin.memory_governed._compress import MermaidCompressor
        from plugin.memory_governed._config import GovernedMemoryConfig
        config = GovernedMemoryConfig()
        config.mermaid_compress.enabled = True
        compressor = MermaidCompressor(config)
        messages = [
            {"role": "user", "content": "Fix the authentication bug"},
            {"role": "assistant", "content": "I decided to use JWT tokens"},
            {"role": "tool", "name": "terminal", "content": "Server started on port 8080"},
        ]
        canvas = compressor.compress(messages)
        assert "graph TD" in canvas


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------

class TestBridge:
    def test_export_candidates(self, config):
        from plugin.memory_governed._bridge import BridgeExporter
        exporter = BridgeExporter(config)
        candidates = [
            {"content": "User prefers concise replies", "target": "user"},
            {"content": "Don't refactor old modules", "target": "memory"},
        ]
        report = exporter.export_candidates(candidates)
        assert report["written"] == 2
        assert report["skipped"] == 0

    def test_rejects_unsafe_target(self, config):
        from plugin.memory_governed._bridge import BridgeExporter
        exporter = BridgeExporter(config)
        candidates = [
            {"content": "Some fact", "target": "general"},
        ]
        report = exporter.export_candidates(candidates)
        assert report["written"] == 0
        assert report["skipped"] == 1

    def test_rejects_secrets(self, config):
        from plugin.memory_governed._bridge import BridgeExporter
        exporter = BridgeExporter(config)
        candidates = [
            {"content": "The API key: sk-abc123def456ghi789jkl0 must be kept secret", "target": "memory"},
        ]
        report = exporter.export_candidates(candidates)
        assert report["written"] == 0

    def test_status(self, config):
        from plugin.memory_governed._bridge import BridgeExporter
        exporter = BridgeExporter(config)
        status = exporter.get_status()
        assert "bridge_dir" in status
        assert "candidate_count" in status

    def test_redact_sensitive(self):
        from plugin.memory_governed._bridge import redact_sensitive
        text = "API key is sk-abc123def456ghi789jkl0"
        redacted = redact_sensitive(text)
        assert "sk-abc123" not in redacted
        assert "[REDACTED]" in redacted

    def test_append_dedup(self, tmp_path):
        """export_candidates appends and deduplicates by id."""
        from plugin.memory_governed._bridge import BridgeExporter
        from plugin.memory_governed._config import GovernedMemoryConfig
        cfg = GovernedMemoryConfig()
        cfg.bridge_dir = str(tmp_path)
        be = BridgeExporter(cfg)

        cand = {"content": "User prefers dark mode", "target": "user"}
        r1 = be.export_candidates([cand])
        assert r1["written"] == 1

        # Same content again — should dedup
        r2 = be.export_candidates([cand])
        assert r2["written"] == 0

        # Different content — should append
        cand2 = {"content": "Always use Python 3.12 plus", "target": "memory"}
        r3 = be.export_candidates([cand2])
        assert r3["written"] == 1

        # File should have exactly 2 lines
        lines = [l for l in (tmp_path / "candidates.jsonl").read_text().splitlines() if l.strip()]
        assert len(lines) == 2

    def test_archive_when_exceeds_limit(self, tmp_path):
        """candidates.jsonl is archived to archive/ when line count exceeds limit."""
        from plugin.memory_governed._bridge import BridgeExporter
        from plugin.memory_governed._config import GovernedMemoryConfig
        cfg = GovernedMemoryConfig()
        cfg.bridge_dir = str(tmp_path)
        be = BridgeExporter(cfg)
        # Temporarily lower the limit for testing
        original_limit = be.MAX_JSONL_LINES
        be.MAX_JSONL_LINES = 3
        try:
            # Write 4 candidates to exceed the limit (archive triggers after write #4)
            for i in range(4):
                be.export_candidates([{
                    "content": f"Candidate number {i} with enough text",
                    "target": "memory",
                }])
            # Archive should have been created
            archive_dir = tmp_path / "archive"
            assert archive_dir.exists(), "archive/ directory not created"
            archives = list(archive_dir.glob("candidates_*.jsonl"))
            assert len(archives) == 1, f"Expected 1 archive, got {len(archives)}"
            # After archive, a subsequent write creates a fresh active file
            be.export_candidates([{"content": "After archive candidate text", "target": "memory"}])
            assert (tmp_path / "candidates.jsonl").exists()
            lines = [l for l in (tmp_path / "candidates.jsonl").read_text().splitlines() if l.strip()]
            assert len(lines) == 1, f"Expected 1 line in fresh file, got {len(lines)}"
        finally:
            be.MAX_JSONL_LINES = original_limit


# ---------------------------------------------------------------------------
# Provider tests
# ---------------------------------------------------------------------------

class TestProvider:
    def test_name(self):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        assert provider.name == "governed"

    def test_tool_schemas(self):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert "governed_search" in names
        assert "governed_audit" in names
        assert "governed_health" in names

    def test_initialize(self, tmp_hermes):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        provider.initialize("test-session", hermes_home=str(tmp_hermes))
        assert provider._recall is not None
        assert provider._l3_writer is not None
        assert provider._write_queue is not None

    def test_system_prompt_block(self, tmp_hermes):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        provider.initialize("test-session", hermes_home=str(tmp_hermes))
        block = provider.system_prompt_block()
        assert "User Profile" in block
        assert "Chinese/English" in block

    def test_sync_turn(self, tmp_hermes):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        provider.initialize("test-session", hermes_home=str(tmp_hermes))
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        provider.sync_turn("Hello", "Hi!", session_id="test", messages=messages)

    def test_on_memory_write(self, tmp_hermes):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        provider.initialize("test-session", hermes_home=str(tmp_hermes))
        provider.on_memory_write("add", "memory", "New rule: always test first")

        memory_md = tmp_hermes / "memory" / "MEMORY.md"
        content = memory_md.read_text(encoding="utf-8")
        assert "always test first" in content

    def test_shutdown(self, tmp_hermes):
        from plugin.memory_governed import GovernedMemoryProvider
        provider = GovernedMemoryProvider()
        provider.initialize("test-session", hermes_home=str(tmp_hermes))
        provider.shutdown()

    def test_all_export(self):
        """Public API is explicitly exported via __all__."""
        from plugin.memory_governed import __all__
        assert "GovernedMemoryProvider" in __all__
        assert "register" in __all__
        assert len(__all__) == 2

    def test_durable_fact_heuristic(self):
        """_looks_like_durable_fact filters non-durable content."""
        from plugin.memory_governed import GovernedMemoryProvider
        fn = GovernedMemoryProvider._looks_like_durable_fact

        # Should match
        assert fn("I prefer dark mode")
        assert fn("always use tabs")
        assert fn("never refactor old modules")
        assert fn("\u4ee5\u540e\u90fd\u7528\u4e2d\u6587")  # 以后都用中文
        assert fn("\u4e0d\u8981\u518d\u5220\u9664\u6587\u4ef6")  # 不要再删除文件
        assert fn("\u504f\u597d\u6df1\u8272\u6a21\u5f0f")  # 偏好深色模式

        # Should NOT match
        assert not fn("short")
        assert not fn("what time is it?")
        assert not fn("\u4eca\u5929\u5929\u6c14\u600e\u4e48\u6837\uff1f")  # 今天天气怎么样？
        assert not fn("buy milk, eggs, bread, butter, cheese, juice")


# ---------------------------------------------------------------------------
# Build tests
# ---------------------------------------------------------------------------

class TestBuild:
    def test_pyproject_toml_exists(self):
        """pyproject.toml is present for pip install."""
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml missing"

    def test_package_importable(self):
        """The plugin package is importable as a Python package."""
        import plugin.memory_governed as pkg
        assert hasattr(pkg, "GovernedMemoryProvider")
        assert hasattr(pkg, "register")

    def test_no_missing_dependencies(self):
        """Core imports don't crash on missing optional deps (lancedb, sentence_transformers)."""
        from plugin.memory_governed._recall import RecallEngine
        from plugin.memory_governed._sync import WriteQueue
        from plugin.memory_governed._bridge import BridgeExporter
        from plugin.memory_governed._compress import MermaidCompressor
        from plugin.memory_governed._config import GovernedMemoryConfig
        # All should import without requiring lancedb/sentence_transformers
        cfg = GovernedMemoryConfig()
        assert RecallEngine(cfg) is not None
        assert WriteQueue(cfg) is not None
