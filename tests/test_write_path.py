# -*- coding: utf-8 -*-
"""Write-path tests (L3 storage, WriteQueue reliability, fact extraction).

Scope: plugin/memory_governed/_sync.py, _migrations.py, _bridge.py, _diag.py.

All tests MUST pass with lancedb / sentence-transformers / fastembed / pyarrow
absent — that is the default deployment shape, so anything that needs them is
skipped or stubbed, never erroring.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugin.memory_governed import _diag
from plugin.memory_governed._bridge import BridgeExporter
from plugin.memory_governed._config import GovernedMemoryConfig, load_governed_config
from plugin.memory_governed._migrations import MigrationRunner
from plugin.memory_governed._sync import (
    L3Writer,
    WriteQueue,
    _resolve_source_rowid,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config(tmp_path):
    cfg = GovernedMemoryConfig()
    cfg.l3_db_path = str(tmp_path / "l3" / "l3.db")
    cfg.l2_db_path = str(tmp_path / "l2")
    cfg.bridge_dir = str(tmp_path / "bridge")
    cfg.l1_memory_path = str(tmp_path / "MEMORY.md")
    cfg.l1_user_path = str(tmp_path / "USER.md")
    (tmp_path / "l3").mkdir(parents=True, exist_ok=True)
    return cfg


class _FakeField:
    """Minimal stand-in for a pyarrow/lancedb schema field."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeL2Store:
    """In-memory stand-in for a LanceDB table (no lancedb in this env)."""

    def __init__(self, names=("content", "category", "source", "timestamp",
                              "vector", "source_rowid")):
        self.schema = [_FakeField(n) for n in names]
        self.rows: list = []

    def add(self, rows) -> None:
        self.rows.extend(rows)


@pytest.fixture(autouse=True)
def _reset_diag():
    """Keep the process-global diagnostics state predictable per test."""
    _diag.reset()
    yield
    _diag.reset()


# ---------------------------------------------------------------------------
# P0-1 — L3 indexes (performance cliff)
# ---------------------------------------------------------------------------

class TestL3Indexes:
    def test_new_database_gets_both_indexes(self, config):
        writer = L3Writer(config)
        writer.write([{"role": "user", "content": "hello world"}], "s1")

        conn = sqlite3.connect(config.l3_db_path)
        names = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        writer.shutdown()

        assert "idx_messages_dedup" in names
        assert "idx_messages_ts" in names

    def test_dedup_lookup_uses_the_index(self, config):
        """EXPLAIN QUERY PLAN must show SEARCH ... USING INDEX, not SCAN."""
        writer = L3Writer(config)
        conn = writer._get_conn()
        for i in range(200):
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, hash) "
                "VALUES (?,?,?,?,?)",
                (f"s{i}", "user", f"message body {i}", time.time(), f"h{i}"),
            )
        conn.commit()
        conn.execute("ANALYZE")

        plan = "\n".join(
            row[-1] for row in conn.execute(
                "EXPLAIN QUERY PLAN SELECT 1 FROM messages "
                "WHERE session_id = ? AND hash = ?", ("s1", "h1")
            )
        )
        writer.shutdown()

        assert "idx_messages_dedup" in plan, f"dedup lookup still scans: {plan}"
        assert "SCAN messages" not in plan, f"full table scan remains: {plan}"

    def test_migration_003_is_idempotent(self, tmp_path):
        """Migration 003 creates the indexes on a legacy DB and is re-runnable."""
        db = tmp_path / "l3.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL, hash TEXT)"
        )
        conn.commit()
        conn.close()

        cfg = GovernedMemoryConfig()
        cfg.l3_db_path = str(db)
        cfg.l2_db_path = str(tmp_path / "l2")

        runner = MigrationRunner(tmp_path / "hermes")
        ran1 = runner.run_pending(None, cfg)
        assert 3 in ran1

        conn = sqlite3.connect(str(db))
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        conn.close()
        assert {"idx_messages_dedup", "idx_messages_ts"} <= names

        # Second run: nothing pending, no error on re-creating the indexes.
        assert runner.run_pending(None, cfg) == []

    def test_migration_003_defers_without_hash_column(self, tmp_path):
        """A DB without messages.hash must defer (not fail, not half-index)."""
        db = tmp_path / "l3.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL)"
        )
        conn.commit()
        conn.close()

        cfg = GovernedMemoryConfig()
        cfg.l3_db_path = str(db)
        runner = MigrationRunner(tmp_path / "hermes")
        # Pretend 001 already ran, so 003 sees a table without `hash`.
        runner.mark_applied(1)
        ran = runner.run_pending(None, cfg)

        assert 3 in ran  # recorded as applied (deferred is not a failure)
        conn = sqlite3.connect(str(db))
        names = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'").fetchall()}
        conn.close()
        assert "idx_messages_dedup" not in names  # nothing to index yet


# ---------------------------------------------------------------------------
# P0-2 — L3Writer thread safety
# ---------------------------------------------------------------------------

class TestL3ThreadSafety:
    def test_concurrent_writes_from_many_threads(self, config):
        """8 threads x 10 messages must all land, with no exceptions."""
        writer = L3Writer(config)
        errors: list = []

        def worker(base: int) -> None:
            try:
                for i in range(10):
                    writer.write(
                        [{"role": "user", "content": f"t{base}-{i} hello world message"}],
                        f"s{base}",
                    )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(b,)) for b in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn = sqlite3.connect(config.l3_db_path)
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
        writer.shutdown()

        assert not errors, f"concurrent writes raised: {errors}"
        assert count == 80, f"only {count}/80 messages survived"

    def test_writer_survives_being_created_on_another_thread(self, config):
        """A connection created on thread A must be usable from thread B."""
        writer = L3Writer(config)
        writer.write([{"role": "user", "content": "created on main thread"}], "s1")

        seen: list = []

        def other_thread() -> None:
            seen.append(
                writer.write([{"role": "user", "content": "written from thread 2"}], "s1")
            )

        t = threading.Thread(target=other_thread)
        t.start()
        t.join()
        writer.shutdown()

        assert seen and list(seen[0].values()), "cross-thread write produced no rowid"

    def test_connection_options(self, config):
        """check_same_thread=False + busy timeout are mandatory for shared use."""
        writer = L3Writer(config)
        conn = writer._get_conn()
        busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        writer.shutdown()

        assert busy_timeout >= 1000, f"busy_timeout too low: {busy_timeout}"
        assert journal.lower() == "wal"


# ---------------------------------------------------------------------------
# P0-3 — FTS divergence must be visible
# ---------------------------------------------------------------------------

class TestFtsObservability:
    def test_legacy_three_column_fts_is_still_filled(self, config, caplog):
        """An FTS table without `timestamp` must not silently diverge."""
        Path(config.l3_db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(config.l3_db_path)
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, role TEXT, content TEXT, timestamp REAL, hash TEXT)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id)"
        )
        conn.commit()
        conn.close()

        writer = L3Writer(config)
        with caplog.at_level(logging.WARNING):
            writer.write([{"role": "user", "content": "hello indexed world"}], "s1")

        conn = sqlite3.connect(config.l3_db_path)
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        conn.close()
        writer.shutdown()

        assert n_msg == n_fts == 1, f"divergence: messages={n_msg} fts={n_fts}"
        assert any("FTS schema mismatch" in r.message for r in caplog.records), (
            "the degraded FTS schema was not reported"
        )

    def test_fts_insert_failure_is_reported_as_data_loss(self, config, caplog, monkeypatch):
        """A failing FTS insert must raise an ERROR-level, unthrottled record."""
        writer = L3Writer(config)
        writer._get_conn()

        monkeypatch.setattr(
            writer, "_fts_insert_columns", lambda conn: ["column_that_does_not_exist"]
        )

        with caplog.at_level(logging.ERROR):
            writer.write([{"role": "user", "content": "row that cannot be indexed"}], "s1")

        conn = sqlite3.connect(config.l3_db_path)
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        conn.close()
        writer.shutdown()

        assert n_msg == 1, "the row must still be archived in messages"
        assert n_fts == 0, "test setup: the FTS insert is expected to fail"
        assert any(r.levelno >= logging.ERROR for r in caplog.records), (
            "FTS insert failure was swallowed"
        )
        assert _diag.stats()["data_loss"].get("l3_fts::fts_insert_failed") == 1

    def test_repeated_failures_are_never_throttled(self, config, caplog, monkeypatch):
        """Unlike degraded events, data-loss records must never be suppressed."""
        writer = L3Writer(config)
        writer._get_conn()
        monkeypatch.setattr(
            writer, "_fts_insert_columns", lambda conn: ["column_that_does_not_exist"]
        )

        with caplog.at_level(logging.ERROR):
            for i in range(3):
                writer.write([{"role": "user", "content": f"unindexable row {i}"}], "s1")
        writer.shutdown()

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 3, f"expected 3 ERROR records, got {len(errors)}"


# ---------------------------------------------------------------------------
# P1-1 — L2 -> L3 audit drill-down (real chain, no provider mocks)
# ---------------------------------------------------------------------------

class TestAuditDrillDown:
    def test_facts_get_source_rowid_end_to_end(self, config):
        """L3Writer.write() -> _index_l2() must attach a resolvable rowid."""
        messages = [
            {"role": "user",
             "content": "I always prefer dark mode. The project deadline is next Friday."},
            {"role": "assistant", "content": "Noted, I will remember the deadline."},
        ]
        writer = L3Writer(config)
        rowid_map = writer.write(messages, "s-drill")

        store = _FakeL2Store()
        queue = WriteQueue(config)
        queue._l2_store = store
        queue._index_l2(messages, rowid_map)
        writer.shutdown()

        assert store.rows, "no facts indexed into L2"
        resolved = [r for r in store.rows if r.get("source_rowid") is not None]
        assert len(resolved) == len(store.rows), (
            f"only {len(resolved)}/{len(store.rows)} facts carry a source_rowid: "
            f"{[r['content'] for r in store.rows]}"
        )
        conn = sqlite3.connect(config.l3_db_path)
        for row in store.rows:
            hit = conn.execute(
                "SELECT content FROM messages WHERE id = ?", (row["source_rowid"],)
            ).fetchone()
            assert hit and row["content"] in hit[0], (
                f"rowid {row['source_rowid']} does not contain {row['content']!r}"
            )
        conn.close()

    def test_resolve_source_rowid_substring_match(self):
        rowid_map = {"Full message. Second sentence.": 7}
        assert _resolve_source_rowid("Second sentence", rowid_map) == 7
        assert _resolve_source_rowid("nothing like it", rowid_map) is None

    def test_resolve_source_rowid_truncated_message_key(self):
        message = "x" * 600
        rowid_map = {message[:500]: 42}
        assert _resolve_source_rowid("x" * 260, rowid_map) == 42

    def test_resolve_source_rowid_empty_inputs(self):
        assert _resolve_source_rowid("", {"a": 1}) is None
        assert _resolve_source_rowid("a", {}) is None
        assert _resolve_source_rowid("a", {"": 1}) is None

    def test_instance_resolver_accepts_messages_fallback(self, config):
        """The WriteQueue method also accepts the original messages list."""
        messages = [{"role": "user", "content": "We decided to use Postgres for the store."}]
        writer = L3Writer(config)
        rowid_map = writer.write(messages, "s-fallback")
        writer.shutdown()

        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts(messages)
        assert facts
        assert queue._resolve_source_rowid(facts[0]["content"], rowid_map, messages) is not None


# ---------------------------------------------------------------------------
# P0 regression — multimodal (list) message content must not break the write path
# ---------------------------------------------------------------------------

class TestMultimodalContent:
    """hermes transcripts carry OpenAI-style content parts (a ``list``).

    Regression: ``session_id + "|" + role + "|" + content`` raised
    ``TypeError: can only concatenate str (not "list") to str`` inside
    ``L3Writer.write`` (rolling back the WHOLE batch — repeated real data loss),
    and ``_split_sentences`` raised
    ``expected string or bytes-like object, got 'list'`` inside
    ``_extract_atomic_facts``. Both paths now normalise through
    ``_synthesize._content_to_text`` before hashing / splitting.
    """

    @staticmethod
    def _text(*texts):
        return [{"type": "text", "text": t} for t in texts]

    def test_l3_write_accepts_list_content(self, config):
        writer = L3Writer(config)
        rowid_map = writer.write(
            [{"role": "user", "content": self._text("我们决定用 PostgreSQL 作为主数据库")}],
            "s-multimodal",
        )
        conn = sqlite3.connect(config.l3_db_path)
        rows = conn.execute("SELECT content FROM messages").fetchall()
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        conn.close()
        writer.shutdown()

        assert len(rows) == 1, "list-content message was not archived"
        assert "PostgreSQL" in rows[0][0]
        assert n_fts == 1, "list-content message was not mirrored into FTS"
        assert rowid_map, "rowid_map empty for list content"

    def test_l3_write_mixed_str_and_list(self, config):
        writer = L3Writer(config)
        writer.write([
            {"role": "user", "content": "I prefer dark mode everywhere"},
            {"role": "assistant", "content": self._text("好的", "我会记录这个决定")},
        ], "s-mixed")
        conn = sqlite3.connect(config.l3_db_path)
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        conn.close()
        writer.shutdown()

        assert n_msg == n_fts == 2, f"mixed batch lost rows: msg={n_msg} fts={n_fts}"

    def test_image_only_message_is_skipped(self, config):
        """A list with no text parts normalises to '' and keeps skip semantics."""
        writer = L3Writer(config)
        writer.write(
            [{"role": "user",
              "content": [{"type": "image_url",
                           "image_url": {"url": "https://example.com/x.png"}}]}],
            "s-img",
        )
        conn = sqlite3.connect(config.l3_db_path)
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()
        writer.shutdown()

        assert n_msg == 0, "an image-only part has no text to archive"

    def test_extract_atomic_facts_handles_list_content(self, config):
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts(
            [{"role": "user", "content": self._text("我们决定用 PostgreSQL 作为主数据库")}]
        )
        assert facts, "list content produced no facts"
        assert all(isinstance(f["content"], str) for f in facts)

    def test_index_l2_end_to_end_with_list_content(self, config):
        writer = L3Writer(config)
        queue = WriteQueue(config)
        messages = [{"role": "user",
                     "content": self._text("部署方案确定用 Docker Compose")}]
        rowid_map = writer.write(messages, "s-list-l2")
        store = _FakeL2Store()
        queue._l2_store = store
        queue._index_l2(messages, rowid_map)
        writer.shutdown()

        assert store.rows, "no facts indexed from list content"

    def test_resolve_source_rowid_handles_list_message(self, config):
        """The messages fallback must normalise, not compare against a raw list."""
        queue = WriteQueue(config)
        messages = [{"role": "user",
                     "content": self._text("We decided to use Postgres for the store.")}]
        rowid_map = {"We decided to use Postgres for the store.": 5}
        assert queue._resolve_source_rowid(
            "use Postgres for the store", rowid_map, messages
        ) == 5


# ---------------------------------------------------------------------------
# P0 regression — L2 write de-duplication (long sessions re-feed their history)
# ---------------------------------------------------------------------------

class TestL2Dedup:
    FACT = "We decided to use PostgreSQL for the store"

    def test_same_content_is_not_rewritten(self, config):
        queue = WriteQueue(config)
        store = _FakeL2Store()
        queue._l2_store = store
        messages = [{"role": "user", "content": self.FACT}]

        queue._index_l2(messages, {})
        first = len(store.rows)
        assert first >= 1
        queue._index_l2(messages, {})  # the long session re-feeds the same turn

        assert len(store.rows) == first, "duplicate facts were re-appended to L2"

    def test_batch_internal_duplicates_are_dropped(self, config):
        queue = WriteQueue(config)
        store = _FakeL2Store()
        queue._l2_store = store

        queue._index_l2([
            {"role": "user", "content": self.FACT},
            {"role": "user", "content": self.FACT},
        ], {})

        contents = [r["content"] for r in store.rows]
        assert contents.count(self.FACT) == 1, f"batch duplicates kept: {contents}"

    def test_dedup_runs_before_embedding(self, config):
        """Dropped facts must not spend an embedding call."""
        queue = WriteQueue(config)
        store = _FakeL2Store()
        queue._l2_store = store
        calls: list = []
        queue._embed_fn = lambda text: (calls.append(text), [0.1, 0.2])[1]
        messages = [{"role": "user", "content": self.FACT}]

        queue._index_l2(messages, {})
        assert calls == [self.FACT]
        queue._index_l2(messages, {})

        assert calls == [self.FACT], "duplicate fact was embedded again"
        assert _diag.metrics().get("facts_deduped") == 1


class TestL2DedupArrowPriority:
    """Regression guard: the Arrow path must be tried BEFORE any pandas path.

    The hermes deploy runtime (``hermes-agent/venv``) ships lancedb + pyarrow +
    numpy but NO pandas and NO pylance. The previous ordering called
    ``to_pandas()`` first, so ``_existing_contents()`` raised
    ModuleNotFoundError, fell through every strategy, returned an empty set,
    and store-level L2 dedup silently no-op'd in production. The project dev
    venv happens to have pandas, so the tests stayed green and the bug escaped.
    See ``_sync._existing_contents`` STRATEGY 1.
    """

    class _Col:
        def __init__(self, data):
            self._data = list(data)

        def to_pylist(self):
            return list(self._data)

    class _Table:
        def __init__(self, data):
            self._data = list(data)
            self.num_rows = len(self._data)
            self.column_names = ["content"]

        def column(self, name):
            assert name == "content", name
            return TestL2DedupArrowPriority._Col(self._data)

    class _Series:
        def __init__(self, data):
            self._data = list(data)

        def tolist(self):
            return list(self._data)

    class _FakeDF:
        """Stand-in for a pandas DataFrame: only ``df["content"].tolist()``."""

        def __init__(self, data):
            self._data = list(data)

        def __getitem__(self, key):
            assert key == "content", key
            return TestL2DedupArrowPriority._Series(self._data)

    class _Query:
        def __init__(self, store):
            self._store = store

        def select(self, cols):
            assert cols == ["content"], cols
            return self

        def limit(self, n):
            self._store.limit_calls.append(n)
            return self

        def to_arrow(self):
            self._store.arrow_calls += 1
            return TestL2DedupArrowPriority._Table(self._store.arrow_rows)

        def to_pandas(self):
            # Reached only if the ordering regressed (pandas tried first).
            self._store.pandas_calls += 1
            return TestL2DedupArrowPriority._FakeDF(self._store.pandas_rows)

    class _Store:
        """Arrow-only table double: no ``to_pandas`` / ``to_lance`` / ``rows``.

        This mirrors the deploy runtime, where the only importable projection
        path is Arrow.
        """

        def __init__(self, arrow_rows, pandas_rows=()):
            self.arrow_rows = list(arrow_rows)
            self.pandas_rows = list(pandas_rows)
            self.arrow_calls = 0
            self.pandas_calls = 0
            self.limit_calls: list = []

        def count_rows(self):
            return len(self.arrow_rows)

        def search(self):
            return TestL2DedupArrowPriority._Query(self)

        def add(self, rows):  # pragma: no cover - reads must not add
            raise AssertionError("add() must not run while reading existing content")

    def test_arrow_only_store_still_dedups(self, config):
        """A runtime without pandas must still dedup — never return an empty set."""
        queue = WriteQueue(config)
        store = self._Store(["alpha fact", "beta fact", None])
        queue._l2_store = store

        assert queue._existing_contents() == {"alpha fact", "beta fact"}
        assert store.arrow_calls == 1, "the Arrow projection scan was not used"
        assert store.limit_calls == [3], store.limit_calls

    def test_arrow_result_wins_over_pandas(self, config):
        """Arrow is tried first: a differing pandas result must never be returned."""
        queue = WriteQueue(config)
        store = self._Store(arrow_rows=["from-arrow"],
                            pandas_rows=["from-pandas"])
        queue._l2_store = store

        assert queue._existing_contents() == {"from-arrow"}
        assert store.arrow_calls == 1
        assert store.pandas_calls == 0, "pandas path ran before the Arrow path"

    def test_zero_rows_short_circuits_before_scan(self, config):
        queue = WriteQueue(config)
        store = self._Store([])
        queue._l2_store = store

        assert queue._existing_contents() == set()
        assert store.arrow_calls == 0, "count_rows()==0 must short-circuit the scan"


# ---------------------------------------------------------------------------
# P1-2 — WriteQueue stop semantics
# ---------------------------------------------------------------------------

class TestWriteQueueStop:
    def test_stop_drains_every_queued_item(self, config, monkeypatch):
        processed: list = []
        queue = WriteQueue(config)
        monkeypatch.setattr(queue, "_process_item", processed.append)

        for i in range(5):
            queue.enqueue([{"role": "user", "content": f"message number {i}"}], "s1")
        queue.stop()

        assert len(processed) == 5, f"stop() dropped items: {len(processed)}/5"

    def test_stop_survives_a_raising_item(self, config, monkeypatch):
        calls: list = []

        def flaky(item):
            calls.append(item)
            if len(calls) == 1:
                raise RuntimeError("boom")

        queue = WriteQueue(config)
        monkeypatch.setattr(queue, "_process_item", flaky)
        queue._queue.put_nowait({"messages": [{"a": 1}]})
        queue._queue.put_nowait({"messages": [{"b": 2}]})

        queue.stop()  # must not propagate

        assert len(calls) == 2, "the first error discarded the rest of the queue"
        assert queue.stats["errors"] == 1

    def test_stop_joins_the_worker(self, config, monkeypatch):
        done = threading.Event()

        def quick(item):
            done.set()

        queue = WriteQueue(config)
        monkeypatch.setattr(queue, "_process_item", quick)
        queue.start()
        queue.enqueue([{"role": "user", "content": "a message that is long enough"}], "s1")
        assert done.wait(5), "worker never processed the item"

        queue.stop(timeout=5.0)
        assert not queue._worker_thread.is_alive(), "stop() returned with a live worker"

    def test_stop_is_idempotent(self, config):
        queue = WriteQueue(config)
        queue.stop()
        queue.stop()

    def test_drain_is_bounded(self, config, monkeypatch):
        """A saturated queue must not block shutdown forever."""
        queue = WriteQueue(config)
        config.sync.write_queue_maxsize = 3
        processed: list = []
        monkeypatch.setattr(queue, "_process_item", processed.append)
        for i in range(50):
            queue._queue.put_nowait({"messages": [{"n": i}]})

        queue.stop(timeout=1.0)
        assert len(processed) <= 50

    def test_worker_records_errors_without_dying(self, config, monkeypatch):
        def boom(item):
            raise RuntimeError("always fails")

        queue = WriteQueue(config)
        monkeypatch.setattr(queue, "_process_item", boom)
        queue.start()
        for i in range(3):
            queue.enqueue([{"role": "user", "content": f"failing message {i}"}], "s1")
        queue.stop(timeout=5.0)

        assert queue.stats["errors"] == 3
        assert not queue._worker_thread.is_alive()


# ---------------------------------------------------------------------------
# P1-3 / P1-4 — write-path config
# ---------------------------------------------------------------------------

class TestWritePathConfig:
    def test_embed_fn_defined_in_constructor(self, config):
        queue = WriteQueue(config)
        assert hasattr(queue, "_embed_fn")
        assert queue._embed_fn is None
        assert queue._embed_model is None

    def test_extraction_is_capped_by_the_write_knob(self, config):
        queue = WriteQueue(config)
        messages = [
            {"role": "user",
             "content": f"The project deadline for milestone {i} is next Friday."}
            for i in range(40)
        ]
        # A read-path cap of 15 must not truncate the write path.
        config.recall.l2_max_results = 15
        assert len(queue._extract_atomic_facts(messages)) == 40

    def test_write_knob_is_independent_and_configurable(self, config):
        config.sync.l2_max_facts_per_turn = 7
        queue = WriteQueue(config)
        messages = [
            {"role": "user",
             "content": f"The project deadline for milestone {i} is next Friday."}
            for i in range(40)
        ]
        assert len(queue._extract_atomic_facts(messages)) == 7

    def test_config_file_accepts_the_new_field(self, tmp_path):
        cfg_file = tmp_path / "governed_memory.json"
        cfg_file.write_text(json.dumps({"sync": {"l2_max_facts_per_turn": "25"}}),
                            encoding="utf-8")
        cfg = load_governed_config(tmp_path)
        assert cfg.sync.l2_max_facts_per_turn == 25


# ---------------------------------------------------------------------------
# P1-5 — sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_decimal_numbers_survive(self):
        sents = _split_sentences("The p99 latency is 12.5 ms in production.")
        assert sents == ["The p99 latency is 12.5 ms in production"]

    def test_version_numbers_survive(self):
        sents = _split_sentences("We are shipping version 3.14 today.")
        assert sents == ["We are shipping version 3.14 today"]

    def test_urls_survive(self):
        text = "We are shipping version 3.14 today. See https://example.com/docs for info."
        sents = _split_sentences(text)
        assert sents == [
            "We are shipping version 3.14 today",
            "See https://example.com/docs for info",
        ]

    def test_ip_and_semver_survive(self):
        sents = _split_sentences("The host is 10.0.0.1 on port 8080.")
        assert sents == ["The host is 10.0.0.1 on port 8080"]

    def test_real_sentence_boundaries_are_kept(self):
        sents = _split_sentences("I always prefer dark mode. The deadline is next Friday.")
        assert sents == ["I always prefer dark mode", "The deadline is next Friday"]

    def test_cjk_terminators_split(self):
        sents = _split_sentences("我们决定用 Postgres。部署时间定在下周五。")
        assert sents == ["我们决定用 Postgres", "部署时间定在下周五"]

    def test_newlines_split(self):
        assert _split_sentences("line one\nline two\n\nline three") == [
            "line one", "line two", "line three",
        ]

    def test_question_mark_is_preserved(self):
        """'?' must survive so that questions can be rejected as facts."""
        assert _split_sentences("What is the deadline?") == ["What is the deadline?"]

    def test_empty_input(self):
        assert _split_sentences("") == []


# ---------------------------------------------------------------------------
# P1-6 — fact judgement (CN + EN)
# ---------------------------------------------------------------------------

class TestLooksLikeFact:
    @pytest.mark.parametrize("text", [
        "我们决定用 PostgreSQL 作为主数据库",
        "部署方案确定用 Docker Compose",
        "注意：生产环境禁止直接改库",
        "以后都用 pnpm 而不是 npm",
        "结论是先做单体架构，以后再拆服务",
        "记住我的偏好是深色主题",
        "这个项目的规则是所有接口必须带版本号",
    ])
    def test_chinese_facts_are_recognised(self, text):
        assert WriteQueue._looks_like_fact(text), f"漏判中文事实: {text}"

    @pytest.mark.parametrize("text", [
        "I always prefer dark mode in every editor",
        "We decided to use Postgres instead of MySQL",
        "The project deadline is next Friday",
        "Remember that the staging server runs on port 8080",
    ])
    def test_english_facts_are_recognised(self, text):
        assert WriteQueue._looks_like_fact(text), f"missed English fact: {text}"

    @pytest.mark.parametrize("text", [
        "你好",
        "谢谢",
        "嗯",
        "ok",
        "thanks",
        "帮我看一下这个报错",
        "请帮我重构这个函数",
        "What is the deployment deadline?",
        "这个方案可以吗？",
        "Traceback (most recent call last):",
        "SELECT * FROM messages WHERE id = 1",
        "def handle_tool_call(self, args):",
        "import sqlite3",
        "https://example.com/docs",
        "TODO",
    ])
    def test_noise_is_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), f"误判为事实: {text}"

    def test_chinese_sentence_shorter_than_six_ascii_chars_is_kept(self):
        """CJK is denser — '我们用 Postgres' must not be dropped for length."""
        assert WriteQueue._looks_like_fact("我们用 Postgres")

    def test_very_long_text_is_rejected(self):
        assert not WriteQueue._looks_like_fact("x" * 501)

    def test_extraction_metrics_are_recorded(self, config):
        _diag.reset()
        queue = WriteQueue(config)
        queue._extract_atomic_facts([
            {"role": "user", "content": "我们决定用 PostgreSQL 作为主数据库"},
        ])
        assert _diag.metrics().get("fact_turns") == 1
        assert _diag.metrics().get("facts_extracted", 0) >= 1

    def test_facts_are_ranked_by_signal_strength(self, config):
        """When the cap bites, signal-carrying facts must survive."""
        config.sync.l2_max_facts_per_turn = 1
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts([
            {"role": "user", "content": "The weather is nice today in the park"},
            {"role": "user", "content": "We decided to use PostgreSQL for the store"},
        ])
        assert len(facts) == 1
        assert "decided" in facts[0]["content"]


# ---------------------------------------------------------------------------
# P2-1 — NULL-vector migration keeps provenance and is recoverable
# ---------------------------------------------------------------------------

class TestNullVectorMigration:
    def test_rows_are_dlq_ed_and_source_rowid_preserved(self, config, tmp_path):
        """No lancedb here: verify the DLQ contract through a stubbed store."""
        queue = WriteQueue(config)

        class _Vec:
            def __init__(self, value):
                self._value = value

            def as_py(self):
                return self._value

        class _Col:
            def __init__(self, values):
                self._values = values

            def __getitem__(self, i):
                return _Vec(self._values[i])

        class _Table:
            num_rows = 2
            column_names = ["content", "category", "source", "timestamp",
                            "vector", "source_rowid", "rowid"]

            def column(self, name):
                data = {
                    "content": ["alpha fact", "beta fact"],
                    "category": ["tech", "work"],
                    "source": ["auto-extract", "auto-extract"],
                    "timestamp": ["t1", "t2"],
                    "vector": [None, None],
                    "source_rowid": [11, 22],
                    "rowid": [0, 1],
                }
                return _Col(data[name])

        class _Store:
            def __init__(self):
                self.deleted: list = []
                self.added: list = []

            def to_arrow(self):
                return _Table()

            def delete(self, where):
                self.deleted.append(where)

            def add(self, rows):
                self.added.extend(rows)

        store = _Store()
        queue._l2_store = store
        queue._embed_model = _FakeEmbedder()

        queue._migrate_null_vectors()

        # The delete+add pair ran, and provenance survived.
        assert store.deleted == ["rowid IN (0, 1)"]
        assert [r["source_rowid"] for r in store.added] == [11, 22]
        assert all(r["vector"] == [0.1, 0.2] for r in store.added)

        dlq = Path(config.l2_db_path).parent / "dlq" / "l2_null_vector_migration.jsonl"
        assert dlq.exists(), "migration rows were not written to the DLQ before deletion"
        lines = [json.loads(l) for l in dlq.read_text(encoding="utf-8").splitlines()]
        assert {row["row"]["source_rowid"] for row in lines} == {11, 22}

    def test_migration_failure_is_reported_as_data_loss(self, config, caplog):
        queue = WriteQueue(config)

        class _Vec:
            def as_py(self):
                return None

        class _Col:
            def __getitem__(self, i):
                return _Vec()

        class _Table:
            num_rows = 1
            column_names = ["content", "category", "source", "timestamp", "vector"]

            def column(self, name):
                return _Col()

        class _Store:
            def to_arrow(self):
                return _Table()

            def delete(self, where):
                raise RuntimeError("lancedb delete exploded")

            def add(self, rows):
                raise AssertionError("add must not run after a failed delete")

        queue._l2_store = _Store()
        queue._embed_model = _FakeEmbedder()

        with caplog.at_level(logging.ERROR):
            queue._migrate_null_vectors()

        assert any(r.levelno >= logging.ERROR for r in caplog.records), (
            "a half-applied migration was swallowed"
        )
        assert _diag.stats()["data_loss"].get("l2_write::null_vector_migration_failed") == 1


class _FakeEmbedder:
    """Minimal embedding backend exposing the legacy .encode() protocol."""

    dim = 2

    class _Result:
        @staticmethod
        def tolist():
            return [[0.1, 0.2], [0.1, 0.2]]

    def encode(self, texts):
        return self._Result()


# ---------------------------------------------------------------------------
# P2-3 — Bridge import safety
# ---------------------------------------------------------------------------

class TestBridgeImportSafety:
    def _bridge(self, config):
        return BridgeExporter(config)

    def test_corrupt_lines_are_preserved_and_quarantined(self, config):
        bridge = self._bridge(config)
        jsonl = bridge._bridge_dir / "candidates.jsonl"
        jsonl.write_text(
            '{"id": "keep", "content": "a durable fact that is long enough", '
            '"tags": ["approved"], "target": "memory"}\n'
            "this-line-is-not-json\n",
            encoding="utf-8",
        )

        report = bridge.import_approved(Path(config.l1_memory_path))

        after = jsonl.read_text(encoding="utf-8")
        assert "this-line-is-not-json" in after, "corrupt line was destroyed"
        assert report["corrupt_lines"] == 1

        quarantine = bridge._bridge_dir / "candidates.corrupt.jsonl"
        assert quarantine.exists()
        assert "this-line-is-not-json" in quarantine.read_text(encoding="utf-8")

    def test_import_is_idempotent_after_l1_failure(self, config, monkeypatch):
        """Mark-first ordering: a failing L1 append must not re-import later."""
        bridge = self._bridge(config)
        jsonl = bridge._bridge_dir / "candidates.jsonl"
        jsonl.write_text(
            json.dumps({"id": "c1", "content": "a durable fact that is long enough",
                        "tags": ["approved"], "target": "memory"}) + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr("builtins.open", _FailingOpen(jsonl))
        report = bridge.import_approved(Path(config.l1_memory_path))
        monkeypatch.undo()

        assert report["imported"] == 0
        assert _diag.stats()["data_loss"].get("bridge::l1_append_failed") == 1

        # Second attempt: the row is already marked imported -> no duplicate.
        second = bridge.import_approved(Path(config.l1_memory_path))
        assert second["imported"] == 0
        assert second["skipped"] >= 1

    def test_no_temp_files_left_behind(self, config):
        bridge = self._bridge(config)
        bridge.export_candidates([
            {"content": "A durable approved fact about deployment", "target": "memory",
             "tags": ["approved"]},
        ])
        bridge.import_approved(Path(config.l1_memory_path))
        leftovers = list(bridge._bridge_dir.glob("*.tmp"))
        assert not leftovers, f"temp files leaked: {leftovers}"


class _FailingOpen:
    """Context-manager factory: fails only when opening the L1 file for append."""

    def __init__(self, allowed_path: Path):
        self._allowed = str(allowed_path)
        self._real_open = open

    def __call__(self, file, mode="r", *args, **kwargs):
        if "a" in mode and str(file) != self._allowed:
            raise OSError("injected L1 append failure")
        return self._real_open(file, mode, *args, **kwargs)


# ---------------------------------------------------------------------------
# P2-4 — secrets are fail-closed: quarantine, never the review feed
# ---------------------------------------------------------------------------

class TestBridgeSecretHandling:
    def test_secret_candidate_is_quarantined_not_stored(self, config):
        bridge = BridgeExporter(config)
        report = bridge.export_candidates([
            {"content": "The API key: sk-abc123def456ghi789jkl0 must be kept secret",
             "target": "memory", "source_path": "ops.md"},
        ])

        assert report["written"] == 0, "secrets must never enter the review feed"
        assert report["quarantined"] == 1

        candidates = (bridge._bridge_dir / "candidates.jsonl").read_text(
            encoding="utf-8").strip()
        assert candidates == "", "candidates.jsonl must stay clean"

        line = (bridge._bridge_dir / "quarantine.jsonl").read_text(
            encoding="utf-8").strip()
        entry = json.loads(line)
        assert "sk-abc123def456ghi789jkl0" not in line, "secret stored verbatim"
        assert "[REDACTED]" in entry["content"]
        assert entry["source"] == "ops.md"
        assert entry["patterns"] == ["openai_style_key"]
        assert bridge.get_status()["quarantine_count"] == 1
        assert _diag.stats()["data_loss"].get("bridge_secret::candidate_quarantined") == 1

    def test_pem_and_bearer_are_detected(self):
        from plugin.memory_governed._bridge import has_secret_like_text

        assert has_secret_like_text("-----BEGIN PRIVATE KEY----- MIIEvQIBADANBg")
        assert has_secret_like_text("-----BEGIN RSA PRIVATE KEY-----")
        assert has_secret_like_text("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.sig")
        assert not has_secret_like_text("we use bearer authentication for the api")

    def test_normal_technical_text_is_not_miskilled(self):
        """Fail-closed means false positives = silently dropped memories.

        These must NEVER match (precision > recall, architect verdict).
        """
        from plugin.memory_governed._bridge import has_secret_like_text

        clean_samples = [
            "our token bucket rate limiter uses a sliding window",
            "the key insight is that caching reduces latency",
            "base64 encoding is commonly used for data transfer",
            "we store the session id as a uuid: 550e8400-e29b-41d4-a716-446655440000",
            "the content hash is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
            "we decided to use postgres instead of mysql",
        ]
        for text in clean_samples:
            assert not has_secret_like_text(text), f"false positive on: {text}"

    def test_escape_hatch_stores_redacted_and_flagged(self, config):
        config.allow_secret_candidates = True
        bridge = BridgeExporter(config)
        report = bridge.export_candidates([
            {"content": "The API key: sk-abc123def456ghi789jkl0 must be kept secret",
             "target": "memory"},
        ])

        assert report["written"] == 1
        assert report["quarantined"] == 0
        line = (bridge._bridge_dir / "candidates.jsonl").read_text(
            encoding="utf-8").strip()
        assert "sk-abc123def456ghi789jkl0" not in line
        assert json.loads(line)["secret_redacted"] is True

        # ...but the L1 hard gate still blocks it from MEMORY.md.
        report2 = bridge.import_approved(Path(config.l1_memory_path), auto_approve=True)
        assert report2["imported"] == 0
        assert report2["blocked_secrets"] == 1

    def test_l1_hard_gate_blocks_flagged_and_historical_rows(self, config):
        """The gate must hold for legacy rows with no flag and no tag."""
        bridge = BridgeExporter(config)
        secret = "The API key: sk-abc123def456ghi789jkl0 must be kept secret"
        rows = [
            {"id": "flagged", "content": "[REDACTED]", "target": "memory",
             "tags": ["approved"], "secret_redacted": True},
            {"id": "tagged", "content": "harmless looking text but tagged", "target": "memory",
             "tags": ["approved", "needs-careful-review"]},
            {"id": "legacy", "content": secret, "target": "memory",
             "tags": ["approved"]},
        ]
        (bridge._bridge_dir / "candidates.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        report = bridge.import_approved(Path(config.l1_memory_path), auto_approve=True)

        assert report["imported"] == 0
        assert report["blocked_secrets"] == 3
        l1 = Path(config.l1_memory_path)
        assert not l1.exists() or "sk-abc123def456ghi789jkl0" not in l1.read_text(
            encoding="utf-8")

    def test_has_secret_like_text_still_detects(self):
        from plugin.memory_governed._bridge import has_secret_like_text

        assert has_secret_like_text("token=abcdefghijklmnop")
        assert not has_secret_like_text("we decided to use postgres")


# ---------------------------------------------------------------------------
# _diag contract
# ---------------------------------------------------------------------------

class TestDiagContract:
    def test_degraded_is_throttled_but_counted(self, caplog):
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                _diag.log_degraded("unit", "reason_a")

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, f"throttling failed: {len(warnings)} records"
        stats = _diag.stats()
        assert stats["degraded"]["unit::reason_a"] == 5
        assert stats["suppressed"] == 4

    def test_data_loss_is_never_throttled(self, caplog):
        with caplog.at_level(logging.ERROR):
            for _ in range(3):
                _diag.log_data_loss("unit", "reason_b")

        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 3
        assert _diag.stats()["data_loss"]["unit::reason_b"] == 3
        assert _diag.stats()["suppressed"] == 0

    def test_different_reasons_are_not_throttled_together(self, caplog):
        with caplog.at_level(logging.WARNING):
            _diag.log_degraded("unit", "reason_c")
            _diag.log_degraded("unit", "reason_d")
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 2

    def test_stats_is_json_serialisable(self):
        _diag.log_degraded("unit", "x")
        _diag.log_data_loss("unit", "y")
        payload = json.dumps(_diag.stats())
        assert json.loads(payload)["degraded"] == {"unit::x": 1}

    def test_exception_detail_is_included(self, caplog):
        with caplog.at_level(logging.ERROR):
            try:
                raise ValueError("kaboom")
            except ValueError as e:
                _diag.log_data_loss("unit", "boom", detail="ctx", exc=e)
        assert any("kaboom" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Graceful degradation — the whole write path without optional dependencies
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_index_l2_without_lancedb_is_reported_not_crashed(self, config, caplog, monkeypatch):
        # 显式模拟 lancedb 缺失，而非依赖真实环境——这样无论本机是否安装
        # lancedb，该用例都稳定验证"优雅降级"路径（本文件 docstring 的契约）。
        import sys

        monkeypatch.setitem(sys.modules, "lancedb", None)
        queue = WriteQueue(config)
        with caplog.at_level(logging.WARNING):
            queue._init_l2()
        assert queue._l2_store is None
        assert queue._embed_fn is None
        assert _diag.stats()["degraded"].get("l2_write::lancedb_missing") == 1

    def test_embed_texts_without_backend_returns_empty(self, config):
        queue = WriteQueue(config)
        assert queue._embed_texts(["hello"]) == []
        facts = [{"content": "we decided to use postgres"}]
        queue._attach_vectors(facts)
        assert "vector" not in facts[0]

    def test_embed_texts_filters_none_entries(self, config):
        queue = WriteQueue(config)

        class _Service:
            available = True
            dim = 2

            @staticmethod
            def embed_batch(texts):
                return [[0.1, 0.2], None]

        queue._embed_model = _Service()
        facts = [{"content": "first fact here"}, {"content": "second fact here"}]
        queue._attach_vectors(facts)
        assert facts[0]["vector"] == [0.1, 0.2]
        assert "vector" not in facts[1]  # aligned, not shifted

    def test_embedding_service_contract_is_honoured(self, config, monkeypatch):
        """_sync must delegate to EmbeddingService (shared singleton contract)."""
        from plugin.memory_governed import _embedding as embedding_module

        seen: dict = {}

        class _Service:
            available = True
            dim = 4
            last_error = ""

            @classmethod
            def get(cls, cfg):
                seen["cfg"] = cfg
                return cls()

            def embed_one(self, text: str):
                return [0.5] * 4

            def embed_batch(self, texts):
                return [[0.5] * 4 for _ in texts]

        monkeypatch.setattr(embedding_module, "EmbeddingService", _Service)

        queue = WriteQueue(config)
        store = _FakeL2Store()
        queue._l2_store = store
        queue._init_embedding_service()

        assert queue._embed_model is not None
        assert callable(queue._embed_fn)
        assert queue._embed_fn("x") == [0.5] * 4
        assert seen["cfg"] is config

        queue._index_l2(
            [{"role": "user", "content": "We decided to use PostgreSQL for the store"}], {}
        )
        assert store.rows and store.rows[0]["vector"] == [0.5] * 4

    def test_embedding_service_failure_degrades(self, config, monkeypatch, caplog):
        from plugin.memory_governed import _embedding as embedding_module

        class _Broken:
            @classmethod
            def get(cls, cfg):
                raise RuntimeError("backend exploded")

        monkeypatch.setattr(embedding_module, "EmbeddingService", _Broken)
        queue = WriteQueue(config)
        with caplog.at_level(logging.WARNING):
            queue._init_embedding_service()

        assert queue._embed_fn is None
        assert queue._embed_model is None
        assert _diag.stats()["degraded"].get("l2_embedding::service_init_failed") == 1

    def test_full_write_roundtrip_without_lancedb(self, config):
        writer = L3Writer(config)
        queue = WriteQueue(config)
        messages = [
            {"role": "user", "content": "我们决定用 PostgreSQL 作为主数据库"},
            {"role": "assistant", "content": "好的，我会记录这个决定。"},
        ]
        rowid_map = writer.write(messages, "s-roundtrip")
        assert rowid_map

        store = _FakeL2Store()
        queue._l2_store = store
        queue._process_item({"messages": messages, "rowid_map": rowid_map})
        writer.shutdown()

        assert store.rows, "no facts extracted in the degraded deployment"
        conn = sqlite3.connect(config.l3_db_path)
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        conn.close()
        assert n_msg == n_fts == 2, f"L3/FTS diverged: {n_msg} vs {n_fts}"
