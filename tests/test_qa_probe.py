"""QA probe tests — independent verification of suspected defects.

These tests assert the CORRECT behaviour (per README design goals and module
docstrings).  A failure here indicates a source-code defect, not a bad test.

Run:
    python -m pytest tests/test_qa_probe.py -v
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugin.memory_governed import GovernedMemoryProvider
from plugin.memory_governed._bridge import BridgeExporter
from plugin.memory_governed._config import load_governed_config
from plugin.memory_governed._recall import RecallEngine, RecallResult
from plugin.memory_governed._sync import L3Writer, WriteQueue


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mk_l3(config, with_fts=True):
    """Create a fresh L3 db, optionally with the 4-column FTS table."""
    conn = sqlite3.connect(config.l3_db_path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL, hash TEXT)"
    )
    if with_fts:
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
        )
    conn.commit()
    conn.close()


def _count_l3_rows(db_path: str, like: str | None = None) -> int:
    """Count rows in L3 `messages`. Returns 0 if the table was never created."""
    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if not exists:
            return 0
        if like:
            return conn.execute(
                "SELECT COUNT(*) FROM messages WHERE content LIKE ?", (f"%{like}%",)
            ).fetchone()[0]
        return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()


def _is_closed(conn) -> bool:
    try:
        conn.execute("SELECT 1")
        return False
    except sqlite3.ProgrammingError:
        return True


# ---------------------------------------------------------------------------
# P0 — Data loss: sync_turn ignores user_content / assistant_content
# ---------------------------------------------------------------------------

class TestSyncTurnDataLoss:
    def test_sync_turn_without_messages_loses_the_turn(self, provider):
        """user_content/assistant_content are declared but never used.

        A host that calls sync_turn("hi", "hello") without ``messages`` gets a
        silent no-op — the turn is never archived anywhere.
        """
        provider.sync_turn(
            "I always prefer dark mode in every editor",
            "Noted, dark mode preference stored.",
            session_id="s1",
        )
        time.sleep(0.3)

        n = _count_l3_rows(provider._config.l3_db_path)
        # BUG: parameters are dead — nothing is persisted, not even the table.
        assert n == 2, (
            "sync_turn ignored user_content/assistant_content: "
            f"{n} rows archived, expected 2 (user + assistant)"
        )

    def test_sync_turn_params_are_dead(self, provider):
        """Objective check: user_content never reaches L3."""
        provider.sync_turn("UNIQUE_SENTINEL_USER_TEXT", "", session_id="s2")
        time.sleep(0.3)
        hit = _count_l3_rows(provider._config.l3_db_path, "UNIQUE_SENTINEL_USER_TEXT")
        assert hit == 1, "user_content is a dead parameter — turn data is dropped"

    def test_sync_turn_with_messages_does_persist(self, provider):
        """Control: the messages= path works, proving only the params are dead."""
        provider.sync_turn(
            "", "",
            session_id="s3",
            messages=[
                {"role": "user", "content": "CONTROL_USER_TEXT"},
                {"role": "assistant", "content": "CONTROL_ASSISTANT_TEXT"},
            ],
        )
        time.sleep(0.3)
        hit = _count_l3_rows(provider._config.l3_db_path, "CONTROL_USER_TEXT")
        assert hit == 1, "the messages= path is the only working write path"


# ---------------------------------------------------------------------------
# P1 — Unbounded prefetch cache (no LRU / no eviction)
# ---------------------------------------------------------------------------

class TestPrefetchCacheBounds:
    def test_prefetch_cache_is_bounded(self, provider):
        """_prefetch_cache grows forever: one entry per unique query string."""
        for i in range(2000):
            provider.queue_prefetch(f"unique query number {i}")
            provider._prefetch_cache[f"unique query number {i}"] = ("x" * 500, time.time())

        assert len(provider._prefetch_cache) <= 256, (
            f"_prefetch_cache unbounded: {len(provider._prefetch_cache)} entries "
            "retained with no TTL sweep or LRU eviction"
        )

    def test_expired_entries_are_never_swept(self, provider):
        """Expired entries stay in the dict forever (only overwritten on re-read)."""
        provider._prefetch_cache["old"] = ("stale", time.time() - 99999)
        provider._cache_hits = 0
        provider.prefetch("old")  # reads it, sees expiry, counts a MISS
        assert "old" not in provider._prefetch_cache, (
            "expired entry was not evicted on read — dict keeps growing"
        )


# ---------------------------------------------------------------------------
# P1 — queue_prefetch spawns one thread per call, no dedup
# ---------------------------------------------------------------------------

# NOTE: the thread-explosion check lives in test_qa_concurrency.py
# (TestPrefetchThreadExplosion). An earlier version of it lived here but used
# `threading.active_count()` deltas, which proved UNRELIABLE — it reported a
# false negative (passed while 20 threads were actually alive). Counting by
# thread identity is used instead. Do not reintroduce the active_count() form.


# ---------------------------------------------------------------------------
# P1 — L3Writer connection is bound to its creating thread
# ---------------------------------------------------------------------------

class TestL3ThreadSafety:
    def test_l3_writer_usable_from_multiple_threads(self, config):
        """Connection is created lazily with default check_same_thread=True.

        The first caller's thread owns the connection; a later caller on a
        different thread raises ProgrammingError, losing the write.
        """
        _mk_l3(config)
        w = L3Writer(config)
        errors = []

        def worker(i):
            try:
                w.write([{"role": "user", "content": f"thread message {i}"}], "s1")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        conn = sqlite3.connect(config.l3_db_path)
        n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.close()

        assert not errors, f"concurrent L3 writes raised: {errors}"
        assert n == 4, f"only {n}/4 messages survived concurrent writes"
        w.shutdown()


# ---------------------------------------------------------------------------
# P0 — FTS insert failure is swallowed → L3 and FTS index diverge silently
# ---------------------------------------------------------------------------

class TestL3IndexDivergence:
    def test_fts_failure_is_reported(self, config, caplog):
        """When the FTS insert fails the row still lands in `messages`.

        Result: the message is archived but permanently unsearchable, and
        nothing is logged — the failure is invisible.
        """
        # FTS table with the WRONG column count -> the 4-value insert fails.
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

        w = L3Writer(config)
        with caplog.at_level("WARNING"):
            w.write([{"role": "user", "content": "hello world"}], "s1")

        conn = sqlite3.connect(config.l3_db_path)
        n_msg = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
        conn.close()

        assert n_msg == n_fts, (
            f"L3 divergence: messages={n_msg} but messages_fts={n_fts}. "
            "Row is archived but permanently unsearchable, and no error surfaced."
        )
        assert caplog.records, (
            "FTS insert failed completely silently (bare `except Exception: pass`)"
        )
        w.shutdown()


# ---------------------------------------------------------------------------
# P2 — SQLite connections leak on exception paths
# ---------------------------------------------------------------------------

class TestConnectionLeaks:
    def test_search_l3_closes_connection_on_error(self, config, monkeypatch):
        _mk_l3(config)
        opened = []
        real_connect = sqlite3.connect

        def tracking(*a, **kw):
            c = real_connect(*a, **kw)
            opened.append(c)
            return c

        monkeypatch.setattr(sqlite3, "connect", tracking)

        engine = RecallEngine(config)

        def boom(_q):
            raise RuntimeError("injected")

        monkeypatch.setattr(engine, "_build_fts_query", boom)
        engine._search_l3("anything")

        assert opened, "no connection was opened (test setup problem)"
        leaked = [c for c in opened if not _is_closed(c)]
        assert not leaked, (
            f"_search_l3 leaked {len(leaked)} sqlite connection(s) — "
            "conn.close() is not in a finally block"
        )

    def test_search_l3_like_closes_connection_on_error(self, config, monkeypatch):
        _mk_l3(config)
        opened = []
        real_connect = sqlite3.connect

        def tracking(*a, **kw):
            c = real_connect(*a, **kw)
            opened.append(c)
            return c

        monkeypatch.setattr(sqlite3, "connect", tracking)

        engine = RecallEngine(config)
        monkeypatch.setattr(
            engine, "_escape_like", lambda t: (_ for _ in ()).throw(RuntimeError("x"))
        )
        engine._search_l3_like("中文")

        leaked = [c for c in opened if not _is_closed(c)]
        assert not leaked, (
            f"_search_l3_like leaked {len(leaked)} sqlite connection(s) on error"
        )

    def test_cjk_segments_open_one_connection_each(self, config, monkeypatch):
        """Mixed EN+CJK query re-opens sqlite once per CJK segment."""
        _mk_l3(config)
        opened = []
        real_connect = sqlite3.connect

        def tracking(*a, **kw):
            c = real_connect(*a, **kw)
            opened.append(c)
            return c

        monkeypatch.setattr(sqlite3, "connect", tracking)

        engine = RecallEngine(config)
        engine._search_l3("python 今天 天气 北京 部署")

        assert len(opened) <= 1, (
            f"opened {len(opened)} sqlite connections for one query "
            "(one per CJK segment + full-table LIKE scan each)"
        )


# ---------------------------------------------------------------------------
# P1 — Write path truncated by a READ-path config knob
# ---------------------------------------------------------------------------

class TestFactExtractionTruncation:
    def test_all_facts_are_extracted(self, config):
        """_extract_atomic_facts caps output at recall.l2_max_results (a read knob)."""
        q = WriteQueue(config)
        msgs = [
            {"role": "user", "content": f"The project deadline for milestone {i} is next Friday."}
            for i in range(40)
        ]
        facts = q._extract_atomic_facts(msgs)
        assert len(facts) == 40, (
            f"only {len(facts)}/40 facts extracted — write path is capped by "
            f"recall.l2_max_results={config.recall.l2_max_results} (a READ setting)"
        )


# ---------------------------------------------------------------------------
# P1 — L2 → L3 audit drill-down never resolves
# ---------------------------------------------------------------------------

class TestSourceRowidPropagation:
    def test_rowid_map_matches_extracted_facts(self, config):
        """rowid_map is keyed by FULL message content; facts are SENTENCES.

        `_index_l2` looks up rmap.get(fact["content"]) — a sentence — against a
        dict keyed by the whole message, so it essentially never matches and
        source_rowid is never populated.
        """
        _mk_l3(config)
        w = L3Writer(config)
        content = "I always prefer dark mode. The project deadline is next Friday."
        rowid_map = w.write([{"role": "user", "content": content}], "s1")

        q = WriteQueue(config)
        facts = q._extract_atomic_facts([{"role": "user", "content": content}])
        assert facts, "no facts extracted (test setup problem)"

        resolved = [
            f["content"] for f in facts
            if rowid_map.get(f["content"]) is not None
        ]
        assert resolved, (
            "source_rowid never resolves: facts are sentence fragments but "
            f"rowid_map is keyed by full message content. facts={[f['content'] for f in facts]}"
        )
        w.shutdown()


# ---------------------------------------------------------------------------
# P1 — CJK multi-term queries with spaces fail to match
# ---------------------------------------------------------------------------

class TestCjkRecall:
    def test_multi_term_cjk_query_matches(self, config):
        """_search_l3_like strips ALL spaces, so 'A B' requires 'AB' adjacency."""
        _mk_l3(config)
        conn = sqlite3.connect(config.l3_db_path)
        now = time.time()
        text = "今天天气不错，北京的空气质量也还可以，适合出门跑步"
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            ("s", "user", text, now),
        )
        conn.execute(
            "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?,?,?,?)",
            (text, "user", "s", now),
        )
        conn.commit()
        conn.close()

        engine = RecallEngine(config)
        results = engine._search_l3("天气 北京")
        assert results, (
            "multi-term CJK query '天气 北京' returned nothing: spaces are stripped "
            "into '天气北京' which requires the terms to be adjacent in the source"
        )


# ---------------------------------------------------------------------------
# P2 — WriteQueue attribute defined outside __init__
# ---------------------------------------------------------------------------

class TestWriteQueueState:
    def test_embed_fn_initialised_in_constructor(self, config):
        q = WriteQueue(config)
        assert hasattr(q, "_embed_fn"), (
            "WriteQueue._embed_fn is only created inside _init_l2(); if _init_l2 "
            "raises early, _index_l2 line 145 raises AttributeError (swallowed)"
        )

    def test_stop_joins_worker_thread(self, config, monkeypatch):
        """stop() does not join the worker → in-flight item is killed at exit."""
        q = WriteQueue(config)
        started = threading.Event()
        release = threading.Event()

        def slow(item):
            started.set()
            release.wait(5)

        monkeypatch.setattr(q, "_process_item", slow)
        q.start()
        q.enqueue([{"role": "user", "content": "hello there friend" * 3}], "s1")
        assert started.wait(3), "worker never started"

        try:
            q.stop()
            assert not q._worker_thread.is_alive(), (
                "stop() returned while the worker was still mid-item — "
                "the daemon thread is killed at interpreter exit and the item is lost"
            )
        finally:
            release.set()

    def test_stop_drain_survives_item_error(self, config, monkeypatch):
        """A raising item aborts the whole drain; remaining items are lost."""
        q = WriteQueue(config)
        calls = []

        def flaky(item):
            calls.append(item)
            if len(calls) == 1:
                raise RuntimeError("boom")

        monkeypatch.setattr(q, "_process_item", flaky)
        q._queue.put_nowait({"messages": [{"a": 1}]})
        q._queue.put_nowait({"messages": [{"b": 2}]})

        try:
            q.stop()
        except RuntimeError:
            pass  # stop() propagates the item error

        assert len(calls) == 2, (
            f"drain aborted after {len(calls)}/2 items — the first error "
            "propagated out of stop() and the rest of the queue was discarded"
        )


# ---------------------------------------------------------------------------
# P2 — L1 cache ignores external file changes
# ---------------------------------------------------------------------------

class TestL1CacheFreshness:
    def test_l1_picks_up_external_file_change(self, config):
        """Docstring promises refresh 'if file changed'; reality is a blind 60s TTL."""
        engine = RecallEngine(config)
        first = engine.get_l1()
        Path(config.l1_memory_path).write_text(
            "# Rules\n- EXTERNALLY EDITED CONTENT", encoding="utf-8"
        )
        second = engine.get_l1()
        assert "EXTERNALLY EDITED" in second, (
            "L1 cache is time-based only (60s TTL) — edits made by scripts or "
            "another process are invisible for up to a minute"
        )


# ---------------------------------------------------------------------------
# P2 — Sentence splitting corrupts numbers / URLs / versions
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_urls_and_versions_survive_splitting(self):
        text = "We are shipping version 3.14 today. See https://example.com/docs for info."
        sents = WriteQueue._split_sentences(text)
        assert "We are shipping version 3.14 today" in sents, (
            f"split on bare '.' shredded the sentence: {sents}"
        )

    def test_decimal_numbers_survive_splitting(self):
        sents = WriteQueue._split_sentences("The p99 latency is 12.5 ms in production.")
        assert "The p99 latency is 12.5 ms in production" in sents, (
            f"decimal split apart: {sents}"
        )


# ---------------------------------------------------------------------------
# P2 — Bridge: secret handling and concurrent-export dedup
# ---------------------------------------------------------------------------

class TestBridgeRobustness:
    def test_concurrent_export_no_duplicates(self, config):
        """Regression guard: read-ids-then-append has no lock.

        Currently PASSES (GIL serialises the critical section), but the
        read-then-append window is a real race on a free-threaded build.
        """
        bridge = BridgeExporter(config)
        results = []
        errors = []

        def worker(i):
            try:
                results.append(
                    bridge.export_candidates([
                        {
                            "content": f"shared durable fact number {i} about deployment",
                            "target": "memory",
                            "source_path": f"file{i}.md",
                        }
                    ])
                )
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent export raised: {errors}"
        total_written = sum(r["written"] for r in results)
        total_lines = bridge.get_status()["candidate_count"]
        assert total_lines == total_written, (
            f"jsonl has {total_lines} lines but {total_written} were reported "
            "written — concurrent exports duplicated entries (no file lock)"
        )

    def test_corrupt_lines_are_preserved_on_import(self, config):
        """import_approved() silently deletes unparsable lines when rewriting."""
        bridge = BridgeExporter(config)
        jsonl = bridge._bridge_dir / "candidates.jsonl"
        jsonl.write_text(
            '{"id": "keep", "content": "a durable fact that is long enough", '
            '"tags": ["approved"], "target": "memory"}\n'
            "this-line-is-not-json\n",
            encoding="utf-8",
        )
        l1 = Path(config.l1_memory_path)
        bridge.import_approved(l1)
        after = jsonl.read_text(encoding="utf-8")
        assert "this-line-is-not-json" in after, (
            "import_approved() silently dropped the corrupt line instead of "
            "quarantining it — data is destroyed on every import cycle"
        )


# ---------------------------------------------------------------------------
# Performance — README claims
# ---------------------------------------------------------------------------

class TestPerformanceClaims:
    def test_prefetch_cache_hit_under_1ms(self, provider):
        provider.queue_prefetch("python rules")
        for _ in range(50):
            if provider._prefetch_cache:
                break
            time.sleep(0.05)

        t0 = time.perf_counter()
        for _ in range(1000):
            provider.prefetch("python rules")
        avg_ms = (time.perf_counter() - t0) / 1000 * 1000
        assert avg_ms < 1.0, f"prefetch cache hit took {avg_ms:.3f}ms (README: <1ms)"

    def test_sync_turn_is_non_blocking(self, provider, config):
        _mk_l3(config)
        messages = [
            {"role": "user", "content": f"message body number {i} with some text" * 20}
            for i in range(200)
        ]
        # warm up connection + table
        provider.sync_turn("", "", session_id="warm", messages=messages[:1])

        t0 = time.perf_counter()
        provider.sync_turn("", "", session_id="perf", messages=messages)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 50, (
            f"sync_turn blocked {elapsed_ms:.1f}ms for 200 messages "
            "(README claims non-blocking, L3 sync <10ms)"
        )


# ---------------------------------------------------------------------------
# Graceful degradation — must NOT raise when optional deps are missing
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    def test_initialize_without_lancedb(self, hermes_home):
        """lancedb / sentence-transformers are absent in this env."""
        p = GovernedMemoryProvider()
        p.initialize("sess", hermes_home=str(hermes_home))
        assert p._recall is not None
        assert p._write_queue is not None
        p.shutdown()

    def test_l2_search_degrades_to_empty(self, config):
        engine = RecallEngine(config)
        assert engine._search_l2("anything") == []

    def test_parallel_recall_works_without_l2(self, config):
        _mk_l3(config)
        engine = RecallEngine(config)
        engine._search_l3("python")
        results = engine.parallel_recall("python")
        assert isinstance(results, list)

    def test_full_turn_roundtrip_without_lancedb(self, provider, config):
        _mk_l3(config)
        provider.sync_turn(
            "", "",
            session_id="rt",
            messages=[
                {"role": "user", "content": "The deployment deadline is next Friday"},
                {"role": "assistant", "content": "Understood, I will note the deadline"},
            ],
        )
        time.sleep(0.5)
        provider.queue_prefetch("deployment deadline")
        for _ in range(60):
            if provider._prefetch_cache:
                break
            time.sleep(0.05)
        out = provider.prefetch("deployment deadline")
        assert "deployment deadline" in out.lower(), (
            f"round-trip lost the archived turn. prefetch returned: {out[:200]!r}"
        )
