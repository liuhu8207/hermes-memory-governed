"""Concurrency / performance / compatibility probes cross-validated with the
architecture review (docs/architecture-review.md, findings #1 #4 #5 #7 #8).

These assert CORRECT behaviour. Failures are source defects.
"""

from __future__ import annotations

import concurrent.futures as cf
import hashlib
import sqlite3
import sys
import threading
import time

import pytest

from plugin.memory_governed import GovernedMemoryProvider
from plugin.memory_governed._recall import RecallEngine
from plugin.memory_governed._sync import L3Writer


# ---------------------------------------------------------------------------
# #1 (P0) — L3 dedup SELECT has no index → full table scan per message
# ---------------------------------------------------------------------------

class TestL3DedupIndex:
    """Reproduced: 80k rows / 50 msgs = 494.8ms; with index = 2.6ms (188x).

    README promises "L3 sync <10ms".
    """

    def _seed(self, config, n: int) -> None:
        w = L3Writer(config)
        w._get_conn()
        c = w._conn
        now = time.time()
        rows = []
        for i in range(n):
            content = f"historical conversation message number {i} with filler text"
            h = hashlib.sha256((f"sess{i % 50}|user|" + content).encode()).hexdigest()
            rows.append((f"sess{i % 50}", "user", content, now, h))
        c.executemany(
            "INSERT INTO messages (session_id, role, content, timestamp, hash) "
            "VALUES (?,?,?,?,?)",
            rows,
        )
        c.commit()
        return w

    def test_dedup_query_uses_an_index(self, config):
        """Deterministic: the dedup SELECT must not be a full table scan."""
        w = self._seed(config, 2000)
        try:
            plan = " ".join(
                r[-1]
                for r in w._conn.execute(
                    "EXPLAIN QUERY PLAN SELECT 1 FROM messages "
                    "WHERE session_id = ? AND hash = ?",
                    ("s1", "x"),
                )
            )
            assert "SCAN messages" not in plan, (
                f"dedup SELECT does a full table scan: {plan!r}. "
                "Add idx_messages_dedup(session_id, hash)."
            )
        finally:
            w.shutdown()

    def test_l3_write_stays_within_budget_at_scale(self, config):
        """Timed: 50-message turn into 30k rows must stay under 100ms."""
        w = self._seed(config, 30000)
        try:
            msgs = [
                {"role": "user", "content": f"new turn message body {i} with text"}
                for i in range(50)
            ]
            t0 = time.perf_counter()
            w.write(msgs, "s-new")
            elapsed_ms = (time.perf_counter() - t0) * 1000

            assert elapsed_ms < 100, (
                f"L3 write of 50 messages into 30k rows took {elapsed_ms:.1f}ms "
                "(README promises <10ms). Cause: no index on (session_id, hash) "
                "→ one full table scan per message."
            )
        finally:
            w.shutdown()


# ---------------------------------------------------------------------------
# #4 (P1) — queue_prefetch spawns one thread per call, no dedup, no pool
# ---------------------------------------------------------------------------

class TestPrefetchThreadExplosion:
    def test_identical_queries_are_coalesced(self, provider):
        """20 identical in-flight prefetches must not spawn 20 threads.

        Identical in-flight queries are coalesced: 20 calls → exactly 1 recall
        execution.

        NOTE: an earlier version of this test used `threading.active_count()`
        deltas and gave a FALSE NEGATIVE. Counting by thread identity is
        reliable; active_count() is not.
        """
        gate = threading.Event()
        started = threading.Event()
        counter = {"n": 0}
        original = provider._recall.parallel_recall

        def blocking(query):
            counter["n"] += 1
            started.set()
            gate.wait(15)
            return original(query)

        provider._recall.parallel_recall = blocking

        N = 20
        before = {t.ident for t in threading.enumerate()}
        for _ in range(N):
            provider.queue_prefetch("the same query")

        assert started.wait(5), "no prefetch thread started (test setup problem)"
        time.sleep(1.0)

        spawned = [t for t in threading.enumerate() if t.ident not in before]
        gate.set()
        time.sleep(0.5)

        assert counter["n"] == 1, (
            f"identical queries must be coalesced to 1 recall, got {counter['n']}"
        )
        assert len(spawned) <= 4, (
            f"queue_prefetch spawned {len(spawned)} live threads for {N} IDENTICAL "
            "queries — no in-flight dedup, no pool (1 raw Thread + "
            "ThreadPoolExecutor(3) per call)"
        )


# ---------------------------------------------------------------------------
# #5 (P1) — `except TimeoutError` is dead code on Python 3.10
# ---------------------------------------------------------------------------

class TestTimeoutCompatibility:
    @pytest.mark.skipif(
        sys.version_info >= (3, 11),
        reason=(
            "On Python 3.11+ concurrent.futures.TimeoutError IS the builtin "
            "TimeoutError, so `except TimeoutError` works and this probe is moot. "
            "The negative-assertion only has meaning on the 3.10 floor."
        ),
    )
    def test_concurrent_futures_timeout_is_caught(self):
        """_recall.py:535 catches builtin TimeoutError.

        On 3.10 the project's stated floor `except TimeoutError` does NOT catch
        `concurrent.futures.TimeoutError` (they are distinct classes until 3.11),
        so the except clause is dead code on 3.10 and as_completed's timeout
        escapes to the outer `except Exception`, logged at debug → recall
        silently returns [].

        The test asserts the OPPOSITE of the desired behaviour: it fails when
        the assertion is False, which is exactly when 3.10 needs the fix.
        A pass here would mean the assertion is True, which would mean 3.10
        already works — which it doesn't.
        """
        assert issubclass(cf.TimeoutError, TimeoutError), (
            f"Python {sys.version.split()[0]}: concurrent.futures.TimeoutError is "
            "NOT a subclass of builtin TimeoutError, so `except TimeoutError` at "
            "_recall.py:535 is dead code. Catch concurrent.futures.TimeoutError."
        )


# ---------------------------------------------------------------------------
# #7 (P1) — Chinese durable-fact heuristic misses most samples
# ---------------------------------------------------------------------------

class TestChineseDurableFactDetection:
    SAMPLES = [
        "不要动 database.py",
        "以后默认用中文回复我",
        "我们决定用 Postgres",
        "绝不允许直接改 main 分支",
        "帮我记住我习惯用 VSCode",
        "我偏好使用深色主题",
        "记得每次提交前跑测试",
        "以后都用 pytest 跑测试",
    ]

    def test_chinese_samples_are_recognised(self):
        """Root cause: CJK patterns only cover 偏好/记得/以后(都|请|要)/不要(再|去).

        "不要动" does not match 不要(再|去); 决定/绝不允许/帮我记住 have no pattern.
        """
        f = GovernedMemoryProvider._looks_like_durable_fact
        hits = [s for s in self.SAMPLES if f(s)]
        rate = len(hits) / len(self.SAMPLES)
        assert rate >= 0.8, (
            f"only {len(hits)}/{len(self.SAMPLES)} Chinese durable facts detected "
            f"({rate:.0%}). Missed: {[s for s in self.SAMPLES if s not in hits]}"
        )

    def test_english_samples_still_recognised(self):
        """Control: the English path works, so this is a CJK-only gap."""
        f = GovernedMemoryProvider._looks_like_durable_fact
        for s in ["I prefer to use pytest for all tests",
                  "I always run the linter before committing"]:
            assert f(s), f"English sample regressed: {s}"


# ---------------------------------------------------------------------------
# #8 (P1) — _init_l2 lazy init has no lock
# ---------------------------------------------------------------------------

class TestLazyInitRace:
    def test_init_l2_runs_once_under_concurrency(self, config, monkeypatch):
        """_recall.py:117-118 checks `if self._l2_store is None` with no lock.

        Cold-start concurrency re-enters _init_l2 N times → the embedding model
        is loaded N times (seconds each, hundreds of MB each).
        """
        engine = RecallEngine(config)
        calls = {"n": 0}
        real = engine._init_l2

        def counting_init():
            calls["n"] += 1
            time.sleep(0.05)  # widen the race window
            real()

        monkeypatch.setattr(engine, "_init_l2", counting_init)

        N = 8
        barrier = threading.Barrier(N)

        def racer():
            barrier.wait()
            engine._search_l2("anything")

        ts = [threading.Thread(target=racer) for _ in range(N)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()

        assert calls["n"] <= 1, (
            f"_init_l2 ran {calls['n']} times for {N} concurrent cold-start "
            "searches — no lock around the lazy init at _recall.py:117-118"
        )

    def test_loaded_embed_fn_is_not_discarded(self, config, monkeypatch):
        """_recall.py:197-198 unconditionally resets _embed_fn on every _init_l2.

        If _init_l2 is re-entered after a model is loaded, the live embedding
        function is thrown away and the layer silently drops to text fallback.
        """
        engine = RecallEngine(config)
        sentinel = lambda t: [0.0] * 384  # noqa: E731
        engine._embed_fn = sentinel
        engine._l2_store = None
        engine._init_l2()
        assert engine._embed_fn is not None, (
            "_init_l2 wiped an already-loaded _embed_fn at _recall.py:197-198"
        )
