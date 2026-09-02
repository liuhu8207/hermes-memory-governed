"""Read-path / embedding-layer regression tests (T03 + T04).

Scope: ``__init__.py`` (prefetch cache, queue_prefetch, prefetch self-heal,
sync_turn content fallback, recall_status), ``_recall.py`` (L1 mtime, L2 lazy
init, L3 CJK recall, sqlite connection hygiene) and ``_embedding.py``
(EmbeddingService singleton / graceful degradation).

All tests must pass on a bare interpreter — lancedb / sentence-transformers /
fastembed / pyarrow are optional and mocked or absent here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from plugin.memory_governed import GovernedMemoryProvider
from plugin.memory_governed._config import load_governed_config
from plugin.memory_governed._embedding import EmbeddingService
from plugin.memory_governed._recall import RecallEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hermes_home(tmp_path):
    """Minimal HERMES_HOME with L1-L4 files + bridge dir."""
    home = tmp_path / ".hermes"
    mem = home / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(
        "# Rules\n- Don't refactor old modules\n- Mobile still uses them",
        encoding="utf-8",
    )
    (mem / "USER.md").write_text(
        "# User\n- Name: Test User\n- Prefers concise replies", encoding="utf-8",
    )
    (mem / "l2").mkdir()
    (mem / "l3").mkdir()
    (mem / "persona.md").write_text(
        "# Profile\n- Language: Chinese/English", encoding="utf-8",
    )
    bridge = home / "cron" / "output" / "scope_recall_bridge"
    bridge.mkdir(parents=True)

    (home / "governed_memory.json").write_text(
        json.dumps(
            {
                "l1_memory_path": str(mem / "MEMORY.md"),
                "l1_user_path": str(mem / "USER.md"),
                "l2_db_path": str(mem / "l2"),
                "l3_db_path": str(mem / "l3" / "l3.db"),
                "l4_persona_path": str(mem / "persona.md"),
                "l4_meta_path": str(mem / "persona_meta.json"),
                "bridge_dir": str(bridge),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return home


@pytest.fixture
def config(hermes_home):
    return load_governed_config(str(hermes_home))


@pytest.fixture
def provider(hermes_home):
    p = GovernedMemoryProvider()
    p.initialize("sess", hermes_home=str(hermes_home))
    yield p
    p.shutdown()


@pytest.fixture(autouse=True)
def _reset_embedding_singleton():
    """EmbeddingService 是进程内单例 —— 每个用例前后重置，避免互相污染。"""
    EmbeddingService.reset()
    yield
    EmbeddingService.reset()


def _mk_l3(config, with_fts=True):
    """Create the L3 db (mirrors L3Writer's schema)."""
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


def _insert_l3(config, content, role="user", session="s1"):
    conn = sqlite3.connect(config.l3_db_path)
    now = time.time()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        (session, role, content, now),
    )
    conn.execute(
        "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?,?,?,?)",
        (content, role, session, now),
    )
    conn.commit()
    conn.close()


def _count_l3(db_path: str, like: str | None = None) -> int:
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


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class _TrackingConnect:
    """Context helper that records every sqlite3 connection created."""

    def __init__(self, monkeypatch):
        self.monkeypatch = monkeypatch
        self.opened: list = []
        self._real = sqlite3.connect

    def __enter__(self):
        real = self._real
        opened = self.opened

        def tracking(*args, **kwargs):
            conn = real(*args, **kwargs)
            opened.append(conn)
            return conn

        self.monkeypatch.setattr(sqlite3, "connect", tracking)
        return self

    def __exit__(self, *exc_info):
        self.monkeypatch.undo()
        return False

    @staticmethod
    def is_closed(conn) -> bool:
        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            return True
        except Exception:  # noqa: BLE001 - 连接已被破坏也算关闭
            return True
        return False

    def leaked(self):
        return [c for c in self.opened if not self.is_closed(c)]


# ---------------------------------------------------------------------------
# P1-1 — bounded LRU prefetch cache
# ---------------------------------------------------------------------------

class TestPrefetchCacheBounded:
    def test_cache_never_exceeds_capacity(self, provider):
        """注入 200 条（直接赋值，最容易绕过 LRU 的路径）后仍受上限约束。"""
        for i in range(200):
            provider._prefetch_cache[f"query-{i}"] = ("x" * 500, time.time())

        assert len(provider._prefetch_cache) <= provider._PREFETCH_CACHE_MAX
        assert len(provider._prefetch_cache) <= 64

    def test_cache_is_lru_ordered(self, provider):
        """写入超容后，最旧的条目被淘汰，最新的保留。"""
        for i in range(64):
            provider._prefetch_cache[f"q{i}"] = (f"v{i}", time.time())
        assert "q0" in provider._prefetch_cache

        provider._prefetch_cache["fresh"] = ("fresh", time.time())
        assert len(provider._prefetch_cache) <= 64
        assert "fresh" in provider._prefetch_cache
        assert "q0" not in provider._prefetch_cache  # 最旧的被淘汰

    def test_expired_entries_are_swept(self, provider):
        """过期条目不会永久赖在缓存里。"""
        provider._prefetch_cache["stale"] = ("old", time.time() - 99999)
        provider._prefetch_cache["fresh"] = ("new", time.time())
        provider._prefetch_cache.sweep_expired()

        assert "stale" not in provider._prefetch_cache
        assert "fresh" in provider._prefetch_cache

    def test_expired_entry_evicted_on_read(self, provider):
        """读到过期条目时顺手删除（而不是留着等覆盖）。"""
        provider._prefetch_cache["old"] = ("stale", time.time() - 99999)
        provider._cache_hits = 0
        provider._cache_misses = 0

        provider.prefetch("old")  # miss

        assert "old" not in provider._prefetch_cache
        assert provider._cache_misses == 1

    def test_sweep_is_bounded_per_write(self, provider):
        """单次写入只清扫一小批，不做全量遍历（避免写入放大）。"""
        assert provider._prefetch_cache.sweep_limit == provider._PREFETCH_CACHE_SWEEP


# ---------------------------------------------------------------------------
# P1-2 — shared executor + in-flight dedup
# ---------------------------------------------------------------------------

class TestQueuePrefetchDedup:
    def test_identical_query_recalls_once(self, provider):
        """同一 query 并发/连续调用只触发一次真正的召回。"""
        gate = threading.Event()
        entered = threading.Event()
        counter = {"n": 0}
        original = provider._recall.parallel_recall

        def blocking_recall(query):
            counter["n"] += 1
            entered.set()
            gate.wait(5)
            return original(query)

        provider._recall.parallel_recall = blocking_recall
        try:
            for _ in range(20):
                provider.queue_prefetch("dup-query")

            assert entered.wait(3), "recall never started (setup problem)"
            time.sleep(0.4)
            assert counter["n"] == 1, f"dedup failed: {counter['n']} recalls for 20 calls"
        finally:
            gate.set()
            time.sleep(0.3)

    def test_fresh_cache_entry_skips_recall(self, provider):
        """缓存未过期时 queue_prefetch 不应再发起召回。"""
        provider._prefetch_cache["warm"] = ("cached", time.time())
        counter = {"n": 0}
        original = provider._recall.parallel_recall

        def counting(query):
            counter["n"] += 1
            return original(query)

        provider._recall.parallel_recall = counting
        provider.queue_prefetch("warm")
        time.sleep(0.5)
        assert counter["n"] == 0

    def test_inflight_is_released_after_failure(self, provider):
        """召回抛异常时也必须摘掉 in-flight 标记，否则该 query 永久无法召回。"""
        from plugin.memory_governed import _inflight, _inflight_lock

        def exploding(_query):
            raise RuntimeError("injected failure")

        provider._recall.parallel_recall = exploding
        provider.queue_prefetch("boom-query")

        assert _wait_until(lambda: "boom-query" not in _inflight)

        with _inflight_lock:
            assert "boom-query" not in _inflight

        # 失败之后仍然可以重新提交（不会被永久锁死）
        assert provider._submit_prefetch("boom-query") is True

    def test_shared_executor_replaces_thread_per_call(self, provider):
        """30 次调用不应该产生 30 个存活线程。"""
        provider._write_queue.stop()  # 排除写队列线程干扰
        base = {t.ident for t in threading.enumerate()}

        gate = threading.Event()
        original = provider._recall.parallel_recall
        provider._recall.parallel_recall = lambda q: (gate.wait(5), original(q))[1]

        for i in range(30):
            provider.queue_prefetch(f"query-{i}")
        time.sleep(0.5)

        spawned = [t for t in threading.enumerate() if t.ident not in base]
        names = [t.name for t in spawned]
        gate.set()
        time.sleep(0.3)

        assert len(spawned) <= 4, f"30 次 queue_prefetch 起了 {len(spawned)} 个线程: {names}"


# ---------------------------------------------------------------------------
# P1-3 — prefetch self-heal
# ---------------------------------------------------------------------------

class TestPrefetchSelfHeal:
    def test_miss_triggers_background_recall(self, provider, config):
        """未命中时不应静默降级：应立即返回 L1，同时在后台补齐完整召回。"""
        _mk_l3(config)
        _insert_l3(config, "The deployment deadline is next Friday", session="rt")

        out = provider.prefetch("deployment deadline")
        assert out, "prefetch 未命中时至少要返回 L1 兜底"
        assert "[User Rules]" in out

        assert _wait_until(lambda: "deployment deadline" in provider._prefetch_cache)

        healed = provider.prefetch("deployment deadline")
        assert "deployment deadline" in healed.lower()

    def test_prefetch_never_blocks(self, provider, config):
        """prefetch 必须严格非阻塞（自愈召回在后台跑）。"""
        _mk_l3(config)
        started = threading.Event()
        original = provider._recall.parallel_recall

        def slow_recall(query):
            started.set()
            time.sleep(1.5)
            return original(query)

        provider._recall.parallel_recall = slow_recall
        t0 = time.perf_counter()
        provider.prefetch("never-seen-query")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 100, f"prefetch 阻塞了 {elapsed_ms:.1f}ms"
        started.wait(2)

    def test_cache_hit_still_under_1ms(self, provider):
        """LRU 改造不能让命中路径退化（架构师基线 p50=0.0004ms）。"""
        provider.queue_prefetch("python rules")
        assert _wait_until(lambda: "python rules" in provider._prefetch_cache)

        provider.prefetch("python rules")  # 预热
        t0 = time.perf_counter()
        for _ in range(2000):
            provider.prefetch("python rules")
        avg_ms = (time.perf_counter() - t0) / 2000 * 1000

        assert avg_ms < 1.0, f"prefetch 命中平均 {avg_ms:.4f}ms（要求 <1ms）"


# ---------------------------------------------------------------------------
# P1-4 — sync_turn content parameters are not dead
# ---------------------------------------------------------------------------

class TestSyncTurnContentParams:
    def test_content_only_turn_is_archived(self, provider):
        """只传 user_content / assistant_content 时必须落库，不能静默丢弃。"""
        provider.sync_turn(
            "SYNTH_USER_TEXT",
            "SYNTH_ASSISTANT_TEXT",
            session_id="s-synth",
        )
        time.sleep(0.3)

        db = provider._config.l3_db_path
        assert _count_l3(db) == 2
        assert _count_l3(db, "SYNTH_USER_TEXT") == 1
        assert _count_l3(db, "SYNTH_ASSISTANT_TEXT") == 1

    def test_empty_content_is_filtered(self, provider):
        """空内容不应产生空行（与真实消息同构，不加额外字段）。"""
        provider.sync_turn("ONLY_USER_TEXT", "", session_id="s-partial")
        time.sleep(0.3)

        db = provider._config.l3_db_path
        assert _count_l3(db) == 1
        assert _count_l3(db, "ONLY_USER_TEXT") == 1

    def test_messages_path_still_wins(self, provider):
        """显式传 messages= 时行为不变。"""
        provider.sync_turn(
            "ignored", "ignored",
            session_id="s-msgs",
            messages=[
                {"role": "user", "content": "EXPLICIT_USER"},
                {"role": "assistant", "content": "EXPLICIT_ASSISTANT"},
            ],
        )
        time.sleep(0.3)

        db = provider._config.l3_db_path
        assert _count_l3(db, "EXPLICIT_USER") == 1
        assert _count_l3(db, "ignored") == 0

    def test_synthesized_messages_are_plain_dicts(self, provider):
        """构造出的消息保持同构：只有 role/content 两个键。"""
        messages = GovernedMemoryProvider._messages_from_contents("u", "a")
        assert messages == [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
        assert GovernedMemoryProvider._messages_from_contents("", "") == []


# ---------------------------------------------------------------------------
# P1-5 — L3 write failure is visible (data-loss logging)
# ---------------------------------------------------------------------------

class TestL3FailureVisibility:
    def test_l3_failure_is_logged_not_swallowed(self, provider, caplog):
        """L3 写失败必须留下 ERROR 级痕迹，而不是只剩 debug。"""
        calls = {"n": 0}

        def exploding(messages, session_id):
            calls["n"] += 1
            raise RuntimeError("injected L3 failure")

        provider._l3_writer.write = exploding
        with caplog.at_level("ERROR"):
            provider.sync_turn("u", "a", session_id="s-fail")

        assert calls["n"] == 1
        assert any(r.levelname == "ERROR" for r in caplog.records), (
            "L3 写失败没有产生任何 ERROR 级日志"
        )

    def test_l2_still_queued_when_l3_fails(self, provider, caplog):
        """L3 失败时 L2 仍然入队 —— 不能因为缺溯源就把整轮数据丢掉。"""
        seen = {}

        def exploding(messages, session_id):
            raise RuntimeError("injected L3 failure")

        def recording(messages, session_id="", rowid_map=None, **kwargs):
            seen["messages"] = messages
            seen["kwargs"] = kwargs

        provider._l3_writer.write = exploding
        provider._write_queue.enqueue = recording
        with caplog.at_level("ERROR"):
            provider.sync_turn("u", "a", session_id="s-fail2")

        assert seen.get("messages"), "L3 失败后 L2 也不再入队 —— 整轮数据全丢"
        # 队列若支持该标记，必须为 True；不支持（旧签名）时至少不能丢数据
        if "provenance_missing" in seen.get("kwargs", {}):
            assert seen["kwargs"]["provenance_missing"] is True

    def test_provenance_flag_is_passed_when_supported(self, provider, monkeypatch):
        """一旦 WriteQueue.enqueue 支持 provenance_missing，必须如实传 True。"""
        import plugin.memory_governed as pkg
        from plugin.memory_governed import _sync

        seen = {}

        def exploding(messages, session_id):
            raise RuntimeError("injected L3 failure")

        def recording(self, messages, session_id="", rowid_map=None,
                      provenance_missing=False):
            seen["messages"] = messages
            seen["provenance_missing"] = provenance_missing

        monkeypatch.setattr(_sync.WriteQueue, "enqueue", recording)
        pkg._ENQUEUE_SUPPORTS_PROVENANCE = None  # 强制重新探测
        try:
            provider._l3_writer.write = exploding
            provider.sync_turn("u", "a", session_id="s-flag")
        finally:
            pkg._ENQUEUE_SUPPORTS_PROVENANCE = None

        assert seen.get("messages")
        assert seen.get("provenance_missing") is True


# ---------------------------------------------------------------------------
# P1-6 / P1-7 / P1-8 / P1-9 / P2-1 / P2-2 — recall engine
# ---------------------------------------------------------------------------

class TestRecallEngineFixes:
    def test_l1_picks_up_external_change(self, config):
        """L1 缓存感知 mtime：外部改动立即可见，不必等 60s TTL。"""
        engine = RecallEngine(config)
        assert "Don't refactor old modules" in engine.get_l1()

        Path(config.l1_memory_path).write_text(
            "# Rules\n- EXTERNALLY EDITED CONTENT", encoding="utf-8",
        )
        assert "EXTERNALLY EDITED" in engine.get_l1()

    def test_l1_survives_missing_file(self, config, caplog):
        """stat 失败时安全降级（保留旧缓存，不抛异常）。"""
        engine = RecallEngine(config)
        first = engine.get_l1()
        Path(config.l1_memory_path).unlink()
        Path(config.l1_user_path).unlink()

        engine._l1_mtime_signature = lambda: (_ for _ in ()).throw(OSError("boom"))
        assert engine.get_l1() == first

    def test_l2_init_runs_once(self, config):
        """冷启动并发只初始化一次；失败后不再重试探测。"""
        engine = RecallEngine(config)
        calls = {"n": 0}
        original = engine._init_l2

        def counting():
            calls["n"] += 1
            original()

        engine._init_l2 = counting

        barrier = threading.Barrier(8)

        def racer():
            barrier.wait()
            engine._search_l2("anything")

        threads = [threading.Thread(target=racer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert calls["n"] == 1, f"_init_l2 跑了 {calls['n']} 次"
        assert engine._l2_init_attempted is True

        engine._search_l2("anything again")
        assert calls["n"] == 1, "初始化失败后仍在每次查询重试探测"

    def test_loaded_embed_fn_is_not_discarded(self, config):
        """_init_l2 不应把已加载的嵌入函数抹掉。"""
        engine = RecallEngine(config)
        sentinel = lambda t: [0.0] * 384  # noqa: E731
        engine._embed_fn = sentinel
        engine._init_l2()
        assert engine._embed_fn is sentinel

    def test_timeout_error_is_caught(self):
        """concurrent.futures.TimeoutError 必须被显式捕获（3.10 上不是内建别名）。"""
        import concurrent.futures as cf
        import inspect

        source = inspect.getsource(RecallEngine.parallel_recall)
        assert "FuturesTimeoutError" in source
        assert issubclass(cf.TimeoutError, (TimeoutError, cf.TimeoutError))

    def test_multi_term_cjk_query_matches(self, config):
        """CJK 多词查询：'天气 北京' 必须能召回，不要求两词相邻。"""
        _mk_l3(config)
        _insert_l3(config, "今天天气不错，北京的空气质量也还可以，适合出门跑步")

        engine = RecallEngine(config)
        results = engine._search_l3("天气 北京")
        assert results, "多词 CJK 查询返回 0 条（空格被拼成连续串）"
        assert any("天气" in r.content for r in results)

    def test_single_cjk_segment_matches(self, config):
        """单个 CJK 段内保持原样匹配（不去空格）。"""
        _mk_l3(config)
        _insert_l3(config, "Python 是最好的编程语言")

        engine = RecallEngine(config)
        results = engine._search_l3("编程语言")
        assert results and any("编程语言" in r.content for r in results)

    def test_cjk_mixed_query_uses_one_connection(self, config, monkeypatch):
        """混合查询只开一个连接（过去是每段一个连接 + 一次全表扫描）。"""
        _mk_l3(config)
        with _TrackingConnect(monkeypatch) as tracker:
            engine = RecallEngine(config)
            engine._search_l3("python 今天 天气 北京 部署")

        assert len(tracker.opened) <= 1, f"开了 {len(tracker.opened)} 个连接"
        assert not tracker.leaked(), "连接没有关闭"

    def test_search_l3_closes_connection_on_error(self, config, monkeypatch):
        _mk_l3(config)
        with _TrackingConnect(monkeypatch) as tracker:
            engine = RecallEngine(config)
            monkeypatch.setattr(
                engine, "_build_fts_query", lambda q: (_ for _ in ()).throw(RuntimeError("x"))
            )
            engine._search_l3("anything")

        assert tracker.opened, "测试前置条件失败：没有打开任何连接"
        assert not tracker.leaked(), "_search_l3 在异常路径泄漏连接"

    def test_search_l3_like_closes_connection_on_error(self, config, monkeypatch):
        _mk_l3(config)
        with _TrackingConnect(monkeypatch) as tracker:
            engine = RecallEngine(config)
            monkeypatch.setattr(
                engine, "_escape_like", lambda t: (_ for _ in ()).throw(RuntimeError("x"))
            )
            engine._search_l3_like("中文")

        assert tracker.opened, "测试前置条件失败：没有打开任何连接"
        assert not tracker.leaked(), "_search_l3_like 在异常路径泄漏连接"

    def test_search_l3_like_reuses_borrowed_connection(self, config, monkeypatch):
        """借用来的连接不应被被调用方关闭。"""
        _mk_l3(config)
        engine = RecallEngine(config)
        conn = engine._open_l3_readonly()
        assert conn is not None
        try:
            engine._search_l3_like("中文", conn=conn)
            conn.execute("SELECT 1")  # 仍然可用
        finally:
            conn.close()

    def test_segment_splitting(self, config):
        engine = RecallEngine(config)
        assert engine._split_like_segments("天气 北京") == ["天气", "北京"]
        assert engine._split_like_segments("天气北京") == ["天气北京"]
        assert engine._split_like_segments("python3.11发布") == ["python3.11", "发布"]
        assert engine._split_like_segments("") == []
        # 多片段时丢弃单字噪声
        assert "的" not in engine._split_like_segments("天气 的 北京")
        assert len(engine._split_like_segments("a b c d e f g h i j k l")) <= 8


# ---------------------------------------------------------------------------
# P2-3 — recall_status
# ---------------------------------------------------------------------------

class TestRecallStatusContract:
    def test_hit_rate_is_consistent(self, provider):
        """hits/misses 在锁内同源读取，命中率不会 >1。"""
        provider._cache_hits = 2
        provider._cache_misses = 1
        status = provider.recall_status()
        assert status.cache_hits == 2
        assert status.cache_misses == 1
        assert abs(status.cache_hit_rate - 2 / 3) < 1e-6

    def test_hit_rate_is_clamped(self, provider):
        provider._cache_hits = 5
        provider._cache_misses = 0
        assert provider.recall_status().cache_hit_rate <= 1.0

        provider._cache_hits = 0
        provider._cache_misses = 3
        assert provider.recall_status().cache_hit_rate >= 0.0

    def test_status_class_is_module_level(self):
        from plugin.memory_governed import _RecallStatus

        assert _RecallStatus is not None
        assert not hasattr(_RecallStatus, "__qualname__") or "." not in _RecallStatus.__qualname__

    def test_returns_none_when_idle(self, provider):
        provider._cache_hits = 0
        provider._cache_misses = 0
        provider._last_recall_count = 0
        assert provider.recall_status() is None


# ---------------------------------------------------------------------------
# T04 — EmbeddingService
# ---------------------------------------------------------------------------

class _FakeEmbedder:
    """Minimal Embedder stand-in (avoids torch / ONNX entirely)."""

    def __init__(self, dim: int = 768, model_name: str = "fake-model"):
        self.dim = dim
        self.model_name = model_name

    def encode(self, texts: list):
        rows = [[float(i + 1)] * self.dim for i, _ in enumerate(texts)]
        return type("R", (), {"tolist": lambda self: rows})()


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class TestEmbeddingService:
    def test_get_returns_singleton(self, config):
        a = EmbeddingService.get(config)
        b = EmbeddingService.get(config)
        assert a is b

    def test_unavailable_when_no_backend(self, config, monkeypatch):
        """默认部署形态（后端全部缺失）：优雅降级，绝不抛异常。"""
        monkeypatch.setattr("plugin.memory_governed._embedding.load_embedder",
                            lambda backend, model: None)
        service = EmbeddingService.get(config)

        assert service.available is False
        assert service.embed_one("hello") is None
        assert service.embed_batch(["a", "b"]) == [None, None]
        assert service.dim == config.vector.dim  # 配置值兜底
        assert service.last_error

    def test_backend_none_is_unavailable(self, config):
        config.vector.backend = "none"
        service = EmbeddingService.get(config)
        assert service.available is False
        assert service.embed_one("x") is None

    def test_local_backend_is_wrapped(self, config, monkeypatch):
        monkeypatch.setattr("plugin.memory_governed._embedding.load_embedder",
                            lambda backend, model: _FakeEmbedder(dim=512))
        service = EmbeddingService.get(config)

        assert service.available is True
        vector = service.embed_one("hello")
        assert vector is not None and len(vector) == 512
        # 维度不一致：用实际维度覆盖配置并告警
        assert service.dim == 512
        assert config.vector.dim == 512

    def test_embed_batch_aligns_with_input(self, config, monkeypatch):
        monkeypatch.setattr("plugin.memory_governed._embedding.load_embedder",
                            lambda backend, model: _FakeEmbedder(dim=4))
        service = EmbeddingService.get(config)

        out = service.embed_batch(["a", "b", "c"])
        assert len(out) == 3
        assert all(v is not None and len(v) == 4 for v in out)
        assert service.embed_batch([]) == []

    def test_api_backend_has_priority(self, config, monkeypatch):
        """配置了 API embedding 时它优先于本地后端。"""
        config.embedding.provider = "openai"
        config.embedding.base_url = "https://api.example.com/v1"
        config.embedding.model = "text-embedding-3-small"
        config.embedding.api_key_env = "GOVERNED_TEST_EMBED_KEY"
        monkeypatch.setenv("GOVERNED_TEST_EMBED_KEY", "sk-test")

        captured = {}

        def fake_post(url, json=None, headers=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["timeout"] = timeout
            texts = json["input"]
            texts = texts if isinstance(texts, list) else [texts]
            return _FakeResponse({
                "data": [
                    {"index": i, "embedding": [0.25] * 1024}
                    for i in range(len(texts))
                ]
            })

        monkeypatch.setattr("httpx.post", fake_post)
        service = EmbeddingService.get(config)

        assert service.available is True
        assert service.backend_name.startswith("api:")
        assert captured["url"] == "https://api.example.com/v1/embeddings"
        assert captured["timeout"] == 10
        assert len(service.embed_one("hello")) == 1024
        assert service.dim == 1024

    def test_api_backend_matching_dim_does_not_misreport(self, config, monkeypatch):
        """API 场景 embedding.dimensions 与实际维度一致时，不误报 mismatch。"""
        config.embedding.provider = "openai"
        config.embedding.base_url = "https://api.example.com/v1"
        config.embedding.model = "text-embedding-3-small"
        config.embedding.api_key_env = "GOVERNED_TEST_EMBED_KEY3"
        config.embedding.dimensions = 1024
        monkeypatch.setenv("GOVERNED_TEST_EMBED_KEY3", "sk-test")

        def fake_post(url, json=None, headers=None, timeout=None):
            texts = json["input"]
            texts = texts if isinstance(texts, list) else [texts]
            return _FakeResponse({
                "data": [
                    {"index": i, "embedding": [0.25] * 1024}
                    for i in range(len(texts))
                ]
            })

        monkeypatch.setattr("httpx.post", fake_post)
        service = EmbeddingService.get(config)

        assert service.dim == 1024
        # 维度一致：不污染本地模型配置字段，也不改动已正确的 dimensions
        assert config.vector.dim == 512
        assert config.embedding.dimensions == 1024

    def test_api_backend_dim_mismatch_updates_dimensions(self, config, monkeypatch):
        """API 实际维度与配置不一致时，覆盖 embedding.dimensions（而非 vector.dim）。"""
        config.embedding.provider = "openai"
        config.embedding.base_url = "https://api.example.com/v1"
        config.embedding.model = "m"
        config.embedding.api_key_env = "GOVERNED_TEST_EMBED_KEY4"
        config.embedding.dimensions = 1024
        monkeypatch.setenv("GOVERNED_TEST_EMBED_KEY4", "sk-test")

        def fake_post(url, json=None, headers=None, timeout=None):
            texts = json["input"]
            texts = texts if isinstance(texts, list) else [texts]
            return _FakeResponse({
                "data": [
                    {"index": i, "embedding": [0.1] * 768}
                    for i in range(len(texts))
                ]
            })

        monkeypatch.setattr("httpx.post", fake_post)
        service = EmbeddingService.get(config)

        assert service.dim == 768
        assert config.embedding.dimensions == 768  # API 场景覆盖 dimensions
        assert config.vector.dim == 512  # 本地 vector.dim 不动

    def test_api_failure_is_negative_cached(self, config, monkeypatch):
        """API 探测失败后不再每次重试。"""
        config.embedding.provider = "openai"
        config.embedding.base_url = "https://api.example.com/v1"
        config.embedding.model = "m"
        config.embedding.api_key_env = "GOVERNED_TEST_EMBED_KEY2"
        monkeypatch.setenv("GOVERNED_TEST_EMBED_KEY2", "sk-test")

        calls = {"n": 0}

        def failing_post(*args, **kwargs):
            calls["n"] += 1
            raise RuntimeError("network down")

        monkeypatch.setattr("httpx.post", failing_post)
        service = EmbeddingService.get(config)

        assert service.available is False
        assert "network down" in service.last_error
        for _ in range(5):
            service.embed_one("x")
        assert calls["n"] == 1, "探测失败后仍在每次调用重试"

    def test_config_change_reprobes(self, config, monkeypatch):
        monkeypatch.setattr("plugin.memory_governed._embedding.load_embedder",
                            lambda backend, model: None)
        first = EmbeddingService.get(config)
        assert first.available is False

        monkeypatch.setattr("plugin.memory_governed._embedding.load_embedder",
                            lambda backend, model: _FakeEmbedder(dim=8))
        config.vector.model = "another-model"
        second = EmbeddingService.get(config)

        assert second is not first
        assert second.available is True

    def test_load_embedder_graceful(self):
        """load_embedder 在可选依赖缺失时返回 None，而不是抛异常。"""
        from plugin.memory_governed._embedding import load_embedder

        result = load_embedder("auto", "all-MiniLM-L6-v2")
        assert result is None or hasattr(result, "dim")

    def test_recall_delegates_to_service(self, config, monkeypatch):
        """_recall._init_l2 委托单例，不再内联降级阶梯。"""
        monkeypatch.setattr("plugin.memory_governed._embedding.load_embedder",
                            lambda backend, model: _FakeEmbedder(dim=16))
        engine = RecallEngine(config)
        engine._init_l2()

        service = engine._embed_service
        assert service is not None
        # 绑定方法每次取属性都是新对象，比较底层函数
        assert engine._embed_fn is not None
        assert engine._embed_fn.__func__ is type(service).embed_one
        assert engine._embed_fn("hello") == service.embed_one("hello")
