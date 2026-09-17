# -*- coding: utf-8 -*-
"""索引可用性懒重探 / 自愈的回归测试（P2，2026-09-17）。

**缺陷**：``KBIndex._available`` 在 ``__init__`` 里一次判定后**永不重探**。
索引初始化那一刻的瞬时失败（网络抖动 / API 限流 / key 临时不可用）会让
**这个进程整程**丢掉语义通道、且悄无声息 —— 与刚修的「索引陈旧不自愈」
同族，都是「一次性判定 + 无自愈 + 不报错」。

**修复**：当前判定为「不可用」时，:attr:`KBIndex.available` 按
``REPROBE_TTL_SECONDS`` 懒重探（可用状态下零成本），恢复即自愈并留下
``reprobe_count`` / ``log_degraded`` 痕迹。特别地，``EmbeddingService`` 是
按配置签名缓存的单例 —— 不 ``reset()`` 丢弃它，重探只会拿到同一个不可用
实例、永远空转（这是本修复最容易漏的一点，下面有专门用例）。

语义边界：``available=False``（后端真不可用）与 ``probe="unsupported"`` /
``fresh=None``（可用性未知）是两个概念；重探失败只维持 ``available=False``，
**不改** ``fresh``。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugin.memory_governed import _diag
from plugin.memory_governed import _kb
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KBIndex, KnowledgeBase


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------

class _ScriptedIndex(KBIndex):
    """把 ``_init`` 换成可控脚本：按调用次序决定每次成功/失败。

    不碰文件系统 —— 用于把「重探时机」从「后端行为」里隔离出来单独验证。
    """

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.init_calls = 0
        self.reprobe_flags = []
        super().__init__(GovernedMemoryConfig(), Path("/nonexistent/kb_index"))

    def _init(self, *, reprobe: bool = False) -> None:
        self.reprobe_flags.append(reprobe)
        ok = bool(self.outcomes[min(self.init_calls, len(self.outcomes) - 1)])
        self.init_calls += 1
        self._available = ok
        self._last_error = "" if ok else "transient: 429 rate limited"
        self._store = object() if ok else None
        self._service = object() if ok else None


class _EmbStub:
    """最小嵌入替身：可用时返回固定向量（任意查询与任意文本距离为 0）。"""

    def __init__(self, available: bool, dim: int = 4):
        self.available = available
        self.dim = dim
        self.last_error = "" if available else "transient: 429 rate limited"

    def embed_one(self, _text):
        return [1.0, 0.0, 0.0, 0.0] if self.available else None


class _FlakyEmbeddingService:
    """模拟 ``EmbeddingService`` 单例：后端恢复后**缓存实例不会自动变好**。

    ``arm(True)`` 只改后端真实状态，**不动**已经缓存下来的不可用实例 ——
    这正是线上场景：单例在启动时探测失败并被缓存，之后端恢复了，但 ``get()``
    仍然把那个坏实例还回来。只有 ``reset()`` 丢弃它才会重新探测。
    """

    backend_ok: bool = False
    singleton = None

    @classmethod
    def get(cls, _config):
        if cls.singleton is None:
            cls.singleton = _EmbStub(available=cls.backend_ok)
        return cls.singleton

    @classmethod
    def reset(cls):
        cls.singleton = None

    @classmethod
    def arm(cls, backend_ok: bool) -> None:
        """后端真实状态变化（缓存实例刻意不动）。"""
        cls.backend_ok = backend_ok

    @classmethod
    def hard_reset(cls) -> None:
        cls.backend_ok = False
        cls.singleton = None


@pytest.fixture
def flaky_embedding(monkeypatch):
    _FlakyEmbeddingService.hard_reset()
    monkeypatch.setattr(_kb, "EmbeddingService", _FlakyEmbeddingService)
    yield _FlakyEmbeddingService
    _FlakyEmbeddingService.hard_reset()


def _diag_count(key: str) -> int:
    return int(_diag.stats()["degraded"].get(key, 0))


# ---------------------------------------------------------------------------
# 1) 重探时机：TTL 内不重探 / TTL 过期才重探 / 可用时零成本
# ---------------------------------------------------------------------------

class TestReprobeTiming:
    def test_within_ttl_does_not_reprobe(self):
        idx = _ScriptedIndex([False, True])  # 首次失败
        assert idx.init_calls == 1 and idx.available is False
        for _ in range(5):
            assert idx.available is False
        assert idx.init_calls == 1, "TTL 内不得重探（否则每次检索都多一次 connect）"
        assert idx.reprobe_count == 0

    def test_reprobe_after_ttl_expires(self):
        idx = _ScriptedIndex([False, True])
        idx._next_reprobe_at = 0.0  # 让 TTL 立即过期
        assert idx.available is True
        assert idx.init_calls == 2
        assert idx.reprobe_flags == [False, True], "重探必须走 reprobe=True 分支"
        assert idx.reprobe_count == 1

    def test_no_reprobe_when_already_available(self):
        """可用状态下 :attr:`available` 零成本：一次属性判断，不碰后端。"""
        idx = _ScriptedIndex([True])
        assert idx.available is True
        for _ in range(10):
            assert idx.available is True
        assert idx.init_calls == 1 and idx.reprobe_count == 0

    def test_each_ttl_window_probes_at_most_once(self):
        idx = _ScriptedIndex([False, False, True])
        # 窗口一：过期后探一次（仍失败）
        idx._next_reprobe_at = 0.0
        assert idx.available is False and idx.reprobe_count == 1
        # 同窗口内再读多次，不再探
        for _ in range(3):
            assert idx.available is False
        assert idx.reprobe_count == 1
        # 窗口二：再过期，探一次（成功）
        idx._next_reprobe_at = 0.0
        assert idx.available is True and idx.reprobe_count == 2


# ---------------------------------------------------------------------------
# 2) 自愈 + 可观测
# ---------------------------------------------------------------------------

class TestReprobeSelfHeal:
    def test_recovery_emits_a_visible_event(self):
        before = _diag_count("kb_index::recovered")
        idx = _ScriptedIndex([False, True])
        idx._next_reprobe_at = 0.0
        assert idx.available is True
        assert _diag_count("kb_index::recovered") > before, (
            "恢复必须留下可观测痕迹 —— 静默自愈和静默降级一样糟"
        )

    def test_still_unavailable_reports_and_does_not_recover(self):
        before = _diag_count("kb_index::reprobe_still_unavailable")
        idx = _ScriptedIndex([False, False])
        idx._next_reprobe_at = 0.0
        assert idx.available is False
        assert idx.reprobe_count == 1
        assert idx.last_error == "transient: 429 rate limited"
        assert _diag_count("kb_index::reprobe_still_unavailable") > before

    def test_reprobe_failure_does_not_pollute_fresh_semantics(self, tmp_path):
        """重探失败 = 「仍然不可用」，不是「陈旧」；``fresh`` 必须是 None（未知）。"""
        wiki = tmp_path / "wiki"
        (wiki / "notes").mkdir(parents=True)
        (wiki / "notes" / "a.md").write_text("---\ntitle: A\n---\nbody",
                                             encoding="utf-8")
        cfg = GovernedMemoryConfig()
        cfg.wiki_dir = str(wiki)
        kb = KnowledgeBase(cfg)

        idx = _ScriptedIndex([False, False])
        idx._next_reprobe_at = 0.0  # 强制重探一次（仍失败）
        state = kb._probe_state(idx)
        assert state["available"] is False
        assert state["probe"] == "unavailable"
        assert state["fresh"] is None, (
            "不可用时新鲜度是**未知**（None），绝不能写成 False 冒充「陈旧」"
        )
        assert state["reprobe_count"] == 1, "重探次数要能透出到状态里"


# ---------------------------------------------------------------------------
# 3) 端到端：缓存单例被 reset 后真正重探，语义通道回来
# ---------------------------------------------------------------------------

def _make_kb(tmp_path: Path) -> KnowledgeBase:
    wiki = tmp_path / "wiki"
    (wiki / "notes").mkdir(parents=True)
    (wiki / "notes" / "Alpha.md").write_text(
        "---\ntitle: Alpha\n---\nalpha body", encoding="utf-8")
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(wiki)
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    cfg.vector.dim = 4
    kb = KnowledgeBase(cfg)
    kb.ensure(sync=False)
    return kb


class TestEndToEndRecovery:
    def test_semantic_channel_returns_after_transient_failure(
            self, tmp_path, flaky_embedding):
        kb = _make_kb(tmp_path)
        idx = kb._index_get()

        # 启动时后端不可用 → 语义通道被关（关键词通道仍工作）
        assert idx.available is False
        assert idx.reprobe_count == 0
        first = kb.search("Alpha", top_k=5)
        assert first, "关键词通道必须不受可用性影响"
        assert all(r["kind"] == "keyword" for r in first), (
            "后端不可用时不该有语义 / 混合命中"
        )
        assert idx.reprobe_count == 0, "TTL 内检索不应触发重探"

        # 后端恢复（缓存单例仍不可用 → 必须靠 reset 才能真正重探）
        flaky_embedding.arm(True)
        idx._next_reprobe_at = 0.0

        second = kb.search("Alpha", top_k=5)
        assert idx.available is True, "瞬时故障必须能自愈，而不是锁死整进程"
        assert idx.reprobe_count == 1
        assert any(r.get("semantic_score", 0.0) > 0.0 for r in second), (
            "可用性恢复后语义通道必须回来"
        )
        snap = second[0]["index_status"]
        assert snap["available"] is True
        assert snap["reprobe_count"] == 1, "自愈这件事要在检索结果里可见"

    def test_recovery_is_idempotent_after_healing(self, tmp_path, flaky_embedding):
        kb = _make_kb(tmp_path)
        idx = kb._index_get()
        flaky_embedding.arm(True)
        idx._next_reprobe_at = 0.0
        assert idx.available is True and idx.reprobe_count == 1
        for _ in range(3):
            assert idx.available is True
        assert idx.reprobe_count == 1, "已可用后不得再重探"

    def test_stats_reports_reprobe_count(self, tmp_path, flaky_embedding):
        kb = _make_kb(tmp_path)
        idx = kb._index_get()
        assert kb.stats()["index_reprobe_count"] == 0
        flaky_embedding.arm(True)
        idx._next_reprobe_at = 0.0
        kb._index_get().available  # 触发自愈
        assert kb.stats()["index_reprobe_count"] == 1
        assert kb.stats()["index_available"] is True


# ---------------------------------------------------------------------------
# 4) 向后兼容 / 替身防御
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_index_without_reprobe_count_is_safe(self, tmp_path):
        """没有 ``reprobe_count`` 的替身索引不能让探测抛异常。"""
        wiki = tmp_path / "wiki"
        (wiki / "notes").mkdir(parents=True)

        class _Bare:
            available = True
            last_error = ""

            def path_mtimes(self):
                return {}

        cfg = GovernedMemoryConfig()
        cfg.wiki_dir = str(wiki)
        kb = KnowledgeBase(cfg)
        state = kb._probe_state(_Bare())
        assert state["available"] is True
        assert state["fresh"] is True
        assert state["reprobe_count"] == 0

    def test_ttl_is_nonzero(self):
        """TTL=0 会让每次检索都重探 —— 用一个下限把它钉死。"""
        assert KBIndex.REPROBE_TTL_SECONDS >= 5.0
