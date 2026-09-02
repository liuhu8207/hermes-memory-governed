# -*- coding: utf-8 -*-
"""L2 向量召回「distance → score」换算的回归防线（P0）。

背景
----
曾经有个 P0：L2 向量召回的 score 换算写成了 ``1.0 - min(d, 1.0)``（假设返回
的是 [0, 1] 的余弦相似度），但 LanceDB 默认 metric 是 L2 欧氏距离 —— 384/512
维向量的典型距离是 1.0~1.5，``min(d, 1.0)`` 恒为 1.0 → score 恒为 0 →
``format_recall`` 的 ``MIN_SCORE`` 把 **100%** 的 L2 命中丢掉。用户侧症状是
"L2 记忆完全不召回"，而当时的测试全都直接构造 ``RecallResult(score=0.9)``，
绕开了真实换算路径，所以公式改回去测试依然全绿。

本文件的目标就是锁死这条路径：

1. ``distance_to_score`` / ``row_to_score`` 的**纯函数**断言（含越界 clamp）。
2. ``_recall._search_l2_vector`` 用 fake LanceDB 表跑通，断言真实调用链上的
   score 与 ``.metric("cosine")``。
3. ``_kb.KBIndex.search`` 用 fake 表跑通，断言两处实现不漂移。
4. 端到端语义断言：真实距离 → ``format_recall`` → ``[Fact]`` 必须出现。

所有用例都不 import 真实的 lancedb / fastembed / sentence-transformers ——
LanceDB 的返回值用 fake 对象构造，保证 bare interpreter 上也能跑（另见
``TestLancedbAbsent``：把 ``lancedb`` 从 ``sys.modules`` 里屏蔽后仍然全绿）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._embedding import EmbeddingService, _FastEmbedBackend
from plugin.memory_governed._kb import KBIndex
from plugin.memory_governed._recall import (
    DEFAULT_COSINE_DISTANCE,
    MIN_SCORE,
    RecallEngine,
    RecallResult,
    distance_to_score,
    row_to_score,
)


# ---------------------------------------------------------------------------
# Fake LanceDB：记录链式调用，返回预设的行
# ---------------------------------------------------------------------------


class _FakeQueryBuilder:
    """``table.search(vec).metric(...).limit(...).to_list()`` 的链式桩。"""

    def __init__(self, rows: List[Dict[str, Any]], calls: List[tuple]) -> None:
        self._rows = rows
        self._calls = calls

    def metric(self, name: str) -> "_FakeQueryBuilder":
        self._calls.append(("metric", name))
        return self

    def limit(self, n: int) -> "_FakeQueryBuilder":
        self._calls.append(("limit", n))
        return self

    def to_list(self) -> List[Dict[str, Any]]:
        self._calls.append(("to_list",))
        return list(self._rows)


class _FakeLanceTable:
    """LanceDB ``Table`` 的最小替身，只实现 ``search()``。"""

    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self._rows = list(rows)
        self.calls: List[tuple] = []

    def search(self, vector: Any) -> _FakeQueryBuilder:
        self.calls.append(("search", vector))
        return _FakeQueryBuilder(self._rows, self.calls)

    @property
    def metrics(self) -> List[str]:
        """本次查询显式声明过的 metric 名。"""
        return [call[1] for call in self.calls if call[0] == "metric" and len(call) > 1]


class _FakeEmbedService:
    """``EmbeddingService`` 的最小替身（``KBIndex.search`` 只用到两个成员）。"""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.available = True

    def embed_one(self, text: str) -> List[float]:
        return [0.1, 0.2, 0.3, 0.4]


def _make_engine(tmp_path: Path) -> RecallEngine:
    """构造一个不走真实 lancedb 的 RecallEngine。"""
    cfg = GovernedMemoryConfig()
    cfg.l1_memory_path = str(tmp_path / "MEMORY.md")
    cfg.l1_user_path = str(tmp_path / "USER.md")
    cfg.l2_db_path = str(tmp_path / "l2")
    cfg.l3_db_path = str(tmp_path / "l3" / "l3.db")
    cfg.l4_persona_path = str(tmp_path / "persona.md")
    return RecallEngine(cfg)


@pytest.fixture(autouse=True)
def _reset_embedding_singleton():
    """EmbeddingService 是进程内单例 —— 每个用例前后重置，避免互相污染。"""
    EmbeddingService.reset()
    yield
    EmbeddingService.reset()


# ---------------------------------------------------------------------------
# 1) 纯函数：distance → score
# ---------------------------------------------------------------------------


class TestDistanceToScore:
    """余弦距离 → 相关性分数的映射表（P0 公式的直接断言）。"""

    @pytest.mark.parametrize(
        "distance, expected",
        [
            (0.0, 1.0),   # 完全相同
            (1.0, 0.5),   # 正交
            (2.0, 0.0),   # 完全相反（余弦距离上界）
            (0.4, 0.8),   # 一般情况，验证 1 - d/2
            (3.0, 0.0),   # 越界：clamp 下界，不能为负
            (-1.0, 1.0),  # 越界：clamp 上界，不能 > 1
        ],
    )
    def test_mapping_table(self, distance: float, expected: float):
        assert distance_to_score(distance) == pytest.approx(expected)

    def test_result_is_always_within_unit_interval(self):
        """任何输入都必须落在 [0, 1]，否则 MIN_SCORE 过滤的语义就不成立。"""
        for distance in (-10.0, -0.001, 0.0, 0.5, 1.0, 1.5, 2.0, 2.0001, 100.0):
            score = distance_to_score(distance)
            assert 0.0 <= score <= 1.0, f"distance={distance} → score={score}"

    def test_monotonically_decreasing(self):
        """距离越大越不相关 —— 保序性是排序正确的前提。"""
        distances = [0.0, 0.25, 0.5, 1.0, 1.25, 1.5, 2.0]
        scores = [distance_to_score(d) for d in distances]
        assert scores == sorted(scores, reverse=True)

    def test_int_input_is_accepted(self):
        """LanceDB 可能返回 numpy / int 标量。"""
        assert distance_to_score(1) == pytest.approx(0.5)
        assert distance_to_score(2) == pytest.approx(0.0)


class TestRowToScore:
    """``_distance`` 字段缺失 / 脏数据时的兜底行为。"""

    def test_missing_field_falls_back_to_orthogonal(self):
        """字段缺失 → 默认距离 1.0 → score 0.5（既不算命中也不丢弃）。"""
        assert row_to_score({"content": "没有 _distance 的行"}) == pytest.approx(0.5)

    def test_missing_field_uses_declared_default(self):
        assert DEFAULT_COSINE_DISTANCE == pytest.approx(1.0)
        assert distance_to_score(DEFAULT_COSINE_DISTANCE) == pytest.approx(0.5)

    def test_none_distance_falls_back(self):
        assert row_to_score({"_distance": None}) == pytest.approx(0.5)

    def test_nan_distance_falls_back(self):
        assert row_to_score({"_distance": float("nan")}) == pytest.approx(0.5)

    def test_non_numeric_distance_falls_back(self):
        assert row_to_score({"_distance": "not-a-number"}) == pytest.approx(0.5)

    def test_non_mapping_row_falls_back(self):
        """行不是 Mapping（比如 pa.StructScalar）时也要给出兜底而不是抛异常。"""
        assert row_to_score(object()) == pytest.approx(0.5)
        assert row_to_score(None) == pytest.approx(0.5)

    def test_row_get_raising_falls_back(self):
        class _Exploding:
            def get(self, key, default=None):
                raise RuntimeError("boom")

        assert row_to_score(_Exploding()) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2) _recall._search_l2_vector：真实换算路径（fake LanceDB 表）
# ---------------------------------------------------------------------------


class TestSearchL2Vector:
    """L2 向量召回必须走 cosine metric + d/2 映射。"""

    def test_scores_follow_distance_mapping(self, tmp_path):
        engine = _make_engine(tmp_path)
        table = _FakeLanceTable([
            {"content": "完全相同", "_distance": 0.0, "category": "fact", "source_rowid": 1},
            {"content": "正交", "_distance": 1.0, "category": "fact", "source_rowid": 2},
            {"content": "反向", "_distance": 2.0, "category": "fact", "source_rowid": 3},
            {"content": "一般", "_distance": 0.4, "category": "fact", "source_rowid": 4},
        ])
        engine._l2_store = table
        engine._embed_fn = lambda text: [0.1, 0.2, 0.3, 0.4]

        results = engine._search_l2_vector("查询")

        assert [r.content for r in results] == ["完全相同", "正交", "反向", "一般"]
        assert [r.score for r in results] == pytest.approx([1.0, 0.5, 0.0, 0.8])
        assert all(r.layer == "l2" and r.source == "lance" for r in results)

    def test_cosine_metric_is_requested_explicitly(self, tmp_path):
        """不显式指定 metric 时 LanceDB 用 L2 欧氏距离 —— 回归根因之一。"""
        engine = _make_engine(tmp_path)
        table = _FakeLanceTable([{"content": "x", "_distance": 1.0}])
        engine._l2_store = table
        engine._embed_fn = lambda text: [0.1]

        engine._search_l2_vector("查询")

        assert table.metrics == ["cosine"]

    def test_typical_distance_survives_min_score(self, tmp_path):
        """真实场景的典型距离 1.0~1.5 必须换算成 > MIN_SCORE 的分数。

        这是 P0 的核心：旧公式 ``1 - min(d, 1)`` 在这个区间恒等于 0。
        """
        engine = _make_engine(tmp_path)
        table = _FakeLanceTable([
            {"content": "数据库连接配置", "_distance": 1.4},
            {"content": "部署服务器", "_distance": 1.2},
        ])
        engine._l2_store = table
        engine._embed_fn = lambda text: [0.1]

        results = engine._search_l2_vector("数据库")

        assert [r.score for r in results] == pytest.approx([0.3, 0.4])
        assert all(r.score >= MIN_SCORE for r in results)

    def test_missing_distance_defaults_to_half(self, tmp_path):
        engine = _make_engine(tmp_path)
        table = _FakeLanceTable([{"content": "缺字段的行"}])
        engine._l2_store = table
        engine._embed_fn = lambda text: [0.1]

        results = engine._search_l2_vector("查询")

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.5)

    def test_search_failure_returns_empty_without_raising(self, tmp_path):
        engine = _make_engine(tmp_path)
        engine._embed_fn = lambda text: [0.1]

        class _Broken:
            def search(self, vector):
                raise RuntimeError("lancedb exploded")

        engine._l2_store = _Broken()
        assert engine._search_l2_vector("查询") == []


# ---------------------------------------------------------------------------
# 3) _kb.KBIndex.search：与 _recall 共用同一换算，防止两份实现漂移
# ---------------------------------------------------------------------------


class TestKbIndexScoreMapping:
    """``_kb`` 的语义检索必须复用 ``_recall`` 的换算。"""

    def _make_index(self, tmp_path: Path, rows: Sequence[Dict[str, Any]]) -> KBIndex:
        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "memory" / "l2")
        cfg.vector.backend = "none"  # 不走真实 embedding 探测
        index = KBIndex.__new__(KBIndex)  # 跳过 _init()（它会 import lancedb）
        index._config = cfg
        index._db_path = tmp_path / "memory" / "kb_index"
        index._service = _FakeEmbedService()
        index._store = _FakeLanceTable(list(rows))
        index._available = True
        index._last_error = ""
        return index

    def test_scores_match_recall_mapping(self, tmp_path):
        index = self._make_index(tmp_path, [
            {"path": "a.md", "text": "完全相同", "_distance": 0.0},
            {"path": "b.md", "text": "正交", "_distance": 1.0},
            {"path": "c.md", "text": "反向", "_distance": 2.0},
            {"path": "d.md", "text": "一般", "_distance": 0.4},
        ])

        out = index.search("查询", k=10)

        assert [r["score"] for r in out] == pytest.approx([1.0, 0.5, 0.0, 0.8])
        assert all(r["kind"] == "semantic" for r in out)

    def test_kb_uses_cosine_metric_too(self, tmp_path):
        index = self._make_index(tmp_path, [{"path": "a.md", "text": "t", "_distance": 1.0}])
        index.search("查询", k=1)
        assert index._store.metrics == ["cosine"]

    def test_missing_distance_defaults_to_half(self, tmp_path):
        index = self._make_index(tmp_path, [{"path": "a.md", "text": "t"}])
        assert index.search("查询", k=1)[0]["score"] == pytest.approx(0.5)

    def test_no_drift_between_recall_and_kb(self, tmp_path):
        """两条路径对同一行必须给出完全相同的分数（不是"接近"，是相等）。"""
        rows = [{"content": "x", "path": "x.md", "text": "x", "_distance": 0.7}]

        engine = _make_engine(tmp_path)
        engine._l2_store = _FakeLanceTable(rows)
        engine._embed_fn = lambda text: [0.1]
        recall_score = engine._search_l2_vector("q")[0].score

        kb_score = self._make_index(tmp_path, rows).search("q", k=1)[0]["score"]

        assert recall_score == kb_score == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 4) 端到端：MIN_SCORE 过滤（P0 的用户可见症状）
# ---------------------------------------------------------------------------


class TestFormatRecallMinScore:
    """``format_recall`` 必须让合格条目出现、让低分条目消失。"""

    def test_low_score_is_dropped_and_high_scores_kept(self, tmp_path):
        engine = _make_engine(tmp_path)
        results = [
            RecallResult(layer="l2", content="低分事实", score=0.05),
            RecallResult(layer="l2", content="中分事实", score=0.5),
            RecallResult(layer="l3", content="高分历史", score=0.9),
        ]

        text = engine.format_recall(results, "L1 规则", l1_budget=800, l23_budget=1200)

        assert "低分事实" not in text
        assert "中分事实" in text
        assert "高分历史" in text

    def test_threshold_boundary_is_inclusive(self, tmp_path):
        """``>= MIN_SCORE``：正好等于阈值要保留。"""
        engine = _make_engine(tmp_path)
        results = [
            RecallResult(layer="l2", content="刚好达标", score=MIN_SCORE),
            RecallResult(layer="l2", content="差一点点", score=MIN_SCORE - 1e-9),
        ]
        text = engine.format_recall(results, "", l1_budget=800, l23_budget=1200)
        assert "刚好达标" in text
        assert "差一点点" not in text

    def test_l2_items_are_tagged_as_fact(self, tmp_path):
        engine = _make_engine(tmp_path)
        results = [RecallResult(layer="l2", content="数据库连接配置", score=0.75)]
        text = engine.format_recall(results, "", l1_budget=800, l23_budget=1200)
        assert "[Fact] 数据库连接配置" in text

    def test_p0_regression_l2_facts_reach_the_prompt(self, tmp_path):
        """端到端复现 P0：真实余弦距离 → score → ``[Fact]`` 必须进入注入上下文。

        这条用例走完整的 ``_search_l2_vector`` → ``format_recall`` 链路。把
        ``distance_to_score`` 换成旧的 ``1.0 - min(d, 1.0)`` 后，1.4 会变成
        score 0.0，``[Fact]`` 随之消失 —— 用例即变红。
        """
        engine = _make_engine(tmp_path)
        engine._l2_store = _FakeLanceTable([
            {"content": "数据库用 PostgreSQL 15，端口 5433", "_distance": 1.4},
            {"content": "前端构建工具换成 Vite", "_distance": 1.2},
        ])
        engine._embed_fn = lambda text: [0.1]

        results = engine._search_l2_vector("数据库连接配置")
        text = engine.format_recall(results, "L1：不要动 database.py", 800, 1200)

        assert "[Fact]" in text, (
            f"L2 内容被 MIN_SCORE 全丢了 —— P0 回归！scores={[r.score for r in results]}"
        )
        assert "PostgreSQL" in text
        assert max(r.score for r in results) > MIN_SCORE


# ---------------------------------------------------------------------------
# 5) 裸模型名补前缀（_FastEmbedBackend）
# ---------------------------------------------------------------------------


class TestFastEmbedModelCandidates:
    """fastembed 的裸模型名必须自动补 ``sentence-transformers/`` 前缀。"""

    @pytest.mark.parametrize(
        "model_name, expected",
        [
            ("BAAI/bge-small-zh-v1.5", ["BAAI/bge-small-zh-v1.5"]),
            (
                "all-MiniLM-L6-v2",
                ["all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2"],
            ),
            ("", []),
            ("   ", []),
        ],
    )
    def test_candidate_expansion(self, model_name: str, expected: List[str]):
        assert _FastEmbedBackend._candidates(model_name) == expected

    def test_bare_name_loads_prefixed_model(self, monkeypatch):
        """裸名：第一个候选（构造即失败）被跳过，补前缀后成功。"""
        constructed: List[str] = []

        class _FakeTextEmbedding:
            def __init__(self, model_name: str = "", **kwargs):
                constructed.append(model_name)
                if model_name != "sentence-transformers/all-MiniLM-L6-v2":
                    raise ValueError(f"Model {model_name} is not supported in TextEmbedding.")

            def embed(self, texts, **kwargs):
                # fastembed 的 embed() 是生成器，逐条产出向量
                for _ in texts:
                    yield [0.0] * 384

        monkeypatch.setitem(
            sys.modules, "fastembed", type(sys)("fastembed")
        )
        monkeypatch.setattr(
            sys.modules["fastembed"], "TextEmbedding", _FakeTextEmbedding, raising=False
        )

        backend = _FastEmbedBackend("all-MiniLM-L6-v2")

        assert backend.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert backend.dim == 384
        assert constructed == [
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
        ]

    def test_namespaced_name_is_not_rewritten(self, monkeypatch):
        """自带命名空间的模型名按原样使用（尊重用户显式指定）。"""
        constructed: List[str] = []

        class _FakeTextEmbedding:
            def __init__(self, model_name: str = "", **kwargs):
                constructed.append(model_name)
                if "/" not in model_name:
                    raise ValueError("unsupported")

            def embed(self, texts, **kwargs):
                # fastembed 的 embed() 是生成器，逐条产出向量
                for _ in texts:
                    yield [0.0] * 512

        monkeypatch.setitem(sys.modules, "fastembed", type(sys)("fastembed"))
        monkeypatch.setattr(
            sys.modules["fastembed"], "TextEmbedding", _FakeTextEmbedding, raising=False
        )

        backend = _FastEmbedBackend("BAAI/bge-small-zh-v1.5")

        assert backend.model_name == "BAAI/bge-small-zh-v1.5"
        assert backend.dim == 512
        assert constructed == ["BAAI/bge-small-zh-v1.5"]

    def test_probe_failure_does_not_escape_the_loop(self, monkeypatch):
        """问题 B：embed() 阶段的失败不能逃逸出去干掉整个 fastembed 后端。

        旧实现把 probe 留在 try 外面，构造成功但 embed 失败时异常会冒泡到
        ``load_embedder``，被记成"fastembed 后端加载失败"直接降级。
        """
        constructed: List[str] = []

        class _FakeTextEmbedding:
            def __init__(self, model_name: str = "", **kwargs):
                constructed.append(model_name)
                # 构造函数不校验模型名 —— 模拟 fastembed 的懒加载

            def embed(self, texts, **kwargs):
                # 裸名在真正加载时失败；补前缀后成功
                if not constructed[-1].startswith("sentence-transformers/"):
                    raise RuntimeError("model download failed")
                # fastembed 的 embed() 是生成器，逐条产出向量
                for _ in texts:
                    yield [0.0] * 384

        monkeypatch.setitem(sys.modules, "fastembed", type(sys)("fastembed"))
        monkeypatch.setattr(
            sys.modules["fastembed"], "TextEmbedding", _FakeTextEmbedding, raising=False
        )

        backend = _FastEmbedBackend("all-MiniLM-L6-v2")

        assert backend.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert backend.dim == 384

    def test_all_candidates_fail_falls_back_to_default(self, monkeypatch):
        """全部候选失败才 fallback 到 fastembed 默认模型。"""
        constructed: List[str] = []

        class _FakeTextEmbedding:
            def __init__(self, model_name: str = "", **kwargs):
                constructed.append(model_name)
                if model_name:
                    raise ValueError("unknown model")

            def embed(self, texts, **kwargs):
                # fastembed 的 embed() 是生成器，逐条产出向量
                for _ in texts:
                    yield [0.0] * 384

        monkeypatch.setitem(sys.modules, "fastembed", type(sys)("fastembed"))
        monkeypatch.setattr(
            sys.modules["fastembed"], "TextEmbedding", _FakeTextEmbedding, raising=False
        )

        backend = _FastEmbedBackend("definitely-not-a-model")

        assert backend.model_name == "fastembed-default"
        assert backend.dim == 384
        assert constructed == [
            "definitely-not-a-model",
            "sentence-transformers/definitely-not-a-model",
            "",  # 最后兜底：TextEmbedding() 无参
        ]


# ---------------------------------------------------------------------------
# 6) lancedb 缺失时仍然优雅降级
# ---------------------------------------------------------------------------


class TestLancedbAbsent:
    """新测试不能硬依赖 lancedb：屏蔽 import 后这组用例仍必须全绿。"""

    def test_pure_mapping_works_without_lancedb(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "lancedb", None)
        assert distance_to_score(0.4) == pytest.approx(0.8)
        assert row_to_score({"_distance": 1.0}) == pytest.approx(0.5)

    def test_search_l2_vector_works_without_lancedb(self, monkeypatch, tmp_path):
        """fake 表 + fake embed：整条换算链路不需要真实 lancedb。"""
        monkeypatch.setitem(sys.modules, "lancedb", None)

        engine = _make_engine(tmp_path)
        engine._l2_store = _FakeLanceTable([{"content": "内容", "_distance": 0.6}])
        engine._embed_fn = lambda text: [0.1]

        results = engine._search_l2_vector("查询")

        assert [r.score for r in results] == pytest.approx([0.7])

    def test_kb_search_works_without_lancedb(self, monkeypatch, tmp_path):
        monkeypatch.setitem(sys.modules, "lancedb", None)

        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "memory" / "l2")
        cfg.vector.backend = "none"
        index = KBIndex.__new__(KBIndex)
        index._config = cfg
        index._db_path = tmp_path / "memory" / "kb_index"
        index._service = _FakeEmbedService()
        index._store = _FakeLanceTable([{"path": "a.md", "text": "内容", "_distance": 1.2}])
        index._available = True
        index._last_error = ""

        assert index.search("查询", k=5)[0]["score"] == pytest.approx(0.4)

    def test_real_kb_init_degrades_gracefully_without_lancedb(self, monkeypatch, tmp_path):
        """真实的 ``KBIndex.__init__`` 在 lancedb 缺失时 unavailable 且不抛异常。"""
        monkeypatch.setitem(sys.modules, "lancedb", None)
        monkeypatch.setitem(sys.modules, "pyarrow", None)

        cfg = GovernedMemoryConfig()
        cfg.l2_db_path = str(tmp_path / "memory" / "l2")
        cfg.vector.backend = "fastembed"  # 让 EmbeddingService 真的去探测
        monkeypatch.setattr(
            "plugin.memory_governed._embedding.load_embedder",
            lambda backend, model: _FakeEmbedService(),
        )

        index = KBIndex(cfg, tmp_path / "memory" / "kb_index")

        assert index.available is False
        assert "lancedb missing" in index.last_error
