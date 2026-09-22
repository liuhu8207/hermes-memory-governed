# -*- coding: utf-8 -*-
"""KB 召回通道 / 分数融合 / 增量索引回归测试（2026-09-16）。

修的三件事：

1. **分数混排**：关键词原始分（0~12）和语义分（0~1）此前被直接 ``max()``
   合并，关键词永远碾压语义 —— 实测 ``hermes 记忆系统`` 返回 1.8/1.6/1.4
   全是关键词分，向量这一路等于白建。归一化后加权融合。
2. **KB 不参与对话召回**：25 篇笔记躺在 vault 里，召回链路完全不碰 KB。
   现在有一条 layer="kb" 的提示通道（只给标题+路径，见 _search_kb）。
3. **无增量索引**：索引停在 9-01，而 ``notes/共享记忆系统开通.md`` 是
   9-04 由 DeepSeek Harness 直接写入 vault 的 —— 外部写入不经过
   ``_kb.add``，只有 mtime 比对能发现。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import (
    _KB_KW_WEIGHT,
    _KB_SEM_WEIGHT,
    KnowledgeBase,
    _Note,
)
from plugin.memory_governed._recall import RecallEngine, RecallResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kb(tmp_path: Path, notes: dict[str, str]) -> KnowledgeBase:
    """建一个真实 vault（写文件）+ 关掉 embedding 的 KB。"""
    wiki = tmp_path / "wiki"
    for rel, body in notes.items():
        p = wiki / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(wiki)
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    kb = KnowledgeBase(cfg)
    kb.ensure(sync=False)
    return kb


class _FakeIndex:
    """记录 upsert/delete 的假索引，用于验证增量同步。"""

    available = True
    last_error = ""

    def __init__(self):
        self.rows: dict[str, tuple] = {}

    def path_mtimes(self):
        return {p: mt for p, (_t, mt) in self.rows.items()}

    def upsert(self, path, text, mtime=0.0):
        self.rows[path] = (text, float(mtime))

    def delete(self, path):
        self.rows.pop(path, None)

    def drop(self):
        self.rows.clear()

    def search(self, query, k):
        return []


class _FakeSemanticIndex(_FakeIndex):
    def __init__(self, scores: dict[str, float]):
        super().__init__()
        self._scores = scores

    def search(self, query, k):
        return [
            {"path": p, "text": f"text of {p}", "score": s, "kind": "semantic"}
            for p, s in list(self._scores.items())[:k]
        ]


# ---------------------------------------------------------------------------
# 1) 关键词分归一化
# ---------------------------------------------------------------------------

class TestKeywordNormalisation:
    def _note(self):
        return _Note(Path("n.md"), "SecretStore 同步", {"tags": ["vault"]}, "正文内容")

    def test_title_hit_saturates_at_one(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        s = kb._keyword_score_norm("SecretStore 同步", self._note())
        assert s == pytest.approx(1.0)

    def test_body_only_hit_is_much_lower_than_title_hit(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        note = _Note(Path("n.md"), "无关标题", {}, "这里提到了 secretstore 同步这件事")
        body = kb._keyword_score_norm("secretstore 同步", note)
        title = kb._keyword_score_norm("SecretStore 同步", self._note())
        assert 0.0 < body < title
        assert body <= 0.5

    def test_norm_is_always_within_unit_interval(self):
        kb = KnowledgeBase.__new__(KnowledgeBase)
        n = self._note()
        for q in ("", "SecretStore", "SecretStore 同步", "不存在的东西"):
            assert 0.0 <= kb._keyword_score_norm(q, n) <= 1.0

    def test_raw_score_is_untouched(self):
        """原始分仍保留（10 分制），供 UI 展示命中强度。"""
        kb = KnowledgeBase.__new__(KnowledgeBase)
        assert kb._keyword_score("SecretStore 同步", self._note()) >= 5.0


# ---------------------------------------------------------------------------
# 2) 融合检索
# ---------------------------------------------------------------------------

class TestFusionSearch:
    def test_semantic_score_is_no_longer_crushed_by_keyword_scale(self, tmp_path):
        """语义分 0.9 必须能压过「仅正文命中」的关键词分。

        旧实现里 0.9 < 关键词的 2.0，语义命中会被排到后面 —— 这就是
        「向量检索形同虚设」的直接原因。
        """
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\n正文提到某个词"})
        kb._index_get = lambda: _FakeSemanticIndex({"notes/a.md": 0.9})
        res = kb.search("毫不相关的查询", top_k=5)
        assert res, "语义命中必须出现在结果里"
        assert res[0]["semantic_score"] == pytest.approx(0.9)
        assert res[0]["kind"] in ("semantic", "hybrid")

    def test_weights_are_reflected_in_hybrid_score(self, tmp_path):
        # 注意：_iter_notes 的 title 取自**文件名**（Obsidian wikilink 解析目标），
        # 不是 frontmatter 的 title —— 文件名必须与查询词一致才有关键词命中。
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({"notes/Alpha.md": 0.5})
        res = kb.search("Alpha", top_k=5)
        top = res[0]
        assert top["keyword_score"] > 0, "关键词这一路必须真正命中，否则测不出融合"
        expected = _KB_KW_WEIGHT * top["keyword_score"] + _KB_SEM_WEIGHT * 0.5
        assert top["score"] == pytest.approx(expected, abs=1e-3)

    def test_disabling_semantic_keeps_keyword_path(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({"notes/Alpha.md": 0.99})
        kb._config.kb.semantic_enabled = False
        res = kb.search("Alpha", top_k=5)
        assert res and res[0]["semantic_score"] == 0.0
        assert res[0]["keyword_score"] > 0

    def test_disabling_keyword_keeps_semantic_path(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({"notes/Alpha.md": 0.8})
        kb._config.kb.keyword_enabled = False
        res = kb.search("Alpha", top_k=5)
        assert res and res[0]["score"] == pytest.approx(0.8)

    def test_both_disabled_returns_nothing(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._config.kb.keyword_enabled = False
        kb._config.kb.semantic_enabled = False
        assert kb.search("Alpha", top_k=5) == []


# ---------------------------------------------------------------------------
# 3) 增量索引
# ---------------------------------------------------------------------------

class TestIncrementalIndex:
    def test_first_sync_adds_every_note(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "notes/a.md": "---\ntitle: A\n---\nalpha",
            "notes/b.md": "---\ntitle: B\n---\nbeta",
        })
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        r = kb.sync_index()
        assert r["ok"] and r["added"] == 2 and r["updated"] == 0

    def test_second_sync_is_a_no_op(self, tmp_path):
        """幂等：mtime 未变就不应重新嵌入（embedding 调用有成本）。"""
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\nalpha"})
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        kb.sync_index()
        r = kb.sync_index()
        assert r["added"] == 0 and r["updated"] == 0 and r["unchanged"] == 1

    def test_modified_note_is_reindexed(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\nalpha"})
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        kb.sync_index()
        # 直接把索引里的 mtime 改成「陈旧」：真实场景下文件写入后的 mtime
        # 变化可能小于 0.5s 容差，用 utime 构造会让测试变得不稳定。
        fake.rows["notes/a.md"] = ("stale text", 0.0)
        r = kb.sync_index()
        assert r["updated"] == 1 and r["added"] == 0
        assert fake.rows["notes/a.md"][0] != "stale text"

    def test_externally_added_note_is_picked_up(self, tmp_path):
        """这正是不做增量索引时漏掉的场景（DSH 直接写 vault）。"""
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\nalpha"})
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        kb.sync_index()
        kb._vault.joinpath("notes/新笔记.md").write_text("---\ntitle: 新\n---\n内容", encoding="utf-8")
        r = kb.sync_index()
        assert r["added"] == 1
        assert "notes/新笔记.md" in fake.rows

    def test_deleted_note_is_removed_from_index(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\nalpha"})
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        kb.sync_index()
        kb._vault.joinpath("notes/a.md").unlink()
        r = kb.sync_index()
        assert r["removed"] == 1 and not fake.rows

    def test_force_reindexes_everything(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\nalpha"})
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        kb.sync_index()
        r = kb.sync_index(force=True)
        assert r["added"] == 1 and r["unchanged"] == 0

    def test_index_stats_reports_freshness(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/a.md": "---\ntitle: A\n---\nalpha"})
        fake = _FakeIndex()
        kb._index_get = lambda: fake
        assert kb.stats()["index_missing"] == 1
        kb.sync_index()
        assert kb.stats()["index_missing"] == 0


# ---------------------------------------------------------------------------
# 4) KB 召回通道
# ---------------------------------------------------------------------------

class _FakeKB:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=10, section=""):
        return self._hits[:top_k]


class TestKbRecallChannel:
    def _engine(self, tmp_path, hits):
        cfg = GovernedMemoryConfig()
        cfg.wiki_dir = str(tmp_path / "wiki")
        cfg.l2_db_path = str(tmp_path / "memory" / "l2")
        cfg.vector.backend = "none"
        return RecallEngine(cfg, kb=_FakeKB(hits))

    def test_disabled_by_default(self, tmp_path):
        """recall_min_score=0.0（默认）→ 通道关闭，行为与旧版一致。"""
        engine = self._engine(tmp_path, [{"title": "T", "path": "notes/T.md", "score": 0.9}])
        assert engine._search_kb("q") == []

    def test_no_kb_injected_is_safe(self, tmp_path):
        cfg = GovernedMemoryConfig()
        cfg.vector.backend = "none"
        assert RecallEngine(cfg)._search_kb("q") == []

    def test_hits_above_threshold_become_recall_results(self, tmp_path):
        engine = self._engine(tmp_path, [
            {"title": "SecretStore 同步", "path": "notes/v.md", "score": 0.82,
             "kind": "hybrid", "section": "notes"},
            {"title": "低分笔记", "path": "notes/low.md", "score": 0.30},
        ])
        engine._config.kb.recall_min_score = 0.5
        out = engine._search_kb("secretstore")
        assert len(out) == 1
        assert out[0].layer == "kb"
        assert out[0].content == "SecretStore 同步"
        assert out[0].source == "notes/v.md"

    def test_search_failure_never_breaks_recall(self, tmp_path):
        class _Boom:
            def search(self, *a, **k):
                raise RuntimeError("boom")
        cfg = GovernedMemoryConfig()
        cfg.vector.backend = "none"
        cfg.kb.recall_min_score = 0.5
        engine = RecallEngine(cfg, kb=_Boom())
        assert engine._search_kb("q") == []

    def test_format_recall_renders_knowledge_section(self, tmp_path):
        engine = self._engine(tmp_path, [])
        results = [
            RecallResult(layer="kb", content="SecretStore 同步",
                         score=0.8, source="notes/v.md"),
            RecallResult(layer="l3", content="历史消息", score=0.6),
        ]
        text = engine.format_recall(results, l1_text="", l1_budget=800, l23_budget=1200)
        assert "[Knowledge]" in text
        assert "《SecretStore 同步》" in text
        assert "notes/v.md" in text
        assert "governed_kb_get" in text

    def test_kb_hint_does_not_consume_l23_budget(self, tmp_path):
        """提示段是独立预算 —— 不能把 L2/L3 的内容挤掉。"""
        engine = self._engine(tmp_path, [])
        results = [
            RecallResult(layer="kb", content="笔记", score=0.9, source="n.md"),
            RecallResult(layer="l3", content="X" * 200, score=0.6),
        ]
        text = engine.format_recall(results, l1_text="", l1_budget=800, l23_budget=1200)
        assert "[Knowledge]" in text and "[History]" in text


# ---------------------------------------------------------------------------
# 5) KBIndex schema 迁移
# ---------------------------------------------------------------------------

class TestIndexSchema:
    def test_upsert_accepts_mtime(self):
        import inspect
        from plugin.memory_governed._kb import KBIndex
        sig = inspect.signature(KBIndex.upsert)
        assert "mtime" in sig.parameters

    def test_path_mtimes_contract(self):
        from plugin.memory_governed._kb import KBIndex
        assert hasattr(KBIndex, "path_mtimes")


# ---------------------------------------------------------------------------
# 6) RRF 融合（kb.fusion = "rrf"，Phase 1 / WeKnora 借鉴）
# ---------------------------------------------------------------------------

class TestRRFusion:
    def test_linear_is_default_and_carries_trace(self, tmp_path):
        """默认 linear：分数语义与旧版一致，trace 标明模式与缺席的名次。"""
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({"notes/Alpha.md": 0.5})
        res = kb.search("Alpha", top_k=5)
        top = res[0]
        expected = _KB_KW_WEIGHT * top["keyword_score"] + _KB_SEM_WEIGHT * 0.5
        assert top["score"] == pytest.approx(expected, abs=1e-3)
        assert top["trace"]["fusion"] == "linear"
        assert top["trace"]["kw_rank"] is None
        assert top["trace"]["sem_rank"] is None
        assert top["trace"]["affinity"] == 1.0

    def test_unknown_fusion_value_falls_back_to_linear(self, tmp_path):
        """脏配置不抛错、不改变行为 —— 读路径绝不因配置挂掉。"""
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({})
        kb._config.kb.fusion = "bogus"
        res = kb.search("Alpha", top_k=5)
        assert res and res[0]["trace"]["fusion"] == "linear"

    def test_pure_keyword_ceiling_stays_below_fused_corridor(self, tmp_path):
        """RRF 下纯关键词第 1 名 = 0.4 —— 依然够不到 recall_min_score 0.45。

        这是关键词走廊（recall_min_kw_score）存在的前提（2026-09-17 P0）：
        换融合算法不能悄悄改变两条准入走廊的语义。
        """
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({})
        kb._config.kb.fusion = "rrf"
        res = kb.search("Alpha", top_k=5)
        assert res
        assert res[0]["score"] == pytest.approx(0.4, abs=1e-3)
        assert res[0]["score"] < 0.45

    def test_both_rank_one_saturates_at_one(self, tmp_path):
        """关键词与语义都排第 1 → 融合分恰好 1.0（归一化天花板）。"""
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({"notes/Alpha.md": 0.9})
        kb._config.kb.fusion = "rrf"
        res = kb.search("Alpha", top_k=5)
        assert res
        assert res[0]["score"] == pytest.approx(1.0, abs=1e-3)
        assert res[0]["kind"] == "hybrid"

    def test_trace_reports_both_channel_ranks(self, tmp_path):
        """trace 是排序诊断：两路名次各自可见，缺席的一路是 None。"""
        kb = _make_kb(tmp_path, {
            "notes/Alpha.md": "---\ntitle: Alpha\n---\nbody",
            "notes/Beta.md": "---\ntitle: Beta\n---\nbody",
        })
        # 语义：Beta 第 1、Alpha 第 2；关键词：只有 Alpha 命中。
        kb._index_get = lambda: _FakeSemanticIndex(
            {"notes/Alpha.md": 0.5, "notes/Beta.md": 0.95})
        kb._config.kb.fusion = "rrf"
        res = kb.search("Alpha", top_k=5)
        by_title = {r["title"]: r for r in res}
        alpha, beta = by_title["Alpha"], by_title["Beta"]
        assert alpha["trace"]["kw_rank"] == 1
        assert alpha["trace"]["sem_rank"] == 2
        assert beta["trace"]["kw_rank"] is None
        assert beta["trace"]["sem_rank"] == 1
        # 双通道确认（0.99）压过单通道语义第 1 名（0.6）
        assert alpha["score"] > beta["score"]
        assert beta["score"] == pytest.approx(0.6, abs=1e-3)


# ---------------------------------------------------------------------------
# 7) MMR 去冗余（kb.mmr_lambda）
# ---------------------------------------------------------------------------

#: MMR 用例语料：A/B 近似重复（测挤出），C 完全不同（测保留）。
_MMR_BODIES = {
    "notes/A.md": "---\ntitle: A\n---\n部署流程内容相同的一段说明文字用作相似度对照基准材料",
    "notes/B.md": "---\ntitle: B\n---\n部署流程内容相同的一段说明文字用作相似度对照基准材料补充",
    "notes/C.md": "---\ntitle: C\n---\n完全不同的另一件事关于天气与散步的随手记录",
}
_MMR_SCORES = {"notes/A.md": 0.95, "notes/B.md": 0.94, "notes/C.md": 0.93}


class TestMMRRerank:
    def _kb(self, tmp_path):
        kb = _make_kb(tmp_path, dict(_MMR_BODIES))
        kb._index_get = lambda: _FakeSemanticIndex(dict(_MMR_SCORES))
        return kb

    def test_disabled_keeps_score_order(self, tmp_path):
        """λ=0（默认）→ 纯融合分排序，A、B 近似重复项相邻。"""
        kb = self._kb(tmp_path)
        res = kb.search("毫不相关的查询词xyz", top_k=2)
        assert [r["title"] for r in res] == ["A", "B"]

    def test_lambda_displaces_near_duplicate(self, tmp_path):
        """λ=0.7 → 近似重复的 B 被多样性项 C 挤掉。"""
        kb = self._kb(tmp_path)
        kb._config.kb.mmr_lambda = 0.7
        res = kb.search("毫不相关的查询词xyz", top_k=2)
        assert [r["title"] for r in res] == ["A", "C"]
        assert all(r["trace"].get("mmr") for r in res)

    def test_mmr_short_circuits_when_pool_not_larger_than_topk(self, tmp_path):
        """候选 ≤ top_k 时不重排（没有可挤的对象），结果原样返回。"""
        kb = self._kb(tmp_path)
        kb._config.kb.mmr_lambda = 0.7
        res = kb.search("毫不相关的查询词xyz", top_k=10)
        assert [r["title"] for r in res] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# 8) 使用度亲和度（kb.affinity_enabled / kb_usage 表）
# ---------------------------------------------------------------------------

class TestAffinityAndUsage:
    def test_disabled_by_default_zero_side_effect(self, tmp_path):
        """默认关：touch 不落表、search 不建库 —— 读路径零副作用。"""
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({})
        kb.touch(["notes/Alpha.md"])
        kb.search("Alpha", top_k=5)
        assert not kb._usage_db.exists()

    def test_touch_increments_when_enabled(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._config.kb.affinity_enabled = True
        rel = kb.search("Alpha", top_k=1)[0]["path"]
        kb.touch([rel])
        kb.touch([rel])
        assert kb._usage_store().hits() == {rel: 2}

    def test_search_reads_but_never_writes_usage(self, tmp_path):
        """搜索不是消费：计数只由召回提示通道的 touch 写入。"""
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({})
        kb._config.kb.affinity_enabled = True
        kb.search("Alpha", top_k=5)
        assert kb._usage_store().hits() == {}

    def test_affinity_boost_caps_at_115_percent(self, tmp_path):
        """乘数**真的**封顶 ×1.15 —— 不只是饱和点恰好取到 1.15。

        旧版只测「命中 8 次 == 1.15」，而那个点公式值与上限重合，表达式本身
        无夹取（hits=1000 → ×1.47）。这里补上远超饱和点的情形，让「封顶」
        成为被断言的行为，而不是注释里的一句话。
        """
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._index_get = lambda: _FakeSemanticIndex({})
        kb._config.kb.affinity_enabled = True
        rel = kb.search("Alpha", top_k=1)[0]["path"]
        assert kb.search("Alpha", top_k=1)[0]["trace"]["affinity"] == 1.0
        for _ in range(8):
            kb.touch([rel])
        assert kb.search("Alpha", top_k=1)[0]["trace"]["affinity"] == pytest.approx(1.15)
        # 远超饱和点：仍必须是 1.15（无夹取时这里是 1.47）
        for _ in range(92):
            kb.touch([rel])
        hot = kb.search("Alpha", top_k=1)[0]
        assert kb._usage_store().hits()[rel] == 100
        assert hot["trace"]["affinity"] == pytest.approx(1.15)

    def test_affinity_moves_ranking_but_not_the_admission_score(self, tmp_path):
        """P0 回归（2026-09-22）：使用度只抬**排序**，不抬**相关分**。

        旧版把乘数算进 ``score``，而 KB 提示通道的准入门槛判的正是 ``score``
        （``_recall.py``），于是纯关键词命中（base=0.4）在 hits≥6 时
        0.4×1.13 = 0.45 就压过部署门槛 —— 把「纯关键词够不到 0.45、必须走
        关键词走廊」这条不变量打开。现在 ``score`` 原生、``rank_key`` 才加成。
        """
        kb = _make_kb(tmp_path, {"notes/Kw.md": "---\ntitle: Kw\n---\nKeywordOnlyToken"})
        kb._index_get = lambda: _FakeSemanticIndex({})  # 语义路空 → 纯关键词命中
        kb._config.kb.affinity_enabled = True
        rel = kb.search("KeywordOnlyToken", top_k=1)[0]["path"]
        base = kb.search("KeywordOnlyToken", top_k=1)[0]["score"]
        for _ in range(8):
            kb.touch([rel])
        hot = kb.search("KeywordOnlyToken", top_k=1)[0]
        assert hot["trace"]["affinity"] == pytest.approx(1.15), "乘数本身仍生效"
        assert hot["score"] == pytest.approx(base), "相关分不得被使用度抬高"
        assert hot["rank_key"] > hot["score"], "加成只体现在排序键上"

    def test_affinity_cannot_push_pure_keyword_hit_through_fused_corridor(self, tmp_path):
        """端到端组合用例：``fusion=rrf`` + ``affinity`` 开 + 高 hits + 纯关键词。

        按功能逐项测**测不到**这个组合 —— 每条断言各自绿，而组合起来门槛被
        绕过。以此锁定：该命中必须**不**经 fused 走廊被放行。
        """
        kb = _make_kb(tmp_path, {"notes/Kw.md": "---\ntitle: Kw\n---\nKeywordOnlyToken"})
        kb._index_get = lambda: _FakeSemanticIndex({})
        kb._config.kb.fusion = "rrf"
        kb._config.kb.affinity_enabled = True
        rel = kb.search("KeywordOnlyToken", top_k=1)[0]["path"]
        for _ in range(8):
            kb.touch([rel])

        cfg = GovernedMemoryConfig()
        cfg.vector.backend = "none"
        cfg.kb.recall_min_score = 0.45   # 部署值
        cfg.kb.recall_min_kw_score = 0.0  # 走廊 A 关闭 ⇒ 只剩 fused 走廊可进
        engine = RecallEngine(cfg, kb=kb)
        assert engine._search_kb("KeywordOnlyToken") == [], (
            "纯关键词命中被使用度加成抬过 0.45，经 fused 走廊混进了召回提示")

    def test_search_returns_every_matching_note(self, tmp_path):
        """回归：一条查询必须能返回**多条**结果。

        ``search()`` 的组装语句曾被一次编辑顶格到 ``for`` 循环外，只剩最后一轮
        迭代的值 —— 每条查询只返回一条结果，而当时**整个测试套件全绿**。
        """
        kb = _make_kb(tmp_path, {
            "notes/Alpha.md": "---\ntitle: Alpha\n---\nSharedToken alpha",
            "notes/Beta.md": "---\ntitle: Beta\n---\nSharedToken beta",
            "notes/Gamma.md": "---\ntitle: Gamma\n---\nSharedToken gamma",
        })
        kb._index_get = lambda: _FakeSemanticIndex({})
        out = kb.search("SharedToken", top_k=5)
        assert len(out) == 3, f"每条查询只返回 {len(out)} 条 —— 组装语句疑似又在循环外"

    def test_usage_store_persists_across_instances(self, tmp_path):
        kb = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb._config.kb.affinity_enabled = True
        kb.touch(["notes/Alpha.md"])
        kb2 = _make_kb(tmp_path, {"notes/Alpha.md": "---\ntitle: Alpha\n---\nbody"})
        kb2._config.kb.affinity_enabled = True
        assert kb2._usage_store().hits() == {"notes/Alpha.md": 1}


# ---------------------------------------------------------------------------
# 9) 召回提示通道：trace 透传 + 命中计数
# ---------------------------------------------------------------------------

class TestKbHintTraceTouch:
    class _KB:
        def __init__(self, hits):
            self._hits = hits
            self.touched = []

        def search(self, query, top_k=10, section=""):
            return self._hits[:top_k]

        def touch(self, paths):
            self.touched.extend(paths)

    def _engine(self, tmp_path, hits):
        cfg = GovernedMemoryConfig()
        cfg.wiki_dir = str(tmp_path / "wiki")
        cfg.l2_db_path = str(tmp_path / "memory" / "l2")
        cfg.vector.backend = "none"
        cfg.kb.recall_min_score = 0.45
        kb = self._KB(hits)
        return RecallEngine(cfg, kb=kb), kb

    def test_trace_passthrough_and_touch_on_admit(self, tmp_path):
        engine, kb = self._engine(tmp_path, [{
            "title": "T", "path": "notes/t.md", "section": "notes",
            "score": 0.9, "keyword_score": 1.0, "semantic_score": 0.5,
            "kind": "hybrid", "index_status": {},
            "trace": {"fusion": "rrf", "kw_rank": 1, "sem_rank": 1},
        }])
        out = engine._search_kb("q")
        assert len(out) == 1
        assert out[0].metadata["trace"]["fusion"] == "rrf"
        # 只记真正注入提示的条目（搜索 ≠ 消费）
        assert kb.touched == ["notes/t.md"]

    def test_rejected_hits_are_not_touched(self, tmp_path):
        engine, kb = self._engine(tmp_path, [{
            "title": "弱", "path": "notes/w.md", "score": 0.30,
            "keyword_score": 0.0, "semantic_score": 0.3,
            "kind": "semantic", "index_status": {},
        }])
        assert engine._search_kb("q") == []
        assert kb.touched == []

    def test_kb_without_touch_still_works(self, tmp_path):
        """老 KB 门面（无 touch 方法）不影响提示通道 —— hasattr 门禁。"""
        cfg = GovernedMemoryConfig()
        cfg.wiki_dir = str(tmp_path / "wiki")
        cfg.vector.backend = "none"
        cfg.kb.recall_min_score = 0.45

        class _NoTouch:
            def search(self, query, top_k=10, section=""):
                return [{"title": "T", "path": "notes/t.md", "score": 0.9,
                         "keyword_score": 1.0, "kind": "hybrid",
                         "index_status": {}}]

        engine = RecallEngine(cfg, kb=_NoTouch())
        assert len(engine._search_kb("q")) == 1
