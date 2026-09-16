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
