# -*- coding: utf-8 -*-
"""2026-10-20 评估报告里三条已复现缺陷的回归测试。

每条都先复现「坏行为」，再断言修复后的行为 —— 只测"现在能过"不够，
要测"当初为什么会坏"。
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from plugin.memory_governed import _vault
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase


def _make_kb(tmp_path, *, rerank=True, top_n=10):
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "l2")
    cfg.kb.rerank_min_score = 0.05
    cfg.reranking.enabled = rerank
    cfg.reranking.top_n = top_n
    _vault.ensure_skeleton(_vault.vault_root(cfg))
    return cfg, KnowledgeBase(cfg)


class _TruncatingReranker:
    """模拟真实 reranker 的预算语义：只给前 ``top_n`` 篇打分。

    ``_rerank.scores`` 的 ``limit = self._resolve_limit(...)`` 只对
    ``materialized[:limit]`` 发请求；超出预算的候选**根本没有分数**。
    """

    def __init__(self, top_n=10):
        self.top_n = top_n
        self.available = True
        self.submitted = []

    def scores(self, query, texts, top_n=None):
        self.submitted = list(texts)
        out = {}
        for i, text in enumerate(list(texts)[:self.top_n]):
            # 真正的命中给高分，其余给低分（模拟 bge-reranker 的两带）
            out[i] = 0.95 if query in text else 0.01
        return out


class TestCjkNotesSurviveTheRerankBudget:
    """P0-1：按路径码点排序 + reranker 截断 = CJK 标题永久不可检索。

    路径排序下 ASCII 标题在前、中文标题（码点大）在后；一旦在用笔记超过
    ``reranking.top_n``，中文标题连被打分的机会都没有，于是被整条丢弃 ——
    查询词与标题完全一致也返回不了。
    """

    def test_exact_cjk_title_is_retrievable(self, tmp_path):
        cfg, kb = _make_kb(tmp_path)
        # 12 篇 ASCII 标题把预算占满，再放一篇中文标题（路径排序时在最后）
        for i in range(12):
            kb.add("alpha note %02d" % i, "unrelated english content %02d" % i)
        kb.add("飞牛 NAS 硬件配置", "nas 硬件 内存 存储 飞牛")

        fake = _TruncatingReranker(top_n=cfg.reranking.top_n)
        kb._get_reranker = lambda: fake  # type: ignore[assignment]

        hits = kb.search("飞牛 NAS 硬件配置", top_k=10)
        titles = [h["title"] for h in hits]
        assert "飞牛 NAS 硬件配置" in titles, (
            "查询词与标题完全一致却检索不到 —— 候选在进入 reranker 之前"
            "被按路径截断了。命中: %r" % titles)

    def test_the_candidates_handed_to_the_reranker_are_relevance_ordered(
            self, tmp_path):
        """必须在**截断之前**按相关性排序，而不是按路径码点。"""
        cfg, kb = _make_kb(tmp_path)
        for i in range(12):
            kb.add("alpha note %02d" % i, "unrelated english content %02d" % i)
        kb.add("飞牛 NAS 硬件配置", "nas 硬件 内存 存储 飞牛")

        fake = _TruncatingReranker(top_n=cfg.reranking.top_n)
        kb._get_reranker = lambda: fake  # type: ignore[assignment]
        kb.search("飞牛 NAS 硬件配置", top_k=10)

        assert fake.submitted, "reranker 没被调用"
        # ⚠️ submitted 记的是**完整**列表；截断发生在 scores 内部的
        # ``texts[:top_n]``。只断言完整列表会让坏实现同样通过（复评指出
        # 这是一条假见证）。必须只看真正进入预算窗口的那一段。
        window = fake.submitted[:cfg.reranking.top_n]
        assert any("飞牛" in t for t in window), (
            "进入 reranker 预算窗口的前 %d 条里没有最相关的候选"
            % cfg.reranking.top_n)


class TestRebuildHonoursTombstones:
    """P0-2：``l2_rebuild`` 从 L3 重新抽取，必须过墓碑门。"""

    def test_a_retracted_sentence_is_not_resurrected(self, tmp_path,
                                                    monkeypatch):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                               / "scripts"))
        import l2_rebuild as rebuild
        from plugin.memory_governed import _lifecycle as life

        sentence = "生产数据库的备份目录是 /srv/backups/db，每天凌晨执行"
        monkeypatch.setattr(rebuild, "MEMORY_DIR", tmp_path)
        (tmp_path / "l2_tombstones.json").write_text(
            json.dumps({life.content_fp(sentence): {
                "ts": "2026-10-20T10:00:00", "reason": "test-retract",
                "content": sentence}}),
            encoding="utf-8")

        facts, stats = rebuild.extract_facts([
            {"role": "user", "content": sentence, "timestamp": 0}])

        assert facts == [], "被撤回的事实又被重建带回来了"
        assert stats.get("tombstone") == 1, "丢弃必须可数，不能静默"

    def test_no_gate_does_not_disable_the_tombstone_gate(self, tmp_path,
                                                          monkeypatch):
        """``--no-gate`` 只能关质量/密钥门，不能复活已撤回的事实。"""
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                               / "scripts"))
        import l2_rebuild as rebuild
        from plugin.memory_governed import _lifecycle as life

        sentence = "生产数据库的备份目录是 /srv/backups/db，每天凌晨执行"
        monkeypatch.setattr(rebuild, "MEMORY_DIR", tmp_path)
        (tmp_path / "l2_tombstones.json").write_text(
            json.dumps({life.content_fp(sentence): {
                "ts": "2026-10-20T10:00:00", "reason": "test-retract",
                "content": sentence}}),
            encoding="utf-8")

        facts, stats = rebuild.extract_facts(
            [{"role": "user", "content": sentence, "timestamp": 0}],
            gate=False)

        assert facts == [], "--no-gate 复活了被撤回的事实"
        assert stats.get("tombstone") == 1

    def test_the_control_still_extracts_without_the_gate(self, tmp_path,
                                                        monkeypatch):
        """正对照：证明这条句子本来抽得出来，不是被别的规则挡掉的。"""
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent
                               / "scripts"))
        import l2_rebuild as rebuild

        sentence = "生产数据库的备份目录是 /srv/backups/db，每天凌晨执行"
        monkeypatch.setattr(rebuild, "MEMORY_DIR", tmp_path)
        facts, _ = rebuild.extract_facts(
            [{"role": "user", "content": sentence, "timestamp": 0}],
            gate=False)
        assert len(facts) == 1


class TestMocLinksResolve:
    """MOC 死链：frontmatter 标题里的 ``/`` 落盘时被替换，链接必须用文件名。"""

    def test_link_target_is_the_filename_not_the_title(self, tmp_path):
        cfg, kb = _make_kb(tmp_path, rerank=False)
        title = "ThinkServer SR588 配套规格（内存/扩展/存储）"
        kb.add(title, "服务器 硬件 规格 内存 扩展 存储")
        kb.rebuild_moc()

        text = (pathlib.Path(cfg.wiki_dir) / "index.md").read_text(
            encoding="utf-8")
        stem = _vault.slugify_filename(title)

        assert "[[%s" % stem in text, (
            "MOC 没有用文件名做链接目标，Obsidian 会解析成死链：\n%s" % text)
        assert "[[%s]]" % title not in text, "标题里的 / 会导致死链"
