# -*- coding: utf-8 -*-
"""Knowledge base (``_kb.py`` / ``_vault.py``) regression tests.

Scope: vault 骨架、frontmatter 解析/序列化、写入 + 自动 wikilink、关键词检索
（中文 bigram 兜底）、反链、section 过滤、索引降级（无 embedding 后端时
语义索引不可用但关键词检索仍工作）。

全部在 bare interpreter 上可跑：不依赖 lancedb / sentence-transformers /
fastembed；通过 ``config.vector.backend = "none"`` 关闭 embedding 后端，
``KBIndex`` 应优雅降级为 unavailable，绝不抛异常。
"""

from __future__ import annotations

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase
from plugin.memory_governed._vault import (
    parse_frontmatter,
    dump_frontmatter,
    slugify_filename,
)


@pytest.fixture
def kb(tmp_path):
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"  # 关闭 embedding，测降级路径
    kb = KnowledgeBase(cfg)
    kb.ensure()
    return kb


# ---------------------------------------------------------------------------
# _vault：frontmatter 解析 / 序列化
# ---------------------------------------------------------------------------

class TestFrontmatter:
    def test_parse_standard(self):
        text = "---\ntitle: 你好\nconcepts: [A, B]\n---\n正文内容"
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "你好"
        assert meta["concepts"] == ["A", "B"]
        assert body == "正文内容"

    def test_parse_no_frontmatter(self):
        meta, body = parse_frontmatter("只有正文\n没有元数据")
        assert meta == {}
        assert body == "只有正文\n没有元数据"

    def test_parse_open_fence_has_whitespace(self):
        """首行围栏允许前后空白（编辑器常见）。"""
        text = "  ---  \ntitle: x\n---\nbody"
        meta, body = parse_frontmatter(text)
        assert meta["title"] == "x"
        assert body == "body"

    def test_roundtrip(self):
        meta = {"title": "T", "tags": ["a", "b"], "concepts": ["c"], "n": 3}
        text = dump_frontmatter(meta)
        parsed, _ = parse_frontmatter(text + "\n\nbody")
        assert parsed["title"] == "T"
        assert parsed["tags"] == ["a", "b"]
        assert parsed["concepts"] == ["c"]
        assert parsed["n"] == 3

    def test_slugify_keeps_chinese(self):
        assert slugify_filename("数据库选型") == "数据库选型"

    def test_slugify_strips_illegal(self):
        assert slugify_filename("a/b:c") == "a b c"


# ---------------------------------------------------------------------------
# _kb：写入 / 检索 / 反链
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_ensure_skeleton(self, kb):
        root = kb._vault
        for sub in ["inbox", "notes", "projects", "areas", "resources", "archive"]:
            assert (root / sub).is_dir()
        assert (root / "index.md").exists()

    def test_add_writes_frontmatter_and_autolink(self, kb):
        kb.add(
            title="数据库选型",
            body="我们用 Postgres，不用 MySQL。",
            section="notes",
            tags=["技术"],
            concepts=["Postgres"],
            source="https://example.com",
        )
        note = kb.get("数据库选型")
        assert note is not None
        assert note["section"] == "notes"
        assert note["meta"]["tags"] == ["技术"]
        assert note["meta"]["concepts"] == ["Postgres"]
        assert note["meta"]["source"] == "https://example.com"
        # 自动 wikilink
        assert "[[Postgres]]" in note["body"]
        assert "MySQL" in note["body"]

    def test_autolink_no_double_link(self, kb):
        kb.add(title="X", body="[[Postgres]] 是数据库", concepts=["Postgres"])
        note = kb.get("X")
        # 已链接的不重复包
        assert note["body"].count("[[Postgres]]") == 1

    def test_add_invalid_section_falls_back_to_inbox(self, kb):
        r = kb.add(title="Y", body="b", section="bogus")
        assert r["section"] == "inbox"

    def test_keyword_search_chinese_bigram(self, kb):
        kb.add(title="数据库选型", body="Postgres 适合 OLTP", concepts=["Postgres"])
        kb.add(title="周末计划", body="去公园散步")
        results = kb.search("数据库")
        titles = {r["title"] for r in results}
        assert "数据库选型" in titles

    def test_keyword_search_by_concept(self, kb):
        kb.add(title="A", body="内容", concepts=["向量检索"])
        results = kb.search("向量检索")
        assert any(r["title"] == "A" for r in results)

    def test_search_empty_query(self, kb):
        assert kb.search("") == []

    def test_backlinks(self, kb):
        kb.add(title="Postgres", body="这是一个关系型数据库")
        kb.add(title="笔记B", body="参考 [[Postgres]] 的做法", concepts=["Postgres"])
        back = kb.links("Postgres")
        assert any("笔记B" in p for p in back)

    def test_list_notes_section_filter(self, kb):
        kb.add(title="n1", body="x", section="notes")
        kb.add(title="n2", body="x", section="inbox")
        notes = kb.list_notes("notes")
        assert {n["title"] for n in notes} == {"n1"}

    def test_index_degraded_but_search_works(self, kb):
        """无 embedding 后端：语义索引 unavailable，但关键词检索仍返回结果。"""
        kb.add(title="降级测试", body="lancedb 未装也不影响关键词")
        assert kb.index_available is False
        results = kb.search("降级测试")
        assert any(r["title"] == "降级测试" for r in results)

    def test_stats_counts(self, kb):
        kb.add(title="a", body="x", section="notes")
        kb.add(title="b", body="x", section="inbox")
        stats = kb.stats()
        assert stats["total"] == 2
        assert stats["by_section"]["notes"] == 1
        assert stats["by_section"]["inbox"] == 1

    def test_search_dedup_semantic_and_keyword(self, kb, monkeypatch):
        """语义 + 关键词命中同一篇时只返回一次（索引路径必须与去重键一致）。"""
        kb.add(title="X", body="内容", concepts=["向量"])

        class _FakeIdx:
            available = True

            def search(self, q, k):
                # 模拟语义检索命中同一篇（相对路径）
                return [{"path": "notes/X.md", "text": "X 内容", "score": 0.9, "kind": "semantic"}]

        monkeypatch.setattr(kb, "_index_get", lambda: _FakeIdx())
        results = kb.search("向量")
        hits = [r for r in results if r["title"] == "X"]
        assert len(hits) == 1

    def test_add_index_uses_relative_path(self, kb, monkeypatch):
        """写入索引的 path 必须是相对 vault 的路径（否则检索去重失效）。"""
        captured = {}

        class _FakeIdx:
            available = True

            def upsert(self, path, text, mtime=0.0):
                captured["path"] = path

        monkeypatch.setattr(kb, "_index_get", lambda: _FakeIdx())
        kb.add(title="Y", body="b")
        assert captured["path"] == "notes/Y.md"
        assert not captured["path"].startswith("\\")
        assert "wiki" not in captured["path"]
