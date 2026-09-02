# -*- coding: utf-8 -*-
"""Phase 3（对话自动沉淀 + 分流 + 链接补全）治理层回归测试。

Scope:
- 密钥闸门：落库前 fail-closed，含密钥一律拒绝，绝不写 vault。
- 置信分流：confidence >= 阈值 → notes；< 阈值 → inbox + review_required；
  不传 confidence → 按显式 section（行为不变）。
- 链接补全：共享 concepts/tags 的笔记自动互加 [[wikilink]]（双向），
  去重、且 inbox 待审不互链。

全部在 bare interpreter 上可跑：``config.vector.backend = "none"``。
"""

from __future__ import annotations

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase


@pytest.fixture
def kb(tmp_path):
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    k = KnowledgeBase(cfg)
    k.ensure()
    return k


# ---------------------------------------------------------------------------
# 密钥闸门（fail-closed）
# ---------------------------------------------------------------------------

class TestSecretGate:
    def test_rejects_openai_style_key_in_body(self, kb):
        r = kb.add(title="笔记", body="我的 key 是 sk-abcdefghijklmnop12345")
        assert r["ok"] is False
        assert "secret" in r["error"]
        assert kb.get("笔记") is None  # 没写盘

    def test_rejects_key_value_secret_in_title(self, kb):
        r = kb.add(title="api_key: sk-abcdefghijklmnop", body="正文")
        assert r["ok"] is False
        assert kb.get("api_key: sk-abcdefghijklmnop") is None

    def test_clean_content_passes(self, kb):
        r = kb.add(title="普通笔记", body="没有密钥的正常内容")
        assert r["ok"] is True
        assert kb.get("普通笔记") is not None


# ---------------------------------------------------------------------------
# 置信分流
# ---------------------------------------------------------------------------

class TestConfidenceRouting:
    def test_high_confidence_goes_notes(self, kb):
        r = kb.add(title="高置信", body="内容", confidence=0.9)
        assert r["section"] == "notes"
        assert r["review_required"] is False
        note = kb.get("高置信")
        assert note["section"] == "notes"
        assert note["meta"]["confidence"] == 0.9
        assert "review_required" not in note["meta"]

    def test_low_confidence_goes_inbox(self, kb):
        r = kb.add(title="低置信", body="内容", confidence=0.3)
        assert r["section"] == "inbox"
        assert r["review_required"] is True
        note = kb.get("低置信")
        assert note["section"] == "inbox"
        assert note["meta"]["review_required"] is True

    def test_no_confidence_uses_explicit_section(self, kb):
        # 不传 confidence：行为不变，显式 section 生效
        r = kb.add(title="手动", body="内容", section="inbox")
        assert r["section"] == "inbox"
        assert r["review_required"] is False

    def test_confidence_overrides_explicit_section(self, kb):
        # 传了 confidence 就由阈值接管 notes/inbox（忽略显式 section）
        r = kb.add(title="覆盖", body="内容", section="inbox", confidence=0.95)
        assert r["section"] == "notes"

    def test_threshold_boundary(self, kb):
        # 默认阈值 0.7：恰好等于阈值应进 notes
        r = kb.add(title="边界", body="内容", confidence=0.7)
        assert r["section"] == "notes"

    def test_custom_threshold(self, tmp_path):
        cfg = GovernedMemoryConfig()
        cfg.wiki_dir = str(tmp_path / "wiki")
        cfg.l2_db_path = str(tmp_path / "memory" / "l2")
        cfg.vector.backend = "none"
        cfg.kb.confidence_threshold = 0.5
        k = KnowledgeBase(cfg)
        k.ensure()
        assert k.add(title="x", body="b", confidence=0.6)["section"] == "notes"
        assert k.add(title="y", body="b", confidence=0.4)["section"] == "inbox"


# ---------------------------------------------------------------------------
# 链接补全（双向 wikilink）
# ---------------------------------------------------------------------------

class TestLinkCompletion:
    def test_shared_concept_creates_bidirectional_links(self, kb):
        kb.add(title="笔记A", body="Postgres 是关系型数据库", concepts=["Postgres"])
        r = kb.add(title="笔记B", body="用 Postgres 做 OLTP", concepts=["Postgres"])
        assert r["links_added"] >= 2  # B→A 正向 + A→B 反向
        a = kb.get("笔记A")
        b = kb.get("笔记B")
        assert "[[笔记B]]" in a["body"]
        assert "[[笔记A]]" in b["body"]

    def test_no_shared_concept_no_links(self, kb):
        kb.add(title="甲", body="内容", concepts=["数据库"])
        r = kb.add(title="乙", body="内容", concepts=["烹饪"])
        assert r["links_added"] == 0
        assert "[[甲]]" not in kb.get("乙")["body"]

    def test_links_dedup_on_update(self, kb):
        kb.add(title="A", body="x", concepts=["共享"])
        kb.add(title="B", body="y", concepts=["共享"])
        # 重复 add 不应重复追加链接
        kb.add(title="B", body="y", concepts=["共享"], update=True)
        a = kb.get("A")
        assert a["body"].count("[[B]]") == 1

    def test_inbox_not_linked(self, kb):
        # 低置信进 inbox：不触发链接补全
        kb.add(title="正库", body="x", concepts=["共享"])
        r = kb.add(title="待审", body="y", concepts=["共享"], confidence=0.3)
        assert r["section"] == "inbox"
        assert r["links_added"] == 0
        assert "[[待审]]" not in kb.get("正库")["body"]

    def test_tag_match_also_links(self, kb):
        kb.add(title="T1", body="x", tags=["技术"])
        r = kb.add(title="T2", body="y", tags=["技术"])
        assert r["links_added"] >= 2


# ---------------------------------------------------------------------------
# review 门（inbox 待审 → approve/reject 闭环）
# ---------------------------------------------------------------------------

class TestReviewGate:
    def test_list_review_empty(self, kb):
        assert kb.list_review() == []

    def test_list_review_shows_pending(self, kb):
        kb.add(title="高", body="x", confidence=0.9)            # notes，不在 review
        kb.add(title="低", body="y", confidence=0.3)            # inbox + review_required
        kb.add(title="手动inbox", body="z", section="inbox")    # 显式 inbox
        titles = {p["title"] for p in kb.list_review()}
        assert "低" in titles
        assert "手动inbox" in titles
        assert "高" not in titles

    def test_approve_moves_to_notes(self, kb):
        kb.add(title="待审", body="内容", confidence=0.3)
        r = kb.approve("待审")
        assert r["ok"] is True
        assert r["section"] == "notes"
        note = kb.get("待审")
        assert note["section"] == "notes"
        assert note["meta"]["type"] == "notes"
        assert "review_required" not in note["meta"]

    def test_reject_moves_to_archive(self, kb):
        kb.add(title="垃圾", body="内容", confidence=0.2)
        r = kb.reject("垃圾")
        assert r["ok"] is True
        assert r["section"] == "archive"
        note = kb.get("垃圾")
        assert note["section"] == "archive"
        assert note["meta"]["review_rejected"] is True

    def test_approve_non_inbox_errors(self, kb):
        kb.add(title="正库", body="x")  # notes
        r = kb.approve("正库")
        assert r["ok"] is False
        assert "not in inbox" in r["error"]

    def test_reject_missing_errors(self, kb):
        r = kb.reject("不存在")
        assert r["ok"] is False
        assert "not found" in r["error"]

    def test_approve_triggers_link(self, kb):
        kb.add(title="正库", body="x", concepts=["共享"])
        kb.add(title="待审", body="y", concepts=["共享"], confidence=0.3)
        # inbox 阶段不互链
        assert "[[待审]]" not in kb.get("正库")["body"]
        r = kb.approve("待审")
        assert r["ok"] is True
        assert r["links_added"] >= 2  # approve 后补双向链接
        assert "[[待审]]" in kb.get("正库")["body"]
        assert "[[正库]]" in kb.get("待审")["body"]

    def test_approve_clears_pending_list(self, kb):
        kb.add(title="待审", body="内容", confidence=0.3)
        assert len(kb.list_review()) == 1
        kb.approve("待审")
        assert kb.list_review() == []
