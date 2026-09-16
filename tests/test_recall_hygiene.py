# -*- coding: utf-8 -*-
"""召回卫生（recall hygiene）回归测试：L3 角色过滤 + format_recall 去重 + KB min_score。

背景（实测于真实部署数据）
--------------------------
1. **L3 角色污染**：L3 存档的 657 条消息里 ``tool`` 角色占 396 条（~60%），
   全是 JSON 结果 / 终端输出 / 文件列表。旧召回不按 role 过滤，这些过程输出会
   以 score=1.0 挤进注入上下文。实测 "linux" 的命中里 tool 占 57/60。
2. **L2 重复挤占预算**：L2 表 2650 行只有 137 个唯一 content（重复率 94.8%），
   ``format_recall`` 对同一 content 的多个副本各占一份预算，把真正多样的记忆
   挤出 1200-token 的 l23 budget。
3. **kb.min_score 是死配置**：``_kb.KnowledgeBase.search`` 完全不读它，配置
   0.0 时表现为"无任何过滤"。

全部在 bare interpreter 上可跑：L3 用真实 sqlite（标准库）+ 内存构造的存档，
召回引擎用 fake 表 / fake 索引，不依赖 lancedb / fastembed。
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase, _KB_SEM_WEIGHT
from plugin.memory_governed._recall import (
    L3_RECALL_ROLES,
    MIN_SCORE,
    RecallEngine,
    RecallResult,
)


# ---------------------------------------------------------------------------
# 公共假件
# ---------------------------------------------------------------------------


def _make_engine(tmp_path) -> RecallEngine:
    """构造一个不触真实 lancedb / embedding 的 RecallEngine。"""
    cfg = GovernedMemoryConfig()
    cfg.l1_memory_path = str(tmp_path / "MEMORY.md")
    cfg.l1_user_path = str(tmp_path / "USER.md")
    cfg.l2_db_path = str(tmp_path / "l2")
    cfg.l3_db_path = str(tmp_path / "l3" / "l3.db")
    cfg.l4_persona_path = str(tmp_path / "persona.md")
    return RecallEngine(cfg)


# ---------------------------------------------------------------------------
# 1) L3 角色过滤
# ---------------------------------------------------------------------------

# 英文行（走 FTS5 分支）：tool 与 user 共享同一关键词。
EN_USER = "linux secretstore deployment notes for the NAS server"
EN_TOOL = "linux secretstore deployment tool output: container list json dump"
EN_ASSIST = "linux kernel update available, assistant summary"

# 中文行（走 CJK LIKE 分支）。
CN_USER = "北京今天天气不错适合出门跑步"
CN_TOOL = "北京今天天气工具调用输出日志块"
CN_ASSIST = "北京项目进度需要尽快跟进"

# 只有 tool 命中的标记串：过滤后应干净返回空。
TOOL_ONLY = "zzztoolonlymarker never recalled from tool role"


@pytest.fixture
def l3_mixed_roles(config):
    """在真实 sqlite 里塞入 user / assistant / tool 三种角色的存档。"""
    conn = sqlite3.connect(config.l3_db_path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL, hash TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
    )
    now = time.time()
    rows = [
        ("user", EN_USER),
        ("tool", EN_TOOL),
        ("assistant", EN_ASSIST),
        ("user", CN_USER),
        ("tool", CN_TOOL),
        ("assistant", CN_ASSIST),
        ("tool", TOOL_ONLY),
    ]
    for i, (role, content) in enumerate(rows):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            ("s-mix", role, content, now - i),
        )
        conn.execute(
            "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?,?,?,?)",
            (content, role, "s-mix", now - i),
        )
    conn.commit()
    conn.close()
    return config


def _contents(engine, query):
    return {r.content for r in engine._search_l3(query)}


class TestL3RoleFilter:
    """L3 召回必须把 tool 角色挡在门外（FTS 与 LIKE 两条分支都要）。"""

    def test_whitelist_contract(self):
        """角色白名单是模块级常量，值为 user/assistant。"""
        assert L3_RECALL_ROLES == ("user", "assistant")

    def test_fts_branch_excludes_tool(self, l3_mixed_roles):
        """英文查询走 FTS5：共享关键词的 tool 行不得出现。"""
        hits = _contents(RecallEngine(l3_mixed_roles), "secretstore")
        assert EN_USER in hits, f"user 行被误伤：{sorted(hits)}"
        assert EN_TOOL not in hits, f"tool 行混入召回：{sorted(hits)}"

    def test_fts_branch_keeps_user_and_assistant(self, l3_mixed_roles):
        hits = _contents(RecallEngine(l3_mixed_roles), "linux")
        assert EN_USER in hits and EN_ASSIST in hits, f"正常行被丢失：{sorted(hits)}"
        assert EN_TOOL not in hits, f"tool 行混入召回：{sorted(hits)}"

    def test_like_branch_excludes_tool(self, l3_mixed_roles):
        """纯中文查询走 LIKE 分支：tool 行不得出现。"""
        hits = _contents(RecallEngine(l3_mixed_roles), "北京")
        assert CN_USER in hits and CN_ASSIST in hits, f"正常中文行被丢失：{sorted(hits)}"
        assert CN_TOOL not in hits, f"tool 行混入 CJK 召回：{sorted(hits)}"

    def test_like_or_chain_is_parenthesized(self, l3_mixed_roles):
        """多片段 OR 与 role 过滤的优先级：只能命中"含任一片段且角色合法"的行。

        若 role 条件被错误地并进 OR 链，任一条 tool 行都会被带出来 —— 这条
        用例会因此变红。
        """
        engine = RecallEngine(l3_mixed_roles)
        hits = _contents(engine, "secretstore 北京")
        assert EN_TOOL not in hits and CN_TOOL not in hits, f"tool 行泄漏：{sorted(hits)}"
        assert EN_USER in hits and CN_USER in hits, f"正常行被丢失：{sorted(hits)}"

    def test_only_tool_matches_returns_empty_cleanly(self, l3_mixed_roles):
        """命中的全是 tool 行时，过滤后必须干净返回 []，不抛异常。"""
        engine = RecallEngine(l3_mixed_roles)
        assert engine._search_l3("zzztoolonlymarker") == []


# ---------------------------------------------------------------------------
# 2) format_recall 去重 + 排序
# ---------------------------------------------------------------------------


class TestFormatRecallDedup:
    """同一 content 的重复副本只能进上下文一次，且高分优先。"""

    def test_duplicate_content_appears_once(self, tmp_path):
        engine = _make_engine(tmp_path)
        results = [
            RecallResult(layer="l2", content="重复事实", score=0.9),
            RecallResult(layer="l2", content="重复事实", score=0.9),
            RecallResult(layer="l2", content="重复事实", score=0.8),
            RecallResult(layer="l3", content="另一条历史", score=0.5),
        ]

        text = engine.format_recall(results, "", l1_budget=800, l23_budget=1200)

        assert text.count("重复事实") == 1, f"重复内容出现了 {text.count('重复事实')} 次"
        assert "另一条历史" in text

    def test_dedup_keeps_highest_score_version(self, tmp_path):
        """L2/L3 撞同一 content 时保留高分那条（tag 随之变化）。"""
        engine = _make_engine(tmp_path)
        results = [
            RecallResult(layer="l2", content="同一内容", score=0.4),
            RecallResult(layer="l3", content="同一内容", score=0.7),
        ]

        text = engine.format_recall(results, "", l1_budget=800, l23_budget=1200)

        assert text.count("同一内容") == 1
        assert "[History] 同一内容" in text
        assert "[Fact] 同一内容" not in text

    def test_high_score_is_ordered_before_low_score(self, tmp_path):
        """预算只够装一条时，留下的必须是高分那条（不能依赖调用方排序）。"""
        engine = _make_engine(tmp_path)
        # 输入顺序故意把低分放前面；预算 1 → budget_chars=4，每条 3 字符只够 1 条。
        results = [
            RecallResult(layer="l2", content="AAA", score=0.3),
            RecallResult(layer="l2", content="BBB", score=0.9),
        ]

        text = engine.format_recall(results, "", l1_budget=800, l23_budget=1)

        assert "BBB" in text, f"高分条目被低分挤掉：{text!r}"
        assert "AAA" not in text, f"低分条目占用了预算：{text!r}"

    def test_dedup_does_not_bypass_min_score(self, tmp_path):
        """去重不能把低于 MIN_SCORE 的条目放进来。"""
        engine = _make_engine(tmp_path)
        results = [
            RecallResult(layer="l2", content="低分重复", score=MIN_SCORE - 0.05),
            RecallResult(layer="l2", content="低分重复", score=MIN_SCORE - 0.05),
            RecallResult(layer="l2", content="达标内容", score=MIN_SCORE + 0.1),
        ]

        text = engine.format_recall(results, "", l1_budget=800, l23_budget=1200)

        assert "低分重复" not in text
        assert "达标内容" in text

    def test_empty_results_returns_empty(self, tmp_path):
        engine = _make_engine(tmp_path)
        assert engine.format_recall([], "", l1_budget=800, l23_budget=1200) == ""


# ---------------------------------------------------------------------------
# 3) KB min_score 真正生效
# ---------------------------------------------------------------------------


def _kb_with_semantic_scores(tmp_path, scores):
    """构造一个 KB，其语义索引固定返回给定分数的笔记骨架。"""
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"  # 关闭真实 embedding 探测
    kb = KnowledgeBase(cfg)
    kb.ensure()

    class _FakeIdx:
        available = True

        def search(self, query, k):
            return [
                {
                    "path": f"notes/n{i}.md",
                    "text": f"note {i} body",
                    "score": s,
                    "kind": "semantic",
                }
                for i, s in enumerate(scores)
            ]

    kb._index_get = lambda: _FakeIdx()  # type: ignore[assignment]
    return kb


class TestKbMinScore:
    """``kb.min_score`` 必须真正过滤低分条目，且默认 0.0 完全兼容旧行为。"""

    def test_default_zero_does_not_filter(self, tmp_path):
        """min_score=0.0 → 结果数不变（向后兼容，等价于旧行为）。"""
        kb = _kb_with_semantic_scores(tmp_path, [0.9, 0.4, 0.1])
        assert kb._config.kb.min_score == 0.0
        assert len(kb.search("q", top_k=10)) == 3

    def test_high_min_score_filters_low_items(self, tmp_path):
        kb = _kb_with_semantic_scores(tmp_path, [0.9, 0.4, 0.1])
        kb._config.kb.min_score = 0.5
        results = kb.search("q", top_k=10)
        # 融合分 = kw*W_kw + sem*W_sem；mock 的笔记不在真实 vault 里，
        # 命中不了关键词，所以只剩语义分量：sem * W_sem。
        assert [r["score"] for r in results] == pytest.approx([0.9 * _KB_SEM_WEIGHT])

    def test_mid_min_score_keeps_items_at_or_above_threshold(self, tmp_path):
        """``>= min_score``：正好等于阈值的条目保留。"""
        kb = _kb_with_semantic_scores(tmp_path, [0.9, 0.4, 0.1])
        kb._config.kb.min_score = 0.4 * _KB_SEM_WEIGHT
        results = kb.search("q", top_k=10)
        assert [r["score"] for r in results] == pytest.approx(
            [0.9 * _KB_SEM_WEIGHT, 0.4 * _KB_SEM_WEIGHT]
        )

    def test_min_score_still_respects_top_k(self, tmp_path):
        kb = _kb_with_semantic_scores(tmp_path, [0.9, 0.8, 0.7, 0.6])
        kb._config.kb.min_score = 0.0
        assert len(kb.search("q", top_k=2)) == 2

    def test_missing_min_score_field_is_defensive(self, tmp_path):
        """旧配置缺 min_score 字段时按 0.0 处理，不能抛异常。"""
        kb = _kb_with_semantic_scores(tmp_path, [0.9, 0.4, 0.1])
        del kb._config.kb.min_score
        assert len(kb.search("q", top_k=10)) == 3

    def test_no_results_when_all_below_threshold(self, tmp_path):
        kb = _kb_with_semantic_scores(tmp_path, [0.3, 0.2, 0.1])
        kb._config.kb.min_score = 0.9
        assert kb.search("q", top_k=10) == []
