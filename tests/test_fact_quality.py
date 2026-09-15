# -*- coding: utf-8 -*-
"""L2 fact-quality gate tests (noise control for semantic memory).

Scope: ``plugin/memory_governed/_sync.py`` — ``WriteQueue._looks_like_fact``,
``_fact_signal_score`` and ``WriteQueue._extract_atomic_facts``.

Background: the real L2 corpus (137 rows) was ~83% noise — self-check report
fragments, markdown structure, assistant narration ("让我检查一下…"), system
notices and status-report metadata rows. Those rows are not merely dead weight
in a VECTOR store: they win recalls that real knowledge should have won, and
nothing downstream can clean them up because the Bridge review gate sits
between L2 and L1. So the gate has to reject them at extraction time.

This module pins the behaviour with a golden sample set taken from that
corpus: every "must drop" and "must keep" string below occurred for real.

The tests hold with lancedb / sentence-transformers / pandas absent (the
default deployment shape): only ``WriteQueue`` pure functions are exercised.
"""

from __future__ import annotations

import pytest

from plugin.memory_governed._sync import (
    WriteQueue,
    _ASSISTANT_MIN_WEIGHTED_LEN,
    _ROLE_WEIGHTS,
    _TITLE_FRAGMENT_MAX_WEIGHT,
    _fact_signal_score,
    _undecorate,
)

# ---------------------------------------------------------------------------
# Golden sample set — every string came out of the real L2 table.
# ---------------------------------------------------------------------------

# Self-check report output, markdown structure, assistant narration, system
# notices, politeness, metadata rows and result announcements: none of these
# are durable knowledge.
MUST_DROP = [
    # assistant process narration
    "让我检查一下记忆系统的状态",
    "让我修复这个 bug",
    "让我验证一下修复是否生效",
    "好的，让我自检一下整个记忆系统的状态",
    # markdown structure
    "| 组件 | 状态 | 说明 |",
    "================================",
    "--------------------------------------",
    "## 方案一：仅共享给机器人（推荐）",
    # system-injected notices
    ("[System note: The previous turn was interrupted by a gateway shutdown; "
     "the gateway is no longer running.]"),
    "Operation interrupted",
    # status-report metadata rows
    "Provider: governed | 可用: ✓",
    "大小: 236KB",
    "时间衰减: 168 小时",
    "状态: ✓ 运行正常",
    "存储: LanceDB (memory/l2/memories.lance/)",
    # self-check boilerplate
    "记忆系统自检报告 (hermes-memory-governed)",
    # politeness
    "有什么我能帮你的吗",
    # assistant promise / outlook about a single run
    "以后管理家用 NAS 就不需要找本地目录了，直接通过 Hermes 的 skill 系统调用",
]

# Real knowledge: user constraints, root causes and technical facts that all
# survived the manual audit of the corpus.
MUST_KEEP = [
    "我在NAS有装secretstore，但本地布署好像不能同步给其它地方装的secretstore",
    "必须同步 rsa.key，否则两边加密的数据不兼容",
    "SecretStore 使用 `ROCKET_TLS` 而不是 `SSL_CERT_FILE`",
    "我家里可以正常访问和使用，但我说的同步是家里再装一个secretstore",
    "都太麻烦，我需要的是本地ai能读取，同时能同步给其它地方使用",
    ("先解决hermes读取吧，但我现在只能用域名才能https访问，"
     "还需要你帮忙解决"),
    "不能同时写入 — 同一时间只在一个实例操作，否则数据库冲突",
    "不管哪个方案，Hermes 都可以通过 Bitwarden CLI 或 API 读取",
]

# The same known-good facts as they were ACTUALLY stored, i.e. wrapped in
# bullets, status symbols and markdown bold. Decoration must not smuggle a
# real fact past the gate in either direction.
MUST_KEEP_DECORATED = [
    "- ⚠️ **必须同步 `rsa.key`**，否则两边加密的数据不兼容",
    "- ⚠️ **不能同时写入** — 同一时间只在一个实例操作，否则数据库冲突",
]


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------

class TestGoldenSampleSet:
    @pytest.mark.parametrize("text", MUST_DROP)
    def test_noise_is_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), f"noise survived: {text}"

    @pytest.mark.parametrize("text", MUST_KEEP)
    def test_real_knowledge_is_kept(self, text):
        assert WriteQueue._looks_like_fact(text), f"real knowledge lost: {text}"

    @pytest.mark.parametrize("text", MUST_KEEP_DECORATED)
    def test_decoration_does_not_hide_real_knowledge(self, text):
        assert WriteQueue._looks_like_fact(text), f"real knowledge lost: {text}"


# ---------------------------------------------------------------------------
# A. markdown structure
# ---------------------------------------------------------------------------

class TestMarkdownStructure:
    @pytest.mark.parametrize("text", [
        "================================",   # '=' rule
        "----",                               # '-' rule
        "______________________________",     # '_' rule
        "| 组件 | 状态 | 说明 |",             # table header row
    ])
    def test_structural_forms_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text

    def test_table_body_row_is_rejected(self):
        assert not WriteQueue._looks_like_fact(
            "| **L2 语义记忆** | ⏳ 待积累 | LanceDB 尚未创建 |"
        )

    @pytest.mark.parametrize("text", [
        "## 方案三：链接分享 + 密码",     # spaces in the title
        "### 方案 A：你手动复制（简单）",  # spaces after the marker
        "# 方案1：Bitwarden CLI",          # no space before '：'
        "## ✅ 对接状态",                  # emoji status heading
        "## 🔧 已修复的 Bug",
        "### 结论",
    ])
    def test_headings_with_spaces_are_rejected(self, text):
        """`\\S+` in the old heading pattern missed any title containing a space."""
        assert not WriteQueue._looks_like_fact(text), text

    @pytest.mark.parametrize("text", [
        "**✅ 正常的部分：**",
        "**⚠️ 缺失的部分：**",
        "**使用方式：**",
        "**注意事项**：",
        "**总结**：原 memory 系统和 governed 系统对接正常",
        "**方案一（age 加密）** 最平衡：",
        "** 现在可以这样使用：",
    ])
    def test_bold_labels_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text

    @pytest.mark.parametrize("text", [
        "L1 手写规则层 (最高信任)",
        "L2 语义记忆 (向量搜索)",
        "知识库 (KB / Obsidian Vault)",
        "项目规则和行为约束",
    ])
    def test_section_titles_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text

    @pytest.mark.parametrize("text", [
        "archive/meetings/  - 17篇会议纪要 (2026-05-06~07)",
        "archive/chats/     - 2篇聊天记录",
        "notes/             - 4篇综合归纳笔记",
        "├── rsa.key             # 加密密钥（必须同步",
        "└── config.json         # 配置文件",
    ])
    def test_directory_and_tree_lines_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text


# ---------------------------------------------------------------------------
# A. system noise
# ---------------------------------------------------------------------------

class TestSystemNoise:
    @pytest.mark.parametrize("text", [
        "[System note: The previous turn was interrupted by a gateway shutdown; "
        "the gateway is no longer running.]",
        "Operation interrupted",
        "gateway shutdown",
        "Report to the user that the session was restored successfully and ask "
        "what they would like to do next",
        "Any restart/shutdown command in the history has already run — do NOT "
        "re-execute or verify it",
    ])
    def test_injected_notices_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text


# ---------------------------------------------------------------------------
# A. status-report metadata rows
# ---------------------------------------------------------------------------

class TestMetadataRows:
    @pytest.mark.parametrize("text", [
        "Provider: governed | 可用: ✓",
        "大小: 236KB",
        "时间衰减: 168 小时",
        "状态: ✓ 运行正常",
        "存储: LanceDB (memory/l2/memories.lance/)",
        "文件: memory/MEMORY.md + USER.md",
        "内容: MEMORY.md 只有一条测试记录",
        "总文件: 32个 .md 文件",
        "索引状态: ✓ 可用",
        "向量索引: LanceDB (memory/kb_index/kb_notes.lance/)",
        "Embedding: BAAI/bge-m3 @ SiliconFlow (1024维)",
        "Reranking: BAAI/bge-reranker-v2-m3",
        "路径: C:/Users/example/wiki/",
    ])
    def test_metadata_rows_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text

    @pytest.mark.parametrize("text", [
        # decoration must not shield a metadata row from the rule
        "- **Provider**: `governed` 已激活",
        "- **L2（语义记忆 - LanceDB）**: 数据库已创建",
        "- **L3（对话存档 - SQLite FTS5）**: 未创建",
        "- **Write Queue**: 空闲状态，无错误",
        "**Bridge**: 目录已配置，等待导出",
    ])
    def test_decorated_metadata_rows_are_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text

    @pytest.mark.parametrize("text", [
        "方案二：文件夹权限控制",
        "方案三：Hermes .env 文件",
        "结论：使用 Tailscale 更简单",
        "注意：生产环境禁止直接改库",
    ])
    def test_prose_colons_are_kept(self, text):
        """A short key list must not swallow prose that merely contains '：'."""
        assert WriteQueue._looks_like_fact(text), text

    def test_undecorate_peels_bullets_and_symbols(self):
        assert _undecorate("- ⚠️ **Provider**: governed") == "**Provider**: governed"
        assert _undecorate("1. 第一步先备份") == "第一步先备份"
        assert _undecorate("**总结**：x") == "**总结**：x"  # bold preserved


# ---------------------------------------------------------------------------
# A. self-check boilerplate and outcome reports
# ---------------------------------------------------------------------------

class TestSelfCheckBoilerplate:
    @pytest.mark.parametrize("text", [
        "记忆系统自检报告 (hermes-memory-governed)",
        "以下是记忆系统的完整状态报告：",
        "| **L2 语义记忆** | ⏳ 待积累 | LanceDB 尚未创建 |",
        "Bug 修复成功",
        "`governed_health` 现在正常工作了",
        "现在可以直接使用：",
    ])
    def test_self_check_output_is_rejected(self, text):
        assert not WriteQueue._looks_like_fact(text), text


# ---------------------------------------------------------------------------
# B. signal scoring and role weighting
# ---------------------------------------------------------------------------

class TestSignalScore:
    def test_user_role_outranks_assistant_role(self):
        """The same words are knowledge when the USER says them."""
        sentence = "我需要本地部署的 SecretStore 能双向同步数据"
        assert (_fact_signal_score(sentence, "user")
                > _fact_signal_score(sentence, "assistant"))

    def test_role_weights_are_applied(self):
        base = _fact_signal_score("部署方案确定用 Docker Compose")
        assert _fact_signal_score("部署方案确定用 Docker Compose", "user") == (
            base + _ROLE_WEIGHTS["user"])
        assert _fact_signal_score("部署方案确定用 Docker Compose", "assistant") == (
            base + _ROLE_WEIGHTS["assistant"])

    def test_default_role_is_neutral(self):
        sentence = "我们决定用 PostgreSQL"
        assert _fact_signal_score(sentence) == _fact_signal_score(sentence, "")

    @pytest.mark.parametrize("text", [
        "我需要的是本地ai能读取，同时能同步给其它地方使用",
        "我家里可以正常访问和使用",
        "必须同步 rsa.key，否则两边加密的数据不兼容",
        "都太麻烦",
    ])
    def test_user_constraints_score_positive(self, text):
        assert _fact_signal_score(text) > 0, text

    def test_positive_markers_outrank_generic_prose(self):
        constraint = "太麻烦，我需要的是本地ai能读取"
        prose = "这是一个方案，这个方案可以使用 Docker Compose"
        assert _fact_signal_score(constraint) > _fact_signal_score(prose)

    def test_soft_wording_is_penalised(self):
        assert _fact_signal_score("可以试试用 Syncthing") < _fact_signal_score(
            "用 Syncthing")


class TestRoleWeightedExtraction:
    def test_user_message_survives_the_cap_over_assistant(self, config):
        """Same sentence, two roles: with cap=1 the user version must win."""
        config.sync.l2_max_facts_per_turn = 1
        queue = WriteQueue(config)
        sentence = "我需要 SecretStore 在本地通过 Bitwarden CLI 读取"
        facts = queue._extract_atomic_facts([
            {"role": "assistant", "content": sentence},
            {"role": "user", "content": sentence},
        ])
        assert len(facts) == 1
        assert facts[0]["content"] == sentence

    def test_user_constraints_survive_over_assistant_prose(self, config):
        config.sync.l2_max_facts_per_turn = 1
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts([
            {"role": "assistant", "content": "这个可以考虑使用下面的方案来处理"},
            {"role": "user", "content": "都太麻烦，我需要的是本地ai能读取"},
        ])
        assert len(facts) == 1
        assert "我需要" in facts[0]["content"]

    def test_signals_key_is_not_leaked_into_facts(self, config):
        """'_signals' is sort metadata and must be stripped before returning."""
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts([
            {"role": "user", "content": "我们决定用 PostgreSQL 作为主数据库"},
        ])
        assert facts
        assert all("_signals" not in fact for fact in facts)

    def test_assistant_short_replies_are_filtered_when_role_known(self):
        """Outcome reports need the role to be recognised reliably."""
        assert not WriteQueue._looks_like_fact("Bug 修复成功", "assistant")
        # Identical text without a role stays subject to the content rules.
        assert not WriteQueue._looks_like_fact("Bug 修复成功")

    def test_assistant_fact_still_extracts_when_it_has_content(self, config):
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts([
            {"role": "assistant",
             "content": "`governed_health` 工具报错是因为引用了未定义的变量 `total`"},
        ])
        assert len(facts) == 1

    def test_role_agnostic_gate_still_rejects_process_narration(self, config):
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts([
            {"role": "assistant", "content": "让我检查一下记忆系统的状态"},
        ])
        assert facts == []


# ---------------------------------------------------------------------------
# Regression: the pre-existing gate must not change behaviour
# ---------------------------------------------------------------------------

class TestPreexistingBehaviour:
    @pytest.mark.parametrize("text", [
        "我们决定用 PostgreSQL 作为主数据库",
        "部署方案确定用 Docker Compose",
        "以后都用 pnpm 而不是 npm",
        "记住我的偏好是深色主题",
        "我们用 Postgres",
        "这个项目的规则是所有接口必须带版本号",
        "I always prefer dark mode in every editor",
        "The project deadline is next Friday",
        "We decided to use Postgres instead of MySQL",
    ])
    def test_known_facts_still_pass(self, text):
        assert WriteQueue._looks_like_fact(text), text

    @pytest.mark.parametrize("text", [
        "你好",
        "谢谢",
        "ok",
        "帮我看一下这个报错",
        "请帮我重构这个函数",
        "What is the deployment deadline?",
        "这个方案可以吗？",
        "Traceback (most recent call last):",
        "SELECT * FROM messages WHERE id = 1",
        "import sqlite3",
        "https://example.com/docs",
        "TODO",
        "x" * 501,
    ])
    def test_known_noise_still_fails(self, text):
        assert not WriteQueue._looks_like_fact(text), text

    def test_facts_ranked_by_signal_strength_is_unchanged(self, config):
        config.sync.l2_max_facts_per_turn = 1
        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts([
            {"role": "user", "content": "The weather is nice today in the park"},
            {"role": "user", "content": "We decided to use PostgreSQL for the store"},
        ])
        assert len(facts) == 1
        assert "decided" in facts[0]["content"]


# ---------------------------------------------------------------------------
# Constants sanity (guards against a future "cleanup" widening the rules)
# ---------------------------------------------------------------------------

class TestThresholdSanity:
    def test_user_weight_is_positive_and_assistant_negative(self):
        assert _ROLE_WEIGHTS["user"] > 0 > _ROLE_WEIGHTS["assistant"]

    def test_title_fragment_rule_is_narrow_enough(self):
        """'记住我的偏好是深色主题' is 33 weighted units — just under the cap,
        but it carries a user marker and must not be treated as a heading."""
        assert _TITLE_FRAGMENT_MAX_WEIGHT == 36
        assert WriteQueue._looks_like_fact("记住我的偏好是深色主题")

    def test_assistant_length_floor_exists(self):
        assert _ASSISTANT_MIN_WEIGHTED_LEN > 0
