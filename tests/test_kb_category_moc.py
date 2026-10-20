# -*- coding: utf-8 -*-
"""受控分类（``category``）与 MOC 导航。

背景：库内 20+ 篇笔记全挤在 ``notes/`` 一个目录里，``tags`` 是中英混杂的自由
文本（``["nas", "硬件", "飞牛", "i3-N305"]`` 里既有概念又有实例还有语言），
既不能被人预测也不能被机器判定归属 —— 找东西只能靠搜索。

做法借鉴社区长期争论后的收敛结论：纯文件夹会造出第二套与链接结构竞争的
并行分类法，纯 tag 不可预测且无法浏览；可维护的是「一个受控字段定类目 +
自动 MOC 按类目建索引」。本模块钉住：

* 词表是封闭集合，非法值归一成 ""（绝不硬猜、绝不抛）；
* 推断有优先级，且优先级经过真实库标定（``_综合归纳`` 必须判 synthesis）；
* CLI 与插件两侧词表/关键词表**同源**（一个入口一种分类 = 两个真相）；
* MOC 按类目分组、归档区不参与、未分类显式列出；
* 迁移默认 dry-run，已有合法分类的不动，推断不出的不写。
"""

from __future__ import annotations

import json

import pytest

from plugin.memory_governed import _vault
from plugin.memory_governed._kb import KnowledgeBase
from plugin.memory_governed._config import GovernedMemoryConfig

import memory_cli


# ---------------------------------------------------------------------------
# 词表封闭性
# ---------------------------------------------------------------------------

class TestControlledVocabulary:
    def test_every_category_is_a_plain_lowercase_slug(self):
        for c in _vault.KB_CATEGORIES:
            assert c == c.lower()
            assert " " not in c

    def test_no_duplicates(self):
        assert len(_vault.KB_CATEGORIES) == len(set(_vault.KB_CATEGORIES))

    @pytest.mark.parametrize("bad", ["", "  ", "NAS", "environments", None, 7,
                                     ["environment"], {"a": 1}])
    def test_illegal_values_normalize_to_empty(self, bad):
        """坏分类归一成 ""，而不是抛异常或猜一个塞回去。"""
        assert _vault.normalize_category(bad) == ""

    @pytest.mark.parametrize("good", ["environment", " ENVIRONMENT ",
                                      "Tooling", "meta"])
    def test_legal_values_survive_case_and_space(self, good):
        assert _vault.normalize_category(good) == good.strip().lower()

    def test_cli_and_plugin_share_one_vocabulary(self):
        """两个入口必须给出同一套词表，否则同一篇笔记有两个真相。"""
        assert memory_cli.KB_CATEGORIES == _vault.KB_CATEGORIES
        assert memory_cli._CATEGORY_HINTS == _vault._CATEGORY_HINTS


# ---------------------------------------------------------------------------
# 推断
# ---------------------------------------------------------------------------

class TestInference:
    @pytest.mark.parametrize("title,tags,expected", [
        # 真实库里的笔记：优先级必须让强信号赢。
        ("hermes_综合归纳", ["自动化"], "synthesis"),
        ("workflow_综合归纳", ["工作流"], "synthesis"),
        ("共享记忆系统开通", ["memory", "setup"], "meta"),
        ("飞牛 NAS 硬件配置", ["nas", "硬件", "飞牛"], "environment"),
        ("本地 PC 硬件与系统环境", ["环境", "硬件", "PC"], "environment"),
        ("用户工作流偏好：先讨论方向再部署", ["偏好", "工作流"], "user"),
        ("browser-skill (bsk) 安装状态与 Windows 升级坑",
         ["browser-skill", "installation"], "tooling"),
        ("MiMo API 端点能力", ["mimo", "api"], "reference"),
        ("联想 ThinkServer SR588 服务器 CPU 支持规格",
         ["服务器", "硬件", "cpu"], "reference"),
    ])
    def test_real_notes_land_where_a_human_would_put_them(
            self, title, tags, expected):
        assert _vault.infer_category(title, tags) == expected

    def test_nothing_to_go_on_returns_empty_not_a_guess(self):
        assert _vault.infer_category("", [], []) == ""

    def test_body_is_only_a_last_resort(self):
        """title/tags 都没信号时才看正文，且只看前 400 字。"""
        assert _vault.infer_category("", [], [], "x" * 500 + " nas 硬件") == ""
        assert _vault.infer_category("", [], [], "nas 硬件配置") == "environment"

    def test_english_tags_are_case_insensitive(self):
        assert _vault.infer_category("", ["NAS", "Hardware"]) == "environment"


class TestWordBoundaryMatching:
    """ASCII 关键词必须走词边界，否则英文笔记大面积误分类。

    裸子串匹配下 ``script`` / ``recipe`` / ``skip`` / ``pipeline`` /
    ``shipping`` 都因为含 "ip" 被判成 environment，``capital`` /
    ``apiary`` 因为含 "api" 被判成 tooling —— 而这些笔记跟环境、工具
    毫无关系。
    """

    @pytest.mark.parametrize("word", [
        "script", "recipe", "skip", "pipeline", "shipping", "capital",
        "apiary", "ubuntu", "capitalize", "spaceship",
    ])
    def test_a_substring_is_not_a_match(self, word):
        got = _vault.infer_category("", [word])
        assert got in ("", "project"), (
            "%r 不该凭子串命中 environment/tooling，实际判成 %r"
            % (word, got))

    @pytest.mark.parametrize("word,expected", [
        ("api", "tooling"), ("ip", "environment"), ("nas", "environment"),
        ("ssh", "environment"), ("vpn", "environment"),
    ])
    def test_the_word_itself_still_matches(self, word, expected):
        assert _vault.infer_category("", [word]) == expected

    def test_a_real_word_containing_a_keyword_is_not_dragged_in(self):
        """``spacecraft`` 含 ``craft`` 不含 ``cpu``；这条盯的是反向。"""
        assert _vault.infer_category("Spacecraft notes", []) in ("", "project")


class TestTitleMarkerWins:
    """标题标志词优先于关键词加权。

    判据是标题的**构词法**而不是"哪个词更强"：中文笔记标题是
    ``<主体>_<属性>`` 结构，结尾的属性词说明这篇笔记是什么类型。
    """

    @pytest.mark.parametrize("title,tags,expected", [
        # 同一批词摆在不同位置，含义完全不同：
        ("hermes_综合归纳", ["自动化"], "synthesis"),
        ("workflow_综合归纳", ["工作流"], "synthesis"),
        ("ZCode provider kind 配置", ["zcode", "config"], "tooling"),
        ("飞牛 NAS 硬件配置", ["nas", "硬件"], "environment"),
    ])
    def test_marker_beats_the_keyword_channel(self, title, tags, expected):
        assert _vault.infer_category(title, tags) == expected

    def test_longest_marker_wins(self):
        """``硬件配置``(4 字) 必须压过 ``配置``(2 字)。"""
        assert _vault.infer_category("飞牛 NAS 硬件配置", []) == "environment"
        assert _vault.infer_category("某工具配置", []) == "tooling"

    def test_ambiguous_marker_only_counts_at_the_tail(self):
        """``配置`` 在句中只是内容描述，只有结尾才构成类型声明。"""
        assert _vault.infer_category("配置管理的心得", []) == ""
        assert _vault.infer_category("Octopus 配置", []) == "tooling"

    def test_marker_needs_a_real_title(self):
        assert _vault._title_marker_category("") == ""
        assert _vault._title_marker_category("   ") == ""



# ---------------------------------------------------------------------------
# 写入路径
# ---------------------------------------------------------------------------

def _write_legacy_note(kb, title, body):
    """按老版本（没有 category 的年代）的形态写一篇笔记。

    迁移面对的就是这批存量件：frontmatter 里根本没有 ``category`` 键。
    直接调 ``_vault.write_note`` 绕开 ``add()`` 的推断，才能造出这个缺口。
    """
    path = kb._vault / "notes" / (title + ".md")
    _vault.write_note(path, {"title": title, "type": "notes",
                             "status": "curated"}, body)
    return path


@pytest.fixture
def kb(tmp_path):
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "l2")
    _vault.ensure_skeleton(_vault.vault_root(cfg))
    return KnowledgeBase(cfg)


class TestWritePath:
    def test_explicit_category_is_kept(self, kb):
        r = kb.add("某篇笔记", "正文内容", category="tooling")
        assert r.get("ok") is True
        meta, _ = _vault.read_note(
            kb._vault / "notes" / "某篇笔记.md")
        assert meta.get("category") == "tooling"

    def test_illegal_explicit_category_falls_back_to_inference(self, kb):
        r = kb.add("飞牛 NAS 硬件配置", "正文", category="not-a-category")
        assert r.get("ok") is True
        meta, _ = _vault.read_note(
            kb._vault / "notes" / "飞牛 NAS 硬件配置.md")
        # 非法值被丢弃后走推断，而不是把脏值写进 frontmatter。
        assert meta.get("category") == "environment"

    def test_uninferable_note_gets_no_category_key_at_all(self, kb):
        """空分类会让 MOC 出现一个名为 "" 的桶 —— 整个键不写更好。"""
        r = kb.add("无题", "完全没有任何信号的一句话")
        assert r.get("ok") is True
        meta, _ = _vault.read_note(kb._vault / "notes" / "无题.md")
        assert "category" not in meta


# ---------------------------------------------------------------------------
# 检索侧：分类必须真的能用来筛
# ---------------------------------------------------------------------------

class TestCategoryFiltersRetrieval:
    """分类不能只活在 index.md 里 —— 「想按类目找」是它存在的理由。"""

    def _seed(self, kb):
        kb.add("飞牛 NAS 硬件配置", "nas 硬件 内存 存储", category="environment")
        kb.add("ZCode provider kind 配置", "zcode config provider",
               category="tooling")
        kb.add("用户工作流偏好", "偏好 工作流", category="user")

    def test_category_filter_narrows_results(self, kb):
        self._seed(kb)
        hits = kb.search("配置", top_k=10, category="tooling")
        assert hits, "按类目筛必须能筛出东西"
        assert all(h["path"].startswith("notes/") for h in hits)
        titles = {h["title"] for h in hits}
        assert "ZCode provider kind 配置" in titles
        assert "飞牛 NAS 硬件配置" not in titles

    def test_illegal_category_returns_nothing_rather_than_everything(self, kb):
        """静默忽略过滤条件 = 「我按 tooling 筛却拿到全部」。"""
        self._seed(kb)
        assert kb.search("配置", top_k=10, category="not-a-category") == []

    def test_empty_category_is_not_a_filter(self, kb):
        self._seed(kb)
        unfiltered = kb.search("配置", top_k=10)
        assert len(unfiltered) >= 2

    def test_category_is_in_the_keyword_channel(self, kb):
        """类目名本身作为一个词参与关键词打分。"""
        kb.add("某篇无关笔记", "完全不相干的内容", category="tooling")
        note = [n for n in kb._iter_notes() if n.title == "某篇无关笔记"][0]
        assert "tooling" in note.short_text.lower()

    def test_section_filter_is_not_bypassed_by_the_semantic_channel(self, kb):
        """回归：过滤必须作用在**两路并集**上，不只关键词候选。

        语义通道按索引返回路径，可能带进 ``notes`` 列表之外的笔记（索引
        投影滞后、笔记被移动）。只在关键词候选上过滤时，``section="inbox"``
        仍会返回 ``notes/`` 里的笔记 —— 门形同虚设。
        """
        self._seed(kb)
        for section in ("notes", "inbox", "archive", "areas"):
            hits = kb.search("配置", top_k=10, section=section)
            for h in hits:
                assert h["section"] == section, (
                    "section=%r 却返回了 %r（来自 %s）"
                    % (section, h["section"], h["path"]))

    def test_category_name_survives_the_reranker(self, kb):
        """类目名是合法关键词，但 reranker 只看正文会把它筛掉。

        回归：``kb-search "tooling" --category tooling`` 曾返回 0 ——
        关键词通道给了 1.0 分，reranker 却因为 "tooling" 不在正文里把它
        当噪声丢弃。类目名必须和正文一起提交给 reranker。
        """
        kb._config.reranking.enabled = True
        kb._config.kb.rerank_min_score = 0.05
        kb.add("某工具安装说明", "安装步骤与排障", category="tooling")
        kb.add("某环境事实", "服务器硬件", category="environment")

        class _R:
            available = True

            def scores(self, query, texts, top_n=None):
                return {i: (0.9 if query in t else 0.01)
                        for i, t in enumerate(texts)}

        kb._get_reranker = lambda: _R()  # type: ignore[assignment]
        hits = kb.search("tooling", top_k=5, category="tooling")
        assert hits, "类目名检索被 reranker 整条筛掉了"
        assert all(h["category"] == "tooling" for h in hits)

    def test_section_and_category_are_independent_gates(self, kb):
        self._seed(kb)
        assert kb.search("配置", top_k=10, section="notes",
                         category="tooling")
        assert kb.search("配置", top_k=10, section="inbox",
                         category="tooling") == []

    def test_cli_search_accepts_and_honours_category(self, kb):
        self._seed(kb)
        # CLI 从 ``config["wiki_dir"]`` 取要搜的库（它必须搜调用方指的库，
        # 而不是默认的 HERMES_HOME vault），所以这里传真实路径。
        cfg = {"wiki_dir": str(kb._vault)}
        errs: list = []
        hits = memory_cli.cmd_kb_search(cfg, "配置", 10, "", errs,
                                       category="tooling")
        assert hits, "CLI 侧也要能按类目筛"
        assert all(h.get("category") == "tooling" for h in hits)

    def test_cli_refuses_an_illegal_category_by_name(self, monkeypatch):
        errs: list = []
        assert memory_cli.cmd_kb_search({}, "x", 10, "", errs,
                                        category="bogus") == []
        assert any("not a controlled category" in e for e in errs)


# ---------------------------------------------------------------------------
# MOC
# ---------------------------------------------------------------------------

class TestMoc:
    def _seed(self, kb):
        kb.add("飞牛 NAS 硬件配置", "nas 硬件", category="environment")
        kb.add("共享记忆系统开通", "hermes 记忆系统", category="meta")
        kb.add("无题", "没有任何信号的一句话")

    def test_groups_by_category_and_lists_the_ungrouped(self, kb):
        self._seed(kb)
        r = kb.rebuild_moc()
        assert r["ok"] is True
        text = (kb._vault / "index.md").read_text(encoding="utf-8")
        assert "## environment" in text
        assert "## meta" in text
        assert "[[飞牛 NAS 硬件配置]]" in text
        # 未分类的必须显式出现，而不是悄悄消失。
        assert "## Uncategorized" in text
        assert "[[无题]]" in text
        assert r["uncategorized"] == 1

    def test_archive_is_not_a_category(self, kb):
        self._seed(kb)
        kb.add("旧笔记", "nas 硬件 旧版", category="environment",
               section="archive")
        r = kb.rebuild_moc()
        text = (kb._vault / "index.md").read_text(encoding="utf-8")
        assert "[[旧笔记]]" not in text
        assert "## archive" not in text

    def test_nested_archive_is_excluded_too(self, kb):
        """``archive/superseded/`` 下的笔记父目录叫 superseded。

        用目录名判归档会漏掉这一层，于是被 supersede 的旧笔记混进导航 ——
        实测库内正是这个形态（``archive/superseded/飞牛 NAS 硬件与网络``）。
        判据必须是 frontmatter 的 ``status``。
        """
        self._seed(kb)
        nested = kb._vault / "archive" / "superseded" / "旧版 NAS.md"
        nested.parent.mkdir(parents=True, exist_ok=True)
        _vault.write_note(nested, {"title": "旧版 NAS", "status": "archived",
                                   "tags": ["nas", "硬件"]}, "nas 硬件 旧版")
        r = kb.rebuild_moc()
        text = (kb._vault / "index.md").read_text(encoding="utf-8")
        assert "[[旧版 NAS]]" not in text
        assert "[[旧笔记]]" not in text

    def test_migrate_leaves_nested_archive_alone(self, kb):
        nested = kb._vault / "archive" / "superseded" / "旧版 NAS.md"
        nested.parent.mkdir(parents=True, exist_ok=True)
        _vault.write_note(nested, {"title": "旧版 NAS", "status": "archived",
                                   "tags": ["nas", "硬件"]}, "nas 硬件 旧版")
        r = kb.migrate_categories(apply=True)
        assert r["proposed"] == []
        assert r["written"] == []

    def test_rebuild_is_idempotent(self, kb):
        self._seed(kb)
        a = kb.rebuild_moc()
        first = (kb._vault / "index.md").read_text(encoding="utf-8")
        b = kb.rebuild_moc()
        assert (kb._vault / "index.md").read_text(encoding="utf-8") == first
        assert a["categories"] == b["categories"]

    def test_index_md_is_not_itself_a_note(self, kb):
        """MOC 是导航脚手架，不能把自己算成一篇被分类的笔记。"""
        self._seed(kb)
        kb.rebuild_moc()
        titles = [n.title for n in kb._iter_notes()]
        assert "知识库导航 (MOC)" not in titles


# ---------------------------------------------------------------------------
# 迁移
# ---------------------------------------------------------------------------

class TestMigrate:
    def test_dry_run_reports_without_writing(self, kb):
        # add() 现在自己就会写 category，所以迁移的缺口要用"写盘时绕过分
        # 类推断"的方式造出来 —— 这正是存量笔记的形态（老版本写的）。
        _write_legacy_note(kb, "飞牛 NAS 硬件配置", "nas 硬件")
        r = kb.migrate_categories(apply=False)
        assert r["applied"] is False
        assert r["written"] == []
        assert [p["category"] for p in r["proposed"]] == ["environment"]
        meta, _ = _vault.read_note(
            kb._vault / "notes" / "飞牛 NAS 硬件配置.md")
        assert "category" not in meta, "dry-run must not touch frontmatter"

    def test_apply_writes_only_what_it_proposed(self, kb):
        _write_legacy_note(kb, "飞牛 NAS 硬件配置", "nas 硬件")
        _write_legacy_note(kb, "无题", "没有任何信号的一句话")
        r = kb.migrate_categories(apply=True)
        assert len(r["written"]) == 1
        assert len(r["skipped"]) == 1
        meta, _ = _vault.read_note(
            kb._vault / "notes" / "飞牛 NAS 硬件配置.md")
        assert meta["category"] == "environment"
        meta2, _ = _vault.read_note(kb._vault / "notes" / "无题.md")
        assert "category" not in meta2, "a guess must never be written"

    def test_a_humans_category_is_never_overwritten(self, kb):
        """人的判断压过机器的推断 —— 已合法分类的笔记一律不动。"""
        kb.add("飞牛 NAS 硬件配置", "nas 硬件", category="reference")
        r = kb.migrate_categories(apply=True)
        assert r["proposed"] == []
        meta, _ = _vault.read_note(
            kb._vault / "notes" / "飞牛 NAS 硬件配置.md")
        assert meta["category"] == "reference"

    def test_archive_is_left_alone(self, kb):
        kb.add("旧笔记", "nas 硬件", section="archive")
        r = kb.migrate_categories(apply=True)
        assert r["proposed"] == []
        assert r["written"] == []
