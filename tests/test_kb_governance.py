# -*- coding: utf-8 -*-
"""双库分工治理测试（Phase 2 / WeKnora 借鉴）。

覆盖五件事：

1. ``status`` 契约：CLI 与插件两侧 ``_status_for_section`` 逐 section 一致
   （一个存储只能有一个答案），写入路径（插件 add / CLI kb-add / relocate）
   都落 status；
2. 治理巡检 ``governance()``：晋升建议（命中 ≥ promote_hits 标
   ``promote_suggest``）、自动归档（自动区闲置 > auto_archive_days 移入
   archive，降级不删除）、dry-run 默认不写盘、无使用度数据时 fail-closed；
3. 包含性查重：新正文与既有笔记互相包含且不琐碎 → 拒绝并给既有路径；
4. CLI ``cmd_kb_govern`` 能出 JSON verdict；
5. 存量迁移脚本 ``kb_frontmatter_migrate``：dry-run 计数、--apply 补
   status、幂等。
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path

import memory_cli
from plugin.memory_governed import KB_GOV_SCHEMA
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase, _status_for_section

_LONG_BODY = (
    "部署注意事项：所有服务必须先在 staging 环境验证通过之后才能发布到"
    "生产环境，发布窗口固定在周二和周四的下午，回滚预案需要提前一天写好"
    "并经过值班负责人确认；灰度期间需要监控错误率与延迟指标，任何异常"
    "超过阈值都要立即停止发布并回滚到上一个稳定版本。这是一段足够长的"
    "正文，用于触发包含性查重的最小长度门槛。"
)


def _make_cfg(tmp_path: Path) -> GovernedMemoryConfig:
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    return cfg


def _make_kb(tmp_path: Path, notes: dict[str, str] | None = None) -> KnowledgeBase:
    cfg = _make_cfg(tmp_path)
    for rel, body in (notes or {}).items():
        p = Path(cfg.wiki_dir) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    kb = KnowledgeBase(cfg)
    kb.ensure(sync=False)
    return kb


# ---------------------------------------------------------------------------
# 1) status 契约：两侧一致 + 写入路径落 status
# ---------------------------------------------------------------------------

class TestStatusContract:
    def test_cli_and_plugin_agree_per_section(self):
        """CLI 拷贝与插件实现必须逐 section 给出同一个答案。"""
        for section in ("inbox", "knowledge", "archive", "notes", "projects",
                        "areas", "resources", "", "运维", "(root)"):
            assert (memory_cli._status_for_section(section)
                    == _status_for_section(section))

    def test_expected_mapping(self):
        assert _status_for_section("inbox") == "draft"
        assert _status_for_section("knowledge") == "draft"
        assert _status_for_section("archive") == "archived"
        assert _status_for_section("notes") == "curated"
        assert _status_for_section("运维") == "curated"

    def test_plugin_add_writes_status(self, tmp_path):
        kb = _make_kb(tmp_path)
        assert kb.add("正库笔记", "正文", section="notes")["ok"]
        assert kb.add("待审笔记", "正文", section="inbox")["ok"]
        assert kb.get("正库笔记")["meta"]["status"] == "curated"
        assert kb.get("待审笔记")["meta"]["status"] == "draft"

    def test_cli_kb_add_writes_status(self, tmp_path):
        cfg = {"wiki_dir": str(tmp_path / "wiki")}
        out = memory_cli.cmd_kb_add(cfg, "CLI笔记", "正文", "inbox",
                                    [], [], None, agent="tester")
        assert out.get("ok") is not False, out
        meta, _ = memory_cli.parse_frontmatter(
            (Path(cfg["wiki_dir"]) / "inbox" / "CLI笔记.md").read_text(
                encoding="utf-8"))
        assert meta["status"] == "draft"

    def test_approve_promotes_status(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.add("晋升对象", "待审正文", section="inbox")
        r = kb.approve("晋升对象")
        assert r["ok"], r
        note = kb.get("晋升对象")
        assert note["meta"]["status"] == "curated"
        assert "review_required" not in note["meta"]

    def test_reject_archives_status(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.add("拒绝对象", "不要的内容", section="inbox")
        r = kb.reject("拒绝对象")
        assert r["ok"], r
        note = kb.get("拒绝对象")
        assert note["meta"]["status"] == "archived"
        assert note["meta"].get("review_rejected") is True

    def test_knowledge_section_now_in_write_whitelist(self, tmp_path):
        """knowledge 进白名单：kb-add --section knowledge 不再回落 inbox。"""
        kb = _make_kb(tmp_path)
        r = kb.add("自动区笔记", "采集来的知识", section="knowledge")
        assert r["ok"] and r["section"] == "knowledge"
        assert kb.get("自动区笔记")["meta"]["status"] == "draft"


# ---------------------------------------------------------------------------
# 2) 治理巡检
# ---------------------------------------------------------------------------

class TestGovernance:
    def _note_rel(self, kb: KnowledgeBase, title: str) -> str:
        for n in kb._iter_notes():
            if n.title == title or n.path.stem == title:
                return kb._rel(n.path)
        raise AssertionError(f"note not found: {title}")

    def test_dry_run_reports_without_writing(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "inbox/候选.md": "---\ntitle: 候选\n---\n待审内容",
        })
        kb._config.kb.affinity_enabled = True
        rel = self._note_rel(kb, "候选")
        for _ in range(3):
            kb.touch([rel])
        report = kb.governance(apply=False)
        assert report["ok"] and report["applied"] is False
        assert [c["path"] for c in report["promote_suggest"]] == [rel]
        # dry-run：没有写盘
        assert "promote_suggest" not in kb.get("候选")["meta"]
        assert report["flagged"] == []

    def test_apply_flags_promote_suggest(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "inbox/候选.md": "---\ntitle: 候选\n---\n待审内容",
        })
        kb._config.kb.affinity_enabled = True
        rel = self._note_rel(kb, "候选")
        for _ in range(3):
            kb.touch([rel])
        report = kb.governance(apply=True)
        assert report["flagged"] == [rel]
        assert kb.get("候选")["meta"]["promote_suggest"] is True
        # 幂等：再跑一次不会重复提议
        again = kb.governance(apply=True)
        assert again["promote_suggest"] == []

    def test_below_threshold_not_flagged(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "inbox/候选.md": "---\ntitle: 候选\n---\n待审内容",
        })
        kb._config.kb.affinity_enabled = True
        rel = self._note_rel(kb, "候选")
        kb.touch([rel])  # 1 次 < promote_hits 3
        assert kb.governance()["promote_suggest"] == []

    def test_idle_auto_zone_note_is_archivable_and_moves(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "knowledge/旧条目.md": "---\ntitle: 旧条目\n---\n很久没人用的知识",
            "inbox/别处.md": "---\ntitle: 别处\n---\n另一篇",
        })
        kb._config.kb.affinity_enabled = True
        # 表里必须先**有**使用度数据，否则 fail-closed 会整体跳过归档
        # （「没数据」≠「没被用过」—— 见 test_fail_closed_when_usage_table_is_empty）。
        kb.touch([self._note_rel(kb, "别处")])
        rel = self._note_rel(kb, "旧条目")
        old = time.time() - 50 * 86400
        os.utime(Path(cfg_path(kb, rel)), (old, old))

        report = kb.governance()
        assert [c["path"] for c in report["archivable"]] == [rel]
        assert report["archived"] == []  # dry-run 不移动

        applied = kb.governance(apply=True)
        # archived 返回的是**移动后**的路径（降级不删除，位置换了）
        assert applied["archived"] == ["archive/旧条目.md"]
        note = kb.get("旧条目")
        assert note["meta"]["status"] == "archived"
        assert note["meta"]["archived_reason"] == "auto-unused"
        # 降级不删除：正文还在（在 archive 区）
        assert "很久没人用的知识" in note["body"]

    def test_recently_used_note_is_not_archivable(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "knowledge/热知识.md": "---\ntitle: 热知识\n---\n常用内容",
        })
        kb._config.kb.affinity_enabled = True
        rel = self._note_rel(kb, "热知识")
        kb.touch([rel])  # 刚命中过 → last_used 是现在
        assert kb.governance()["archivable"] == []

    def test_fail_closed_without_usage_data(self, tmp_path):
        """affinity 关（无使用度数据）→ 归档评估整体跳过并说明原因。"""
        kb = _make_kb(tmp_path, {
            "knowledge/旧条目.md": "---\ntitle: 旧条目\n---\n内容",
        })
        old = time.time() - 50 * 86400
        os.utime(tmp_path / "wiki" / "knowledge" / "旧条目.md", (old, old))
        report = kb.governance(apply=True)
        assert report["usage_data"] is False
        assert report["archivable"] == [] and report["archived"] == []
        assert "fail-closed" in report["reason"]
        # 文件没被动过
        assert (tmp_path / "wiki" / "knowledge" / "旧条目.md").exists()

    def test_fail_closed_when_usage_table_is_empty(self, tmp_path):
        """affinity **开**、但表里一行都没有 ⇒ 与「关」同等对待，跳过归档。

        P0 回归（2026-09-22）：此前只判开关，不判表是否为空。于是「affinity 开着
        但还没攒到数据」时，第一次巡检会按 mtime 把整个自动区误清 —— 而本方法
        docstring 承诺的是「关**或表空**」，实现只做了前一半。
        真实 vault 今日半径碰巧为 0（自动区为空 + mtime 都是新的），但
        ``add()`` 落低置信笔记进 ``inbox/`` 是常规路径，两个条件都不是保证。
        """
        kb = _make_kb(tmp_path, {
            "knowledge/旧条目.md": "---\ntitle: 旧条目\n---\n内容",
        })
        kb._config.kb.affinity_enabled = True      # 开，但从没 touch 过
        old = time.time() - 50 * 86400
        os.utime(tmp_path / "wiki" / "knowledge" / "旧条目.md", (old, old))
        report = kb.governance(apply=True)
        assert report["usage_data"] is False, "空表必须算作「没有使用度数据」"
        assert report["archivable"] == [] and report["archived"] == []
        assert (tmp_path / "wiki" / "knowledge" / "旧条目.md").exists(), "文件不该被移走"

    def test_promote_suggest_clears_on_approve(self, tmp_path):
        kb = _make_kb(tmp_path, {
            "inbox/候选.md": "---\ntitle: 候选\n---\n待审内容",
        })
        kb._config.kb.affinity_enabled = True
        rel = self._note_rel(kb, "候选")
        for _ in range(3):
            kb.touch([rel])
        kb.governance(apply=True)
        assert kb.approve("候选")["ok"]
        assert "promote_suggest" not in kb.get("候选")["meta"]


def cfg_path(kb: KnowledgeBase, rel: str) -> Path:
    """相对 vault 路径 → 绝对路径（测试内工具）。"""
    return Path(kb._vault) / rel


# ---------------------------------------------------------------------------
# 3) 包含性查重（双写入路径）
# ---------------------------------------------------------------------------

class TestContainmentDedup:
    def test_plugin_refuses_duplicate_body(self, tmp_path):
        kb = _make_kb(tmp_path)
        assert kb.add("部署注意事项", _LONG_BODY, section="notes")["ok"]
        r = kb.add("发布流程提醒", _LONG_BODY, section="inbox")
        assert r["ok"] is False and r.get("duplicate") is True
        assert "notes" in r["path"]

    def test_plugin_allows_same_title_update(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.add("部署注意事项", _LONG_BODY, section="notes")
        r = kb.add("部署注意事项", _LONG_BODY + "补充一句修订。",
                   section="notes", update=True)
        assert r["ok"], r

    def test_plugin_ignores_short_bodies(self, tmp_path):
        kb = _make_kb(tmp_path)
        assert kb.add("确认甲", "好的收到", section="notes")["ok"]
        assert kb.add("确认乙", "好的收到", section="notes")["ok"]

    def test_cli_refuses_duplicate_body(self, tmp_path):
        cfg = {"wiki_dir": str(tmp_path / "wiki")}
        first = memory_cli.cmd_kb_add(cfg, "部署注意事项", _LONG_BODY,
                                      "notes", [], [], None, agent="tester")
        assert first.get("ok") is not False, first
        second = memory_cli.cmd_kb_add(cfg, "发布流程提醒", _LONG_BODY,
                                       "inbox", [], [], None, agent="tester")
        assert second["ok"] is False
        assert second.get("path", "").endswith("部署注意事项.md")


# ---------------------------------------------------------------------------
# 4) CLI kb-govern + MCP 工具
# ---------------------------------------------------------------------------

class TestGovernEntrypoints:
    def test_cli_kb_govern_returns_json_verdict(self, tmp_path):
        cfg = {"wiki_dir": str(tmp_path / "wiki"),
               "l2_db_path": str(tmp_path / "memory" / "l2")}
        (Path(cfg["wiki_dir"]) / "inbox").mkdir(parents=True)
        (Path(cfg["wiki_dir"]) / "inbox" / "x.md").write_text(
            "---\ntitle: x\n---\nbody", encoding="utf-8")
        out = memory_cli.cmd_kb_govern(cfg, apply=False)
        assert out["ok"] is True and out["applied"] is False

    def test_cli_kb_govern_survives_bad_config(self, tmp_path):
        cfg = {
            "wiki_dir": str(tmp_path / "does-not-exist"),
            # 显式给沙箱路径：CLI 现在会为**为空**的 l2_db_path 派生
            # ``$HERMES_HOME/memory/l2``（usage 库落在它旁边）。不给的话这条
            # 测试会去读真实存储 —— 测试必须自洽，不能依赖开发者的机器。
            "l2_db_path": str(tmp_path / "memory" / "l2"),
        }
        out = memory_cli.cmd_kb_govern(cfg, apply=False)
        # 空库不是错误：walk 返回空，报告照常
        assert out["ok"] is True

    def test_govern_schema_registered(self):
        assert KB_GOV_SCHEMA["name"] == "governed_kb_govern"
        from plugin.memory_governed import GovernedMemoryProvider
        # get_tool_schemas 不依赖实例状态 —— 直接在类上取 schema 列表
        names = [s["name"]
                 for s in GovernedMemoryProvider.get_tool_schemas(None)]
        assert "governed_kb_govern" in names
        assert names.count("governed_kb_govern") == 1


# ---------------------------------------------------------------------------
# 5) 存量迁移脚本
# ---------------------------------------------------------------------------

def _load_migrate_module():
    path = (Path(__file__).resolve().parents[1]
            / "scripts" / "kb_frontmatter_migrate.py")
    spec = importlib.util.spec_from_file_location("kb_frontmatter_migrate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: 迁移脚本测试的语料：三种 status 归属 + 一篇已有 status（跳过计数）。
_MIGRATE_FILES = {
    "notes/a.md": "---\ntitle: A\n---\n人工整理的笔记",
    "inbox/b.md": "---\ntitle: B\n---\n待审笔记",
    "knowledge/c.md": "---\ntitle: C\n---\n自动采集",
    "archive/d.md": "---\ntitle: D\nstatus: curated\n---\n已有状态",
}


class TestFrontmatterMigrate:
    def _vault(self, tmp_path: Path) -> dict:
        vault = tmp_path / "wiki"
        for rel, body in _MIGRATE_FILES.items():
            p = vault / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(body, encoding="utf-8")
        return {"wiki_dir": str(vault)}

    def test_dry_run_counts_without_writing(self, tmp_path):
        mod = _load_migrate_module()
        cfg = self._vault(tmp_path)
        would, already, errors = mod._plan(cfg)
        assert errors == []
        assert already == 1  # archive/d.md 已有 status
        assert len(would) == 3
        statuses = {item[0].name: item[3] for item in would}
        assert statuses == {"a.md": "curated", "b.md": "draft", "c.md": "draft"}
        # dry-run 不写盘
        assert "status" not in memory_cli.parse_frontmatter(
            (Path(cfg["wiki_dir"]) / "notes" / "a.md").read_text(
                encoding="utf-8"))[0]

    def test_apply_writes_and_is_idempotent(self, tmp_path):
        mod = _load_migrate_module()
        cfg = self._vault(tmp_path)
        would, _already, _errors = mod._plan(cfg)
        for path, meta, body, status, _section in would:
            mod._apply_one(path, meta, body, status)
        # 写入正确 + 原正文保留
        a_meta, a_body = memory_cli.parse_frontmatter(
            (Path(cfg["wiki_dir"]) / "notes" / "a.md").read_text(
                encoding="utf-8"))
        assert a_meta["status"] == "curated"
        assert "人工整理的笔记" in a_body
        # 幂等：再扫一遍没有待办
        would2, already2, _ = mod._plan(cfg)
        assert would2 == [] and already2 == 4
