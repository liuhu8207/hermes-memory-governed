# -*- coding: utf-8 -*-
"""护栏棘轮：``memory_cli.py`` 与 ``scripts/`` 的静默异常不得继续增长。

``plugin/`` 内是**零容忍**（整包必须为 0）。CLI 与脚本历史存量很大
（201 处静默、51 处急切字符串化），一次归零不现实。

这里用**按文件**冻结的棘轮：
* 任何文件的计数不得超过它自己的基线（防止「这里删一处、那里加一处」腾挪）；
* 基线只允许下调；
* 新增站点必须带 ``# silent-ok:`` 标记或改成可见降级。

基线冻结于 2026-10-20（复评第二轮）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_no_eager_exception_format import _scan as eager_scan
from tests.test_recall_degraded_visibility import _violations as silent_scan

REPO = Path(__file__).resolve().parent.parent

#: file -> (silent, eager)。只列有违规的文件；未列出的文件基线是 (0, 0)。
BASELINE = {
    "memory_cli.py": (75, 35),
    "chat_extractor.py": (2, 0),
    "clipper_sync.py": (2, 0),
    "cron_memory_daily.py": (2, 1),
    "dashboard_gen.py": (3, 0),
    "feishu_sync.py": (4, 0),
    "hermes_env.py": (2, 0),
    "hgm_mcp.py": (12, 4),
    "kb_frontmatter_migrate.py": (5, 3),
    "knowledge_collector.py": (4, 0),
    "knowledge_graph.py": (1, 0),
    "knowledge_links.py": (1, 0),
    "knowledge_processor.py": (3, 0),
    "knowledge_sync.py": (5, 1),
    "l2_capacity.py": (1, 1),
    "l2_rebuild.py": (6, 0),
    "l3_retention.py": (2, 0),
    "l4_persona_daily.py": (7, 0),
    "meeting_recap.py": (4, 0),
    "memory_dream.py": (4, 0),
    "memory_health_report.py": (6, 3),
    "memory_pipeline.py": (1, 0),
    "memory_promote.py": (2, 0),
    "obsidian_sync.py": (5, 1),
    "redact_l3_archive.py": (1, 0),
    "scope_recall_bridge.py": (7, 0),
    "wb_pretool_hook.py": (6, 1),
    "wb_prompt_hook.py": (9, 0),
    "wb_session_hook.py": (10, 0),
    "wiki_nblm_daily.py": (1, 0),
    "wiki_nblm_induction.py": (3, 0),
    "wiki_utils.py": (5, 1),
}


def _targets():
    return [REPO / "memory_cli.py"] + sorted((REPO / "scripts").glob("*.py"))


def _counts():
    """返回 ``{filename: (silent, eager)}``，包含零违规文件。"""
    out = {}
    for path in _targets():
        text = path.read_text(encoding="utf-8")
        out[path.name] = (len(silent_scan(text, filename=path.name)),
                          len(eager_scan(text)))
    return out


def test_the_scan_actually_reads_files():
    files = _targets()
    assert len(files) > 20, "扫描目标为空或路径写错，棘轮会恒绿"
    assert (REPO / "memory_cli.py").exists()


def test_no_file_exceeds_its_own_baseline():
    counts = _counts()
    offenders = []
    for name, (silent, eager) in sorted(counts.items()):
        base_s, base_e = BASELINE.get(name, (0, 0))
        if silent > base_s:
            offenders.append("%s: silent %d > baseline %d" % (name, silent, base_s))
        if eager > base_e:
            offenders.append("%s: eager %d > baseline %d" % (name, eager, base_e))
    assert not offenders, (
        "静默吞异常 / 急切字符串化超过了**该文件**的基线。新增站点请带 "
        "`# silent-ok: already-reported — <理由>` 或改成 log_degraded / raise。\n  "
        + "\n  ".join(offenders))


def test_baselines_are_not_stale():
    """基线只许下调：如果某文件已经低于基线，请同步收紧数字。"""
    counts = _counts()
    stale = []
    for name, (base_s, base_e) in BASELINE.items():
        silent, eager = counts.get(name, (0, 0))
        if silent < base_s or eager < base_e:
            stale.append("%s: now (%d,%d) vs baseline (%d,%d)"
                         % (name, silent, eager, base_s, base_e))
    assert not stale, (
        "基线落后于实际；请把 BASELINE 收紧到当前值：\n  " + "\n  ".join(stale))


def test_the_ratchet_catches_a_new_violation(tmp_path, monkeypatch):
    """正对照：证明棘轮真的会红，而不是只会恒绿。"""
    import tests.test_guardrail_ratchet as self_mod

    bad = tmp_path / "bad.py"
    bad.write_text(
        "def f():\n"
        "    try:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        pass\n",
        encoding="utf-8")
    monkeypatch.setattr(self_mod, "_targets", lambda: [bad])
    monkeypatch.setattr(self_mod, "BASELINE", {})
    counts = self_mod._counts()
    assert counts["bad.py"][0] > 0
