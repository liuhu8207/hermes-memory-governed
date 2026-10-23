# -*- coding: utf-8 -*-
"""L2 quality gate + periodic sweep wiring."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from plugin.memory_governed._quality import admission_rejection, classify_fact
from plugin.memory_governed._sync import external_write_verdict


def test_test_residue_is_flagged_and_rejected():
    text = "自动归档测试：agent codex 在 2026-10-19 通过 hgm_remember 自动归档对话到 L3"
    verdict = classify_fact(text)
    assert verdict and verdict["kind"] == "test_residue"
    assert admission_rejection(text) == "test_residue"
    ok, reason = external_write_verdict(text)
    assert ok is False and reason == "test_residue"


def test_plan_test_result_is_flagged():
    assert classify_fact("方案2上线：hgm_remember 和 hgm_kb_add 自动归档对话到 L3")["kind"] \
        == "test_residue"


def test_assistant_fragment_is_flagged_and_rejected():
    text = "最后追问当年读取慢的具体场景（大文件顺序读 / 大量小文件）"
    assert classify_fact(text)["kind"] == "assistant_fragment"
    assert admission_rejection(text) == "assistant_fragment"


def test_assistant_status_report_is_rejected_only_with_assistant_context():
    text = ("本轮三个问题全部修复且均为根因修复：① cron 4/9 失败→9/9 全绿，"
            "测试 2032 passed")
    verdict = classify_fact(text, role="assistant", source="auto-extract")
    assert verdict and verdict["kind"] == "assistant_fragment"
    assert admission_rejection(text, role="assistant") == "assistant_fragment"
    # The same prose from an external fact writer is not rejected by text alone.
    assert classify_fact(text) is None


def test_a_real_fact_is_not_flagged():
    text = "Hermes Gateway 重启必须用 schtasks /run /tn Hermes_Gateway 计划任务"
    assert classify_fact(text) is None
    ok, reason = external_write_verdict(text)
    assert ok is True, reason


def test_stale_marker_is_reported_not_rejected():
    text = "公开仓库路径泄漏尚未修复，已确认随 origin/main 推送"
    assert classify_fact(text)["kind"] == "stale_marker"
    assert admission_rejection(text) is None


def test_cron_runs_the_quality_sweep():
    path = Path(__file__).resolve().parent.parent / "scripts" / "cron_memory_daily.py"
    spec = importlib.util.spec_from_file_location("cron_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod          # dataclasses 需要模块已注册
    spec.loader.exec_module(mod)
    steps = {step.label: step for step in mod.STEPS}
    assert "l2-quality" in steps
    assert "--apply" in steps["l2-quality"].args
    assert "test_residue" in steps["l2-quality"].args
