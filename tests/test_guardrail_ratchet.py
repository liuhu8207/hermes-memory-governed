# -*- coding: utf-8 -*-
"""护栏棘轮：``memory_cli.py`` 与 ``scripts/`` 的静默异常不得继续增长。

``plugin/`` 内是**零容忍**（整包必须为 0）。CLI 与脚本是另一类代码 ——
它们直接面对终端用户和 cron，历史存量很大（202 处静默、52 处急切字符串化），
一次要求归零会把工作量和风险都推到不可控。

这里用**棘轮**：冻结当前基线，只允许减少、不允许增加。每一处新增都必须
带 ``# silent-ok:`` 标记或改写成可见降级，否则 CI 变红。这样盲区不会继续
变宽，也不必为了通过检查而做大范围重写。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_no_eager_exception_format import _scan as eager_scan
from tests.test_recall_degraded_visibility import _violations as silent_scan

REPO = Path(__file__).resolve().parent.parent

#: 冻结于 2026-10-20。只允许下调；下调后请同步更新这两个数字。
BASELINE_SILENT = 202
BASELINE_EAGER = 52


def _targets():
    return [REPO / "memory_cli.py"] + sorted((REPO / "scripts").glob("*.py"))


def _counts():
    silent = eager = 0
    detail = []
    for path in _targets():
        text = path.read_text(encoding="utf-8")
        s = silent_scan(text, filename=path.name)
        e = eager_scan(text)
        silent += len(s)
        eager += len(e)
        if s or e:
            detail.append((path.name, len(s), len(e)))
    return silent, eager, detail


def test_the_scan_actually_reads_files():
    files = _targets()
    assert len(files) > 20, "扫描目标为空或路径写错，棘轮会恒绿"
    assert (REPO / "memory_cli.py").exists()


def test_silent_exception_count_did_not_grow():
    silent, _eager, detail = _counts()
    assert silent <= BASELINE_SILENT, (
        "memory_cli.py / scripts/ 的静默吞异常从 %d 涨到 %d。"
        "新增站点请带 `# silent-ok: already-reported — <理由>` 或改成 "
        "log_degraded / raise。\n逐文件：%r"
        % (BASELINE_SILENT, silent, detail))


def test_eager_exception_stringification_did_not_grow():
    _silent, eager, detail = _counts()
    assert eager <= BASELINE_EAGER, (
        "memory_cli.py / scripts/ 的急切异常字符串化从 %d 涨到 %d。"
        "改用 _safe_str / exc=。\n逐文件：%r"
        % (BASELINE_EAGER, eager, detail))


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
    monkeypatch.setattr(self_mod, "BASELINE_SILENT", 0)
    silent, _eager, _d = self_mod._counts()
    assert silent > 0
