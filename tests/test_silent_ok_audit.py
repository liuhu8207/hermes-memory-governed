# -*- coding: utf-8 -*-
"""``scripts/silent_ok_audit.py`` 的行为契约（CI 闸门）与"判定只有一份"的证明。

为什么单独一个文件
------------------
审计器此前**没有任何调用方**（CI 不跑、测试不 import），于是它坏了三轮没人发现。
本文件把它钉住：

1. **干净树绿 / 脏树红**都变成**测试**（不再靠人记得跑）；
2. **判定只有一份**：断言本仓库的 pytest 闸门与 CI 审计器用的是**同一个函数对象**
   （``guardrail_rules.violations``），而不是两份会漂移的复制品；
3. **文档示例 ≠ 真实代码**：docstring 里讲解 ``# silent-ok:`` 用法的文字不得被
   当成豁免标记（旧版审计器因此恒红）。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import guardrail_rules as rules

_SCRIPT = REPO / "scripts" / "silent_ok_audit.py"
_SPEC = importlib.util.spec_from_file_location("silent_ok_audit", _SCRIPT)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


# ---------------------------------------------------------------------------
# 1. 判定只有一份
# ---------------------------------------------------------------------------

def test_judge_is_shared_not_duplicated():
    """闸门与审计器**共用同一个函数对象**（改一次两边同步，不存在第二套判定）。"""
    import tests.test_recall_degraded_visibility as visibility

    assert visibility._violations is rules.violations
    assert audit.rules is rules                     # 审计器 import 的就是它
    assert visibility._silent_ok_kind_reason("x") is None
    assert visibility._silent_ok_kind_reason(
        "# silent-ok: value-fallback — 理由") == ("value-fallback", "理由")


def test_shared_marker_parsing_rejects_unknown_kinds():
    """标记解析只认 ``SILENT_OK_KINDS``（两边同一常量，不许各写一份）。"""
    assert rules.parse_marker("# silent-ok: already-reported — 谁在报").known
    assert not rules.parse_marker("# silent-ok: 随便写 — x").known
    assert rules.parse_marker("# silent-ok:") is None      # 空 payload 不算标记
    assert rules.parse_marker("没有标记") is None


# ---------------------------------------------------------------------------
# 2. 文档示例不算代码
# ---------------------------------------------------------------------------

def test_docstring_example_is_not_an_exemption():
    """docstring 里讲解用法的 ``# silent-ok:`` **不豁免**任何处理器（旧版会误判）。

    这是"干净树恒红"的直接成因：旧审计器按原始文本行扫标记，把自己脚下的
    docstring 也扫进去了。
    """
    src = ('"""\n'
           '讲解：except 行可写 ``# silent-ok: value-fallback — 理由`` 来豁免。\n'
           '"""\n'
           'def f():\n'
           '    try:\n'
           '        return []\n'
           '    except Exception:\n'
           '        pass\n')
    # 第 7 行 ``except Exception: pass`` 仍判违规 —— 文档里的标记没有救它。
    assert rules.violations(src) == [7]
    result = rules.audit_source(src, filename="m.py")
    assert any(where == "docstring" for _f, _n, where, _k in result.unattached)


def test_standalone_comment_example_is_unattached():
    """模块顶部注释里举的例子也不生效（只列出，不豁免）。"""
    src = ('# 示例：``# silent-ok: value-fallback``（可接一句理由）。\n'
           'def f():\n'
           '    try:\n'
           '        return []\n'
           '    except Exception:\n'
           '        pass\n')
    assert rules.violations(src) == [5]
    result = rules.audit_source(src, filename="m.py")
    assert [w for _f, _n, w, _k in result.unattached] == ["standalone-comment"]


# ---------------------------------------------------------------------------
# 3. 理由层：能验证的验证，不能验证的**不假装**
# ---------------------------------------------------------------------------

def test_fabricated_reason_is_rejected():
    """``调用方会上报`` 这类无主语的话 → 阻断（这是问题 3 的核心）。"""
    verdict = rules.verify_reason("already-reported", "调用方会上报")
    assert verdict.problems, "编造的理由竟然通过了"
    assert not verdict.verified


def test_reason_must_name_a_reporter_or_a_channel():
    """点名了机制或渠道才过关；两者皆无即空白支票。"""
    named = rules.verify_reason(
        "already-reported", "内层 _ingest.fetch_url 已 log_degraded 上报")
    assert not named.problems
    channel = rules.verify_reason("already-reported", "失败聚合进返回的 errors")
    assert not channel.problems
    assert channel.needs_human          # 点名了渠道，但没给可自动校验的标识


def test_reason_reference_must_exist():
    """理由引用的标识必须真实存在；``--reason-key`` 可显式注入运行时才知道的键。"""
    reason = '内层 log_degraded("ghost", "no_such_key") 已上报'
    empty = rules.SymbolIndex()
    assert rules.verify_reason("already-reported", reason, empty).problems

    injected = rules.SymbolIndex.from_sources([], extra_keys=["ghost::no_such_key"])
    assert not rules.verify_reason("already-reported", reason, injected).problems


def test_symbol_index_sees_real_identifiers():
    """静态索引能认出真实存在的函数 / 模块级常量 / diag 键。"""
    index = rules.SymbolIndex.from_sources([
        ("_ingest", 'MIN = 1\ndef fetch_url():\n    log_degraded("ingest", "fetch_failed")\n'),
    ])
    assert index.resolve("_ingest.fetch_url")
    assert index.resolve("MIN")
    assert index.resolve_key("ingest::fetch_failed")
    assert not index.resolve("_ingest.no_such_func")


# ---------------------------------------------------------------------------
# 4. 干净树绿 / 脏树红（**双向**）
# ---------------------------------------------------------------------------

def test_audit_check_is_green_on_this_repo():
    """干净树：``--check`` 退出码 0（CI 里跑的就是这条命令）。"""
    assert audit.main(["--check"]) == audit.EXIT_OK


def test_audit_check_is_red_on_a_dirty_tree(tmp_path, monkeypatch, capsys):
    """脏树：造四类坏站点，``--check`` 必须非零，且报出**正确文件与行号**。"""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "_diag.py").write_text("x = 1\n", encoding="utf-8")     # 锚点
    (pkg / "_recall.py").write_text("x = 1\n", encoding="utf-8")   # 锚点
    (pkg / "bad.py").write_text(
        "def f():\n"                                              # 1-5：裸吞
        "    try:\n"
        "        return []\n"
        "    except Exception:\n"
        "        pass\n"
        "def g():\n"                                              # 6-10：理由编造
        "    try:\n"
        "        return []\n"
        "    except Exception:  # silent-ok: already-reported — 调用方会上报\n"
        "        return []\n"
        "def h():\n"                                              # 11-15：未知种类
        "    try:\n"
        "        return []\n"
        "    except Exception:  # silent-ok: mystery-kind — 我编的\n"
        "        return []\n"
        "def i():\n"                                              # 16-20：引用不存在
        "    try:\n"
        "        return []\n"
        '    except Exception:  # silent-ok: already-reported — 内层已 log_degraded("ghost","no_such_key") 上报\n'
        "        return []\n",
        encoding="utf-8")

    monkeypatch.setattr(rules, "package_files", lambda *a, **k: sorted(pkg.rglob("*.py")))
    monkeypatch.setattr(rules, "package_anchor_error", lambda *a, **k: "")
    monkeypatch.setattr(audit, "REPO", tmp_path)

    assert audit.main(["--check"]) == audit.EXIT_BLOCKED
    out = capsys.readouterr().out
    for expected in ("bad.py:4", "bad.py:9", "bad.py:14", "bad.py:19"):
        assert expected in out, f"未报出 {expected}：\n{out}"
    assert "0 违规" not in out


def test_strict_upgrades_needs_human_to_blocking(tmp_path, monkeypatch):
    """``--strict``：需人工确认项升级为阻断（默认不阻断，但**不假装验证过**）。"""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "_diag.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "_recall.py").write_text("x = 1\n", encoding="utf-8")
    (pkg / "vague.py").write_text(
        "def g():\n"
        "    try:\n"
        "        return []\n"
        "    except Exception:  # silent-ok: already-reported — 聚合进 errors\n"
        "        return []\n",
        encoding="utf-8")
    monkeypatch.setattr(rules, "package_files", lambda *a, **k: sorted(pkg.rglob("*.py")))
    monkeypatch.setattr(rules, "package_anchor_error", lambda *a, **k: "")
    monkeypatch.setattr(audit, "REPO", tmp_path)

    assert audit.main(["--check"]) == audit.EXIT_OK                 # 默认不阻断
    assert audit.main(["--check", "--strict"]) == audit.EXIT_BLOCKED


def test_reason_key_cli_option_is_wired():
    """``--reason-key`` 真的接到了索引上（命令行 → SymbolIndex）。"""
    reason = '内层 log_degraded("ghost", "no_such_key") 已上报'
    assert rules.verify_reason("already-reported", reason,
                               rules.SymbolIndex()).problems
    keys = audit._load_reason_keys(["ghost::no_such_key"], None)
    index = rules.SymbolIndex.from_sources([], extra_keys=keys)
    assert not rules.verify_reason("already-reported", reason, index).problems
