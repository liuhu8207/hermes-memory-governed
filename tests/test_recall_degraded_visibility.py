# -*- coding: utf-8 -*-
"""``_recall.py`` 降级可见性回归测试 + "静默吞异常" AST 护栏。

为什么需要
----------
本项目最典型的故障是**"安静地失效"**：嵌入后端挂了与"确实没匹配到"对调用方
**完全一样**（都返回 ``[]``），若只在 ``logger.debug`` 留痕，正常运行（默认
level）下根本看不到 —— 排查召回问题时输出不可信。

``_diag.log_degraded`` 是本项目既有机制（WARNING + (component, reason) 计数，
``governed_health`` 可见），但 ``_recall.py`` 里有一批 except 处理器**漏了**
它，只写 ``logger.debug``。本文件三件事：

1. **可见性**：逐处造异常，断言①函数仍返回 ``[]``（降级行为不变）②
   ``_diag.stats()["degraded"]`` 出现对应 ``(component, reason)`` 计数。
2. **不过度上报**：正常的"没匹配"（非 except 的 ``return []``）**不得**计数。
3. **AST 护栏**：扫 ``_recall.py``，任一 ``except`` 处理器既不调用
   ``log_degraded``/``log_data_loss``、又不重新抛出，即判失败 —— 以后再加一处
   静默吞异常会直接红。判**结构位置**而非"出现过"：上报/裸重抛必须是
   handler 体的**直接语句**（``if False: raise`` / 嵌套 ``def h(): raise`` /
   ``if False: log_degraded`` 都不算），且 ``Return(Call(diag))`` 也算上报；
   ``suppress``（无 ``ExceptHandler`` 节点）另按"**提及即拦**"单扫。
4. **诊断层永不抛**：``_diag._format`` 对坏 ``__str__`` 的 ``exc``/``detail``
   退回安全文本；``log_degraded`` 在坏异常上也不得抛（调用点都在 ``except``
   里，抛一次就把优雅降级变成崩溃）。

用**计数**断言，不断言日志行：``log_degraded`` 有 60 秒节流
（``_diag.THROTTLE_SECONDS``），同键第二次不发日志但**永远计数**。测试先
``_diag.reset()``。
"""

from __future__ import annotations

import ast
import os
import re
import sqlite3
from pathlib import Path

import pytest

import plugin.memory_governed._recall as recall_mod
from plugin.memory_governed import _diag
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._recall import RecallEngine


#: 被护栏扫描的源文件。
_RECALL_PATH = (Path(__file__).resolve().parents[1]
                / "plugin" / "memory_governed" / "_recall.py")

#: 只在**值级**防御里捕获的窄异常类型：解析坏配置值 / 坏行字段时返回默认值。
#: 这些是**有意的**安静回退（不是子系统故障），不应上报（否则 per-row 计数噪音
#: 就是另一种"输出不可信"）。
#:
#: ⚠️ 但**类型本身不足以豁免**：豁免必须由**逐站点**的显式标记 ``# silent-ok:``
#: 驱动 —— 否则明天有人用 ``except OSError`` 包住一次真实的磁盘读失败，也会被
#: 类型规则默默放过。护栏规则：``caught ⊆ _NARROW_ERROR_TYPES`` **且**该处理器
#: 带 ``silent-ok`` 标记，才豁免。
_NARROW_ERROR_TYPES = {"TypeError", "ValueError", "OSError", "ImportError"}

#: 站点级静默豁免标记（注释不进 AST，用正则扫源文本行）。
_SILENT_OK_RE = re.compile(r"#\s*silent-ok:")


# ---------------------------------------------------------------------------
# AST 护栏
# ---------------------------------------------------------------------------

def _caught_names(handler: ast.ExceptHandler) -> set:
    """handler 捕获的异常类型名集合；裸 ``except:`` 记为 ``{"__bare__"}``。"""
    t = handler.type
    if t is None:
        return {"__bare__"}
    if isinstance(t, ast.Tuple):
        names = set()
        for elt in t.elts:
            names.add(elt.id if isinstance(elt, ast.Name) else "<non-name>")
        return names
    if isinstance(t, ast.Name):
        return {t.id}
    return {"<non-name>"}


def _direct_diag_call(handler: ast.ExceptHandler) -> bool:
    """handler 体的**直接语句**里是否上报（``log_degraded`` / ``log_data_loss``）。

    认两种直接语句：``Expr(Call(diag))`` 与 ``Return(Call(diag))``（``return
    log_degraded(...)`` 确实报了，不能算违规 —— 过严会在安全方向挡住合法代码）。
    **不认**嵌在 ``if`` / ``def`` / ``try`` 里的调用（``if False:
    log_degraded(...)`` 是 QA 实测的绕过手法）。
    """
    for stmt in handler.body:
        call = None
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        elif isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            call = stmt.value
        if call is None:
            continue
        func = call.func
        if isinstance(func, ast.Name) and func.id in ("log_degraded", "log_data_loss"):
            return True
    return False


def _direct_bare_reraise(handler: ast.ExceptHandler) -> bool:
    """handler 体的**直接语句**里是否有裸重抛（``raise`` 不带表达式）。

    只认顶层 —— 嵌在 ``if`` / ``def`` 里的 ``raise`` **不算**（``if False:
    raise`` 与 handler 内嵌套 ``def h(): raise`` 都是 QA 实测的绕过手法）。
    """
    for stmt in handler.body:
        if isinstance(stmt, ast.Raise) and stmt.exc is None:
            return True
    return False


def _defines_diag(handler: ast.ExceptHandler) -> bool:
    """handler 体的直接语句是否**定义**了 diag 函数（``_diag`` 导入回退引导块）。

    引导块在 ``log_degraded`` 尚未绑定时运行，**不可能**调用自己 —— 唯一豁免。
    """
    for stmt in handler.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name in ("log_degraded", "log_data_loss")):
            return True
    return False


def _has_silent_ok_at(lines: list, lineno: int) -> bool:
    """指定行是否有 ``# silent-ok:`` 标记（注释不进 AST，查源文本行）。"""
    return 1 <= lineno <= len(lines) and bool(_SILENT_OK_RE.search(lines[lineno - 1]))


def _has_silent_ok(handler: ast.ExceptHandler, lines: list) -> bool:
    """处理器跨度内是否有 ``# silent-ok:`` 标记。"""
    start = handler.lineno
    end = handler.body[-1].end_lineno if handler.body else handler.lineno
    return any(_SILENT_OK_RE.search(lines[i]) for i in range(start - 1, end))


def _bootstrap_spans(tree: ast.AST) -> list:
    """返回 ``_diag`` 导入回退块的 ``(start, end)`` 行区间。

    该块（在 ``_diag`` 尚未可用时定义 ``log_degraded`` / ``log_data_loss``）
    **不可能**调用尚未定义的自己，因此整块豁免 —— 包括兜底实现内部的自身防御
    （它同样必须"永不抛"，无法把异常交给一个还不存在的上报函数）。区间**之外**
    的一切仍须上报或重抛。
    """
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _defines_diag(node):
            end = max((getattr(s, "end_lineno", None) or s.lineno) for s in node.body)
            spans.append((node.lineno, end))
    return spans


def _in_bootstrap(lineno: int, spans: list) -> bool:
    """行号是否落在某个引导区间内。"""
    return any(start <= lineno <= end for start, end in spans)


def _suppress_mentions(tree: ast.AST, lines: list) -> list:
    """**提及即拦**：模块里凡出现 ``suppress`` 的行都判违规（除非带标记）。

    ``contextlib.suppress`` 静默吞异常却**没有 ExceptHandler 节点**，是护栏的
    盲区。按"形状"匹配（``with contextlib.suppress(...)``）会被任何一次间接绕过：
    ``getattr(contextlib, 'suppress')`` / ``functools.partial(contextlib.suppress,
    ...)`` / 本地别名 ``sup = contextlib.suppress`` / 包装名 ``mysuppress`` ——
    但它们**最终都得写出 ``suppress`` 这个名字或这个字符串**。所以改为扫所有
    ``Name`` / ``Attribute`` / ``Constant(str)`` / ``ImportFrom`` 名中含
    ``suppress``（不区分大小写）者，命中即违规，除非该行带 ``# silent-ok:`` 标记。
    误报可接受：真要用就显式加标记并写理由 —— 那正是我们要的"显式决定"。
    """
    out = set()
    for node in ast.walk(tree):
        hit = False
        if isinstance(node, ast.Name):
            hit = "suppress" in node.id.lower()
        elif isinstance(node, ast.Attribute):
            hit = "suppress" in node.attr.lower()
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # 专治 ``getattr(contextlib, 'suppress')``（suppress 是字符串常量）。
            hit = "suppress" in node.value.lower()
        elif isinstance(node, ast.ImportFrom):
            # 按**原始名**或别名任一含 suppress 即命中（``suppress as _sup`` 也算）。
            hit = any("suppress" in a.name.lower()
                      or (a.asname is not None and "suppress" in a.asname.lower())
                      for a in node.names)
        if hit and not _has_silent_ok_at(lines, node.lineno):
            out.add(node.lineno)
    return sorted(out)


def _violations(source: str) -> list:
    """返回"静默吞异常"的行号列表（空 = 合规）。

    豁免只有三种：① ``_diag`` 导入回退引导块（整块）；② **窄类型**值级回退
    **且**该处理器带显式 ``# silent-ok:`` 标记；③ 结构合法的重抛 / 上报 ——
    大前提是**判结构位置**而非"出现过"：裸重抛与 ``log_degraded`` 调用必须是
    handler 体的**直接语句**（``if False: raise`` / 嵌套 ``def h(): raise`` /
    ``if False: log_degraded`` 都不算），且 ``Return(Call(diag))`` 也算上报。
    此外 ``suppress`` 无 ExceptHandler 节点，改由"提及即拦"单独扫。
    """
    lines = source.splitlines()
    tree = ast.parse(source)
    spans = _bootstrap_spans(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if _in_bootstrap(node.lineno, spans):
            continue
        if _direct_diag_call(node) or _direct_bare_reraise(node):
            continue
        caught = _caught_names(node)
        if caught and caught <= _NARROW_ERROR_TYPES and _has_silent_ok(node, lines):
            continue
        out.append(node.lineno)
    out.extend(_suppress_mentions(tree, lines))
    return sorted(set(out))


def test_guard_helper_flags_a_synthetic_silent_swallow():
    """护栏自身有效性：含静默吞异常的片段必须被判违规（否则护栏是空转）。"""
    bad = "def f():\n    try:\n        return []\n    except Exception:\n        pass\n"
    assert _violations(bad) == [4]
    good = ("def f():\n    try:\n        return []\n    except Exception as e:\n"
            "        log_degraded('x', 'y', exc=e)\n        return []\n")
    assert _violations(good) == []
    reraise = ("def f():\n    try:\n        return []\n    except Exception:\n"
               "        raise\n")
    assert _violations(reraise) == []


def test_narrow_exemption_is_marker_driven_not_type_driven():
    """窄类型**不加标记**仍判违规；加了 ``# silent-ok:`` 才豁免。"""
    unmarked = ("def f():\n    try:\n        x = int('a')\n"
                "    except (TypeError, ValueError):\n        return 0\n")
    assert _violations(unmarked) == [4]          # 窄类型、无标记 → 违规
    marked = ("def f():\n    try:\n        x = int('a')\n"
              "    except (TypeError, ValueError):  # silent-ok: value-fallback\n"
              "        return 0\n")
    assert _violations(marked) == []             # 标记驱动豁免
    # 广类型即使带标记也不豁免（宽 catch 一律必须上报）。
    broad_marked = ("def f():\n    try:\n        return []\n"
                    "    except Exception:  # silent-ok: value-fallback\n"
                    "        return []\n")
    assert _violations(broad_marked) == [4]


def test_guard_requires_direct_statement_not_any_nesting():
    """上报/重抛必须是 handler 体的**直接语句**，不能只"出现过"。"""
    # 直接语句 → 合规。
    direct_report = ("def f():\n    try:\n        return []\n"
                     "    except Exception as e:\n        log_degraded('x', 'y', exc=e)\n")
    assert _violations(direct_report) == []
    direct_raise = ("def f():\n    try:\n        return []\n"
                    "    except Exception:\n        raise\n")
    assert _violations(direct_raise) == []
    # 带表达式的 raise 不是"裸重抛"（无法保证异常继续传播）。
    raise_exc = ("def f():\n    try:\n        return []\n"
                 "    except Exception as e:\n        raise e\n")
    assert _violations(raise_exc) == [4]


def test_guard_rejects_known_bypasses():
    """QA 给的 4 个绕过片段，收紧后必须**全部判红**。

    收紧前护栏用 ``ast.walk`` 找"出现过"的 Raise/Call，这些片段都能骗过它：
    ``if False: raise`` / 嵌套 ``def h(): raise`` / ``if False: log_degraded``
    都能命中 walk；``contextlib.suppress`` 更是连 ExceptHandler 节点都没有。
    """
    cases = {
        "contextlib.suppress": (
            "import contextlib\n"
            "with contextlib.suppress(ValueError):\n"
            "    x = int('a')\n"),
        "if False: raise": (
            "def f():\n"
            "    try:\n"
            "        return []\n"
            "    except Exception:\n"
            "        if False:\n"
            "            raise\n"),
        "nested def with raise": (
            "def f():\n"
            "    try:\n"
            "        return []\n"
            "    except Exception:\n"
            "        def h():\n"
            "            raise\n"),
        "if False: log_degraded": (
            "def f():\n"
            "    try:\n"
            "        return []\n"
            "    except Exception as e:\n"
            "        if False:\n"
            "            log_degraded('c', 'r', exc=e)\n"),
    }
    for name, src in cases.items():
        assert _violations(src), f"绕过片段未被判红（护栏仍可绕过）：{name}"


def test_return_of_diag_call_is_not_a_violation():
    """``return log_degraded(...)`` 确实上报了，不得误判为静默吞异常（过严假阳）。"""
    src = ("def f():\n    try:\n        return []\n"
           "    except Exception as e:\n        return log_degraded('x', 'y', exc=e)\n")
    assert _violations(src) == []
    # ``return log_data_loss(...)`` 同理。
    src2 = ("def f():\n    try:\n        return []\n"
            "    except Exception as e:\n        return log_data_loss('x', 'y', exc=e)\n")
    assert _violations(src2) == []


def test_suppress_is_caught_and_marker_exempts():
    """``suppress`` 提及即拦（默认判红）；带 ``# silent-ok:`` 标记才豁免。"""
    plain = ("import contextlib\n"
             "with contextlib.suppress(OSError):\n"
             "    x = 1\n")
    assert _violations(plain) == [2]
    marked = ("import contextlib\n"
              "with contextlib.suppress(OSError):  # silent-ok: value-fallback\n"
              "    x = 1\n")
    assert _violations(marked) == []
    # 别名导入：命中的是 import 名（第 1 行）。
    aliased = ("from contextlib import suppress as _sup\n"
               "with _sup(OSError):\n"
               "    x = 1\n")
    assert _violations(aliased) == [1]


def test_suppress_indirect_bypasses_are_caught():
    """QA 的 4 种**间接**绕法，改"提及即拦"后必须**全部判红**。

    按形状匹配时它们都溜过；"提及即拦"下它们最终都得写出 ``suppress`` 这个名字
    或这个字符串，故全部命中。
    """
    cases = {
        "getattr": (
            "import contextlib\n"
            "with getattr(contextlib, 'suppress')(OSError):\n"
            "    x = 1\n"),
        "functools.partial": (
            "import contextlib, functools\n"
            "with functools.partial(contextlib.suppress, OSError)():\n"
            "    x = 1\n"),
        "local-wrapper-name": (
            "def mysuppress():\n"
            "    import contextlib\n"
            "    return contextlib.suppress(OSError)\n"
            "with mysuppress():\n"
            "    x = 1\n"),
        "local-alias": (
            "import contextlib\n"
            "sup = contextlib.suppress\n"
            "with sup(OSError):\n"
            "    x = 1\n"),
    }
    for name, src in cases.items():
        assert _violations(src), f"间接绕法未被判红（护栏仍可绕过）：{name}"


def test_recall_has_no_silent_exception_swallow():
    """``_recall.py`` 里不得存在"既不 report 又不 re-raise"的 except 处理器。"""
    source = _RECALL_PATH.read_text(encoding="utf-8")
    bad = _violations(source)
    assert bad == [], (
        "静默吞异常（except 既不 log_degraded/log_data_loss 又不 re-raise）"
        f"位于 _recall.py 行：{bad}")
    # 计数守卫：这个修复点必须≥9（防止有人把护栏改成永远空集）。
    assert len([n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.ExceptHandler)]) >= 9


# ---------------------------------------------------------------------------
# 可见性：except 路径必须计数
# ---------------------------------------------------------------------------

def _engine(tmp_path) -> RecallEngine:
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.l3_db_path = str(tmp_path / "l3" / "l3.db")
    cfg.vector.backend = "none"
    eng = RecallEngine(cfg)
    eng._l2_init_attempted = True
    eng._l2_store = object()
    eng._embed_fn = lambda q: [0.0] * 4
    return eng


def test_l2_search_failure_is_visible(tmp_path, monkeypatch):
    """L2 检索抛异常：仍返回 ``[]``，但 ``l2::search_failed`` 被计数（不再静默）。"""
    _diag.reset()
    eng = _engine(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("embedding backend down")

    monkeypatch.setattr(eng, "_search_l2_vector", boom)
    out = eng._search_l2("q")

    assert out == []                       # 降级行为不变：仍空手
    assert _diag.stats()["degraded"].get("l2::search_failed") == 1


def test_l3_connect_failure_is_visible(tmp_path, monkeypatch):
    """L3 连接抛异常：仍返回 ``[]``，但 ``l3::connect_failed`` 被计数。"""
    _diag.reset()
    eng = _engine(tmp_path)
    db_file = tmp_path / "l3" / "l3.db"
    db_file.parent.mkdir(parents=True, exist_ok=True)
    db_file.write_bytes(b"")               # 让 Path.exists() 为真，走到 connect

    def boom(*a, **k):
        raise sqlite3.OperationalError("cannot open")

    monkeypatch.setattr(recall_mod.sqlite3, "connect", boom)
    out = eng._search_l3("q")

    assert out == []
    assert _diag.stats()["degraded"].get("l3::connect_failed") == 1


def test_sequential_recall_layer_failure_is_visible(tmp_path, monkeypatch):
    """顺序召回里某层抛异常：该层被跳过并计数，其余层仍返回。"""
    _diag.reset()
    eng = _engine(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("layer down")

    monkeypatch.setattr(eng, "_search_l3", boom)
    monkeypatch.setattr(eng, "_search_l2", lambda *a, **k: [])
    monkeypatch.setattr(eng, "_search_kb", lambda *a, **k: [])
    out = eng._sequential_recall("q")

    assert out == []
    assert _diag.stats()["degraded"].get("recall::sequential_layer_failed") == 1


def test_eval_query_failure_is_visible(tmp_path, monkeypatch):
    """评测门 ``admitted_l2``：``parallel_recall`` 抛异常仍返回 ``[]``，但被计数。"""
    _diag.reset()
    eng = _engine(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("parallel recall exploded")

    monkeypatch.setattr(eng, "parallel_recall", boom)
    out = eng.admitted_l2("q")

    assert out == []                       # 评测门不因单条查询崩掉
    assert _diag.stats()["degraded"].get("recall::eval_query_failed") == 1


# ---------------------------------------------------------------------------
# 不过度上报：正常"没匹配"不得计数
# ---------------------------------------------------------------------------

def test_no_match_path_does_not_report(tmp_path):
    """非 except 的 ``return []``（L2 未初始化 / 确实没命中）不得产生计数。"""
    _diag.reset()
    eng = _engine(tmp_path)
    eng._l2_store = None                   # 走到 _search_l2 的非 except return []
    assert eng._search_l2("q") == []

    degraded = _diag.stats()["degraded"]
    assert degraded.get("l2::search_failed") is None
    assert not any(k.startswith("l2::") for k in degraded), degraded


# ---------------------------------------------------------------------------
# A：诊断层"永不抛" —— 坏 __str__ 不得把优雅降级变成崩溃
# ---------------------------------------------------------------------------

class _Unstr(Exception):
    """``__str__`` 必抛的异常：QA 反例（异常文本急切求值崩溃）的最小复现。"""

    def __str__(self) -> str:                       # pragma: no cover - 故意抛
        raise RuntimeError("no str for you")


def test_diag_format_survives_broken_str():
    """``_diag._format`` 对坏 ``__str__`` 的 exc 必须退回安全文本，绝不冒泡。"""
    text = _diag._format("c", "r", "static", _Unstr())
    assert "c: r" in text
    assert "static" in text
    assert "<unrepr" in text                        # 安全占位，而非抛出


def test_diag_format_survives_broken_detail_str():
    """``detail`` 的 ``__str__`` 抛错时同样不得冒泡。"""

    class _BadDetail:
        def __str__(self):
            raise RuntimeError("bad detail str")

    text = _diag._format("c", "r", _BadDetail(), None)
    assert "c: r" in text
    assert "<unrepr" in text


def test_log_degraded_never_raises_on_broken_exc():
    """``log_degraded`` 的"永不抛异常"契约：坏 ``exc`` 也不得抛。"""
    _diag.reset()
    _diag.log_degraded("x", "y", exc=_Unstr())      # 不得抛
    assert _diag.stats()["degraded"].get("x::y") == 1


def test_l2_search_failure_with_broken_str_still_returns_empty(tmp_path, monkeypatch):
    """A 的关键回归：异常 ``__str__`` 抛错时，旧代码会 RAISED，新代码须仍返回 ``[]`` 并计数。"""
    _diag.reset()
    eng = _engine(tmp_path)

    def boom(*a, **k):
        raise _Unstr("embedding backend down")

    monkeypatch.setattr(eng, "_search_l2_vector", boom)
    out = eng._search_l2("q")                       # 不得抛
    assert out == []
    assert _diag.stats()["degraded"].get("l2::search_failed") == 1


# ---------------------------------------------------------------------------
# C：L1 mtime —— 坏值静默、子系统级 stat 失败上报、缺席不误报
# ---------------------------------------------------------------------------

def _patch_os_stat(monkeypatch, needle: str, exc: BaseException) -> None:
    """让结尾为 ``needle`` 的路径在 ``stat`` 时抛 ``exc``（其余放行）。

    打 ``os.stat`` 而非 ``Path.stat``：pathlib 的 ``exists()``/``stat()`` 最终都走
    ``os.stat``，对 3.13 的 C 加速实现同样有效（已实测拦截成功）。
    """
    real_stat = os.stat

    def fake_stat(path, *a, **k):
        if str(path).endswith(needle):
            raise exc
        return real_stat(path, *a, **k)

    monkeypatch.setattr(os, "stat", fake_stat)


def test_l1_mtime_stat_oserror_is_reported(tmp_path, monkeypatch):
    """C：真子系统级 stat 失败（OSError）**必须上报** ``l1::mtime_stat_failed``。"""
    _diag.reset()
    eng = _engine(tmp_path)
    boom = tmp_path / "boom_l1.md"
    boom.write_text("x", encoding="utf-8")          # 存在：排除"缺席"这一正常情况
    eng._config.l1_memory_path = str(boom)
    eng._config.l1_user_path = str(tmp_path / "absent_user.md")
    _patch_os_stat(monkeypatch, "boom_l1.md", OSError("stat denied"))

    sig = eng._l1_mtime_signature()
    assert sig == 0.0
    assert _diag.stats()["degraded"].get("l1::mtime_stat_failed") == 1


def test_l1_mtime_bad_value_is_silent(tmp_path, monkeypatch):
    """C：坏路径值（ValueError）保持静默 —— 由 ``exists()`` 内部消化，不得上报。

    ``_l1_mtime_signature`` 已**删除**不可达的 ``except ValueError`` 死分支：
    ``Path.exists()`` 自身就 ``except ValueError: return False``，非法路径在
    ``if not p.exists()`` 处即 ``continue``。本用例锁住这个可观测行为。
    """
    _diag.reset()
    eng = _engine(tmp_path)
    eng._config.l1_memory_path = str(tmp_path / "bad_l1.md")
    eng._config.l1_user_path = str(tmp_path / "absent_user.md")
    _patch_os_stat(monkeypatch, "bad_l1.md", ValueError("embedded null byte"))

    sig = eng._l1_mtime_signature()
    assert sig == 0.0
    assert not any(k.startswith("l1::") for k in _diag.stats()["degraded"])


def test_l1_absent_file_is_not_reported(tmp_path):
    """C 回归：文件**缺席**（正常）不得被 ``FileNotFoundError`` 误报为 stat 失败。"""
    _diag.reset()
    eng = _engine(tmp_path)
    eng._config.l1_memory_path = str(tmp_path / "nope_a.md")
    eng._config.l1_user_path = str(tmp_path / "nope_b.md")

    assert eng._l1_mtime_signature() == 0.0
    assert not any(k.startswith("l1::") for k in _diag.stats()["degraded"])
