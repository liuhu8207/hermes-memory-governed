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
3. **AST 护栏（整包）**：扫 ``plugin/memory_governed/**/*.py``（**递归 glob** 动态
   发现，非手写列表），任一 ``except`` 处理器既不调用 ``log_degraded``/``log_data_loss``、
   又不重新抛出，即判失败 —— 以后再加一处静默吞异常会直接红。判**结构位置**而非
   "出现过"：上报/重抛（**任意形态**，含 ``raise ... from e``）必须是 handler 体的
   **直接语句**（``if False: raise`` / 嵌套 ``def h(): raise`` / ``if False:
   log_degraded`` 都不算），且 ``Return(Call(diag))`` 也算上报；``suppress``（无
   ``ExceptHandler`` 节点）另按"**提及即拦**"单扫（对字符串常量做**常量折叠**，
   故 ``'sup'+'press'`` / ``'sup' 'press'`` 都命中；动态拼串见"已知边界"）。豁免按
   **标记种类**分：``# silent-ok: value-fallback — <理由>``（仅窄类型/控制流型）与
   ``# silent-ok: already-reported — <理由>``（任意类型，理由须非空）；未知种类不
   豁免。另有对 ``_diag.py``（诊断实现本身，上报会无限递归）的**模块级豁免**。
4. **诊断层永不抛**：``_diag._format`` 对坏 ``__str__`` 的 ``exc``/``detail``
   退回安全文本；``log_degraded`` 在坏异常上也不得抛（调用点都在 ``except``
   里，抛一次就把优雅降级变成崩溃）。

用**计数**断言，不断言日志行：``log_degraded`` 有 60 秒节流
（``_diag.THROTTLE_SECONDS``），同键第二次不发日志但**永远计数**。测试先
``_diag.reset()``。
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import json
import os
import queue
import re
import sqlite3
import sys
from pathlib import Path

import pytest

import plugin.memory_governed._recall as recall_mod
from plugin.memory_governed import _diag
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._recall import RecallEngine


#: 被护栏扫描的源文件。
_RECALL_PATH = (Path(__file__).resolve().parents[1]
                / "plugin" / "memory_governed" / "_recall.py")

#: ``_diag`` 实现本身（唯一被整块豁免的模块，见 ``_SELF_EXEMPT_MODULES``）。
_DIAG_PATH = (Path(__file__).resolve().parents[1]
              / "plugin" / "memory_governed" / "_diag.py")

#: 被测包目录。整包扫描用 **递归 glob** 动态发现 ``**/*.py``（**不手写文件名列表**
#: —— 手写列表会让将来新增的模块/子包自动成盲区）。
_PACKAGE_DIR = (Path(__file__).resolve().parents[1]
                / "plugin" / "memory_governed")

#: 扫描集合的**锚点** / 模块数**下限**：定义见 ``guardrail_rules``（三个调用方
#: 共用同一份阈值 —— 本文件、``tests/test_no_eager_exception_format.py``、
#: ``scripts/silent_ok_audit.py``）。绑定在 ``_rules`` 导入之后（见下方）。

# ---------------------------------------------------------------------------
# 判定逻辑：**全部**来自 ``guardrail_rules``（唯一实现，与 CI 审计器共用）
# ---------------------------------------------------------------------------
#
# 护栏此前把判定写在本文件里，CI 侧的 ``scripts/silent_ok_audit.py`` 另有一份
# "只认标记存在与否"的正则扫描 —— 两份判定必然漂移（护栏收紧了、审计器还按旧
# 语义放行，或反之），而且**没人跑**的那份等于不存在。
#
# 判定现统一放在仓库根的 ``guardrail_rules.py``：本文件与 CI 审计器都 import 它，
# **改一次、两边同步变化**。下面只做同名绑定 —— 本文件既有的私有名（尤其
# ``tests/test_guardrail_ratchet.py`` 在用的 ``_violations``）保持不变。
#
# 语义说明（抽离未改变任何判定）：
#   * 结构层（``_violations``）与抽离前**完全一致**，故既有测试的红/绿不变；
#   * 新增的**理由层**校验（点名谁在报 / 引用是否存在）只有审计器用 —— 闸门不
#     因此变红，但共用同一份标记解析，不存在第二套判定。
#   * docstring 覆盖的行会被置空（``guardrail_rules.code_lines``）：文档里讲解
#     ``# silent-ok:`` / ``contextlib.suppress`` 的示例不再被当成真实代码。

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import guardrail_rules as _rules

#: 窄异常类型（**真实类**，按继承关系判定子类）。
_NARROW_ERROR_CLASSES = _rules.NARROW_ERROR_CLASSES
#: 控制流型异常（**闭合集合**，禁止模糊匹配）。
_CONTROL_FLOW_CLASSES = _rules.CONTROL_FLOW_CLASSES
#: 异常名 → 真实类的解析命名空间。
_EXC_NAMESPACES = _rules.EXC_NAMESPACES
#: 唯一被整块豁免的模块。
_SELF_EXEMPT_MODULES = _rules.SELF_EXEMPT_MODULES
#: 标记正则 / 已知种类 / 种类-理由分隔符。
_SILENT_OK_RE = _rules.SILENT_OK_RE
_SILENT_OK_KINDS = _rules.SILENT_OK_KINDS
_SILENT_OK_SEP_RE = _rules.SILENT_OK_SEP_RE

#: 扫描集合守卫（锚点 + 模块数下限）—— **唯一实现**在 ``guardrail_rules``。
_PACKAGE_ANCHORS = _rules.PACKAGE_ANCHORS
_PACKAGE_MIN_MODULES = _rules.PACKAGE_MIN_MODULES
_package_py_files = _rules.package_files
_package_anchor_error = _rules.package_anchor_error

_name_of = _rules._name_of
_caught_names = _rules.caught_names
_resolve_exception_class = _rules.resolve_exception_class
_all_caught_are_narrow = _rules.all_caught_are_narrow
_direct_diag_call = _rules.direct_diag_call
_direct_reraise = _rules.direct_reraise
_defines_diag = _rules.defines_diag
_bootstrap_spans = _rules.bootstrap_spans
_in_bootstrap = _rules.in_bootstrap
_fold_str_constants = _rules.fold_str_constants
_suppress_mentions = _rules.suppress_mentions
_line_has_known_marker = _rules.line_has_known_marker
_handler_marker_verdict = _rules.handler_marker_verdict
_violations = _rules.violations


def _silent_ok_kind_reason(line: str):
    """兼容旧名：解析一行里的 ``# silent-ok:`` 标记 → ``(kind, reason)``。

    真正的实现在 ``guardrail_rules.parse_marker``（返回 ``Marker``）；这里保留旧
    的**元组**返回，便于既有用例/外部脚本沿用。
    """
    marker = _rules.parse_marker(line)
    return None if marker is None else (marker.kind, marker.reason)


def _silent_ok_audit_is_the_same_judge() -> bool:
    """**同一份判定**的自证：本文件的 ``_violations`` 就是共享模块的那个函数。"""
    return _violations is _rules.violations


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


def test_narrow_subclass_is_exemptable_by_marker():
    """缺陷 A 回归：窄类型的**子类**带 ``silent-ok`` 标记即豁免，去掉标记仍判红。

    旧护栏按**字面名**比对 ``{TypeError,ValueError,OSError,ImportError}``，
    ``FileNotFoundError`` / ``UnicodeEncodeError`` 不在其中 → 无法加标记、被迫
    上报（噪音）。改为按真实类的 ``issubclass`` 判定后应豁免。
    """
    fne_marked = ("def f():\n    try:\n        return []\n"
                  "    except FileNotFoundError:  # silent-ok: value-fallback — absent\n"
                  "        return []\n")
    assert _violations(fne_marked) == []        # FileNotFoundError ⊂ OSError
    ue_marked = ("def f():\n    try:\n        return []\n"
                 "    except UnicodeEncodeError:  # silent-ok: value-fallback — replace\n"
                 "        return []\n")
    assert _violations(ue_marked) == []         # UnicodeEncodeError ⊂ ValueError
    fne_unmarked = ("def f():\n    try:\n        return []\n"
                    "    except FileNotFoundError:\n        return []\n")
    assert _violations(fne_unmarked) == [4]     # 子类仍需标记，否则判红


def test_attribute_written_exception_is_resolved_and_exemptable():
    """缺陷 B 回归：``json.JSONDecodeError``（``ast.Attribute``）→ 解析到真实类后
    带标记即豁免；旧护栏对 ``ast.Attribute`` 返回 ``{"<non-name>"}`` 无法豁免。"""
    marked = ("def f():\n    try:\n        return []\n"
              "    except json.JSONDecodeError:  # silent-ok: value-fallback — truncated\n"
              "        return []\n")
    assert _violations(marked) == []
    unmarked = ("def f():\n    try:\n        return []\n"
                "    except json.JSONDecodeError:\n        return []\n")
    assert _violations(unmarked) == [4]


def test_tightened_guardrail_still_rejects_real_swallows():
    """回改后护栏**未变松**：删标记 / 只写 logger.debug / 未解析的类型 → 全红。"""
    no_marker = ("def f():\n    try:\n        return []\n"
                 "    except FileNotFoundError:\n        return []\n")
    assert _violations(no_marker) == [4]        # 删掉 silent-ok 标记
    debug_only = ("def f():\n    try:\n        return []\n"
                  "    except ValueError:\n        logger.debug('x')\n        return []\n")
    assert _violations(debug_only) == [4]       # logger.debug 冒充上报不算
    custom_marked = ("def f():\n    try:\n        return []\n"
                     "    except MyCustomError:  # silent-ok: value-fallback\n"
                     "        return []\n")
    assert _violations(custom_marked) == [4]    # 解析不到的自定义异常 → 保守判宽
    dotted_unknown = ("def f():\n    try:\n        return []\n"
                      "    except pkg.WeirdError:  # silent-ok: value-fallback\n"
                      "        return []\n")
    assert _violations(dotted_unknown) == [4]   # Attribute 名字解析不到 → 判宽


def test_guard_requires_direct_statement_not_any_nesting():
    """上报/重抛必须是 handler 体的**直接语句**，不能只"出现过"。"""
    # 直接语句 → 合规。
    direct_report = ("def f():\n    try:\n        return []\n"
                     "    except Exception as e:\n        log_degraded('x', 'y', exc=e)\n")
    assert _violations(direct_report) == []
    direct_raise = ("def f():\n    try:\n        return []\n"
                    "    except Exception:\n        raise\n")
    assert _violations(direct_raise) == []
    # ``raise e`` 也是**直接语句里的 raise**：异常继续传播、上游必然记录 → 合规
    # （"必须是直接语句"已独立防住 ``if False: raise`` / 嵌套 def 的假重抛）。
    raise_exc = ("def f():\n    try:\n        return []\n"
                 "    except Exception as e:\n        raise e\n")
    assert _violations(raise_exc) == []


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


def test_suppress_built_by_constant_folding_is_caught():
    """常量折叠：显式 ``'sup'+'press'``（含多级）与隐式拼接都必须判红。

    此前"提及即拦"只匹配**单个**字符串常量，``getattr(c, 'sup'+'press')``
    （``BinOp(Add)``）能溜过。折叠后与隐式拼接 ``'sup' 'press'`` 一并命中。
    """
    explicit = ("import contextlib\n"
                "with getattr(contextlib, 'sup' + 'press')(OSError):\n"
                "    x = 1\n")
    assert _violations(explicit) == [2]           # 'sup'+'press' 显式折叠
    implicit = ("import contextlib\n"
                "with getattr(contextlib, 'sup' 'press')(OSError):\n"
                "    x = 1\n")
    assert _violations(implicit) == [2]           # 隐式拼接（解析期已折成单常量）
    nested = ("import contextlib\n"
              "with getattr(contextlib, 'sup' + 'pr' + 'ess')(OSError):\n"
              "    x = 1\n")
    assert _violations(nested) == [2]             # 多级嵌套折叠
    # 折叠结果不含 suppress 时不误报。
    assert _violations("x = 'sup' + 'port'\n") == []
    # 动态拼串（掺入变量）**不折叠、不误报** —— 见文件头"已知边界"。
    dynamic = ("import contextlib\n"
               "a = 'sup'\n"
               "with getattr(contextlib, a + 'press')(OSError):\n"
               "    x = 1\n")
    assert _violations(dynamic) == []


def test_recall_has_no_silent_exception_swallow():
    """``_recall.py`` 里不得存在"既不 report 又不 re-raise"的 except 处理器。"""
    source = _RECALL_PATH.read_text(encoding="utf-8")
    bad = _violations(source, filename=_RECALL_PATH.name)
    assert bad == [], (
        "静默吞异常（except 既不 log_degraded/log_data_loss 又不 re-raise）"
        f"位于 _recall.py 行：{bad}")
    # 计数守卫：这个修复点必须≥9（防止有人把护栏改成永远空集）。
    assert len([n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.ExceptHandler)]) >= 9


def test_diag_is_the_only_self_exempt_module():
    """``_SELF_EXEMPT_MODULES`` **只允许**含 ``_diag.py``（防有人往里塞别的文件）。"""
    assert _SELF_EXEMPT_MODULES == frozenset({"_diag.py"})


def test_diag_module_is_exempt_from_both_checks():
    """``_diag.py`` 整块豁免：不再因 ``_safe_str`` 兜底 / ``suppressed`` 提及被判红。

    它是 ``log_degraded`` 的实现本身（上报会无限递归，其兜底只可能标记）；且
    ``stats()`` 的 ``"suppressed"`` 是**文档化契约键**、不能改名 ⇒ suppress 误报。
    """
    src = _DIAG_PATH.read_text(encoding="utf-8")
    assert _violations(src, filename=_DIAG_PATH.name) == []
    # 豁免**按文件名**生效：同一段源码不报名字时仍会被扫（证明豁免在起作用，
    # 而不是"这段源码本来就干净"）。
    assert _violations(src) != []


def test_accepts_any_direct_raise_but_not_fake_reraises():
    """任意形态的**直接语句** ``raise`` 合规；``if False: raise`` / 嵌套 def 仍红。"""
    from_raise = ("def f():\n    try:\n        return []\n"
                  "    except Exception as e:\n        raise RuntimeError('x') from e\n")
    assert _violations(from_raise) == []        # raise X(...) from e
    reraises = ("def f():\n    try:\n        return []\n"
                "    except Exception as e:\n        raise e\n")
    assert _violations(reraises) == []          # raise e
    if_false = ("def f():\n    try:\n        return []\n"
                "    except Exception:\n        if False:\n            raise\n")
    assert _violations(if_false) == [4]         # 假重抛（嵌在 if 里）→ 仍红
    nested = ("def f():\n    try:\n        return []\n"
              "    except Exception:\n        def h():\n            raise\n")
    assert _violations(nested) == [4]           # 假重抛（嵌套 def）→ 仍红


def test_control_flow_exceptions_are_exemptable_by_marker():
    """``queue.Empty`` 等控制流型异常带标记即豁免；无标记 / 宽 ``Exception`` 带标记仍红。"""
    for cls in ("queue.Empty", "queue.Full", "asyncio.QueueEmpty",
                "asyncio.QueueFull", "StopIteration"):
        marked = ("def f():\n    try:\n        return []\n"
                  f"    except {cls}:  # silent-ok: value-fallback — control flow\n"
                  "        return []\n")
        assert _violations(marked) == [], cls
    empty_unmarked = ("def f():\n    try:\n        return []\n"
                      "    except queue.Empty:\n        return []\n")
    assert _violations(empty_unmarked) == [4]   # 控制流型仍需标记
    broad_marked = ("def f():\n    try:\n        return []\n"
                    "    except Exception:  # silent-ok: value-fallback\n        return []\n")
    assert _violations(broad_marked) == [4]     # 宽类型带标记仍红（闭合性）


def test_keyerror_is_narrow_and_marker_exemptable():
    """``KeyError``（``cache[k]`` 缺失 / ``del`` 竞态）带标记即豁免，无标记仍红。"""
    marked = ("def f():\n    try:\n        return []\n"
              "    except KeyError:  # silent-ok: value-fallback — missing key\n"
              "        return []\n")
    assert _violations(marked) == []
    unmarked = ("def f():\n    try:\n        return []\n"
                "    except KeyError:\n        return []\n")
    assert _violations(unmarked) == [4]


# ---------------------------------------------------------------------------
# 追加5：标记**分种类**（value-fallback / already-reported）
# ---------------------------------------------------------------------------

def test_marker_kind_gates_which_exceptions_can_be_exempt():
    """标记分种类 —— ``value-fallback`` 只保窄/控制流型；``already-reported`` 保任意
    类型（含宽 ``Exception``）。覆盖 team-lead 的 5 条用例。"""
    broad_value = ("def f():\n    try:\n        return []\n"
                   "    except Exception:  # silent-ok: value-fallback — oops\n"
                   "        return []\n")
    assert _violations(broad_value) == [4]       # ① 宽 + value-fallback → 红（种类用错）
    broad_already = ("def f():\n    try:\n        return []\n"
                     "    except Exception:  # silent-ok: already-reported — 内层 _ingest.x 已 log_degraded 上报\n"
                     "        return []\n")
    assert _violations(broad_already) == []      # ② 宽 + already-reported → 绿
    narrow_already = ("def f():\n    try:\n        return []\n"
                      "    except KeyError:  # silent-ok: already-reported — 内层已 log_degraded 记录\n"
                      "        return []\n")
    assert _violations(narrow_already) == []     # ③ 窄 + already-reported → 绿
    unknown_kind = ("def f():\n    try:\n        return []\n"
                    "    except Exception:  # silent-ok: 随便写\n"
                    "        return []\n")
    assert _violations(unknown_kind) == [4]      # ④ 未知种类 → 红
    unmarked = ("def f():\n    try:\n        return []\n"
                "    except Exception:\n        return []\n")
    assert _violations(unmarked) == [4]          # ⑤ 无标记 → 红


def test_already_reported_requires_nonempty_reason():
    """``already-reported`` **理由必须非空**（空理由 = 空白支票，不豁免）。"""
    no_reason = ("def f():\n    try:\n        return []\n"
                 "    except Exception:  # silent-ok: already-reported\n"
                 "        return []\n")
    assert _violations(no_reason) == [4]
    blank_reason = ("def f():\n    try:\n        return []\n"
                    "    except Exception:  # silent-ok: already-reported —   \n"
                    "        return []\n")
    assert _violations(blank_reason) == [4]      # 只写了分隔符、理由仍为空 → 红
    filled = ("def f():\n    try:\n        return []\n"
              "    except Exception:  # silent-ok: already-reported — 内层 fetch_url 已报\n"
              "        return []\n")
    assert _violations(filled) == []


def test_unknown_marker_kind_is_rejected_by_both_checks():
    """未知种类**两个检查都不认**（handler 检查 + suppress 检查）。"""
    unknown_handler = ("def f():\n    try:\n        return []\n"
                       "    except Exception:  # silent-ok: mystery — x\n"
                       "        return []\n")
    assert _violations(unknown_handler) == [4]
    unknown_suppress = ("import contextlib\n"
                        "with contextlib.suppress(OSError):  # silent-ok: mystery — x\n"
                        "    x = 1\n")
    assert _violations(unknown_suppress) == [2]
    known_suppress = ("import contextlib\n"
                      "with contextlib.suppress(OSError):  # silent-ok: value-fallback — ok\n"
                      "    x = 1\n")
    assert _violations(known_suppress) == []


# ---------------------------------------------------------------------------
# 追加6：整包扫描（**递归** glob 动态发现，覆盖全部模块 / 子包）
# ---------------------------------------------------------------------------

#: 上面已在 ``_rules`` 导入后绑定了 ``_package_py_files`` / ``_package_anchor_error``
#: （唯一实现在 ``guardrail_rules``：``package_files`` = **递归** rglob；
#: ``package_anchor_error`` = 锚点 + 模块数下限）。这里**不再**各写一份 ——
#: 兄弟文件 ``tests/test_no_eager_exception_format.py`` 与 CI 审计器用的是同一个
#: 函数对象（``tests/test_no_eager_exception_format.py::
#: test_guard_constants_match_the_sibling_file`` 会断言这一点）。


def test_whole_package_has_no_silent_exception_swallow():
    """整包扫描 ``plugin/memory_governed/**/*.py``（**递归 glob**），断言全空。

    失败时给出 ``文件:行`` 清单（便于逐条路由给对应负责人）。``_diag.py`` 的整块
    豁免按文件名照常生效（故其 suppress 提及不入列）。用**递归**发现（``rglob``）
    而非顶层 glob —— 将来加子包也自动纳入。先过锚点校验（防扫空集 / 路径写错）。
    """
    anchor_err = _package_anchor_error(_PACKAGE_DIR)
    assert anchor_err == "", anchor_err
    offenders = []
    for path in _package_py_files(_PACKAGE_DIR):
        src = path.read_text(encoding="utf-8")
        for lineno in _violations(src, filename=path.name):
            offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], (
        "整包静默吞异常（文件:行）：\n  " + "\n  ".join(offenders))


def test_package_scan_guard_fires_on_wrong_root():
    """守卫真的在工作：把扫描根路径故意写错 → 锚点校验失败（而非静默空集恒绿）。"""
    assert _package_anchor_error(_PACKAGE_DIR / "no_such_subdir") != ""
    # 反向证明：真实 root 通过锚点校验（锚点齐全 + 数量达标）。
    assert _package_anchor_error(_PACKAGE_DIR) == ""


def test_package_scan_recurses_into_subpackages(tmp_path):
    """**递归**发现：子目录里埋的坏模块也必须被扫到（防新子包成盲区）。

    team-lead 实测"顶层自动纳入、子目录漏扫"；本用例把坏模块放进子目录，证明
    ``rglob`` 会捞到它（旧的顶层 glob 会漏）。用 ``tmp_path`` 造/删，不污染仓库。
    """
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "_recall.py").write_text("x = 1\n", encoding="utf-8")     # 锚点
    (pkg / "_diag.py").write_text("x = 1\n", encoding="utf-8")       # 锚点
    (pkg / "sub" / "bad.py").write_text(
        "def f():\n    try:\n        return []\n"
        "    except Exception:\n        pass\n", encoding="utf-8")
    offenders = []
    for path in _package_py_files(pkg):
        src = path.read_text(encoding="utf-8")
        for lineno in _violations(src, filename=path.name):
            offenders.append(f"{path.name}:{lineno}")
    assert offenders == ["bad.py:4"], offenders       # 子目录里的坏模块被抓到


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
