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

#: 扫描集合的**锚点**：无论模块数怎么变，都必须包含这些（路径写错 / glob 失效 →
#: 立刻红，防"扫到空集恒绿"）。
_PACKAGE_ANCHORS = ("_diag.py", "_recall.py")

#: 包内模块数**下限**。与实际一致（当前 17）；**新增/删除模块时同步上调/调整** ——
#: 不留余量，防"少扫几个也不红"。与 `_PACKAGE_ANCHORS` 合起来同时防"扫空集"与
#: "路径写错"。
_PACKAGE_MIN_MODULES = 17

#: 只在**值级**防御里捕获的窄异常类型（**真实类**，按继承关系判定）：解析坏配置
#: 值 / 坏行字段时返回默认值。这些是**有意的**安静回退（不是子系统故障），不应
#: 上报（否则 per-row 计数噪音就是另一种"输出不可信"）。
#:
#: ⚠️ 但**类型本身不足以豁免**：豁免必须由**逐站点**的显式标记 ``# silent-ok:``
#: 驱动 —— 否则明天有人用 ``except OSError`` 包住一次真实的磁盘读失败，也会被
#: 类型规则默默放过。护栏规则：捕获到的类 ``⊆ _NARROW_ERROR_CLASSES``（按
#: ``issubclass`` **继承关系**判定，所以 ``FileNotFoundError ⊂ OSError`` /
#: ``UnicodeEncodeError ⊂ ValueError`` / ``json.JSONDecodeError ⊂ ValueError``
#: 同样可豁免）**或**是 ``_CONTROL_FLOW_CLASSES`` 之一，**且**该处理器带
#: ``silent-ok`` 标记，才豁免。``KeyError``（``cache[k]`` 缺失 / ``del`` 竞态）
#: 也是值级回退，一并纳入。
_NARROW_ERROR_CLASSES: tuple = (TypeError, ValueError, OSError, ImportError, KeyError)

#: **控制流型**异常（正常控制流，不是故障）：worker 每秒 poll 队列，空/满都是
#: 预期路径 —— 上报会把计数废掉（每秒 +1）。与窄类型**同权**：**仍需标记**才豁免。
#: ⚠️ **闭合集合**：必须是真实类，**禁止**"名字含 ``Empty``/``Full`` 即豁免"
#: （那种模糊匹配将来能被人拿去豁免宽 ``except Exception``）。
_CONTROL_FLOW_CLASSES: tuple = (
    queue.Empty, queue.Full, asyncio.QueueEmpty, asyncio.QueueFull, StopIteration,
)

#: 解析 ``except`` 里写的异常**名** → 真实类的标准库命名空间（``except`` 里的名字
#: 通常来自它们）。解析不到的一律**保守按宽类型**处理（不豁免）。
_EXC_NAMESPACES = (builtins, json, sqlite3, queue, asyncio)

#: **唯一**被整块豁免的模块。``_diag.py`` 是 ``log_degraded`` 的**实现本身** ——
#: 它上报会**无限递归**（格式化路径本身走 ``_safe_str``），所以它的兜底**只可能
#: 标记、不可能上报**；且它的 ``suppressed`` 是 ``stats()`` 的**文档化契约键**
#: （不能改名），故对本文件的"提及即拦 suppress"全是**误报**。因此对它**同时豁免**
#: 两个检查。模块级豁免由此**命名常量**驱动（显式、可审计），并在
#: ``test_diag_is_the_only_self_exempt_module`` 里断言**只含** ``_diag.py``。
#:
#: ⚠️ **已知例外**：``_diag.py:72/76`` 两条 ``value-fallback`` 挂在**宽 ``except
#: Exception``** 上，违反"VF 仅窄类型"规则，仅因**整模块豁免**才过关。语义**正当**
#: ——那是 ``_safe_str``"永不抛"的终极兜底（异常文本本身可能就是坏的，无法再依赖
#: 任何可能抛异常的上报路径）。此例外**只被本模块豁免覆盖，不可被其它模块引用**：
#: 别的文件若把 ``value-fallback`` 挂在宽 ``Exception`` 上，仍照常判红。
_SELF_EXEMPT_MODULES = frozenset({"_diag.py"})

#: 站点级静默豁免标记（注释不进 AST，用正则扫源文本行）。捕获 ``silent-ok:`` 之后
#: 的整段 payload（**种类** + 分隔符 + 理由），再手工拆分 —— 因为豁免依据按**种类**
#: 分（见下）。
_SILENT_OK_RE = re.compile(r"#\s*silent-ok:\s*(.*)$")

#: **已知**的标记种类。种类不同 = 正当性不同，豁免依据因此**可分类、可审计**：
#:  - ``value-fallback``：**值级回退**（坏配置值 / 坏行字段 / 可选的缺席文件），
#:    **仅**窄类型 **或** 控制流型可豁免；
#:  - ``already-reported``：**已在别处（内层）记录过**，此处再记即**双重计数**，
#:    **任意**捕获类型（含宽 ``Exception``）可豁免，但**理由必须非空**且看得出
#:    是谁在报（如"内层 _ingest.transcribe_* 已 log_degraded"）。
#: ⚠️ **只认这两种**：``# silent-ok: 随便写`` / 未知种类 / 空 payload **一律不豁免**
#: —— 否则标记就成了"随便写个词就能让护栏闭嘴"的空白支票。
_SILENT_OK_KINDS = frozenset({"value-fallback", "already-reported"})

#: 标记里"种类"与"理由"之间的分隔符（em-dash / en-dash / 破折号）。
_SILENT_OK_SEP_RE = re.compile(r"\s*[—–]\s*|\s*--\s*")


# ---------------------------------------------------------------------------
# AST 护栏
# ---------------------------------------------------------------------------

def _name_of(node: ast.AST) -> str:
    """``except`` 子句里的类型节点 → 名字。

    ``ast.Name`` 取 ``id``；``ast.Attribute``（如 ``json.JSONDecodeError``）取
    ``attr`` —— 否则会被错判成 ``<non-name>``，从而无法走继承关系豁免
    （``JSONDecodeError ⊂ ValueError``，本就是值级回退）。
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "<non-name>"


def _caught_names(handler: ast.ExceptHandler) -> set:
    """handler 捕获的异常类型名集合；裸 ``except:`` 记为 ``{"__bare__"}``。"""
    t = handler.type
    if t is None:
        return {"__bare__"}
    if isinstance(t, ast.Tuple):
        return {_name_of(elt) for elt in t.elts}
    return {_name_of(t)}


def _resolve_exception_class(name: str):
    """``except`` 里写的异常**名** → 真实的标准库异常类；解析不到返回 ``None``。

    用真实类（而非字面名）判定才能识别**子类**：``FileNotFoundError`` /
    ``UnicodeEncodeError`` / ``json.JSONDecodeError`` 字面名都不在白名单里，但
    语义上都是值级回退。
    """
    for ns in _EXC_NAMESPACES:
        cls = getattr(ns, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return cls
    return None


def _all_caught_are_narrow(caught: set) -> bool:
    """捕获集合是否**全部**为豁免-friendly（窄类型 **或** 控制流型）。

    任一名字解析不到（自定义 / 动态异常 / ``<non-name>``）、出现裸 ``except``
    （``__bare__``）、或解析出的类既不是窄类型子类也不是控制流型类 → 判**非窄**
    （不豁免）。宁严勿松：豁免须同时满足"捕获的确实是真实窄/控制流类" 与 "显式标记"。
    """
    if not caught or "__bare__" in caught or "<non-name>" in caught:
        return False
    for name in caught:
        cls = _resolve_exception_class(name)
        if cls is None:
            return False
        if not (issubclass(cls, _NARROW_ERROR_CLASSES)
                or issubclass(cls, _CONTROL_FLOW_CLASSES)):
            return False
    return True


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


def _direct_reraise(handler: ast.ExceptHandler) -> bool:
    """handler 体的**直接语句**里是否有 ``raise``（**任意形态**）。

    只认顶层 —— 嵌在 ``if`` / ``def`` 里的 ``raise`` **不算**（``if False:
    raise`` 与 handler 内嵌套 ``def h(): raise`` 都是 QA 实测的绕过手法）。

    形态不限（``raise`` / ``raise e`` / ``raise X(...) from e``）：只要它是 handler
    的**直接语句**，异常就继续向上传播、上游必然记录 —— 可见性已达成，同层**不必**
    再报一次（否则 ``raise ... from e`` 会被迫在前面加一个 ``log_degraded``，造成
    **双重计数**）。"必须是直接语句"这一条已独立防住了 ``if False: raise`` /
    嵌套 ``def h(): raise`` 那类**假重抛**。
    """
    for stmt in handler.body:
        if isinstance(stmt, ast.Raise):
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


def _silent_ok_kind_reason(line: str):
    """解析一行里的 ``# silent-ok:`` 标记 → ``(kind, reason)``；无/空标记返回 ``None``。

    只做**解析**；合法性（种类是否已知、理由是否满足该种类要求）交给调用方按
    ``_SILENT_OK_KINDS`` 判定。
    """
    m = _SILENT_OK_RE.search(line)
    if m is None:
        return None
    payload = m.group(1).strip()
    if not payload:
        return None
    parts = _SILENT_OK_SEP_RE.split(payload, maxsplit=1)
    head = parts[0].strip()
    reason = parts[1].strip() if len(parts) > 1 else ""
    kind = head.split()[0] if head else ""
    return kind, reason


def _line_has_known_marker(line: str) -> bool:
    """某行是否有**已知种类**的标记（未知种类 / 空 payload 不算）。"""
    parsed = _silent_ok_kind_reason(line)
    return parsed is not None and parsed[0] in _SILENT_OK_KINDS


def _handler_marker_verdict(handler: ast.ExceptHandler, lines: list, caught: set) -> bool:
    """按标记**种类**判定 handler 是否豁免（只认 ``_SILENT_OK_KINDS``）。

    - ``value-fallback``：仅当捕获集合**全**为窄类型/控制流型时豁免（同旧语义）；
    - ``already-reported``：**任意**捕获类型可豁免，但**理由必须非空**；
    - 未知种类 / 空 payload：不豁免。
    """
    end = handler.body[-1].end_lineno if handler.body else handler.lineno
    for i in range(handler.lineno - 1, end):
        if i < 0 or i >= len(lines):
            continue
        parsed = _silent_ok_kind_reason(lines[i])
        if parsed is None:
            continue
        kind, reason = parsed
        if kind == "value-fallback" and _all_caught_are_narrow(caught):
            return True
        if kind == "already-reported" and reason:
            return True
    return False


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


# ---------------------------------------------------------------------------
# 已知边界（**接受为不检测** —— 需刻意构造才可绕过，语义级检测复杂度远超收益）
#   - 运行时拼串绕过"提及即拦"：``"{a}{b}".format(a="sup", b="press")`` /
#     ``"%s%s" % ("sup", "press")`` / ``"".join(["sup", "press"])`` —— 运行时才产生
#     "suppress"，源码里**不含任何 suppress 字面量**。（**纯字面量**的 ``'sup'+'press'``
#     已由 ``_fold_str_constants`` 折叠后拦下；只有掺入变量/调用的**动态**拼串才落到
#     这条边界。）
#   - 自定义上下文管理器吞异常：``with MyCtx(): ...`` 且 ``MyCtx.__exit__`` 返回
#     ``True`` —— 既无 ``suppress`` 字面、又无 ``ExceptHandler`` 节点；要检测得做
#     ``__exit__`` 返回值的语义分析，且必须**自己写一个类**才能造出来。
# 两条都需**刻意构造**（不是顺手写法），故**不检测**；记录在此，以免误以为护栏全覆盖。
# ---------------------------------------------------------------------------


def _fold_str_constants(node: ast.AST):
    """对**纯字符串字面量**表达式做常量折叠 → ``str``；含非字面量则返回 ``None``。

    Python 解析期已把隐式拼接 ``'sup' 'press'`` 折成单个 ``Constant``；本函数额外
    处理显式 ``BinOp(Add)``（``'sup'+'press'``），并**递归**到多级嵌套
    （``'sup'+'pr'+'ess'``）。只要有一侧不是字面量（变量 / 调用 / 非 ``Add``）→ 返回
    ``None``：**不猜**，避免把动态拼串误判为命中。
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_str_constants(node.left)
        right = _fold_str_constants(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _suppress_mentions(tree: ast.AST, lines: list) -> list:
    """**提及即拦**：模块里凡出现 ``suppress`` 的行都判违规（除非带标记）。

    ``contextlib.suppress`` 静默吞异常却**没有 ExceptHandler 节点**，是护栏的
    盲区。按"形状"匹配（``with contextlib.suppress(...)``）会被任何一次间接绕过：
    ``getattr(contextlib, 'suppress')`` / ``functools.partial(contextlib.suppress,
    ...)`` / 本地别名 ``sup = contextlib.suppress`` / 包装名 ``mysuppress`` ——
    但它们**最终都得写出 ``suppress`` 这个名字或这个字符串**。所以扫所有
    ``Name`` / ``Attribute`` / ``Constant(str)`` / ``ImportFrom`` 名中含
    ``suppress``（不区分大小写）者，命中即违规，除非该行带已知种类的
    ``# silent-ok:`` 标记。

    对**字符串常量**再做**常量折叠**：``getattr(contextlib, 'sup'+'press')`` 用
    ``BinOp(Add)`` 拼出的字符串此前能溜过（只匹配单个常量），折叠后即命中；隐式
    拼接 ``'sup' 'press'`` 解析期本就折成单个 ``Constant``，同样命中。纯字面量才
    折叠；动态拼串见文件头"已知边界"。误报可接受：真要用就显式加标记并写理由。
    """
    out = set()
    for node in ast.walk(tree):
        hit = False
        if isinstance(node, ast.Name):
            hit = "suppress" in node.id.lower()
        elif isinstance(node, ast.Attribute):
            hit = "suppress" in node.attr.lower()
        elif isinstance(node, (ast.Constant, ast.BinOp)):
            # Constant(str)：专治 ``getattr(contextlib, 'suppress')``；
            # BinOp：常量折叠后专治 ``'sup'+'press'``（含多级嵌套）。
            folded = _fold_str_constants(node)
            hit = folded is not None and "suppress" in folded.lower()
        elif isinstance(node, ast.ImportFrom):
            # 按**原始名**或别名任一含 suppress 即命中（``suppress as _sup`` 也算）。
            hit = any("suppress" in a.name.lower()
                      or (a.asname is not None and "suppress" in a.asname.lower())
                      for a in node.names)
        if hit and not (1 <= node.lineno <= len(lines)
                        and _line_has_known_marker(lines[node.lineno - 1])):
            out.add(node.lineno)
    return sorted(out)


def _violations(source: str, *, filename: str = "") -> list:
    """返回"静默吞异常"的行号列表（空 = 合规）。

    ``filename``（模块名，如 ``"_diag.py"``）落在 ``_SELF_EXEMPT_MODULES`` 时**整块
    豁免**（两个检查都豁免，理由见该常量注释）—— 由**命名常量**驱动，显式可审计。

    其余豁免四种：① ``_diag`` 导入回退引导块（整块）；② **标记驱动**的豁免，按标记
    **种类**判：``value-fallback`` 仅当捕获集合全为**窄类型**（按真实类继承关系判定，
    子类同样豁免）或**控制流型**（``queue.Empty`` 等）时豁免，``already-reported``
    （任意类型，含宽 ``Exception``，**理由须非空**）表示"内层已记录、此处再记即双重
    计数"；未知种类不豁免；③ **直接语句**里的 ``raise``（任意形态：``raise`` /
    ``raise e`` / ``raise X() from e``，异常继续向上传播）；④ **直接语句**里的
    ``log_degraded`` / ``log_data_loss`` 调用（含 ``return log_degraded(...)``）。
    大前提是**判结构位置**而非"出现过"：``if False: raise`` / 嵌套 ``def h():
    raise`` / ``if False: log_degraded`` 都不算。
    此外 ``suppress`` 无 ExceptHandler 节点，改由"提及即拦"单独扫。
    """
    lines = source.splitlines()
    tree = ast.parse(source)
    # _diag.py 是 log_degraded 的实现本身：整块豁免（两个检查）。由 _SELF_EXEMPT_MODULES
    # 这个**命名常量**驱动 —— 不是"扫到某字符串就 continue"的无声形式。
    if Path(filename).name in _SELF_EXEMPT_MODULES:
        return []
    spans = _bootstrap_spans(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if _in_bootstrap(node.lineno, spans):
            continue
        if _direct_diag_call(node) or _direct_reraise(node):
            continue
        caught = _caught_names(node)
        if _handler_marker_verdict(node, lines, caught):
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

def _package_py_files(root: Path) -> list:
    """**递归**发现 ``root`` 下全部 ``*.py``（子包也纳入，避免新子目录成盲区）。"""
    return sorted(root.rglob("*.py"))


def _package_anchor_error(root: Path) -> str:
    """校验扫描集合含锚点且数量达标：通过返回 ``""``，否则返回错误描述。

    **不用"总数 ≥ 15"这种带余量的阈值**（少扫两个也不红），改为①必需模块**锚点**
    + ②数量**下限**（与实际一致）。二者合起来同时防"扫空集恒绿"与"路径写错"。
    """
    names = {p.name for p in _package_py_files(root)}
    missing = [a for a in _PACKAGE_ANCHORS if a not in names]
    if missing:
        return f"扫描集合缺少锚点模块 {missing}（root={root}）"
    if len(names) < _PACKAGE_MIN_MODULES:
        return (f"仅发现 {len(names)} 个模块，少于下限 {_PACKAGE_MIN_MODULES}"
                f"（root={root}）；新增模块时请同步上调该下限")
    return ""


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
