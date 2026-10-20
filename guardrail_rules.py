# -*- coding: utf-8 -*-
"""``# silent-ok:`` 护栏的**唯一**判定实现（测试 与 CI 审计器 共用）。

为什么抽成独立模块
------------------
护栏此前**只活在测试里**（``tests/test_recall_degraded_visibility.py``）。测试
之外没有人调用它 —— 而"没人跑的检查等于不存在"：棘轮从 ``38526e3`` 起就是红的，
但因为没人跑，整整几轮没人发现。

``scripts/silent_ok_audit.py`` 想给 CI 用同一套判定，却又复制了一份**只认标记
存在与否**的正则扫描 —— 两份判定必然漂移（护栏收紧了、审计器还按旧语义放行；
或反之）。所以这里把判定逻辑抽成**唯一一份**，两边 import：

* ``tests/test_recall_degraded_visibility.py``（pytest 闸门）
* ``scripts/silent_ok_audit.py``（CI 阻塞步骤 / 人工复核清单）

判定分两层
----------
1. **结构层**（``_violations`` / ``_violation_details``）：``except`` 处理器有没有
   上报 / 重抛 / 合法标记。**与测试既有语义完全一致**（不新增红），保证抽离不
   改变测试行为。
2. **理由层**（``verify_reason``）：标记里的理由到底**说了什么**、引用是否真实
   存在。只有审计器用（测试闸门不因此变红），但**与结构层共用**同一份标记解析
   （``parse_marker`` / ``SILENT_OK_KINDS``）—— 不重写第二套判定。

文档示例 vs 真实代码
--------------------
``ast.parse`` 出来的节点不含注释；docstring 则是**真实节点**（``Expr(Constant
str)``）。旧审计器按**原始文本行**扫标记，于是把 docstring / 模块注释里讲解
``# silent-ok:`` 的示例也当成豁免标记（假阳性），甚至扫到自己脚下的
``scripts/silent_ok_audit.py``。故本模块一律以 **AST 为锚**：
* 标记只在 ``ExceptHandler`` 的**行区间内**（或 suppress 提及行上）才生效；
* docstring 覆盖的行一律**置空**（``code_lines``），既防示例误判为标记，也防
  文档里写 ``contextlib.suppress`` 被"提及即拦"误伤。
"""

from __future__ import annotations

import ast
import asyncio
import builtins
import json
import queue
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent

#: 零容忍扫描区（与测试 ``_PACKAGE_DIR`` 同一目录，只是搬到这里给两边共用）。
PACKAGE_DIR = REPO_ROOT / "plugin" / "memory_governed"

#: 扫描集合的**锚点**：无论模块数怎么变，都必须包含这些（防"扫到空集恒绿"）。
PACKAGE_ANCHORS: tuple = ("_diag.py", "_recall.py")


# ---------------------------------------------------------------------------
# 常量（沿用测试的命名，抽离后语义不变）
# ---------------------------------------------------------------------------

#: 只在**值级**防御里捕获的窄异常类型（**真实类**，按继承关系判定）。
NARROW_ERROR_CLASSES: tuple = (TypeError, ValueError, OSError, ImportError, KeyError)

#: **控制流型**异常（正常控制流，不是故障）。与窄类型**同权**：仍需标记才豁免。
CONTROL_FLOW_CLASSES: tuple = (
    queue.Empty, queue.Full, asyncio.QueueEmpty, asyncio.QueueFull, StopIteration,
)

#: 解析 ``except`` 里写的异常**名** → 真实类的标准库命名空间。
EXC_NAMESPACES: tuple = (builtins, json, sqlite3, queue, asyncio)

#: **唯一**被整块豁免的模块（``log_degraded`` 的实现本身，上报会无限递归）。
SELF_EXEMPT_MODULES = frozenset({"_diag.py"})

#: 站点级静默豁免标记（注释不进 AST，用正则扫源文本行）。
SILENT_OK_RE = re.compile(r"#\s*silent-ok:\s*(.*)$")

#: **已知**的标记种类（未知种类 / 空 payload **一律不豁免**）。
SILENT_OK_KINDS = frozenset({"value-fallback", "already-reported"})

#: 标记里"种类"与"理由"之间的分隔符（em-dash / en-dash / 破折号）。
SILENT_OK_SEP_RE = re.compile(r"\s*[—–]\s*|\s*--\s*")


# ---------------------------------------------------------------------------
# 理由层常量：什么算"点名了谁在报"
# ---------------------------------------------------------------------------

#: **上报机制**闭集：理由里出现其中一个，即"点名了机制"（不含"谁"）。
#: 闭集而非模糊匹配 —— 模糊匹配会被人拿去用随便一个词蒙混。
REPORTER_MECHANISMS: frozenset = frozenset({
    "log_degraded", "log_data_loss",
    "logger.warning", "logger.error", "logger.exception", "logger.critical",
    "logging.warning", "logging.error", "logging.exception", "logging.critical",
    "warnings.warn", "print", "sys.stderr", "stderr", "raise",
})

#: **可见渠道**闭集：不是"谁在报"，而是"失败以什么形式被下游看到"。出现其一即
#: 说明这条静默**有出口**（返回值 / 报告），但仍未点名具体上报者 ⇒ 需人工确认。
#: ⚠️ 刻意**不含**裸的 ``调用方`` / ``上层`` / ``会处理`` —— 那正是"理由编造"
#: 的典型话术（谁都能写，无人可验证）。
VISIBLE_CHANNELS: tuple = (
    "errors", "error_result", "result", "report", "stderr",
    "返回值", "返回结构", "结果", "报告",
)

#: reason 里形如 ``comp::reason`` 的 ``_diag`` 计数键。
KEY_IN_TEXT_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)::([A-Za-z0-9_]+)")

#: reason 里形如 ``log_degraded("ingest","fetch_failed")`` 的字面上报调用。
#: ⚠️ 只用于**解析理由文本**；建立"仓库里有哪些键"的索引走 **AST**
#: （``diag_call_key``），不能反过来用这条正则扫源码 —— 否则注释里举的例子会让
#: 被引用者"存在"。
DIAG_CALL_IN_TEXT_RE = re.compile(
    r"log_(?:degraded|data_loss)\s*\(\s*[\"']([\w.-]+)[\"']\s*,\s*[\"']([\w.-]+)[\"']")

#: 理由里的**点分式**引用（``_ingest.fetch_url`` / ``logger.warning``）。
DOTTED_REF_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")

#: 理由里的**裸引用**：下划线开头（``_safe_str``）或大驼峰（``ItemResult``）——
#: 普通英文小写词（``per``/``item``/``log``）**不算引用**，否则满篇假阳性。
BARE_REF_RE = re.compile(r"(?<![A-Za-z0-9_.])(_?[A-Za-z][A-Za-z0-9_]*)")


# ---------------------------------------------------------------------------
# 数据载体
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Marker:
    """一个解析出来的 ``# silent-ok:`` 标记。"""

    kind: str
    reason: str
    line: int = 0

    @property
    def known(self) -> bool:
        """种类是否已知（未知种类一律不豁免）。"""
        return self.kind in SILENT_OK_KINDS


@dataclass(frozen=True)
class Violation:
    """一处结构层违规。"""

    lineno: int
    category: str
    detail: str


@dataclass(frozen=True)
class Finding:
    """审计器的一行输出（结构层 + 理由层统一）。"""

    file: str
    line: int
    function: str
    category: str
    detail: str
    blocking: bool


@dataclass
class ReasonVerdict:
    """一条理由的验证结论（**可验证的部分**才标 verified）。"""

    kind: str
    reason: str
    mechanisms: tuple = ()
    channels: tuple = ()
    refs: tuple = ()            #: 引用了且**确实存在**的标识
    unknown_refs: tuple = ()    #: 引用了但**不存在**的标识（编造）
    diag_keys: tuple = ()       #: 引用了且存在的 ``comp::reason`` 键
    unknown_keys: tuple = ()    #: 引用了但不存在的键
    problems: tuple = ()        #: 阻断级问题（人类可读）
    needs_human: bool = False   #: 通过了结构要求，但"谁在报"未被自动验证

    @property
    def verified(self) -> bool:
        """是否**被自动验证过**（有可解析且存在的引用）。"""
        return bool(self.refs or self.diag_keys) and not self.problems

    @property
    def ok(self) -> bool:
        """是否可作为放行依据（无阻断问题）。"""
        return not self.problems


@dataclass
class SymbolIndex:
    """"仓库里真实存在哪些标识"的静态索引（用于理由的存在性检查）。"""

    keys: frozenset = frozenset()      #: ``comp::reason``
    idents: frozenset = frozenset()    #: 函数 / 类 / logger 名
    modules: frozenset = frozenset()   #: 模块名（``_ingest`` 等）

    def resolve_key(self, key: str) -> bool:
        """``comp::reason`` 是否真实存在。"""
        return key in self.keys

    def resolve(self, token: str) -> bool:
        """一个引用 token 是否真实存在。

        * ``comp::reason`` → 键索引；
        * ``mod.func`` → 模块名在索引里 **且** 末段是已定义的函数/类；
        * 裸名 → 必须是已定义的函数/类/logger 名。
        解析不到一律 ``False``（**不猜**，宁可列为"需人工确认"）。
        """
        if not token:
            return False
        if "::" in token:
            return self.resolve_key(token)
        if "." in token:
            parts = token.split(".")
            head, tail = parts[0], parts[-1]
            if head in self.modules or head in self.idents:
                return tail in self.idents
            return False
        return token in self.idents or token in self.modules

    @classmethod
    def from_sources(cls, sources: Iterable[tuple[str, str]],
                     extra_keys: Sequence[str] = (),
                     extra_idents: Sequence[str] = ()) -> "SymbolIndex":
        """从 ``(模块名, 源码)`` 序列建立索引；``extra_*`` 由 ``--reason-key`` 注入。

        索引只认 **AST 里的真实调用 / 定义**，不扫文本 —— docstring 或注释里写的
        示例调用不能被当成"真实存在的键"，否则存在性检查就成了自我实现的预言
        （理由引用什么，什么就"存在"）。
        """
        keys: set = set()
        # ``dir(builtins)``：``None`` / ``ValueError`` 这类**确实存在**的名字，
        # 理由里点名它们不算编造（但不因此豁免"点名谁在报"的要求）。
        idents: set = set(dir(builtins)) | {"logger"}
        modules: set = set()
        for module_name, source in sources:
            modules.add(module_name)
            modules.add(module_name.split(".")[0])
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    idents.add(node.name)
                # diag 键一律**按 AST 调用**抽取，不扫文本：注释里写的
                # ``log_degraded("ghost","x")`` 若被当成真实调用，存在性检查就成了
                # "自我实现的预言"（理由引用什么，什么就"存在"）。
                key = diag_call_key(node)
                if key is not None:
                    keys.add(key)
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "getLogger"
                        and node.args
                        and isinstance(node.args[0], ast.Constant)
                        and isinstance(node.args[0].value, str)):
                    idents.add(node.args[0].value.rsplit(".", 1)[-1])
            # 模块级常量（``MIN_SCORE = 0.1``）也是可引用的真实标识。
            for node in getattr(tree, "body", []):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            idents.add(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target,
                                                                    ast.Name):
                    idents.add(node.target.id)
        return cls(keys=frozenset(keys | set(extra_keys)),
                   idents=frozenset(idents | set(extra_idents)),
                   modules=frozenset(modules))


# ---------------------------------------------------------------------------
# 源码预处理：docstring 置空（"只扫 AST" 的地基）
# ---------------------------------------------------------------------------

def _docstring_spans(tree: ast.AST) -> list:
    """返回所有 docstring 的 ``(start, end)`` 行区间（1-based，闭区间）。"""
    spans = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            spans.append((first.lineno, first.end_lineno or first.lineno))
    return spans


def docstring_line_numbers(source: str) -> set:
    """docstring 覆盖的**行号**集合（1-based）。"""
    return set(_expand_spans(_docstring_spans(ast.parse(source))))


def _expand_spans(spans: Iterable[tuple]) -> Iterator[int]:
    for start, end in spans:
        for line in range(start, end + 1):
            yield line


def code_lines(source: str) -> list:
    """源码行列表，**docstring 覆盖的行被置为空串**。

    ``ast.parse`` 天然不含注释，docstring 却是真实节点 —— 本函数把后者抹掉，
    于是"文档/示例里提到 ``# silent-ok:`` / ``contextlib.suppress``"不再被当成
    真实代码。这是审计器能在干净树上恒绿的关键。
    """
    lines = source.splitlines()
    for lineno in docstring_line_numbers(source):
        if 1 <= lineno <= len(lines):
            lines[lineno - 1] = ""
    return lines


# ---------------------------------------------------------------------------
# 标记解析
# ---------------------------------------------------------------------------

def parse_marker(line: str) -> Marker | None:
    """解析一行里的 ``# silent-ok:`` 标记 → ``Marker``；无/空标记返回 ``None``。

    只做**解析**；合法性（种类是否已知、理由是否满足该种类要求）交给
    ``marker_verdict`` / ``verify_reason`` 判定。
    """
    m = SILENT_OK_RE.search(line)
    if m is None:
        return None
    payload = m.group(1).strip()
    if not payload:
        return None
    parts = SILENT_OK_SEP_RE.split(payload, maxsplit=1)
    head = parts[0].strip()
    reason = parts[1].strip() if len(parts) > 1 else ""
    kind = head.split()[0] if head else ""
    return Marker(kind=kind, reason=reason)


def line_has_known_marker(line: str) -> bool:
    """某行是否有**已知种类**的标记（未知种类 / 空 payload 不算）。"""
    marker = parse_marker(line)
    return marker is not None and marker.known


# ---------------------------------------------------------------------------
# 结构层判定（与测试既有语义一致）
# ---------------------------------------------------------------------------

def _name_of(node: ast.AST) -> str:
    """``except`` 子句里的类型节点 → 名字（``ast.Attribute`` 取 ``attr``）。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return "<non-name>"


def caught_names(handler: ast.ExceptHandler) -> set:
    """handler 捕获的异常类型名集合；裸 ``except:`` 记为 ``{"__bare__"}``。"""
    t = handler.type
    if t is None:
        return {"__bare__"}
    if isinstance(t, ast.Tuple):
        return {_name_of(elt) for elt in t.elts}
    return {_name_of(t)}


def resolve_exception_class(name: str):
    """``except`` 里写的异常**名** → 真实的标准库异常类；解析不到返回 ``None``。"""
    for ns in EXC_NAMESPACES:
        cls = getattr(ns, name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            return cls
    return None


def all_caught_are_narrow(caught: set) -> bool:
    """捕获集合是否**全部**为豁免-friendly（窄类型 **或** 控制流型）。"""
    if not caught or "__bare__" in caught or "<non-name>" in caught:
        return False
    for name in caught:
        cls = resolve_exception_class(name)
        if cls is None:
            return False
        if not (issubclass(cls, NARROW_ERROR_CLASSES)
                or issubclass(cls, CONTROL_FLOW_CLASSES)):
            return False
    return True


def direct_diag_call(handler: ast.ExceptHandler) -> bool:
    """handler 体的**直接语句**里是否上报（``log_degraded`` / ``log_data_loss``）。"""
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
        # ``_diag.log_degraded(...)`` 同样是**真的上报**（模块限定名，避免任意
        # 对象的同名方法被当成上报 —— 那会开一个"随便调个 log_degraded 就免检"的口子）。
        if (isinstance(func, ast.Attribute)
                and func.attr in ("log_degraded", "log_data_loss")
                and isinstance(func.value, ast.Name)
                and func.value.id == "_diag"):
            return True
    return False


def direct_reraise(handler: ast.ExceptHandler) -> bool:
    """handler 体的**直接语句**里是否有 ``raise``（**任意形态**）。"""
    for stmt in handler.body:
        if isinstance(stmt, ast.Raise):
            return True
    return False


def defines_diag(handler: ast.ExceptHandler) -> bool:
    """handler 体的直接语句是否**定义**了 diag 函数（``_diag`` 导入回退引导块）。"""
    for stmt in handler.body:
        if (isinstance(stmt, ast.FunctionDef)
                and stmt.name in ("log_degraded", "log_data_loss")):
            return True
    return False


def bootstrap_spans(tree: ast.AST) -> list:
    """``_diag`` 导入回退块的 ``(start, end)`` 行区间（整块豁免）。"""
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and defines_diag(node):
            end = max((getattr(s, "end_lineno", None) or s.lineno) for s in node.body)
            spans.append((node.lineno, end))
    return spans


def in_bootstrap(lineno: int, spans: list) -> bool:
    """行号是否落在某个引导区间内。"""
    return any(start <= lineno <= end for start, end in spans)


def handler_line_range(handler: ast.ExceptHandler) -> tuple:
    """handler 的**完整行区间**（含 ``except`` 行自身，1-based 闭区间）。

    标记只有落在这个区间里才算"附着在这个处理器上" —— 文档/注释里讲解
    ``# silent-ok:`` 用法的文字（通常在模块顶部或 docstring 里）因此**不会被
    误认成豁免标记**。
    """
    start = handler.lineno
    end = max((getattr(s, "end_lineno", None) or s.lineno) for s in handler.body) \
        if handler.body else handler.lineno
    return start, end


def handler_marker(handler: ast.ExceptHandler, lines: list) -> Marker | None:
    """取 handler 行区间内的**第一个**标记（``lines`` 须已 docstring 置空）。"""
    start, end = handler_line_range(handler)
    for lineno in range(start, end + 1):
        if lineno - 1 < 0 or lineno - 1 >= len(lines):
            continue
        marker = parse_marker(lines[lineno - 1])
        if marker is not None:
            return Marker(kind=marker.kind, reason=marker.reason, line=lineno)
    return None


def handler_marker_verdict(handler: ast.ExceptHandler, lines: list,
                           caught: set) -> tuple:
    """按标记**种类**判定 handler 是否豁免 → ``(exempt, category, detail)``。

    - ``value-fallback``：仅当捕获集合**全**为窄类型/控制流型时豁免；
    - ``already-reported``：**任意**捕获类型可豁免，但**理由必须非空**；
    - 未知种类 / 空 payload / 无标记：不豁免（``category`` 说明是哪种）。
    """
    marker = handler_marker(handler, lines)
    if marker is None:
        return False, "silent-swallow", "except 既不 log_degraded/log_data_loss 又不 re-raise"
    if not marker.known:
        return False, "unknown-marker-kind", (
            "标记种类未知：%r（只认 %s）" % (marker.kind, sorted(SILENT_OK_KINDS)))
    if marker.kind == "value-fallback":
        if all_caught_are_narrow(caught):
            return True, "exempt", "value-fallback（窄类型/控制流型 + 标记）"
        return False, "value-fallback-on-broad-catch", (
            "value-fallback 只能用于窄异常/控制流异常，此处捕获 %s"
            % sorted(caught))
    # already-reported：任意类型，但理由必须非空。
    if not marker.reason:
        return False, "empty-reason", "already-reported 的理由为空（须点名谁在报）"
    return True, "exempt", "already-reported（理由非空）"


def fold_str_constants(node: ast.AST):
    """对**纯字符串字面量**表达式做常量折叠 → ``str``；含非字面量返回 ``None``。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = fold_str_constants(node.left)
        right = fold_str_constants(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def suppress_mentions(tree: ast.AST, lines: list,
                      doc_lines: set | None = None) -> list:
    """**提及即拦**：出现 ``suppress`` 的行都判违规（除非该行带已知种类的标记）。

    ``doc_lines``（docstring 行号）里的节点**跳过**：文档里讲解
    ``contextlib.suppress`` 的危险性不该把自己变成违规。
    """
    doc_lines = doc_lines or set()
    out = set()
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) in doc_lines:
            continue
        hit = False
        if isinstance(node, ast.Name):
            hit = "suppress" in node.id.lower()
        elif isinstance(node, ast.Attribute):
            hit = "suppress" in node.attr.lower()
        elif isinstance(node, (ast.Constant, ast.BinOp)):
            folded = fold_str_constants(node)
            hit = folded is not None and "suppress" in folded.lower()
        elif isinstance(node, ast.ImportFrom):
            hit = any("suppress" in a.name.lower()
                      or (a.asname is not None and "suppress" in a.asname.lower())
                      for a in node.names)
        if hit and not (1 <= node.lineno <= len(lines)
                        and line_has_known_marker(lines[node.lineno - 1])):
            out.add(node.lineno)
    return sorted(out)


def function_spans(tree: ast.AST) -> dict:
    """``行号 -> 所属函数名``（外层优先，最佳努力）。"""
    spans: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno,
                              getattr(node, "end_lineno", node.lineno) + 1):
                spans.setdefault(line, node.name)
    return spans


def violation_details(source: str, *, filename: str = "") -> list:
    """结构层违规明细 → ``[Violation(lineno, category, detail)]``。

    ``filename``（模块名）落在 ``SELF_EXEMPT_MODULES`` 时**整块豁免**。
    豁免路径：① ``_diag`` 导入回退块；② 标记驱动（按种类判）；③ 直接语句
    ``raise``；④ 直接语句 ``log_degraded`` / ``log_data_loss``；⑤ suppress
    另行"提及即拦"。
    """
    tree = ast.parse(source)
    if Path(filename).name in SELF_EXEMPT_MODULES:
        return []
    lines = code_lines(source)
    doc_lines = docstring_line_numbers(source)
    spans = bootstrap_spans(tree)
    out: list = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if in_bootstrap(node.lineno, spans):
            continue
        if direct_diag_call(node) or direct_reraise(node):
            continue
        caught = caught_names(node)
        exempt, category, detail = handler_marker_verdict(node, lines, caught)
        if exempt:
            continue
        out.append(Violation(node.lineno, category, detail))
    for lineno in suppress_mentions(tree, lines, doc_lines):
        out.append(Violation(lineno, "suppress-mention",
                             "出现 suppress（无 ExceptHandler 节点，按提及即拦）"))
    return sorted(out, key=lambda v: (v.lineno, v.category))


def violations(source: str, *, filename: str = "") -> list:
    """结构层违规**行号**列表（测试的既有入口，语义不变）。"""
    return sorted({v.lineno for v in violation_details(source, filename=filename)})


# ---------------------------------------------------------------------------
# 理由层：验证"理由说的是不是真的"（**只验证可验证的部分**）
# ---------------------------------------------------------------------------

def diag_call_key(node: ast.AST) -> str | None:
    """``log_degraded("a","b")`` / ``log_data_loss("a","b")`` 的 **AST 调用节点**
    → ``"a::b"``；不是这种调用（或参数不是字面量）返回 ``None``。

    用于建立"仓库里真实存在哪些 ``_diag`` 计数键"的索引。**只认 AST 调用**、
    不认文本：注释里举例写的 ``log_degraded("ghost","x")`` 不能让 ``ghost::x``
    变成"存在的键"。
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name not in ("log_degraded", "log_data_loss"):
        return None
    if len(node.args) < 2:
        return None
    comp, reason = node.args[0], node.args[1]
    if (isinstance(comp, ast.Constant) and isinstance(comp.value, str)
            and isinstance(reason, ast.Constant)
            and isinstance(reason.value, str)):
        return "%s::%s" % (comp.value, reason.value)
    return None


def _is_camel(token: str) -> bool:
    """是否"看起来像一个类名/类型名"（首字母大写 或 含大写驼峰）。"""
    if not token:
        return False
    return token[0].isupper() or (len(token) > 1 and token[1].isupper())


def extract_references(reason: str) -> tuple:
    """从理由里抽出**看起来像代码引用**的 token → ``(tokens, diag_keys)``。

    刻意保守：只有①点分式（``_ingest.fetch_url``）②下划线开头（``_safe_str``）
    ③大驼峰（``ItemResult``）④``comp::reason`` ⑤``log_degraded("a","b")``
    才算引用。普通英文小写词（``per`` / ``item`` / ``log``）**不算** —— 否则
    每条中文理由都会被拆出一堆"不存在的标识"假阳性。
    """
    if not reason:
        return (), ()
    tokens: list = []
    for token in DOTTED_REF_RE.findall(reason):
        if token not in REPORTER_MECHANISMS and token not in tokens:
            tokens.append(token)
    for token in BARE_REF_RE.findall(reason):
        if token in REPORTER_MECHANISMS:
            continue
        if token.startswith("_") or _is_camel(token):
            if token not in tokens:
                tokens.append(token)
    keys: list = []
    for comp, rsn in KEY_IN_TEXT_RE.findall(reason):
        key = "%s::%s" % (comp, rsn)
        if key not in keys:
            keys.append(key)
    for comp, rsn in DIAG_CALL_IN_TEXT_RE.findall(reason):
        key = "%s::%s" % (comp, rsn)
        if key not in keys:
            keys.append(key)
    return tuple(tokens), tuple(keys)


def verify_reason(kind: str, reason: str,
                  symbols: SymbolIndex | None = None) -> ReasonVerdict:
    """验证一条标记理由 → ``ReasonVerdict``。

    * ``already-reported``：**必须**点名上报者（机制 或 可见渠道），否则是空白
      支票（``调用方会上报`` 这种话谁都能写）。
    * 理由里**引用到的标识必须真实存在**（静态索引 + ``--reason-key`` 注入）。
      引用了不存在的标识 = 编造 ⇒ 阻断。
    * **没能自动验证的部分不会假装通过**：只点名机制/渠道但无可解析引用时，
      ``needs_human=True`` —— 审计器会把它列进"需人工确认"清单。
    """
    low = reason.lower()
    mechanisms = tuple(m for m in sorted(REPORTER_MECHANISMS) if m.lower() in low)
    channels = tuple(c for c in VISIBLE_CHANNELS if c.lower() in low)
    refs, diag_keys = extract_references(reason)
    unknown_refs = tuple(r for r in refs
                         if symbols is not None and not symbols.resolve(r))
    unknown_keys = tuple(k for k in diag_keys
                         if symbols is not None and not symbols.resolve_key(k))
    problems: list = []

    for ref in unknown_refs:
        problems.append("理由引用了不存在的标识：%s" % ref)
    for key in unknown_keys:
        problems.append("理由引用了不存在的 _diag 计数键：%s" % key)

    if kind != "already-reported":
        # value-fallback：理由可选；只校验引用真实性（上面已做），不做"点名"要求。
        return ReasonVerdict(
            kind=kind, reason=reason, mechanisms=mechanisms, channels=channels,
            refs=tuple(r for r in refs if r not in unknown_refs),
            unknown_refs=unknown_refs,
            diag_keys=tuple(k for k in diag_keys if k not in unknown_keys),
            unknown_keys=unknown_keys,
            problems=tuple(problems),
            needs_human=False,
        )

    if not reason:
        problems.append("already-reported 的理由为空：必须点名谁在报")
    elif not mechanisms and not channels:
        problems.append(
            "already-reported 的理由未点名上报者：既无上报机制（%s…）也无可见渠道"
            "（%s…）；「调用方会上报」这类无主语的话不算"
            % (", ".join(sorted(REPORTER_MECHANISMS)[:3]),
               ", ".join(VISIBLE_CHANNELS[:4])))

    resolved_refs = tuple(r for r in refs if r not in unknown_refs)
    resolved_keys = tuple(k for k in diag_keys if k not in unknown_keys)
    # 点名了机制/渠道，但没给出任何**可解析**的上报者 ⇒ 只能靠人看。
    needs_human = (not problems and not resolved_refs and not resolved_keys)
    return ReasonVerdict(
        kind=kind, reason=reason, mechanisms=mechanisms, channels=channels,
        refs=resolved_refs, unknown_refs=unknown_refs,
        diag_keys=resolved_keys, unknown_keys=unknown_keys,
        problems=tuple(problems), needs_human=needs_human,
    )


# ---------------------------------------------------------------------------
# 审计面：给 CI / 人工复核用的完整扫描
# ---------------------------------------------------------------------------

@dataclass
class MarkerSite:
    """一个**生效中**的标记（附着在 except 处理器 / suppress 行上）。"""

    file: str
    line: int
    function: str
    kind: str
    reason: str
    caught: tuple = ()
    handler_line: int = 0
    verdict: ReasonVerdict | None = None


@dataclass
class AuditResult:
    """一次扫描的完整结果。"""

    files: list = field(default_factory=list)
    handler_count: int = 0
    findings: list = field(default_factory=list)
    sites: list = field(default_factory=list)
    unattached: list = field(default_factory=list)

    @property
    def blocking(self) -> list:
        return [f for f in self.findings if f.blocking]

    @property
    def needs_human(self) -> list:
        return [f for f in self.findings if not f.blocking]


def package_files(root: Path = PACKAGE_DIR) -> list:
    """**递归**发现包内全部 ``*.py``（子包也纳入，避免新子目录成盲区）。"""
    return sorted(root.rglob("*.py"))


def package_anchor_error(root: Path = PACKAGE_DIR) -> str:
    """校验扫描集合含锚点且数量达标：通过返回 ``""``，否则返回错误描述。"""
    names = {p.name for p in package_files(root)}
    missing = [a for a in PACKAGE_ANCHORS if a not in names]
    if missing:
        return f"扫描集合缺少锚点模块 {missing}（root={root}）"
    if not names:
        return f"扫描集合为空（root={root}）"
    return ""


def audit_source(source: str, *, filename: str = "", file_label: str = "",
                 symbols: SymbolIndex | None = None) -> AuditResult:
    """扫单个源文本 → ``AuditResult``（结构层 + 理由层 + 未附着标记）。"""
    label = file_label or filename or "<source>"
    tree = ast.parse(source)
    lines = code_lines(source)
    raw_lines = source.splitlines()
    doc_lines = docstring_line_numbers(source)
    funcs = function_spans(tree)
    spans = bootstrap_spans(tree)
    exempt_module = Path(filename).name in SELF_EXEMPT_MODULES

    result = AuditResult(files=[label])
    sites: list = []
    findings: list = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            result.handler_count += 1

    if not exempt_module:
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if in_bootstrap(node.lineno, spans):
                continue
            if direct_diag_call(node) or direct_reraise(node):
                continue
            caught = caught_names(node)
            exempt, category, detail = handler_marker_verdict(node, lines, caught)
            if not exempt:
                findings.append(Finding(
                    file=label, line=node.lineno,
                    function=funcs.get(node.lineno, ""),
                    category=category, detail=detail, blocking=True))
                continue
            marker = handler_marker(node, lines)
            if marker is None or not marker.known:
                continue
            verdict = verify_reason(marker.kind, marker.reason, symbols)
            sites.append(MarkerSite(
                file=label, line=marker.line, function=funcs.get(marker.line, ""),
                kind=marker.kind, reason=marker.reason, caught=tuple(sorted(caught)),
                handler_line=node.lineno, verdict=verdict))
        for lineno in suppress_mentions(tree, lines, doc_lines):
            findings.append(Finding(
                file=label, line=lineno, function=funcs.get(lineno, ""),
                category="suppress-mention",
                detail="出现 suppress（无 ExceptHandler 节点，按提及即拦）",
                blocking=True))
    else:
        # 整块豁免的模块（_diag.py）：仍把它带标记的地方列出来供人工复核，
        # 但不判违规（理由见 SELF_EXEMPT_MODULES）。
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            marker = handler_marker(node, lines)
            if marker is None or not marker.known:
                continue
            verdict = verify_reason(marker.kind, marker.reason, symbols)
            sites.append(MarkerSite(
                file=label, line=marker.line, function=funcs.get(marker.line, ""),
                kind=marker.kind, reason=marker.reason,
                caught=tuple(sorted(caught_names(node))),
                handler_line=node.lineno, verdict=verdict))

    # 理由层的阻断问题（只有审计器才查；测试闸门不因此变红）。
    for site in sites:
        verdict = site.verdict
        if verdict is None:
            continue
        # 只有 ``already-reported`` 的理由是**唯一的豁免依据**，故只有它把理由
        # 问题当阻断；``value-fallback`` 的理由是补充说明，编造风险低，只列入
        # 需人工确认（不假装验证过，也不假装它是阻断级事实）。
        hard = site.kind == "already-reported"
        for problem in verdict.problems:
            category = ("reason-unknown-reference" if "不存在" in problem
                        else "reason-unnamed-reporter")
            findings.append(Finding(
                file=label, line=site.line, function=site.function,
                category=category,
                detail=problem if hard else problem + "（value-fallback，仅供参考）",
                blocking=hard))
        if not verdict.problems and verdict.needs_human:
            findings.append(Finding(
                file=label, line=site.line, function=site.function,
                category="reason-needs-human",
                detail=("已点名%s，但未给出可自动校验的上报者标识 —— 需人工确认"
                        % ("机制 %s" % ",".join(verdict.mechanisms)
                           if verdict.mechanisms
                           else "渠道 %s" % ",".join(verdict.channels))),
                blocking=False))

    result.findings = sorted(findings, key=lambda f: (f.line, f.category))
    result.sites = sites

    # 未附着的标记：文档/注释里讲解用法的示例（不生效，列出以供核对）。
    attached = {s.line for s in sites}
    attached |= set(_expand_spans((h.lineno, handler_line_range(h)[1])
                                  for h in _iter_handlers(tree)))
    for lineno, raw in enumerate(raw_lines, 1):
        marker = parse_marker(raw)
        if marker is None or lineno in attached:
            continue
        if lineno in doc_lines:
            kind = "docstring"
        elif raw.strip().startswith("#"):
            kind = "standalone-comment"
        else:
            kind = "code-line"
        result.unattached.append((label, lineno, kind, marker.kind))
    return result


def _iter_handlers(tree: ast.AST) -> Iterator[ast.ExceptHandler]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            yield node


def scan_paths(paths: Sequence[Path], *, root: Path,
               symbols: SymbolIndex | None = None) -> AuditResult:
    """扫多个文件 → 合并后的 ``AuditResult``。"""
    merged = AuditResult()
    for path in paths:
        source = path.read_text(encoding="utf-8")
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = path.name
        part = audit_source(source, filename=path.name, file_label=rel,
                            symbols=symbols)
        merged.files.append(rel)
        merged.handler_count += part.handler_count
        merged.findings.extend(part.findings)
        merged.sites.extend(part.sites)
        merged.unattached.extend(part.unattached)
    merged.findings.sort(key=lambda f: (f.file, f.line, f.category))
    merged.sites.sort(key=lambda s: (s.file, s.line))
    return merged
