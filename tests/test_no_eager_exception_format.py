# -*- coding: utf-8 -*-
"""常驻回归：``except ... as e`` 处理器体内**不得急切字符串化** ``e``。

为什么单独成一个文件（**勿与 ``test_recall_degraded_visibility.py`` 合并**）
--------------------------------------------------------------------------
本文件与 ``tests/test_recall_degraded_visibility.py`` 是**两个独立关注点**，
判据不同、机理不同，是**两个 bug 类**，请勿合并成一个：

* ``test_recall_degraded_visibility.py`` 管的是"异常被**吞掉没说**" ——
  某个 ``except`` 既不 ``log_degraded`` / ``log_data_loss``、又不 re-raise
  （AST **结构**护栏，看"有没有上报/重抛"）。
* 本文件管的是"异常被**急切字符串化**" —— 处理器里写了 ``str(e)`` /
  ``f"...{e}"`` / ``"...% e"`` / ``"...".format(e)``。一旦 ``e.__str__`` 抛错，
  就**把优雅降级变成崩溃**：本该吞下并记录，却在外层炸掉。

两者互补：一个错的修复（补 `log_degraded`）不会碰另一个，反之亦然。

为什么值得常驻
--------------
这类缺陷在真实改造中已在 **4 个文件、17 处**被踩到
（``_recall.py`` 的 ``detail=str(e)``、``_embedding.py``、``_ingest.py``、``_kb.py``）。
它不会自然消失，只会被重新写出来。

判据
----
**命中**（=急切、危险），均在 ``except ... as e`` 的 handler 体内：
  * ``str(e)`` / ``repr(e)``
  * ``f"...{e}..."``（``FormattedValue`` 的值恰为 ``Name(e)``，含 ``!r`` / ``!s``）
  * ``"...% e"``（``%`` 的**直接操作数**为 ``e``；或 ``tuple`` 的某**直接元素**为 ``e``）
  * ``"...".format(e)``（``e`` 是 ``.format`` 的**直接实参**）

**排除**（=安全，不得误报）：
  * ``exc=e`` —— 把对象交给 ``_diag``，由其内部安全格式化
  * ``raise ... from e`` —— 只设置 ``__cause__``，不调用 ``str()``
  * ``_safe_str(e)`` —— 已是安全形态（本项目"永不抛"的字符串化）
  * ``type(e).__name__`` / ``e.attr`` —— 属性访问，不是 ``str(e)``
  * ``logger.debug("...%s", e)`` —— logging 的 %-实参**惰性**，仅在真的 emit 时才格式化

已知边界（**当前包内不存在，故不加固** —— 勿为不存在的情况加复杂度）
------------------------------------------------------------------
若将来出现 ``"...".format(**{"x": e})`` 这类**间接**形态（把 ``e`` 藏进
kwargs/字典再展开），本扫描器会漏。届时按需加固即可。
"""

from __future__ import annotations

import ast
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import guardrail_rules as _rules

#: 被扫描的包目录。**递归**发现（``rglob``）—— 与兄弟文件
#: ``tests/test_recall_degraded_visibility.py`` 同一口径；旧的**非递归** ``glob``
#: 会在包内出现子目录时把它整个漏掉（QA 埋文件实验证明：
#: ``plugin/memory_governed/sub/buried.py`` 非递归扫不到、递归能扫到）。
_PKG_DIR = (_REPO_ROOT / "plugin" / "memory_governed")

#: 扫描集合守卫（锚点 + 模块数下限）与发现逻辑：**唯一实现**在
#: ``guardrail_rules``，本文件不再自造一套阈值（否则与兄弟文件互相打架）。
_PACKAGE_ANCHORS = _rules.PACKAGE_ANCHORS
_PACKAGE_MIN_MODULES = _rules.PACKAGE_MIN_MODULES
_pkg_py_files = _rules.package_files
_pkg_anchor_error = _rules.package_anchor_error


def _scan(source: str) -> list:
    """扫描源码，返回 ``[(lineno, pattern, line_text)]``（已按行去重排序）。

    只关心 ``except ... as e`` 绑定了名字的处理器；未绑定名字（``except E:``）无
    ``e`` 可被字符串化，跳过。
    """
    lines = source.splitlines()
    tree = ast.parse(source)
    hits = []
    for handler in ast.walk(tree):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        ename = handler.name
        for node in ast.walk(handler):
            # str(e) / repr(e)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("str", "repr") and len(node.args) == 1
                    and isinstance(node.args[0], ast.Name)
                    and node.args[0].id == ename):
                hits.append((node.lineno, f"{node.func.id}({ename})"))
            # "...".format(e) —— 只认"直接实参恰为 e"（_safe_str(e) 等包装不算）
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "format"
                    and any(isinstance(a, ast.Name) and a.id == ename
                            for a in node.args)):
                hits.append((node.lineno, f'"...".format({ename})'))
            # f"...{e}..." —— FormattedValue 的值恰为 Name(e)
            if isinstance(node, ast.JoinedStr):
                for part in node.values:
                    if (isinstance(part, ast.FormattedValue)
                            and isinstance(part.value, ast.Name)
                            and part.value.id == ename):
                        hits.append((node.lineno, f'f"...{{{ename}}}..."'))
            # "..." % e —— 只认 e 是 % 的**直接操作数**（不是 _safe_str(e)/getattr(e,..)）
            if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod)
                    and _is_direct_operand(node.right, ename)):
                hits.append((node.lineno, f'"..." % {ename}'))
    uniq = sorted(set(hits))
    return [(ln, pat, lines[ln - 1].strip() if 0 < ln <= len(lines) else "")
            for ln, pat in uniq]


def _is_direct_operand(node: ast.AST, name: str) -> bool:
    """``node`` 是否**就是**被 ``%`` 格式化的对象本身（直接操作数）。

    ``"...%s..." % e`` / ``"..." % (a, e)`` → 对象直接交给 ``%``（急切，命中）。
    ``"...%s" % _safe_str(e)`` / ``getattr(e, ...)`` → ``e`` 只是 helper 的**实参**，
    不是被格式化的值（安全，不命中）。
    """
    if isinstance(node, ast.Name) and node.id == name:
        return True
    if isinstance(node, ast.Tuple):
        return any(isinstance(elt, ast.Name) and elt.id == name
                   for elt in node.elts)
    return False


# ---------------------------------------------------------------------------
# 扫描器**自身有效性**（防"哪天退化成恒绿也没人知道"）
# ---------------------------------------------------------------------------

def test_scanner_detects_eager_stringification():
    """真命中必须被识别：f-string / % 直接操作数 / str(e) 各一，共 3 命中。"""
    src = (
        "def a():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        return f'x: {e}'\n"           # HIT: f-string
        "def b():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        return 'x: %s' % e\n"         # HIT: % 直接操作数
        "def c():\n"
        "    try:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        return str(e)\n"              # HIT: str()
    )
    hits = _scan(src)
    assert len(hits) == 3, hits
    patterns = sorted(pat for _ln, pat, _txt in hits)
    assert patterns == ['"..." % e', "f\"...{e}...\"", "str(e)"], patterns


def test_scanner_ignores_safe_forms():
    """安全写法必须**不**被误报（含收窄后的 ``%`` 直接操作数口径）。"""
    src = (
        "def a():\n"                                       # exc=e
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        log_degraded('c', 'r', exc=e)\n"
        "def b():\n"                                       # raise ... from e
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        raise RuntimeError('x') from e\n"
        "def c():\n"                                       # _safe_str(e)
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        return f'x: {_safe_str(e)}'\n"
        "def d():\n"                                       # % with _safe_str
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        return 'x: %s' % _safe_str(e)\n"
        "def f():\n"                                       # tuple: getattr + _safe_str
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        return 'x: %s' % (getattr(e, 'errno', None), _safe_str(e))\n"
        "def g():\n"                                       # 属性访问
        "    try:\n        pass\n"
        "    except Exception as e:\n"
        "        return type(e).__name__\n"
    )
    assert _scan(src) == []


# ---------------------------------------------------------------------------
# 全包回归：命中必须为 0
# ---------------------------------------------------------------------------

def test_package_has_no_eager_exception_stringification():
    """``plugin/memory_governed/**/*.py``（**递归** rglob）内不得出现急切字符串化。

    先过 ``guardrail_rules.package_anchor_error``（锚点 + 模块数下限）—— 旧的
    ``assert files`` 只防"扫到空集"：扫描根写错到某个只含 1 个 ``.py`` 的目录时
    集合非空、断言照过、检查**空转恒绿**。这两层守卫就是为那种"安静地失效"准备的。
    """
    anchor_err = _pkg_anchor_error(_PKG_DIR)
    assert anchor_err == "", anchor_err

    offenders = {}
    for p in _pkg_py_files(_PKG_DIR):
        hits = _scan(p.read_text(encoding="utf-8"))
        if hits:
            offenders[p.relative_to(_PKG_DIR).as_posix()] = hits

    assert offenders == {}, (
        "存在对 except 绑定异常对象的**急切字符串化**"
        "（坏 ``__str__`` 会把优雅降级变成崩溃）—— 改用 ``_safe_str`` / ``exc=e``：\n"
        + "\n".join(f"  {name}:{ln}  [{pat}]  {txt}"
                    for name, hits in offenders.items()
                    for ln, pat, txt in hits))


# ---------------------------------------------------------------------------
# 扫描集合守卫（防"扫错根 / 扫一半"时空转恒绿）与递归发现
# ---------------------------------------------------------------------------

def test_package_scan_guard_fires_on_wrong_root():
    """扫描根写错 → 守卫红（而不是"集合非空就放行"地恒绿）。

    三种错法各试一次：①不存在的目录 ②只有 1 个 ``.py`` 的目录 ③锚点齐全但
    数量远低于下限的目录（这一条专门证"数量下限"这层不是摆设）。
    """
    # ① 不存在的目录 → 空集
    assert _pkg_anchor_error(_PKG_DIR / "no_such_subdir") != ""
    # ② 锚点齐全、但只有 2 个模块 → 数量下限拦住（tmp 目录，不污染仓库）
    with tempfile.TemporaryDirectory() as tmp:
        tiny = Path(tmp) / "pkg"
        tiny.mkdir()
        for anchor in _PACKAGE_ANCHORS:
            (tiny / anchor).write_text("x = 1\n", encoding="utf-8")
        err = _pkg_anchor_error(tiny)
        assert err != "", "只有 2 个模块却通过了守卫 —— 数量下限失效"
        assert "少于下限" in err, err
    # 反向证明：真实 root 通过守卫
    assert _pkg_anchor_error(_PKG_DIR) == ""


def test_package_scan_guard_fires_when_root_has_one_module(tmp_path):
    """扫描根指向**只有 1 个 ``.py``** 的目录 → 整包测试必须红，而不是恒绿。

    这是旧版 ``assert files`` 挡不住的情形：集合非空（1 个文件），断言通过，
    于是"扫了个寂寞"也是绿。
    """
    tiny = tmp_path / "pkg"
    tiny.mkdir()
    (tiny / "_recall.py").write_text("x = 1\n", encoding="utf-8")
    assert len(_pkg_py_files(tiny)) == 1          # 集合非空 —— 旧断言会放行
    assert _pkg_anchor_error(tiny) != ""          # 新守卫拦住


def test_package_scan_recurses_into_subpackages(tmp_path):
    """**递归**发现：子目录里埋的坏模块必须被扫到（防新子包成盲区）。

    QA 的埋文件实验：顶层 ``glob`` 漏掉 ``sub/buried.py``，``rglob`` 能捞到。
    这里用 ``tmp_path`` 复现，不污染仓库。
    """
    pkg = tmp_path / "pkg"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "_recall.py").write_text("x = 1\n", encoding="utf-8")     # 锚点
    (pkg / "_diag.py").write_text("x = 1\n", encoding="utf-8")       # 锚点
    (pkg / "sub" / "buried.py").write_text(
        "def f():\n    try:\n        return []\n"
        "    except Exception as e:\n        return f'x: {e}'\n",
        encoding="utf-8")

    found = {}
    for p in _pkg_py_files(pkg):
        hits = _scan(p.read_text(encoding="utf-8"))
        if hits:
            found[p.relative_to(pkg).as_posix()] = hits
    assert list(found) == ["sub/buried.py"], found      # 子目录里的坏模块被抓到
    assert found["sub/buried.py"][0][0] == 5            # 行号正确


def test_guard_constants_match_the_sibling_file():
    """两个护栏测试的守卫**口径一致**（同一个常量对象，不是一个抄一份）。"""
    import tests.test_recall_degraded_visibility as sibling

    assert _PACKAGE_ANCHORS is sibling._PACKAGE_ANCHORS
    assert _PACKAGE_MIN_MODULES == sibling._PACKAGE_MIN_MODULES
    assert _pkg_py_files is sibling._package_py_files
    assert _pkg_anchor_error is sibling._package_anchor_error
    assert _rules.package_anchor_error(_PKG_DIR) == ""
