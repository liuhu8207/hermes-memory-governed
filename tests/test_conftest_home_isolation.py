# -*- coding: utf-8 -*-
"""仓库根 ``conftest.py`` 的隔离判定口径：**落在生产目录之内** ⇒ 必须重定向。

漏洞（QA 复评实测）
------------------
判据原本是 ``_normalize(a) == _normalize(b)``（**相等**）。于是
``HERMES_HOME`` = 生产目录的**任意一个子目录**（如 ``%LOCALAPPDATA%/hermes/
memory``）会被判成"不是生产" ⇒ ``redirected: False`` ⇒ 整个 pytest 会话仍**在
生产目录之内**读写（2026-10-20 事故的同一种形状：测试写生产索引）。

同一份"相等"口径在 ``scripts/hermes_env.py::is_production_home()`` 里还有一份
—— 两道防线用同一个错口径，就不是互补而是**同漏**。现在 conftest 直接复用
``hermes_env.is_production_home()``（判定**单一实现**），本文件钉住它。

重定向矩阵（本文件逐格断言）
----------------------------
==============================  ========  ==========================
``HERMES_HOME`` 取值              重定向    理由
==============================  ========  ==========================
未设                              ✅       没有 home 可谈
恰好等于生产目录                   ✅       经典形态
生产目录的子目录（``memory``）      ✅       **本轮补的漏洞**
带尾部分隔符（``…/hermes/``）       ✅       归一化后仍落在之内
大小写不同的写法                   ✅       Windows 上 normcase 后同一路径
生产目录**之外**的隔离目录          ❌       运维显式指定 ⇒ 尊重它
==============================  ========  ==========================

⚠️ 本文件**绝不允许**碰到真实存储：``production_home()`` 一律通过
``LOCALAPPDATA`` 指向 tmp 目录（``monkeypatch``）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFTEST = REPO_ROOT / "conftest.py"

#: 每次重载用一个**不同**的模块名：conftest 的决策在**模块级**执行，用同一个名字
#: 会拿到上一次 exec 留下的模块对象（属性不变 ⇒ 看不出这次到底重没重定向）。
_LOAD_SEQ = {"n": 0}


def _reload_conftest(monkeypatch, hermes_home: str | None, production: Path):
    """在假的生产目录 + 指定 ``HERMES_HOME`` 下，**重新执行** conftest 的模块级逻辑。

    Args:
        hermes_home: 要设进 ``os.environ`` 的值；``None`` 表示**不设**。
        production: 假的生产 home（``LOCALAPPDATA`` 被指向它的父目录）。

    Returns:
        重载出来的 conftest 模块对象（可读 ``_ORIGINAL_HOME`` / ``_SESSION_HOME``）。
    """
    monkeypatch.setenv("LOCALAPPDATA", str(production.parent))
    if hermes_home is None:
        monkeypatch.delenv("HERMES_HOME", raising=False)
    else:
        monkeypatch.setenv("HERMES_HOME", hermes_home)

    _LOAD_SEQ["n"] += 1
    name = "conftest_under_test_%d" % _LOAD_SEQ["n"]
    spec = importlib.util.spec_from_file_location(name, CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_production(tmp_path) -> Path:
    """一个**假的**生产 home（真生产目录绝不参与本文件的任何断言）。"""
    production = tmp_path / "localappdata" / "hermes"
    production.mkdir(parents=True)
    return production


def _assert_redirected(module, production: Path) -> None:
    """重定向**确实发生**，且结果**不在**生产目录之内。"""
    assert module._SESSION_HOME is not None, "落在生产目录之内却没有被重定向"
    effective = Path(os.environ["HERMES_HOME"])
    assert effective == module._SESSION_HOME
    assert not module._hermes_env.is_production_home(effective), (
        "重定向后的 home 仍落在生产目录之内：%s ⟵ 生产 %s"
        % (effective, production))


def _assert_respected(module, expected: str, production: Path) -> None:
    """调用方显式给的隔离 home 被**尊重**（不重定向）。"""
    assert module._SESSION_HOME is None, (
        "生产目录之外的隔离 home 被无谓地换掉了：%s" % module._SESSION_HOME)
    assert os.environ["HERMES_HOME"] == expected
    assert not module._hermes_env.is_production_home(Path(expected))


# ---------------------------------------------------------------------------
# 重定向矩阵
# ---------------------------------------------------------------------------

def test_unset_home_is_redirected(monkeypatch, fake_production):
    """①未设 ⇒ 重定向（没有 home 可谈）。"""
    module = _reload_conftest(monkeypatch, None, fake_production)
    assert module._ORIGINAL_HOME == ""
    _assert_redirected(module, fake_production)


def test_home_equal_to_production_is_redirected(monkeypatch, fake_production):
    """②恰好等于生产目录 ⇒ 重定向（经典形态）。"""
    module = _reload_conftest(monkeypatch, str(fake_production), fake_production)
    _assert_redirected(module, fake_production)


def test_subdirectory_of_production_is_redirected(monkeypatch, fake_production):
    """③**子目录** ⇒ 重定向 —— 本轮补的漏洞（旧"相等"口径会放行）。"""
    nested = fake_production / "memory" / "kb_index"
    nested.mkdir(parents=True)
    module = _reload_conftest(monkeypatch, str(nested), fake_production)

    _assert_redirected(module, fake_production)
    # 反向证明这一格在旧口径下**会漏**：相等判定说"不是生产"。
    assert os.path.normcase(os.path.abspath(str(nested))) != os.path.normcase(
        os.path.abspath(str(fake_production))), "前提：子目录 ≠ 生产目录"


def test_trailing_separator_is_redirected(monkeypatch, fake_production):
    """④带尾部分隔符 ⇒ 重定向（归一化后仍落在之内）。"""
    for raw in (str(fake_production) + os.sep,
                str(fake_production) + (os.altsep or os.sep)):
        module = _reload_conftest(monkeypatch, raw, fake_production)
        _assert_redirected(module, fake_production)


def test_case_variant_is_redirected(monkeypatch, fake_production):
    """⑤大小写不同的写法 ⇒ 重定向（Windows 上 normcase 后是同一路径）。"""
    variant = Path(str(fake_production).upper())
    if os.path.normcase(str(variant)) != os.path.normcase(str(fake_production)):
        pytest.skip("该平台路径大小写敏感，无法构造等价的大小写变体")

    module = _reload_conftest(monkeypatch, str(variant), fake_production)
    _assert_redirected(module, fake_production)


def test_isolated_home_outside_production_is_respected(monkeypatch, tmp_path,
                                                       fake_production):
    """⑥生产目录**之外**的隔离目录 ⇒ **尊重**它（运维显式指定，不瞎换）。"""
    isolated = tmp_path / "ops-isolated-home"
    isolated.mkdir()

    module = _reload_conftest(monkeypatch, str(isolated), fake_production)
    _assert_respected(module, str(isolated), fake_production)


# ---------------------------------------------------------------------------
# 判定口径本身：conftest 与 hermes_env 必须是**同一份**实现
# ---------------------------------------------------------------------------

def test_conftest_reuses_the_single_implementation(monkeypatch, fake_production):
    """conftest 不再自带一份路径比较 —— 它调用 ``hermes_env.is_production_home``。

    这是"两道防线同漏"的根因：各写一份判定，就会以**同一种方式**错。这里钉住
    conftest 的判定函数**就是** hermes_env 的那个函数对象。
    """
    module = _reload_conftest(monkeypatch, str(fake_production), fake_production)

    # 行为一致：子目录这一格是分水岭（旧"相等"口径这里返回 False）。
    nested = fake_production / "memory"
    assert module._is_production_home(str(nested)) is True, (
        "子目录必须判成'落在生产之内'（旧口径这里会返回 False）")
    assert module._hermes_env.is_production_home(nested) is True

    # **委派**证明：换掉 hermes_env 的实现，conftest 的结论必须跟着变 ——
    # 若 conftest 自带一份比较，这里就不会变（那正是"同漏"的形状）。
    monkeypatch.setattr(module._hermes_env, "is_production_home",
                        lambda path: False)
    assert module._is_production_home(str(fake_production)) is False, (
        "conftest 没有复用 hermes_env 的判定 —— 两份实现必然再次同漏")
