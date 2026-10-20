# -*- coding: utf-8 -*-
"""仓库级测试会话策略：**测试会话永远不指向生产存储**。

为什么需要这一层
----------------
``scripts/hermes_env.py`` 现在**拒绝**在 ``HERMES_HOME`` 未设置、或指向生产目录时
继续（2026-10-20 生产事故：一次全量 pytest 把生产 KB 索引从 52 行写坏成 1 行，
残留行是测试夹具 ``notes/好笔记.md``）。

而本机 ``HERMES_HOME`` 是**用户级环境变量**、且就指着生产目录（``%LOCALAPPDATA%/
hermes``）—— 也就是说"一个测试会话指向生产存储"**不需要任何人为失误**，它是默认
状态。同时 CI（``.github/workflows/ci.yml``）根本没有设 ``HERMES_HOME``。

所以隔离必须由测试会话自己保证：这里在 **conftest 导入期**（早于任何测试模块被
导入、也早于任何 ``scripts/`` 脚本被加载）把 ``HERMES_HOME`` 指到一个**仓库外**的
临时目录。

规则
----
* ``HERMES_HOME`` 已设且**不落在**生产目录之内 ⇒ 尊重它（运维显式指定的隔离目录）。
* ``HERMES_HOME`` 未设，或**落在生产目录之内**（含恰好相等、子目录、尾部分隔符、
  大小写不同的写法）⇒ 换成一个仓库外的临时目录。

判定为什么必须是"落在之内"而不是"相等"
--------------------------------------
用"相等"判定时，``HERMES_HOME`` = 生产目录的**任意一个子目录**（例如
``%LOCALAPPDATA%/hermes/memory``）会被判成"不是生产" ⇒ 不重定向 ⇒ 整个 pytest
会话仍然**在生产目录之内**读写 —— 隔离形同虚设（QA 复评实测出来的漏洞）。

而且"相等"这个口径曾经是**两份**：本文件一份、``scripts/hermes_env.py`` 一份。
两道防线用同一个错口径，就不是互补而是**同漏**。所以这里不再自己实现路径比较，
而是 import ``scripts/hermes_env.py`` 复用它的 :func:`is_production_home` ——
判定逻辑**单一实现**，改一次两道防线同时生效。

为什么放在 conftest 而不是某个测试模块里
--------------------------------------
若干测试模块会在**测试执行期**加载 ``scripts/`` 下的脚本，而这些脚本在 **import
期**就调用 ``bootstrap()`` —— 那时必须已经有一个可用的隔离 home。放在 conftest
的导入期是唯一能覆盖"import 期"的时机（测试模块的 import 发生在 conftest 之后）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_HERMES_ENV_SCRIPT = _REPO_ROOT / "scripts" / "hermes_env.py"


def _load_hermes_env():
    """按路径加载 ``scripts/hermes_env.py``（它不是包，只能按文件加载）。

    加载它是**刻意**的：conftest 与 ``bootstrap()`` 是两道防线，判定"是不是生产
    存储"的口径必须只有一份实现 —— 否则它们会以**同一种方式**漏（见模块文档）。
    ``hermes_env`` 在 import 期**不碰文件系统**（只定义函数与 logger），所以在
    conftest 的导入期加载它是安全的。
    """
    spec = importlib.util.spec_from_file_location(
        "hermes_env_for_conftest", _HERMES_ENV_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "conftest 依赖 %s 判定生产存储，加载失败 ⇒ 拒绝继续（宁可炸在导入期，"
            "也不能在没有隔离判定的情况下跑测试）" % _HERMES_ENV_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # 登记进 sys.modules：``@dataclass`` 会按 ``cls.__module__`` 回去找模块。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_hermes_env = _load_hermes_env()


def _production_home() -> Path:
    """平台默认的生产 home（单一实现：``scripts/hermes_env.py``）。"""
    return _hermes_env.production_home()


def _is_production_home(raw: str) -> bool:
    """``raw`` 是否**落在生产目录之内**（含与它相等）。

    判定口径必须是"落在之内"：``HERMES_HOME`` 指到生产目录的**子目录**时，"相等"
    会判成"不是生产" ⇒ 不重定向 ⇒ pytest 会话仍在生产目录之内读写。
    """
    return _hermes_env.is_production_home(Path(raw))


#: 会话开始时 ``HERMES_HOME`` 的原值。
_ORIGINAL_HOME = (os.environ.get("HERMES_HOME") or "").strip()

#: 本会话真正使用的隔离 home；``None`` 表示"沿用调用方给的那个"。
_SESSION_HOME: Path | None = None

if not _ORIGINAL_HOME or _is_production_home(_ORIGINAL_HOME):
    # 生产 home（或压根没设）⇒ 换成一个仓库外的临时目录。
    _SESSION_HOME = Path(tempfile.mkdtemp(prefix="hermes-memory-governed-tests-"))
    os.environ["HERMES_HOME"] = str(_SESSION_HOME)


# ⚠️ 刻意**不**在会话结束时删掉这个临时目录：
#   * 它在系统临时目录里，只有几个空目录，交给操作系统清理即可；
#   * ``shutil.rmtree`` 会被各种宿主环境的"批量删除保护"拦截（直接 SystemExit），
#     在测试进程收尾时炸掉会把整次运行的退出码改成非 0 —— 为了清理一个空目录
#     而让 CI 变红，不划算。
