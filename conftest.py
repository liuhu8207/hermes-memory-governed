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
* ``HERMES_HOME`` 已设且**不是**生产目录 ⇒ 尊重它（运维显式指定的隔离目录）。
* ``HERMES_HOME`` 未设，或**就是**生产目录 ⇒ 换成一个仓库外的临时目录。

为什么放在 conftest 而不是某个测试模块里
--------------------------------------
若干测试模块会在**测试执行期**加载 ``scripts/`` 下的脚本，而这些脚本在 **import
期**就调用 ``bootstrap()`` —— 那时必须已经有一个可用的隔离 home。放在 conftest
的导入期是唯一能覆盖"import 期"的时机（测试模块的 import 发生在 conftest 之后）。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def _production_home() -> Path:
    """平台默认的生产 home（与 ``scripts/hermes_env.py`` 同规则）。"""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "hermes"


def _normalize(path: Path) -> str:
    """路径归一化（用于比较两个路径是不是同一个）。"""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _is_production_home(raw: str) -> bool:
    return _normalize(Path(raw)) == _normalize(_production_home())


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
