"""Shared environment helpers for Hermes memory governance scripts.

隔离契约（2026-10-20 生产事故后固化）
======================================

事故：2026-10-20 19:56，一次全量 pytest 把**生产 KB 索引**从 52 行写坏成 1 行，
残留行是测试夹具 ``notes/好笔记.md``。vault 正本（125 个 .md）完好，丢的只是索引投影。

根因链（本模块的四处行为各贡献一环）
------------------------------------

1. ``load_dotenv(overwrite=True)`` —— **默认覆盖**进程环境。于是「测试认真设了
   ``HERMES_HOME=tmp``」会被 ``.env`` 里的值顶掉：**隔离不是没设，是被绕过了**。
   更致命的是 ``.env`` 里的 ``HERMES_HOME`` 会让 ``os.environ["HERMES_HOME"]``
   与 ``bootstrap()`` 解析出的 home **分叉** —— 索引路径按前者解析（生产），
   vault 按后者遍历（tmp），``sync_index`` 于是把 52 行真实笔记全判成 orphan 删光。
2. ``HERMES_HOME`` 缺失时**静默回退** ``%LOCALAPPDATA%/hermes``（生产）。
   本机 ``HERMES_HOME`` 还是**用户级环境变量**（就指着生产），所以"测试会话指向
   生产存储"不需要任何人为失误就会发生 —— 它是**默认状态**。
3. 写进 ``os.environ`` 后**永不复原** —— 生产 ``.env``（26KB，含
   ``SILICONFLOW_API_KEY`` 等 API 密钥）留在测试进程里，并污染同进程后续所有测试。
4. vault（``wiki_dir``）与索引（``$HERMES_HOME/memory/kb_index``）**可以不是同一对**
   —— 错配正是那次删空索引的成因，而当时**没有任何一处提示**。

现在的契约
----------

* ``HERMES_HOME`` 未设 / 指向**生产存储**（含它的**子目录**）⇒ 在**测试会话**
  或 ``HGM_STRICT_ISOLATION=1`` 下**拒绝**（抛 :class:`HermesHomeNotConfiguredError`）。
  ⚠️ **正常脚本不在此列**：cron / 运维脚本**必须**能操作生产存储，否则
  ``kb_reindex``、``restore_drill``、``l2_capacity`` 会全部拒绝运行 —— 那不是
  安全，是让备份与清理停摆。所以非严格进程里，未设 / 指向生产只是**解析到生产
  目录并打一条醒目 WARNING**（点名是哪个目录），**不拒绝**。
* 显式授权：``HGM_ALLOW_PRODUCTION=1`` 或 ``bootstrap(..., allow_production=True)``
  ⇒ 连**严格模式**下也放行（WARNING 照打，点明当前用的是哪个 home）。没有这个
  opt-in 时，"严格模式 + 生产 home" 一定是拒绝。
  （判据见 :func:`strict_isolation` / :func:`production_allowed`——**以代码为准**，
  本段是代码行为的描述，不是代码要去满足的愿望。）
* ``load_dotenv`` 默认 ``overwrite=False``：**已显式设置的环境变量永不被覆盖**；
  ``bootstrap()`` 结束前会把 ``os.environ["HERMES_HOME"]`` **重新对齐**到解析结果，
  杜绝"env 变量与 home 分叉"。
* ``.env`` 只提供**数据**（API 密钥等），不提供**路径决定权** —— 它永远不能决定
  存储位置。
* vault 与索引**必须同源**；错配时打 WARNING（闸门由 ``plugin/memory_governed/_kb.py``
  负责，本模块只提示）。
* ``load_dotenv`` 返回它**实际写入**的键；:func:`hermes_env_scope` 退出时**全部还原**。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

#: 本模块的 logger。挂在 ``hermes_governed`` 之下 ⇒ ``setup_logging()`` 装好的
#: stderr handler 照样接得住；没有 handler 时 Python 的 lastResort 也会把 WARNING
#: 打到 stderr —— WARNING 必须**看得见**，静默的警告等于没有警告。
logger = logging.getLogger("hermes_governed.env")

#: 允许「直接指向生产 home」的显式开关（环境变量名）。
PRODUCTION_HOME_FLAG = "HGM_ALLOW_PRODUCTION"

#: 视作"真"的环境变量取值。
_TRUTHY = frozenset({"1", "true", "yes", "on", "y"})

#: ``bootstrap()`` 会在 home 下创建这些目录（保持历史行为）。
_BOOTSTRAP_SUBDIRS = ("memory", "memory/l2", "memory/l3", "cron/output")


class HermesHomeNotConfiguredError(RuntimeError):
    """``HERMES_HOME`` 没设置、或指向生产目录而未显式 opt-in —— 拒绝继续。

    抛它而不是"猜一个"的理由：猜错的代价是把生产索引删空（2026-10-20 事故）。
    """


@dataclass(frozen=True)
class EnvChange:
    """``load_dotenv`` 实际写入 ``os.environ`` 的一条记录。

    ``previous`` 是写入前的值；``None`` 表示该键**此前不存在** —— 还原时应当
    **删除**它，而不是写回空串（空串会被下游读成「已设置但为空」）。
    """

    key: str
    value: str
    previous: Optional[str] = None

    def restore(self) -> None:
        """把该键还原到写入前的状态。"""
        if self.previous is None:
            os.environ.pop(self.key, None)
        else:
            os.environ[self.key] = self.previous


# ---------------------------------------------------------------------------
# 路径比较（不触碰文件系统 —— ``resolve()`` 会，且会在路径不存在时抛错）
# ---------------------------------------------------------------------------

def _normalize(path: Path) -> str:
    """路径归一化（展开 ``~`` → 绝对化 → 大小写归一），用于**比较**两个路径。"""
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _is_within(child: Path, parent: Path) -> bool:
    """``child`` 是否落在 ``parent`` 之内（含与 ``parent`` 相等）。"""
    child_key = _normalize(child)
    parent_key = _normalize(parent)
    if child_key == parent_key:
        return True
    return child_key.startswith(parent_key.rstrip(os.sep) + os.sep)


# ---------------------------------------------------------------------------
# 生产 home 的判定与 opt-in
# ---------------------------------------------------------------------------

def production_home() -> Path:
    """平台默认的「生产 home」（与 ``plugin/memory_governed/_config.py`` 同规则）。

    Windows: ``%LOCALAPPDATA%/hermes``；其它平台: ``~/hermes``。
    """
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "hermes"


def is_production_home(path: Path) -> bool:
    """``path`` 是否**落在**平台默认的生产 home **之内**（含与它相等）。

    判定口径是"落在之内"而不是"相等"：``HERMES_HOME`` = 生产目录的**子目录**
    （如 ``%LOCALAPPDATA%/hermes/memory``）时，"相等"会判成"不是生产" ⇒ 放行 ⇒
    进程仍然**在生产目录之内**读写。测试会话的两道防线（``conftest.py`` 的导入期
    重定向、``bootstrap()`` 的严格模式拒绝）**共用这一个函数**，正是为了避免
    "两道防线以同一种方式漏"。

    影响面（改动前后逐条对比，只有前两项是行为变化）：

    * **严格模式**（pytest / ``HGM_STRICT_ISOLATION=1``）+ 未 opt-in：子目录
      **以前放行、现在拒绝** —— 这就是要修的漏洞，也正是 pytest 会话撞上的形态。
    * **非严格进程**：子目录**以前静默放行、现在多打一条 WARNING**。不阻断、不改
      解析结果 —— cron / 运维脚本照常跑。
    * **已显式 opt-in**（``HGM_ALLOW_PRODUCTION=1`` / ``allow_production=True``）：
      ``bootstrap()`` 走的是 ``if strict and not allow`` ⇒ ``allow`` 为真时**根本
      不看本函数的结论** ⇒ 授权放行路径**不受影响**。
    * 恰好等于生产 home：与改动前完全一致（本来就 True）。
    """
    return _is_within(path, production_home())


def production_allowed(allow_production: Optional[bool] = None) -> bool:
    """是否允许指向生产 home。

    两个来源，任一为真即为允许：

    1. ``allow_production`` 参数（调用方显式声明）；
    2. 进程环境里的 ``HGM_ALLOW_PRODUCTION=1``。

    刻意**不从任何 ``.env`` 读这个开关**：授权必须来自**进程调用**，否则加载器
    就会自我授权 —— "显式 opt-in" 一旦可以自我签发，就不再显式了。
    """
    if allow_production is not None:
        return bool(allow_production)
    return (os.environ.get(PRODUCTION_HOME_FLAG) or "").strip().lower() in _TRUTHY


def strict_isolation() -> bool:
    """是否强制隔离（拒绝生产 home）。

    默认只在**测试会话**里强制：``conftest.py`` 已经把 ``HERMES_HOME`` 指到
    临时目录，这里是第二道防线（防止测试绕过 conftest 直接 import 脚本）。
    生产/cron 的正常脚本必须能操作生产存储，否则 ``kb_reindex``、
    ``restore_drill``、``l2_capacity`` 等运维命令会全部拒绝运行 —— 那不是
    安全，是让备份和清理停摆。

    ``HGM_STRICT_ISOLATION=1`` 可显式打开（CI 或临时排障时使用）。
    """
    if (os.environ.get("HGM_STRICT_ISOLATION") or "").strip().lower() in _TRUTHY:
        return True
    if (os.environ.get("PYTEST_CURRENT_TEST") or "").strip():
        return True
    return "pytest" in sys.modules


class HGMRepoNotConfiguredError(RuntimeError):
    """A deployed script cannot locate the repository that holds ``plugin/``."""


def _source_repo_from_config() -> "Optional[Path]":
    """Read ``source_repo`` from ``$HERMES_HOME/governed_memory.json``.

    A deployment copies ``scripts/`` into ``$HERMES_HOME/scripts`` and the plugin
    into ``$HERMES_HOME/plugins``; the checkout that actually contains
    ``plugin/memory_governed`` is therefore not implied by the script path.
    ``source_repo`` is the explicit pointer for that layout.
    """
    raw_home = (os.environ.get("HERMES_HOME") or "").strip()
    if not raw_home:
        return None
    cfg_path = Path(raw_home).expanduser() / "governed_memory.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # silent-ok: value-fallback — config missing/corrupt → no source_repo
        return None
    raw = str(data.get("source_repo") or "").strip() if isinstance(data, dict) else ""
    return Path(raw).expanduser() if raw else None


def resolve_repo(script_file: str) -> Path:
    """Locate the checkout containing ``plugin/memory_governed``.

    Resolution order:
      1. ``HGM_REPO`` (explicit, always wins);
      2. the script's parent.parent (works for a repo checkout);
      3. ``source_repo`` in ``$HERMES_HOME/governed_memory.json`` (deployed
         scripts; the path is intentionally data, not a hardcoded machine path).

    Raises :class:`HGMRepoNotConfiguredError` with the candidates it tried.
    """
    here = Path(script_file).resolve().parent.parent
    candidates: List[Path] = []
    configured = (os.environ.get("HGM_REPO") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(here)
    from_config = _source_repo_from_config()
    if from_config is not None:
        candidates.append(from_config)

    for candidate in candidates:
        if (candidate / "plugin" / "memory_governed" / "__init__.py").is_file():
            return candidate
    raise HGMRepoNotConfiguredError(
        "cannot locate the HGM repository (looked for "
        "plugin/memory_governed/__init__.py). Tried: %s. Set HGM_REPO or "
        "source_repo in $HERMES_HOME/governed_memory.json."
        % ", ".join(str(c) for c in candidates))


# ---------------------------------------------------------------------------
# .env 读取
# ---------------------------------------------------------------------------

def load_dotenv(path: Path, overwrite: bool = False) -> List[EnvChange]:
    """把 ``.env`` 的 ``KEY=VALUE`` 读进 ``os.environ``，返回**实际写入**的记录。

    与事故前的旧版有三处差异：

    * ``overwrite`` 默认从 ``True`` 改成 ``False`` —— **已显式设置的环境变量永不被
      ``.env`` 覆盖**。``HERMES_HOME`` 尤其关键：测试设了 tmp home 却被 ``.env``
      顶成生产 home，"隔离"就被绕过了。
    * 返回值从 ``None`` 变成记录列表：调用方（:func:`hermes_env_scope`）才能在退出时
      **还原**这些键 —— 旧版把生产 ``.env``（含 API 密钥）留在进程里且永不复原。
    * 键名为空则跳过（与 ``plugin/memory_governed/_config.py`` 的解析口径一致）。

    ``overwrite=True`` 仍然可用，但调用方必须写明理由 —— 没有理由的覆盖语义就是
    事故本身。
    """
    path = Path(path)
    if not path.exists():
        return []
    changes: List[EnvChange] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        if key in os.environ and not overwrite:
            continue
        changes.append(EnvChange(key=key, value=value,
                                 previous=os.environ.get(key)))
        os.environ[key] = value
    return changes


def restore_env(changes: List[EnvChange]) -> None:
    """把 ``load_dotenv`` 写入的键**逐个还原**（此前不存在的键删除）。"""
    for change in reversed(changes):
        change.restore()


# ---------------------------------------------------------------------------
# vault 解析与"同源"检查
# ---------------------------------------------------------------------------

def _declared_wiki_dir(hermes_home: Path) -> Optional[Path]:
    """``$HERMES_HOME/governed_memory.json`` 里**声明**的 vault 路径（没有则 ``None``）。

    "声明"是同源的关键：home 自己的配置指出来的 vault，按构造就属于这个 home ——
    即使它在 home 之外（生产部署就是这样：vault 在 Obsidian 目录
    ``C:/Users/<user>/wiki``，索引在 ``%LOCALAPPDATA%/hermes/memory/kb_index``）。
    """
    config_path = hermes_home / "governed_memory.json"
    if not config_path.exists():
        return None
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # silent-ok: value-fallback — 坏配置退回默认 vault 路径
        raw = None
    if not isinstance(raw, dict):
        return None
    configured = str(raw.get("wiki_dir") or "").strip()
    return Path(configured).expanduser() if configured else None


def resolve_wiki_dir(hermes_home: Path) -> Path:
    """解析 vault 根目录（与 ``plugin/memory_governed/_config.py`` 同规则）。

    优先级：``GOVERNED_WIKI_DIR`` 环境变量 → ``$HERMES_HOME/governed_memory.json``
    的 ``wiki_dir`` → ``$HERMES_HOME/wiki``。
    """
    override = (os.environ.get("GOVERNED_WIKI_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    declared = _declared_wiki_dir(hermes_home)
    if declared is not None:
        return declared
    return hermes_home / "wiki"


def _warn_if_vault_mismatched(hermes_home: Path) -> None:
    """vault 与索引不是同一对时，打一条醒目 WARNING。

    索引写在 ``$HERMES_HOME/memory/kb_index``，vault 读 ``wiki_dir``。两者不是同一对
    时，索引里的行在"当前 vault"里**一个都不存在** → ``sync_index`` 把它们全判成
    orphan 删光（2026-10-20 生产事故：52 行 → 1 行）。

    判定"同一对"的规则（**刻意不把"vault 在 home 之外"当成错配**）：

    * vault 落在 home 之内 ⇒ 同源；
    * vault 由 **home 自己的配置**声明 ⇒ 同源（生产部署就是这种：vault 在
      Obsidian 目录，索引在 home 里）；
    * 其余（vault 由 ``GOVERNED_WIKI_DIR`` 指到 home 之外、且 home 的配置并没有
      声明它）⇒ **错配**，报警。

    这里只**提示**，不做闸门 —— 闸门归 ``plugin/memory_governed/_kb.py``。
    """
    effective = resolve_wiki_dir(hermes_home)
    if _is_within(effective, hermes_home):
        return
    declared = _declared_wiki_dir(hermes_home)
    if declared is not None and _normalize(declared) == _normalize(effective):
        return
    logger.warning(
        "[HERMES_ENV][错配] vault 与索引不同源：wiki_dir=%s 不在 HERMES_HOME=%s 之内，"
        "而索引会写在 %s。索引里的行会被判成 orphan 并删光（2026-10-20 生产事故的成因）"
        "—— 请让两者同源。",
        effective, hermes_home, hermes_home / "memory" / "kb_index")


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------

def _missing_home_message() -> str:
    """``HERMES_HOME`` 未设置时的报错文案（必须让人一眼知道下一步做什么）。"""
    return (
        "HERMES_HOME 未设置，拒绝猜测存储位置。\n"
        "  · 隔离存储（测试 / 一次性脚本）：请显式设置 HERMES_HOME=<隔离目录>，"
        "例如 pytest 前 export HERMES_HOME=$(mktemp -d)。\n"
        "  · 确实要直接操作生产存储：请显式 opt-in —— 设 HGM_ALLOW_PRODUCTION=1，"
        "或调用 bootstrap(..., allow_production=True)。\n"
        "  · 平台默认的生产目录是 %s；不再静默回退到它 —— 猜错一次的代价是把生产索引"
        "删空（2026-10-20 事故）。" % production_home())


def _production_home_message(hermes_home: Path) -> str:
    """解析出的 home 就是生产目录、而调用方未 opt-in 时的报错文案。"""
    return (
        "HERMES_HOME 指向生产存储目录：%s —— 已拒绝。\n"
        "  · 隔离存储：请把 HERMES_HOME 指向一个隔离目录。\n"
        "  · 确实要直接操作生产存储：请显式 opt-in —— 设 HGM_ALLOW_PRODUCTION=1，"
        "或调用 bootstrap(..., allow_production=True)。\n"
        "  · 本机 HERMES_HOME 是用户级环境变量，一个测试会话会**默认**继承它 —— "
        "所以这一条不是多余检查。" % hermes_home)


def bootstrap(script_file: str, *, allow_production: Optional[bool] = None) -> Path:
    """解析 ``HERMES_HOME`` 并加载对应的 ``.env``；返回 home 目录。

    隔离契约（详见模块文档）：

    * ``HERMES_HOME`` **未设** ⇒ 抛 :class:`HermesHomeNotConfiguredError`，
      绝不静默回退到生产目录。
    * 解析出的 home **就是生产目录** ⇒ 同样必须显式 opt-in
      （``HGM_ALLOW_PRODUCTION=1`` 或 ``allow_production=True``），否则拒绝；
      允许时打一条醒目 WARNING，点明当前用的是哪个 home。
    * ``.env`` **不得**改写已显式设置的环境变量（尤其 ``HERMES_HOME``）；
      函数结束前会把 ``os.environ["HERMES_HOME"]`` **重新对齐**到解析结果，
      杜绝"env 变量与 home 分叉"（索引写生产、vault 读 tmp 的那条事故链）。

    Parameters
    ----------
    script_file:
        调用方脚本的 ``__file__``。用于定位"仓库自带"的 ``.env``（开发检出）——
        部署时 ``scripts/`` 就住在 ``$HERMES_HOME/scripts`` 下，那时"仓库自带的
        ``.env``"**就是** home 的 ``.env``。
    allow_production:
        ``None``（默认）表示"听 ``HGM_ALLOW_PRODUCTION`` 的"；``True`` 强制允许
        指向生产（仍会打 WARNING）；``False`` 强制不允许。
    """
    script_dir = Path(script_file).resolve().parent
    repo_or_home = (script_dir.parent if script_dir.name.lower() == "scripts"
                    else script_dir)

    raw_home = (os.environ.get("HERMES_HOME") or "").strip()
    allow = production_allowed(allow_production)
    strict = strict_isolation()

    if raw_home:
        hermes_home = Path(raw_home).expanduser()
    else:
        if strict and not allow:
            raise HermesHomeNotConfiguredError(_missing_home_message())
        hermes_home = production_home()
        logger.warning(
            "[HERMES_ENV][生产] HERMES_HOME 未设置，按平台默认指向生产目录：%s。"
            "本次运行会直接读写生产存储。", hermes_home)

    if is_production_home(hermes_home):
        if strict and not allow:
            raise HermesHomeNotConfiguredError(_production_home_message(hermes_home))
        logger.warning(
            "[HERMES_ENV][生产] 本次运行指向生产目录：%s。请确认这是有意为之 —— "
            "写操作会落到生产数据上。", hermes_home)

    # ``.env`` 只提供"数据"，绝不提供"路径决定权"：它不得改写已显式设置的变量。
    # 顺序：先 home 的（更权威：它就是这个存储的配置），再仓库自带的 —— 部署时
    # 两者是同一个文件，开发检出时 home 的配置说了算（与事故前"home 后加载、
    # 覆盖仓库值"的结果一致，但不再覆盖**进程里已设**的值）。
    load_dotenv(hermes_home / ".env")
    load_dotenv(repo_or_home / ".env")

    # 重新对齐：env 变量与解析结果永远一致（``.env`` 里的 HERMES_HOME 不作数）。
    os.environ["HERMES_HOME"] = str(hermes_home)

    _warn_if_vault_mismatched(hermes_home)

    for subdir in _BOOTSTRAP_SUBDIRS:
        (hermes_home / subdir).mkdir(parents=True, exist_ok=True)

    return hermes_home


@contextlib.contextmanager
def hermes_env_scope(script_file: str,
                     *,
                     allow_production: Optional[bool] = None) -> Iterator[Path]:
    """``bootstrap()`` 的作用域版本：退出时**还原**它改动过的所有环境变量。

    用途：测试与短命脚本。``.env`` 里的生产密钥不得留在进程里、更不得污染同进程
    后续所有测试 —— 退出时逐键还原（此前不存在的键**删除**，而不是写回空串：
    空串会被下游读成「已设置但为空」）。

    用法::

        with hermes_env_scope(__file__) as home:
            ...  # home 是隔离目录；退出后 os.environ 回到进来时的样子
    """
    snapshot = dict(os.environ)
    try:
        yield bootstrap(script_file, allow_production=allow_production)
    finally:
        _restore_snapshot(snapshot)


def _restore_snapshot(snapshot: Dict[str, str]) -> None:
    """把 ``os.environ`` 还原成 ``snapshot`` 的样子（多出来的删掉、改动的改回）。"""
    for key in [key for key in os.environ if key not in snapshot]:
        del os.environ[key]
    for key, value in snapshot.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# 其它小工具（保持原有语义）
# ---------------------------------------------------------------------------

def setup_logging(name: str | None = None, level: int | None = None) -> logging.Logger:
    if level is None:
        env_level = os.getenv("HERMES_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    fmt = os.getenv("HERMES_LOG_FORMAT", "%(asctime)s %(name)s %(levelname)s %(message)s")
    datefmt = os.getenv("HERMES_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger("hermes_governed")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)

    # 局部名不叫 ``logger`` —— 模块级已有同名的模块 logger，遮蔽会让读者以为
    # 拿到的是同一个对象。
    script_logger = logging.getLogger(f"hermes_governed.{name}") if name else root
    return script_logger


def env_path(key: str, default: Path) -> Path:
    """Read an env var as a Path, falling back to default."""
    val = os.environ.get(key)
    return Path(val) if val else default


def env_int(key: str, default: int = 0) -> int:
    """Read an env var as an int, falling back to default on error."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def load_config(hermes_home: Path) -> dict:
    """Load config.yaml (minimal loader, no PyYAML dependency)."""
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: basic key: value parsing
        config = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                config[key.strip()] = value.strip().strip('"').strip("'")
        return config
