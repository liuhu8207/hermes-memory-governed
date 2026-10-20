# -*- coding: utf-8 -*-
"""``scripts/hermes_env.py`` 的隔离契约（2026-10-20 生产事故后新增）。

事故
----
2026-10-20 19:56，一次全量 pytest 把**生产 KB 索引**从 52 行写坏成 1 行，残留行
是测试夹具 ``notes/好笔记.md``。vault 正本（125 个 .md）完好，丢的只是索引投影。

根因全在 ``scripts/hermes_env.py``：

* ``load_dotenv(overwrite=True)`` —— **默认覆盖**已显式设置的环境变量。测试认真设了
  ``HERMES_HOME=tmp`` 也会被 ``.env`` 里的值顶掉：**隔离不是没设，是被绕过了**。
* ``HERMES_HOME`` 缺失时**静默回退**生产目录（本机它还是**用户级环境变量**，
  一个测试会话会**默认**继承 ⇒ "指向生产"不需要任何人为失误）。
* 写进 ``os.environ`` 后**永不复原** —— 生产 ``.env``（26KB，含 API 密钥）留在测试
  进程里，并污染同进程后续所有测试。
* vault（``wiki_dir``）与索引（``$HERMES_HOME/memory/kb_index``）**可以不是同一对**
  —— 错配正是删空索引的成因，而当时没有任何一处提示。

本文件把这些契约**钉死**，每条都有断言：

1. ``HERMES_HOME`` 未设 ⇒ ``bootstrap()`` 拒绝，**绝不**返回生产路径；
2. ``.env`` 里的 ``HERMES_HOME`` **盖不掉**已显式设置的隔离 home；
3. 隔离 home 下调用 ``bootstrap()`` ⇒ 生产密钥**不进** ``os.environ``；
4. 退出上下文管理器 ⇒ 被设置的键**全部还原**；
5. 显式 opt-in（``HGM_ALLOW_PRODUCTION=1``）⇒ 允许指向生产，且**有 WARNING**；
6. 回归：现有调用 ``bootstrap()`` 的脚本**仍然正常工作**（逐个 ``--help``）。

⚠️ 本文件**绝不允许**碰到真实存储：所有 home 都是 ``tmp_path``，"生产目录"一律用
``monkeypatch.setattr(hermes_env, "production_home", ...)`` 换成一个 tmp 目录。
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "hermes_env.py"

#: 事故里被注入测试进程的那类密钥**名字**（测试里绝不写真值）。
PRODUCTION_SECRET_KEYS = ("SILICONFLOW_API_KEY", "DEEPSEEK_API_KEY")


def _load_module():
    """按路径加载 ``scripts/hermes_env.py``（它不是包）。"""
    spec = importlib.util.spec_from_file_location("hermes_env_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # 必须先登记进 sys.modules：``@dataclass`` 要按 ``cls.__module__`` 回去找模块，
    # 否则解析字符串注解时找不到 ``EnvChange``。
    sys.modules["hermes_env_under_test"] = module
    spec.loader.exec_module(module)
    return module


hermes_env = _load_module()


def _env_warnings(caplog) -> list:
    """本模块发出的 WARNING 文本（只取 ``hermes_governed.env``，不掺别人的日志）。"""
    return [record.getMessage() for record in caplog.records
            if record.name == "hermes_governed.env"
            and record.levelno >= logging.WARNING]


def _bootstrap_scripts() -> list:
    """所有在 import 期调用 ``bootstrap(__file__)`` 的脚本。"""
    return sorted(path for path in (REPO / "scripts").glob("*.py")
                  if "bootstrap(__file__)" in path.read_text(encoding="utf-8"))


def _has_argparse(script: Path) -> bool:
    """脚本是否用 argparse（决定能不能安全地跑 ``--help``）。"""
    return "argparse" in script.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. HERMES_HOME 未设 ⇒ 拒绝
# ---------------------------------------------------------------------------

class TestHomeMustBeExplicit:
    def test_unset_home_is_refused_and_never_returns_production(self, tmp_path,
                                                                 monkeypatch):
        """未设 ``HERMES_HOME`` ⇒ 抛错；**绝不**回退到生产目录。"""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        # 把"生产目录"指向一个 tmp 目录：若实现偷偷回退到它，这里立刻露馅。
        fake_production = tmp_path / "production"
        fake_production.mkdir()
        monkeypatch.setattr(hermes_env, "production_home", lambda: fake_production)

        with pytest.raises(hermes_env.HermesHomeNotConfiguredError) as excinfo:
            hermes_env.bootstrap(SCRIPT)

        message = str(excinfo.value)
        assert "HERMES_HOME" in message, "报错必须点名 HERMES_HOME"
        assert hermes_env.PRODUCTION_HOME_FLAG in message, (
            "报错必须告诉用户怎么显式 opt-in")

        # 拒绝 ⇒ 没有落任何东西：既没建目录，也没设环境变量。
        assert list(fake_production.iterdir()) == [], "拒绝后仍动了生产目录"
        assert os.environ.get("HERMES_HOME") is None

    def test_home_that_is_the_production_directory_is_refused_without_opt_in(
            self, tmp_path, monkeypatch):
        """本机 ``HERMES_HOME`` 是**用户级环境变量**且就指着生产 —— 必须拒绝。"""
        fake_production = tmp_path / "production"
        fake_production.mkdir()
        monkeypatch.setattr(hermes_env, "production_home", lambda: fake_production)
        monkeypatch.setenv("HERMES_HOME", str(fake_production))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with pytest.raises(hermes_env.HermesHomeNotConfiguredError) as excinfo:
            hermes_env.bootstrap(SCRIPT)

        assert "HERMES_HOME" in str(excinfo.value)
        assert list(fake_production.iterdir()) == [], "拒绝后仍动了生产目录"

    def test_missing_home_message_names_both_ways_out(self, tmp_path, monkeypatch):
        """报错文案必须同时给出两条出路：隔离 home / 显式 opt-in。"""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(hermes_env, "production_home",
                            lambda: tmp_path / "production")
        with pytest.raises(hermes_env.HermesHomeNotConfiguredError) as excinfo:
            hermes_env.bootstrap(SCRIPT)
        message = str(excinfo.value)
        assert "HGM_ALLOW_PRODUCTION" in message
        assert "隔离" in message


# ---------------------------------------------------------------------------
# 2. .env 不得改写已显式设置的环境变量
# ---------------------------------------------------------------------------

class TestDotenvCannotOverrideExplicitEnvironment:
    def test_env_file_cannot_redirect_hermes_home(self, tmp_path, monkeypatch):
        """``.env`` 里的 ``HERMES_HOME=/production`` 盖不掉已显式设置的隔离 home。"""
        home = tmp_path / "home"
        home.mkdir()
        production = tmp_path / "production"
        (home / ".env").write_text(
            "HERMES_HOME=%s\n" % production.as_posix(), encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        resolved = hermes_env.bootstrap(SCRIPT)

        assert resolved == home
        assert os.environ["HERMES_HOME"] == str(home)
        assert Path(os.environ["HERMES_HOME"]) != production

    def test_env_var_and_resolved_home_never_diverge(self, tmp_path, monkeypatch):
        """事故的真凶：env 变量与解析出的 home **分叉**。

        分叉时索引写一处、vault 读另一处 → 索引里的行全被判成 orphan 删光。
        """
        home = tmp_path / "home"
        home.mkdir()
        (home / ".env").write_text(
            "HERMES_HOME=%s\n" % (tmp_path / "production").as_posix(),
            encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        resolved = hermes_env.bootstrap(SCRIPT)

        assert Path(os.environ["HERMES_HOME"]) == resolved

    def test_load_dotenv_does_not_overwrite_by_default(self, tmp_path, monkeypatch):
        """``overwrite`` 默认值必须是 ``False``。"""
        path = tmp_path / ".env"
        path.write_text("HERMES_HOME=/production\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        changes = hermes_env.load_dotenv(path)

        assert changes == [], "默认不该写入任何键"
        assert os.environ["HERMES_HOME"] == str(tmp_path / "home")

    def test_load_dotenv_overwrite_is_still_available_but_explicit(
            self, tmp_path, monkeypatch):
        """显式 ``overwrite=True`` 仍然可用（逃生舱），但必须由调用方点名。"""
        path = tmp_path / ".env"
        path.write_text("HERMES_HOME=/production\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))

        changes = hermes_env.load_dotenv(path, overwrite=True)

        assert [change.key for change in changes] == ["HERMES_HOME"]
        assert os.environ["HERMES_HOME"] == "/production"


# ---------------------------------------------------------------------------
# 3. 生产密钥不得进入测试进程
# ---------------------------------------------------------------------------

class TestProductionSecretsStayOutOfTheProcess:
    def test_isolated_home_does_not_inject_production_secrets(self, tmp_path,
                                                               monkeypatch):
        """隔离 home 下调用 ``bootstrap()`` ⇒ 生产 ``.env`` 的密钥不进 ``os.environ``。"""
        production = tmp_path / "production"
        production.mkdir()
        (production / ".env").write_text(
            "SILICONFLOW_API_KEY=sk-production-value\n"
            "DEEPSEEK_API_KEY=sk-production-value-2\n", encoding="utf-8")
        monkeypatch.setattr(hermes_env, "production_home", lambda: production)

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)
        for key in PRODUCTION_SECRET_KEYS:
            monkeypatch.delenv(key, raising=False)

        hermes_env.bootstrap(SCRIPT)

        for key in PRODUCTION_SECRET_KEYS:
            assert key not in os.environ, f"{key} 被注入测试进程"
        assert os.environ["HERMES_HOME"] == str(home)

    def test_refused_bootstrap_injects_nothing(self, tmp_path, monkeypatch):
        """被拒绝的那次调用**一个键都不许留下**（旧版会留下 26KB 生产 .env）。"""
        production = tmp_path / "production"
        production.mkdir()
        (production / ".env").write_text(
            "SILICONFLOW_API_KEY=sk-production-value\n", encoding="utf-8")
        monkeypatch.setattr(hermes_env, "production_home", lambda: production)
        monkeypatch.setenv("HERMES_HOME", str(production))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)
        monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)

        with pytest.raises(hermes_env.HermesHomeNotConfiguredError):
            hermes_env.bootstrap(SCRIPT)

        assert "SILICONFLOW_API_KEY" not in os.environ


# ---------------------------------------------------------------------------
# 4. 上下文管理器：退出时全部还原
# ---------------------------------------------------------------------------

class TestScopeRestoresEnvironment:
    def test_keys_added_inside_the_scope_are_removed(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        (home / ".env").write_text(
            "HGM_SCOPE_ONLY=secret\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("HGM_SCOPE_ONLY", raising=False)
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with hermes_env.hermes_env_scope(SCRIPT) as resolved:
            assert resolved == home
            assert os.environ["HGM_SCOPE_ONLY"] == "secret"
            os.environ["HGM_ADDED_IN_SCOPE"] = "x"

        assert "HGM_SCOPE_ONLY" not in os.environ, "``.env`` 的键必须被还原"
        assert "HGM_ADDED_IN_SCOPE" not in os.environ, "作用域内新增的键必须被还原"

    def test_hermes_home_is_restored(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with hermes_env.hermes_env_scope(SCRIPT):
            assert os.environ["HERMES_HOME"] == str(home)

        assert os.environ["HERMES_HOME"] == str(home)

    def test_pre_existing_values_survive_the_scope(self, tmp_path, monkeypatch):
        """作用域**之前**就在环境里的值，退出后必须一字不差。"""
        home = tmp_path / "home"
        home.mkdir()
        (home / ".env").write_text("HGM_SCOPE_ONLY=from-dotenv\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HGM_SCOPE_ONLY", "original")
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with hermes_env.hermes_env_scope(SCRIPT):
            # ``.env`` 不得改写已显式设置的值（事故的 (a)）。
            assert os.environ["HGM_SCOPE_ONLY"] == "original"

        assert os.environ["HGM_SCOPE_ONLY"] == "original"

    def test_load_dotenv_reports_what_it_wrote_and_can_restore(self, tmp_path,
                                                               monkeypatch):
        path = tmp_path / ".env"
        path.write_text("HGM_ONE=1\nHGM_TWO=2\n", encoding="utf-8")
        monkeypatch.delenv("HGM_ONE", raising=False)
        monkeypatch.delenv("HGM_TWO", raising=False)

        changes = hermes_env.load_dotenv(path)

        assert {change.key for change in changes} == {"HGM_ONE", "HGM_TWO"}
        assert os.environ["HGM_ONE"] == "1"
        hermes_env.restore_env(changes)
        assert "HGM_ONE" not in os.environ
        assert "HGM_TWO" not in os.environ


# ---------------------------------------------------------------------------
# 5. 显式 opt-in：允许指向生产，且必须打 WARNING
# ---------------------------------------------------------------------------

class TestExplicitProductionOptIn:
    def test_flag_allows_falling_back_to_production_with_warning(
            self, tmp_path, monkeypatch, caplog):
        fake_production = tmp_path / "production"
        fake_production.mkdir()
        monkeypatch.setattr(hermes_env, "production_home", lambda: fake_production)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv(hermes_env.PRODUCTION_HOME_FLAG, "1")

        with caplog.at_level(logging.WARNING, logger="hermes_governed.env"):
            resolved = hermes_env.bootstrap(SCRIPT)

        assert resolved == fake_production
        warnings = _env_warnings(caplog)
        assert warnings, "允许指向生产时必须打 WARNING"
        assert any(str(fake_production) in message for message in warnings), (
            "WARNING 必须点明当前用的是哪个 home")

    def test_flag_allows_an_explicit_production_home_with_warning(
            self, tmp_path, monkeypatch, caplog):
        fake_production = tmp_path / "production"
        fake_production.mkdir()
        monkeypatch.setattr(hermes_env, "production_home", lambda: fake_production)
        monkeypatch.setenv("HERMES_HOME", str(fake_production))
        monkeypatch.setenv(hermes_env.PRODUCTION_HOME_FLAG, "1")

        with caplog.at_level(logging.WARNING, logger="hermes_governed.env"):
            resolved = hermes_env.bootstrap(SCRIPT)

        assert resolved == fake_production
        assert _env_warnings(caplog), "允许指向生产时必须打 WARNING"

    def test_flag_is_not_read_from_any_env_file(self, tmp_path, monkeypatch):
        """opt-in 只能来自**进程调用** —— 否则加载器就能自我授权。"""
        fake_production = tmp_path / "production"
        fake_production.mkdir()
        monkeypatch.setattr(hermes_env, "production_home", lambda: fake_production)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)
        (tmp_path / ".env").write_text(
            "%s=1\n" % hermes_env.PRODUCTION_HOME_FLAG, encoding="utf-8")

        with pytest.raises(hermes_env.HermesHomeNotConfiguredError):
            hermes_env.bootstrap(SCRIPT)


# ---------------------------------------------------------------------------
# 6. vault 与索引必须同源（错配要提示）
# ---------------------------------------------------------------------------

class TestVaultAndIndexMustShareOneSource:
    def test_mismatched_vault_is_reported(self, tmp_path, monkeypatch, caplog):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("GOVERNED_WIKI_DIR", str(tmp_path / "elsewhere" / "wiki"))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with caplog.at_level(logging.WARNING, logger="hermes_governed.env"):
            hermes_env.bootstrap(SCRIPT)

        warnings = _env_warnings(caplog)
        assert any("错配" in message for message in warnings), (
            "vault 与索引不同源必须提示 —— 这正是删空索引的成因")
        assert any("wiki" in message for message in warnings)

    def test_same_source_does_not_warn(self, tmp_path, monkeypatch, caplog):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("GOVERNED_WIKI_DIR", raising=False)
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with caplog.at_level(logging.WARNING, logger="hermes_governed.env"):
            hermes_env.bootstrap(SCRIPT)

        assert _env_warnings(caplog) == [], "同源时不该报 WARNING"

    def test_vault_declared_by_the_home_itself_is_not_a_mismatch(
            self, tmp_path, monkeypatch, caplog):
        """生产部署就是这种布局：vault 在 Obsidian 目录（home 之外），索引在 home 里。

        vault 由 **home 自己的配置**声明 ⇒ 按构造同源 ⇒ **不该**报 WARNING。
        """
        home = tmp_path / "home"
        home.mkdir()
        vault = tmp_path / "obsidian-vault"
        vault.mkdir()
        (home / "governed_memory.json").write_text(
            json.dumps({"wiki_dir": str(vault)}), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.delenv("GOVERNED_WIKI_DIR", raising=False)
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with caplog.at_level(logging.WARNING, logger="hermes_governed.env"):
            hermes_env.bootstrap(SCRIPT)

        assert _env_warnings(caplog) == [], (
            "home 自己声明的 vault 是合法布局，不该被当成错配")

    def test_override_contradicting_the_home_config_is_a_mismatch(
            self, tmp_path, monkeypatch, caplog):
        """环境变量把 vault 指到别处、而 home 的配置说的是另一个 ⇒ 错配。"""
        home = tmp_path / "home"
        home.mkdir()
        declared = tmp_path / "declared-vault"
        declared.mkdir()
        (home / "governed_memory.json").write_text(
            json.dumps({"wiki_dir": str(declared)}), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("GOVERNED_WIKI_DIR", str(tmp_path / "other" / "wiki"))
        monkeypatch.delenv(hermes_env.PRODUCTION_HOME_FLAG, raising=False)

        with caplog.at_level(logging.WARNING, logger="hermes_governed.env"):
            hermes_env.bootstrap(SCRIPT)

        assert any("错配" in message for message in _env_warnings(caplog))


# ---------------------------------------------------------------------------
# 7. 回归：现有调用 bootstrap() 的脚本仍然正常工作
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def script_runs(tmp_path_factory):
    """每个脚本都在**隔离 home**下跑一次，结果缓存（一个模块只跑一遍）。"""
    home = tmp_path_factory.mktemp("script_home")
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env.pop(hermes_env.PRODUCTION_HOME_FLAG, None)

    results = {}
    for script in _bootstrap_scripts():
        if _has_argparse(script):
            command = [sys.executable, str(script), "--help"]
        else:
            # 没有 argparse 的脚本不能跑 ``--help``（会真的执行主流程）——
            # 只做一次"导入即成功"的检查。``runpy`` 不会像 ``python x.py`` 那样把
            # 脚本所在目录放进 ``sys.path``，所以这里手动补上（脚本要 import
            # 同目录的 ``hermes_env``）。
            command = [sys.executable, "-c",
                       "import runpy,sys;sys.path.insert(0, %r);sys.argv=[%r];"
                       "runpy.run_path(%r, run_name='__not_main__')"
                       % (str(script.parent), str(script), str(script))]
        results[script.name] = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300,
            env=env, cwd=str(REPO))
    return results


@pytest.mark.parametrize("script_name", [path.name for path in _bootstrap_scripts()])
def test_script_still_runs(script_name, script_runs):
    """回归：``bootstrap()`` 改了行为之后，这些脚本**照常**能跑起来。"""
    process = script_runs[script_name]
    assert process.returncode == 0, (
        "%s 跑不起来：\n--- stdout ---\n%s\n--- stderr ---\n%s"
        % (script_name, process.stdout, process.stderr))


def test_script_refuses_loudly_when_home_is_unset():
    """回归（反面）：``HERMES_HOME`` 没设时，脚本必须**响亮地拒绝**，而不是猜生产。"""
    env = dict(os.environ)
    env.pop("HERMES_HOME", None)
    env.pop(hermes_env.PRODUCTION_HOME_FLAG, None)
    process = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "memory_pipeline.py"), "--help"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=300, env=env, cwd=str(REPO))

    assert process.returncode != 0, "没设 HERMES_HOME 时不该继续跑"
    assert "HERMES_HOME" in process.stderr, (
        "拒绝必须说清楚原因：\n%s" % process.stderr)
    assert hermes_env.PRODUCTION_HOME_FLAG in process.stderr
