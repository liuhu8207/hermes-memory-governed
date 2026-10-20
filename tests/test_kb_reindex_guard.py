# -*- coding: utf-8 -*-
"""kb_reindex must not clear the index when the vault looks empty."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from plugin.memory_governed import _kb as kb_mod

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "kb_reindex.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("kb_reindex_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script(monkeypatch):
    mod = _load_script()
    # 跳过仓库定位（脚本会找 plugin/memory_governed/_kb.py）
    monkeypatch.setattr(mod, "_repo_root", lambda: SCRIPT.parent.parent)
    return mod


class _FakeKB:
    def __init__(self, paths):
        self._paths = paths
        self.synced = False
        self.allow_purge = None
        self.last_index_status = {"fresh": 0}

    def scan_vault_paths(self):
        return self._paths

    def sync_index(self, force=False, allow_purge=False):
        self.synced = True
        self.allow_purge = allow_purge
        return {"ok": True, "errors": []}


def test_empty_vault_is_refused_and_does_not_touch_the_index(script, monkeypatch):
    fake = _FakeKB({})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py"])

    rc = script.main()

    assert rc == 1, "空 vault 必须是非零退出"
    assert fake.synced is False, "空 vault 时绝不能调用 sync_index（会清空索引）"


def test_allow_empty_is_an_explicit_opt_in(script, monkeypatch):
    fake = _FakeKB({})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py", "--allow-empty", "--yes"])

    rc = script.main()

    assert rc == 0
    assert fake.synced is True


def test_allow_empty_needs_yes_in_a_non_interactive_environment(script, monkeypatch):
    """cron / CI 里没有 TTY ⇒ 没人能确认"清空整个索引" ⇒ 必须显式 ``--yes``。

    静默执行破坏性操作是最坏的组合：既没人授权，也没人知道它发生过。
    """
    fake = _FakeKB({})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py", "--allow-empty"])

    rc = script.main()

    assert rc != 0, "非交互环境下 --allow-empty 没有 --yes 必须拒绝"
    assert fake.synced is False, "被拒绝时绝不能调用 sync_index"


def test_allow_empty_is_forwarded_as_allow_purge(script, monkeypatch):
    """``--allow-empty`` 的意图必须**透传**到 ``sync_index(allow_purge=True)``。

    ``KnowledgeBase.sync_index`` 现在默认 fail-closed（扫描为空时拒绝删索引行），
    所以脚本这个"显式确认 vault 就是空的"的开关必须传下去，否则
    ``--allow-empty`` 会变成"清空不了"——一个静默失效的 CLI 参数。
    """
    fake = _FakeKB({})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py", "--allow-empty", "--yes"])

    script.main()

    assert fake.allow_purge is True, "--allow-empty 必须透传成 allow_purge=True"


def test_default_run_does_not_ask_for_a_purge(script, monkeypatch):
    """默认（没有 --allow-empty）绝不能请求清空索引。"""
    fake = _FakeKB({"notes/a.md": Path("a.md")})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py"])

    script.main()

    assert fake.synced is True
    assert fake.allow_purge is False


def test_a_populated_vault_syncs_normally(script, monkeypatch):
    fake = _FakeKB({"notes/a.md": Path("a.md")})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py"])

    rc = script.main()

    assert rc == 0
    assert fake.synced is True
