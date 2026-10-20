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
        self.last_index_status = {"fresh": 0}

    def scan_vault_paths(self):
        return self._paths

    def sync_index(self, force=False):
        self.synced = True
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
    monkeypatch.setattr("sys.argv", ["kb_reindex.py", "--allow-empty"])

    rc = script.main()

    assert rc == 0
    assert fake.synced is True


def test_a_populated_vault_syncs_normally(script, monkeypatch):
    fake = _FakeKB({"notes/a.md": Path("a.md")})
    monkeypatch.setattr(kb_mod, "KnowledgeBase", lambda cfg: fake)
    monkeypatch.setattr("sys.argv", ["kb_reindex.py"])

    rc = script.main()

    assert rc == 0
    assert fake.synced is True
