# -*- coding: utf-8 -*-
"""KB index location must not depend on the process cwd."""

from __future__ import annotations

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase


def test_relative_l2_path_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = "memory/l2"      # 相对路径
    with pytest.raises(ValueError, match="absolute"):
        KnowledgeBase(cfg)


def test_empty_l2_path_uses_hermes_home_not_cwd(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = ""
    kb = KnowledgeBase(cfg)
    assert kb._index_db == home / "memory" / "kb_index"
    assert not (tmp_path / "kb_index").exists()


def test_empty_l2_path_without_hermes_home_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = ""
    with pytest.raises(ValueError, match="HERMES_HOME"):
        KnowledgeBase(cfg)
