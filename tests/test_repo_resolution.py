# -*- coding: utf-8 -*-
"""Deployed scripts locate the checkout via source_repo / HGM_REPO."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from scripts import hermes_env

REPO = Path(__file__).resolve().parent.parent


def _deployed(tmp_path):
    scripts = tmp_path / "hermes_home" / "scripts"
    scripts.mkdir(parents=True)
    return scripts / "cron_memory_daily.py"


def test_source_repo_in_governed_config(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "governed_memory.json").write_text(
        json.dumps({"source_repo": str(REPO)}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HGM_REPO", raising=False)

    assert hermes_env.resolve_repo(str(_deployed(tmp_path))) == REPO


def test_hgm_repo_wins(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "governed_memory.json").write_text(
        json.dumps({"source_repo": str(tmp_path / "wrong")}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HGM_REPO", str(REPO))

    assert hermes_env.resolve_repo(str(_deployed(tmp_path))) == REPO


def test_missing_raises_a_clear_error(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    (home / "governed_memory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HGM_REPO", raising=False)

    try:
        hermes_env.resolve_repo(str(_deployed(tmp_path)))
    except hermes_env.HGMRepoNotConfiguredError as exc:
        assert "source_repo" in str(exc)
    else:
        raise AssertionError("resolve_repo should fail without a checkout")


def test_cron_build_env_drops_pythonpath(monkeypatch):
    path = REPO / "scripts" / "cron_memory_daily.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("cron_env_probe", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    monkeypatch.setenv("PYTHONPATH", r"C:\wrong\path")
    env = mod._build_env()
    assert "PYTHONPATH" not in env
    assert env["HGM_REPO"] == str(REPO)
