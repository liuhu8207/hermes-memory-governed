# -*- coding: utf-8 -*-
"""Deployed hgm_mcp.py must locate the checkout via HGM_REPO."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "scripts" / "hgm_mcp.py"


def test_deployed_copy_uses_hgm_repo(tmp_path, monkeypatch):
    scripts = tmp_path / "hermes_home" / "scripts"
    scripts.mkdir(parents=True)
    target = scripts / "hgm_mcp.py"
    shutil.copy2(SOURCE, target)

    monkeypatch.setenv("HGM_REPO", str(REPO))
    spec = importlib.util.spec_from_file_location("hgm_mcp_deployed_probe", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    assert mod.REPO == REPO


def test_without_hgm_repo_it_stays_at_the_deployed_home(tmp_path, monkeypatch):
    scripts = tmp_path / "hermes_home" / "scripts"
    scripts.mkdir(parents=True)
    target = scripts / "hgm_mcp.py"
    shutil.copy2(SOURCE, target)

    monkeypatch.delenv("HGM_REPO", raising=False)
    spec = importlib.util.spec_from_file_location("hgm_mcp_deployed_probe2", target)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    assert mod.REPO == tmp_path / "hermes_home"
