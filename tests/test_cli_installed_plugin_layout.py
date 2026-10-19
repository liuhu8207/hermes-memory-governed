from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import memory_cli as cli


def test_plugin_module_resolves_installed_hermes_layout(tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    plugins = tmp_path / "plugins"
    package = plugins / "governed"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.setattr(cli, "__file__", str(scripts / "memory_cli.py"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_PLUGIN_CACHE", {})
    monkeypatch.setattr(sys, "path", list(sys.path))
    imported = []
    monkeypatch.setattr(importlib, "import_module",
                        lambda name: imported.append(name) or SimpleNamespace())

    cli.plugin_module("_lifecycle")

    assert imported == ["governed._lifecycle"]
    assert str(plugins) in sys.path
