# -*- coding: utf-8 -*-
"""The ``hgm_transcribe`` MCP tool: its schema, and its delegation to the CLI.

Kept separate from ``test_hgm_mcp.py`` (which owns the live-server protocol
contract) because these are fast, in-process checks: they load the server module
directly and never spawn a host. Everything is patched — no audio, no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "scripts" / "hgm_mcp.py"


def _module():
    spec = importlib.util.spec_from_file_location("hgm_mcp_transcribe_under_test",
                                                  SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_is_listed_with_a_path_argument():
    mod = _module()
    tools = {t["name"]: t for t in mod.TOOLS}
    assert "hgm_transcribe" in tools, "MCP 工具面里没有转写工具"

    tool = tools["hgm_transcribe"]
    assert tool["description"].strip()
    schema = tool["inputSchema"]
    assert schema["type"] == "object"
    assert "path" in schema["properties"]
    assert schema["required"] == ["path"]
    assert "chunk_minutes" in schema["properties"]


def test_tool_is_registered_for_dispatch():
    mod = _module()
    assert mod.HANDLERS.get("hgm_transcribe") is mod.tool_transcribe


def test_description_says_it_stores_nothing():
    """The contract with the caller: raw text out, persist via hgm_kb_add."""
    mod = _module()
    desc = {t["name"]: t for t in mod.TOOLS}["hgm_transcribe"]["description"]
    assert "hgm_kb_add" in desc
    assert "split" in desc.lower()


def test_empty_path_raises():
    mod = _module()
    with pytest.raises(ValueError, match="path is required"):
        mod.tool_transcribe({"path": "   "})
    with pytest.raises(ValueError, match="path is required"):
        mod.tool_transcribe({})


def test_it_delegates_to_the_cli_with_expected_arguments(monkeypatch):
    mod = _module()
    import memory_cli

    captured = {}

    monkeypatch.setattr(memory_cli, "load_config", lambda: {"sentinel": True})

    def _fake_cmd_transcribe(config, path, timeout=None, chunk_minutes=None):
        captured.update(config=config, path=path,
                        timeout=timeout, chunk_minutes=chunk_minutes)
        return {"ok": True, "text": "hi", "chunks": 2}

    monkeypatch.setattr(memory_cli, "cmd_transcribe", _fake_cmd_transcribe)

    out = mod.tool_transcribe({"path": "C:/Users/<user>/rec.m4a",
                               "chunk_minutes": 5})
    payload = json.loads(out)

    assert payload["ok"] is True
    assert captured["config"] == {"sentinel": True}
    assert captured["path"] == "C:/Users/<user>/rec.m4a"
    assert captured["timeout"] is None
    assert captured["chunk_minutes"] == 5


def test_chunk_minutes_is_optional(monkeypatch):
    mod = _module()
    import memory_cli

    captured = {}
    monkeypatch.setattr(memory_cli, "load_config", lambda: {})
    monkeypatch.setattr(memory_cli, "cmd_transcribe",
                        lambda config, path, timeout=None, chunk_minutes=None:
                        captured.update(chunk_minutes=chunk_minutes)
                        or {"ok": True, "text": "t"})

    mod.tool_transcribe({"path": "a.m4a"})
    assert captured["chunk_minutes"] is None
