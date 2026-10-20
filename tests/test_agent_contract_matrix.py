# -*- coding: utf-8 -*-
"""多入口契约：插件 / CLI / MCP 对同一份 vault 必须给出同一结果。"""

from __future__ import annotations

import json
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

import memory_cli
from plugin.memory_governed import _vault
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase


def _kb(tmp_path):
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "l2")
    _vault.ensure_skeleton(_vault.vault_root(cfg))
    return cfg, KnowledgeBase(cfg)


def test_plugin_and_cli_moc_are_byte_identical(tmp_path):
    cfg, kb = _kb(tmp_path)
    # frontmatter title 含 "/"，落盘文件名会换成空格 —— 这是别名路径
    _vault.write_note(kb._vault / "notes" / "SR588 配套规格（内存 扩展 存储）.md",
                      {"title": "SR588 配套规格（内存/扩展/存储）",
                       "status": "curated", "category": "reference",
                       "tags": ["服务器"], "concepts": []},
                      "服务器 硬件 规格")
    kb.rebuild_moc()
    plugin_text = (kb._vault / "index.md").read_text(encoding="utf-8")

    memory_cli.cmd_kb_moc({"wiki_dir": str(kb._vault)})
    cli_text = (kb._vault / "index.md").read_text(encoding="utf-8")

    assert plugin_text == cli_text, (
        "插件与 CLI 生成的 MOC 不一致：同一个 vault 两个答案\n"
        "--- plugin ---\n%s\n--- cli ---\n%s" % (plugin_text, cli_text))


def test_three_write_surfaces_normalise_the_same_fields(tmp_path, monkeypatch):
    cfg, kb = _kb(tmp_path)

    # 1) plugin
    r1 = kb.add("契约 插件", "正文一", category="tooling",
                tags=["a"], concepts=["c"])

    # 2) CLI
    r2 = memory_cli.cmd_kb_add(
        {"wiki_dir": cfg.wiki_dir, "kb": {"confidence_threshold": 0.7}},
        "契约 CLI", "正文二", "notes", ["a"], ["c"], None,
        agent="cli-test", category="tooling")

    # 3) MCP
    home = tmp_path / "home"
    (home / "memory" / "l2").mkdir(parents=True)
    (home / "scripts").mkdir()
    shutil.copy(Path(memory_cli.__file__), home / "scripts" / "memory_cli.py")
    (home / "governed_memory.json").write_text(json.dumps({
        "scripts_dir": str(home / "scripts"),
        "wiki_dir": cfg.wiki_dir,
        "kb": {"moc_auto_rebuild": True}}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MEMORY_DIR", str(home / "memory"))
    spec = importlib.util.spec_from_file_location(
        "hgm_mcp_contract",
        Path(__file__).resolve().parent.parent / "scripts" / "hgm_mcp.py")
    mcp = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent.parent / "scripts"))
    spec.loader.exec_module(mcp)
    r3 = json.loads(mcp.tool_kb_add({
        "title": "契约 MCP", "body": "正文三",
        "category": "tooling", "tags": ["a"], "concepts": ["c"]}))

    assert r1["ok"] and r2["ok"] and r3["ok"]

    for name in ("契约 插件", "契约 CLI", "契约 MCP"):
        meta, _ = _vault.read_note(kb._vault / "notes" / (name + ".md"))
        assert meta.get("category") == "tooling", name
        assert meta.get("tags") == ["a"], name
        assert meta.get("concepts") == ["c"], name
        assert meta.get("status") == "curated", name

    index = (kb._vault / "index.md").read_text(encoding="utf-8")
    for name in ("契约 插件", "契约 CLI", "契约 MCP"):
        assert "[[%s]]" % name in index



def test_keyword_search_parity_between_plugin_and_cli(tmp_path):
    """semantic 关闭时，同一查询在两个入口必须给出同样的顺序。"""
    cfg, kb = _kb(tmp_path)
    cfg.kb.semantic_enabled = False
    cfg.kb.keyword_enabled = True
    cfg.kb.affinity_enabled = False
    cfg.reranking.enabled = False
    kb.add("飞牛 NAS 硬件配置", "nas 硬件 内存", category="environment")
    kb.add("本地 PC 硬件", "pc 硬件 环境", category="environment")
    kb.add("ZCode 配置", "zcode provider 配置", category="tooling")

    plugin = [h["title"] for h in kb.search("NAS 硬件", top_k=5)]
    cli_cfg = {"wiki_dir": cfg.wiki_dir,
               "reranking": {"enabled": False},
               "kb": {"semantic_enabled": False, "keyword_enabled": True,
                      "affinity_enabled": False,
                      "moc_auto_rebuild": True}}
    cli = [h["title"] for h in memory_cli.cmd_kb_search(
        cli_cfg, "NAS 硬件", 5, "", [])]
    assert plugin == cli, "插件与 CLI 的检索顺序不一致：%r vs %r" % (plugin, cli)


def test_both_write_surfaces_refuse_secrets(tmp_path):
    """错误路径也要有同一套语义：两处都必须拒绝含密钥的正文。"""
    cfg, kb = _kb(tmp_path)
    secret = "sk-" + "A" * 24

    r1 = kb.add("插件密钥测试", "token = " + secret, category="tooling")
    r2 = memory_cli.cmd_kb_add(
        {"wiki_dir": cfg.wiki_dir}, "CLI 密钥测试", "token = " + secret,
        "notes", [], [], None, agent="cli-test", category="tooling")

    assert r1.get("ok") is False
    assert r2.get("ok") is False
    assert not (kb._vault / "notes" / "插件密钥测试.md").exists()
    assert not (kb._vault / "notes" / "CLI 密钥测试.md").exists()
