"""CLI kb-search must use the plugin's hybrid rank, not its own keyword scan."""

from __future__ import annotations

import pytest

from pathlib import Path

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import (
    KnowledgeBase,
)




@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """把 ``HERMES_HOME`` 指到本测试的 tmp 目录。

    ``cmd_kb_search`` 的**索引路径**来自 ``HERMES_HOME``（``wiki_dir`` 才是
    调用方给的）⇒ 不隔离的话，这些用例会共用同一个索引，于是别的用例写进去
    的行会出现在本用例的检索结果里（"库里没这篇"和"这篇在别的用例里"长得
    一模一样）。旧实现靠"把不是我的行全删掉"把污染盖住了 —— 那正是本次
    P0 要禁的删除行为；护栏修好后，污染必须从**测试隔离**上解。
    """
    home = tmp_path / "home"
    (home / "memory").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home

def _vault(tmp_path: Path) -> dict:
    wiki = tmp_path / "wiki"
    notes = wiki / "notes"
    notes.mkdir(parents=True)
    (notes / "目标笔记.md").write_text(
        "---\ntitle: \"目标笔记\"\n---\n\n飞牛 NAS 硬件配置的正文说明。\n",
        encoding="utf-8")
    (notes / "杂谈.md").write_text(
        "---\ntitle: \"杂谈\"\n---\n\n提到硬件两个字很多次的一次闲聊，与飞牛无关。\n",
        encoding="utf-8")
    return {"wiki_dir": str(wiki), "kb": {"recall_min_kw_score": 0.0}}


def test_empty_body_notes_never_surface(tmp_path):
    """A scaffold file with an empty body matches nothing on either channel.

    The vector store proposes its ``limit`` nearest neighbours for every
    query, so an empty note - whose embedding carries no meaning - can be
    ranked first for an unrelated search. Three such files were doing exactly
    that in the live vault.
    """
    import memory_cli

    wiki = tmp_path / "wiki"
    notes = wiki / "notes"
    notes.mkdir(parents=True)
    (notes / "目标笔记.md").write_text(
        "---\ntitle: \"目标笔记\"\n---\n\n飞牛 NAS 硬件配置的正文说明。\n",
                        encoding="utf-8")
    (notes / "空壳.md").write_text(
        "---\nstatus: \"curated\"\n---\n\n\n\n", encoding="utf-8")

    config = {"wiki_dir": str(wiki), "kb": {"recall_min_kw_score": 0.0}}
    for q in ("飞牛 NAS 硬件配置", "量子计算 拓扑"):
        hits = memory_cli.cmd_kb_search(config, q, 5, "")
        titles = [h["title"] for h in hits]
        assert "空壳" not in titles, f"{q}: empty note surfaced ({titles})"


def test_semantic_floor_is_configurable_and_defaults_off():
    """``kb.semantic_min_score`` exists, defaults to 0, and is applied."""
    from plugin.memory_governed._config import load_governed_config

    cfg = load_governed_config("Z:/nonexistent-probe")
    assert cfg.kb.semantic_min_score == 0.0


def test_keyword_and_semantic_scores_are_reported(tmp_path):
    """The CLI must expose both channel scores, not a bare term count."""
    import memory_cli

    config = _vault(tmp_path)
    hits = memory_cli.cmd_kb_search(config, "飞牛 NAS 硬件配置", 5, "", [])
    assert hits, hits
    top = hits[0]
    assert top["title"] == "目标笔记"
    assert top["keyword_score"] > 0
    assert "semantic_score" in top
    assert "kind" in top


def test_an_exact_title_beats_a_note_that_merely_mentions_more(tmp_path):
    """One store, one ordering: the plugin's rank, not raw term coverage."""
    import memory_cli

    config = _vault(tmp_path)
    hits = memory_cli.cmd_kb_search(config, "飞牛 NAS 硬件配置", 5, "", [])
    # A bag-of-words scan can rank a chatty note first; the hybrid ranker must not.
    assert hits[0]["title"] == "目标笔记"


def test_hybrid_failure_falls_back_to_keyword(tmp_path, monkeypatch):
    """A bare interpreter must still answer — and say why it degraded."""
    import memory_cli

    config = _vault(tmp_path)

    def _boom(*a, **kw):
        raise ImportError("no lancedb on this interpreter")

    monkeypatch.setattr(memory_cli, "plugin_config", _boom)
    errors: list = []
    hits = memory_cli.cmd_kb_search(config, "飞牛 NAS 硬件配置", 5, "", errors)

    assert hits, "the keyword fallback must still find the note"
    assert hits[0]["title"] == "目标笔记"
    assert any("degraded to keyword scan" in e for e in errors), errors


def test_the_fallback_and_the_primary_are_the_same_search(tmp_path, monkeypatch):
    """The fallback must be the pre-existing scan, not a new second implementation."""
    import memory_cli

    config = _vault(tmp_path)
    monkeypatch.setattr(memory_cli, "plugin_config",
                        lambda *a, **kw: (_ for _ in ()).throw(ImportError("x")))
    direct = memory_cli._kb_keyword_search(config, "飞牛 NAS 硬件配置", 5, "", [])
    via_cli = memory_cli.cmd_kb_search(config, "飞牛 NAS 硬件配置", 5, "", [])
    assert direct == via_cli


def test_plugin_search_is_reachable_for_the_hybrid_path(tmp_path):
    """The plugin the CLI delegates to accepts the same vault."""
    from plugin.memory_governed._config import load_governed_config  # noqa: F401

    config = _vault(tmp_path)
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = config["wiki_dir"]
    kb = KnowledgeBase(cfg)
    out = kb.search("飞牛 NAS 硬件配置", top_k=5)
    assert out
    assert "keyword_score" in out[0]
    assert "semantic_score" in out[0]
