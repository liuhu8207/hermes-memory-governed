# -*- coding: utf-8 -*-
"""memory_cli eval must not measure a KB gold set as L2 facts."""

from __future__ import annotations

import json

import memory_cli


def _kb_hits(config, query, top_k, section, errors, category=""):
    table = {
        "q1": [{"title": "笔记 A"}],
        "q2": [{"title": "笔记 B"}],
        "neg": [{"title": "噪声"}],
    }
    return table.get(query, [])


def test_list_expect_auto_selects_kb_and_reports_hits(tmp_path, monkeypatch):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([
        {"q": "q1", "expect": ["笔记 A"]},
        {"q": "q2", "expect": ["笔记 B"]},
        {"q": "neg", "expect": []},
    ], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(memory_cli, "cmd_kb_search", _kb_hits)

    out = memory_cli.cmd_eval({}, str(gold), top_k=5)

    assert out["ok"] is True
    assert out["path"] == "kb"
    assert out["hit_at_k"] == 1.0
    assert out["negatives"] == 1
    assert out["negative_false_positive_rate"] == 1.0


def test_explicit_l2_rejects_list_expect(tmp_path):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([{"q": "q1", "expect": ["笔记 A"]}]),
                    encoding="utf-8")

    out = memory_cli.cmd_eval({}, str(gold), path="l2")

    assert out["ok"] is False
    assert "KB gold" in out["error"]


def test_title_map_is_applied_to_both_sides(tmp_path, monkeypatch):
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps([
        {"q": "q1", "expect": ["<PROJECT> 笔记"]},
    ], ensure_ascii=False), encoding="utf-8")
    mapping = tmp_path / "map.json"
    mapping.write_text(json.dumps({"Hermes": "<PROJECT>"}), encoding="utf-8")

    monkeypatch.setattr(
        memory_cli, "cmd_kb_search",
        lambda *a, **k: [{"title": "Hermes 笔记"}])
    out = memory_cli.cmd_eval({}, str(gold), path="kb",
                              title_map=str(mapping))

    assert out["hit_at_k"] == 1.0
    assert out["title_map_used"] is True
