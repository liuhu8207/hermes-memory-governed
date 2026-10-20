# -*- coding: utf-8 -*-
"""kb_eval 的指标计算：正样本与负样本分别计分。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "kb_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("kb_eval_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _KB:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, q, top_k=5):
        return [{"title": t} for t in self.mapping.get(q, [])]


def test_positive_and_negative_metrics():
    mod = _load()
    kb = _KB({
        "q1": ["A"],
        "q2": ["X", "B"],
        "q3": ["X", "Y"],
        "neg": ["noise"],
    })
    gold = [
        {"q": "q1", "expect": ["A"]},
        {"q": "q2", "expect": ["B"]},
        {"q": "q3", "expect": ["C"]},
        {"q": "neg", "expect": []},
        {"q": "neg2", "expect": []},
    ]
    out = mod.evaluate(kb, gold, 5)
    assert out["p_at_1"] == 0.3333
    assert out["recall_at_5"] == 0.6667
    assert out["negative_false_positive_rate"] == 0.5
    assert len(out["misses"]) == 1
    assert len(out["negative_false_positives"]) == 1


def test_all_misses_are_reported():
    mod = _load()
    out = mod.evaluate(_KB({}), [{"q": "x", "expect": ["A"]}], 5)
    assert out["p_at_1"] == 0.0
    assert out["recall_at_5"] == 0.0
    assert out["misses"][0]["q"] == "x"



def test_zero_positives_is_a_failure_not_an_empty_scoreboard():
    mod = _load()
    out = mod.validate({"positives": 0, "ok": True}, {
        "semantic_enabled": True, "index_available": True})
    assert out["ok"] is False
    assert "no positive queries" in out["error"]


def test_unavailable_semantic_channel_is_a_failure():
    mod = _load()
    out = mod.validate({"positives": 5, "ok": True}, {
        "semantic_enabled": True, "index_available": False})
    assert out["ok"] is False
    assert "semantic channel" in out["error"]


def test_healthy_channels_keep_ok():
    mod = _load()
    out = mod.validate({"positives": 5, "ok": True}, {
        "semantic_enabled": True, "index_available": True,
        "reranker_available": True})
    assert out["ok"] is True
    assert "error" not in out
