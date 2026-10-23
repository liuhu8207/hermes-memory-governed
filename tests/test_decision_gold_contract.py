# -*- coding: utf-8 -*-
"""Decision gold set must remain replayable across Jev/local backends."""

import json
from pathlib import Path

GOLD = Path(__file__).resolve().parent.parent / "evals" / "decision_gold.seed.json"


def test_decision_gold_contract():
    data = json.loads(GOLD.read_text(encoding="utf-8"))
    assert data["version"] == 1
    assert data["operation"] == "kb_novelty"
    labels = set(data["labels"])
    assert labels == {"duplicate", "extends", "conflicts", "new", "uncertain"}
    items = data["items"]
    assert len(items) >= 10
    assert len({item["id"] for item in items}) == len(items)
    assert {item["expect"] for item in items} == labels
    for item in items:
        assert item["new_note"]["title"].strip()
        assert item["new_note"]["body"].strip()
        assert isinstance(item["candidates"], list)
        assert item["candidates"]
        assert item["rationale"].strip()
