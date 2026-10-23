# -*- coding: utf-8 -*-
"""Rule labels for the private real-data decision set builder."""

from scripts.decision_gold_build import _candidate, _entities, _public_node, _tokens


def _node(node_id, title, body, *, tags=(), concepts=(), kind="l2"):
    return {
        "id": node_id,
        "kind": kind,
        "status": "active",
        "title": title,
        "body": body,
        "tags": list(tags),
        "concepts": list(concepts),
        "_tokens": _tokens(body),
        "_title_tokens": _tokens(title),
        "_title_entities": _entities(title),
        "_entities": _entities(body),
        "_tags": {x.casefold() for x in tags},
        "_concepts": {x.casefold() for x in concepts},
    }


def test_builder_labels_duplicate_extends_new_and_temporal_uncertainty():
    canonical = _node(
        "kb:a", "SR588 CPU 支持规格",
        "SR588 支持双路 Xeon Scalable 处理器。",
        tags=["sr588", "cpu"], concepts=["Xeon Scalable"], kind="kb")
    duplicate = _node(
        "archive:a", "SR588 CPU 支持规格",
        "SR588 支持双路 Xeon Scalable 处理器。", tags=["sr588", "cpu"])
    extension = _node(
        "kb:b", "SR588 存储扩展",
        "SR588 支持八个 3.5 英寸盘位或十六个 2.5 英寸盘位。",
        tags=["sr588", "storage"], kind="kb")
    unrelated = _node("l2:z", "科幻电影清单", "记录三部计划观看的电影。")
    old_state = _node(
        "archive:old", "SR588 支持规格", "2026-10-20 检测失败：SR588 支持信息不可用。")
    current = _node(
        "current:x", "SR588 支持规格", "2026-10-23 当前检测成功：SR588 支持信息可用。",
        kind="current-state")

    assert _candidate(canonical, duplicate)["label"] == "duplicate"
    assert _candidate(canonical, extension)["label"] == "extends"
    assert _candidate(canonical, unrelated)["label"] == "new"
    assert _candidate(old_state, current)["label"] == "uncertain"


def test_public_node_strips_internal_scoring_fields():
    node = _node("l2:a", "A", "body")
    public = _public_node(node)
    assert set(public) == {"title", "body", "tags", "concepts"}
