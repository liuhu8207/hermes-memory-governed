# -*- coding: utf-8 -*-
"""Controlled L2 category vocabulary and legacy normalization."""

from plugin.memory_governed._categories import (
    CONTROLLED_CATEGORIES,
    classify_fact_category,
    normalize_legacy_category,
)


def test_controlled_categories_match_kb_taxonomy():
    assert set(CONTROLLED_CATEGORIES) == {
        "environment", "user", "tooling", "workflow",
        "reference", "project", "synthesis", "meta",
    }


def test_fact_classifier_uses_controlled_values():
    assert classify_fact_category("Unraid 导入 TrueNAS ZFS 池前先 export") == "environment"
    assert classify_fact_category("OpenCode 必须配置 OPENCODE_HOME") == "tooling"
    assert classify_fact_category("hermes-memory-governed 的 L2 写入质量门") == "meta"
    assert classify_fact_category("我喜欢先讨论方向再动手") == "user"
    assert classify_fact_category("Project deadline is Friday") == "project"
    assert classify_fact_category("部署流程规定必须先备份") == "workflow"


def test_legacy_categories_normalize_to_controlled_values():
    assert normalize_legacy_category("tech") == "tooling"
    assert normalize_legacy_category("work") == "project"
    assert normalize_legacy_category("life") == "user"
    assert normalize_legacy_category("hgm-ops") == "meta"
    assert normalize_legacy_category("environment") == "environment"
    assert normalize_legacy_category("unknown") == ""
