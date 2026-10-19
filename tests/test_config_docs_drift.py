# -*- coding: utf-8 -*-
"""文档 / 示例配置与 ``_config.py`` 的字段一致性守卫。

背景：``recall.l2_min_score`` / ``l2_fusion`` / ``kb.fusion`` /
``kb.mmr_lambda`` / ``kb.affinity_enabled`` / ``kb.semantic_min_score`` /
``kb.rerank_min_score`` 等字段在 ``_config.py`` 里早已落地并被部署实际开启，
但 ``docs/configuration.md`` 与 ``config/governed_memory.example.json`` 都没跟上
—— 照文档抄配置的人拿不到这些开关，也读不到它们的默认值和标定依据。

本模块把「代码即正本」钉住：
* ``_config.py`` 的每个 dataclass 字段，示例配置里都要出现（或其默认值可见）；
* 示例配置出现的键，必须是真实字段（防手滑拼错）；
* ``docs/configuration.md`` 必须逐一提及这些字段名（防文档再次静默过期）。
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from plugin.memory_governed import _config

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / "config" / "governed_memory.example.json"
DOC = REPO / "docs" / "configuration.md"

#: ``_config.py`` 里这些 section 名 -> JSON 里的键名。
SECTION_KEY = {
    "RecallConfig": "recall",
    "SyncConfig": "sync",
    "PersonaConfig": "persona",
    "TextractConfig": "tencent_extract",
    "EmbeddingConfig": "embedding",
    "RerankingConfig": "reranking",
    "MermaidConfig": "mermaid_compress",
    "VectorConfig": "vector",
    "KnowledgeConfig": "kb",
    "AsrConfig": "asr",
    "SynthesisConfig": "synthesis",
}

#: 只在运行时推导、不应出现在示例配置里的字段。
RUNTIME_ONLY = {"project_scope"}


def _example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def _doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


def _field_names(cls) -> set:
    return {f.name for f in dataclasses.fields(cls)}


class TestExampleCoversEveryCodeField:
    @pytest.mark.parametrize("cls_name", sorted(SECTION_KEY))
    def test_every_field_is_listed(self, cls_name):
        """示例配置要么写出该键，要么文档写明它靠默认值。"""
        cls = getattr(_config, cls_name)
        key = SECTION_KEY[cls_name]
        section = _example().get(key, {})
        listed = set(section)
        expected = _field_names(cls) - RUNTIME_ONLY

        missing = sorted(expected - listed)
        assert not missing, (
            "%s 有字段没进示例配置: %s" % (key, ", ".join(missing))
        )

    def test_no_key_in_example_that_code_does_not_read(self):
        """示例配置里的键必须对应真实字段（防拼错 / 防过期残留）。"""
        example = _example()
        for cls_name, key in SECTION_KEY.items():
            cls = getattr(_config, cls_name)
            known = _field_names(cls)
            for k in example.get(key, {}):
                assert k in known, (
                    "%s.%s 不是 _config.py 的字段" % (key, k)
                )


class TestDocsNameEveryTunable:
    @pytest.mark.parametrize("cls_name", sorted(SECTION_KEY))
    def test_field_is_documented(self, cls_name):
        cls = getattr(_config, cls_name)
        doc = _doc_text()
        missing = sorted(
            n for n in _field_names(cls) - RUNTIME_ONLY if n not in doc
        )
        assert not missing, (
            "docs/configuration.md 没提到 %s 的字段: %s"
            % (SECTION_KEY[cls_name], ", ".join(missing))
        )

    def test_the_two_gates_keep_their_own_names(self):
        """关键词与融合分是两条独立走廊，文档不能把它们混成一个阈值。"""
        doc = _doc_text()
        for name in ("recall_min_score", "recall_min_kw_score"):
            assert name in doc
        assert "两条" in doc or "two gates" in doc.lower() or "OR" in doc
