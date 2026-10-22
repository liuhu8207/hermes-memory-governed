# -*- coding: utf-8 -*-
"""Deterministic query-entry routing tests."""

from __future__ import annotations

import json

from plugin.memory_governed._config import GovernedMemoryConfig, RoutingConfig
from plugin.memory_governed._config import load_governed_config
from plugin.memory_governed._kb import KnowledgeBase
from plugin.memory_governed._routing import QueryRouter


def _router(enabled=True) -> QueryRouter:
    return QueryRouter(RoutingConfig(enabled=enabled))


def test_router_disabled_is_passthrough():
    route = _router(False).route("SR588 支持哪些 CPU")
    assert route.enabled is False
    assert route.mode == "general"
    assert route.exact_terms == ()


def test_config_loads_and_clamps_routing(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "governed_memory.json").write_text(json.dumps({
        "routing": {
            "enabled": True,
            "max_exact_terms": 100,
            "exact_keyword_threshold": 2.0,
            "max_exact_candidates": 0,
        }
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = load_governed_config(home)
    assert cfg.routing.enabled is True
    assert cfg.routing.max_exact_terms == 50
    assert cfg.routing.exact_keyword_threshold == 1.0
    assert cfg.routing.max_exact_candidates == 1


def test_router_detects_exact_identifiers_and_versions():
    route = _router().route("查一下 SR588 和 dsh 0.2.0-rc.2，路径 C:/ops/a.md")
    assert route.mode == "exact"
    assert route.fast_keyword is True
    assert "SR588" in route.exact_terms
    assert "0.2.0-rc.2" in route.exact_terms
    assert any(term.endswith("a.md") for term in route.exact_terms)


def test_router_distinguishes_navigation_episodic_and_conceptual():
    assert _router().route("知识库里哪篇写了 cron 维护").mode == "navigational"
    assert _router().route("你记得我上次说过的偏好").mode == "episodic"
    assert _router().route("为什么混合检索和纯向量有什么区别").mode == "conceptual"
    assert _router().route("推荐一部科幻电影").mode == "general"


def _kb(tmp_path, *, max_exact_candidates=5):
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    cfg.routing = RoutingConfig(enabled=True,
                                max_exact_candidates=max_exact_candidates)
    cfg.reranking.enabled = True
    cfg.kb.rerank_min_score = 0.08
    kb = KnowledgeBase(cfg)
    kb.ensure()
    return kb


def test_exact_title_match_skips_reranker(tmp_path, monkeypatch):
    kb = _kb(tmp_path)
    kb.add(title="SR588 支持规格", body="支持 Xeon Silver 处理器。",
           section="notes")

    def _must_not_call():
        raise AssertionError("exact title route should not call reranker")

    monkeypatch.setattr(kb, "_get_reranker", _must_not_call)
    hits = kb.search("SR588 支持哪些 CPU", top_k=5)
    assert hits
    assert hits[0]["title"] == "SR588 支持规格"
    assert hits[0]["trace"]["route"]["mode"] == "exact"
    assert hits[0]["trace"]["route"]["rerank"] == "routed_keyword"


def test_ambiguous_exact_match_keeps_reranker(tmp_path, monkeypatch):
    kb = _kb(tmp_path, max_exact_candidates=2)
    for index in range(3):
        kb.add(title=f"SR588 支持规格 {index}", body="相关处理器说明。",
               section="notes")

    class _Reranker:
        available = True

        def __init__(self):
            self.calls = 0

        def scores(self, query, texts, top_n=None):
            self.calls += 1
            return {i: 0.9 for i in range(len(texts))}

    reranker = _Reranker()
    monkeypatch.setattr(kb, "_get_reranker", lambda: reranker)
    hits = kb.search("SR588 支持哪些 CPU", top_k=5)
    assert hits
    assert reranker.calls == 1
    assert hits[0]["trace"]["route"]["rerank"] == "reranked"
