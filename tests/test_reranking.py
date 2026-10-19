# -*- coding: utf-8 -*-
"""Reranking（P2）回归测试：API reranker + L2 接入点。

覆盖任务书的 6 条硬要求 + 边界：

1. 未配置 → 直通（原列表、``available`` False、**无** degraded 计数）
2. 配置开 + 调用成功 → 顺序改变、集合不变（重排不能丢条目）
3. 调用抛异常 → 原样返回 + ``rerank::call_failed`` +1（不是抛出来）
4. 返回格式不符 → 原样返回 + ``rerank::bad_response`` +1
5. 开关关 → 直通（哪怕配了 provider，且不发 API 调用）
6. 准入门槛没被动（接入点前后 ``_l2_admitted`` 判定一致；端到端
   ``format_recall`` 注入集合不变）

外加边界：``top_n`` 语义（只提交前 N、尾部原序、不丢条目）、``native_score``
不被融合已存值覆盖、L2 分数带保持（不改层间量纲）、其它层分数不动、
``parallel_recall`` / ``_sequential_recall`` 同一接入点、配置加载与默认关。

全部离线：``httpx.post`` 用 monkeypatch 注入假实现，不打真实网络。
degraded 计数先 ``_diag.reset()``，用 ``_diag.stats()`` 读（**不用**
``grep -rc | awk`` 统计 —— 本机路径含 ``C:`` 会把 ``$2`` 劈歪得 0）。
"""

from __future__ import annotations

import json

import pytest

from plugin.memory_governed import _diag
from plugin.memory_governed._config import (
    GovernedMemoryConfig,
    RerankingConfig,
    load_governed_config,
)
from plugin.memory_governed._recall import (
    RecallEngine,
    RecallResult,
    _l2_admitted,
    layer_score_floor,
)
from plugin.memory_governed._rerank import Reranker

#: 测试专用密钥环境变量名（刻意取一个真实 .env 里不会有的名字）
_KEY_ENV = "GOVERNED_TEST_RERANK_KEY"


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

def _reranking_cfg(**overrides) -> RerankingConfig:
    """一份"生产形态"的 reranking 配置（siliconflow + bge-reranker-v2-m3）。"""
    cfg = RerankingConfig()
    cfg.enabled = True
    cfg.provider = "siliconflow"
    cfg.api_key_env = _KEY_ENV
    cfg.base_url = "https://api.siliconflow.cn/v1"
    cfg.model = "BAAI/bge-reranker-v2-m3"
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class _RerankResp:
    """``/rerank`` 的最小响应（含 ``raise_for_status``）。"""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _HttpErrorResp:
    """非 2xx：``raise_for_status`` 抛（传输层失败 → call_failed）。"""

    def raise_for_status(self):
        raise RuntimeError("503 from fake gateway")

    def json(self):
        return {}


class _BadJsonResp:
    """JSON 解析失败（``json()`` 抛 → call_failed）。"""

    def raise_for_status(self):
        return None

    def json(self):
        raise ValueError("no json for you")


def _reversing_post(url, *, json=None, headers=None, timeout=None):
    """假 API：按**文档下标反转**打分（第 0 篇最低、最后一篇最高）。"""
    docs = json["documents"]
    n = len(docs)
    return _RerankResp({
        "results": [
            {"index": i, "relevance_score": (i + 1) / n}
            for i in range(n)
        ]
    })


def _install_post(monkeypatch, handler) -> None:
    """把 ``httpx.post`` 换成假实现（与 test_read_path / test_env_secret 同款）。"""
    monkeypatch.setattr("httpx.post", handler)


def _capture_documents(store: dict):
    """假 API：记录提交的 ``documents``（验 top_n 语义用），并按下标反转打分。"""
    def _post(url, *, json=None, headers=None, timeout=None):
        store["documents"] = list(json["documents"])
        docs = json["documents"]
        n = len(docs)
        return _RerankResp({
            "results": [
                {"index": i, "relevance_score": (i + 1) / n}
                for i in range(n)
            ]
        })
    return _post


class _ExplodingContent:
    """访问 ``.content`` 即抛 —— 模拟畸形候选（F-A2：``__str__`` 会炸的对象）。"""

    layer = "l2"
    score = 0.9
    source = "lance"
    metadata = None

    @property
    def content(self):
        raise RuntimeError("content explodes")


def _l2_items(*specs):
    """(content, score) → RecallResult 列表（layer=l2）。"""
    return [
        RecallResult(layer="l2", content=content, score=score,
                     source="lance", metadata={"source_rowid": i})
        for i, (content, score) in enumerate(specs)
    ]


def _engine_cfg(tmp_path, reranking: RerankingConfig) -> GovernedMemoryConfig:
    """引擎用配置：reranking 段可指定，其余路径指向不存在的文件（各层静默空）。"""
    cfg = GovernedMemoryConfig()
    cfg.l2_db_path = str(tmp_path / "absent" / "l2")
    cfg.l3_db_path = str(tmp_path / "absent" / "l3.db")
    cfg.l4_persona_path = str(tmp_path / "absent" / "persona.md")
    cfg.reranking = reranking
    return cfg


# ---------------------------------------------------------------------------
# 1) 未配置 → 直通
# ---------------------------------------------------------------------------

def test_unconfigured_passthrough(monkeypatch):
    """未配置（provider 空）：available False、rerank 原样返回、无 degraded。"""
    _diag.reset()
    rer = Reranker(RerankingConfig())
    assert rer.available is False

    items = _l2_items(("a", 0.9), ("b", 0.8))
    out = rer.rerank("查询", items)
    assert [i.content for i in out] == ["a", "b"]
    assert {id(i) for i in out} == {id(i) for i in items}
    # 未配置不是故障：不得产生任何 degraded 计数
    assert _diag.stats()["degraded"] == {}


def test_none_config_passthrough():
    """config 段整个缺失（None）时按未配置处理，不抛。"""
    rer = Reranker(None)
    assert rer.available is False
    items = _l2_items(("a", 0.9), ("b", 0.8))
    assert [i.content for i in rer.rerank("q", items)] == ["a", "b"]


def test_api_key_missing_passthrough(tmp_path, monkeypatch):
    """配了 provider 但密钥缺失（env 无、.env 也无）：直通且无 degraded。"""
    _diag.reset()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))  # 隔离 .env 回退
    monkeypatch.delenv(_KEY_ENV, raising=False)
    rer = Reranker(_reranking_cfg())
    assert rer.available is False

    items = _l2_items(("a", 0.9), ("b", 0.8))
    assert [i.content for i in rer.rerank("q", items)] == ["a", "b"]
    assert _diag.stats()["degraded"] == {}


# ---------------------------------------------------------------------------
# 2) 配置开 + 调用成功 → 顺序改变、集合不变
# ---------------------------------------------------------------------------

def test_success_reorders_and_preserves_set(monkeypatch):
    """mock API 返回反转分数：顺序改变、集合不变、请求形状正确、无 degraded。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    assert rer.available is True

    items = _l2_items(("alpha", 0.9), ("beta", 0.8),
                      ("gamma", 0.7), ("delta", 0.6))
    captured = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        docs = json["documents"]
        n = len(docs)
        return _RerankResp({
            "results": [
                {"index": i, "relevance_score": (i + 1) / n}
                for i in range(n)
            ]
        })

    _install_post(monkeypatch, fake_post)
    out = rer.rerank("查询", items)

    # 顺序改变（按下标反转）
    assert [i.content for i in out] == ["delta", "gamma", "beta", "alpha"]
    # 集合不变：同一批对象，一个不多一个不少
    assert {id(i) for i in out} == {id(i) for i in items}
    # 请求形状：URL / model / query / documents / 鉴权 / 超时
    assert captured["url"] == "https://api.siliconflow.cn/v1/rerank"
    assert captured["json"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["json"]["query"] == "查询"
    assert captured["json"]["documents"] == ["alpha", "beta", "gamma", "delta"]
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["timeout"] == 10
    # 成功路径无 degraded
    assert _diag.stats()["degraded"] == {}


def test_tuple_items_are_supported(monkeypatch):
    """接口文档形态 ``(content, metadata)`` 元组也能重排。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = [("alpha", {"k": 1}), ("beta", {"k": 2})]
    _install_post(monkeypatch, _reversing_post)
    out = rer.rerank("q", items)
    assert [content for content, _meta in out] == ["beta", "alpha"]


# ---------------------------------------------------------------------------
# 3) 调用抛异常 → 原样返回 + rerank::call_failed
# ---------------------------------------------------------------------------

def test_call_failure_passthrough_and_counted(monkeypatch):
    """API 调用抛异常：不抛出、原样返回、rerank::call_failed +1。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = _l2_items(("a", 0.9), ("b", 0.8))

    def boom(url, **kwargs):
        raise RuntimeError("network down")

    _install_post(monkeypatch, boom)
    out = rer.rerank("q", items)  # 不得抛

    assert [i.content for i in out] == ["a", "b"]
    assert _diag.stats()["degraded"].get("rerank::call_failed") == 1


def test_http_error_status_is_call_failed(monkeypatch):
    """非 2xx（raise_for_status 抛）记 call_failed，不是 bad_response。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = _l2_items(("a", 0.9), ("b", 0.8))
    _install_post(monkeypatch, lambda url, **kw: _HttpErrorResp())
    out = rer.rerank("q", items)
    assert [i.content for i in out] == ["a", "b"]
    assert _diag.stats()["degraded"].get("rerank::call_failed") == 1


def test_json_decode_error_is_call_failed(monkeypatch):
    """响应不是合法 JSON：call_failed（传输/解析层），原样返回。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = _l2_items(("a", 0.9), ("b", 0.8))
    _install_post(monkeypatch, lambda url, **kw: _BadJsonResp())
    out = rer.rerank("q", items)
    assert [i.content for i in out] == ["a", "b"]
    assert _diag.stats()["degraded"].get("rerank::call_failed") == 1


# ---------------------------------------------------------------------------
# 4) 返回格式不符 → 原样返回 + rerank::bad_response
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    ["not", "a", "dict"],                                    # 顶层不是 dict
    {"results": "not-a-list"},                               # results 类型错
    {"results": []},                                         # results 为空
    {"results": [{"index": "x", "relevance_score": 0.5}]},   # 无可用条目
    {},                                                      # 缺 results
])
def test_bad_response_passthrough_and_counted(monkeypatch, payload):
    """返回格式不符：原样返回 + rerank::bad_response +1（不是 call_failed）。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = _l2_items(("a", 0.9), ("b", 0.8))
    _install_post(monkeypatch, lambda url, **kw: _RerankResp(payload))
    out = rer.rerank("q", items)  # 不得抛

    assert [i.content for i in out] == ["a", "b"]
    assert _diag.stats()["degraded"].get("rerank::bad_response") == 1
    assert _diag.stats()["degraded"].get("rerank::call_failed") is None


def test_partial_scores_are_usable(monkeypatch):
    """部分条目有分：可用（有分的排前、没分的原序接后），不算 bad_response。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = _l2_items(("a", 0.9), ("b", 0.8), ("c", 0.7))
    payload = {"results": [{"index": 1, "relevance_score": 0.99}, {"junk": True}]}
    _install_post(monkeypatch, lambda url, **kw: _RerankResp(payload))
    out = rer.rerank("q", items)

    assert [i.content for i in out] == ["b", "a", "c"]
    assert {id(i) for i in out} == {id(i) for i in items}
    assert _diag.stats()["degraded"] == {}


# ---------------------------------------------------------------------------
# 5) 开关关 → 直通（哪怕配了 provider）
# ---------------------------------------------------------------------------

def test_disabled_reranker_never_calls_api(monkeypatch):
    """enabled=False：rerank 原样返回，且**不发** API 调用、无 degraded。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg(enabled=False))
    assert rer.available is True  # 配了 provider + 密钥：可用性为真

    calls = []
    _install_post(monkeypatch, lambda url, **kw: calls.append(url))
    items = _l2_items(("a", 0.9), ("b", 0.8))
    out = rer.rerank("q", items)

    assert [i.content for i in out] == ["a", "b"]
    assert calls == []
    assert _diag.stats()["degraded"] == {}


def test_engine_passthrough_when_disabled(tmp_path, monkeypatch):
    """接入点：开关关时 ``_rerank_l2`` 原样返回（同对象），不碰分数/元数据。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg(enabled=False)))
    items = _l2_items(("a", 0.9), ("b", 0.8))
    original_scores = [r.score for r in items]

    out = eng._rerank_l2("q", items)

    assert out is items
    assert [r.score for r in out] == original_scores
    assert all("rerank_rank" not in (r.metadata or {}) for r in out)
    assert all("native_score" not in (r.metadata or {}) for r in out)
    assert _diag.stats()["degraded"] == {}


def test_engine_passthrough_when_unconfigured(tmp_path, monkeypatch):
    """接入点：未配置（provider 空）时直通，无 degraded。"""
    _diag.reset()
    eng = RecallEngine(_engine_cfg(tmp_path, RerankingConfig()))
    items = _l2_items(("a", 0.9), ("b", 0.8))
    out = eng._rerank_l2("q", items)
    assert out is items
    assert _diag.stats()["degraded"] == {}


# ---------------------------------------------------------------------------
# 6) 准入门槛没被动
# ---------------------------------------------------------------------------

def test_rerank_preserves_admission_basis(tmp_path, monkeypatch):
    """接入点前后 ``_l2_admitted`` 判定一致；原生分落袋 native_score。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    cfg = _engine_cfg(tmp_path, _reranking_cfg())
    cfg.recall.l2_min_score = 0.76
    eng = RecallEngine(cfg)
    floor = layer_score_floor("l2", cfg.recall)

    items = _l2_items(("high-1", 0.82), ("low-1", 0.75),
                      ("high-2", 0.90), ("low-2", 0.74))
    original = {r.content: r.score for r in items}
    before = {r.content: _l2_admitted(r, floor) for r in items}
    assert before == {"high-1": True, "low-1": False,
                      "high-2": True, "low-2": False}

    _install_post(monkeypatch, _reversing_post)
    out = eng._rerank_l2("q", items)

    after = {r.content: _l2_admitted(r, floor) for r in out if r.layer == "l2"}
    assert after == before  # 门槛判定逐条一致
    # 原生分落袋：重排后 score 已改，native_score 仍是重排前的分数
    for r in out:
        if r.layer == "l2":
            assert r.metadata["native_score"] == original[r.content]
    # 顺序确实被重排改了（否则本测试是空转）
    assert [r.content for r in out if r.layer == "l2"] != [
        "high-2", "high-1", "low-1", "low-2"]


def test_format_recall_admitted_set_unchanged(tmp_path, monkeypatch):
    """端到端：rerank 开/关两种配置下，format_recall 注入的条目**集合**一致。"""
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    l2_specs = [("fact-high-1", 0.82), ("fact-low-1", 0.75),
                ("fact-high-2", 0.90), ("fact-low-2", 0.74)]
    l3_specs = [("hist-strong", 0.6), ("hist-weak", 0.05)]

    def _run(enabled: bool) -> str:
        cfg = _engine_cfg(tmp_path, _reranking_cfg(enabled=enabled))
        cfg.recall.l2_min_score = 0.76
        eng = RecallEngine(cfg)
        l2 = _l2_items(*l2_specs)
        l3 = [RecallResult(layer="l3", content=c, score=s, source="sqlite")
              for c, s in l3_specs]
        monkeypatch.setattr(eng, "_search_l2", lambda q, scope=None: l2)
        monkeypatch.setattr(eng, "_search_l3", lambda q: l3)
        _install_post(monkeypatch, _reversing_post)
        results = eng.parallel_recall("查询")
        return eng.format_recall(results, l1_text="",
                                 l1_budget=800, l23_budget=100000)

    def _injected(text: str) -> set:
        return {
            line.split("] ", 1)[1]
            for line in text.splitlines()
            if line.startswith(("[Fact]", "[History]"))
        }

    off_text = _run(enabled=False)
    on_text = _run(enabled=True)

    # 集合不变：重排只改顺序，不改谁过门槛
    assert _injected(off_text) == _injected(on_text)
    assert _injected(off_text) == {"fact-high-1", "fact-high-2", "hist-strong"}
    # 顺序确实变了（否则本测试是空转）
    assert off_text != on_text


def test_native_score_not_overwritten_when_fusion_present(tmp_path, monkeypatch):
    """l2_fusion 已存 native_score（原生余弦/子串分）时，重排不覆盖它。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    items = [
        RecallResult(layer="l2", content="a", score=0.5,
                     metadata={"native_score": 0.99}),
        RecallResult(layer="l2", content="b", score=0.4,
                     metadata={"native_score": 0.98}),
    ]
    _install_post(monkeypatch, _reversing_post)
    out = eng._rerank_l2("q", items)

    # 重排顺序 b, a；准入依据仍是融合时存的原生分
    assert [r.content for r in out] == ["b", "a"]
    assert [r.metadata["native_score"] for r in out] == [0.98, 0.99]


# ---------------------------------------------------------------------------
# 边界：top_n 语义 / 分数带 / 其它层 / 两条召回路径
# ---------------------------------------------------------------------------

def test_top_n_limits_submissions_without_dropping(monkeypatch):
    """top_n=3：只提交前 3 篇，尾部原序接后，集合不变。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg(top_n=3))
    items = _l2_items(*[(f"c{i}", 1.0 - i / 100) for i in range(12)])
    submitted = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        submitted["documents"] = json["documents"]
        docs = json["documents"]
        n = len(docs)
        return _RerankResp({
            "results": [
                {"index": i, "relevance_score": (i + 1) / n}
                for i in range(n)
            ]
        })

    _install_post(monkeypatch, fake_post)
    out = rer.rerank("q", items)

    assert submitted["documents"] == ["c0", "c1", "c2"]
    assert [i.content for i in out][:3] == ["c2", "c1", "c0"]
    assert [i.content for i in out][3:] == [f"c{i}" for i in range(3, 12)]
    assert {id(i) for i in out} == {id(i) for i in items}
    assert _diag.stats()["degraded"] == {}


def test_top_n_zero_means_all_and_explicit_arg_overrides(monkeypatch):
    """top_n<=0 全量提交；显式实参覆盖配置值。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg(top_n=2))
    items = _l2_items(("a", 0.9), ("b", 0.8), ("c", 0.7))
    submitted = {}

    def fake_post(url, *, json=None, headers=None, timeout=None):
        submitted["documents"] = json["documents"]
        return _reversing_post(url, json=json, headers=headers, timeout=timeout)

    _install_post(monkeypatch, fake_post)

    # 配置 top_n=2：只提交前 2 篇
    out = rer.rerank("q", items)
    assert submitted["documents"] == ["a", "b"]
    assert [i.content for i in out] == ["b", "a", "c"]

    # 显式 top_n=0：全量提交
    out = rer.rerank("q", items, top_n=0)
    assert submitted["documents"] == ["a", "b", "c"]
    assert [i.content for i in out] == ["c", "b", "a"]

    # 显式 top_n=1：只提交第 1 篇（单篇"重排"=原序），集合不变
    out = rer.rerank("q", items, top_n=1)
    assert submitted["documents"] == ["a"]
    assert [i.content for i in out] == ["a", "b", "c"]


def test_empty_query_passthrough(monkeypatch):
    """空查询没有信号可重排：直通，不发调用。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    calls = []
    _install_post(monkeypatch, lambda url, **kw: calls.append(url))
    items = _l2_items(("a", 0.9), ("b", 0.8))
    assert [i.content for i in rer.rerank("   ", items)] == ["a", "b"]
    assert calls == []


def test_score_band_and_other_layers_untouched(tmp_path, monkeypatch):
    """L2 分数带 [lo, hi] 保持（不改层间量纲）；L3/L4 分数不动；L4 仍居首。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    l2 = _l2_items(("f0", 0.73), ("f1", 0.80), ("f2", 0.91))
    l3 = RecallResult(layer="l3", content="h", score=0.55, source="sqlite")
    l4 = RecallResult(layer="l4", content="persona", score=1.0, source="persona")
    results = [l3, *l2, l4]

    _install_post(monkeypatch, _reversing_post)
    out = eng._rerank_l2("q", results)

    l2_out = [r for r in out if r.layer == "l2"]
    # 分数带保持：全部落在原 [0.73, 0.91] 内（层间量纲未动）
    assert all(0.73 <= r.score <= 0.91 for r in l2_out)
    # 顺序按重排（反转）
    assert [r.content for r in l2_out] == ["f2", "f1", "f0"]
    # 其它层原样
    l3_out = next(r for r in out if r.layer == "l3")
    assert l3_out.score == 0.55 and "rerank_rank" not in (l3_out.metadata or {})
    # L4 score=1.0 仍居首
    assert out[0].layer == "l4" and out[0].score == 1.0


def test_parallel_recall_applies_rerank(tmp_path, monkeypatch):
    """接入点①：parallel_recall 对 L2 应用重排（线程池路径）。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    l2 = _l2_items(("a", 0.9), ("b", 0.8), ("c", 0.7))
    monkeypatch.setattr(eng, "_search_l2", lambda q, scope=None: l2)
    _install_post(monkeypatch, _reversing_post)

    out = eng.parallel_recall("q")

    assert [r.content for r in out if r.layer == "l2"] == ["c", "b", "a"]
    assert all("rerank_rank" in (r.metadata or {})
               for r in out if r.layer == "l2")
    assert _diag.stats()["degraded"] == {}


def test_sequential_recall_applies_rerank(tmp_path, monkeypatch):
    """接入点②：_sequential_recall（关池后的顺序路径）同样应用重排。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    l2 = _l2_items(("a", 0.9), ("b", 0.8), ("c", 0.7))
    monkeypatch.setattr(eng, "_search_l2", lambda q, scope=None: l2)
    _install_post(monkeypatch, _reversing_post)

    out = eng._sequential_recall("q")

    assert [r.content for r in out if r.layer == "l2"] == ["c", "b", "a"]


def test_single_l2_candidate_skips_rerank(tmp_path, monkeypatch):
    """只有 1 条 L2 候选：重排无意义，直通（不发 API 调用）。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    items = _l2_items(("only", 0.9))
    calls = []
    _install_post(monkeypatch, lambda url, **kw: calls.append(url))
    out = eng._rerank_l2("q", items)
    assert out is items
    assert calls == []


def test_rerank_l2_failure_still_returns_results(tmp_path, monkeypatch):
    """接入点：API 失败时 parallel_recall 仍返回完整结果（只是没重排）。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    l2 = _l2_items(("a", 0.9), ("b", 0.8))
    monkeypatch.setattr(eng, "_search_l2", lambda q, scope=None: l2)

    def boom(url, **kwargs):
        raise RuntimeError("network down")

    _install_post(monkeypatch, boom)
    out = eng.parallel_recall("q")

    # 顺序保持原分数序（未重排），条目一个不少
    assert [r.content for r in out if r.layer == "l2"] == ["a", "b"]
    assert _diag.stats()["degraded"].get("rerank::call_failed") == 1


# ---------------------------------------------------------------------------
# 配置：默认关 + JSON 加载 + 数值纠偏
# ---------------------------------------------------------------------------

def test_enabled_defaults_off():
    """铁律：新行为默认关。"""
    assert RerankingConfig().enabled is False


def test_config_loads_reranking_section(tmp_path):
    """governed_memory.json 的 reranking 段可加载（含 top_n 数值纠偏）。"""
    (tmp_path / "governed_memory.json").write_text(json.dumps({
        "reranking": {
            "enabled": True,
            "provider": "siliconflow",
            "api_key_env": "SILICONFLOW_API_KEY",
            "base_url": "https://api.siliconflow.cn/v1",
            "model": "BAAI/bge-reranker-v2-m3",
            "top_n": "5",
        }
    }), encoding="utf-8")
    cfg = load_governed_config(tmp_path)
    assert cfg.reranking.enabled is True
    assert cfg.reranking.provider == "siliconflow"
    assert cfg.reranking.model == "BAAI/bge-reranker-v2-m3"
    # "5"（字符串）被 _NUMERIC_FIELDS 纠偏成 int 5
    assert cfg.reranking.top_n == 5
    assert isinstance(cfg.reranking.top_n, int)


def test_config_default_reranking_stays_off(tmp_path):
    """没有 reranking 段时：enabled 保持默认 False（现行为不变）。"""
    cfg = load_governed_config(tmp_path)
    assert cfg.reranking.enabled is False
    assert cfg.reranking.provider == ""


# ---------------------------------------------------------------------------
# QA 复核回归（F-A3 / F-A1 / F-A2，2026-10-19）
# ---------------------------------------------------------------------------

def test_config_top_n_zero_means_all_not_ten(monkeypatch):
    """F-A3：配置 ``top_n=0`` 必须"全量提交"，不能被 ``or 10`` 静默吞成 10。

    回归的真实缺陷：``int(getattr(cfg, "top_n", 10) or 10)`` 里 ``0 or 10``
    → 10，于是"想全量重排"的部署**静默**只提交前 10 条且不报任何错。
    """
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg(top_n=0))
    submitted = {}
    _install_post(monkeypatch, _capture_documents(submitted))
    items = _l2_items(*[(f"c{i}", 1.0 - i / 100) for i in range(12)])

    out = rer.rerank("q", items)

    assert submitted["documents"] == [f"c{i}" for i in range(12)]  # 全量，不是 10
    assert {id(i) for i in out} == {id(i) for i in items}
    assert _diag.stats()["degraded"] == {}


def test_config_top_n_null_falls_back_to_default_ten(monkeypatch):
    """F-A3 对照：**只有缺值/None** 才回退默认 10（``null`` 与缺字段同义）。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg(top_n=None))
    submitted = {}
    _install_post(monkeypatch, _capture_documents(submitted))
    items = _l2_items(*[(f"c{i}", 1.0 - i / 100) for i in range(12)])

    rer.rerank("q", items)

    assert submitted["documents"] == [f"c{i}" for i in range(10)]
    assert _diag.stats()["degraded"] == {}


def test_config_top_n_negative_means_all(monkeypatch):
    """F-A3 同族：负值配置同样按文档语义 = 全量提交。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg(top_n=-1))
    submitted = {}
    _install_post(monkeypatch, _capture_documents(submitted))
    items = _l2_items(*[(f"c{i}", 1.0 - i / 100) for i in range(12)])

    rer.rerank("q", items)

    assert submitted["documents"] == [f"c{i}" for i in range(12)]


def test_rerank_with_none_items_raises_directly(monkeypatch):
    """F-A1 契约（诚实版）：Reranker 层对畸形入参**不承诺**不抛。

    读路径的安全由接入层保证（见下一条）——本测试把"分层契约"钉死：
    文档写的是"对构造良好的 items 永不抛"，这里证明它没有过度承诺。
    """
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    with pytest.raises(TypeError):
        rer.rerank("q", None)


def test_rerank_l2_absorbs_malformed_items_failure(tmp_path, monkeypatch):
    """F-A1 接入层：rerank() 抛 TypeError 不炸读路径，条目一条不少。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    l2 = _l2_items(("a", 0.9), ("b", 0.8))
    l3 = RecallResult(layer="l3", content="h", score=0.55, source="sqlite")
    monkeypatch.setattr(eng, "_search_l2", lambda q, scope=None: l2)
    monkeypatch.setattr(eng, "_search_l3", lambda q: [l3])

    def _broken_rerank(self, query, items, top_n=None):
        # 忠实模拟 "Reranker.rerank(q, None)" 的失败形态：materialization
        # （list(items)）在 rerank() 内部抛出，由接入层兜底网吸收。
        raise TypeError("list() argument must be iterable, not NoneType")

    monkeypatch.setattr(Reranker, "rerank", _broken_rerank)
    out = eng.parallel_recall("q")  # 不得抛

    assert {id(r) for r in out if r.layer == "l2"} == {id(r) for r in l2}
    assert [r.content for r in out if r.layer == "l3"] == ["h"]
    assert _diag.stats()["degraded"].get("rerank::rerank_raised") == 1


def test_rerank_l2_absorbs_exploding_content(tmp_path, monkeypatch):
    """F-A2 接入层：元素 ``.content`` 访问即抛 → 直通、不发 API、条目不少。"""
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    bad = _ExplodingContent()
    good = RecallResult(layer="l2", content="good", score=0.8)
    items = [bad, good]
    calls = []
    _install_post(monkeypatch, lambda url, **kw: calls.append(url))

    out = eng._rerank_l2("q", items)  # 不得抛

    assert out is items
    assert {id(r) for r in out} == {id(bad), id(good)}
    assert calls == []  # 畸形入参在提交前就炸，不该发出任何 API 调用
    assert _diag.stats()["degraded"].get("rerank::rerank_raised") == 1


def test_rerank_l2_handles_none_return(tmp_path, monkeypatch):
    """接入层：rerank() 返回 None（契约违规）不得让 ``len(None)`` 逃逸。

    这条是兜底网的边界：``len(reranked)`` 在 try 之外，``None`` 会在那里
    自己抛 —— 所以契约违规分支必须先用 ``is None`` 短路。
    """
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    eng = RecallEngine(_engine_cfg(tmp_path, _reranking_cfg()))
    l2 = _l2_items(("a", 0.9), ("b", 0.8))
    monkeypatch.setattr(Reranker, "rerank",
                        lambda self, q, items, top_n=None: None)
    out = eng._rerank_l2("q", l2)  # 不得抛
    assert out is l2
    assert _diag.stats()["degraded"].get("rerank::bad_response") == 1


def test_rerank_returns_new_list_not_same_object(monkeypatch):
    """措辞钉桩：``rerank()`` 返回**新列表**（元素同批、顺序同），非入参对象。

    "返回同一对象"只在接入层成立；这里把差异写死，防止将来照着旧注释
    在这一层写 ``out is items`` 断言。
    """
    _diag.reset()
    monkeypatch.setenv(_KEY_ENV, "sk-test")
    rer = Reranker(_reranking_cfg())
    items = _l2_items(("a", 0.9), ("b", 0.8))
    out = rer.rerank("q", items)
    assert out is not items
    assert [i.content for i in out] == [i.content for i in items]
    assert {id(i) for i in out} == {id(i) for i in items}
