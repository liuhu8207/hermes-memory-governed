# -*- coding: utf-8 -*-
"""L2 双路 RRF 融合（``recall.l2_fusion``）测试（Phase 1 / WeKnora 借鉴）。

覆盖五件事：

1. 默认关闭 → 历史二选一链路一个字节都不变；
2. 开启 → 两路并集 + RRF 排名 + ``native_score`` / ``trace``；
3. 同一事实（source_rowid 相同）跨通道合并为一条；
4. ``format_recall`` 的 L2 门槛对**原生分**生效 —— 排序用融合分，
   准入看原生分（拿余弦标尺去量 RRF 排名分会把整层砍空）；
5. 嵌入不可用 / 单路为空 → 直返原生分，门槛语义与历史一致。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._recall import (
    RecallEngine,
    RecallResult,
    _fuse_l2_channels,
)


def _hit(content: str, score: float, rid=None) -> RecallResult:
    return RecallResult(layer="l2", content=content, score=score,
                        source="l2", metadata={"source_rowid": rid})


def _engine(tmp_path: Path, fusion: bool) -> RecallEngine:
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    cfg.recall.l2_fusion = fusion
    cfg.recall.l2_min_score = 0.76
    eng = RecallEngine(cfg)
    # 跳过真实 LanceDB / 嵌入初始化：下面会把两条渠道方法整个替换掉。
    eng._l2_init_attempted = True
    eng._l2_store = object()
    eng._embed_fn = lambda q: [0.0] * 4
    return eng


class TestL2FusionChain:
    def test_fusion_off_keeps_legacy_chain(self, tmp_path):
        """默认关闭：向量可用则只跑向量 —— 词法通道被整体跳过（历史行为）。"""
        eng = _engine(tmp_path, fusion=False)
        eng._search_l2_vector = lambda q, p=None: [_hit("vec-only", 0.9, 1)]
        eng._search_l2_text = lambda q, p=None: [_hit("lex-only", 0.8, 2)]
        out = eng._search_l2("q")
        assert [r.content for r in out] == ["vec-only"]

    def test_fusion_on_unions_both_channels(self, tmp_path):
        eng = _engine(tmp_path, fusion=True)
        eng._search_l2_vector = lambda q, p=None: [_hit("vec-only", 0.9, 1)]
        eng._search_l2_text = lambda q, p=None: [_hit("lex-only", 0.8, 2)]
        out = eng._search_l2("q")
        assert {r.content for r in out} == {"vec-only", "lex-only"}
        for r in out:
            assert r.metadata["trace"]["fusion"] == "rrf"
            assert "native_score" in r.metadata

    def test_same_fact_merges_across_channels(self, tmp_path):
        """同一事实双通道确认 → 单条记录，trace 记下两路名次。"""
        eng = _engine(tmp_path, fusion=True)
        eng._search_l2_vector = lambda q, p=None: [_hit("同一事实", 0.9, 7)]
        eng._search_l2_text = lambda q, p=None: [_hit("同一事实", 0.8, 7)]
        out = eng._search_l2("q")
        assert len(out) == 1
        r = out[0]
        assert r.metadata["trace"]["channels"] == ["vector", "lexical"]
        assert r.metadata["trace"]["vector_rank"] == 1
        assert r.metadata["trace"]["lexical_rank"] == 1
        assert r.metadata["native_score"] == pytest.approx(0.9)
        # 双通道各自第 1 名 → 归一化融合分恰好 1.0
        assert r.score == pytest.approx(1.0)

    def test_lexical_only_hit_carries_text_native(self, tmp_path):
        """只在词法路命中的事实也进入结果（旧链路下它会被向量路闷死）。"""
        eng = _engine(tmp_path, fusion=True)
        eng._search_l2_vector = lambda q, p=None: [_hit("向量事实", 0.9, 1)]
        eng._search_l2_text = lambda q, p=None: [
            _hit("向量事实", 0.9, 1), _hit("精确术语事实", 0.8, 2)]
        out = eng._search_l2("q")
        by_content = {r.content: r for r in out}
        assert "精确术语事实" in by_content
        assert by_content["精确术语事实"].metadata["native_score"] == pytest.approx(0.8)
        assert by_content["精确术语事实"].metadata["trace"]["channels"] == ["lexical"]

    def test_fusion_without_embed_falls_back_to_lexical(self, tmp_path):
        """嵌入不可用 → 走历史词法兜底，不套融合、无 native_score。"""
        eng = _engine(tmp_path, fusion=True)
        eng._embed_fn = None
        eng._search_l2_text = lambda q, p=None: [_hit("lex-only", 0.8, 2)]
        out = eng._search_l2("q")
        assert [r.content for r in out] == ["lex-only"]
        assert "native_score" not in out[0].metadata

    def test_vector_empty_returns_lexical_raw(self, tmp_path):
        """单路直返：另一路为空时不值得开融合，原生分原样保留。"""
        eng = _engine(tmp_path, fusion=True)
        eng._search_l2_vector = lambda q, p=None: []
        eng._search_l2_text = lambda q, p=None: [_hit("lex", 0.85, 2)]
        out = eng._search_l2("q")
        assert [r.content for r in out] == ["lex"]
        assert out[0].score == pytest.approx(0.85)
        assert "native_score" not in out[0].metadata


class TestFuseDeterminism:
    def test_ties_break_by_content_not_set_order(self, tmp_path):
        """同分 tiebreak 按内容字典序 —— set 并集迭代不能泄漏到输出顺序。"""
        vec = [_hit("乙", 0.9, 1), _hit("甲", 0.9, 2)]
        lex = [_hit("丙", 0.8, 3), _hit("丁", 0.7, 4)]
        first = [(r.content, r.score) for r in _fuse_l2_channels(vec, lex)]
        for _ in range(5):
            again = [(r.content, r.score) for r in _fuse_l2_channels(vec, lex)]
            assert again == first

    def test_double_confirmed_beats_single_channel(self, tmp_path):
        """双通道第 1+第 2 名 ≈ 0.992 > 单通道第 2 名 61/62 ≈ 0.984。

        同时钉住单通道归一化语义：单通道第 1 名仍是满分 1.0（与双通道
        满分并列，由 tiebreak 决定次序，不强行压过）。
        """
        vec = [_hit("双确认", 0.51, 1), _hit("纯向量2", 0.3, 3)]
        lex = [_hit("词法头", 0.6, 9), _hit("双确认", 0.5, 1)]
        fused = {r.content: r for r in _fuse_l2_channels(vec, lex)}
        assert fused["双确认"].score == pytest.approx(
            (0.5 / 61 + 0.5 / 62) * 61, abs=1e-3)
        # 单通道第 2 名：归一化按出现通道折算 = (k+1)/(k+rank)
        assert fused["纯向量2"].score == pytest.approx(61 / 62, abs=1e-3)
        assert fused["词法头"].score == pytest.approx(1.0, abs=1e-3)
        assert fused["双确认"].score > fused["纯向量2"].score


class TestFormatFloorOnNative:
    def test_low_fused_high_native_is_kept(self, tmp_path):
        """融合分 0.68 < 0.76 但原生分 0.9 → 必须保留。

        这是「排序用融合分，准入看原生分」的正向用例：若门槛误套在融合分上，
        深名次的强证据会被整层砍空。
        """
        eng = _engine(tmp_path, fusion=True)
        vec = [_hit(f"噪音{i:02d}", 0.95 - i * 0.001, 500 + i) for i in range(29)]
        vec.append(_hit("目标", 0.9, 1))
        lex = [_hit(f"干扰{j:02d}", 0.92 - j * 0.001, 700 + j) for j in range(29)]
        lex.append(_hit("目标", 0.85, 1))
        fused = _fuse_l2_channels(vec, lex)
        target = next(r for r in fused if r.content == "目标")
        # 两路都排第 30 → 融合分 = 61/90 ≈ 0.678 < 0.76
        assert target.score < 0.76
        assert target.metadata["native_score"] >= 0.76
        text = eng.format_recall(fused, l1_text="", l1_budget=800,
                                 l23_budget=1200)
        assert "目标" in text

    def test_high_fused_low_native_is_dropped(self, tmp_path):
        """双路第 1 名（融合分 1.0）但原生分 0.7 < 0.76 → 必须剔除。

        反向用例：若门槛只看融合分，弱证据会借双通道确认混进上下文。
        """
        eng = _engine(tmp_path, fusion=True)
        fused = _fuse_l2_channels([_hit("弱证据", 0.7, 1)],
                                  [_hit("弱证据", 0.7, 1)])
        assert fused[0].score == pytest.approx(1.0)
        assert fused[0].metadata["native_score"] == pytest.approx(0.7)
        text = eng.format_recall(fused, l1_text="", l1_budget=800,
                                 l23_budget=1200)
        assert "弱证据" not in text

    def test_unfused_hits_still_floor_on_score(self, tmp_path):
        """未融合的历史命中（无 native_score）按 score 过门槛 —— 行为不变。"""
        eng = _engine(tmp_path, fusion=False)
        results = [
            _hit("够分的事实", 0.9),
            _hit("不够分的事实", 0.5),
        ]
        text = eng.format_recall(results, l1_text="", l1_budget=800,
                                 l23_budget=1200)
        assert "够分的事实" in text
        assert "不够分的事实" not in text
