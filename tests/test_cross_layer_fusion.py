# -*- coding: utf-8 -*-
"""L2 ↔ L3 跨层 RRF 融合（``recall.cross_layer_fusion``）测试。

为什么需要跨层 RRF：L2 是 1024 维余弦映射（``(1+cos)/2`` 标尺），L3 是 FTS5
原始分（0.3~0.7），两者**不同源**，混排时直接按 ``score`` 排序没有可比性
（``format_recall`` 的历史注释记着：套 L2 的 0.76 门槛会误砍 L3 的 82/91 条）。
RRF 只用**名次**，天然回避量纲问题。

四条，各自能独立推翻实现：

1. 不同源分数混排时 L3 不被砍，跨层 trace 生效；
2. 准入仍看**原生分** —— 防「融合分回流当准入门槛」的回归网；
3. 单层缺失时回退，开/关两种配置输出一致；
4. 关闭 = 历史行为：融合根本不被调用，命中原分原样进入结果。

全部跑在**最小假数据**上（直接构造 ``RecallResult``），不依赖 lancedb / 嵌入。
"""

from __future__ import annotations

from pathlib import Path

import pytest

import plugin.memory_governed._recall as recall_mod
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._recall import (
    RecallEngine,
    RecallResult,
    _fuse_cross_layer,
    _l2_admitted,
)


#: 与部署一致的 L2 余弦标尺门槛（``recall.l2_min_score``）。
_L2_FLOOR = 0.76


def _l2(content: str, score: float, *, native=None, rid=None) -> RecallResult:
    """构造一条 L2 命中。``native`` 非 None 时写入 ``metadata["native_score"]``。"""
    meta = {}
    if native is not None:
        meta["native_score"] = native
    if rid is not None:
        meta["source_rowid"] = rid
    return RecallResult(layer="l2", content=content, score=score,
                        source="lance", metadata=meta)


def _l3(content: str, score: float, *, rid=None) -> RecallResult:
    """构造一条 L3 命中（FTS5 原始分标尺）。"""
    meta = {}
    if rid is not None:
        meta["source_rowid"] = rid
    return RecallResult(layer="l3", content=content, score=score,
                        source="sqlite", metadata=meta)


def _engine(tmp_path: Path, cross: bool) -> RecallEngine:
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    cfg.recall.l2_min_score = _L2_FLOOR
    cfg.recall.cross_layer_fusion = cross
    eng = RecallEngine(cfg)
    # 跳过真实 LanceDB / 嵌入初始化：测试直接构造命中喂给 format_recall。
    eng._l2_init_attempted = True
    eng._l2_store = object()
    eng._embed_fn = lambda q: [0.0] * 4
    return eng


class TestCrossLayerFusion:
    def test_l3_survives_incomparable_score_mix(self, tmp_path, monkeypatch):
        """不同源分数混排：L2 0.80（余弦标尺）与 L3 0.50（FTS5 原始分）都保留。

        L3 的 0.50 若被误套 L2 的 0.76 门槛就会被整层砍掉 —— 这正是跨层 RRF
        要解决的问题。断言两条都在结果里，且跨层 trace 生效。
        """
        eng = _engine(tmp_path, cross=True)
        l2 = _l2("L2事实", 0.80)          # 无 source_rowid → 键退化为 content
        l3 = _l3("L3历史", 0.50)

        captured: dict = {}
        real = recall_mod._fuse_cross_layer

        def _spy(a, b):
            out = real(a, b)
            captured["out"] = out
            return out

        monkeypatch.setattr(recall_mod, "_fuse_cross_layer", _spy)
        text = eng.format_recall([l2, l3], l1_text="", l1_budget=800,
                                 l23_budget=1200)

        assert "L2事实" in text and "L3历史" in text
        assert {r.content for r in captured["out"]} == {"L2事实", "L3历史"}
        for r in captured["out"]:
            assert r.metadata["trace"]["fusion"] == "rrf-cross-layer"
        # 两条内容不同 → 各是单通道命中，但两条路都进了融合。
        all_channels = {c for r in captured["out"]
                        for c in r.metadata["trace"]["channels"]}
        assert all_channels == {"l2", "l3"}

    def test_admission_still_on_native_score(self, tmp_path):
        """准入仍看**原生分**：融合分是排名量纲，绝不能回流当门槛。

        构造一条 L2：原生分 0.10 < 0.76 门槛，但 score=1.0（若参与融合会排最前
        —— 单路第 1 名归一化即为满分）。它**必须**被挡掉，否则「拿 A 的尺子量
        B」的旧病会借融合分复活。
        """
        eng = _engine(tmp_path, cross=True)
        weak = _l2("弱L2", 1.0, native=0.10)   # 融合会排很前，但原生分不够
        strong_l3 = _l3("L3历史", 0.50)

        assert not _l2_admitted(weak, _L2_FLOOR)   # 底层准入规则挡住它

        text = eng.format_recall([weak, strong_l3], l1_text="", l1_budget=800,
                                 l23_budget=1200)
        assert "弱L2" not in text                  # 融合分没有把它捞回来
        assert "L3历史" in text

    def test_single_layer_falls_back(self, tmp_path):
        """单层缺失（只有 L3 或只有 L2）→ 不融合，开/关两种配置输出一致。"""
        on = _engine(tmp_path, cross=True)
        off = _engine(tmp_path, cross=False)

        only_l3 = [_l3("L3历史A", 0.6), _l3("L3历史B", 0.4)]
        assert (on.format_recall(list(only_l3), l1_text="", l1_budget=800,
                                 l23_budget=1200)
                == off.format_recall(list(only_l3), l1_text="", l1_budget=800,
                                     l23_budget=1200))

        only_l2 = [_l2("L2事实A", 0.9), _l2("L2事实B", 0.8)]
        assert (on.format_recall(list(only_l2), l1_text="", l1_budget=800,
                                 l23_budget=1200)
                == off.format_recall(list(only_l2), l1_text="", l1_budget=800,
                                     l23_budget=1200))

    def test_off_never_invokes_fusion(self, tmp_path, monkeypatch):
        """关闭 = 历史行为：融合根本不被调用，命中原分原样进入结果。"""
        eng = _engine(tmp_path, cross=False)
        l2 = _l2("L2事实", 0.80)
        l3 = _l3("L3历史", 0.50)

        calls: list = []
        real = recall_mod._fuse_cross_layer

        def _spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(recall_mod, "_fuse_cross_layer", _spy)
        text = eng.format_recall([l2, l3], l1_text="", l1_budget=800,
                                 l23_budget=1200)

        assert calls == []                          # 融合未被触碰
        assert "rrf-cross-layer" not in text
        # 关闭 → 命中原样进入结果：score 仍是各条原始分，无融合痕迹。
        assert l2.score == pytest.approx(0.80)
        assert l3.score == pytest.approx(0.50)
        assert "native_score" not in l2.metadata
        assert "trace" not in l2.metadata
        assert "trace" not in l3.metadata
        # 原分降序：L2(0.80) 排在 L3(0.50) 之前 —— 历史排序行为保留。
        assert text.index("L2事实") < text.index("L3历史")


#: 迭代序与普通 ``set`` **相反**的 set —— 用来强制暴露「输出依赖 set 迭代序」。
#: 内核按 ``set`` 并集迭代键；把它换成这个，预排序列表顺序即翻转。若最终排序
#: 非全序，输出顺序会跟着翻 —— 这正是随哈希种子漂移的根因，可在同一进程内复现
#: （无需换 ``PYTHONHASHSEED``）。
class _ReversedSet(set):
    used = False

    def __init__(self, *args, **kwargs):
        _ReversedSet.used = True
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return iter(list(super().__iter__())[::-1])


class TestDeterministicOrder:
    """融合输出顺序必须与 ``set`` 迭代序无关（即与哈希种子无关）。

    内核用 ``set`` 并集迭代键、再排序；若最终排序键**非全序**，「同内容不同键」
    （一边有 ``source_rowid`` 一边没有、两边都是本层 rank 1 ⇒ 融合分同为 1.0）
    会退化到集合迭代序，随哈希种子漂移 —— 注入 tag（Fact/History）非确定。
    """

    def test_output_order_does_not_depend_on_input_order(self):
        """同一进程内换入参顺序，输出恒为全序规范序。

        注意：进程内哈希种子固定、且 ``set`` 迭代序与插入顺序无关，所以单靠
        「换顺序」**测不出**漂移；真正对抗根因见
        :meth:`test_output_order_ignores_set_iteration_order`。本用例把全序契约
        钉成确定值，并证明输出不再依赖入参排列。
        """
        def _groups(tie_l2_first: bool, tie_l3_first: bool):
            l2 = [_l2("DUP", 0.9, rid=42), _l2("L2other", 0.5, rid=7)]
            l3 = [_l3("DUP", 0.5), _l3("L3other", 0.3)]
            if not tie_l2_first:
                l2.reverse()
            if not tie_l3_first:
                l3.reverse()
            return l2, l3

        orders = set()
        for a in (True, False):
            for b in (True, False):
                l2, l3 = _groups(a, b)
                orders.add(tuple(r.layer for r in _fuse_cross_layer(l2, l3)))

        # DUP 两条融合分同为 1.0：按 repr(key) 定序，"('content'…)" < "('rid'…)"
        # → 键退化的一方（l3）在前；0.9839 的两条按内容 "L2other" < "L3other"。
        assert orders == {("l3", "l2", "l2", "l3")}

    def test_output_order_ignores_set_iteration_order(self, monkeypatch):
        """把内核的 ``set`` 换成**反序迭代**的 set，输出必须逐字不变。

        这是对根因的**正面**对抗：内核按 ``set`` 并集迭代键。用一个反序 set 让
        预排序列表顺序翻转 —— 旧的非全序排序会让同分同内容的那一对跟着翻
        （实测 OLD: normal=l2,l3… vs reversed=l3,l2…），全序排序下则恒定。
        """
        def _groups():
            l2 = [_l2("DUP", 0.9, rid=42), _l2("L2other", 0.5, rid=7)]
            l3 = [_l3("DUP", 0.5), _l3("L3other", 0.3)]
            return l2, l3

        normal = tuple(r.layer for r in _fuse_cross_layer(*_groups()))

        _ReversedSet.used = False
        monkeypatch.setattr(recall_mod, "set", _ReversedSet, raising=False)
        reversed_iter = tuple(r.layer for r in _fuse_cross_layer(*_groups()))

        assert _ReversedSet.used, "反序 set 未被内核采纳，用例失去判别力"
        assert normal == reversed_iter == ("l3", "l2", "l2", "l3")
