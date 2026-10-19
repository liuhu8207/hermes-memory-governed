# -*- coding: utf-8 -*-
"""跨通道证据救援回归测试（2026-10-19）。

背景（不是假设，是当天实测）
============================
``recall_l2`` 在向量通道产出**任何**命中时直接 return —— 这是"二选一"，
不是融合（``recall_l2`` 的 docstring 自己就是这么写的）。后果是一条字面
全命中的事实被一条擦线通过的噪声彻底顶掉：

    query  「家庭网络代理节点挂在哪个地址」
    真答案  0.7588（< floor 0.77，落进 near_miss）
    实际返回 0.7703 「OpenCode 必须配置 OPENCODE_HOME…」——毫不相关

两者只差 0.0115。把一个帧度上的二元判决（过线/不过线）当成了一个
可信赖的排序结论。

这里测的是救援逻辑本身，而不是假向量空间：``_l2_semantic_hits`` 被打桩，
词法侧用真实的 L2 表 + ``l2_lexical_score``。两个变量分开控制，才好在"词法
没有满分证据"、"预算已满"、"projects 不可见"这些分支上做反例。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import memory_cli as cli

_SEMANTIC_NOISE = "OpenCode 必须配置 OPENCODE_HOME 才能识别身份"
#: 3 个查询词项全部出现 → ``l2_lexical_score`` 饱和到 1.0（值域上界）
_FULL = "SecretStore 端口 密码 都记在 NAS 上"
#: 只有 2/3 → 0.6667，够不上"完全覆盖"
_PARTIAL = "SecretStore 端口 在别处"
_QUERY = "SecretStore 端口 密码"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """HERMES_HOME + 空 L2 目录，接到 CLI 上。"""
    home = tmp_path / ".hermes"
    (home / "memory" / "l2").mkdir(parents=True)
    (home / "governed_memory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _seed(home: Path, texts) -> None:
    lancedb = pytest.importorskip("lancedb")
    l2 = home / "memory" / "l2"
    l2.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(l2))
    if "memories" in db.list_tables().tables:
        db.drop_table("memories")
    rows = []
    for i, item in enumerate(texts):
        content, project = item if isinstance(item, tuple) else (item, None)
        rows.append({
            "content": content, "category": "other", "source": "test",
            "timestamp": "2026-09-16T00:00:00", "vector": [0.0, 0.0],
            "source_rowid": i, "role": "agent", "agent": "tester",
            "project": project,
        })
    db.create_table("memories", data=rows)


def _patch_semantic(monkeypatch, hits, floor: float = 0.77) -> None:
    """让"向量通道说了什么"成为可控变量。"""
    def _fake(query, top_k, project, errors):
        return {"available": True, "hits": list(hits), "floor": floor,
                "floor_source": "default", "filtered_out": 3, "truncated": 0,
                "reason": "", "backend": "fake:test",
                "near_miss": [{"content": _FULL, "score": round(floor - 0.01,
                                                                4)}]}
    monkeypatch.setattr(cli, "_l2_semantic_hits", _fake)


def _noise_hit(score: float = 0.7703):
    return {"layer": "l2", "content": _SEMANTIC_NOISE, "score": score,
            "score_basis": "semantic-cosine"}


class TestFullCoverageEvidenceComesBack:
    """真答案被向量通道丢掉、但词法完全覆盖时，它必须还能被看见。"""

    def test_a_row_the_vector_channel_dropped_is_appended(self, store, monkeypatch):
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_FULL])
        res = cli.recall_l2(_QUERY, 5, [], project="")
        hits = res["hits"]
        assert hits[0]["content"] == _SEMANTIC_NOISE     # 向量通道的顺序没动
        rescued = [h for h in hits if h.get("rescued")]
        assert [h["content"] for h in rescued] == [_FULL]
        assert all(h["score_basis"] == "lexical-dis-max" for h in rescued)

    def test_the_rescue_is_said_out_loud_in_the_ranking(self, store, monkeypatch):
        # 一条带着不同量纲分数的行混进结果却不说明，等于让调用方无从证伪。
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_FULL])
        ranking = cli.recall_l2(_QUERY, 5, [], project="")["ranking"]
        assert ranking["rescued"] == 1
        assert "rescued" in ranking["note"]
        # 量纲没被偷偷换掉：basis 仍然是产生主体命中的那个通道。
        assert ranking["basis"] == "semantic-cosine"

    def test_partial_coverage_is_not_enough_to_rescue(self, store, monkeypatch):
        """饱和上界才是门槛：只差一个词就不算"完全覆盖"。"""
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_PARTIAL])
        hits = cli.recall_l2(_QUERY, 5, [], project="")["hits"]
        assert [h["content"] for h in hits] == [_SEMANTIC_NOISE]
        assert cli.l2_lexical_score(_QUERY, _PARTIAL) < 1.0

    def test_a_row_the_vector_channel_already_ranked_is_not_duplicated(
            self, store, monkeypatch):
        _seed(store, [_FULL])
        already = {"layer": "l2", "content": _FULL, "score": 0.85,
                   "score_basis": "semantic-cosine"}
        _patch_semantic(monkeypatch, [already, _noise_hit()])
        hits = cli.recall_l2(_QUERY, 5, [], project="")["hits"]
        assert [h["content"] for h in hits] == [_FULL, _SEMANTIC_NOISE]
        assert not any(h.get("rescued") for h in hits)


class TestTheRescueStaysInsideItsBudget:
    def test_a_full_top_k_leaves_no_room(self, store, monkeypatch):
        """调用方的 top_k 是预算：救援不能把结果撑出去。"""
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_FULL])
        hits = cli.recall_l2(_QUERY, 1, [], project="")["hits"]
        assert len(hits) == 1
        assert hits[0]["content"] == _SEMANTIC_NOISE

    def test_multiple_rescues_are_capped_by_top_k(self, store, monkeypatch):
        second = "SecretStore 端口 密码 的第二份记录"
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_FULL, second])
        hits = cli.recall_l2(_QUERY, 2, [], project="")["hits"]
        assert len(hits) == 2          # 1 条向量 + 1 条救援 = 预算用完
        assert hits[1].get("rescued") is True


class TestProjectVisibilityStillHolds:
    def test_another_projects_row_is_not_rescued(self, store, monkeypatch):
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [(_FULL, "someone-else")])
        hits = cli.recall_l2(_QUERY, 5, [], project="mine")["hits"]
        assert [h["content"] for h in hits] == [_SEMANTIC_NOISE]

    def test_a_global_row_is_visible_to_every_project(self, store, monkeypatch):
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [(_FULL, None)])
        hits = cli.recall_l2(_QUERY, 5, [], project="mine")["hits"]
        assert any(h.get("rescued") for h in hits)


class TestTheRescueDoesNotTouchTheOtherPaths:
    def test_lexical_only_is_untouched(self, store, monkeypatch):
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_FULL])
        hits = cli.recall_l2(_QUERY, 5, [], project="",
                             lexical_only=True)["hits"]
        assert [h["content"] for h in hits] == [_FULL]
        assert not any(h.get("rescued") for h in hits)

    def test_an_empty_vector_channel_degrades_rather_than_rescues(
            self, store, monkeypatch):
        # 向量通道交白卷走的是既有的降级路径，不是救援路径。
        _patch_semantic(monkeypatch, [])
        _seed(store, [_FULL])
        res = cli.recall_l2(_QUERY, 5, [], project="")
        assert res["ranking"]["basis"] == "lexical-dis-max"
        assert "rescued" not in res["ranking"]
        assert [h["content"] for h in res["hits"]] == [_FULL]

    def test_an_unavailable_vector_channel_is_still_a_degradation(
            self, store, monkeypatch):
        def _dead(query, top_k, project, errors):
            return {"available": False, "hits": [], "floor": 0.77,
                    "floor_source": "default", "filtered_out": 0,
                    "truncated": 0, "reason": "no backend", "backend": "",
                    "near_miss": []}
        monkeypatch.setattr(cli, "_l2_semantic_hits", _dead)
        _seed(store, [_FULL])
        res = cli.recall_l2(_QUERY, 5, [], project="")
        assert res["ranking"]["basis"] == "lexical-dis-max"
        assert "rescued" not in res["ranking"]


class TestTheRealQuestionItExistsFor:
    def test_the_answer_survives_a_marginally_winning_decoy(self, store, monkeypatch):
        """2026-10-18 那道题的抽象形态：擦线者赢了，全中者被丢掉。"""
        _patch_semantic(monkeypatch, [_noise_hit()])
        _seed(store, [_FULL])
        res = cli.recall_l2(_QUERY, 5, [], project="")
        contents = [h["content"] for h in res["hits"]]
        # 噪声保住了它的第一名（救援不重排），但答案没有消失。
        assert contents[0] == _SEMANTIC_NOISE
        assert _FULL in contents
        # 而且它是可识别的：调用方能看出这条不是同一个盘子里的东西。
        assert res["ranking"]["rescued"] >= 1
