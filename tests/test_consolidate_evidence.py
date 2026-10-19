# -*- coding: utf-8 -*-
# ruff: noqa: F811 — pytest fixture 跨文件复用（store 由 import 引入，
# 各测试以参数消费它时被静态分析视为重绑定；pytest 标准模式）。
"""P3 测试：consolidate 的证据链（双向可追溯）与置信度（有定义且被用起来）。

背景：外部方案曾提议新建 ``beliefs.jsonl``（confidence/evidence/superseded_by/
stale 归档），评审结论是**拒绝新建存储** —— 与已有 ``cmd_consolidate`` 大面积
重叠且引入序列化/覆盖丢数据的新问题。P3 = 在现有 consolidate 上补它真正缺的
两部分：证据链与置信度。本文件钉住的正是这两部分的行为：

1. ``--apply`` 成功一簇后，归档 JSONL 里**每条**被合并记录都有
   ``merged_from`` + ``merged_into``，且与原条/合并条互为对应（从任一端都能
   走到另一端）；
2. 候选文件 ``consolidate_<ts>.json`` 的簇按 confidence **降序**（构造时让
   小簇的发现序在前，排序错了这个测试会红）；
3. confidence 随簇大小**单调不减**（size 2 vs size 4 的簇对比）；
4. LLM 空输出（拒绝合并）时：原条目保留、归档**不写**（没合并就不能写
   merged_into）、墓碑也不写；
5. 失败路径仍保留原条目（回归：归档失败时一条都不许删）。

LLM 一律 monkeypatch，绝不外呼；向量走假嵌入（同
``tests.test_cli_security_and_recall`` 的 fixture）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import memory_cli as cli
from plugin.memory_governed import _llm as llm_mod

# 复用 HERMES_HOME store fixture、假嵌入与向量播种（绝不外呼）
from tests.test_cli_security_and_recall import (  # noqa: F401 — pytest fixture
    _fake_vec,
    _install_fake_embedding,
    _seed_vectors,
    store,
)

# 近重复基座：中文 + 无 fake 轴词汇 → 向量落在兜底轴（余弦 1.0），词法
# bigram 高重叠 —— 唯一变量就是聚簇/证据链逻辑本身。后缀只改词法不改语义，
# 实测 Jaccard(基座, 基座+后缀) ≈ 0.92，稳过 CONS_JACCARD_MIN。
_BASE = "构建机 build-07 使用 Ubuntu 24.04 编译内核模块"
_NEAR_A = _BASE
_NEAR_B = _BASE + "（每日凌晨重建）"
_UNRELATED = "办公室绿植每周三由行政浇水一次"


def _archive_records(store) -> list:
    path = store.home / "memory" / "l2_archive.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _table_contents(store) -> list:
    lancedb = pytest.importorskip("lancedb")
    db = lancedb.connect(str(store.home / "memory" / "l2"))
    return db.open_table("memories").to_arrow()["content"].to_pylist()


def _fp(text: str) -> str:
    """内容指纹：与墓碑/查重同一份实现（_lifecycle.content_fp）。"""
    return cli.plugin_module("_lifecycle").content_fp(text)


def _patch_llm(monkeypatch, result) -> None:
    """替换共享 LLM 调用（consolidate 走 module 属性查找，patch 即生效）。"""
    monkeypatch.setattr(llm_mod, "chat_completion",
                        lambda *a, **k: result)


def _seed_with_categories(home, items) -> None:
    """按 (content, category) 播种 —— 分类不同保证两组近重复互不吸收，
    构造 size 2 与 size 4 两个簇时唯一变量就是簇大小。"""
    lancedb = pytest.importorskip("lancedb")

    l2 = home / "memory" / "l2"
    l2.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(l2))
    if "memories" in db.table_names():
        db.drop_table("memories")
    db.create_table("memories", data=[{
        "content": t, "category": cat, "source": "test",
        "timestamp": "2026-09-16T00:00:00", "vector": _fake_vec(t),
        "source_rowid": None, "role": "agent", "agent": "tester",
        "project": None,
    } for t, cat in items])


# ---------------------------------------------------------------------------
# 1) 证据链：--apply 成功后归档 JSONL 双向可追溯
# ---------------------------------------------------------------------------

class TestEvidenceChain:
    def test_apply_writes_bidirectional_evidence_chain(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_NEAR_A, _NEAR_B, _UNRELATED])
        merged = "构建机 build-07 每日凌晨重建，使用 Ubuntu 24.04 编译内核模块"
        _patch_llm(monkeypatch, merged)

        out = cli.cmd_consolidate({}, apply=True, agent="tester")
        assert out["ok"] is True and out["errors"] == []
        assert len(out["merged"]) == 1

        records = _archive_records(store)
        superseded = [r for r in records if r["state"] == "superseded"]
        merged_rows = [r for r in records if r["state"] == "merged"]
        # 每条被合并记录 + 合并后单条各一条归档：2 + 1
        assert len(superseded) == 2 and len(merged_rows) == 1

        fp_a, fp_b, fp_m = _fp(_NEAR_A), _fp(_NEAR_B), _fp(merged)
        by_content = {r["content"]: r for r in superseded}
        assert set(by_content) == {_NEAR_A, _NEAR_B}
        for rec in superseded:
            # merged_into：合并后单条的摘要 + 指纹（这条被谁取代了）
            assert rec["merged_into"]["fp"] == fp_m
            assert rec["merged_into"]["content"] == merged[:80]
            # merged_from：同簇其它成员的摘要 + 指纹（同簇还有谁）
            others = rec["merged_from"]
            assert len(others) == 1
            if rec["content"] == _NEAR_A:
                assert others[0]["fp"] == fp_b
                assert others[0]["content"] == _NEAR_B[:80]
            else:
                assert others[0]["fp"] == fp_a
                assert others[0]["content"] == _NEAR_A[:80]
        # 合并后单条那一侧：fp 与 merged_from 列出的全部原条 —— 与各原条的
        # merged_into 互为对应（从归档任一条出发都能走到另一端）
        row = merged_rows[0]
        assert row["fp"] == fp_m
        assert row["content"] == merged
        assert {r["fp"] for r in row["merged_from"]} == {fp_a, fp_b}
        assert {r["merged_into"]["fp"] for r in superseded} == {row["fp"]}


# ---------------------------------------------------------------------------
# 2) + 3) 置信度：定义单调；候选文件按它降序
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_is_monotone_in_cluster_size(self):
        """定义：size/(size+1)，只按证据数单调映射到 (0,1)，无模型分数。"""
        vals = [cli._cons_confidence(n)
                for n in range(2, cli.CONS_MAX_CLUSTER + 1)]
        assert all(b > a for a, b in zip(vals, vals[1:])), "必须单调不减"
        assert all(0.0 < v < 1.0 for v in vals), "必须落在 (0,1)"
        assert vals[0] == pytest.approx(2 / 3)      # 两条证据
        assert vals[-1] == pytest.approx(5 / 6)     # 单簇上限五条证据
        # 防御性下限：入参异常也不出定义域（单行不成簇，最小值即 2）
        assert cli._cons_confidence(0) == cli._cons_confidence(2)
        assert cli._cons_confidence(1) == cli._cons_confidence(2)

    def test_candidates_file_sorted_by_confidence_desc(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        # 两个簇：note 一对（size 2）+ other 四条（size 4）。分类不同保证
        # 互不吸收，唯一变量就是簇大小；行序让小簇**先被发现** —— 文件里
        # 若是大簇在前，那只能是 confidence 降序排序的功劳。
        items = [(_BASE + "（丁）", "note"), (_BASE + "（戊）", "note"),
                 (_BASE, "other"), (_BASE + "（甲）", "other"),
                 (_BASE + "（乙）", "other"), (_BASE + "（丙）", "other")]
        _seed_with_categories(store.home, items)

        # 前置条件：发现序确实是 [小簇, 大簇]（否则排序测试无意义）
        rows = [{"content": t, "category": cat, "vector": _fake_vec(t)}
                for t, cat in items]
        assert [len(c) for c in cli._cons_clusters(rows)] == [2, 4]

        out = cli.cmd_consolidate({}, apply=False)
        assert out["ok"] is True and out["clusters"] == 2
        data = json.loads(Path(out["candidates_file"]).read_text(
            encoding="utf-8"))
        sizes = [c["size"] for c in data["clusters"]]
        confs = [c["confidence"] for c in data["clusters"]]
        assert sizes == [4, 2], "候选文件未按 confidence 降序排列"
        assert confs == sorted(confs, reverse=True)
        # size 4 的簇置信度严格高于 size 2 的簇：证据越多越可信
        assert confs[0] == pytest.approx(4 / 5)
        assert confs[1] == pytest.approx(2 / 3)
        assert confs[0] > confs[1]
        # evidence = 成员数，与 size 同向；定义写进文件本身
        assert all(c["evidence"] == c["size"] for c in data["clusters"])
        assert "confidence" in data["confidence_note"]


# ---------------------------------------------------------------------------
# 4) + 5) 保留路径：拒绝合并不动数据；失败路径也不动数据
# ---------------------------------------------------------------------------

class TestPreservePaths:
    def test_declined_merge_keeps_originals_and_writes_no_archive(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_NEAR_A, _NEAR_B, _UNRELATED])
        _patch_llm(monkeypatch, "   ")  # 空输出 = 模型拒绝合并

        out = cli.cmd_consolidate({}, apply=True)
        assert len(out["skipped"]) == 1
        skip = out["skipped"][0]
        assert skip["cluster"] == 0 and skip["reason"] == "declined"
        # skipped 也带 confidence 与 evidence（成员数）
        assert skip["confidence"] == pytest.approx(2 / 3)
        assert skip["evidence"] == 2
        assert out["merged"] == []
        # 原条目一条不少；没合并就不能写 merged_into —— 归档和墓碑都不写
        assert set(_table_contents(store)) == {_NEAR_A, _NEAR_B, _UNRELATED}
        assert _archive_records(store) == []
        assert not (store.home / "memory" / "l2_tombstones.json").exists()

    def test_llm_unavailable_skip_carries_confidence_and_evidence(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_NEAR_A, _NEAR_B, _UNRELATED])
        _patch_llm(monkeypatch, None)  # LLM 不可用 ≠ 拒绝合并，两个 reason 要能区分

        out = cli.cmd_consolidate({}, apply=True)
        assert len(out["skipped"]) == 1
        assert out["skipped"][0]["reason"] == "llm_unavailable"
        assert out["skipped"][0]["confidence"] == pytest.approx(2 / 3)
        assert out["skipped"][0]["evidence"] == 2
        assert set(_table_contents(store)) == {_NEAR_A, _NEAR_B, _UNRELATED}

    def test_archive_failure_preserves_originals(self, store, monkeypatch):
        """回归：归档失败（删除前的第一步）时，原条目一条都不许丢，
        errors 如实带上簇号 / confidence / evidence。"""
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_NEAR_A, _NEAR_B, _UNRELATED])
        _patch_llm(monkeypatch, "构建机 build-07 每日凌晨重建并用 Ubuntu 24.04 编译内核模块")
        life = cli.plugin_module("_lifecycle")

        def _boom(*a, **k):
            raise OSError("archive disk full")

        monkeypatch.setattr(life, "archive_append", _boom)

        out = cli.cmd_consolidate({}, apply=True, agent="tester")
        assert out["merged"] == []
        assert len(out["errors"]) == 1
        err = out["errors"][0]
        assert err["cluster"] == 0
        assert err["confidence"] == pytest.approx(2 / 3)
        assert err["evidence"] == 2
        # 只留异常类型名：异常文本绝不急切求值（__str__ 自身抛错会把
        # "如实上报"变成二次崩溃 —— 全包已修 17 处同类缺陷）
        assert err["error"] == "OSError"
        # 归档先于删除：第一步就炸 ⇒ 删除从未发生，原条目全在
        assert set(_table_contents(store)) == {_NEAR_A, _NEAR_B, _UNRELATED}
        assert _archive_records(store) == []
