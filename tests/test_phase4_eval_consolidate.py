# -*- coding: utf-8 -*-
# ruff: noqa: F811 — pytest fixture 跨文件复用（store 由 import 引入，
# 各测试以参数消费它时被静态分析视为重绑定；pytest 标准模式）。
"""Phase 4 测试：巩固合并 / 检索评测 / 富化 / 信封与退出码（WeKnora 借鉴）。

1. **consolidate**：同类 + Jaccard≥0.55 + 余弦≥0.86 才成簇；dry-run 只报
   不动；--apply 时 LLM 合并（空输出 = 拒绝保留原条；LLM/嵌入不可用 =
   跳过且不碰数据），成功路径原条归档（reason=consolidated）+ 墓碑 + 插入
   合并条；
2. **eval**：self-snapshot 自评 hit@k（通道健康门）与自定义 QA 文件的
   miss 报告；
3. **enrich**：``kb.enrich`` 开启才调 LLM，摘要/自问进 frontmatter，
   失败不阻断写入；
4. **信封与退出码**：``--envelope`` 出 ``{ok, data, meta}``；破坏性操作
   缺 ``--yes`` → exit 10 且不执行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import memory_cli as cli
from plugin.memory_governed import _kb as kb_mod
from plugin.memory_governed import _llm as llm_mod

# 复用 HERMES_HOME store fixture、假嵌入与向量播种（绝不外呼）
from tests.test_cli_security_and_recall import (  # noqa: F401 — pytest fixture
    _install_fake_embedding,
    _seed_vectors,
    store,
)
from tests.test_kb_governance import _make_kb  # noqa: F401

_FACT = "家用NAS 192.0.2.62 上重度使用 Docker 部署服务"
_FACT_B = "SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE"
# 近重复对：中文 + 无 fake 轴词汇 → 两向量都落在兜底轴上（余弦 1.0），
# 词法 bigram 高重叠 —— 唯一变量就是聚簇逻辑本身。
_NEAR_A = "构建机 build-07 使用 Ubuntu 24.04 编译内核模块"
_NEAR_B = "构建机 build-07 使用 Ubuntu 24.04 编译内核模块（每日凌晨重建）"
_UNRELATED = "办公室绿植每周三由行政浇水一次"


def _table_contents(store) -> list:
    lancedb = pytest.importorskip("lancedb")
    db = lancedb.connect(str(store.home / "memory" / "l2"))
    return db.open_table("memories").to_arrow()["content"].to_pylist()


def _archive_records(store) -> list:
    path = store.home / "memory" / "l2_archive.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _patch_llm(monkeypatch, result) -> None:
    """替换共享 LLM 调用（consolidate 走 module 属性查找，patch 即生效）。"""
    monkeypatch.setattr(llm_mod, "chat_completion",
                        lambda *a, **k: result)


# ---------------------------------------------------------------------------
# 1) consolidate：聚簇纯函数
# ---------------------------------------------------------------------------

class TestConsClusters:
    def test_same_category_and_both_thresholds_cluster(self):
        rows = [
            {"content": _NEAR_A, "category": "other", "vector": [1.0, 0.0]},
            {"content": _NEAR_B, "category": "other", "vector": [1.0, 0.05]},
        ]
        assert cli._cons_clusters(rows) == [[0, 1]]

    def test_different_category_never_clusters(self):
        rows = [
            {"content": _NEAR_A, "category": "other", "vector": [1.0, 0.0]},
            {"content": _NEAR_B, "category": "note", "vector": [1.0, 0.0]},
        ]
        assert cli._cons_clusters(rows) == []

    def test_low_jaccard_never_clusters(self):
        rows = [
            {"content": _NEAR_A, "category": "other", "vector": [1.0, 0.0]},
            {"content": _UNRELATED, "category": "other", "vector": [1.0, 0.0]},
        ]
        assert cli._cons_clusters(rows) == []

    def test_low_cosine_never_clusters(self):
        # 文本近重复但向量正交 → 词法像、语义不像：双条件缺一不可
        rows = [
            {"content": _NEAR_A, "category": "other", "vector": [1.0, 0.0]},
            {"content": _NEAR_B, "category": "other", "vector": [0.0, 1.0]},
        ]
        assert cli._cons_clusters(rows) == []

    def test_cluster_size_is_capped(self):
        """单簇上限：6 条互相近重复不许并成一簇（合并有损，越大越危险）。"""
        rows = [{"content": _NEAR_A, "category": "other", "vector": [1.0, 0.0]}
                for _ in range(6)]
        got = cli._cons_clusters(rows)
        assert len(got) == 1
        assert len(got[0]) == cli.CONS_MAX_CLUSTER < 6

    def test_no_transitive_chain(self):
        """代表制：A~B 且 B~C、但 A≁C 时，C **不得**被链进同一簇。

        并查集（传递闭包）会把 {A,B,C} 并成一簇 —— 哪怕 A 与 C 毫不相干；
        链式扩张没有上界，而 LLM 合并是**有损**的（误合并比漏合并贵）。
        """
        a, b, c = "aaaabbbb", "aaaabbbbcccc", "bbbbcccc"
        jab = cli._cons_jaccard(cli._cons_bigrams(a), cli._cons_bigrams(b))
        jbc = cli._cons_jaccard(cli._cons_bigrams(b), cli._cons_bigrams(c))
        jac = cli._cons_jaccard(cli._cons_bigrams(a), cli._cons_bigrams(c))
        assert jab >= cli.CONS_JACCARD_MIN and jbc >= cli.CONS_JACCARD_MIN
        assert jac < cli.CONS_JACCARD_MIN, "前置条件没造对，用例无意义"
        rows = [{"content": t, "category": "other", "vector": [1.0, 0.0]}
                for t in (a, b, c)]
        assert cli._cons_clusters(rows) == [[0, 1]], "C 被链式并进来了"


# ---------------------------------------------------------------------------
# 2) consolidate：命令级（dry-run / apply / 各失败路径）
# ---------------------------------------------------------------------------

class TestConsolidateCommand:
    def _seed(self, store):
        _seed_vectors(store.home, [_NEAR_A, _NEAR_B, _UNRELATED])

    def test_dry_run_reports_without_touching_data(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        self._seed(store)

        out = cli.cmd_consolidate({}, apply=False)
        assert out["ok"] is True and out["applied"] is False
        assert out["clusters"] == 1
        assert out["merged"] == [] and _archive_records(store) == []
        assert len(_table_contents(store)) == 3

    def test_dry_run_writes_the_candidates_file(self, store, monkeypatch):
        """候选整份落文件：几十上百个簇没法靠 stdout 的 80 字前缀人工审，
        而 ``--apply`` 有损 —— 审必须在动手之前。"""
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        self._seed(store)

        out = cli.cmd_consolidate({}, apply=False)
        path = Path(out["candidates_file"])
        assert path.exists(), "候选文件没落盘"
        assert path.parent == store.home / "memory"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["clusters"], "候选文件里没有簇"
        assert data["thresholds"]["max_cluster"] == cli.CONS_MAX_CLUSTER
        member = data["clusters"][0]["members"][0]
        assert member["content"] and "index" in member

    def test_apply_merges_and_supersedes_originals(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        self._seed(store)
        merged = "构建机 build-07 每日凌晨重建，使用 Ubuntu 24.04 编译内核模块"
        _patch_llm(monkeypatch, merged)

        out = cli.cmd_consolidate({}, apply=True, agent="tester")
        assert out["ok"] is True and out["applied"] is True
        assert out["errors"] == [] and out["skipped"] == []
        assert len(out["merged"]) == 1

        remaining = _table_contents(store)
        assert _NEAR_A not in remaining and _NEAR_B not in remaining
        assert merged in remaining and _UNRELATED in remaining

        records = _archive_records(store)
        assert len(records) == 2
        assert all(r["state"] == "superseded" for r in records)
        assert all(r["reason"] == "consolidated" for r in records)

    def test_apply_declined_keeps_originals(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        self._seed(store)
        _patch_llm(monkeypatch, "   ")  # 空输出 = 模型拒绝合并

        out = cli.cmd_consolidate({}, apply=True)
        assert out["skipped"] == [{"cluster": 0, "reason": "declined"}]
        assert out["merged"] == []
        assert len(_table_contents(store)) == 3
        assert _archive_records(store) == []

    def test_apply_llm_unavailable_skips_without_mutation(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        self._seed(store)
        _patch_llm(monkeypatch, None)

        out = cli.cmd_consolidate({}, apply=True)
        assert out["skipped"] == [{"cluster": 0, "reason": "llm_unavailable"}]
        assert len(_table_contents(store)) == 3

    def test_apply_without_embedding_refuses_entire_round(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch, available=False)
        self._seed(store)

        out = cli.cmd_consolidate({}, apply=True)
        assert out["ok"] is False and out["applied"] is False
        assert "embedding_unavailable" in str(out)
        assert len(_table_contents(store)) == 3


# ---------------------------------------------------------------------------
# 3) eval：自评 hit@k 与自定义集 miss
# ---------------------------------------------------------------------------

class TestEval:
    def test_self_snapshot_recalls_itself(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        assert cli.cmd_remember({}, _FACT, "dsh")["ok"] is True

        out = cli.cmd_eval({}, "")
        assert out["ok"] is True, out
        assert out["basis"] == "self-snapshot"
        assert out["evaluated"] >= 1
        assert out["hit_at_k"] == 1.0, out

    def test_file_qa_reports_misses(self, store, monkeypatch, tmp_path):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT, "dsh")
        qa = tmp_path / "qa.json"
        qa.write_text(json.dumps([
            {"q": _FACT, "expect": "绝不存在的期望内容"}],
            ensure_ascii=False), encoding="utf-8")

        out = cli.cmd_eval({}, str(qa))
        assert out["ok"] is True
        assert out["basis"] == "file"
        assert out["hit_at_k"] == 0.0
        assert len(out["misses"]) == 1

    def test_unreadable_qa_file_is_refused(self, store):
        out = cli.cmd_eval({}, str(Path("does") / "not" / "exist.json"))
        assert out["ok"] is False

    def test_min_hit_below_the_floor_fails(self, store, monkeypatch):
        """没有下限时 hit@k 只是个数字 —— 实测它在 floor∈[0.0, 0.99] 全程恒为
        1.0，只有 >1.0 才掉下去。有了下限才配叫验收门（``ok:false`` → 非零退出）。"""
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        assert cli.cmd_remember({}, _FACT, "dsh")["ok"] is True

        ok = cli.cmd_eval({}, "", min_hit=0.5)
        assert ok["ok"] is True and ok["min_hit_at_k"] == 0.5

        bad = cli.cmd_eval({}, "", min_hit=1.01)
        assert bad["ok"] is False
        assert "below the required" in bad["error"]

    def test_report_names_the_engine_and_the_sample_pool(self, store, monkeypatch):
        """上个版本的门对本次交付全盲、却自称是它的验收门 —— 现在必须写明用的是
        哪条路、从多大的池子里抽的样（否则"这扇门测的是哪条路"又变成一个谜）。"""
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        assert cli.cmd_remember({}, _FACT, "dsh")["ok"] is True

        out = cli.cmd_eval({}, "")
        assert out["engine"] == "cli-recall_l2", out
        assert out["sampled_from"] >= 1
        assert "snapshot_truncated" in out


# ---------------------------------------------------------------------------
# 4) enrich：默认关零调用；开启后摘要/自问进 frontmatter；失败不阻断
# ---------------------------------------------------------------------------

class TestEnrich:
    _RESPONSE = json.dumps({
        "summary": "NAS 上以 Docker 重度部署服务",
        "questions": ["NAS 上跑什么？", "部署密度如何？", "有几个服务？", "第四条被截断"],
    }, ensure_ascii=False)

    def _patch_kb_llm(self, monkeypatch, result, calls: list):
        def _fake(*a, **k):
            calls.append(1)
            return result
        monkeypatch.setattr(kb_mod, "chat_completion", _fake)

    def test_off_by_default_no_llm_call(self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_kb_llm(monkeypatch, self._RESPONSE, calls)
        kb = _make_kb(tmp_path)
        assert kb.add("富化关闭", "正文内容", section="notes")["ok"]
        assert calls == []
        assert "summary" not in kb.get("富化关闭")["meta"]

    def test_enriched_note_carries_summary_and_questions(
            self, tmp_path, monkeypatch):
        calls: list = []
        self._patch_kb_llm(monkeypatch, self._RESPONSE, calls)
        kb = _make_kb(tmp_path)
        kb._config.kb.enrich = True
        assert kb.add("富化开启", "正文内容", section="notes")["ok"]
        assert calls == [1]
        meta = kb.get("富化开启")["meta"]
        assert meta["summary"]
        assert len(meta["questions"]) == 3  # 最多 3 条

    def test_llm_failure_never_blocks_the_write(self, tmp_path, monkeypatch):
        self._patch_kb_llm(monkeypatch, None, [])
        kb = _make_kb(tmp_path)
        kb._config.kb.enrich = True
        out = kb.add("富化失败", "正文内容", section="notes")
        assert out["ok"] is True
        assert "summary" not in kb.get("富化失败")["meta"]


# ---------------------------------------------------------------------------
# 5) 信封与退出码
# ---------------------------------------------------------------------------

class TestEnvelopeAndExitCodes:
    def _run(self, monkeypatch, capsys, argv):
        monkeypatch.setattr(sys, "argv", ["memory_cli.py"] + argv)
        code = cli.main()
        return code, json.loads(capsys.readouterr().out)

    def test_retract_without_yes_exits_10_and_does_nothing(
            self, store, monkeypatch, capsys):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_B, "dsh")

        code, out = self._run(monkeypatch, capsys,
                              ["retract", "ROCKET_TLS"])
        assert code == 10
        assert out["ok"] is False and "--yes" in out["hint"]
        assert _FACT_B in _table_contents(store)

    def test_eval_min_hit_below_the_floor_exits_nonzero(
            self, store, monkeypatch, capsys):
        """验收门必须**能失败**：低于下限 ⇒ ``ok:false`` ⇒ 非零退出。

        改之前它连下限参数都没有，hit@k 恒为 1.0，等于一个永远通过的门。
        """
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT, "dsh")

        code, out = self._run(monkeypatch, capsys,
                              ["eval", "--min-hit", "1.01"])
        assert code == 1, out
        assert out["ok"] is False and "below the required" in out["error"]

    def test_retract_dry_run_needs_no_yes(self, store, monkeypatch, capsys):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_B, "dsh")
        code, out = self._run(monkeypatch, capsys,
                              ["retract", "ROCKET_TLS", "--dry-run"])
        assert code == 0
        assert out["dry_run"] is True and out["matched"] == 1

    def test_consolidate_apply_without_yes_exits_10(
            self, store, monkeypatch, capsys):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_NEAR_A, _NEAR_B])
        code, out = self._run(monkeypatch, capsys,
                              ["consolidate", "--apply"])
        assert code == 10
        assert len(_table_contents(store)) == 2  # 什么都没动

    def test_envelope_wraps_ok_data_meta(self, store, monkeypatch, capsys):
        code, out = self._run(monkeypatch, capsys,
                              ["kb-search", "不存在的查询", "--envelope"])
        assert code in (0, 1)
        assert set(out) == {"ok", "data", "meta"}
        assert out["meta"]["cmd"] == "kb-search"
        assert "results" in out["data"]
        assert out["ok"] is True

    def test_envelope_ok_false_on_refusal(self, store, monkeypatch, capsys):
        code, out = self._run(monkeypatch, capsys,
                              ["retract", "", "--dry-run", "--envelope"])
        assert code == 1
        assert out["ok"] is False
        assert out["meta"]["cmd"] == "retract"
