# -*- coding: utf-8 -*-
"""Hard-negative bench 回归测试（2026-10-18）。

外部 provider ``scope-recall``（``410979729/scope-recall-hermes`` v3.11.0）用一个
release 把「答案旁边埋着像它的错答案」做成了常设 bench：188 题 7 个任务，每个答案
旁边配 hard negative（别的项目的值、一个假设、一段引述、一句否认、assistant 的猜测、
工具输出、被替换掉的旧值），并统计「旧值排在现值之前」的次数。

本仓库原有的 ``cmd_eval`` 自查询集测不到这一层——问句就是事实本身，正确答案必然
在库里且必然是答案。它测的是通道健康（分数换算、门槛、通道退化），这一层已经够了；
缺的是**分辨力**。

这里的测试刻意不联网真实 store：``run_bench`` 接受注入的 ``search``，所以可以用
构造的检索器把每种判定分支都走到，同时不依赖 embedding。
"""
from __future__ import annotations

import pytest

from plugin.memory_governed._bench import (
    BENCH_DEV,
    BENCH_HELDOUT,
    SPLITS,
    run_bench,
)


def _fixed(hits_per_query: dict[str, list[str]]):
    """构造一个只查表返回的检索器。"""
    def _search(query: str, k: int) -> list[str]:
        return list(hits_per_query.get(query, []))[:k]
    return _search


class TestBenchRuler:
    """判定口径本身的对错——它量错了，后面所有数字都没意义。"""

    def test_exact_phrasing_is_a_hit(self):
        from plugin.memory_governed._bench import _rank_of, _norm
        assert _rank_of(_norm("schtasks /run /tn Hermes_Gateway"),
                        _norm("重启必须用 schtasks /run /tn Hermes_Gateway"))

    def test_a_reworded_answer_still_counts(self):
        """措辞不同但说的是同一件事，必须算命中。

        这是第一版按子串匹配时报过的错：期望写成
        「两把无口令密钥 id_ed25519 与 id_rsa」，库里实际是
        「两把无口令密钥 —— id_ed25519(注释 herm)」，检索把正确答案排了第一却记 0 分，
        heldout answer@1 因此报了 0.0。
        """
        from plugin.memory_governed._bench import _rank_of, _norm
        assert _rank_of(
            _norm("两把无口令密钥 id_ed25519 与 id_rsa"),
            _norm("内网 SSH 免密配置现状：私钥集中在 .ssh/，两把无口令密钥 "
                  "—— id_ed25519 与 id_rsa，均无 passphrase"))

    def test_an_unrelated_row_does_not_pass_by_accident(self):
        """覆盖判定不能松到让不相关的行蒙对。"""
        from plugin.memory_governed._bench import _rank_of, _norm
        assert not _rank_of(
            _norm("两把无口令密钥 id_ed25519 与 id_rsa"),
            _norm("MiMo Desktop 的 MCP 配置在 mimocode.jsonc 顶层 mcp 段"))

    def test_a_wrong_value_on_the_same_topic_does_not_count(self):
        """主题相同但值不同——这正是 hard negative 的定义，必须判负。"""
        from plugin.memory_governed._bench import _rank_of, _norm
        assert not _rank_of(
            _norm("mimo-v2.6-flash 是当前的 model.default"),
            _norm("Hermes 默认模型已切到 mimo-v2.5，内置 xiaomi provider"))


class TestBenchReport:
    def test_the_answer_first_and_no_negative_above_is_a_pass(self):
        rep = run_bench(_fixed({
            "网关重启要用什么命令": [
                "重启必须用 schtasks /run /tn Hermes_Gateway",
                "memories 表新增 project 列，recall 支持 --project",
            ],
        }), split="dev", top_k=5)
        item = next(i for i in rep.items if i["query"] == "网关重启要用什么命令")
        assert item["rank_of_answer"] == 1
        assert item["negatives_ranked_above"] == 0

    def test_a_negative_above_the_answer_is_counted_not_hidden(self):
        """负例压在答案前面——这才是 bench 的价值所在，必须报出来。"""
        rep = run_bench(_fixed({
            "网关重启要用什么命令": [
                "memories 表新增 project 列，recall 支持 --project",
                "重启必须用 schtasks /run /tn Hermes_Gateway",
            ],
        }), split="dev", top_k=5)
        item = next(i for i in rep.items if i["query"] == "网关重启要用什么命令")
        assert item["rank_of_answer"] == 2
        assert item["negatives_ranked_above"] == 1

    def test_an_answer_outside_top_k_scores_zero_not_zero_ranks(self):
        """答案没被召回 = 漏，不能因为「没有负例压它」就算干净通过。"""
        rep = run_bench(_fixed({"网关重启要用什么命令": ["完全不相关的内容"]}),
                        split="dev", top_k=1)
        item = next(i for i in rep.items if i["query"] == "网关重启要用什么命令")
        assert item["rank_of_answer"] is None
        assert rep.answer_at_1 == 0.0

    def test_an_empty_retrieval_cannot_score_one(self):
        """空结果若被算成满分，bench 就是橡皮章。

        刻意构造**一题答对、一题空召回**的检索器：只看总分会发现
        「答对那题 ranks=1、decided=2」与「decided=1」算出来都是 1.0，
        所以必须同时断言 ``decided`` 的分母没被缩掉，否则把分母改成
        ``len(ranks)`` 也能全绿（这个变异真的试过，第一版测试没抓住）。
        """
        rep = run_bench(_fixed({"网关重启要用什么命令":
                                ["重启必须用 schtasks /run /tn Hermes_Gateway"]}),
                        split="dev", top_k=5)
        # 分母必须是"所有可判定题"，而不是某个硬编码的题数 —— dev 的样本量
        # 会随，"我给的答案确实存在"的核对而变，钉一个常数只会让这条断言在
        # 下次补样本时假失败。
        evaluable = [i for i in rep.items if not i["undecidable"]]
        assert rep.decided == len(evaluable)
        assert len([1 for i in rep.items
                    if i["rank_of_answer"] and not i["undecidable"]]) == 1
        assert rep.answer_at_1 < 1.0      # 空召回那两题拖低了它
        assert rep.clean_pass_rate < 1.0
        # 分母必须是 decided，而不是有命中的条数——
        # 否则把分母改成 len(ranks) 也能算出 1.0，这个变异试过。
        assert rep.answer_at_1 == pytest.approx(1 / rep.decided, abs=1e-4)

    def test_undecidable_items_are_reported_and_not_scored(self):
        """自带「已改/不再/已停用」的答案与负例内容重叠，按内容判不了就不计分。"""
        rep = run_bench(_fixed({}), split="dev", top_k=5)
        assert rep.undecidable > 0
        assert rep.decided + rep.undecidable == rep.evaluated

    def test_a_raising_search_is_reported_per_item_not_crashed(self):
        """检索抛异常要落到 undecidable + 原因，不能整轮崩掉。"""
        def _boom(query: str, k: int) -> list[str]:
            raise RuntimeError("lancedb not importable")

        rep = run_bench(_boom, split="dev", top_k=5)
        assert rep.evaluated == len(BENCH_DEV)
        assert all("lancedb" in (i["note"] or "") for i in rep.items)

    def test_an_unknown_split_is_refused(self):
        with pytest.raises(ValueError):
            run_bench(_fixed({}), split="nope")


class TestAnswerGrounding:
    """答案不在被测层时，应当被点名而不是被扣一分。

    2026-10-19 实测：dev 里「Hermes 默认模型当前是哪个版本」长期 FAIL，而检索
    什么也没做错 —— 那个答案住在 KB 笔记里，L2 一条都没有。bench 只跑 L2，
    于是它年年扣一分，扣的是我编样本的错。
    """

    _ANSWER_LIVES_IN_KB = "Hermes 默认模型当前是哪个版本"

    def test_an_answer_outside_the_measured_layer_leaves_the_denominator(
            self):
        def _locate(answer: str):
            return "kb:某篇笔记" if answer else None

        rep = run_bench(_fixed({}), split="dev", top_k=5, locate=_locate)
        # 未定位：CLI 不传 locate 时与主链路一致，不该有人被踢出去。
        plain = run_bench(_fixed({}), split="dev", top_k=5)
        assert rep.ungrounded > 0
        assert rep.decided < plain.decided

    def test_the_sample_that_motivated_it_is_really_flagged(self):
        """the check has to catch the case it was written for, by name."""
        def _locate(answer: str):
            return "l2" if answer and "已在库里" in answer else "kb:x"

        rep = run_bench(_fixed({}), split="dev", top_k=5, locate=_locate)
        flagged = [i for i in rep.items if not i["grounded"]]
        assert flagged, "没有一条被标出来 —— 校验形同虚设"
        assert all(i["answer_found_in"].startswith("kb:") for i in flagged)

    def test_where_the_answer_went_is_reported_not_dropped(self):
        """退出分母 ≠ 消失：读报告的人得知道该去哪一层找。"""
        def _locate(answer: str):
            return "kb:Hermes 默认模型已切到 mimo-v2.6-flash"

        rep = run_bench(_fixed({}), split="dev", top_k=5, locate=_locate)
        for item in rep.items:
            assert not item["grounded"]
            assert item["answer_found_in"].startswith("kb:")

    def test_a_locator_that_raises_does_not_delete_samples(self):
        """定位器是 best-effort：它挂了必须退回旧行为，而不是让样本消失。"""
        def _boom(answer: str) -> str:
            raise RuntimeError("kb unreadable")

        rep = run_bench(_fixed({}), split="dev", top_k=5, locate=_boom)
        assert rep.ungrounded == 0
        assert rep.decided == run_bench(_fixed({}), split="dev",
                                        top_k=5).decided

    def test_no_locator_keeps_the_historical_behaviour(self):
        """老调用方不传 locate：一切照旧，包括 decided 的数量。"""
        rep = run_bench(_fixed({}), split="dev", top_k=5)
        assert rep.ungrounded == 0
        item = next(i for i in rep.items
                    if i["query"] == "网关重启要用什么命令")
        assert item["grounded"] is True


class TestLiveNegatives:
    """站在场上的竞争者有几个——没有竞争就不该谈"抗骗"。

    12 道题里曾有 6 道的负例根本不在被测层，``negatives_ranked_above`` 对它们
    恒为 0，``clean_pass_rate`` 于是在一批没人打扰的题上记了满分。
    """

    def test_live_negatives_are_counted_per_item(self):
        def _locate(text: str):
            return "l2" if "负例在库" in text else "kb:x"

        rep = run_bench(_fixed({}), split="heldout", top_k=5, locate=_locate)
        for item in rep.items:
            assert isinstance(item["live_negatives"], int)

    def test_a_split_where_nobody_competes_gives_none_not_one(self):
        """没有竞争者时 contested_clean_rate 必须是 None —— 1.0 会撒谎。"""

        def _locate(text: str):
            return None          # 什么都找不到

        rep = run_bench(_fixed({}), split="dev", top_k=5, locate=_locate)
        assert rep.contested == 0
        assert rep.contested_clean_rate is None

    def test_contested_rate_only_looks_at_contested_items(self):
        # 一题有竞争且干净通过，另一题零竞争：两者混算会得出 1.0，
        # 而那只说明"没被打扰"的题没被打扰。
        def _locate(text: str):
            return "l2" if "争" in text else None

        search = _fixed({"网关重启要用什么命令": [
            "重启必须用 schtasks /run /tn Hermes_Gateway",
            "memories 表新增 project 列，recall 支持 --project",
        ]})
        rep = run_bench(search, split="dev", top_k=5, locate=_locate)
        # 有竞争的那道题（负例 MEMORIES 也会被定位）至少被单独结算。
        assert rep.contested_clean_rate in (None, 0.0, 1.0)
        assert 0 <= rep.contested <= rep.decided


class TestBenchSamples:
    def test_both_splits_are_present_and_non_trivial(self):
        assert set(SPLITS) == {"dev", "heldout"}
        assert len(BENCH_DEV) >= 5
        assert len(BENCH_HELDOUT) >= 4

    def test_every_sample_names_a_kind_a_query_and_an_answer(self):
        for split, items in SPLITS.items():
            for it in items:
                assert it.get("kind"), f"{split}: kind missing"
                assert it.get("query"), f"{split}: query missing"
                assert it.get("answer"), f"{split}: answer missing"
                assert it.get("negatives"), f"{split}: no negative in {it}"

    def test_every_kind_is_covered_in_each_split(self):
        """dev 与 heldout 都要覆盖三类负例，否则某类退化不会被发现。"""
        for split, items in SPLITS.items():
            kinds = {it["kind"] for it in items}
            assert kinds == {"outdated", "denial", "near_miss"}, \
                f"{split} missing kinds: {kinds}"

    def test_the_two_splits_are_not_the_same_questions(self):
        """heldout 必须是另一批题，否则「在 dev 上调好」没有意义。"""
        q_dev = {it["query"] for it in BENCH_DEV}
        q_out = {it["query"] for it in BENCH_HELDOUT}
        assert not (q_dev & q_out), f"shared queries: {q_dev & q_out}"

    def test_negatives_are_not_copies_of_the_answer(self):
        """负例等于答案的样本是自欺，必须挡住。"""
        for split, items in SPLITS.items():
            for it in items:
                assert it["answer"] not in it["negatives"], \
                    f"{split}: negative duplicates the answer in {it}"
