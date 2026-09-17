# -*- coding: utf-8 -*-
"""两个 P0 的回归测试（2026-09-17）。

**P0-1 索引陈旧既不报也不自愈**：索引同步只在进程启动时跑一次，会话中新写入
的笔记（`kb-add` 之外的外部直写 vault / 同步盘 / Obsidian 手工编辑）进不了索引，
``kb-search`` 搜不到却什么都不说。

这里的做法是把语义限制在「只有入过索引的笔记才有语义分」的替身上 —— 这才能
真正复现线上故障。如果用真实关键词命中来测，笔记会因为**字面匹配**照样被搜到，
测出来的永远是绿的（故障精制地绕过测试，正是这类 bug 活了这么久的原因）。

**P0-2 关键词通道被融合权重结构性卡死**：融合分是 ``0.4*kw + 0.6*sem``，
而部署的 ``recall_min_score = 0.45`` 直接套在融合分上，纯关键词命中的理论
上限就是 0.4 —— 再准也够不到门槛。改成逐通道判定后这条通路才重新可用。

两个测试都要求：在修复前失败、修复后通过。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase
from plugin.memory_governed._recall import RecallEngine

NOTE_A = "---\ntitle: A\n---\nalpha body"
NOTE_B = "---\ntitle: B\n---\nbeta body"


def _make_kb(tmp_path: Path, notes: dict) -> KnowledgeBase:
    """真实 vault（写文件）+ embedding 关闭的 KB。"""
    wiki = tmp_path / "wiki"
    for rel, body in notes.items():
        p = wiki / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(wiki)
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    kb = KnowledgeBase(cfg)
    kb.ensure(sync=False)
    return kb


class _SemanticIndex:
    """语义替身：**只有入过索引的笔记才可能出现在语义结果里**。

    这是复现线上故障的关键：真实的 LanceDB 索引当然如此，新的外部写入必须先
    upsert 才能被语义检索到。``path_mtimes`` / ``upsert`` / ``delete`` 都按
    真实 KBIndex 的契约实现，并在 count 里记账，好量化「有没有白重嵌」。
    """

    available = True
    last_error = ""

    def __init__(self, score: float = 0.8):
        self.rows: dict[str, tuple] = {}   # path -> (text, mtime)
        self.score = score
        self.upserts: list[str] = []

    def path_mtimes(self):
        return {p: mt for p, (_t, mt) in self.rows.items()}

    def upsert(self, path, text, mtime=0.0):
        self.upserts.append(path)
        self.rows[path] = (text, float(mtime))

    def delete(self, path):
        self.rows.pop(path, None)

    def drop(self):
        self.rows.clear()

    def search(self, query, k):
        return [
            {"path": p, "text": t, "score": self.score, "kind": "semantic"}
            for p, (t, _m) in list(self.rows.items())[:k]
        ]


def _install(kb: KnowledgeBase, idx: _SemanticIndex) -> _SemanticIndex:
    kb._index_get = lambda: idx
    return idx


def _make_upsert_fail(idx: _SemanticIndex) -> None:
    """让这个索引的 ``upsert`` 从此只记账、**不落库**。

    模拟「同步跑了却仍然对不上」：文件在同步期间持续被改、upsert 写入失败、
    或后端拒绝写入。这条路径要验的是「重探之后仍然不新鲜时必须**如实报**，
    不能因为发生过自愈就强行折成 ``fresh=True``」—— 那才是真信号。

    必须**就地**改同一个索引对象：换一个新的替身会把已入索引的笔记一起清空，
    测出来的是「全没了」而不是「补不上」（本机实测过这个假失败）。
    """
    def upsert(path, text, mtime=0.0):
        idx.upserts.append(path)   # 只记账，rows 保持不变

    idx.upsert = upsert


# ---------------------------------------------------------------------------
# P0-1
# ---------------------------------------------------------------------------

class TestSearchSelfHealsStaleIndex:
    """① kb-add 新笔记 ② 外部直写 vault ③ 修改已有内容 —— 都要能当场搜到。"""

    def test_external_write_is_searchable_immediately(self, tmp_path):
        """外部直写 vault 的新笔记，不显式调 sync_index 也必须在结果里。"""
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()

        # 模拟外部写入（DSH / Obsidian / 同步盘）：完全绕过 _kb.add
        external = tmp_path / "wiki" / "notes" / "外写的.md"
        external.write_text("---\ntitle: 外写的\n---\n外部写入的正文",
                            encoding="utf-8")

        hits = kb.search("alpha", top_k=10)
        paths = [h["path"] for h in hits]
        assert "notes/外写的.md" in paths, (
            "外部写入的笔记没被检索到 —— 索引自愈没跑。search 前的 mtime "
            "探测应该发现它与索引不一致并触发增量同步"
        )
        assert "notes/外写的.md" in idx.rows, "笔记必须真的进了索引投影"

    def test_edited_note_searches_new_content_not_old(self, tmp_path):
        """改了内容之后，索引里留的必须是新文本，不能还是旧的。"""
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()
        assert "alpha body" in idx.rows["notes/a.md"][0]

        target = tmp_path / "wiki" / "notes" / "a.md"
        target.write_text("---\ntitle: A\n---\n改过之后的正文", encoding="utf-8")
        # 显式把 mtime 推到未来：文件系统时钟粒度（以及 0.5s 容差）会让同一
        # tick 内的重写看起来「没变」。真实外部编辑发生在几秒之后，不会踩到
        # 这个窗口；这里必须把它构造出来，否则测的其实是精度问题而不是自愈。
        future = time.time() + 10.0
        os.utime(target, (future, future))

        kb.search("alpha", top_k=10)
        assert "改过之后的正文" in idx.rows["notes/a.md"][0], (
            "索引里还是旧文本 —— 修改过的笔记没有被重嵌，用户会读到过期内容"
        )

    def test_index_status_is_exposed_in_search_output(self, tmp_path):
        """陈旧状态必须出现在检索结果里，不能只在 health 里算着玩。

        契约（2026-09-17 修正）：自愈**成功之后重新探测**，
        ``fresh`` / ``missing`` / ``stale`` / ``orphaned`` 报的是**当前真值**；
        同步**前**的差异计数显式留在 ``pre_sync_*``。

        旧实现只挂同步前那一份快照，于是一个字典里同时是 ``synced=True`` 和
        ``fresh=False, missing=1`` —— 而 ``search()`` 正是在自愈**之后**才构建
        结果的，那句「结果可能不全」是误报（实测：自愈 missing 1→0、indexed
        26→27 都成功）。「陈旧必须可见」这条原意改由 ``pre_sync_missing`` 守住。
        """
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()  # 先对齐

        fresh = kb.search("alpha", top_k=10)[0]["index_status"]
        assert fresh["fresh"] is True
        assert fresh["missing"] == 0 and fresh["stale"] == 0

        (tmp_path / "wiki" / "notes" / "b.md").write_text(NOTE_B, encoding="utf-8")
        healed = kb.search("alpha", top_k=10)[0]["index_status"]
        assert healed["synced"] is True, "必须标明已经自愈过"
        # 同步前的差异 —— 「这一条发生时『结果可能不全』」仍然必须可见
        assert healed["pre_sync_missing"] == 1, (
            "自愈前漏了 1 篇这件事必须还查得到 —— 不能因为改报『已新鲜』就抹掉"
            "（这正是本次修正不能丢掉的东西）")
        assert healed["pre_sync_stale"] == 0
        assert healed["pre_sync_orphaned"] == 0
        # 同步后的当前真值 —— 已经补上了，就不该再报「结果可能不全」
        assert healed["fresh"] is True, (
            "自愈成功后 fresh 仍是 False —— search() 的结果其实是完整的，"
            "这是误报（修复前：synced=True 却 fresh=False、missing=1）")
        assert healed["missing"] == 0, "重探之后 missing 必须是当前真值 0"
        assert "notes/b.md" in idx.rows, "笔记必须真的进了索引投影"

    def test_second_search_reports_fresh_current_truth(self, tmp_path):
        """连续两次 ``search()``：第二次的快照必须报 ``fresh: true``。

        修复前最刺眼的地方就是这里：自愈明明成功了（missing 1→0），检索结果其实
        完整，快照却报 ``fresh=false, missing=1``。用户被告知「结果可能不全」，
        而结果一点都不少。
        """
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()
        (tmp_path / "wiki" / "notes" / "b.md").write_text(NOTE_B, encoding="utf-8")

        first = kb.search("alpha", top_k=10)[0]["index_status"]
        assert first["synced"] is True, "第一次必须检测到陈旧并自愈"
        assert first["pre_sync_missing"] == 1, "自愈前的差异必须留痕"
        assert first["fresh"] is True, "自愈成功那一次就该报当前真值"

        upserts_after_first = len(idx.upserts)
        second = kb.search("alpha", top_k=10)[0]["index_status"]
        assert second["fresh"] is True, "第二次检索必须报新鲜"
        assert second["missing"] == 0 and second["stale"] == 0
        assert second["synced"] is False, "第二次没有再同步，不该标 synced"
        assert len(idx.upserts) == upserts_after_first, (
            "第二次检索又重嵌了 %d 篇 —— 新鲜时重探不得变成重建"
            % (len(idx.upserts) - upserts_after_first))

    def test_still_stale_after_sync_is_reported_honestly(self, tmp_path):
        """重探后仍然不新鲜时必须如实报，不能因为「自愈过」就折成 True。

        构造：先用正常索引对齐（a.md 在库里），再换成 upsert 不落库的替身，
        然后外部写入 b.md —— 同步**跑了**但补不上。这正是「文件持续变动 /
        upsert 失败」的形状。
        """
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()          # a.md 正常入索引
        _make_upsert_fail(idx)   # 此后 upsert 补不上
        (tmp_path / "wiki" / "notes" / "b.md").write_text(NOTE_B, encoding="utf-8")

        status = kb.search("alpha", top_k=10)[0]["index_status"]
        assert status["synced"] is True, "确实尝试同步过"
        assert status["fresh"] is False, (
            "同步了却仍然对不上时必须继续报不新鲜 —— 这是真信号，"
            "谎报新鲜等于把『结果可能不全』重新变回静默失败")
        assert status["missing"] == 1, "当前真值：确实还差 b.md 这一篇"
        assert status["pre_sync_missing"] == 1, "同步前的计数同样要留"

    def test_reprobe_happens_only_when_a_sync_actually_ran(self, tmp_path):
        """重探只允许发生在**本次真的同步过**之后。

        性能硬约束的另一半：新鲜时每次检索只该探测一次，不能为「取当前真值」
        白花第二次 stat + ``path_mtimes``。同步过之后才允许重探一次。
        """
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()

        calls = {"n": 0}
        base = idx.path_mtimes

        def counted():
            calls["n"] += 1
            return base()

        idx.path_mtimes = counted

        calls["n"] = 0
        kb.search("alpha", top_k=10)
        assert calls["n"] == 1, (
            "新鲜时探测了 %d 次 —— 重探不得变成每次检索的固定开销" % calls["n"])

        (tmp_path / "wiki" / "notes" / "b.md").write_text(NOTE_B, encoding="utf-8")
        calls["n"] = 0
        kb.search("alpha", top_k=10)
        assert calls["n"] == 2, (
            "同步过之后应当重探一次（1 次判陈旧 + 1 次取当前真值），实际 %d 次"
            % calls["n"])

    def test_index_status_stays_a_pure_current_truth_diagnostic(self, tmp_path):
        """``index_status()`` 是纯诊断（当前真值），不带 ``synced`` / ``pre_sync_*``。

        这两套语义必须分开：一个是「此刻索引什么状态」，一个是「刚才那次自愈
        做了什么」。混在一起就是本次缺陷的形状。
        """
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        _install(kb, _SemanticIndex())
        kb.sync_index()
        (tmp_path / "wiki" / "notes" / "b.md").write_text(NOTE_B, encoding="utf-8")

        diag = kb.index_status()
        assert diag["fresh"] is False and diag["missing"] == 1, "当前真值：还没同步"
        assert "synced" not in diag, "纯诊断不该带『本次是否自愈过』"
        assert "pre_sync_missing" not in diag, "纯诊断不该带同步前快照"

    def test_second_search_does_not_reembed_everything(self, tmp_path):
        """性能硬约束：新鲜时多次检索不能反复重建索引。"""
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A, "notes/b.md": NOTE_B})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()
        before = len(idx.upserts)

        for _ in range(5):
            kb.search("alpha", top_k=10)

        assert len(idx.upserts) == before, (
            "每次检索都在重新嵌入 —— 探测必须比一次重建便宜得多，"
            f"这 5 次搜索白白重嵌了 {len(idx.upserts) - before} 篇"
        )

    def test_probe_does_not_read_file_contents(self, tmp_path):
        """自愈探测只能 stat，不能 read —— 这是它比重建便宜的前提。"""
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A, "notes/b.md": NOTE_B})
        idx = _install(kb, _SemanticIndex())
        kb.sync_index()

        calls = {"read_text": 0}
        base = Path.read_text

        def counting_read_text(self_, *a, **kw):  # noqa: ANN001 - 替身
            calls["read_text"] += 1
            return base(self_, *a, **kw)

        Path.read_text = counting_read_text  # type: ignore[assignment]
        try:
            kb.index_status()
        finally:
            Path.read_text = base  # type: ignore[assignment]

        assert calls["read_text"] == 0, (
            f"探测读进了 {calls['read_text']} 个文件的内容 —— mtime 就够了，"
            "读内容会把每次检索都变成一次全库扫描"
        )

    def test_missing_note_is_reported_by_stats(self, tmp_path):
        """``stats()`` 的陈旧计数必须和自愈判定一致（同一份 mtime 容差）。"""
        kb = _make_kb(tmp_path, {"notes/a.md": NOTE_A})
        _install(kb, _SemanticIndex())
        assert kb.stats()["index_missing"] == 1
        kb.search("alpha", top_k=10)  # 触发自愈
        stats = kb.stats()
        assert stats["index_missing"] == 0 and stats["index_stale"] == 0


# ---------------------------------------------------------------------------
# P0-2
# ---------------------------------------------------------------------------

class _FakeKB:
    """按调用方给的命中列表回应 ``search``，并记录收到的 top_k。"""

    def __init__(self, hits):
        self._hits = hits
        self.requested_top_k: list[int] = []

    def search(self, query, top_k=10, section=""):
        self.requested_top_k.append(top_k)
        return self._hits[:top_k]


def _hit(path, *, kw, fused=0.0, sem=0.0, title="T"):
    return {"path": path, "title": title or path, "score": fused,
            "keyword_score": kw, "semantic_score": sem,
            "kind": "keyword" if kw else "semantic", "section": "notes"}


def _engine(tmp_path: Path, hits) -> RecallEngine:
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    return RecallEngine(cfg, kb=_FakeKB(hits))


class TestKeywordRecallChannel:
    """``kw >= recall_min_kw_score`` OR ``fused >= recall_min_score``。"""

    def test_pure_keyword_hit_is_admitted(self, tmp_path):
        """满格关键词命中（融合分被 0.4 权重压到 0.4）必须能通过 0.45 的门。"""
        engine = _engine(tmp_path, [
            _hit("notes/target.md", kw=1.0, fused=0.4, title="目标笔记"),
        ])
        engine._config.kb.recall_min_score = 0.45
        engine._config.kb.recall_min_kw_score = 1.0

        out = engine._search_kb("q")
        assert [r.source for r in out] == ["notes/target.md"], (
            "纯关键词命中融合分上限就是 0.4（0.4*kw_norm），只按融合分判定的话"
            "它永远够不到 0.45 —— 关键词通道等于被结构性关死"
        )
        assert out[0].metadata["admit_via"] == "keyword"

    def test_keyword_hit_buried_deep_in_rank_still_surfaces(self, tmp_path):
        """按融合分排在很后面的强命中，也必须能进最终的 top-N。

        只放开阈值、不放开候选池是不够的：生产链路先按融合分截断到
        ``recall_max_notes``，0.4 封顶的命中连候选都进不去（实测排第 27 位）。
        """
        hits = [_hit(f"notes/filler{i}.md", kw=0.0, fused=0.9 - i * 0.001)
                for i in range(25)]
        hits.append(_hit("notes/target.md", kw=1.0, fused=0.4, title="目标笔记"))

        engine = _engine(tmp_path, hits)
        engine._config.kb.recall_min_score = 0.45
        engine._config.kb.recall_min_kw_score = 1.0
        engine._config.kb.recall_max_notes = 3

        out = engine._search_kb("q")
        sources = [r.source for r in out]
        assert "notes/target.md" in sources, (
            "笔记被 top_k 截断掉了：候选池没放开，或者放行后没有按放行依据重排"
        )
        assert len(out) <= 3, "输出上限仍然必须是 recall_max_notes"
        assert engine._kb.requested_top_k[0] > 3, "候选池必须大于输出上限"

    def test_low_keyword_noise_is_still_rejected(self, tmp_path):
        """噪音守门：关键词分不够就是不放行，不能因为开了门槛就全进。"""
        engine = _engine(tmp_path, [
            _hit("notes/noise.md", kw=0.19, fused=0.41, title="无关笔记"),
        ])
        engine._config.kb.recall_min_score = 0.45
        engine._config.kb.recall_min_kw_score = 1.0
        assert engine._search_kb("q") == []

    def test_default_zero_changes_nothing(self, tmp_path):
        """两个门槛都 =0（代码默认）→ 通道关闭，行为与旧版完全一致。"""
        engine = _engine(tmp_path, [
            _hit("notes/target.md", kw=1.0, fused=0.9, title="目标笔记"),
        ])
        assert engine._config.kb.recall_min_kw_score == 0.0
        assert engine._search_kb("q") == []

    def test_fused_channel_still_works_alone(self, tmp_path):
        """只配 recall_min_score（老配置）时，语义/融合通道不受影响。"""
        engine = _engine(tmp_path, [
            _hit("notes/sem.md", kw=0.0, fused=0.9, sem=0.9, title="语义命中"),
            _hit("notes/low.md", kw=0.0, fused=0.30, sem=0.5, title="低分"),
        ])
        engine._config.kb.recall_min_score = 0.45
        engine._config.kb.recall_min_kw_score = 0.0

        out = engine._search_kb("q")
        assert [r.source for r in out] == ["notes/sem.md"]
        assert out[0].metadata["admit_via"] == "fused"

    @pytest.mark.parametrize("min_kw", [0.5, 0.6, 0.8, 1.0])
    def test_thresholds_in_the_calibrated_band_all_admit_the_title_hit(
            self, tmp_path, min_kw):
        """标定可用区间的任意阈值都要放行满格命中（见 reports 的敏感性分析）。"""
        engine = _engine(tmp_path, [
            _hit("notes/target.md", kw=1.0, fused=0.4, title="目标笔记"),
        ])
        engine._config.kb.recall_min_score = 0.45
        engine._config.kb.recall_min_kw_score = min_kw
        assert len(engine._search_kb("q")) == 1
