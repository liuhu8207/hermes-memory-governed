# -*- coding: utf-8 -*-
"""``_relocate`` / ``approve`` 在旧笔记删不掉时必须**报失败**（2026-09-17）。

**缺陷**：``_relocate`` 把新文件写进 notes，然后

    try:
        note.path.unlink()
    except OSError as e:
        logger.warning(...)          # 只 warning
    ...
    return {"ok": True, ...}         # 然后照常报成功

Windows 上文件被别的线程/进程持有句柄时 ``unlink`` 抛 ``PermissionError``
（errno 13 / **WinError 32**）。此时 ``approve`` 返回 ``ok=True``、旧文件却留在
inbox —— 笔记**同时存在于 inbox 和 notes**。这是真实产品缺陷：重复数据，而调用
方完全不知情（`test_phase3.py::TestReviewGate::test_approve_clears_pending_list`
偶发失败的根因）。

**修法**：``unlink`` 经 ``_unlink_with_retry`` 退避重试（WinError 32 绝大多数是
瞬时占用）；重试用尽仍失败就 ``ok=False`` 并带原因，同时**回滚**刚写的副本，把
现场还原成「只存在于 inbox」。回滚也失败时明确说明「两处都有一份，需人工处理」——
调用方要能区分「什么都没发生」和「现在库里有两份」。

本文件的两条铁律：
1. 没做完的移动**绝不能**报 ``ok=True``；
2. 失败后至少 inbox 里那份还在，``list_review()`` 必须**看得到它** ——
   「批准成功」和「还在待审」不能同时成立。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase, _unlink_with_retry

TITLE = "待审"
BODY = "待审正文 le-canary"


@pytest.fixture
def kb(tmp_path):
    """与 test_phase3.py 同一个 fixture 形状（backend=none）。"""
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    k = KnowledgeBase(cfg)
    k.ensure()
    return k


def _add_pending(kb: KnowledgeBase) -> Path:
    """加一篇 inbox 待审笔记，返回它在磁盘上的路径。"""
    r = kb.add(title=TITLE, body=BODY, confidence=0.3)
    assert r["ok"] is True, r
    pending = kb.list_review()
    assert len(pending) == 1, "前置条件：inbox 里应当正好有一篇待审"
    return kb._vault / pending[0]["path"]


def _notes_copy(kb: KnowledgeBase) -> Path:
    moving = sorted((kb._vault / "notes").glob("*.md"))
    assert len(moving) <= 1, "notes 里不该有同名副本摞在一起"
    return moving[0] if moving else None


# ---------------------------------------------------------------------------
# _unlink_with_retry
# ---------------------------------------------------------------------------

class TestUnlinkWithRetry:
    """重试助手的契约：成功返回 None，失败返回**带原因**的字符串。"""

    def test_successful_unlink_returns_none(self, tmp_path):
        p = tmp_path / "x.md"
        p.write_text("x", encoding="utf-8")
        assert _unlink_with_retry(p) is None
        assert not p.exists()

    def test_already_gone_counts_as_success(self, tmp_path):
        """早就没了 = 目的已达成。报成失败会让调用方拒绝一次成功的移动。"""
        assert _unlink_with_retry(tmp_path / "never-existed.md") is None

    def test_persistent_oserror_returns_a_reason_with_errno(self, tmp_path,
                                                            monkeypatch):
        calls = {"n": 0}
        real = Path.unlink

        def boom(self, *a, **kw):
            calls["n"] += 1
            raise OSError(13, "The process cannot access the file because it is "
                              "being used by another process")

        monkeypatch.setattr(Path, "unlink", boom)
        reason = _unlink_with_retry(tmp_path / "locked.md", base_delay=0.001)

        assert reason is not None, "失败必须返回原因，不能返回 None"
        assert "errno=13" in reason, reason
        assert calls["n"] >= 2, "应当重试过，而不是一次就放弃"
        monkeypatch.setattr(Path, "unlink", real)

    def test_transient_lock_is_retried_until_it_clears(self, tmp_path,
                                                       monkeypatch):
        """占用在第 2 次之后解开 → 整体必须成功（这正是 WinError 32 的常见形态）。"""
        p = tmp_path / "locked.md"
        p.write_text("x", encoding="utf-8")
        state = {"n": 0}
        real = Path.unlink

        def flaky(self, *a, **kw):
            state["n"] += 1
            if state["n"] < 3:
                raise OSError(13, "locked")
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "unlink", flaky)
        try:
            assert _unlink_with_retry(p, base_delay=0.001) is None
        finally:
            monkeypatch.setattr(Path, "unlink", real)
        assert not p.exists()


# ---------------------------------------------------------------------------
# approve / _relocate 的真实 Windows 文件锁
# ---------------------------------------------------------------------------

class TestApproveDoesNotPretendSuccess:
    """用**真实**文件句柄制造 WinError 32，不用替身。"""

    def test_happy_path_still_moves_the_note(self, kb):
        """没被占用时必须照常成功 —— 修复不能把正常路径弄坏。"""
        src = _add_pending(kb)
        r = kb.approve(TITLE)
        assert r["ok"] is True, r
        assert not src.exists(), "旧文件必须被删掉"
        assert kb.list_review() == [], "批准之后待审列表必须清空"

    @pytest.mark.skipif(
        os.name != "nt",
        reason="依赖 Windows 的强制文件锁：句柄未释放时 unlink 抛 WinError 32。"
               "POSIX 上删除被打开的文件是**成功**的，所以这个场景在那里不存在 —— "
               "契约本身由下方用 monkeypatch 的确定性用例在所有平台覆盖")
    def test_locked_original_reports_failure_not_ok(self, kb):
        """句柄一直不放：必须报失败，且还清清楚楚留在待审列表里。"""
        src = _add_pending(kb)
        handle = open(src, "r+b")   # 持有句柄 → unlink 抛 WinError 32
        try:
            r = kb.approve(TITLE)
        finally:
            handle.close()

        assert r["ok"] is False, (
            "旧笔记删不掉却返回 ok=True —— approve 谎报成功，"
            "笔记会同时留在 inbox 和 notes（重复数据）"
        )
        assert r["rolled_back"] is True, "应当回滚刚写的副本，还原成只存在于 inbox"
        assert r["partial"] is False
        assert src.exists(), "旧笔记确实还在 inbox，这正是不能报成功的原因"
        assert _notes_copy(kb) is None, "回滚后 notes 里不该留下副本"
        pending = kb.list_review()
        assert len(pending) == 1, (
            "批准失败后待审列表必须仍然看得到它：%s" % pending)

    def test_a_permanent_unlink_failure_is_reported_deterministically(
            self, kb, monkeypatch):
        """同一个契约，**不依赖文件锁语义**，因此每个平台都验证得到。

        上面那条用真实句柄更像线上，但只在 Windows 成立。这条把 unlink 对**这一个
        文件**钉成永久失败，于是「旧笔记删不掉就必须报失败、并回滚刚写的副本」
        在 POSIX 上也有人看着 —— 那正是它要挡的重复数据（同时留在 inbox 和 notes）。
        """
        src = _add_pending(kb)
        real = Path.unlink

        def locked_for_src_only(self, *a, **kw):
            if self == src:
                raise PermissionError(13, "being used by another process")
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "unlink", locked_for_src_only)
        try:
            r = kb.approve(TITLE)
        finally:
            monkeypatch.setattr(Path, "unlink", real)

        assert r["ok"] is False, (
            "旧笔记删不掉却返回 ok=True —— approve 谎报成功，"
            "笔记会同时留在 inbox 和 notes（重复数据）")
        assert r["rolled_back"] is True, "应当回滚刚写的副本，还原成只存在于 inbox"
        assert r["partial"] is False
        assert src.exists(), "旧笔记确实还在 inbox，这正是不能报成功的原因"
        assert _notes_copy(kb) is None, "回滚后 notes 里不该留下副本"
        assert len(kb.list_review()) == 1, (
            "批准失败后待审列表必须仍然看得到它：%s" % kb.list_review())

    def test_lock_clearing_mid_retry_lets_the_move_finish(self, kb, monkeypatch):
        """占用在第 3 次才解开 → ``approve`` 整体必须成功。

        这条是**确定性**的守卫（上面那条用真实句柄，更像线上但受调度时序影响，
        不能单独拿来证明「重试真的会救回来」）：限定只对这个文件前两次失败，
        其余路径照常走真实实现。
        """
        src = _add_pending(kb)
        state = {"n": 0}
        real = Path.unlink

        def flaky(self, *a, **kw):
            if self == src:
                state["n"] += 1
                if state["n"] < 3:
                    raise PermissionError(13, "being used by another process")
            return real(self, *a, **kw)

        monkeypatch.setattr(Path, "unlink", flaky)
        r = kb.approve(TITLE)

        assert state["n"] >= 2, "应当真的重试过，而不是一次就放弃"
        assert r["ok"] is True, "瞬时占用必须被重试救回来：%s" % r
        assert not src.exists(), "旧文件最终必须被删掉"
        assert kb.list_review() == [], "移动成功后待审列表必须清空"
        assert _notes_copy(kb) is not None, "新位置必须真的有这份笔记"

    @pytest.mark.skipif(
        os.name != "nt",
        reason="同上：靠真实句柄制造 WinError 32，POSIX 无此语义。"
               "重试的确定性与契约覆盖由 monkeypatch 版用例承担")
    def test_transient_lock_is_retried_and_the_move_succeeds(self, kb):
        """占位句柄很快就放开 → 重试应当救回这次移动。

        这是真实工况：杀软扫盘、索引器、并发读都是**短时**占用。没有重试就会把
        这些全判成失败，误报率会非常高。
        """
        src = _add_pending(kb)
        handle = open(src, "r+b")
        threading.Timer(0.03, handle.close).start()
        try:
            r = kb.approve(TITLE)
        finally:
            try:
                handle.close()
            except OSError:
                pass

        assert r["ok"] is True, "瞬时占用应当被重试救回来：%s" % r
        assert not src.exists(), "旧文件最终必须被删掉"
        assert kb.list_review() == [], "移动成功后待审列表必须清空"


# ---------------------------------------------------------------------------
# 回滚也失败 —— 两处都有一份
# ---------------------------------------------------------------------------

class TestRollbackFailureIsStated:
    """回滚失败时必须明说「现在两处都有一份」，不能只给一个干巴巴的 False。"""

    def test_both_copies_are_reported(self, kb, monkeypatch):
        def always_locked(self, *a, **kw):
            raise OSError(13, "being used by another process")

        monkeypatch.setattr(Path, "unlink", always_locked)
        src = _add_pending(kb)
        r = kb.approve(TITLE)

        assert r["ok"] is False
        assert r["rolled_back"] is False, "回滚确实失败了"
        assert r["partial"] is True, "必须标出半成品状态"
        assert src.exists()
        assert _notes_copy(kb) is not None, "此刻库里确实有两份"
        assert "BOTH" in r["error"], (
            "错误信息必须点明『两处都有一份』，调用方才知道要人工去重：\n%s"
            % r["error"])
