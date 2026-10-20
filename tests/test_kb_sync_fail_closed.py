# -*- coding: utf-8 -*-
"""KB 索引同步必须 **fail-closed**：vault 读失败不得删空索引。

背景（生产事故，独立评估者隔离复现）
------------------------------------
``sync_index(_vault_mtimes={}, _note_paths={})``（= **vault 遍历失败/未就绪**）
在有 52 行的索引上返回 ``{'ok': True, 'removed': 5, 'errors': 0}`` ——
**整表被删光，而且被报告成成功**。随后 ``_ensure_fresh_index`` 仍报
``fresh: True, synced: True``，**状态在说谎**。

生产侧观测到过索引行数 52→13→1→2→0；下一次 ``kb-search`` 触发重建后行数回到
52（能自愈），但每次删空+重建 ≈ +100 个版本，与"版本膨胀"问题叠加。

三条根因，本文件逐条钉住
------------------------
1. **"vault 真的空" 与 "vault 读失败" 在调用方是同一现象** ⇒ 扫描为空时
   **一行都不许删**，必须显式 ``allow_purge=True`` 才放行。
2. **部分扫描失败**（只扫到 3 篇却按 52 篇处理）同样会大面积误删 ⇒ 扫到的
   少于预期、或 orphan 的**文件其实还在磁盘上**，都要 fail-closed。
3. **写失败会毁掉旧数据**：``KBIndex.upsert`` 旧实现是"先删后加"，``add``
   失败时旧行已被删掉，而失败只被 log 掉、**不抛** ⇒ 调用方以为写成功了。
   现在改成"先写成功、再替换"（``update`` 原地改写 / ``add`` 新增），写失败
   时旧行原样保留。

用**真实 LanceDB**（不用假索引）：断言的是"索引里到底还剩几行"，假对象证明
不了删空。另配一个假嵌入服务（确定性向量），避免联网。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugin.memory_governed import _diag
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KBIndex, KnowledgeBase

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

#: 生产索引行数（复现脚本用的规模）。
_N = 52


class _FakeService:
    """可用但极简的嵌入服务：确定性向量，不联网。"""

    available = True
    dim = 4
    model_name = "fake"

    def embed_one(self, text: str):
        return [float(len(text) % 7), 0.2, 0.3, 0.4]


class _DeadService:
    """嵌入后端**挂了**：``embed_one`` 返回 ``None``（不抛、不写、不报错）。

    这是独立评估者实测到的那个洞的最小复现：旧实现里 ``upsert`` 拿到 None 就
    直接 return —— 既不 add、也**不抛** ⇒ ``sync_index`` 的 except 根本不触发
    ⇒ ``errors: 0``，而索引行数已经从 4 掉到 0。

    注意 ``available`` 仍是 True：后端"在"，只是每次调用都失败 —— 这正是
    "embedding 挂掉但索引后端仍可用"的生产形态。
    """

    available = True
    dim = 4
    model_name = "dead"

    def embed_one(self, text: str):
        return None


class _SwitchableService:
    """可切换的嵌入服务：先用"好的"把索引行造出来，再切到"挂了"。

    生产形态是"索引行早已存在，embedding **之后**挂了" —— 所以测试前置必须
    能用好的服务把行建好，再让后续写入全部失败。
    """

    available = True
    dim = 4
    model_name = "switch"

    def __init__(self):
        self.dead = False

    def embed_one(self, text: str):
        if self.dead:
            return None
        return [float(len(text) % 7), 0.2, 0.3, 0.4]


def _make_kb(tmp_path, monkeypatch, *, vault_notes: int = 0,
             index_rows: int = 0, service=None,
             dead_after_setup: bool = False,
             index_paths: "list[str] | None" = None):
    """建 KB：真实 vault 目录 + 真实 LanceDB 索引（假嵌入服务）。

    Args:
        vault_notes: vault 里放几篇笔记。
        index_rows: 索引里预置几行（路径为 ``notes/nNN.md``）。
        service: 嵌入服务**实例**（默认 ``_FakeService()``）。
        dead_after_setup: 造完索引行之后把服务切到"挂了"（模拟 embedding
            在索引建好之后才故障）。
        index_paths: 显式指定索引行的路径（默认按 ``notes/nNN.md`` 生成）。
            用来造"路径集变化"——索引里的旧路径在磁盘上**已经不存在**。

    Returns:
        ``(kb, idx)``；``idx`` 就是 KB 手里那个索引对象（可直接查行数）。
    """
    pytest.importorskip("lancedb")
    svc = service if service is not None else _SwitchableService()
    monkeypatch.setattr(
        "plugin.memory_governed._kb.EmbeddingService.get",
        staticmethod(lambda cfg: svc))

    wiki = tmp_path / "wiki"
    (wiki / "notes").mkdir(parents=True, exist_ok=True)
    for i in range(vault_notes):
        (wiki / "notes" / ("n%02d.md" % i)).write_text(
            "标题 %d\n\n正文 %d\n" % (i, i), encoding="utf-8")

    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(wiki)
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    kb = KnowledgeBase(cfg)

    idx = KBIndex(cfg, kb._index_db)
    assert idx.available, idx.last_error
    kb._index = idx
    paths = index_paths or ["notes/n%02d.md" % i for i in range(index_rows)]
    for i, rel in enumerate(paths):
        # 旧行：mtime 固定 1.0，与真实文件 mtime 不同 ⇒ 会被判 stale。
        # 这里不校验 upsert 的返回值（旧实现返回 None，新实现返回 bool）——
        # 断言的是**行数**，那才是"索引里还剩什么"的唯一事实。
        idx.upsert(rel, "old text %d" % i, mtime=1.0)
    assert idx.row_count() == len(paths), "测试前置：索引里应已有 N 行"
    if dead_after_setup:
        svc.dead = True
    return kb, idx


def _break_write_for(monkeypatch, idx, target: str) -> None:
    """让 ``target`` 这条路径的写入失败（模拟磁盘满），其余路径照常。"""
    real_update = idx._store.update

    def boom_update(where=None, values=None, **kwargs):
        if where and target in str(where):
            raise OSError("disk full")
        return real_update(where=where, values=values, **kwargs)

    def boom_add(data, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(idx._store, "update", boom_update)
    monkeypatch.setattr(idx._store, "add", boom_add)


# ---------------------------------------------------------------------------
# 1. 扫描为空 ⇒ 一行都不许删
# ---------------------------------------------------------------------------

def test_empty_vault_scan_never_purges_the_index(tmp_path, monkeypatch):
    """复现事故：52 行索引 + 空扫描 ⇒ 仍是 52 行，ok=False + reason，有计数。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=_N, index_rows=_N)
    assert idx.row_count() == _N

    _diag.reset()
    result = kb.sync_index(_vault_mtimes={}, _note_paths={})

    assert result["ok"] is False
    assert result["reason"] == "vault_scan_empty"
    assert result["removed"] == 0
    assert result["purged"] is False
    assert idx.row_count() == _N, "vault 扫描为空时绝不能删索引行"
    assert _diag.stats()["degraded"].get("kb_index::vault_scan_empty") == 1


def test_allow_purge_is_the_explicit_opt_in(tmp_path, monkeypatch):
    """``allow_purge=True`` 才允许删光（= 调用方确认"vault 就是空的"）。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=0, index_rows=10)
    assert idx.row_count() == 10

    # 默认：拒绝。
    refused = kb.sync_index(_vault_mtimes={}, _note_paths={})
    assert refused["ok"] is False
    assert refused["reason"] == "vault_scan_empty"
    assert idx.row_count() == 10

    # 显式确认：允许删。
    allowed = kb.sync_index(_vault_mtimes={}, _note_paths={}, allow_purge=True)
    assert allowed["ok"] is True
    assert allowed["removed"] == 10
    assert idx.row_count() == 0


# ---------------------------------------------------------------------------
# 2. 写失败 ⇒ 不删（且旧行必须还在）
# ---------------------------------------------------------------------------

def test_failed_write_keeps_the_old_row(tmp_path, monkeypatch):
    """upsert 失败 ⇒ 旧行原样保留（不再"先删后加"把数据毁掉），且计入 errors。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=2, index_rows=2)
    assert idx.row_count() == 2
    _break_write_for(monkeypatch, idx, "notes/n00.md")

    result = kb.sync_index()

    assert result["errors"] >= 1, "写失败必须计入 errors"
    assert result["ok"] is False, "写失败不能报 ok=True"
    mtimes = idx.path_mtimes()
    assert mtimes.get("notes/n00.md") == 1.0, \
        "写失败的路径，旧行必须原样保留（旧实现会先删后加 → 旧行消失）"
    assert idx.row_count() == 2, "写失败不得删掉任何索引行"


# ---------------------------------------------------------------------------
# 3. 状态不得说谎
# ---------------------------------------------------------------------------

def test_ensure_fresh_index_reports_the_refusal_honestly(tmp_path, monkeypatch):
    """同步被拒后**不得**报 ``fresh: True, synced: True``。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=0, index_rows=_N)

    status = kb._ensure_fresh_index()

    assert status["synced"] is False
    assert status["fresh"] is not True
    assert status["sync_reason"] == "vault_scan_empty"
    assert idx.row_count() == _N
    # 同一份结论也要能从 last_index_status 看到（CLI 读的是它）。
    assert kb.last_index_status["synced"] is False


def test_index_status_exposes_row_count_and_last_sync(tmp_path, monkeypatch):
    """状态里要能一眼看出**索引行数**与**上次同步是否成功**。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=3, index_rows=_N)
    kb.sync_index(_vault_mtimes={}, _note_paths={})

    status = kb.index_status()
    assert status["index_rows"] == _N, "索引行数必须独立可见"
    assert status["last_sync_ok"] is False
    assert status["last_sync_reason"] == "vault_scan_empty"


# ---------------------------------------------------------------------------
# 4. 健康路径回归
# ---------------------------------------------------------------------------

def test_healthy_vault_syncs_normally(tmp_path, monkeypatch):
    """vault 有 N 篇 ⇒ 正常同步，行数 = N，第二次同步是 no-op。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=5, index_rows=0)

    first = kb.sync_index()
    assert first["ok"] is True
    assert first["added"] == 5
    assert first["removed"] == 0
    assert first["errors"] == 0
    assert idx.row_count() == 5

    second = kb.sync_index()
    assert second["ok"] is True
    assert second["added"] == 0
    assert second["updated"] == 0
    assert second["removed"] == 0
    assert second["errors"] == 0
    assert idx.row_count() == 5


def test_a_genuinely_deleted_note_is_still_purged(tmp_path, monkeypatch):
    """真删掉的笔记**仍然要**从索引里删掉（fail-closed 不能把正常清理也堵死）。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=3, index_rows=3)
    # 删掉 vault 里的第 3 篇（文件真的不在了）→ 它才是合法 orphan。
    (Path(kb._vault) / "notes" / "n02.md").unlink()

    result = kb.sync_index()

    assert result["ok"] is True
    assert result["removed"] == 1
    assert idx.row_count() == 2
    assert "notes/n02.md" not in idx.path_mtimes()


def test_deleting_the_last_note_needs_the_explicit_opt_in(tmp_path, monkeypatch):
    """删掉**最后一篇** ⇒ 扫描变成空 ⇒ 与"vault 读失败"无法区分 ⇒ 拒绝删。

    这是本修复**有意收紧**的地方：旧实现会把"用户删光了"和"挂载掉了"当成同一
    件事处理，于是后者把 52 行索引一次删光。要清空必须显式 ``allow_purge=True``。
    """
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=1, index_rows=1)
    (Path(kb._vault) / "notes" / "n00.md").unlink()

    refused = kb.sync_index()
    assert refused["ok"] is False
    assert refused["reason"] == "vault_scan_empty"
    assert refused["removed"] == 0
    assert idx.row_count() == 1, "扫描为空时旧行必须留着"

    allowed = kb.sync_index(allow_purge=True)
    assert allowed["ok"] is True
    assert allowed["removed"] == 1
    assert idx.row_count() == 0


# ---------------------------------------------------------------------------
# 5. 部分扫描失败 ⇒ 同样 fail-closed
# ---------------------------------------------------------------------------

def test_partial_scan_is_refused_and_nothing_is_purged(tmp_path, monkeypatch):
    """只扫到 3 篇（vault 里有 52 篇）⇒ 拒绝删，52 行一个不少。

    49 个"孤儿"的文件**其实还在磁盘上** —— 扫描漏了它们，不是它们被删了。
    """
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=_N, index_rows=_N)
    paths = kb.scan_vault_paths()
    assert len(paths) == _N

    partial = dict(list(paths.items())[:3])
    result = kb.sync_index(
        _vault_mtimes={rel: 1.0 for rel in partial},
        _note_paths=partial,
    )

    assert result["ok"] is False
    assert result["reason"] == "vault_scan_incomplete"
    assert result["removed"] == 0
    assert idx.row_count() == _N, "部分扫描失败不得删掉任何索引行"


def test_expected_notes_catches_a_short_scan(tmp_path, monkeypatch):
    """调用方知道"vault 应该有 52 篇"时，扫到 3 篇 ⇒ 同样 fail-closed。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=3, index_rows=_N)
    paths = kb.scan_vault_paths()
    assert len(paths) == 3

    result = kb.sync_index(
        _vault_mtimes={rel: 1.0 for rel in paths},
        _note_paths=paths,
        expected_notes=_N,
    )

    assert result["ok"] is False
    assert result["reason"] == "vault_scan_incomplete"
    assert result["removed"] == 0
    assert idx.row_count() == _N


# ---------------------------------------------------------------------------
# 6. 规格修正：upsert 失败必须**可见**，否则修不干净
#    （旧实现里 embed_one 返 None → upsert 直接 return → errors: 0，
#      而行数已经从 4 掉到 0 —— sync_index 根本看不到失败。）
# ---------------------------------------------------------------------------

def test_upsert_failure_is_visible_to_the_caller(tmp_path, monkeypatch):
    """``upsert`` 的失败必须**暴露出来**（返回 False），不能被内部吞掉。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=1, index_rows=0,
                       service=_DeadService())

    assert idx.upsert("notes/n00.md", "正文", mtime=1.0) is False, \
        "嵌入不可用时 upsert 必须报失败，不能静默 return"


def test_embedding_down_yields_errors_and_keeps_every_row(tmp_path, monkeypatch):
    """embedding 不可用 ⇒ ``errors > 0`` 且**行数一条都不许减少**。"""
    kb, idx = _make_kb(tmp_path, monkeypatch, vault_notes=2, index_rows=2,
                       dead_after_setup=True)
    before = idx.row_count()
    assert before == 2

    result = kb.sync_index()

    assert result["errors"] > 0, "写入失败必须被 sync_index 看见（errors>0）"
    assert result["ok"] is False
    assert idx.row_count() == before, "写失败时既有行一条都不许消失"


def test_path_set_change_with_embedding_down_deletes_nothing(tmp_path, monkeypatch):
    """路径集变化 + embedding 挂掉 ⇒ **不删任何行**，且状态不报 healthy。

    独立评估者实测的组合：``{'ok': True, 'added': 4, 'removed': 4, 'errors': 0}``、
    行数 4→0。之所以比"扫描为空"更危险：扫描为空时后续检索还能靠重建自愈；
    而"删空 + embedding 同时坏"⇒ 重建路径本身也依赖 embedding ⇒ **索引长期为 0
    且无法自愈**，状态却还报健康。

    索引里除了 4 个"旧路径"（文件已不在 ⇒ 孤儿），还留着 4 行**仍在 vault 里**
    的笔记 —— 那几行就是**配对证据**（索引确实属于这个 vault），否则护栏会以
    "索引可能属于别的 vault"为由直接拒绝整轮同步，压根走不到写入路径。
    """
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=4,                                  # 当前 vault：4 篇
        index_paths=["notes/n%02d.md" % i for i in range(4)]   # 仍在 vault 里（证据）
                     + ["notes/old%d.md" % i for i in range(4)],  # 旧路径：孤儿
        dead_after_setup=True,
    )
    assert idx.row_count() == 8

    result = kb.sync_index()

    assert result["errors"] > 0, "embedding 挂掉必须报 errors>0"
    assert result["ok"] is False
    assert result["removed"] == 0, "写失败时不得删任何行"
    assert idx.row_count() == 8, "行数不得减少（否则重建也救不回来）"
    assert result.get("purge_skipped") == "sync_write_failed"


def test_state_does_not_claim_healthy_after_a_failed_sync(tmp_path, monkeypatch):
    """写失败后，``_ensure_fresh_index`` **不得**报 ``fresh: True, synced: True``。"""
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=4,
        index_paths=["notes/n%02d.md" % i for i in range(4)]
                     + ["notes/old%d.md" % i for i in range(4)],
        dead_after_setup=True,
    )

    status = kb._ensure_fresh_index()

    assert status["synced"] is False
    assert status["fresh"] is not True
    assert status["sync_errors"] > 0
    assert idx.row_count() == 8


def test_allow_purge_does_not_override_the_write_failure_protection(
        tmp_path, monkeypatch):
    """``allow_purge=True`` 是"确认 vault 为空"的开关，**不能**盖过写失败保护。"""
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=4,
        index_paths=["notes/old%d.md" % i for i in range(4)],
        dead_after_setup=True,
    )

    result = kb.sync_index(allow_purge=True)

    assert result["errors"] > 0
    assert result["removed"] == 0
    assert idx.row_count() == 4, "写失败时即使 allow_purge 也不许删行"


def test_one_unreadable_note_does_not_block_orphan_cleanup(tmp_path, monkeypatch):
    """**读**失败（per-note 坏文件）不该阻断孤儿清理；**写**失败才该。

    两者都是真错误、都要计入 ``errors``；但只有"向量写不进去"才说明这一轮的
    写入环境坏了 —— 用一篇读不出来的笔记挡住整个清理，会让一行陈旧数据
    永久留在索引里（检索还会把它当命中返回）。
    """
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=2,
        index_paths=["notes/n00.md", "notes/gone.md"],   # n00 还在（配对证据）、gone 是孤儿
    )
    assert idx.row_count() == 2

    real_read = Path.read_text

    def poisoned(self, *a, **kw):
        if self.name == "n01.md":
            raise PermissionError(13, "Permission denied")
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", poisoned)

    result = kb.sync_index()

    assert result["errors"] > 0, "读失败必须计入 errors"
    assert result.get("purge_skipped") is None, \
        "读失败不该阻断删除（那是写失败才该做的）"
    assert result["removed"] == 1, "合法 orphan 仍要清掉"
    paths_left = set(idx.path_mtimes())
    assert "notes/gone.md" not in paths_left, "合法 orphan 必须被清掉"
    assert "notes/n00.md" in paths_left, "写得进去的那篇必须在索引里"
    assert "notes/n01.md" not in paths_left, "读失败的那篇本来就没写进去"


# ---------------------------------------------------------------------------
# 7. 索引↔vault 配对：索引和 vault 来自两个世界时，**一行都不能删**
#    （生产事故实测形状：索引 52 行真实笔记 + 当前 vault 只有 1 篇测试夹具
#      ⇒ 旧实现把 52 行全判成 orphan 删光，而且报 ok:true）
# ---------------------------------------------------------------------------

def test_index_belonging_to_another_vault_is_never_purged(tmp_path, monkeypatch):
    """索引里的行在当前 vault 里**一个都不存在** ⇒ 拒绝删（配对错位，不是清理）。"""
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=1,                                   # 当前 vault 只有 1 篇
        index_paths=["notes/prod%d.md" % i for i in range(52)],  # 别的 vault 的 52 行
    )
    assert idx.row_count() == 52

    result = kb.sync_index()

    assert result["ok"] is False
    assert result["reason"] in ("index_vault_unknown", "index_vault_mismatch")
    assert result["removed"] == 0, "配对错位时一行都不许删"
    assert idx.row_count() == 52, "52 行一个都不许少"


def test_a_recorded_vault_mismatch_is_reported(tmp_path, monkeypatch):
    """sidecar 记录的 vault 与当前 vault 不一致 ⇒ 报 ``index_vault_mismatch``。"""
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=1,
        index_paths=["notes/prod%d.md" % i for i in range(4)],
    )
    kb._index_vault_marker().write_text(
        json.dumps({"vault_root": str(Path(kb._vault).parent / "另一个vault")},
                   ensure_ascii=False),
        encoding="utf-8")

    result = kb.sync_index()

    assert result["ok"] is False
    assert result["reason"] == "index_vault_mismatch"
    assert result["removed"] == 0
    assert idx.row_count() == 4


def test_a_purge_that_would_empty_the_index_is_refused(tmp_path, monkeypatch):
    """这次 purge 会把索引**清空** ⇒ 拒绝（清空是"重置"不是"清理"）。

    这与"索引与 vault 毫无关系"是**同一个条件**（可证：存在配对证据 ⇒ 至少有一行
    不是 orphan ⇒ purge 不可能清空索引），所以报的是**病因**
    ``index_vault_unknown``，而不是"症状"（原 ``purge_to_zero``，已删）。
    本用例同时**钉住** ``index_vault_unknown`` 是**可达的**（若死规则还在，
    这里报出来的就会是 ``purge_to_zero``）。
    """
    kb, idx = _make_kb(
        tmp_path, monkeypatch,
        vault_notes=1,
        index_paths=["notes/gone.md"],
    )
    # sidecar 记的是**当前** vault ⇒ 配对没错 ⇒ 拦下它的只能是"索引里的行在
    # 当前 vault 里一个都不存在"这一条。
    kb._index_vault_marker().write_text(
        json.dumps({"vault_root": str(Path(kb._vault).resolve())},
                   ensure_ascii=False),
        encoding="utf-8")

    refused = kb.sync_index()
    assert refused["ok"] is False
    assert refused["reason"] == "index_vault_unknown"
    assert refused["removed"] == 0
    assert idx.row_count() == 1
