# -*- coding: utf-8 -*-
"""``project`` 维度对插件读路径生效的回归测试（P0，2026-09-17）。

**缺陷**：``project`` 维度只装在 CLI 的一条读路径上（``memory_cli.py`` 的
``recall_l2``），而 Hermes 插件走 ``_recall.py`` —— 全文件里
``project`` 出现 0 次。同一份存储、两个入口、两套答案：CLI 会按项目收窄，
插件却把所有项目的事实一锅端注入上下文。

**修复**：给 L2 检索加可选 ``project`` 作用域（向量与文本两条路径都走同一套
行级判定），作用域取值来自 ``config.recall.project_scope`` 三态：

* ``None`` —— 不过滤（向后兼容，行为与修复前一致）
* ``""`` —— 只看全局（``project IS NULL``）
* ``"名字"`` —— 该项目 **+** 全局

行级语义刻意与 CLI 一致（``if project and pj and pj != project: continue``）：
保留本项目 + 全局。全库事实（"SecretStore 跑在 NAS 上"）在任何项目下问都成立，
**别的项目**的事实才是必须挡住的。

本文件的行级替身（``_FakeL2Store``）在 ``where(..., prefilter=True)`` 上做真过滤，
与生产 LanceDB 的「先过滤再取 top-N」语义一致。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig, load_governed_config
from plugin.memory_governed._recall import (
    RecallEngine,
    l2_project_allows,
    l2_project_where,
)


# ---------------------------------------------------------------------------
# 行级语义（纯函数，三态 × 三类行）
# ---------------------------------------------------------------------------

class TestRowLevelFilter:
    """``l2_project_allows`` 必须与 CLI 的行级判定逐字对齐。"""

    @pytest.mark.parametrize("row_project", ["A", "B", "", None])
    def test_scope_none_allows_everything(self, row_project):
        """``None`` = 不过滤 -> 任何行都放行（旧行为不变）。"""
        assert l2_project_allows(None, row_project) is True

    def test_scope_empty_allows_only_globals(self):
        """``""`` = 只看全局：只有 ``project IS NULL/''`` 的行放行。"""
        assert l2_project_allows("", None) is True
        assert l2_project_allows("", "") is True
        assert l2_project_allows("", "A") is False
        assert l2_project_allows("", "B") is False

    def test_scope_name_keeps_own_project_plus_globals(self):
        """``"A"``：A 项目 + 全局放行，B 项目挡住。"""
        assert l2_project_allows("A", None) is True
        assert l2_project_allows("A", "") is True
        assert l2_project_allows("A", "A") is True
        assert l2_project_allows("A", "B") is False

    def test_missing_project_field_is_treated_as_global(self):
        """行里没有 ``project`` 字段（旧 schema）→ 按全局处理，不丢行。"""
        assert l2_project_allows("A", None) is True
        assert l2_project_allows("", None) is True


class TestWherePredicate:
    """``l2_project_where`` 生成的 SQL 谓词必须与行级判定等价。"""

    def test_empty_scope_is_globals_only(self):
        assert l2_project_where("") == "project IS NULL"

    def test_named_scope_includes_globals(self):
        assert l2_project_where("A") == "project IS NULL OR project = 'A'"

    def test_single_quote_in_scope_is_escaped(self):
        """目录名带引号时不能把谓词拼坏（SQL 字面量转义 ``'`` -> ``''``）。"""
        assert l2_project_where("O'Brien") == (
            "project IS NULL OR project = 'O''Brien'"
        )


# ---------------------------------------------------------------------------
# L2 检索替身：真实行级过滤（向量 where / 文本扫描两条路径共用）
# ---------------------------------------------------------------------------

#: 三类行：项目 A / 项目 B / 全局（NULL）。query "alpha" 三类都含。
ROWS: List[Dict[str, Any]] = [
    {"content": "alpha from project A", "project": "A", "category": "c",
     "source_rowid": 11, "_distance": 0.0},
    {"content": "alpha from project B", "project": "B", "category": "c",
     "source_rowid": 22, "_distance": 0.1},
    {"content": "alpha global knowledge", "project": None, "category": "c",
     "source_rowid": 33, "_distance": 0.2},
]


def _eval_where(predicate: str, row: Dict[str, Any]) -> bool:
    """只认代码实际产出的两种谓词形态，用于在替身里复算 LanceDB 的过滤。"""
    pj = row.get("project")
    if predicate == "project IS NULL":
        return pj is None
    m = re.fullmatch(r"project IS NULL OR project = '(.*)'", predicate)
    if m:
        literal = m.group(1).replace("''", "'")
        return pj is None or pj == literal
    raise AssertionError(f"unexpected predicate: {predicate!r}")


class _FakeSearch:
    """``table.search(vec)`` 返回的链式查询对象。"""

    def __init__(self, rows: List[Dict[str, Any]], *, allow_prefilter: bool):
        self._rows = rows
        self._allow_prefilter = allow_prefilter
        self._predicate: Optional[str] = None
        self._limit: Optional[int] = None
        self.predicate_used: Optional[str] = None
        self.prefilter_flag: Optional[bool] = None

    def metric(self, _metric: str) -> "_FakeSearch":
        return self

    def where(self, predicate: str, prefilter: bool = False) -> "_FakeSearch":
        if not self._allow_prefilter:
            # 模拟旧版 LanceDB：没有 prefilter 参数 -> 抛异常，触发降级路径
            raise TypeError("where() got an unexpected keyword argument 'prefilter'")
        self._predicate = predicate
        self.predicate_used = predicate
        self.prefilter_flag = prefilter
        return self

    def limit(self, n: int) -> "_FakeSearch":
        self._limit = int(n)
        return self

    def to_list(self) -> List[Dict[str, Any]]:
        rows = self._rows
        if self._predicate is not None:
            rows = [r for r in rows if _eval_where(self._predicate, r)]
        if self._limit is not None:
            rows = rows[:self._limit]
        return list(rows)


class _FakeColumn:
    def __init__(self, values: List[Any]):
        self._values = values

    def to_pylist(self) -> List[Any]:
        return list(self._values)


class _FakeArrow:
    """``table.to_arrow()`` 的最小接口，供文本兜底路径扫描。"""

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self.num_rows = len(rows)
        self.column_names = list(rows[0].keys()) if rows else []

    def column(self, name: str) -> _FakeColumn:
        return _FakeColumn([r.get(name) for r in self._rows])


class _FakeL2Store:
    """LanceDB ``memories`` 表的替身：向量搜索 + 全表扫描两条路径。"""

    def __init__(self, rows: List[Dict[str, Any]], *, allow_prefilter: bool = True):
        self._rows = rows
        self._allow_prefilter = allow_prefilter
        self.last_search: Optional[_FakeSearch] = None

    def search(self, _vec, vector_column_name=None) -> _FakeSearch:
        # 桩必须跟真 LanceDB 的签名一致：生产代码会显式传 vector_column_name
        # 来指定向量列（不传时真后端会抛错，而那个错会被降级逻辑吞掉成空结果）。
        self.last_search = _FakeSearch(self._rows, allow_prefilter=self._allow_prefilter)
        return self.last_search

    def to_arrow(self) -> _FakeArrow:
        return _FakeArrow(self._rows)


def _engine(rows: List[Dict[str, Any]], *, embed: bool = True,
            allow_prefilter: bool = True) -> RecallEngine:
    """装了 L2 替身的引擎（``_l2_init_attempted`` 置位，跳过真实加载）。"""
    cfg = GovernedMemoryConfig()
    cfg.vector.backend = "none"
    cfg.recall.l2_max_results = 10
    engine = RecallEngine(cfg)
    engine._l2_init_attempted = True
    engine._l2_store = _FakeL2Store(rows, allow_prefilter=allow_prefilter)
    engine._embed_fn = (lambda _q: [0.0, 0.0]) if embed else None
    return engine


def _sources(results) -> List[str]:
    return [r.metadata.get("source_rowid") for r in results]


# ---------------------------------------------------------------------------
# 向量路径：三类行 × 三种作用域
# ---------------------------------------------------------------------------

class TestL2VectorScope:
    def test_scope_none_returns_all_three_rows(self):
        engine = _engine(ROWS)
        out = engine._search_l2_vector("alpha", None)
        assert _sources(out) == [11, 22, 33]
        # 未过滤时不应调用 where
        assert engine._l2_store.last_search.predicate_used is None

    def test_scope_empty_returns_only_global(self):
        engine = _engine(ROWS)
        out = engine._search_l2_vector("alpha", "")
        assert _sources(out) == [33], "只看全局时，A/B 两个项目的行都必须被挡住"

    def test_scope_a_returns_own_project_plus_global(self):
        engine = _engine(ROWS)
        out = engine._search_l2_vector("alpha", "A")
        assert _sources(out) == [11, 33], "A 作用域应保留 A 项目 + 全局，且不含 B"
        assert 22 not in _sources(out)

    def test_scope_b_returns_own_project_plus_global(self):
        engine = _engine(ROWS)
        out = engine._search_l2_vector("alpha", "B")
        assert _sources(out) == [22, 33]
        assert 11 not in _sources(out)

    def test_where_receives_prefilter_true(self):
        """必须用 prefilter：在取 top-N **之前**过滤，才能与 CLI 全表扫描等价。"""
        engine = _engine(ROWS)
        engine._search_l2_vector("alpha", "A")
        assert engine._l2_store.last_search.prefilter_flag is True
        assert engine._l2_store.last_search.predicate_used == (
            "project IS NULL OR project = 'A'"
        )


class TestL2VectorScopeFallback:
    """旧版 LanceDB 不支持 ``prefilter`` 时的降级路径仍要正确过滤。"""

    def test_fallback_still_scopes_correctly(self):
        engine = _engine(ROWS, allow_prefilter=False)
        out = engine._search_l2_vector("alpha", "A")
        assert _sources(out) == [11, 33], "降级路径漏过滤 —— 会向别的项目泄漏事实"

    def test_fallback_scope_empty_returns_only_global(self):
        engine = _engine(ROWS, allow_prefilter=False)
        out = engine._search_l2_vector("alpha", "")
        assert _sources(out) == [33]

    def test_fallback_widens_candidate_pool(self):
        """降级路径必须先多取候选再过滤，否则别的项目的行会占满名额。"""
        from plugin.memory_governed._recall import _L2_SCOPE_OVERFETCH

        many = [{"content": "alpha", "project": "B", "source_rowid": 900 + i,
                 "_distance": 0.01 * i} for i in range(5)]
        many.append({"content": "alpha", "project": "A", "source_rowid": 1,
                     "_distance": 0.9})
        engine = _engine(many, allow_prefilter=False)
        engine._config.recall.l2_max_results = 10
        out = engine._search_l2_vector("alpha", "A")
        assert _sources(out) == [1], "降级时没多取候选 —— A 项目的行会被 B 挤掉"
        assert _L2_SCOPE_OVERFETCH >= 1


# ---------------------------------------------------------------------------
# 文本兜底路径：同一个作用域语义
# ---------------------------------------------------------------------------

class TestL2TextScope:
    def test_scope_none_returns_all_rows(self):
        engine = _engine(ROWS, embed=False)
        out = engine._search_l2_text("alpha", None)
        assert sorted(_sources(out)) == [11, 22, 33]

    def test_scope_empty_returns_only_global(self):
        engine = _engine(ROWS, embed=False)
        out = engine._search_l2_text("alpha", "")
        assert _sources(out) == [33]

    def test_scope_a_returns_own_project_plus_global(self):
        engine = _engine(ROWS, embed=False)
        out = engine._search_l2_text("alpha", "A")
        assert sorted(_sources(out)) == [11, 33]

    def test_non_matching_content_is_filtered_by_query(self):
        """作用域之外仍要按查询命中过滤（别顺手把过滤写成了放行一切）。"""
        engine = _engine(ROWS, embed=False)
        assert engine._search_l2_text("zzz-not-present", "A") == []


# ---------------------------------------------------------------------------
# 可观测性 —— 「被过滤过」这件事必须看得见，不能静默
# ---------------------------------------------------------------------------

class TestScopeObservability:
    def test_last_l2_scope_records_effective_scope(self):
        engine = _engine(ROWS)
        engine._search_l2("alpha", "A")
        assert engine.last_l2_scope == "A"

    def test_last_l2_scope_is_none_when_unfiltered(self):
        engine = _engine(ROWS)
        engine._search_l2("alpha", None)
        assert engine.last_l2_scope is None

    def test_metadata_carries_scope_and_row_project(self):
        engine = _engine(ROWS)
        out = engine._search_l2("alpha", "A")
        by_row = {r.metadata.get("source_rowid"): r.metadata for r in out}
        assert by_row[11]["project_scope"] == "A"
        assert by_row[11]["project"] == "A", "命中的行要带上它自己的归属"
        assert by_row[33]["project_scope"] == "A"
        assert "project" not in by_row[33], "全局行没有项目归属，不该伪造一个"

    def test_scope_emits_info_log(self, caplog):
        engine = _engine(ROWS)
        with caplog.at_level("INFO", logger="plugin.memory_governed._recall"):
            engine._search_l2("alpha", "A")
        assert any("project scope" in rec.getMessage() for rec in caplog.records), (
            "作用域生效时必须留下可观测痕迹（日志）"
        )

    def test_no_scope_does_not_log(self, caplog):
        engine = _engine(ROWS)
        with caplog.at_level("INFO", logger="plugin.memory_governed._recall"):
            engine._search_l2("alpha", None)
        assert not any("project scope" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# 配置接线：config.recall.project_scope -> 实际调用
# ---------------------------------------------------------------------------

class TestConfigWiring:
    def test_l2_scope_reads_config(self):
        engine = _engine(ROWS)
        assert engine.l2_scope() is None
        engine._config.recall.project_scope = "A"
        assert engine.l2_scope() == "A"
        engine._config.recall.project_scope = ""
        assert engine.l2_scope() == ""

    def test_l2_scope_is_defensive_without_recall_section(self):
        """替身对象没有 ``recall`` 段时退回 ``None``（不过滤），不能抛异常。"""
        class _Bare:
            recall = None

        engine = RecallEngine.__new__(RecallEngine)
        engine._config = _Bare()
        assert engine.l2_scope() is None

    def test_sequential_recall_passes_scope_to_l2(self):
        engine = _engine(ROWS)
        engine._config.recall.project_scope = "A"
        seen: Dict[str, Any] = {}

        def _spy_l2(query, project=None):
            seen["project"] = project
            return []

        engine._search_l2 = _spy_l2  # type: ignore[assignment]
        engine._search_l3 = lambda _q: []  # type: ignore[assignment]
        engine._get_l4_result = lambda: None  # type: ignore[assignment]
        engine._search_kb = lambda _q: []  # type: ignore[assignment]
        engine._sequential_recall("alpha")
        assert seen["project"] == "A", (
            "顺序回退路径没有把作用域传下去 —— 主路径与回退路径会出现两套答案"
        )

    def test_parallel_recall_passes_scope_to_l2(self):
        engine = _engine(ROWS)
        engine._config.recall.project_scope = "A"
        seen: Dict[str, Any] = {}

        def _spy_l2(query, project=None):
            seen["project"] = project
            return []

        engine._search_l2 = _spy_l2  # type: ignore[assignment]
        engine._search_l3 = lambda _q: []  # type: ignore[assignment]
        engine._get_l4_result = lambda: None  # type: ignore[assignment]
        engine._search_kb = lambda _q: []  # type: ignore[assignment]
        engine.parallel_recall("alpha")
        assert seen.get("project") == "A", (
            "并行主路径没有把作用域传给 L2 —— 生产链路等于没修"
        )

    def test_project_scope_loads_from_json(self, tmp_path):
        """JSON 里的字符串必须原样装载（不能被数值字段校验误伤）。"""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        (home / "governed_memory.json").write_text(
            json.dumps({"recall": {"project_scope": "A"}}), encoding="utf-8")
        cfg = load_governed_config(home)
        assert cfg.recall.project_scope == "A"

    def test_project_scope_empty_string_survives_json(self, tmp_path):
        """``""`` 是与 ``None`` 不同的第三态，不能被当成「未设置」丢掉。"""
        home = tmp_path / ".hermes"
        home.mkdir(parents=True)
        (home / "governed_memory.json").write_text(
            json.dumps({"recall": {"project_scope": ""}}), encoding="utf-8")
        cfg = load_governed_config(home)
        assert cfg.recall.project_scope == ""

    def test_project_scope_defaults_to_none(self):
        """默认不设 -> 不过滤，向后兼容旧配置。"""
        assert GovernedMemoryConfig().recall.project_scope is None
