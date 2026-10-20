"""Daily L2 capacity sweep: least-used rows are archived and removed."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from plugin.memory_governed import _lifecycle as life


class _Arrow:
    def __init__(self, rows):
        self._rows = rows
        self.num_rows = len(rows)

    def to_pylist(self):
        return list(self._rows)


def _matches(row, predicate: str) -> bool:
    """解析 ``col = 'value' AND ...`` / ``col = 123`` 谓词（够测试用）。"""
    for part in predicate.split(" AND "):
        m = re.match(r"(\w+)\s*=\s*'((?:[^']|'')*)'$", part.strip())
        if m:
            col, val = m.group(1), m.group(2).replace("''", "'")
            if str(row.get(col)) != val:
                return False
            continue
        m = re.match(r"(\w+)\s*=\s*(-?\d+)$", part.strip())
        if m:
            col, val = m.group(1), int(m.group(2))
            if row.get(col) != val:
                return False
            continue
        raise AssertionError("unparsed predicate part: %r" % part)
    return True


class _Table:
    def __init__(self, rows):
        self.rows = list(rows)
        self.deleted_predicates = []

    def to_arrow(self):
        return _Arrow(self.rows)

    def count_rows(self, filter: str = ""):
        return sum(1 for r in self.rows if _matches(r, filter))

    def delete(self, predicate: str):
        self.deleted_predicates.append(predicate)
        self.rows = [r for r in self.rows if not _matches(r, predicate)]


def _usage(contents):
    return {life.content_fp(c): float(i) for i, c in enumerate(contents)}


def test_enforce_capacity_evicts_least_used(monkeypatch, tmp_path):
    rows = [
        {"content": "least used", "agent": "hermes", "timestamp": "2026-10-01"},
        {"content": "middle", "agent": "hermes", "timestamp": "2026-10-02"},
        {"content": "most used", "agent": "hermes", "timestamp": "2026-10-03"},
    ]
    table = _Table(rows)
    archived = []
    monkeypatch.setattr(life, "backup_table", lambda *a, **k: tmp_path / "backup")
    monkeypatch.setattr(life, "archive_append", lambda _dir, rec: archived.append(rec))
    monkeypatch.setattr(life, "usage_last_used", lambda _dir: {
        life.content_fp("least used"): 0.0,
        life.content_fp("middle"): 1.0,
        life.content_fp("most used"): 2.0,
    })

    demoted = life.enforce_capacity(table, Path(tmp_path), 2, agent=None)

    assert demoted == ["least used"]
    assert [r["content"] for r in table.rows] == ["middle", "most used"]
    assert archived and archived[0]["reason"] == "capacity-demoted"


def test_enforce_capacity_respects_agent_scope(monkeypatch, tmp_path):
    rows = [
        {"content": "a", "agent": "hermes", "timestamp": "1"},
        {"content": "b", "agent": "hermes", "timestamp": "2"},
        {"content": "c", "agent": "workbuddy", "timestamp": "3"},
    ]
    table = _Table(rows)
    monkeypatch.setattr(life, "backup_table", lambda *a, **k: tmp_path / "backup")
    monkeypatch.setattr(life, "archive_append", lambda *a, **k: None)
    monkeypatch.setattr(life, "usage_last_used", lambda _dir: _usage(["a", "b", "c"]))

    demoted = life.enforce_capacity(table, Path(tmp_path), 1, agent="hermes")

    assert demoted == ["a"]
    assert [r["content"] for r in table.rows] == ["b", "c"]


def test_no_usage_data_means_no_eviction(monkeypatch, tmp_path):
    """没有使用度数据时按时间删最旧 = 会删掉被召回过的事实（实测）。

    没有数据就跳过（fail-closed），并让调用方能区分「不需要」和「拒绝」。
    """
    rows = [
        {"content": "old but recalled", "agent": "hermes", "timestamp": "1"},
        {"content": "new", "agent": "hermes", "timestamp": "2"},
    ]
    table = _Table(rows)
    monkeypatch.setattr(life, "usage_last_used", lambda _dir: {})
    report: dict = {}

    demoted = life.enforce_capacity(table, Path(tmp_path), 1, report=report)

    assert demoted == []
    assert len(table.rows) == 2, "无使用度数据时不得删除"
    assert report.get("skipped_reason") == "no usage data"


def test_backup_failure_is_an_error_not_a_silent_noop(monkeypatch, tmp_path):
    rows = [
        {"content": "a", "agent": "hermes", "timestamp": "1"},
        {"content": "b", "agent": "hermes", "timestamp": "2"},
    ]
    table = _Table(rows)
    monkeypatch.setattr(life, "backup_table", lambda *a, **k: None)
    monkeypatch.setattr(life, "usage_last_used", lambda _dir: _usage(["a", "b"]))

    with pytest.raises(life.CapacityRefused):
        life.enforce_capacity(table, Path(tmp_path), 1)
    assert len(table.rows) == 2, "备份失败时一行都不能删"


def test_duplicate_content_does_not_delete_unrelated_rows(monkeypatch, tmp_path):
    """只按 content 删会连带删掉同内容的其它行（实测 1 个受害者删 3 行）。"""
    rows = [
        {"content": "same text", "agent": "hermes", "timestamp": "1"},
        {"content": "same text", "agent": "hermes", "timestamp": "2"},
        {"content": "same text", "agent": "hermes", "timestamp": "3"},
        {"content": "other", "agent": "hermes", "timestamp": "4"},
    ]
    table = _Table(rows)
    report: dict = {}
    monkeypatch.setattr(life, "backup_table", lambda *a, **k: tmp_path / "backup")
    monkeypatch.setattr(life, "archive_append", lambda *a, **k: None)
    monkeypatch.setattr(life, "usage_last_used", lambda _dir: {
        life.content_fp("same text"): 0.0,
        life.content_fp("other"): 9.0,
    })

    demoted = life.enforce_capacity(table, Path(tmp_path), 3, report=report)

    assert len(demoted) == 1
    # 只删掉时间戳最早的那一行，同内容的其它行保留
    assert len(table.rows) == 3
    assert sorted(r["timestamp"] for r in table.rows if r["content"] == "same text") == ["2", "3"]
    assert report.get("rows_removed") == 1
