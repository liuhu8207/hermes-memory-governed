"""Daily L2 capacity sweep: least-used rows are archived and removed."""

from __future__ import annotations

from pathlib import Path

from plugin.memory_governed import _lifecycle as life


class _Arrow:
    def __init__(self, rows):
        self._rows = rows
        self.num_rows = len(rows)

    def to_pylist(self):
        return list(self._rows)


class _Table:
    def __init__(self, rows):
        self.rows = list(rows)

    def to_arrow(self):
        return _Arrow(self.rows)

    def delete(self, predicate: str):
        prefix = "content = '"
        assert predicate.startswith(prefix)
        value = predicate[len(prefix):-1].replace("''", "'")
        self.rows = [r for r in self.rows if r.get("content") != value]


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
    monkeypatch.setattr(life, "usage_last_used", lambda _dir: {})

    demoted = life.enforce_capacity(table, Path(tmp_path), 1, agent="hermes")

    assert demoted == ["a"]
    assert [r["content"] for r in table.rows] == ["b", "c"]
