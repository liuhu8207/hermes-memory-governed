"""Regression tests for L4 persona generation (``scripts/l4_persona_daily.py``).

Covers the P1 under-count defect:

* the minimal deployment runtime ships ``lancedb`` / ``pyarrow`` but **not**
  pandas, so the old ``table.to_pandas()`` call raised ``ModuleNotFoundError``
  that was silently swallowed, leaving L2 facts permanently empty;
* the L3 read was hard-coded to ``limit=50`` against a 657-row archive.

The L2 tests inject a fake ``lancedb`` module whose ``to_pandas`` always raises,
mirroring the deployment runtime, and assert the pyarrow path still succeeds.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
for _p in (str(ROOT), str(SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _FakeArrowTable:
    """Minimal stand-in for ``pyarrow.Table`` used by ``_table_records``."""

    def __init__(self, records: list[dict], names: list[str]) -> None:
        self._records = [dict(r) for r in records]
        self.schema = types.SimpleNamespace(names=list(names))

    def select(self, columns):
        columns = list(columns)
        return _FakeArrowTable(
            [{k: r.get(k) for k in columns} for r in self._records], columns
        )

    def to_pylist(self) -> list[dict]:
        return [dict(r) for r in self._records]


class _FakeLanceTable:
    """LanceDB-like table whose ``head`` works but ``to_pandas`` is broken."""

    def __init__(self, records: list[dict]) -> None:
        self._records = [dict(r) for r in records]

    def count_rows(self) -> int:
        return len(self._records)

    def head(self, n=5) -> _FakeArrowTable:
        recs = self._records[:n]
        names = list(recs[0].keys()) if recs else ["content", "category"]
        return _FakeArrowTable(recs, names)

    def to_arrow(self) -> _FakeArrowTable:
        names = list(self._records[0].keys()) if self._records else []
        return _FakeArrowTable(self._records, names)

    def to_pandas(self):
        # Mirrors the deployment runtime where pandas is NOT installed.
        raise ModuleNotFoundError("No module named 'pandas'")

    def to_pandas_broken(self):  # pragma: no cover - clarity helper
        return self.to_pandas()


def _make_fake_lancedb(table: _FakeLanceTable):
    """A fake ``lancedb`` module exposing ``connect`` -> db with one table."""
    fake_db = types.SimpleNamespace(
        list_tables=lambda: types.SimpleNamespace(tables=["memories"]),
        open_table=lambda name: table,
    )
    return types.SimpleNamespace(connect=lambda path: fake_db)


@pytest.fixture
def l4(tmp_path, monkeypatch):
    """Import ``l4_persona_daily`` against an isolated, temp HERMES_HOME."""
    home = tmp_path / "hermes_home"
    (home / "memory" / "l2").mkdir(parents=True)
    (home / "memory" / "l3").mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_MEMORY_DIR", "PERSONA_PATH", "PERSONA_META_PATH", "L3_DB_PATH"):
        monkeypatch.delenv(var, raising=False)

    sys.modules.pop("l4_persona_daily", None)
    import l4_persona_daily as module  # noqa: WPS433 - deliberate late import

    return module


def test_table_records_uses_arrow_when_pandas_raises():
    """``_table_records`` must succeed through pyarrow even if pandas is gone."""
    table = _FakeLanceTable([
        {"content": "fact A", "category": "proj", "source": "s",
         "timestamp": "t", "vector": [0.0] * 8},
        {"content": "fact B", "category": "pref", "source": "s",
         "timestamp": "t", "vector": [0.0] * 8},
    ])

    import l4_persona_daily as m

    records = m._table_records(table, 10)

    assert [r["content"] for r in records] == ["fact A", "fact B"]
    assert all("vector" not in r for r in records)  # projection drops embeddings


def test_table_records_respects_limit_and_zero():
    import l4_persona_daily as m

    table = _FakeLanceTable([{"content": f"{i}", "category": "c"} for i in range(5)])
    assert len(m._table_records(table, 3)) == 3
    assert m._table_records(table, 0) == []


def test_load_l2_facts_reads_via_arrow(l4, monkeypatch):
    """End-to-end ``load_l2_facts`` works with a broken pandas, via pyarrow."""
    table = _FakeLanceTable([
        {"content": "deploy uses ffmpeg", "category": "project"},
        {"content": "prefers terse replies", "category": "preference"},
    ])
    monkeypatch.setitem(sys.modules, "lancedb", _make_fake_lancedb(table))

    facts = l4.load_l2_facts()

    assert len(facts) == 2
    assert {f["category"] for f in facts} == {"project", "preference"}


def test_load_l2_facts_logs_warning_on_failure(l4, monkeypatch, caplog):
    """A read failure must not be swallowed silently — it logs a warning."""

    def _boom(path):
        raise RuntimeError("lance broken")

    fake = types.SimpleNamespace(connect=_boom)
    monkeypatch.setitem(sys.modules, "lancedb", fake)

    with caplog.at_level("WARNING", logger="hermes_governed.l4_persona"):
        facts = l4.load_l2_facts()

    assert facts == []
    assert any("Failed to read L2 facts" in r.message for r in caplog.records)


def test_generate_persona_populates_fact_categories(l4):
    l2 = [
        {"content": "a", "category": "project"},
        {"content": "b", "category": "project"},
        {"content": "c", "category": "preference"},
        {"content": "d"},  # missing category -> "other"
    ]
    persona = l4.generate_persona({}, l2, ["m1", "m2", "m3"])

    assert persona["fact_categories"] == {"project": 2, "preference": 1, "other": 1}
    assert persona["conversation_count"] == 3
    # Real L2 content is carried through (not just an L1 echo).
    assert persona["key_facts"] == ["a", "b", "c", "d"]


def test_generate_persona_bounds_key_facts_and_dedups(l4):
    l2 = [{"content": f"f{i}", "category": "c"} for i in range(50)]
    l2.append({"content": "f0", "category": "c"})  # duplicate

    persona = l4.generate_persona({}, l2, [], max_facts=5, fact_chars=100)

    assert len(persona["key_facts"]) == 5
    assert persona["key_facts"] == [f"f{i}" for i in range(5)]
    # category counting still sees every row, including the duplicate
    assert persona["fact_categories"] == {"c": 51}


def test_key_facts_are_redacted(l4):
    l2 = [{"content": "api_key=sk-ABCDEFGHIJKLMNOPqrst", "category": "c"}]
    persona = l4.generate_persona({}, l2, [])
    assert "sk-ABCDEFGHIJKLMNOPqrst" not in persona["key_facts"][0]
    assert "[REDACTED]" in persona["key_facts"][0]


def _seed_l3(l4, rows: list[tuple[str, str, float]]) -> None:
    """Create the L3 ``messages`` table and insert ``(role, content, timestamp)``."""
    db_path = l4.L3_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, timestamp REAL)"
    )
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, timestamp) "
        "VALUES ('s', ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_load_l3_recent_filters_tool_role(l4):
    """Only user/assistant dialogue is returned; tool dumps are excluded."""
    now = time.time()
    _seed_l3(l4, [
        ("user", "hello", now - 10),
        ("assistant", "hi there", now - 9),
        ("tool", '{"output": "dump"}', now - 8),
        ("user", "another", now - 7),
        ("tool", "terminal dump", now - 6),
    ])

    recent = l4.load_l3_recent()

    assert sorted(recent) == ["another", "hello", "hi there"]
    assert all("dump" not in r for r in recent)
    # roles=None opts out of filtering (includes tool rows)
    assert len(l4.load_l3_recent(roles=None)) == 5


def test_load_l3_recent_window_excludes_old(l4):
    """Default window drops ages beyond ``days`` and honours the cap."""
    now = time.time()
    rows = [("user", f"recent {i}", now - i * 60) for i in range(80)]
    rows.append(("user", "ancient", now - 100 * 86400))
    _seed_l3(l4, rows)

    recent = l4.load_l3_recent()  # default: 90 days, cap 2000
    assert len(recent) == 80
    assert "ancient" not in recent

    assert len(l4.load_l3_recent(limit=50)) == 50
    assert len(l4.load_l3_recent(days=None)) == 81


def test_count_l3_messages_counts_all_roles(l4):
    now = time.time()
    _seed_l3(l4, [
        ("user", "a", now),
        ("assistant", "b", now),
        ("tool", "c", now),
    ])
    assert l4.count_l3_messages() == 3
    assert len(l4.load_l3_recent()) == 2  # dialogue only


def test_count_l3_messages_zero_without_db(l4):
    assert l4.count_l3_messages() == 0  # fixture has no l3.db


def test_generate_persona_records_total_messages(l4):
    persona = l4.generate_persona({}, [], ["m1", "m2"], message_count_total=657)
    assert persona["conversation_count"] == 2
    assert persona["message_count_total"] == 657


def test_load_l3_recent_logs_warning_on_failure(l4, caplog):
    """A malformed L3 db must warn instead of returning [] silently."""
    db_path = l4.L3_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text("this is not a sqlite database", encoding="utf-8")

    with caplog.at_level("WARNING", logger="hermes_governed.l4_persona"):
        recent = l4.load_l3_recent()

    assert recent == []
    assert any(
        rec.levelno >= logging.WARNING and "L3" in rec.getMessage()
        for rec in caplog.records
    )
