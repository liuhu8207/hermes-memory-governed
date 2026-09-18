# -*- coding: utf-8 -*-
"""Opening (or creating) the L2 table when another process is doing it too.

LanceDB creates a table in more than one step — the directory appears before
``_versions`` does. Measured 2026-09-17, four agents writing at once on a fresh
HERMES_HOME, two of them died with:

    RuntimeError: Table 'memories' exists but could not be loaded
    (it may be corrupt or incomplete): ... memories.lance was not found:
    Not found: .../memories.lance/_versions

and printed no JSON at all. Both halves of that are tested here: the retry that
waits the other process out, and the ``errors`` sink that turns exhaustion into
something the caller can report instead of a traceback.

Scope note: the structured-refusal conversion inside ``cmd_remember`` is not
unit-tested here — reaching it needs the embedding backend and hashing plumbing
of a real write. It is covered by the live check that motivated this file
(12/12 concurrent cold-start writes, 0 tracebacks, against 2-of-4 failing
before).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import memory_cli as cli                                   # noqa: E402
from plugin.memory_governed import _sync                   # noqa: E402

COLS = [c for c, _t in _sync.L2_PROVENANCE_COLUMNS]


class _Col:
    def __init__(self, name):
        self.name = name


class _Cfg:
    """``open_l2_table`` only reads ``l2_db_path``; ``lancedb.connect`` is stubbed."""

    l2_db_path = "C:/tmp/unused-by-this-test"


class _Table:
    def __init__(self):
        self.schema = [_Col(c) for c in COLS]
        self.added = []

    def add_columns(self, field):
        self.added.append(field)


class _DB:
    """A connection whose create/open behaviour is scripted per call."""

    def __init__(self, visible=(), open_raises=0, create_raises=0):
        self._visible = list(visible)
        self._open_raises = open_raises
        self._create_raises = create_raises
        self.created = 0
        self.opened = 0

    def list_tables(self):
        return list(self._visible)

    def open_table(self, _name):
        self.opened += 1
        if self._open_raises > 0:
            self._open_raises -= 1
            raise RuntimeError("Table 'memories' exists but could not be loaded "
                               "(it may be corrupt or incomplete)")
        if "memories" not in self._visible:
            self._visible.append("memories")
        return _Table()

    def create_table(self, _name, schema=None):
        if self._create_raises > 0:
            self._create_raises -= 1
            # Losing the race means the *winner's* table is now there to open —
            # that is the whole reason a retry can succeed. Modelling only the
            # exception would have the loser retry into the same empty directory
            # for ever, which is not what happens.
            if "memories" not in self._visible:
                self._visible.append("memories")
            raise RuntimeError("another process is creating this table")
        self.created += 1
        self._visible.append("memories")
        return _Table()


@pytest.fixture
def no_backoff(monkeypatch):
    """Retries are the subject; the sleeping is not."""
    monkeypatch.setattr(cli, "_L2_OPEN_BACKOFF", 0)
    return monkeypatch


def _wire(monkeypatch, db):
    lancedb = pytest.importorskip("lancedb")

    monkeypatch.setattr(lancedb, "connect", lambda _path: db)
    return db


class TestOpenRetries:
    def test_a_half_made_table_is_waited_out(self, monkeypatch, no_backoff):
        """The exact production symptom: listed, then unopenable."""
        db = _wire(monkeypatch, _DB(visible=("memories",), open_raises=2))
        errors: list = []
        table = cli.open_l2_table(_Cfg(), create_dim=8, errors=errors)

        assert table is not None, "应当重试到对手写完，而不是直接失败"
        assert errors == []
        assert db.opened == 3

    def test_losing_the_create_race_ends_in_a_usable_table(self, monkeypatch, no_backoff):
        db = _wire(monkeypatch, _DB(visible=(), create_raises=1))
        created: list = []
        errors: list = []
        table = cli.open_l2_table(_Cfg(), create_dim=8, created=created,
                                  errors=errors)

        assert table is not None
        assert db.opened >= 1, "输掉建表后应当改为打开对手建好的表"
        assert created == [], "表不是本进程建的，就不能声称冷启动"
        assert errors == []

    def test_it_still_creates_when_nobody_races(self, monkeypatch, no_backoff):
        db = _wire(monkeypatch, _DB(visible=()))
        created: list = []
        table = cli.open_l2_table(_Cfg(), create_dim=8, created=created)

        assert table is not None and db.created == 1
        assert created == [True]

    def test_read_only_caller_still_gets_none(self, monkeypatch, no_backoff):
        """``cmd_agents`` must not conjure a table just by being asked."""
        db = _wire(monkeypatch, _DB(visible=()))
        assert cli.open_l2_table(_Cfg(), create_dim=0) is None
        assert db.created == 0

    def test_exhaustion_reports_why_and_returns_none(self, monkeypatch, no_backoff):
        _wire(monkeypatch, _DB(visible=("memories",), open_raises=99))
        errors: list = []
        assert cli.open_l2_table(_Cfg(), create_dim=8, errors=errors) is None
        assert errors and "l2_open_failed" in errors[0]
        assert "RuntimeError" in errors[0], "要带上异常类型，否则读的人不知道是什么坏了"

    def test_exhaustion_without_a_sink_is_still_none(self, monkeypatch, no_backoff):
        _wire(monkeypatch, _DB(visible=("memories",), open_raises=99))
        assert cli.open_l2_table(_Cfg(), create_dim=8) is None

    def test_retries_are_bounded(self, monkeypatch, no_backoff):
        db = _wire(monkeypatch, _DB(visible=("memories",), open_raises=99))
        cli.open_l2_table(_Cfg(), create_dim=8)
        assert db.opened == cli._L2_OPEN_RETRIES


class TestTableNames:
    @pytest.mark.parametrize("listing, expected", [
        (["a", "b"], ["a", "b"]),
        ([type("T", (), {"name": "a"})(), "b"], ["a", "b"]),
    ])
    def test_it_normalises_across_client_shapes(self, listing, expected):
        class DB:
            def list_tables(self):
                return listing

        assert cli._l2_table_names(DB()) == expected

    def test_it_falls_back_to_the_older_api(self):
        class DB:
            def table_names(self):
                return ["memories"]

        assert cli._l2_table_names(DB()) == ["memories"]
