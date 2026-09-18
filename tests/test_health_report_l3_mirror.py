# -*- coding: utf-8 -*-
"""The health report must notice when the L3 text index has drifted.

Why this exists
---------------
`_sync` mirrors every L3 row into `messages_fts`, and its docstring states the
contract plainly: a row that lands in `messages` but not in the index is
**data loss, not a debug-level curiosity**. What nothing did was *re-check* the
contract afterwards. The health report printed "L3 FTS5 — indexed" and stopped,
so a divergence in either direction was invisible.

Measured 2026-09-18, on the real store: one FTS-only row existed — a session
from a one-off test whose cleanup deleted from `messages` without touching the
index — and **no component reported it**. The report now does, with the two
directions graded differently, because they are not equally bad:

* in `messages`, not in the index → that text can never be found again → **error**
* only in the index → search still returns the text, but the source row is gone,
  which means a delete happened outside the product → **warning**
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "memory_health_report.py"

# The report imports ``hermes_env`` as a sibling (that is how the installer
# deploys it, side by side in ``$HERMES_HOME/scripts``). The file is in the repo
# too, so putting ``scripts/`` on the path is all the test needs.
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


def _load():
    spec = importlib.util.spec_from_file_location("health_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_l3(root: Path, rows, fts_only=()) -> None:
    """Build an L3 database at ``root/l3/l3.db``.

    ``rows`` are mirrored into both tables; ``fts_only`` are inserted into the
    index alone, which is the orphan shape seen in the real store.
    """
    d = root / "l3"
    d.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(d / "l3.db")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
                 " role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)")
    conn.execute("CREATE VIRTUAL TABLE messages_fts "
                 "USING fts5(content, role, session_id, timestamp)")
    for i, (sess, role, text) in enumerate(rows, 1):
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp, hash)"
                     " VALUES (?,?,?,?,?)", (sess, role, text, 1.0, "h%d" % i))
        conn.execute("INSERT INTO messages_fts (content, role, session_id, timestamp)"
                     " VALUES (?,?,?,?)", (text, role, sess, "1.0"))
    for sess, role, text in fts_only:
        conn.execute("INSERT INTO messages_fts (content, role, session_id, timestamp)"
                     " VALUES (?,?,?,?)", (text, role, sess, "1.0"))
    conn.commit()
    conn.close()


@pytest.fixture
def report(tmp_path, monkeypatch):
    mod = _load()
    monkeypatch.setattr(mod, "MEMORY_DIR", tmp_path)
    return mod, tmp_path


def _run(mod):
    h = mod.HealthCheck()
    mod.check_l3(h)
    return h


class TestTheMirrorIsVerified:
    def test_an_aligned_store_reports_no_problem(self, report):
        mod, root = report
        _make_l3(root, [("s", "user", "hello"), ("s", "assistant", "hi")])
        h = _run(mod)
        assert "L3 FTS mirror" not in h.errors
        assert "L3 FTS mirror" not in h.warnings
        assert any(c["name"] == "L3 FTS mirror" and c["ok"] for c in h.checks)

    def test_a_row_missing_from_the_index_is_an_error(self, report):
        """The serious direction: the text is gone from search for good."""
        mod, root = report
        _make_l3(root, [("s", "user", "hello")])
        # Delete from the source table only — exactly what an out-of-band
        # cleanup does, and what leaves a message unsearchable forever.
        conn = sqlite3.connect(root / "l3" / "l3.db")
        conn.execute("INSERT INTO messages (session_id, role, content, timestamp, hash)"
                     " VALUES ('s','user','只有源表有','1.0','hx')")
        conn.commit()
        conn.close()

        h = _run(mod)
        assert "L3 FTS mirror" in h.errors, (
            "源表有、索引没有 —— 这正是会被永久搜不到的那类，必须算 error")
        assert not h.warnings

    def test_an_index_only_row_is_a_warning_not_an_error(self, report):
        """The shape found in the real store: searchable, but its source is gone."""
        mod, root = report
        _make_l3(root, [("s", "user", "hello")],
                 fts_only=[("upgrade-e2e-test", "assistant", "只在索引里的残留")])
        h = _run(mod)
        assert "L3 FTS mirror" in h.warnings, (
            "索引多出来的行要报出来（它说明有一次未经产品的删除），"
            "但检索仍能返回原文，不该判成数据丢失")
        assert "L3 FTS mirror" not in h.errors

    def test_a_missing_index_table_is_only_a_warning(self, report):
        """L3 without FTS is degraded, not broken — every layer must still be readable."""
        mod, root = report
        d = root / "l3"
        d.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(d / "l3.db")
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT,"
                     " role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)")
        conn.commit()
        conn.close()

        h = _run(mod)
        assert "L3 FTS5" in h.warnings
        assert not h.errors

    def test_a_missing_database_is_reported(self, report):
        mod, root = report          # nothing created at all
        h = _run(mod)
        assert "L3 SQLite" in h.errors
