# -*- coding: utf-8 -*-
"""L3 recall caps one session from monopolizing the result window."""

from __future__ import annotations

import sqlite3
import time

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._recall import RecallEngine


def test_l3_per_session_cap_preserves_other_sessions(tmp_path):
    db = tmp_path / "l3.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)")
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5("
        "content, role, session_id, timestamp)")
    now = time.time()
    rows = []
    for index in range(6):
        rows.append((f"big", "user", f"共享记忆 大会话 {index}", now - index))
    for index in range(3):
        rows.append((f"small", "user", f"共享记忆 小会话 {index}", now - 10 - index))
    for session_id, role, content, timestamp in rows:
        cur = conn.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) "
            "VALUES (?,?,?,?)", (session_id, role, content, timestamp))
        conn.execute(
            "INSERT INTO messages_fts(rowid, content, role, session_id, timestamp) "
            "VALUES (?,?,?,?,?)",
            (cur.lastrowid, content, role, session_id, timestamp))
    conn.commit()
    conn.close()

    cfg = GovernedMemoryConfig()
    cfg.l3_db_path = str(db)
    cfg.recall.l3_max_results = 5
    cfg.recall.l3_max_per_session = 2
    results = RecallEngine(cfg)._search_l3("共享记忆")
    sessions = [str((r.metadata or {}).get("session_id") or "") for r in results]
    assert sessions.count("big") == 2
    assert sessions.count("small") == 2
    assert len(results) == 4
