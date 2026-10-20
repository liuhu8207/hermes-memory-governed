# -*- coding: utf-8 -*-
"""L3 writer must keep messages.id and messages_fts.rowid aligned."""

from __future__ import annotations

import sqlite3

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._sync import L3Writer


def test_writer_uses_messages_rowid_for_fts(tmp_path):
    cfg = GovernedMemoryConfig()
    cfg.l3_db_path = str(tmp_path / "l3.db")

    # 先写一行让 L3Writer 建表，再制造 id 空洞。
    writer = L3Writer(cfg)
    writer.write([{"role": "user", "content": "seed row"}], session_id="s0")
    writer.shutdown()

    # 占一个高位 id，让下一条消息的 id 与 FTS 自增序列拉开。
    conn = sqlite3.connect(cfg.l3_db_path)
    conn.execute(
        "INSERT INTO messages (id, session_id, role, content, timestamp, hash) "
        "VALUES (100, 's0', 'user', 'old row', 1.0, 'x')")
    conn.execute(
        "INSERT INTO messages_fts (rowid, content, role, session_id, timestamp) "
        "VALUES (100, 'old row', 'user', 's0', 1.0)")
    conn.commit()
    conn.close()

    writer = L3Writer(cfg)
    writer.write([{"role": "user", "content": "new row after a gap"}],
                 session_id="s1")
    writer.shutdown()

    conn = sqlite3.connect(cfg.l3_db_path)
    try:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        msg_rowid = conn.execute(
            "SELECT id FROM messages WHERE content = 'new row after a gap'"
        ).fetchone()[0]
        fts_rowid = conn.execute(
            "SELECT rowid FROM messages_fts WHERE content = 'new row after a gap'"
        ).fetchone()[0]
        assert msg_rowid == fts_rowid, (
            "messages.id=%s but messages_fts.rowid=%s — the indices drifted"
            % (msg_rowid, fts_rowid))
    finally:
        conn.close()
