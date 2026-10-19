"""L3 archive redaction: secrets and inline images must not persist."""

from __future__ import annotations

import sqlite3

from plugin.memory_governed._sync import L3Writer, redact_for_archive


def test_redact_for_archive_removes_secrets_and_images():
    text = (
        "api_key=sk-abcdefghijklmnopqrstuvwx "
        "password: hunter2 "
        "image=data:image/png;base64,AAAA"
    )
    out = redact_for_archive(text)
    assert "sk-abcdefghijklmnopqrstuvwx" not in out
    assert "hunter2" not in out
    assert "data:image/png;base64" not in out
    assert "[REDACTED" in out


def test_redact_for_archive_keeps_clean_text():
    text = "Hermes Gateway 重启必须用 schtasks /run /tn Hermes_Gateway"
    assert redact_for_archive(text) == text


def test_l3_writer_stores_redacted_content(config):
    secret = "sk-" + "A" * 24
    writer = L3Writer(config)
    writer.write(
        [{"role": "tool", "content": f"key={secret} password: hunter2"}],
        session_id="s1",
    )
    writer.shutdown()

    conn = sqlite3.connect(config.l3_db_path)
    try:
        content = conn.execute("SELECT content FROM messages").fetchone()[0]
        fts = conn.execute("SELECT content FROM messages_fts").fetchone()[0]
    finally:
        conn.close()

    for stored in (content, fts):
        assert secret not in stored
        assert "hunter2" not in stored
        assert "[REDACTED" in stored


def test_search_l3_redacts_legacy_rows(hermes_home, monkeypatch):
    db = hermes_home / "memory" / "l3" / "l3.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL, metadata TEXT, hash TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
    )
    secret = "sk-" + "B" * 24
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        ("legacy", "tool", f"secret token {secret}", 1.0),
    )
    conn.execute(
        "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?,?,?,?)",
        (f"secret token {secret}", "tool", "legacy", 1.0),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    import memory_cli

    hits = memory_cli.search_l3("secret", 5)
    assert hits
    assert secret not in hits[0]["content"]
    assert "[REDACTED" in hits[0]["content"]
