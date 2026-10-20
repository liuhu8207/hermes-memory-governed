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


def test_policy_prose_is_not_mistaken_for_a_credential():
    r"""「密码：必须至少 12 位」是规则，不是凭据。

    旧的 ``\S{3,}`` 把冒号后的说明词当成了密码值，脱敏成
    ``[REDACTED] 12 位`` —— 把一句规则改成了半句话。
    """
    for text in (
        "密码：必须至少 12 位",
        "口令：请设置为复杂口令",
        "password: min 12 chars",
        "password: at least 12 characters",
    ):
        assert redact_for_archive(text) == text


def test_real_credentials_are_still_redacted():
    """正对照：别把规则修得太宽，真凭据必须继续拦下。"""
    for text in ("密码: abc123", "password: hunter2", "token: sk-abcdefghijklmnop1234"):
        assert "[REDACTED" in redact_for_archive(text)


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
