"""T03 acceptance spec — S-11 Chinese multi-term recall.

Written BEFORE the fix so it doubles as the executable acceptance criteria.
Acceptance targets agreed with the architect (docs/architecture-review.md, T03):

    MUST FIX (currently broken):
        天气 北京  -> >= 2 hits
        会议 下午  -> >= 1 hit

    MUST PRESERVE (regression protection — do not break these):
        天气               -> exactly the 2 rows containing 天气
        北京               -> exactly the 2 rows containing 北京
        weather 天气 北京   -> >= 2 hits (mixed EN+CJK path)

Root cause: _recall.py:312-314. For a pure-CJK query _build_fts_query returns
"", so _search_l3 takes the early-return branch and hands the WHOLE query to
_search_l3_like, which does query.replace(" ", "") — requiring the terms to be
adjacent in the source text. The segmentation logic at _recall.py:362-383 is
already implemented but only runs when the query has a non-CJK part.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from plugin.memory_governed._recall import RecallEngine

# Four archived turns. 天气 appears in MSG_A and MSG_B; 北京 in MSG_B and MSG_D;
# 会议 and 下午 co-occur only in MSG_C and are NOT adjacent.
MSG_A = "今天天气不错，适合出门跑步"
MSG_B = "北京的空气质量今天也可以，天气也不错"
MSG_C = "会议改到下午三点了"
MSG_D = "北京的项目进度怎么样了"

ALL_MESSAGES = [MSG_A, MSG_B, MSG_C, MSG_D]


@pytest.fixture
def l3_with_chinese(config):
    """Seed L3 (messages + messages_fts) with the four Chinese turns."""
    conn = sqlite3.connect(config.l3_db_path)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "session_id TEXT, role TEXT, content TEXT, timestamp REAL, hash TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE messages_fts USING fts5(content, role, session_id, timestamp)"
    )
    now = time.time()
    for i, text in enumerate(ALL_MESSAGES):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
            ("s-cn", "user", text, now - i),
        )
        conn.execute(
            "INSERT INTO messages_fts (content, role, session_id, timestamp) VALUES (?,?,?,?)",
            (text, "user", "s-cn", now - i),
        )
    conn.commit()
    conn.close()
    return config


def _hits(engine, query):
    return {r.content for r in engine._search_l3(query)}


# ---------------------------------------------------------------------------
# MUST FIX — these define "S-11 is fixed"
# ---------------------------------------------------------------------------

class TestMultiTermCjkMustFix:
    def test_two_separate_terms_both_recalled(self, l3_with_chinese):
        """'天气 北京' must match rows containing either term."""
        hits = _hits(RecallEngine(l3_with_chinese), "天气 北京")
        assert len(hits) >= 2, (
            f"'天气 北京' returned {len(hits)} hits ({sorted(hits)}), expected >= 2. "
            "Pure-CJK queries take the _recall.py:312-314 early-return and are "
            "space-stripped into '天气北京', which requires term adjacency."
        )

    def test_two_separate_terms_hit_the_right_rows(self, l3_with_chinese):
        """'天气 北京' must surface the 天气 rows AND the 北京 rows."""
        hits = _hits(RecallEngine(l3_with_chinese), "天气 北京")
        assert MSG_A in hits, f"'天气 北京' missed the 天气 row: {MSG_A}"
        assert MSG_D in hits, f"'天气 北京' missed the 北京 row: {MSG_D}"

    def test_meeting_afternoon_recalled(self, l3_with_chinese):
        """'会议 下午' must match the row where both appear (non-adjacent)."""
        hits = _hits(RecallEngine(l3_with_chinese), "会议 下午")
        assert len(hits) >= 1, (
            f"'会议 下午' returned 0 hits, expected >= 1 ({MSG_C})"
        )
        assert MSG_C in hits


# ---------------------------------------------------------------------------
# MUST PRESERVE — regression protection, do not change this behaviour
# ---------------------------------------------------------------------------

class TestSingleTermCjkMustPreserve:
    def test_single_term_weather(self, l3_with_chinese):
        hits = _hits(RecallEngine(l3_with_chinese), "天气")
        assert hits == {MSG_A, MSG_B}, (
            f"single-term '天气' regressed: {sorted(hits)}"
        )

    def test_single_term_beijing(self, l3_with_chinese):
        hits = _hits(RecallEngine(l3_with_chinese), "北京")
        assert hits == {MSG_B, MSG_D}, (
            f"single-term '北京' regressed: {sorted(hits)}"
        )

    def test_single_term_does_not_over_match(self, l3_with_chinese):
        """A single term must NOT drag in unrelated rows."""
        hits = _hits(RecallEngine(l3_with_chinese), "会议")
        assert hits == {MSG_C}, f"single-term '会议' over-matched: {sorted(hits)}"


class TestMixedQueryMustPreserve:
    def test_mixed_en_cjk_still_works(self, l3_with_chinese):
        """The mixed path (_recall.py:319-383) must not regress."""
        hits = _hits(RecallEngine(l3_with_chinese), "weather 天气 北京")
        assert len(hits) >= 2, (
            f"mixed query 'weather 天气 北京' returned {len(hits)} hits, expected >= 2"
        )

    def test_mixed_query_does_not_over_match(self, l3_with_chinese):
        """Segmented CJK must stay precise — no unrelated rows."""
        hits = _hits(RecallEngine(l3_with_chinese), "weather 会议 下午")
        assert MSG_A not in hits and MSG_D not in hits, (
            f"mixed query pulled in unrelated rows: {sorted(hits)}"
        )


# ---------------------------------------------------------------------------
# T02 — rowid provenance contract (implementation-agnostic)
# ---------------------------------------------------------------------------

class TestRowidProvenanceContract:
    def test_facts_are_traceable_to_their_source_message(self, config, monkeypatch):
        """S-12 acceptance: every extracted fact must resolve to an L3 rowid.

        Asserted end-to-end and implementation-agnostically — it must hold
        whether the fix keys rowid_map by message index or by text.

        Current state: 0/N resolve, because rowid_map is keyed by the FULL
        message while facts are sentence fragments (_sync.py:135, :483).
        """
        from plugin.memory_governed._sync import L3Writer, WriteQueue

        messages = [
            {"role": "user", "content": "我们决定用 Postgres 而不是 MySQL。部署时间定在下周五。"},
            {"role": "assistant", "content": "好的，我会记录 Postgres 这个决定。"},
        ]
        writer = L3Writer(config)
        rowid_map = writer.write(messages, session_id="s-prov")

        queue = WriteQueue(config)
        facts = queue._extract_atomic_facts(messages)

        assert facts, "no facts extracted (test setup problem)"
        assert rowid_map, "L3 write produced no rowids (test setup problem)"

        resolved = [
            f["content"]
            for f in facts
            if queue._resolve_source_rowid(f["content"], rowid_map, messages) is not None
            if hasattr(queue, "_resolve_source_rowid")
        ]
        # Fall back to the shipped lookup while the helper does not exist yet.
        if not hasattr(queue, "_resolve_source_rowid"):
            resolved = [f["content"] for f in facts if rowid_map.get(f["content"]) is not None]

        assert len(resolved) == len(facts), (
            f"only {len(resolved)}/{len(facts)} facts trace back to an L3 row. "
            "rowid_map is keyed by full message content but facts are sentence "
            "fragments, so source_rowid never resolves (_sync.py:135, :483)."
        )
        writer.shutdown()
