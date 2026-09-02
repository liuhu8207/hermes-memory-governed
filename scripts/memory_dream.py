#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory dream: consolidate old L3 sessions into compact L2 summaries.

Inspired by OpenSquilla's dream consolidation — but safety-first:
- Opt-in: does nothing unless --yes (or --dry-run for preview)
- Old sessions (default >= 30 days) are summarized into compact
  "dream:*" facts written back to L2 (category="dream")
- Original L3 messages are NOT deleted (use l3_retention.py for that)
- LLM summarization is optional: with LLM_API_KEY/LLM_BASE_URL/LLM_MODEL
  set it produces abstractive summaries; without, it falls back to
  extractive summaries (first user message + key sentences)

Usage:
    python scripts/memory_dream.py --dry-run --days 30
    python scripts/memory_dream.py --days 30 --yes
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager

logger = setup_logging("memory_dream")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
L2_DB_PATH = env_path("L2_DB_PATH", MEMORY_DIR / "l2")
DREAM_STATE = env_path("DREAM_STATE_DIR", HERMES_HOME / "state")


def load_old_sessions(days: int, limit: int) -> list[dict[str, Any]]:
    """Group L3 messages older than `days` by session_id."""
    if not L3_DB_PATH.exists():
        logger.warning("L3 database not found: %s", L3_DB_PATH)
        return []
    cutoff = time.time() - days * 86400
    conn = sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True)
    sessions = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT session_id FROM messages WHERE timestamp < ? AND session_id != '' LIMIT ?",
            (cutoff, limit),
        ).fetchall()
    ]
    out = []
    for sid in sessions:
        rows = conn.execute(
            "SELECT id, role, content, timestamp FROM messages WHERE session_id = ? AND content != '' ORDER BY timestamp ASC",
            (sid,),
        ).fetchall()
        if not rows:
            continue
        out.append({
            "session_id": sid,
            "messages": [{"rowid": r[0], "role": r[1], "content": r[2], "timestamp": r[3]} for r in rows],
        })
    conn.close()
    logger.info("found %d old sessions (>= %d days)", len(out), days)
    return out


def llm_available() -> bool:
    return bool(os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL") and os.getenv("LLM_MODEL"))


def llm_summarize(messages: list[dict[str, Any]]) -> str | None:
    """Abstractive summary via OpenAI-compatible API; None on any failure."""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")
    convo = "\n".join(
        f"[{m['role']}]: {(m['content'] or '')[:400]}" for m in messages[:40]
    )
    prompt = (
        "请把以下旧对话压缩为一段 100 字以内的记忆摘要，"
        "保留关键决策、事实和偏好，去掉寒暄：\n\n" + convo
    )
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={"model": model, "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 300, "temperature": 0.2},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.warning("LLM summarize failed: %s", e)
        return None


def extractive_summary(messages: list[dict[str, Any]]) -> str:
    """Extractive fallback: first user message + longest assistant statement."""
    user_first = next((m["content"] for m in messages if m["role"] == "user"), "")
    asst_long = max(
        (m["content"] for m in messages if m["role"] == "assistant"),
        key=len, default="",
    )
    parts = []
    if user_first:
        parts.append("主题: " + user_first[:200])
    if asst_long:
        parts.append("要点: " + asst_long[:200])
    return " | ".join(parts) or "（无内容）"


def write_dream_fact(summary: str, session_id: str, source_rowid: int | None, timestamp: float) -> None:
    """Write a dream summary fact into L2."""
    import lancedb

    L2_DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(L2_DB_PATH))
    if "memories" not in db.list_tables().tables:
        logger.warning("L2 memories table not found — run the agent once or l2_rebuild.py first")
        return
    table = db.open_table("memories")

    fact: dict[str, Any] = {
        "content": f"dream: {session_id} — {summary}"[:600],
        "category": "dream",
        "source": "memory-dream",
        "timestamp": datetime.now().isoformat(),
    }
    # Fill vector if an embedding backend is available
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from plugin.memory_governed._embedding import load_embedder  # noqa: E402
    except ImportError:
        try:
            from memory_governed._embedding import load_embedder  # type: ignore[no-redef]
        except ImportError:
            load_embedder = None

    embedder = None
    if load_embedder is not None:
        try:
            from plugin.memory_governed._config import load_governed_config  # noqa: E402

            cfg = load_governed_config(HERMES_HOME)
            embedder = load_embedder(cfg.vector.backend, cfg.vector.model)
        except Exception:
            embedder = None

    has_vec_col = "vector" in [f.name for f in table.schema]
    if has_vec_col and embedder is not None:
        encoded = embedder.encode([fact["content"]])
        # fastembed backend 返回一个只有 .tolist() 的 "R" 对象（无 __iter__），
        # sentence-transformers 返回 numpy 数组。统一走 .tolist() 兜底。
        rows = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        if rows:
            fact["vector"] = [float(x) for x in rows[0]] if not isinstance(rows[0], list) else list(rows[0])
    if "source_rowid" in [f.name for f in table.schema] and source_rowid is not None:
        fact["source_rowid"] = int(source_rowid)
    table.add([fact])


def main() -> int:
    parser = argparse.ArgumentParser(description="Dream consolidation (opt-in)")
    parser.add_argument("--days", type=int, default=30, help="Consolidate sessions older than N days")
    parser.add_argument("--limit", type=int, default=50, help="Max sessions per run")
    parser.add_argument("--dry-run", action="store_true", help="Preview summaries without writing")
    parser.add_argument("--yes", action="store_true", help="Confirm writing dream facts to L2")
    args = parser.parse_args()

    sessions = load_old_sessions(args.days, args.limit)
    if not sessions:
        print("no old sessions to consolidate")
        return 0

    use_llm = llm_available()
    print(f"old sessions: {len(sessions)} | LLM: {'abstractive' if use_llm else 'extractive fallback'}")

    state = StateManager(DREAM_STATE, "dream")

    written = 0
    for session in sessions:
        # Dedup: one dream fact per session, ever
        if state.is_processed(session["session_id"]):
            continue

        summary = None
        if use_llm:
            summary = llm_summarize(session["messages"])
        if not summary:
            summary = extractive_summary(session["messages"])

        first_rowid = session["messages"][0].get("rowid")
        print(f"  [{session['session_id'][:40]}] {summary[:100]}")

        if not args.dry_run and args.yes:
            write_dream_fact(summary, session["session_id"], first_rowid, time.time())
            state.mark_processed(session["session_id"])
            written += 1

    if args.dry_run:
        print("(dry-run — nothing written)")
    elif args.yes:
        print(f"wrote {written} dream facts to L2")
    else:
        print("nothing written — pass --yes to write dream facts to L2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
