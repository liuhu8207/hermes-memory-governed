#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract valuable knowledge from L3 conversation archives.

This script is READ-ONLY with respect to L3.
It only reads from L3 and writes knowledge notes to wiki/knowledge/chats/.

Adapted from hermes-memory-system/scripts/chat_extractor.py (2026-08-28)
for the governed L3 schema (flat `messages` table keyed by session_id,
unix-float timestamps) instead of the legacy conversations/messages schema.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager

logger = setup_logging("chat_extractor")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "chat")


def get_recent_conversations(days: int = 7, limit: int = 50) -> list[dict[str, Any]]:
    """Read recent conversations grouped by session_id from the flat L3 schema."""
    if not L3_DB_PATH.exists():
        logger.warning("L3 database not found: %s", L3_DB_PATH)
        return []

    cutoff = time.time() - days * 86400
    with sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True) as conn:
        sessions = [
            r[0]
            for r in conn.execute(
                """SELECT DISTINCT session_id FROM messages
                   WHERE timestamp > ? AND session_id != ''
                   ORDER BY timestamp DESC LIMIT ?""",
                (cutoff, limit),
            ).fetchall()
        ]

        conversations = []
        for session_id in sessions:
            rows = conn.execute(
                """SELECT role, content, timestamp FROM messages
                   WHERE session_id = ? AND content != ''
                   ORDER BY timestamp ASC""",
                (session_id,),
            ).fetchall()
            if not rows:
                continue

            messages = [
                {"role": role, "content": content, "created_at": ts}
                for role, content, ts in rows
            ]
            first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
            title = (first_user[:40] + ("…" if len(first_user) > 40 else "")) or session_id
            last_ts = max((m["created_at"] for m in messages if m["created_at"]), default=time.time())

            conversations.append({
                "id": session_id,
                "session_key": session_id,
                "title": title,
                "created_at": last_ts,
                "messages": messages,
            })

    logger.info("found %d conversations from last %d days", len(conversations), days)
    return conversations


def filter_valuable_messages(messages: list[dict[str, Any]], min_length: int = 50) -> list[dict[str, Any]]:
    """Heuristic: keep user messages that look like knowledge-bearing statements."""
    valuable = []

    def _weighted_len(s: str) -> int:
        # CJK chars carry ~3x the information of an ASCII char
        cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
        return (len(s) - cjk) + cjk * 3

    keywords = [
        # 中文
        "实现", "方案", "问题", "解决", "配置", "部署", "优化", "设计", "架构", "原理",
        # 英文
        "implement", "solution", "issue", "fix", "config", "deploy", "optimize",
        "design", "architecture", "how to",
    ]
    for msg in messages:
        content = msg.get("content", "")
        if _weighted_len(content) < min_length:
            continue
        if msg.get("role") != "user":
            continue
        low = content.lower()
        if any(kw in low for kw in keywords):
            valuable.append(msg)
    return valuable


def extract_chat_knowledge(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for conv in conversations:
        valuable_msgs = filter_valuable_messages(conv["messages"])
        if not valuable_msgs:
            continue

        combined_content = "\n\n".join(
            f"[{_fmt_ts(msg.get('created_at'))}] {msg['content']}"
            for msg in valuable_msgs[:10]
        )

        items.append({
            "type": "chat",
            "source": conv.get("session_key", "unknown"),
            "title": conv.get("title", ""),
            "content": combined_content,
            "date": conv.get("created_at", time.time()),
            "message_count": len(valuable_msgs),
        })

    logger.info("extracted %d chat items", len(items))
    return items


def generate_chat_note(item: dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    date = _fmt_date(item.get("date"))
    title = item.get("title", "Chat")
    source = item.get("source", "unknown")

    content = item.get("content", "")
    lines = [f"> {line}" for line in content.split("\n") if line.strip()]

    note = f"""---
title: {title}
type: chat
source: {source}
date: {date}
category: 技术
concepts: []
tags: [聊天记录, chat-extractor]
---

# {title}

## 对话摘要

{chr(10).join(lines[:20])}

---
*来源: {source}*
*消息数: {item.get("message_count", 0)}*
*提取时间: {now}*
"""
    return note


def process_conversations(conversations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = extract_chat_knowledge(conversations)
    results: list[dict[str, Any]] = []

    state = get_state()
    output_dir = KNOWLEDGE_DIR / "chats"
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in items:
        conv_id = item.get("source", "")
        if state.is_processed(conv_id):
            continue

        note = generate_chat_note(item)
        date = _fmt_date(item.get("date"))
        raw_title = item.get("title", "chat")[:40].strip()
        safe_title = re.sub(r'[<>:"/\\|?*]', "_", raw_title).strip() or "chat"
        output_path = output_dir / f"{date}_{safe_title}.md"
        output_path.write_text(note, encoding="utf-8")

        results.append({"path": str(output_path), "title": item.get("title", "")})
        state.mark_processed(conv_id)
        logger.info("generated chat note: %s", output_path)

    state.last_run = datetime.now().isoformat()
    state.save()
    return results


def _fmt_ts(ts: Any) -> str:
    """Format a unix-float timestamp as local time string."""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return str(ts) if ts else ""


def _fmt_date(ts: Any) -> str:
    """Format a unix-float timestamp as YYYY-MM-DD."""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return datetime.now().strftime("%Y-%m-%d")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Chat knowledge extractor (read-only from L3)")
    parser.add_argument("--days", type=int, default=7, help="Extract conversations from last N days")
    parser.add_argument("--limit", type=int, default=50, help="Max conversations to process")
    args = parser.parse_args()

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    conversations = get_recent_conversations(days=args.days, limit=args.limit)
    results = process_conversations(conversations)

    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
