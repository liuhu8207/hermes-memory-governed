#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Meeting recap - extract and summarize meeting notes from L3 or Feishu.

Features:
- Extract meeting conversations from L3
- Generate structured meeting notes
- Extract action items and decisions
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager, retry

logger = setup_logging("meeting_recap")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"

MEETING_KEYWORDS = ["会议", "讨论", "meeting", "sync", "复盘", "review", "standup", "评审"]


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "meeting")


def get_recent_meetings(days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
    if not L3_DB_PATH.exists():
        return []

    cutoff = time.time() - days * 86400
    with sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True) as conn:
        sessions = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT session_id FROM messages WHERE timestamp > ? "
                "AND session_id != '' ORDER BY timestamp DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
        ]

        meetings = []
        for session_id in sessions:
            rows = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? "
                "AND content != '' ORDER BY timestamp ASC",
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

            is_meeting = any(kw in session_id.lower() for kw in MEETING_KEYWORDS)
            if not is_meeting:
                for m in messages[:5]:
                    if m["content"] and any(kw in m["content"].lower() for kw in MEETING_KEYWORDS):
                        is_meeting = True
                        break
            if not is_meeting:
                continue

            meetings.append({
                "id": session_id,
                "session_key": session_id,
                "title": title,
                "created_at": last_ts,
                "messages": messages,
            })

    logger.info("found %d meetings from last %d days", len(meetings), days)
    return meetings


def generate_recap_prompt(meeting: dict[str, Any]) -> str:
    messages_text = []
    for msg in meeting["messages"][:50]:
        role = msg.get("role", "unknown")
        content = (msg.get("content", "") or "")[:500]
        if content.strip():
            messages_text.append(f"[{role}]: {content}")

    conversation = "\n".join(messages_text)

    return f"""请根据以下会议对话生成结构化的会议纪要。

会议标题: {meeting.get('title', 'Untitled')}
会议时间: {meeting.get('created_at', 'Unknown')}

对话内容:
{conversation}

请返回 JSON 格式:
{{
  "title": "会议标题",
  "summary": "200字以内的会议摘要",
  "decisions": ["决策1", "决策2"],
  "action_items": [
    {{"owner": "负责人", "task": "任务描述", "deadline": "截止日期"}}
  ],
  "key_points": ["要点1", "要点2"],
  "participants": ["参与者1", "参与者2"]
}}"""


@retry(max_attempts=3, delay=2.0)
def call_llm(prompt: str) -> str:
    api_key = os.getenv("LLM_API_KEY") or os.getenv("PERSONA_LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("PERSONA_LLM_BASE_URL")
    model = os.getenv("LLM_MODEL") or os.getenv("PERSONA_LLM_MODEL")
    if not api_key or not base_url or not model:
        return ""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1500,
        "temperature": 0.3,
    }
    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


def parse_llm_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return {
            "title": "",
            "summary": text[:200],
            "decisions": [],
            "action_items": [],
            "key_points": [],
            "participants": [],
        }


def generate_recap_basic(meeting: dict[str, Any]) -> dict[str, Any]:
    messages = meeting.get("messages", [])
    user_msgs = [m for m in messages if m.get("role") == "user"]

    summary = ""
    for msg in user_msgs[:3]:
        content = msg.get("content", "")
        if content and len(content) > 50:
            summary = content[:300]
            break

    participants = []
    for msg in messages:
        role = msg.get("role", "")
        if role and role not in participants:
            participants.append(role)

    return {
        "title": meeting.get("title", ""),
        "summary": summary or "无摘要",
        "decisions": [],
        "action_items": [],
        "key_points": [],
        "participants": participants,
    }


def generate_meeting_note(recap: dict[str, Any], meeting: dict[str, Any]) -> str:
    now = datetime.now().strftime("%Y-%m-%d")
    try:
        date = datetime.fromtimestamp(float(meeting.get("created_at", time.time()))).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        date = now
    title = recap.get("title") or meeting.get("title", "Meeting")

    decisions = "\n".join(f"- {d}" for d in recap.get("decisions", []))
    action_items = "\n".join(
        f"- [ ] {item.get('task', '')} (@{item.get('owner', '')}) {item.get('deadline', '')}"
        for item in recap.get("action_items", [])
    )
    key_points = "\n".join(f"- {p}" for p in recap.get("key_points", []))
    participants = ", ".join(recap.get("participants", []))

    note = f"""---
title: {title}
type: meeting
source: l3
date: {now}
category: 工作
concepts: []
tags: [会议, 复盘]
---

# {title}

**日期**: {date}
**参与者**: {participants}

## 摘要
{recap.get('summary', '')}

## 决策
{decisions or '- 无'}

## 待办事项
{action_items or '- 无'}

## 关键要点
{key_points or '- 无'}

---
*来源: L3 会话归档*
*处理时间: {now}*
"""
    return note


def process_meeting(meeting: dict[str, Any]) -> dict[str, Any] | None:
    state = get_state()
    conv_id = meeting.get("id", "")
    if state.is_processed(conv_id):
        return None

    prompt = generate_recap_prompt(meeting)
    response = call_llm(prompt)

    if response:
        recap = parse_llm_response(response)
    else:
        recap = generate_recap_basic(meeting)

    if not recap.get("title"):
        recap["title"] = meeting.get("title", "Meeting")

    note = generate_meeting_note(recap, meeting)

    output_dir = KNOWLEDGE_DIR / "meetings"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        date = datetime.fromtimestamp(float(meeting.get("created_at", time.time()))).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        date = datetime.now().strftime("%Y-%m-%d")
    raw_title = str(recap.get("title", "meeting"))[:40].strip()
    safe_title = re.sub(r'[<>:"/\\|?*]', "_", raw_title).strip() or "meeting"
    output_path = output_dir / f"{date}_{safe_title}.md"
    output_path.write_text(note, encoding="utf-8")

    state.mark_processed(conv_id)
    logger.info("generated meeting recap: %s", output_path)
    return {"path": str(output_path), "title": recap.get("title", "")}


def process_all_meetings(days: int = 7) -> list[dict[str, Any]]:
    meetings = get_recent_meetings(days=days)
    results: list[dict[str, Any]] = []

    for meeting in meetings:
        result = process_meeting(meeting)
        if result:
            results.append(result)

    state = get_state()
    state.last_run = datetime.now().isoformat()
    return results


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Meeting recap generator")
    parser.add_argument("--days", type=int, default=7, help="Process meetings from last N days")
    parser.add_argument("--limit", type=int, default=20, help="Max meetings to process")
    args = parser.parse_args()

    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    results = process_all_meetings(days=args.days)

    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
