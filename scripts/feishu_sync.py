#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feishu document sync - import documents from Feishu to knowledge base.

Features:
- Import Feishu documents as markdown
- Support Feishu wiki spaces
- Auto-categorize imported documents
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager, atomic_write_json, retry

logger = setup_logging("feishu_sync")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"
STATE_FILE = KNOWLEDGE_DIR / ".feishu_state.json"

FEISHU_API_BASE = "https://open.feishu.cn/open-apis"


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "feishu")


def get_tenant_access_token(app_id: str, app_secret: str) -> str | None:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{FEISHU_API_BASE}/auth/v3/tenant_access_token/internal",
                json={"app_id": app_id, "app_secret": app_secret},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                return data.get("tenant_access_token")
    except Exception as exc:
        logger.warning("failed to get tenant access token: %s", exc)
    return None


@retry(max_attempts=3, delay=2.0)
def get_document_content(token: str, document_id: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{FEISHU_API_BASE}/docx/v1/documents/{document_id}/blocks",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": 500},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                return data.get("data", {})
    except Exception as exc:
        logger.warning("failed to get document %s: %s", document_id, exc)
    return None


def get_wiki_spaces(token: str) -> list[dict[str, Any]]:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{FEISHU_API_BASE}/wiki/v2/spaces",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": 50},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                return data.get("data", {}).get("items", [])
    except Exception as exc:
        logger.warning("failed to get wiki spaces: %s", exc)
    return []


def get_wiki_nodes(token: str, space_id: str) -> list[dict[str, Any]]:
    try:
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"{FEISHU_API_BASE}/wiki/v2/spaces/{space_id}/nodes",
                headers={"Authorization": f"Bearer {token}"},
                params={"page_size": 50},
            )
            response.raise_for_status()
            data = response.json()
            if data.get("code") == 0:
                return data.get("data", {}).get("items", [])
    except Exception as exc:
        logger.warning("failed to get wiki nodes: %s", exc)
    return []


def blocks_to_markdown(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        block_type = block.get("block_type")
        content = block.get("text", {})
        elements = content.get("elements", [])

        text = ""
        for elem in elements:
            text_run = elem.get("text_run", {})
            if text_run:
                text += text_run.get("content", "")
            mention = elem.get("mention", {})
            if mention:
                text += f"@{mention.get('name', 'user')}"

        if block_type == 2:
            lines.append(text)
        elif block_type == 3:
            lines.append(f"# {text}")
        elif block_type == 4:
            lines.append(f"## {text}")
        elif block_type == 5:
            lines.append(f"### {text}")
        elif block_type == 12:
            lines.append(f"- {text}")
        elif block_type == 15:
            lines.append(f"```")
            lines.append(text)
            lines.append("```")
        elif text:
            lines.append(text)

    return "\n\n".join(lines)


def save_document(doc: dict[str, Any], content: str) -> Path:
    title = doc.get("title", "Untitled")
    doc_id = doc.get("document_id", "unknown")
    now = datetime.now().strftime("%Y-%m-%d")

    safe_title = re.sub(r'[^\w.-]', '_', title)[:50]

    note = f"""---
title: {title}
type: feishu
source: feishu
document_id: {doc_id}
date: {now}
category: 工作
concepts: []
tags: [飞书, 文档]
---

# {title}

{content}

---
*来源: 飞书文档*
*导入时间: {now}*
"""

    output_dir = KNOWLEDGE_DIR / "feishu"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_title}.md"
    output_path.write_text(note, encoding="utf-8")

    logger.info("saved document: %s", output_path)
    return output_path


def sync_documents(token: str, document_ids: list[str]) -> list[dict[str, Any]]:
    state = get_state()
    results: list[dict[str, Any]] = []

    for doc_id in document_ids:
        if state.is_processed(doc_id):
            continue

        doc_data = get_document_content(token, doc_id)
        if not doc_data:
            continue

        blocks = doc_data.get("items", [])
        content = blocks_to_markdown(blocks)

        doc_title = content.split("\n")[0].strip("# ") if content else doc_id
        doc = {"document_id": doc_id, "title": doc_title}

        output_path = save_document(doc, content)
        state.mark_processed(doc_id)
        results.append({"path": str(output_path), "title": doc_title})

    state.last_run = datetime.now().isoformat()
    return results


def sync_wiki_space(token: str, space_id: str) -> list[dict[str, Any]]:
    nodes = get_wiki_nodes(token, space_id)
    document_ids = []
    for node in nodes:
        obj_type = node.get("obj_type")
        obj_token = node.get("obj_token")
        if obj_type == "doc" and obj_token:
            document_ids.append(obj_token)
    return sync_documents(token, document_ids)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Feishu document sync")
    parser.add_argument("--app-id", help="Feishu App ID")
    parser.add_argument("--app-secret", help="Feishu App Secret")
    parser.add_argument("--document", action="append", help="Document ID to sync")
    parser.add_argument("--space-id", help="Wiki space ID to sync")
    parser.add_argument("--list-spaces", action="store_true", help="List wiki spaces")
    args = parser.parse_args()

    app_id = args.app_id or os.getenv("FEISHU_APP_ID", "")
    app_secret = args.app_secret or os.getenv("FEISHU_APP_SECRET", "")

    if not app_id or not app_secret:
        logger.error("FEISHU_APP_ID and FEISHU_APP_SECRET required")
        print("Error: Set FEISHU_APP_ID and FEISHU_APP_SECRET in .env or use --app-id/--app-secret")
        return 1

    token = get_tenant_access_token(app_id, app_secret)
    if not token:
        logger.error("failed to get access token")
        return 1

    if args.list_spaces:
        spaces = get_wiki_spaces(token)
        for space in spaces:
            print(json.dumps({"id": space.get("space_id"), "name": space.get("name")}, ensure_ascii=False))
        return 0

    results: list[dict[str, Any]] = []
    if args.document:
        results.extend(sync_documents(token, args.document))
    if args.space_id:
        results.extend(sync_wiki_space(token, args.space_id))

    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
