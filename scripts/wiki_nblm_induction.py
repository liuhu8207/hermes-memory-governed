#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Induce multi-page wiki topics with NotebookLM and write to L2.

OPTIONAL: Requires GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS to be set.
When not configured, this script is a no-op.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_int, env_path, load_config, setup_logging
from wiki_utils import (
    async_retry,
    collect_pages,
    extract_summary,
    find_candidates,
    load_log,
    save_log,
    slugify,
)

logger = setup_logging("wiki_nblm_induction")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")


def is_available() -> bool:
    """Check if NotebookLM is configured. Requires Google API credentials."""
    return bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
OUT_DIR = env_path("NBLM_INDUCTION_OUTPUT_DIR", HERMES_HOME / "cron" / "output")
INDUCTION_LOG = env_path("NBLM_INDUCTION_LOG", OUT_DIR / "nblm_induction_log.json")
LINGER_DAYS = env_int("NBLM_INDUCTION_LINGER_DAYS", 14)
MAX_TOPICS_PER_RUN = env_int("NBLM_INDUCTION_MAX_TOPICS_PER_RUN", 1)
MAX_SOURCES_PER_TOPIC = env_int("NBLM_INDUCTION_MAX_SOURCES_PER_TOPIC", 12)

THRESHOLDS = {
    "tag": env_int("NBLM_INDUCTION_TAG_THRESHOLD", 3),
    "concept": env_int("NBLM_INDUCTION_CONCEPT_THRESHOLD", 4),
    "entity": env_int("NBLM_INDUCTION_ENTITY_THRESHOLD", 4),
}

sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(HERMES_HOME))
sys.path.insert(0, str(HERMES_HOME / "hermes-agent"))


def build_instruction_source(kind: str, topic: str, pages: list[dict[str, Any]]) -> str:
    sources = "\n".join(f"- {page['title']}: {page['path']}" for page in pages)
    return f"""# 归纳任务：{topic}

请基于本 notebook 中的 wiki sources，生成一篇中文综合归纳报告。

要求：
1. 只基于 sources，不编造未出现的信息。
2. 先给出执行摘要，再按主题分节。
3. 提炼跨页面共性、差异、演化线索和可执行结论。
4. 保留关键技术细节、决策背景、约束和风险。
5. 文末列出来源标题，方便回溯。

topic_kind: {kind}
topic: {topic}

候选来源：
{sources}
"""


async def open_notebooklm_client() -> Any:
    try:
        from notebooklm import NotebookLMClient  # type: ignore
    except Exception as exc:
        raise RuntimeError("NotebookLM client is unavailable. Run install.ps1 -InstallNotebookLM and log in first.") from exc

    storage_home = os.getenv("NOTEBOOKLM_HOME")
    from_storage = NotebookLMClient.from_storage
    try:
        return await (from_storage(storage_home) if storage_home else from_storage())
    except TypeError:
        return await from_storage()


async def generate_notebooklm_report(client: Any, kind: str, topic: str, pages: list[dict[str, Any]]) -> dict[str, str] | None:
    from notebooklm import ReportFormat  # type: ignore

    selected = sorted(pages, key=lambda page: page["mtime"], reverse=True)[:MAX_SOURCES_PER_TOPIC]
    notebook = await client.notebooks.create(f"Wiki induction - {topic}"[:100])
    notebook_id = notebook.id
    try:
        await client.sources.add_text(
            notebook_id=notebook_id,
            title=f"归纳任务 - {topic}"[:100],
            content=build_instruction_source(kind, topic, selected),
            wait=True,
        )
        for page in selected:
            content = f"# {page['title']}\n\nsource_path: {page['path']}\n\n{page['body']}"
            await client.sources.add_text(notebook_id=notebook_id, title=page["title"][:100], content=content, wait=True)

        format_name = os.getenv("NBLM_INDUCTION_REPORT_FORMAT", "BRIEFING_DOC").upper()
        report_format = getattr(ReportFormat, format_name, ReportFormat.BRIEFING_DOC)
        status = await client.artifacts.generate_report(
            notebook_id=notebook_id,
            report_format=report_format,
            language=os.getenv("NBLM_LANGUAGE", "zh"),
        )
        timeout = int(os.getenv("NBLM_INDUCTION_TIMEOUT", os.getenv("NBLM_TIMEOUT", "300")))
        final = await client.artifacts.wait_for_completion(notebook_id, status.task_id, timeout=timeout)
        if not final.is_complete:
            return None

        artifacts = await client.artifacts.list(notebook_id)
        report_art = next((item for item in reversed(artifacts) if item.kind == "report" and item.is_completed), None)
        if not report_art:
            return None

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUT_DIR / f"nblm_induction_{slugify(topic)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        downloaded = await client.artifacts.download_report(
            notebook_id=notebook_id,
            artifact_id=report_art.id,
            output_path=str(output_path),
        )
        report_content = Path(downloaded).read_text(encoding="utf-8", errors="replace")
        return {
            "title": f"{topic} 综合归纳",
            "topic": topic,
            "kind": kind,
            "summary": extract_summary(report_content),
            "report_content": report_content,
            "output_file": str(output_path),
            "source_paths": json.dumps([page["path"] for page in selected], ensure_ascii=False),
        }
    finally:
        await client.notebooks.delete(notebook_id)


def add_induction_to_l2(result: dict[str, str]) -> dict[str, Any]:
    from plugins.memory.holographic.store import MemoryStore  # type: ignore

    l2_config = load_config(HERMES_HOME)
    l2_text = (
        f"【NotebookLM多页归纳】{result['title']}\n\n"
        f"主题：{result['topic']}\n"
        f"来源页面：{result['source_paths']}\n"
        f"输出文件：{result['output_file']}\n"
        f"摘要：{result['summary']}\n\n"
        f"--- NotebookLM 归纳报告 ---\n"
        f"{result['report_content'][:7000]}"
    )
    store = MemoryStore(db_path=str(l2_config.db_path))
    fact_id = store.add_fact(
        content=l2_text,
        category="wiki_induction",
        tags=f"notebooklm,{result['kind']},{result['topic']}",
    )
    return {"fact_id": fact_id, "category": "wiki_induction"}


async def main(dry_run: bool = False, max_topics: int = MAX_TOPICS_PER_RUN) -> int:
    if not is_available():
        logger.info("NotebookLM not configured (no GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS), skipping")
        return 0

    pages = collect_pages(WIKI_DIR, max_body_length=12000)
    logger.info("pages=%d", len(pages))
    candidates = find_candidates(pages, THRESHOLDS)
    if not candidates:
        logger.info("no candidate topic reached thresholds")
        return 0

    log = load_log(INDUCTION_LOG)
    cutoff = (datetime.now() - timedelta(days=LINGER_DAYS)).timestamp()
    pending = [(kind, topic, items) for kind, topic, items in candidates if log.get(f"{kind}:{topic}", 0) <= cutoff]
    logger.info("candidates=%s", [(kind, topic, len(items)) for kind, topic, items in candidates])
    logger.info("pending_topics=%d", len(pending))
    for kind, topic, items in pending[:max_topics]:
        logger.info("  - [%s] %s pages=%d", kind, topic, len(items))

    if dry_run:
        logger.info("dry_run=True; NotebookLM/L2 were not called")
        return 0
    if not pending:
        logger.info("all candidate topics were processed recently")
        return 0

    completed: list[str] = []
    client_context = await open_notebooklm_client()
    async with client_context as client:
        for kind, topic, matched_pages in pending[:max_topics]:
            key = f"{kind}:{topic}"
            logger.info("processing=%s pages=%d", key, len(matched_pages))

            @async_retry(max_attempts=2, delay=3.0)
            async def _generate() -> dict[str, str] | None:
                return await generate_notebooklm_report(client, kind, topic, matched_pages)

            try:
                result = await _generate()
            except Exception as exc:
                logger.error("notebooklm_failed=%s error=%s", key, exc)
                continue
            if not result:
                logger.warning("notebooklm_failed=%s", key)
                continue
            try:
                entry = add_induction_to_l2(result)
                logger.info("l2_id=%s category=wiki_induction", str(entry.get("id", ""))[:8])
            except Exception as exc:
                logger.error("l2_write_failed=%s", exc)
                continue
            log[key] = time.time()
            completed.append(key)
            save_log(INDUCTION_LOG, log)

    logger.info("completed=%d topics=%s", len(completed), completed)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only scan candidate topics")
    parser.add_argument("--max-topics", type=int, default=MAX_TOPICS_PER_RUN)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(dry_run=args.dry_run, max_topics=args.max_topics)))
