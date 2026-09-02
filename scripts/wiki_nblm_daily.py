#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Incrementally deepen wiki pages with NotebookLM and write results to L2.

OPTIONAL: Requires GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS to be set.
When not configured, this script is a no-op.

This is a portable template. The NotebookLM client API differs across local
setups, so non-dry-run mode expects a compatible `notebooklm` package or a
small adapter with the methods used in `process_page`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_int, env_path, load_config, setup_logging
from wiki_utils import atomic_write_json, extract_summary, slugify

logger = setup_logging("wiki_nblm_daily")

SCRIPT_DIR = Path(__file__).resolve().parent
HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")


def is_available() -> bool:
    """Check if NotebookLM is configured. Requires Google API credentials."""
    return bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    )
STATE_FILE = env_path("NBLM_STATE_PATH", SCRIPT_DIR / "wiki-nblm-state.json")
MAX_PER_RUN = env_int("NBLM_MAX_PER_RUN", 2)

sys.path.insert(0, str(MEMORY_DIR))
sys.path.insert(0, str(HERMES_HOME))
sys.path.insert(0, str(HERMES_HOME / "hermes-agent"))


def load_state() -> dict[str, Any]:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8", errors="replace"))
    return {"processed": []}


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(STATE_FILE, state)


def get_all_wiki_pages() -> list[Path]:
    pages: list[Path] = []
    if not WIKI_DIR.exists():
        return pages
    for path in WIKI_DIR.rglob("*.md"):
        parts = {part.lower() for part in path.parts}
        if ("memory" in parts and "by-date" in parts) or path.name in {"index.md", "log.md", "SCHEMA.md"}:
            continue
        pages.append(path)
    return sorted(pages, key=lambda item: item.stat().st_mtime, reverse=True)


def extract_title(content: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    return match.group(1).strip() if match else fallback


async def process_page(client: Any, notebook_id: str, page_path: Path) -> dict[str, str] | None:
    from notebooklm import ReportFormat  # type: ignore

    content = page_path.read_text(encoding="utf-8", errors="replace")
    title = extract_title(content, page_path.stem)
    await client.sources.add_text(notebook_id=notebook_id, title=title[:100], content=content, wait=True)
    status = await client.artifacts.generate_report(
        notebook_id=notebook_id,
        report_format=ReportFormat.BRIEFING_DOC,
        language=os.getenv("NBLM_LANGUAGE", "zh"),
    )
    final = await client.artifacts.wait_for_completion(notebook_id, status.task_id, timeout=int(os.getenv("NBLM_TIMEOUT", "180")))
    if not final.is_complete:
        return None

    artifacts = await client.artifacts.list(notebook_id)
    report_art = next((item for item in artifacts if item.kind == "report" and item.is_completed), None)
    if not report_art:
        return None

    output_dir = env_path("NBLM_OUTPUT_DIR", HERMES_HOME / "cron" / "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"wiki_rpt_{slugify(title)}.md"
    downloaded = await client.artifacts.download_report(
        notebook_id=notebook_id,
        artifact_id=report_art.id,
        output_path=str(output_path),
    )
    report_content = Path(downloaded).read_text(encoding="utf-8", errors="replace")
    return {
        "title": title,
        "wiki_path": str(page_path),
        "summary": extract_summary(report_content),
        "report_content": report_content,
    }


def add_to_l2(result: dict[str, str]) -> dict[str, Any]:
    from plugins.memory.holographic.store import MemoryStore  # type: ignore

    l2_config = load_config(HERMES_HOME)
    l2_text = (
        f"【Wiki深化】{result['title']}\n\n"
        f"原始：{result['wiki_path']}\n"
        f"摘要：{result['summary']}\n\n"
        f"--- NotebookLM 报告 ---\n"
        f"{result['report_content'][:5000]}"
    )
    store = MemoryStore(db_path=str(l2_config.db_path))
    fact_id = store.add_fact(
        content=l2_text,
        category="wiki_deepened",
        tags=f"notebooklm,{result['title']}",
    )
    return {"fact_id": fact_id, "category": "wiki_deepened"}


async def main(dry_run: bool = False, max_per_run: int = MAX_PER_RUN) -> int:
    if not is_available():
        logger.info("NotebookLM not configured (no GOOGLE_API_KEY or GOOGLE_APPLICATION_CREDENTIALS), skipping")
        return 0

    state = load_state()
    processed_set = set(state.get("processed") or [])
    all_pages = get_all_wiki_pages()
    pending = [page for page in all_pages if str(page) not in processed_set]

    if not pending:
        logger.info("[%s] all %d pages processed", datetime.now().date(), len(processed_set))
        return 0

    to_process = pending[-max_per_run:]
    logger.info("pending=%d this_run=%d", len(pending), len(to_process))
    for page in to_process:
        label = page.relative_to(WIKI_DIR) if page.is_relative_to(WIKI_DIR) else page
        logger.info("  - %s", label)

    if dry_run:
        logger.info("dry_run=True; queue checked only, NotebookLM and L2 were not called")
        return 0

    try:
        from notebooklm import NotebookLMClient  # type: ignore
    except Exception as exc:
        raise RuntimeError("NotebookLM client is unavailable. Install/provide an adapter, or use --dry-run.") from exc

    results: list[dict[str, str]] = []
    storage_home = os.getenv("NOTEBOOKLM_HOME")
    from_storage = NotebookLMClient.from_storage
    try:
        client_context = await (from_storage(storage_home) if storage_home else from_storage())
    except TypeError:
        client_context = await from_storage()
    async with client_context as client:
        notebook = await client.notebooks.create(os.getenv("NBLM_NOTEBOOK_NAME", "Wiki daily deepening"))
        notebook_id = notebook.id
        try:
            for page_path in to_process:
                logger.info("processing=%s", page_path.name)
                result = await process_page(client, notebook_id, page_path)
                if not result:
                    logger.warning("failed=%s", page_path.name)
                    continue
                entry = add_to_l2(result)
                processed_set.add(str(page_path))
                state["processed"] = sorted(processed_set)
                save_state(state)
                logger.info("ok=%s l2=%s", page_path.name, str(entry.get("id", ""))[:8])
                results.append(result)
        finally:
            await client.notebooks.delete(notebook_id)

    logger.info("completed=%d/%d total_processed=%d/%d", len(results), len(to_process), len(processed_set), len(all_pages))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only show pending pages")
    parser.add_argument("--max-per-run", type=int, default=MAX_PER_RUN)
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(dry_run=args.dry_run, max_per_run=args.max_per_run)))
