#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clipper sync - sync clipped content from Obsidian Web Clipper and other tools.

Supported sources:
- Obsidian Web Clipper (default vault/clippings/)
- Xiaohongshu importer
- Manual clips via --clip
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager

logger = setup_logging("clipper_sync")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"

CLIPPER_DIRS = [
    HERMES_HOME / "clippings",
    Path.home() / "Documents" / "Obsidian" / "clippings",
    WIKI_DIR / "clippings",
]


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "clipper")


def find_clipper_dirs() -> list[Path]:
    dirs = []
    for d in CLIPPER_DIRS:
        if d.exists():
            dirs.append(d)
    config_dir = os.getenv("CLIPPER_DIR")
    if config_dir:
        p = Path(config_dir)
        if p.exists():
            dirs.append(p)
    return dirs


def parse_clipper_file(path: Path) -> dict[str, Any] | None:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")

        title = path.stem
        url = ""
        source = "clipper"

        if content.startswith("---"):
            end = content.find("---", 3)
            if end > 0:
                frontmatter = content[3:end].strip()
                for line in frontmatter.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip().lower()
                        value = value.strip().strip('"').strip("'")
                        if key == "title":
                            title = value
                        elif key in ("url", "source", "link"):
                            url = value
                        elif key == "source":
                            source = value
                content = content[end + 3:].strip()

        if not title:
            title = path.stem

        return {
            "type": "clip",
            "path": str(path),
            "title": title,
            "content": content[:10000],
            "url": url,
            "source": source,
            "collected_at": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.warning("failed to parse clip %s: %s", path, exc)
        return None


def collect_from_clipper_dirs() -> list[dict[str, Any]]:
    state = get_state()
    items: list[dict[str, Any]] = []

    for clipper_dir in find_clipper_dirs():
        for md_file in clipper_dir.rglob("*.md"):
            file_id = f"clip:{md_file}"
            if state.is_processed(file_id):
                continue
            item = parse_clipper_file(md_file)
            if item:
                items.append(item)
                state.mark_processed(file_id)

        for html_file in clipper_dir.rglob("*.html"):
            file_id = f"clip:{html_file}"
            if state.is_processed(file_id):
                continue
            try:
                content = html_file.read_text(encoding="utf-8", errors="replace")
                text = re.sub(r"<[^>]+>", " ", content)
                text = re.sub(r"\s+", " ", text).strip()
                items.append({
                    "type": "clip",
                    "path": str(html_file),
                    "title": html_file.stem,
                    "content": text[:10000],
                    "url": "",
                    "source": "clipper",
                    "collected_at": datetime.now().isoformat(),
                })
                state.mark_processed(file_id)
            except Exception as exc:
                logger.warning("failed to read %s: %s", html_file, exc)

    logger.info("found %d clipped items", len(items))
    return items


def save_clip(item: dict[str, Any]) -> Path:
    title = item.get("title", "clip")
    safe_title = re.sub(r'[^\w.-]', '_', title)[:50]
    now = datetime.now().strftime("%Y-%m-%d")

    url_line = f"\n*来源: {item['url']}*" if item.get("url") else ""

    note = f"""---
title: {item.get('title', 'clip')}
type: clip
source: {item.get('source', 'clipper')}
date: {now}
category: 其他
concepts: []
tags: [剪藏]
---

# {item.get('title', 'clip')}

{item.get('content', '')}
{url_line}
---
*剪藏时间: {now}*
"""

    output_dir = KNOWLEDGE_DIR / "clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{safe_title}.md"
    output_path.write_text(note, encoding="utf-8")

    logger.info("saved clip: %s", output_path)
    return output_path


def sync_clips() -> list[dict[str, Any]]:
    items = collect_from_clipper_dirs()
    results: list[dict[str, Any]] = []

    for item in items:
        output_path = save_clip(item)
        results.append({"path": str(output_path), "title": item.get("title", "")})

    state = get_state()
    state.last_run = datetime.now().isoformat()
    return results


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Clipper content sync")
    parser.add_argument("--clip", action="append", help="Manual clip text")
    parser.add_argument("--list-dirs", action="store_true", help="List clipper directories")
    args = parser.parse_args()

    if args.list_dirs:
        for d in find_clipper_dirs():
            print(d)
        return 0

    if args.clip:
        for text in args.clip:
            item = {
                "type": "clip",
                "title": "manual clip",
                "content": text,
                "source": "manual",
                "collected_at": datetime.now().isoformat(),
            }
            output_path = save_clip(item)
            print(json.dumps({"path": str(output_path)}, ensure_ascii=False))

    results = sync_clips()
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
