#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build Obsidian-compatible backlinks across knowledge notes.

Scans $WIKI_DIR/knowledge/**/*.md, parses frontmatter (title/tags/concepts),
and rewrites the "## 相关笔记" (Related Notes) section of every note with
[[wikilinks]] to notes sharing tags.

- Obsidian: [[name]] resolves to links and powers the backlinks panel.
- Other editors: [[name]] is plain text — the notes stay fully readable.

Idempotent: existing "## 相关笔记" sections are replaced on each run.
Run as the last step of knowledge_sync (or standalone).

Usage:
    python scripts/knowledge_links.py            # rebuild all links
    python scripts/knowledge_links.py --dry-run  # preview without writing
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("knowledge_links")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"

RELATED_SECTION = "## 相关笔记"
MAX_RELATED = 5


def parse_frontmatter(text: str) -> dict[str, Any]:
    """Lightweight frontmatter parser (no yaml dependency)."""
    meta: dict[str, Any] = {}
    if not text.startswith("---"):
        return meta
    end = text.find("\n---", 3)
    if end < 0:
        return meta
    block = text[3:end]
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
            meta[key] = items
        else:
            meta[key] = value.strip("'\"")
    return meta


def scan_notes(knowledge_dir: Path) -> list[dict[str, Any]]:
    """Scan all notes: path, title (file stem), tags, concepts."""
    notes = []
    if not knowledge_dir.exists():
        return notes
    for md in sorted(knowledge_dir.rglob("*.md")):
        if md.name.startswith("."):
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        meta = parse_frontmatter(text)
        notes.append({
            "path": md,
            "stem": md.stem,  # filename without extension → Obsidian wikilink target
            "title": meta.get("title") or md.stem,
            "tags": set(meta.get("tags", [])),
            "concepts": set(meta.get("concepts", [])),
        })
    return notes


def related_links(note: dict[str, Any], all_notes: list[dict[str, Any]]) -> list[str]:
    """Find notes sharing tags/concepts with this note (excluding self)."""
    own_sig = note["tags"] | note["concepts"]
    scored: list[tuple[int, dict[str, Any]]] = []
    for other in all_notes:
        if other["path"] == note["path"]:
            continue
        shared = len(own_sig & (other["tags"] | other["concepts"]))
        if shared > 0:
            scored.append((shared, other))
    scored.sort(key=lambda x: (-x[0], x[1]["title"]))
    links = []
    for _, other in scored[:MAX_RELATED]:
        # [[stem|title]] — Obsidian resolves by filename, alias shows title.
        # Trim the target: trailing spaces in filenames break wikilink resolution.
        stem = other["stem"].strip()
        alias = other["title"].strip()
        if alias == stem:
            links.append(f"[[{stem}]]")
        else:
            links.append(f"[[{stem}|{alias}]]")
    return links


def rewrite_related_section(note: dict[str, Any], links: list[str]) -> bool:
    """Replace the '## 相关笔记' section (or append it). Idempotent."""
    path = note["path"]
    text = path.read_text(encoding="utf-8", errors="replace")
    section = RELATED_SECTION + "\n"
    if links:
        section += "\n".join(f"- {l}" for l in links) + "\n"
    else:
        section += "- 暂无\n"

    pattern = re.compile(r"^## 相关笔记.*$", re.M | re.S)
    if pattern.search(text):
        # Replace from the section header to the end (it is the last section)
        new_text = pattern.sub(section.rstrip("\n"), text.rstrip("\n")) + "\n"
    else:
        new_text = text.rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n"

    if new_text == text:
        return False
    path.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Obsidian-compatible related links")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    notes = scan_notes(KNOWLEDGE_DIR)
    logger.info("scanned %d notes in %s", len(notes), KNOWLEDGE_DIR)
    if not notes:
        print("no notes found — nothing to link")
        return 0

    changed = 0
    for note in notes:
        links = related_links(note, notes)
        if args.dry_run:
            print(f"  {note['path'].name}: {len(links)} related -> {links[:3]}")
            continue
        if rewrite_related_section(note, links):
            changed += 1
            logger.info("linked %s -> %d notes", note["path"].name, len(links))

    if not args.dry_run:
        logger.info("linked %d notes", changed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
