#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Knowledge graph builder for Obsidian visualization."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import atomic_write_json, load_log, save_log

logger = setup_logging("knowledge_graph")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"
GRAPH_FILE = KNOWLEDGE_DIR / "graph.json"
STATE_FILE = KNOWLEDGE_DIR / ".graph_state.json"


def load_state() -> dict[str, Any]:
    return load_log(STATE_FILE)


def save_state(state: dict[str, Any]) -> None:
    atomic_write_json(STATE_FILE, state)


def parse_frontmatter(content: str) -> dict[str, Any]:
    match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
    if not match:
        return {}
    meta: dict[str, Any] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = [v.strip().strip('"').strip("'") for v in value[1:-1].split(",") if v.strip()]
        meta[key] = value
    return meta


def scan_notes() -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for subdir in ["documents", "voices", "web", "chats", "inductions", "concepts"]:
        dir_path = KNOWLEDGE_DIR / subdir
        if not dir_path.exists():
            continue
        for md_file in dir_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                meta = parse_frontmatter(content)
                notes.append({
                    "path": str(md_file.relative_to(WIKI_DIR)),
                    "name": md_file.stem,
                    "type": meta.get("type", subdir),
                    "category": meta.get("category", "其他"),
                    "concepts": meta.get("concepts", []) if isinstance(meta.get("concepts"), list) else [],
                    "tags": meta.get("tags", []) if isinstance(meta.get("tags"), list) else [],
                })
            except Exception as exc:
                logger.warning("failed to parse %s: %s", md_file, exc)
    logger.info("scanned %d notes", len(notes))
    return notes


def build_graph(notes: list[dict[str, Any]]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    for note in notes:
        node_id = note["name"]
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "path": note["path"],
                "type": note["type"],
                "category": note["category"],
                "size": 1,
            }
        else:
            nodes[node_id]["size"] += 1

        for concept in note["concepts"]:
            concept_name = concept.strip("[]")
            if concept_name not in nodes:
                nodes[concept_name] = {
                    "id": concept_name,
                    "type": "concept",
                    "category": "",
                    "size": 0,
                }
            nodes[concept_name]["size"] += 1
            edges.append({"source": node_id, "target": concept_name, "type": "concept"})

        for tag in note["tags"]:
            tag_name = tag.strip("#")
            if tag_name not in nodes:
                nodes[tag_name] = {
                    "id": tag_name,
                    "type": "tag",
                    "category": "",
                    "size": 0,
                }
            nodes[tag_name]["size"] += 1
            edges.append({"source": node_id, "target": tag_name, "type": "tag"})

    concept_cooccurrence: dict[tuple[str, str], int] = defaultdict(int)
    for note in notes:
        concepts = [c.strip("[]") for c in note["concepts"]]
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                pair = tuple(sorted([c1, c2]))
                concept_cooccurrence[pair] += 1

    for (c1, c2), weight in concept_cooccurrence.items():
        if weight >= 2:
            edges.append({"source": c1, "target": c2, "type": "related", "weight": weight})

    graph = {
        "nodes": list(nodes.values()),
        "edges": edges,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_notes": len(notes),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
    }

    logger.info("built graph: %d nodes, %d edges", len(nodes), len(edges))
    return graph


def generate_concept_pages(notes: list[dict[str, Any]], graph: dict[str, Any]) -> None:
    concept_dir = KNOWLEDGE_DIR / "concepts"
    concept_dir.mkdir(parents=True, exist_ok=True)

    concept_notes: dict[str, list[str]] = defaultdict(list)
    for note in notes:
        for concept in note["concepts"]:
            concept_name = concept.strip("[]")
            concept_notes[concept_name].append(note["name"])

    for concept_name, related_notes in concept_notes.items():
        safe_name = re.sub(r'[^\w.-]', '_', concept_name)[:50]
        note_path = concept_dir / f"{safe_name}.md"

        related_list = "\n".join(f"- [[{n}]]" for n in related_notes[:20])

        note_content = f"""---
title: {concept_name}
type: concept
created: {datetime.now().strftime('%Y-%m-%d')}
---

# {concept_name}

## 相关笔记

{related_list}

---
*自动更新于 {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
        note_path.write_text(note_content, encoding="utf-8")

    logger.info("generated %d concept pages", len(concept_notes))


def generate_category_index(notes: list[dict[str, Any]]) -> None:
    category_dir = KNOWLEDGE_DIR / "categories"
    category_dir.mkdir(parents=True, exist_ok=True)

    category_notes: dict[str, list[str]] = defaultdict(list)
    for note in notes:
        category = note.get("category", "其他")
        category_notes[category].append(note["name"])

    for category, note_names in category_notes.items():
        safe_name = re.sub(r'[^\w.-]', '_', category)[:30]
        index_path = category_dir / f"_{safe_name}.md"

        notes_list = "\n".join(f"- [[{n}]]" for n in note_names[:50])

        index_content = f"""---
title: {category}
type: category
---

# {category}

## 笔记列表

{notes_list}

---
*共 {len(note_names)} 篇笔记*
"""
        index_path.write_text(index_content, encoding="utf-8")

    logger.info("generated %d category indexes", len(category_notes))


def generate_moc(notes: list[dict[str, Any]], graph: dict[str, Any]) -> None:
    moc_path = KNOWLEDGE_DIR / "_MOC.md"

    by_type: dict[str, list[str]] = defaultdict(list)
    for note in notes:
        by_type[note.get("type", "other")].append(note["name"])

    type_sections = []
    for type_name, note_names in by_type.items():
        type_list = "\n".join(f"- [[{n}]]" for n in note_names[:30])
        type_sections.append(f"## {type_name}\n\n{type_list}")

    categories = set(n.get("category", "") for n in notes if n.get("category"))
    category_links = " ".join(f"[[_{c}]]" for c in sorted(categories))

    moc_content = f"""---
title: Knowledge Map of Content
type: moc
---

# Knowledge Map of Content

## 分类索引

{category_links}

## 统计

- 总笔记数: {len(notes)}
- 总节点数: {graph['metadata']['total_nodes']}
- 总边数: {graph['metadata']['total_edges']}
- 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}

{chr(10).join(type_sections)}
"""
    moc_path.write_text(moc_content, encoding="utf-8")
    logger.info("generated MOC: %s", moc_path)


def build_all() -> dict[str, Any]:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    notes = scan_notes()
    graph = build_graph(notes)

    atomic_write_json(GRAPH_FILE, graph)
    generate_concept_pages(notes, graph)
    generate_category_index(notes)
    generate_moc(notes, graph)

    state = load_state()
    state["last_run"] = datetime.now().isoformat()
    state["notes_count"] = len(notes)
    save_state(state)

    return graph["metadata"]


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge graph builder")
    parser.add_argument("--scan-only", action="store_true", help="Only scan and print stats")
    args = parser.parse_args()

    if args.scan_only:
        notes = scan_notes()
        print(json.dumps({"notes": len(notes)}, ensure_ascii=False))
        return 0

    metadata = build_all()
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
