#!/usr/bin/env python3
"""Audit KB wikilinks without misclassifying concept placeholders as failures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, resolve_repo

HERMES_HOME = bootstrap(__file__)
REPO = resolve_repo(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def audit(vault: Path) -> dict:
    from plugin.memory_governed._kb import KnowledgeBase, _LINK_RE
    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed import _vault

    cfg = load_governed_config(HERMES_HOME)
    cfg.wiki_dir = str(vault)
    kb = KnowledgeBase(cfg)
    notes = []
    for rel, path in kb.scan_vault_paths().items():
        meta, body = _vault.parse_frontmatter(
            path.read_text(encoding="utf-8", errors="replace"))
        notes.append({"rel": rel, "stem": path.stem, "meta": meta, "body": body})

    valid = set()
    concepts = set()
    for note in notes:
        stem = note["stem"]
        valid.add(stem.casefold())
        valid.add(str(note["meta"].get("title") or stem).casefold())
        valid.update(str(x).casefold() for x in (note["meta"].get("aliases") or []))
        concepts.update(str(x).casefold() for x in (note["meta"].get("concepts") or []))

    live = [n for n in notes
            if str(n["meta"].get("status") or "").casefold() != "archived"
            and n["stem"] != "index"]
    incoming = defaultdict(set)
    outgoing = {}
    unresolved = []
    for note in live:
        key = note["stem"].casefold()
        links = set()
        for raw in _LINK_RE.findall(note["body"]):
            target = raw.strip()
            target_key = target.casefold()
            links.add(target_key)
            if target_key in valid:
                incoming[target_key].add(key)
            else:
                kind = "concept" if target_key in concepts else "unknown"
                unresolved.append({"kind": kind, "target": target,
                                   "source": note["rel"]})
        outgoing[key] = links
    orphans = sorted(
        note["stem"] for note in live
        if not outgoing[note["stem"].casefold()]
        and not incoming.get(note["stem"].casefold())
    )
    return {
        "ok": True,
        "notes": len(notes),
        "live_notes": len(live),
        "orphans": orphans,
        "concept_links": sum(1 for x in unresolved if x["kind"] == "concept"),
        "unknown_links": sum(1 for x in unresolved if x["kind"] == "unknown"),
        "unresolved": unresolved,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit KB wikilink graph")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero when true orphans or unknown targets remain")
    args = parser.parse_args()
    from plugin.memory_governed._config import load_governed_config
    result = audit(Path(load_governed_config(HERMES_HOME).wiki_dir))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            if key != "unresolved":
                print(f"{key}: {value}")
    if args.strict and (result["orphans"] or result["unknown_links"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
