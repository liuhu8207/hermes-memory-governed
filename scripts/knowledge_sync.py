#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Knowledge sync pipeline - orchestrates knowledge collection and processing.

Works with or without Obsidian: output is plain markdown under
$WIKI_DIR/knowledge/. If you point WIKI_DIR at an Obsidian vault, the
notes simply appear in Obsidian as well — nothing else changes.

Integration points:
- Reads from L3 (chat_extractor.py) - READ-ONLY
- Reads from wiki induction output (wiki_nblm_induction.py)
- Reads from Feishu (feishu_sync.py)
- Reads from clipper (clipper_sync.py)
- Generates meeting recaps (meeting_recap.py)
- Generates dashboard (dashboard_gen.py)
- Writes to knowledge/ directory (plain markdown + frontmatter)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager

logger = setup_logging("knowledge_sync")

SCRIPT_DIR = Path(__file__).resolve().parent
HERMES_HOME = bootstrap(__file__)
SCRIPTS_DIR = Path(__file__).resolve().parent
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"
CRON_OUTPUT_DIR = HERMES_HOME / "cron" / "output"


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "sync")


def run_script(script_name: str, args: list[str] | None = None) -> dict[str, Any]:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        # Optional sub-scripts (feishu/clipper/graph/...) may not be migrated —
        # report as skipped instead of failing the whole pipeline.
        logger.info("skipping optional sub-script (not present): %s", script_name)
        return {"ok": True, "skipped": True, "output": f"[skipped: {script_name} not installed]"}

    # Run via runpy with SCRIPT_DIR injected into sys.path: works in both
    # standard Python and embedded/._pth-isolated interpreters (e.g. AutoClaw)
    # where `python script.py` cannot import sibling modules.
    runner_args = list(args or [])
    runner_code = (
        "import sys, runpy\n"
        f"sys.path.insert(0, r'{SCRIPT_DIR}')\n"
        f"sys.argv = [r'{script_path}'] + {runner_args!r}\n"
        f"runpy.run_path(r'{script_path}', run_name='__main__')\n"
    )
    cmd = [sys.executable, "-c", runner_code]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            cwd=str(HERMES_HOME),
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip(), "output": output}
        return {"ok": True, "output": output}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def parse_collected_items(output: str) -> list[dict[str, Any]]:
    output = output.strip()
    if not output:
        return []
    try:
        parsed = json.loads(output)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    items = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("skipping non-json collector output line: %s", line[:120])
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items


def collect_and_process(urls: list[str] | None = None, files: list[str] | None = None, clips: list[str] | None = None) -> dict[str, Any]:
    logger.info("=== Step 1: Collect ===")
    collect_args = []
    if urls:
        for url in urls:
            collect_args.extend(["--url", url])
    if files:
        for f in files:
            collect_args.extend(["--file", f])
    if clips:
        for clip in clips:
            collect_args.extend(["--clip", clip])

    collect_result = run_script("knowledge_collector.py", collect_args)
    if not collect_result["ok"]:
        return {"step": "collect", **collect_result}

    items = parse_collected_items(collect_result["output"])

    if not items:
        logger.info("no new items to process")
        return {"step": "collect", "ok": True, "items": 0}

    logger.info("collected %d items", len(items))

    logger.info("=== Step 2: Process ===")
    temp_file = HERMES_HOME / ".tmp" / "collect_items.json"
    temp_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")

    process_result = run_script("knowledge_processor.py", ["--input", str(temp_file)])
    temp_file.unlink(missing_ok=True)

    if not process_result["ok"]:
        return {"step": "process", **process_result}

    logger.info("=== Step 3: Extract Chats ===")
    chat_result = run_script("chat_extractor.py", ["--days", "7"])

    logger.info("=== Step 4: Sync Wiki Inductions ===")
    induction_result = sync_wiki_inductions()

    logger.info("=== Step 5: Build Graph ===")
    graph_result = run_script("knowledge_graph.py")

    return {
        "ok": True,
        "collect": len(items),
        "process": process_result.get("output", ""),
        "chat": chat_result.get("output", ""),
        "induction": induction_result,
        "graph": graph_result.get("output", ""),
    }


def sync_wiki_inductions() -> dict[str, Any]:
    state = get_state()
    synced = 0

    induction_sources = [
        (CRON_OUTPUT_DIR / "nblm_induction", "*.md"),
        (CRON_OUTPUT_DIR, "nblm_induction_*.md"),
    ]

    for induction_dir, pattern in induction_sources:
        if not induction_dir.exists():
            continue
        for md_file in induction_dir.glob(pattern):
            file_id = f"induction:{md_file.name}"
            if state.is_processed(file_id):
                continue

            try:
                content = md_file.read_text(encoding="utf-8", errors="replace")
                output_path = KNOWLEDGE_DIR / "inductions" / md_file.name
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(content, encoding="utf-8")
                state.mark_processed(file_id)
                synced += 1
                logger.info("synced induction: %s", md_file.name)
            except Exception as exc:
                logger.warning("failed to sync induction %s: %s", md_file.name, exc)

    state.last_run = datetime.now().isoformat()
    return {"synced": synced}


def sync_chats_only() -> dict[str, Any]:
    logger.info("=== Chat Sync Only ===")
    result = run_script("chat_extractor.py", ["--days", "30"])
    if result["ok"]:
        graph_result = run_script("knowledge_graph.py")
        return {"ok": True, "chats": result.get("output", ""), "graph": graph_result.get("output", "")}
    return result


def rebuild_graph() -> dict[str, Any]:
    logger.info("=== Rebuild Graph ===")
    result = run_script("knowledge_graph.py")
    return {"ok": result["ok"], "output": result.get("output", "")}


def full_sync(days: int = 7, dashboard: bool = True) -> dict[str, Any]:
    logger.info("=== Full Knowledge Sync ===")
    results = {}

    chat_result = run_script("chat_extractor.py", ["--days", str(days)])
    results["chats"] = chat_result.get("output", "")

    meeting_result = run_script("meeting_recap.py", ["--days", str(days)])
    results["meetings"] = meeting_result.get("output", "")

    clip_result = run_script("clipper_sync.py")
    results["clips"] = clip_result.get("output", "")

    results["inductions"] = sync_wiki_inductions()

    graph_result = run_script("knowledge_graph.py")
    results["graph"] = graph_result.get("output", "")

    if dashboard:
        dashboard_result = run_script("dashboard_gen.py")
        results["dashboard"] = dashboard_result.get("output", "")

    # Rebuild Obsidian-compatible related links across all knowledge notes
    links_result = run_script("knowledge_links.py")
    results["links"] = links_result.get("output", "")

    logger.info("sync completed")
    return {"ok": True, **results}


def sync_meetings_only() -> dict[str, Any]:
    logger.info("=== Meeting Recap Only ===")
    result = run_script("meeting_recap.py", ["--days", "30"])
    if result["ok"]:
        graph_result = run_script("knowledge_graph.py")
        return {"ok": True, "meetings": result.get("output", ""), "graph": graph_result.get("output", "")}
    return result


def sync_clips_only() -> dict[str, Any]:
    logger.info("=== Clipper Sync Only ===")
    result = run_script("clipper_sync.py")
    if result["ok"]:
        graph_result = run_script("knowledge_graph.py")
        return {"ok": True, "clips": result.get("output", ""), "graph": graph_result.get("output", "")}
    return result


def generate_dashboard_only() -> dict[str, Any]:
    logger.info("=== Dashboard Only ===")
    result = run_script("dashboard_gen.py", ["--open"])
    return {"ok": result["ok"], "output": result.get("output", "")}


def sync_feishu(app_id: str, app_secret: str, documents: list[str] | None = None, space_id: str | None = None) -> dict[str, Any]:
    logger.info("=== Feishu Sync ===")
    args = ["--app-id", app_id, "--app-secret", app_secret]
    if documents:
        for doc in documents:
            args.extend(["--document", doc])
    if space_id:
        args.extend(["--space-id", space_id])
    result = run_script("feishu_sync.py", args)
    if result["ok"]:
        graph_result = run_script("knowledge_graph.py")
        return {"ok": True, "feishu": result.get("output", ""), "graph": graph_result.get("output", "")}
    return result


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge sync pipeline (Obsidian optional)")
    parser.add_argument("--url", action="append", help="URL to collect")
    parser.add_argument("--file", action="append", help="File to collect")
    parser.add_argument("--clip", action="append", help="Clipboard text to collect")
    parser.add_argument("--chats-only", action="store_true", help="Only sync chats")
    parser.add_argument("--meetings-only", action="store_true", help="Only sync meetings")
    parser.add_argument("--clips-only", action="store_true", help="Only sync clips")
    parser.add_argument("--dashboard", action="store_true", help="Only generate dashboard")
    parser.add_argument("--full", action="store_true", help="Run full knowledge sync")
    parser.add_argument("--sync-inductions", action="store_true", help="Only sync wiki induction outputs")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip dashboard in --full mode")
    parser.add_argument("--days", type=int, default=7, help="Days of L3 data to process")
    parser.add_argument("--feishu", action="store_true", help="Sync from Feishu")
    parser.add_argument("--feishu-app-id", help="Feishu App ID")
    parser.add_argument("--feishu-app-secret", help="Feishu App Secret")
    parser.add_argument("--feishu-document", action="append", help="Feishu document ID")
    parser.add_argument("--feishu-space-id", help="Feishu wiki space ID")
    parser.add_argument("--rebuild-graph", action="store_true", help="Only rebuild graph")
    args = parser.parse_args()

    if args.full:
        result = full_sync(days=args.days, dashboard=not args.no_dashboard)
    elif args.sync_inductions:
        result = {"ok": True, **sync_wiki_inductions()}
    elif args.rebuild_graph:
        result = rebuild_graph()
    elif args.chats_only:
        result = sync_chats_only()
    elif args.meetings_only:
        result = sync_meetings_only()
    elif args.clips_only:
        result = sync_clips_only()
    elif args.dashboard:
        result = generate_dashboard_only()
    elif args.feishu:
        app_id = args.feishu_app_id or os.getenv("FEISHU_APP_ID", "")
        app_secret = args.feishu_app_secret or os.getenv("FEISHU_APP_SECRET", "")
        result = sync_feishu(app_id, app_secret, args.feishu_document, args.feishu_space_id)
    else:
        result = collect_and_process(urls=args.url, files=args.file, clips=args.clip)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
