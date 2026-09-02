#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified maintenance pipeline for Hermes memory governance.

Usage:
    python memory_pipeline.py daily           # run daily maintenance
    python memory_pipeline.py bridge          # export Bridge candidates
    python memory_pipeline.py bridge --dry-run
    python memory_pipeline.py health          # run health report
    python memory_pipeline.py persona         # regenerate L4 persona
    python memory_pipeline.py daily --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("pipeline")

HERMES_HOME = bootstrap(__file__)
SCRIPTS_DIR = Path(__file__).resolve().parent
REPORT_JSON = env_path(
    "MEMORY_PIPELINE_REPORT_JSON",
    HERMES_HOME / "cron" / "output" / "memory_pipeline_last.json",
)

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{15,}"),
    re.compile(
        r"(?i)(api[_-]?key|app[_-]?secret|token|password|passwd|pwd|secret)"
        r"\s*[:=]\s*([^\s,;，；]+)"
    ),
]


def redact_and_trim(text: str, limit: int = 4000) -> str:
    redacted = text or ""
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda m: f"{m.group(1)}=[REDACTED]" if m.lastindex == 2 else "[REDACTED]",
            redacted,
        )
    if len(redacted) > limit:
        return redacted[:limit] + "\n...[truncated]"
    return redacted


@dataclass
class StepResult:
    name: str
    script: str
    ok: bool
    returncode: int
    duration_seconds: float
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False


def run_script(name: str, script_name: str, args: Sequence[str] = (), timeout: int = 900) -> StepResult:
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        return StepResult(
            name=name,
            script=script_name,
            ok=False,
            returncode=127,
            duration_seconds=0,
            stderr=f"script not found: {script_path}",
        )

    started = datetime.now()
    cmd = [sys.executable, str(script_path), *args]
    logger.info("step=%s script=%s args=%s", name, script_name, list(args))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=str(HERMES_HOME),
        )
        duration = (datetime.now() - started).total_seconds()
        return StepResult(
            name=name,
            script=script_name,
            ok=result.returncode == 0,
            returncode=result.returncode,
            duration_seconds=round(duration, 2),
            stdout=redact_and_trim(result.stdout.strip()),
            stderr=redact_and_trim(result.stderr.strip()),
        )
    except subprocess.TimeoutExpired as exc:
        duration = (datetime.now() - started).total_seconds()
        return StepResult(
            name=name,
            script=script_name,
            ok=False,
            returncode=124,
            duration_seconds=round(duration, 2),
            stdout=redact_and_trim((exc.stdout or "").strip()) if isinstance(exc.stdout, str) else "",
            stderr="timeout",
        )


def cmd_daily(args):
    """Run daily maintenance pipeline."""
    dry_run = getattr(args, "dry_run", False)
    with_nblm = getattr(args, "with_notebooklm", False)
    steps = [
        ("bridge", "scope_recall_bridge.py", ["export"] + (["--dry-run"] if dry_run else [])),
        ("health", "memory_health_report.py", []),
    ]

    # NotebookLM steps: only if configured AND user opted in
    if with_nblm:
        nblm_available = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if nblm_available:
            steps.insert(0, ("nblm_induction", "wiki_nblm_induction.py", []))
            steps.insert(0, ("nblm_daily", "wiki_nblm_daily.py", []))
        else:
            logger.info("NotebookLM not configured, skipping nblm steps")

    results = []
    for name, script, script_args in steps:
        if dry_run:
            logger.info("[dry-run] would run: %s %s", script, script_args)
            continue
        result = run_script(name, script, script_args)
        results.append(result)
        status = "OK" if result.ok else "FAIL"
        logger.info("step=%s status=%s duration=%.1fs", name, status, result.duration_seconds)

    _write_report(results)
    all_ok = all(r.ok for r in results)
    logger.info("daily pipeline: %s (%d steps)", "OK" if all_ok else "FAILED", len(results))
    return 0 if all_ok else 1


def cmd_bridge(args):
    """Export Bridge candidates."""
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        logger.info("[dry-run] would export Bridge candidates")
        return 0

    result = run_script("bridge", "scope_recall_bridge.py", ["export"])
    logger.info("bridge export: %s", "OK" if result.ok else "FAIL")
    if result.stderr:
        logger.error(result.stderr)
    return 0 if result.ok else 1


def cmd_health(args):
    """Run health report."""
    result = run_script("health", "memory_health_report.py", [])
    if result.stdout:
        print(result.stdout)
    return 0 if result.ok else 1


def cmd_persona(args):
    """Regenerate L4 persona."""
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        logger.info("[dry-run] would regenerate L4 persona")
        return 0

    result = run_script("persona", "l4_persona_daily.py", [])
    logger.info("persona generation: %s", "OK" if result.ok else "FAIL")
    return 0 if result.ok else 1


def _write_report(results: list[StepResult]) -> None:
    """Write pipeline execution report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "hermes_home": str(HERMES_HOME),
        "steps": [asdict(r) for r in results],
        "all_ok": all(r.ok for r in results),
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("report written to %s", REPORT_JSON)


def main():
    parser = argparse.ArgumentParser(description="Hermes memory governance pipeline")
    sub = parser.add_subparsers(dest="command")

    p_daily = sub.add_parser("daily", help="Run daily maintenance")
    p_daily.add_argument("--dry-run", action="store_true")
    p_daily.add_argument("--with-notebooklm", action="store_true",
                         help="Include NotebookLM steps (requires GOOGLE_API_KEY)")

    p_bridge = sub.add_parser("bridge", help="Export Bridge candidates")
    p_bridge.add_argument("--dry-run", action="store_true")

    sub.add_parser("health", help="Run health report")

    p_persona = sub.add_parser("persona", help="Regenerate L4 persona")
    p_persona.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "daily":
        sys.exit(cmd_daily(args))
    elif args.command == "bridge":
        sys.exit(cmd_bridge(args))
    elif args.command == "health":
        sys.exit(cmd_health(args))
    elif args.command == "persona":
        sys.exit(cmd_persona(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
