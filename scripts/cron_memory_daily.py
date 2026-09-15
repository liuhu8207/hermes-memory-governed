#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily memory-maintenance orchestrator for a single hermes cron job.

Why this wrapper exists
-----------------------
``hermes cron create --script`` cannot pass arguments to the script, but the
maintenance steps need subcommands (``scope_recall_bridge.py export``,
``memory_pipeline.py persona`` …).  This wrapper runs each step as its own
subprocess and prints a compact summary to stdout, which ``--no-agent`` cron
jobs deliver verbatim.

Mainland-China network preset: ``HF_ENDPOINT=https://hf-mirror.com`` +
``HF_HUB_DISABLE_XET=1`` are injected for every step (embedding downloads).

Steps (in order)
----------------
1. bridge  — ``scope_recall_bridge.py export``   Bridge candidate export
2. persona — ``memory_pipeline.py persona``       L4 persona rebuild
3. promote — ``memory_promote.py``                L3 recurring → Bridge
4. health  — ``memory_health_report.py``          health report

Each step is failure-isolated: a failing step is reported but the remaining
steps still run.  The wrapper exits non-zero if any step failed (so the cron
run is flagged) while still having attempted every step.

Usage
-----
    python cron_memory_daily.py             # real run
    python cron_memory_daily.py --dry-run   # orchestration check, no writes
    python cron_memory_daily.py --verbose   # also dump each step's output
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

#: Scripts live next to this file (HERMES_HOME/scripts).  cron resolves the
#: script relative to that dir and runs it with cwd = scripts dir.
SCRIPTS_DIR = Path(__file__).resolve().parent

#: Embedding/network presets for mainland-China networks (model downloads).
ENV_PRESET: Dict[str, str] = {
    "HF_ENDPOINT": "https://hf-mirror.com",
    "HF_HUB_DISABLE_XET": "1",
}

#: Per-step hard timeout in seconds (mirrors memory_pipeline.run_script).
STEP_TIMEOUT_SECONDS: int = 900


@dataclass(frozen=True)
class Step:
    """One maintenance step: a script invoked with fixed arguments."""

    label: str
    script: str
    args: Tuple[str, ...] = ()
    #: Whether the step understands ``--dry-run`` (steps that don't are skipped
    #: during a dry run so nothing is written).
    supports_dry_run: bool = False


#: Executed top-to-bottom.
#: ``persona`` goes through ``memory_pipeline.py persona`` per the task spec;
#: note it just delegates to ``l4_persona_daily.py`` (memory_pipeline.py:185),
#: so both entries are identical and both under-count (see module fix note).
STEPS: List[Step] = [
    Step("bridge", "scope_recall_bridge.py", ("export",), supports_dry_run=True),
    Step("persona", "memory_pipeline.py", ("persona",), supports_dry_run=True),
    Step("promote", "memory_promote.py", (), supports_dry_run=True),
    Step("health", "memory_health_report.py", (), supports_dry_run=False),
]


@dataclass
class StepOutcome:
    """Result of a single step (never raises out of the runner)."""

    label: str
    skipped: bool
    ok: bool
    returncode: int
    duration: float
    stdout: str = ""
    stderr: str = ""


def _build_env() -> Dict[str, str]:
    """Child environment: inherit os.environ, then force the CN presets."""
    env = dict(os.environ)
    env.update(ENV_PRESET)
    return env


def _run_step(step: Step, dry_run: bool, env: Dict[str, str]) -> StepOutcome:
    """Run one step as an isolated subprocess.  Never raises."""
    script_path = SCRIPTS_DIR / step.script
    if not script_path.exists():
        return StepOutcome(
            label=step.label, skipped=False, ok=False, returncode=127,
            duration=0.0, stderr=f"script not found: {script_path}",
        )

    args = list(step.args)
    if dry_run:
        if step.supports_dry_run:
            args.append("--dry-run")
        else:
            # No --dry-run support → skip so a dry run stays side-effect free.
            return StepOutcome(
                label=step.label, skipped=True, ok=True, returncode=0,
                duration=0.0, stdout="(dry-run: step has no --dry-run flag, skipped)",
            )

    cmd = [sys.executable, str(script_path), *args]
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=STEP_TIMEOUT_SECONDS,
            cwd=str(SCRIPTS_DIR),
            env=env,
        )
        duration = time.monotonic() - started
        return StepOutcome(
            label=step.label, skipped=False, ok=result.returncode == 0,
            returncode=result.returncode, duration=duration,
            stdout=result.stdout or "", stderr=result.stderr or "",
        )
    except subprocess.TimeoutExpired:
        duration = time.monotonic() - started
        return StepOutcome(
            label=step.label, skipped=False, ok=False, returncode=124,
            duration=duration, stderr=f"timeout after {STEP_TIMEOUT_SECONDS}s",
        )
    except Exception as exc:  # noqa: BLE001 — a step must never crash the wrapper
        duration = time.monotonic() - started
        return StepOutcome(
            label=step.label, skipped=False, ok=False, returncode=1,
            duration=duration, stderr=f"{type(exc).__name__}: {exc}",
        )


def _last_line(text: str, limit: int = 200) -> str:
    """Last non-empty, stripped line of *text* (a traceback's exception line)."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line:
            return line[:limit]
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily memory maintenance orchestrator"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="check orchestration without writing anything")
    parser.add_argument("--verbose", action="store_true",
                        help="also print each step's captured stdout/stderr")
    args = parser.parse_args()

    env = _build_env()
    started_at = datetime.now()
    print(
        f"[cron_memory_daily] start {started_at.isoformat(timespec='seconds')} "
        f"dry_run={args.dry_run} steps={len(STEPS)}"
    )

    outcomes: List[StepOutcome] = []
    for index, step in enumerate(STEPS, 1):
        outcome = _run_step(step, args.dry_run, env)
        outcomes.append(outcome)

        status = "SKIP" if outcome.skipped else ("OK" if outcome.ok else "FAIL")
        detail = ""
        if status == "FAIL":
            detail = " — " + (
                _last_line(outcome.stderr) or _last_line(outcome.stdout)
                or f"rc={outcome.returncode}"
            )
        print(
            f"  [{index}/{len(STEPS)}] {step.label:<8} {status:<4} "
            f"rc={outcome.returncode:<3} {outcome.duration:5.1f}s{detail}"
        )
        if args.verbose:
            for stream_name, stream_text in (("stdout", outcome.stdout),
                                             ("stderr", outcome.stderr)):
                for line in (stream_text or "").splitlines():
                    print(f"      {stream_name}: {line}")

    failed = [o for o in outcomes if not o.ok and not o.skipped]
    skipped = [o for o in outcomes if o.skipped]
    ok_count = len(outcomes) - len(failed) - len(skipped)
    total = sum(o.duration for o in outcomes)
    print(
        f"[cron_memory_daily] summary: {ok_count} ok, {len(failed)} failed, "
        f"{len(skipped)} skipped, {total:.1f}s total"
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
