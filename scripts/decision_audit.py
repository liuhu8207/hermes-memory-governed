#!/usr/bin/env python3
"""Summarize the optional decision-layer JSONL audit.

The audit stores metadata, not note bodies. This command is read-only and safe
to run while a persistent MCP process is appending new rows.

Usage:
    python scripts/decision_audit.py
    python scripts/decision_audit.py --path /path/to/decision_audit.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap

HERMES_HOME = bootstrap(__file__)


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(float(ordered[index]), 3)


def summarize(path: Path) -> dict:
    rows: list[dict] = []
    malformed = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:  # silent-ok: already-reported — counted in the malformed return value
                    malformed += 1
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except FileNotFoundError:  # silent-ok: already-reported — returned as ok=false with a stable error tag
        return {"ok": False, "error": "audit file not found", "path": str(path)}
    except OSError as exc:  # silent-ok: already-reported — returned as ok=false with the exception type
        return {"ok": False,
                "error": "cannot read audit: " + type(exc).__name__,
                "path": str(path)}

    latencies = [float(r.get("latency_ms", 0) or 0) for r in rows]
    fallback = [r for r in rows if bool(r.get("fallback"))]
    actions = Counter(str(r.get("action") or "unknown") for r in rows)
    chosen = Counter(str(r.get("chosen") or "unknown") for r in rows)
    operations = Counter(str(r.get("operation") or "unknown") for r in rows)
    sources = Counter(str(r.get("source") or "unknown") for r in rows)
    confidences = [float(r.get("confidence", 0.0) or 0.0) for r in rows
                   if not r.get("fallback")]
    return {
        "ok": True,
        "path": str(path),
        "rows": len(rows),
        "malformed": malformed,
        "operations": dict(operations.most_common()),
        "sources": dict(sources.most_common()),
        "actions": dict(actions.most_common()),
        "chosen": dict(chosen.most_common()),
        "fallback": len(fallback),
        "fallback_rate": round(len(fallback) / len(rows), 4) if rows else 0.0,
        "confidence_mean": (round(sum(confidences) / len(confidences), 4)
                            if confidences else 0.0),
        "latency_ms_p50": _percentile(latencies, 0.50),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "first_ts": str(rows[0].get("ts") or "") if rows else "",
        "last_ts": str(rows[-1].get("ts") or "") if rows else "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize decision audit JSONL")
    parser.add_argument("--path", default="",
                        help="Default: $HERMES_HOME/memory/decision_audit.jsonl")
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args()
    path = Path(args.path) if args.path else (
        Path(HERMES_HOME) / "memory" / "decision_audit.jsonl")
    result = summarize(path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
