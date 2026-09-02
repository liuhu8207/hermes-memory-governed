#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Memory health report for the governed memory system.

Checks:
- L1 file existence and freshness
- L2 LanceDB table status
- L3 SQLite table counts and FTS5 status
- L4 persona freshness
- Bridge candidate counts
- Write queue stats (if accessible)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("health_report")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
BRIDGE_DIR = env_path("BRIDGE_DIR", HERMES_HOME / "cron" / "output" / "scope_recall_bridge")
REPORT_PATH = env_path("HEALTH_REPORT_PATH", HERMES_HOME / "health_report.md")
REPORT_JSON = env_path("HEALTH_REPORT_JSON", HERMES_HOME / "health_report.json")


class HealthCheck:
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []  # real failures — only these fail the exit code

    def check(self, name: str, ok: bool, detail: str = ""):
        status = "✅" if ok else "❌"
        self.checks.append({"name": name, "ok": ok, "detail": detail})
        line = f"{status} {name}"
        if detail:
            line += f" — {detail}"
        print(line)
        if not ok:
            self.errors.append(name)

    def check_warning(self, name: str, detail: str = ""):
        self.checks.append({"name": name, "ok": True, "detail": detail})
        print(f"⚠️  {name} — {detail}")
        self.warnings.append(name)

    def summary(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "total": len(self.checks),
            "ok": sum(1 for c in self.checks if c["ok"]),
            "warnings": len(self.warnings),
            "errors": len(self.errors),
            "checks": self.checks,
        }


def check_l1(h: HealthCheck):
    """Check L1 handwritten files."""
    memory_file = MEMORY_DIR / "MEMORY.md"
    user_file = MEMORY_DIR / "USER.md"

    h.check("L1 MEMORY.md", memory_file.exists(),
            f"{memory_file.stat().st_size} bytes" if memory_file.exists() else "not found")
    h.check("L1 USER.md", user_file.exists(),
            f"{user_file.stat().st_size} bytes" if user_file.exists() else "not found")

    # Freshness
    for name, path in [("MEMORY.md", memory_file), ("USER.md", user_file)]:
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            if age_days > 90:
                h.check_warning(f"L1 {name} freshness", f"last modified {age_days} days ago")


def check_l2(h: HealthCheck):
    """Check L2 LanceDB."""
    l2_dir = MEMORY_DIR / "l2"
    if not l2_dir.exists():
        h.check("L2 LanceDB", False, "directory not found")
        return

    try:
        import lancedb
        db = lancedb.connect(str(l2_dir))
        tables = db.list_tables().tables
        h.check("L2 LanceDB", True, f"{len(tables)} tables")

        if "memories" in tables:
            table = db.open_table("memories")
            count = table.count_rows()
            h.check("L2 memories table", True, f"{count} rows")
        else:
            h.check_warning("L2 memories table", "not created yet")
    except ImportError:
        h.check_warning("L2 LanceDB", "lancedb not installed")
    except Exception as e:
        h.check("L2 LanceDB", False, str(e))


def check_l3(h: HealthCheck):
    """Check L3 SQLite."""
    l3_db = MEMORY_DIR / "l3" / "l3.db"
    if not l3_db.exists():
        h.check("L3 SQLite", False, "database not found")
        return

    try:
        conn = sqlite3.connect(f"file:{l3_db}?mode=ro", uri=True)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]

        h.check("L3 SQLite", True, f"{len(tables)} tables")

        if "messages" in tables:
            count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            h.check("L3 messages", True, f"{count} rows")

        if "messages_fts" in tables:
            h.check("L3 FTS5", True, "indexed")
        else:
            h.check_warning("L3 FTS5", "not created")

        conn.close()
    except Exception as e:
        h.check("L3 SQLite", False, str(e))


def check_l4(h: HealthCheck):
    """Check L4 persona."""
    persona_file = MEMORY_DIR / "persona.md"
    meta_file = MEMORY_DIR / "persona_meta.json"

    h.check("L4 persona.md", persona_file.exists(),
            f"{persona_file.stat().st_size} bytes" if persona_file.exists() else "not found")

    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            last_gen = meta.get("last_generated")
            if last_gen:
                h.check("L4 metadata", True, f"last generated: {last_gen}")
        except Exception:
            pass


def check_bridge(h: HealthCheck):
    """Check Bridge status."""
    jsonl = BRIDGE_DIR / "candidates.jsonl"
    report = BRIDGE_DIR / "last_export.json"

    if jsonl.exists():
        count = sum(1 for _ in jsonl.open(encoding="utf-8"))
        h.check("Bridge candidates", True, f"{count} candidates")
    else:
        h.check_warning("Bridge candidates", "no candidates file")

    if report.exists():
        try:
            r = json.loads(report.read_text(encoding="utf-8"))
            h.check("Bridge export", True, f"last: {r.get('timestamp', '?')}")
        except Exception:
            pass


def check_config(h: HealthCheck):
    """Check configuration."""
    config_file = HERMES_HOME / "governed_memory.json"
    if config_file.exists():
        h.check("Config file", True, "loaded")
    else:
        h.check_warning("Config file", "not found — using defaults")


def generate_report(h: HealthCheck):
    """Write health report to markdown and JSON."""
    # Markdown
    lines = [
        "# Memory Health Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for check in h.checks:
        status = "✅" if check["ok"] else "❌"
        line = f"- {status} {check['name']}"
        if check["detail"]:
            line += f": {check['detail']}"
        lines.append(line)

    if h.warnings:
        lines.extend(["", "## Warnings", ""])
        for w in h.warnings:
            lines.append(f"- {w}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # JSON
    REPORT_JSON.write_text(
        json.dumps(h.summary(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Health report written to %s", REPORT_PATH)


def main():
    h = HealthCheck()

    print("=== Hermes Memory Health Report ===\n")

    check_config(h)
    check_l1(h)
    check_l2(h)
    check_l3(h)
    check_l4(h)
    check_bridge(h)

    print(f"\n{'='*40}")
    print(f"Total: {len(h.checks)} | OK: {sum(1 for c in h.checks if c['ok'])} "
          f"| Warnings: {len(h.warnings)} | Errors: {len(h.errors)}")

    generate_report(h)

    return 1 if h.errors else 0


if __name__ == "__main__":
    sys.exit(main())
