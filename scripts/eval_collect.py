#!/usr/bin/env python3
"""Collect candidate evaluation queries from the L3 conversation archive.

The point is to evaluate retrieval against **how the user actually asks**, not
against note titles. This is read-only: it never writes to L3 and never sends
anything off the machine.

Usage:
    python scripts/eval_collect.py --days 14
    python scripts/eval_collect.py --out D:/path/candidates.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hermes_env import bootstrap, env_path

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")

_SKIP = re.compile(r"^(/|!|python |pip |git |cd |ls |dir )", re.IGNORECASE)


def _clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect eval query candidates")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", default=str(HERMES_HOME / "evals" / "candidates.jsonl"))
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=200)
    args = parser.parse_args()

    if not L3_DB_PATH.exists():
        print(json.dumps({"ok": False, "error": "L3 database not found",
                          "path": str(L3_DB_PATH)}, indent=2))
        return 1

    since = (datetime.now() - timedelta(days=max(1, args.days))).timestamp()
    conn = sqlite3.connect("file:%s?mode=ro" % L3_DB_PATH.as_posix(), uri=True)
    try:
        rows = conn.execute(
            "SELECT role, content, timestamp, session_id FROM messages "
            "WHERE timestamp >= ? ORDER BY id", (since,)).fetchall()
    finally:
        conn.close()

    seen = set()
    out = []
    for role, content, ts, session in rows:
        if str(role).lower() != "user":
            continue
        text = _clean(content)
        if not (args.min_chars <= len(text) <= args.max_chars):
            continue
        if _SKIP.match(text) or text.startswith(("http://", "https://")):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"q": text,
                    "date": datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")
                    if ts else "",
                    "session": session or ""})

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n",
                    encoding="utf-8")
    print(json.dumps({"ok": True, "candidates": len(out), "out": str(dest),
                      "days": args.days}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
