#!/usr/bin/env python3
"""List every ``# silent-ok:`` exemption for review.

The guard can check that a reason exists, but not that it is true. This script
makes the review practical: it prints file:line, the enclosing function, the
kind and the author-supplied reason, so a human can grep the referenced error
handler in seconds.

Usage:
    python scripts/silent_ok_audit.py
    python scripts/silent_ok_audit.py --json
    python scripts/silent_ok_audit.py --check   # non-zero on empty/short reasons
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_PATTERN = re.compile(r"#\s*silent-ok:\s*(.*)$")


def _targets():
    files = sorted((REPO / "plugin" / "memory_governed").glob("*.py"))
    files += [REPO / "memory_cli.py"]
    files += sorted((REPO / "scripts").glob("*.py"))
    return [p for p in files if p.exists()]


def _function_map(source: str) -> dict:
    """line -> enclosing function name (best effort, AST-based)."""
    tree = ast.parse(source)
    spans = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno,
                              getattr(node, "end_lineno", node.lineno) + 1):
                spans.setdefault(line, node.name)
    return spans


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit silent-ok exemptions")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="Fail when a reason is empty or shorter than 8 chars")
    args = parser.parse_args()

    items = []
    for path in _targets():
        source = path.read_text(encoding="utf-8")
        funcs = _function_map(source)
        for lineno, line in enumerate(source.splitlines(), 1):
            m = _PATTERN.search(line)
            if not m:
                continue
            payload = m.group(1).strip()
            kind, _sep, reason = payload.partition("—")
            items.append({
                "file": path.relative_to(REPO).as_posix(),
                "line": lineno,
                "function": funcs.get(lineno, ""),
                "kind": kind.strip(),
                "reason": reason.strip(),
                "short": len(reason.strip()) < 8,
            })

    summary = {"total": len(items),
               "by_kind": {},
               "short_reasons": [i for i in items if i["short"]]}
    for item in items:
        summary["by_kind"][item["kind"]] = summary["by_kind"].get(item["kind"], 0) + 1

    if args.json:
        print(json.dumps({"summary": summary, "items": items},
                         ensure_ascii=False, indent=2))
    else:
        print("silent-ok exemptions: %d" % len(items))
        for item in items:
            print("%-42s:%-4d %-28s %-16s %s"
                  % (item["file"], item["line"], item["function"],
                     item["kind"], item["reason"]))
        if summary["short_reasons"]:
            print("\nshort/empty reasons: %d" % len(summary["short_reasons"]))
    return 1 if args.check and summary["short_reasons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
