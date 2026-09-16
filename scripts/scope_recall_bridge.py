#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scope Recall bridge: export reviewed durable-memory candidates.

Usage:
    python scope_recall_bridge.py export       # write candidates.jsonl
    python scope_recall_bridge.py export --dry-run
    python scope_recall_bridge.py status       # inspect bridge state
    python scope_recall_bridge.py validate     # validate candidates file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, setup_logging

logger = setup_logging("scope_recall_bridge")

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
CRON_OUTPUT = env_path("CRON_OUTPUT_DIR", HERMES_HOME / "cron" / "output")
BRIDGE_DIR = env_path("BRIDGE_DIR", CRON_OUTPUT / "scope_recall_bridge")
BRIDGE_JSONL = env_path("BRIDGE_JSONL", BRIDGE_DIR / "candidates.jsonl")
BRIDGE_REPORT = env_path("BRIDGE_REPORT", BRIDGE_DIR / "last_export.json")

SAFE_TARGETS = {"user", "memory", "project", "ops"}
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{15,}"),
    re.compile(
        r"(?i)(api[_-]?key|app[_-]?secret|token|password)\s*[:=]\s*([^\s,;]+)"
    ),
]

_HELPERS: tuple | None = None


def _governed_helpers():
    """导入统一准入闸门（dev / installed 两种布局），结果缓存。

    桥接的门槛只应有一份实现：插件侧 ``on_session_end``、``memory_promote``、
    本脚本、``import_approved`` 共用 ``screen_bridge_content``。实测本脚本
    此前把 persona.md 全文（其实是 USER.md 模板骨架 + 真实内容混在一起）
    当作 target=user 候选导出，promote 之后模板会写进 L1 —— 而 L1 又会被
    下一次 persona 构建读回去，形成闭环污染。

    导入失败时降级为「不过滤 + 不净化」，绝不因为插件不可用而中断导出。
    """
    global _HELPERS
    if _HELPERS is not None:
        return _HELPERS

    project_root = Path(__file__).resolve().parent.parent
    candidates = []
    if (project_root / "plugin" / "memory_governed").exists():
        candidates.append(str(project_root))          # dev layout
    installed_parent = HERMES_HOME / "plugins"
    if (installed_parent / "governed").exists():
        candidates.append(str(installed_parent))      # installed layout

    last_error = None
    for cand in candidates:
        if cand in sys.path:
            sys.path.remove(cand)
        sys.path.insert(0, cand)
        try:
            try:
                from plugin.memory_governed._bridge import screen_bridge_content
                from plugin.memory_governed._sync import strip_template_fragments
            except ImportError:
                from governed._bridge import screen_bridge_content
                from governed._sync import strip_template_fragments
            _HELPERS = (screen_bridge_content, strip_template_fragments)
            return _HELPERS
        except ImportError as e:
            last_error = e

    logger.warning("governed gate unavailable (%s) — exporting unfiltered", last_error)
    _HELPERS = (lambda _t: None, lambda t: t)
    return _HELPERS


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}


def redact_sensitive(text: str) -> str:
    redacted = text or ""
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda m: f"{m.group(1)}=[REDACTED]" if m.lastindex == 2 else "[REDACTED]",
            redacted,
        )
    return redacted


def has_secret_like_text(text: str) -> bool:
    return redact_sensitive(text) != (text or "")


def stable_id(prefix: str, content: str, source_path: str) -> str:
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(source_path.encode("utf-8", errors="replace"))
    digest.update(content.encode("utf-8", errors="replace"))
    return f"{prefix}-{digest.hexdigest()[:24]}"


def compact_summary(text: str, limit: int = 240) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip().strip("#").strip()
        if not line or line.startswith(("---", "source_path:", "输出文件:")):
            continue
        lines.append(line)
        if sum(len(item) for item in lines) >= limit:
            break
    return " ".join(lines)[:limit].strip()


def extract_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()[:120]
    return fallback[:120]


def source_record(path: Path, target: str, source_kind: str, source_trust: float) -> dict | None:
    content = read_text(path)
    if not content.strip():
        return None
    if has_secret_like_text(content):
        logger.warning("skip_secret_like_source=%s", path)
        return None

    # 先净化模板骨架，再过统一闸门（见 _governed_helpers 的说明）。
    # 净化必须发生在 title/summary 之前 —— 否则 summary 会把
    # "直接编辑此文件" 之类的脚手架当成摘要。
    screen, strip_template = _governed_helpers()
    content = strip_template(content)
    if not content.strip():
        logger.info("skip_template_only_source=%s", path)
        return None
    reason = screen(content)
    if reason is not None:
        logger.info("skip_bridge_source reason=%s path=%s", reason, path)
        return None

    title = extract_title(content, path.stem)
    summary = compact_summary(content)
    safe_content = content[:6000].strip()

    if source_kind == "persona":
        memory_type = "user"
        tags = ["persona", "review-required"]
    elif source_kind == "ops":
        memory_type = "ops"
        tags = ["ops", "review-required"]
    else:
        memory_type = target
        tags = ["knowledge-governance", source_kind, "review-required"]

    return {
        "id": stable_id("hermes", safe_content, str(path)),
        "target": target,
        "content": safe_content,
        "summary": summary or title,
        "memory_type": memory_type,
        "entities": [],
        "tags": tags,
        "source": "hermes-memory-governed",
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
        "metadata": {
            "source_system": "hermes-memory-governed",
            "source_kind": source_kind,
            "source_path": str(path),
            "source_title": title,
            "source_trust": source_trust,
            "import_mode": "read_only",
        },
    }


def collect_candidates() -> list[dict]:
    """Collect Bridge candidates from all sources."""
    candidates = []

    # L4 persona
    persona_path = MEMORY_DIR / "persona.md"
    if persona_path.exists():
        rec = source_record(persona_path, "user", "persona", 0.9)
        if rec:
            candidates.append(rec)

    # Induction outputs
    induction_patterns = [
        CRON_OUTPUT / "nblm_induction_*.md",
        CRON_OUTPUT / "induction_*.md",
        CRON_OUTPUT / "nblm_induction" / "*.md",
    ]
    for pattern in induction_patterns:
        for path in pattern.parent.glob(pattern.name):
            if path.is_file():
                rec = source_record(path, "memory", "induction", 0.7)
                if rec:
                    candidates.append(rec)

    # Wiki knowledge (if exists)
    knowledge_dir = WIKI_DIR / "knowledge"
    if knowledge_dir.exists():
        for subdir in ["concepts", "categories"]:
            for path in (knowledge_dir / subdir).glob("*.md"):
                if path.is_file():
                    rec = source_record(path, "project", "wiki", 0.6)
                    if rec:
                        candidates.append(rec)

    logger.info("collected %d candidates", len(candidates))
    return candidates


def cmd_export(args):
    dry_run = getattr(args, "dry_run", False)
    candidates = collect_candidates()

    if dry_run:
        print(f"[dry-run] Would export {len(candidates)} candidates:")
        for c in candidates:
            print(f"  - {c['target']}: {c['summary'][:60]}...")
        return 0

    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(BRIDGE_JSONL, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            written += 1

    report = {
        "timestamp": datetime.now().isoformat(),
        "candidates_total": len(candidates),
        "written": written,
        "jsonl_path": str(BRIDGE_JSONL),
    }
    BRIDGE_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("exported %d candidates to %s", written, BRIDGE_JSONL)
    return 0


def cmd_status(args):
    status = {
        "bridge_dir": str(BRIDGE_DIR),
        "jsonl_exists": BRIDGE_JSONL.exists(),
        "candidate_count": 0,
        "last_export": None,
    }

    if BRIDGE_JSONL.exists():
        with open(BRIDGE_JSONL, encoding="utf-8") as f:
            status["candidate_count"] = sum(1 for _ in f)

    if BRIDGE_REPORT.exists():
        status["last_export"] = read_json(BRIDGE_REPORT)

    print(json.dumps(status, indent=2, ensure_ascii=False))
    return 0


def cmd_validate(args):
    if not BRIDGE_JSONL.exists():
        print("No candidates file found")
        return 1

    valid = 0
    invalid = 0
    with open(BRIDGE_JSONL, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                if c.get("content") and c.get("target") in SAFE_TARGETS:
                    valid += 1
                else:
                    invalid += 1
                    print(f"  line {line_num}: invalid candidate")
            except json.JSONDecodeError:
                invalid += 1
                print(f"  line {line_num}: invalid JSON")

    print(f"Valid: {valid}, Invalid: {invalid}")
    return 0 if invalid == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="Scope Recall bridge")
    sub = parser.add_subparsers(dest="command")

    p_export = sub.add_parser("export", help="Export candidates")
    p_export.add_argument("--dry-run", action="store_true")

    sub.add_parser("status", help="Show bridge status")
    sub.add_parser("validate", help="Validate candidates file")

    args = parser.parse_args()

    if args.command == "export":
        sys.exit(cmd_export(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    elif args.command == "validate":
        sys.exit(cmd_validate(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
