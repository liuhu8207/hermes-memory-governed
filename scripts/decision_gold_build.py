#!/usr/bin/env python3
"""Build a private 100-item KB-novelty decision set from real local memory.

The output is intentionally written under ``$HERMES_HOME/evals`` and never
under the repository by default. Labels are rule-derived, deterministic, and
carry provenance so they can be audited before the decision layer is enabled.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap, resolve_repo

HERMES_HOME = bootstrap(__file__)
REPO = resolve_repo(__file__)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_OPPOSITES = (
    ("已恢复", "失败"), ("恢复", "不可用"), ("已修复", "未修复"),
    ("authorized", "unauthorized"), ("enabled", "disabled"),
    ("成功", "失败"), ("可用", "不可用"), ("true", "false"),
)
_TIME_RE = re.compile(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b|截至|当时|曾经|此前")
_ENTITY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{2,}")
_GENERIC_ENTITIES = {
    "api", "http", "https", "json", "true", "false", "enabled", "disabled",
    "success", "failed", "error", "p0", "p1", "hgm", "mcp", "l2", "l3",
    "agent", "cli", "hgm_mcp.py", "memory_cli.py", "hermes-memory-governed",
}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _tokens(text: str) -> set[str]:
    from plugin.memory_governed._text import normalize_for_match

    raw = normalize_for_match(text).lower()
    ascii_words = set(re.findall(r"[a-z0-9_.:-]{3,}", raw))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", raw))
    bigrams = {cjk[i:i + 2] for i in range(max(0, len(cjk) - 1))}
    return ascii_words | bigrams


def _entities(text: str) -> set[str]:
    return {x.casefold() for x in _ENTITY_RE.findall(text)}


def _containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _opposition(text_a: str, text_b: str) -> str:
    lower_a, lower_b = text_a.casefold(), text_b.casefold()
    for left, right in _OPPOSITES:
        if (left in lower_a and right in lower_b) or (
                right in lower_a and left in lower_b):
            return f"{left}<->{right}"
    return ""


def _load_nodes() -> list[dict]:
    from plugin.memory_governed._bridge import has_secret_like_text
    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed._diag import log_degraded
    from plugin.memory_governed._kb import KnowledgeBase
    from plugin.memory_governed._quality import classify_fact
    from plugin.memory_governed import _vault
    import lancedb

    cfg = load_governed_config(HERMES_HOME)
    kb = KnowledgeBase(cfg)
    nodes = []
    for rel, path in kb.scan_vault_paths().items():
        meta, body = _vault.parse_frontmatter(
            path.read_text(encoding="utf-8", errors="replace"))
        title = str(meta.get("title") or path.stem)
        if not body.strip():
            continue
        text = title + "\n" + body
        if has_secret_like_text(text):
            continue
        nodes.append({
            "id": "kb:" + rel,
            "kind": "kb",
            "status": str(meta.get("status") or ""),
            "title": title,
            "body": body,
            "tags": [str(x) for x in (meta.get("tags") or [])],
            "concepts": [str(x) for x in (meta.get("concepts") or [])],
            "superseded_by": str(meta.get("superseded_by") or ""),
            "archived_reason": str(meta.get("archived_reason") or ""),
            "timestamp": str(meta.get("updated") or meta.get("created") or ""),
        })

    table = lancedb.connect(str(cfg.l2_db_path)).open_table("memories")
    for index, row in enumerate(table.to_arrow().to_pylist()):
        content = _clean(row.get("content"))
        if not content or has_secret_like_text(content):
            continue
        verdict = classify_fact(content, role=str(row.get("role") or ""),
                                source=str(row.get("source") or ""))
        if verdict and verdict.get("kind") in ("test_residue", "assistant_fragment"):
            continue
        nodes.append({
            "id": f"l2:{index}",
            "kind": "l2",
            "status": "active",
            "title": content[:100],
            "body": content,
            "tags": [],
            "concepts": [],
            "timestamp": str(row.get("timestamp") or ""),
            "category": str(row.get("category") or ""),
        })

    archive = Path(cfg.l2_db_path).parent / "l2_archive.jsonl"
    if archive.exists():
        for index, line in enumerate(archive.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:  # silent-ok: value-fallback — malformed archive line is skipped
                continue
            content = _clean(row.get("content"))
            if not content or has_secret_like_text(content):
                continue
            verdict = classify_fact(content, role=str(row.get("role") or ""),
                                    source=str(row.get("source") or ""))
            if verdict and verdict.get("kind") in ("test_residue", "assistant_fragment"):
                continue
            nodes.append({
                "id": f"archive:{index}",
                "kind": "archive",
                "status": str(row.get("state") or "superseded"),
                "title": content[:100],
                "body": content,
                "tags": [],
                "concepts": [],
                "timestamp": str(row.get("timestamp") or row.get("ts") or ""),
                "archive_reason": str(row.get("reason") or ""),
            })
    for node in nodes:
        node["_tokens"] = _tokens(node["body"])
        node["_title_tokens"] = _tokens(node["title"])
        node["_title_entities"] = _entities(node["title"])
        node["_entities"] = _entities(node["body"])
        node["_tags"] = {x.casefold() for x in node["tags"]}
        node["_concepts"] = {x.casefold() for x in node["concepts"]}
    return nodes


def _load_current_state_nodes() -> list[dict]:
    """Add small current-state observations used to pair with real stale records."""
    from plugin.memory_governed._config import load_governed_config

    cfg = load_governed_config(HERMES_HOME)
    nodes = []
    jobs_path = HERMES_HOME / "cron" / "jobs.json"
    if jobs_path.exists():
        try:
            raw_jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
            jobs = raw_jobs.get("jobs", []) if isinstance(raw_jobs, dict) else raw_jobs
            job = next((x for x in jobs if x.get("name") == "memory-daily-maintenance"),
                       None)
            if job:
                body = (f"截至 2026-10-23，memory-daily-maintenance 当前状态为成功："
                        f"last_status={job.get('last_status')}，"
                        f"failure_streak={job.get('failure_streak', 0)}；"
                        "每日维护全部步骤成功。")
                nodes.append({
                    "id": "current:cron",
                    "kind": "current-state",
                    "status": "current",
                    "title": "HGM 每日维护当前状态",
                    "body": body,
                    "tags": ["hgm", "cron"],
                    "concepts": [],
                    "timestamp": str(job.get("last_run_at") or ""),
                })
        except Exception as e:  # noqa: BLE001 - optional current-state enrichment
            log_degraded("decision_gold", "current_state_failed", exc=e)
    for node in nodes:
        node["_tokens"] = _tokens(node["body"])
        node["_title_tokens"] = _tokens(node["title"])
        node["_title_entities"] = _entities(node["title"])
        node["_entities"] = _entities(node["body"])
        node["_tags"] = {x.casefold() for x in node["tags"]}
        node["_concepts"] = set()
    return nodes


def _candidate(a: dict, b: dict) -> dict:
    sim = _jaccard(a["_tokens"], b["_tokens"])
    contain = _containment(a["_tokens"], b["_tokens"])
    title_contain = _containment(a["_title_tokens"], b["_title_tokens"])
    shared_tags = a["_tags"] & b["_tags"]
    shared_concepts = a["_concepts"] & b["_concepts"]
    shared_entities = a["_entities"] & b["_entities"]
    specific_entities = shared_entities - _GENERIC_ENTITIES
    shared_title_entities = ((a["_title_entities"] & b["_title_entities"])
                             - _GENERIC_ENTITIES)
    identity_topic = bool(shared_tags or shared_concepts or specific_entities)
    primary_topic = bool(shared_concepts or shared_title_entities)
    strong_topic = bool(shared_tags or shared_concepts or specific_entities
                        or contain >= 0.18)
    opposition = _opposition(a["body"], b["body"])
    temporal = bool(_TIME_RE.search(a["body"] + " " + b["body"]))

    label = "uncertain"
    score = 0.0
    reason = "insufficient evidence"
    archive_reason = (a.get("archive_reason", "") + " " +
                      b.get("archive_reason", "")).casefold()
    superseded_match = (
        bool(a.get("superseded_by")) and a["superseded_by"] == b["title"]
    ) or (bool(b.get("superseded_by")) and b["superseded_by"] == a["title"])
    template_agent_setup = ("agent 名为" in a["body"].casefold()
                            and "agent 名为" in b["body"].casefold())
    duplicate_evidence = not template_agent_setup and (
        superseded_match
        or (contain >= 0.78 and (bool(shared_tags)
                                 or title_contain >= 0.65
                                 or len(shared_title_entities) >= 2))
        or (sim >= 0.62 and len(shared_tags) >= 2 and title_contain >= 0.30)
        or (len(shared_tags) >= 2 and title_contain >= 0.20 and contain >= 0.30)
        or (title_contain >= 0.65 and contain >= 0.25
            and (bool(shared_tags or shared_concepts)
                 or len(shared_title_entities) >= 2))
    )
    if (opposition and identity_topic and contain >= 0.12 and not temporal
            and (a["kind"] == "current-state" or b["kind"] == "current-state"
                 or "superseded by" in archive_reason and "restored" in archive_reason)):
        label, score, reason = "conflicts", 0.85, "opposite current-state claims: " + opposition
    elif opposition and identity_topic and temporal:
        label, score, reason = "uncertain", 0.75, "opposite status with temporal qualifiers"
    elif duplicate_evidence and strong_topic:
        label, score, reason = "duplicate", max(contain, sim), "high text containment on the same topic"
    elif ((primary_topic and contain >= 0.12)
          or (bool(shared_tags) and title_contain >= 0.20)):
        label, score, reason = "extends", 0.55 + contain, "same entity/topic with distinct detail"
    elif not primary_topic and not shared_tags and not shared_concepts and sim < 0.025:
        label, score, reason = "new", 1.0 - sim, "no shared tags, concepts, entities, or material text overlap"
    elif (identity_topic or contain >= 0.08) and sim < 0.10:
        label, score, reason = "uncertain", 0.5 + contain, "weak lexical overlap without enough topic evidence"

    return {"a": a, "b": b, "label": label, "score": score,
            "reason": reason, "similarity": sim, "containment": contain,
            "shared_tags": sorted(shared_tags), "shared_entities": sorted(shared_entities)}


def _pick(pool: list[dict], label: str, quota: int,
          used: set[tuple[str, str]]) -> list[dict]:
    picked = []
    ordered = sorted(pool, key=lambda x: (-x["score"], x["a"]["id"], x["b"]["id"]))
    for item in ordered:
        key = tuple(sorted((item["a"]["id"], item["b"]["id"])))
        if key in used or item["label"] != label:
            continue
        used.add(key)
        picked.append(item)
        if len(picked) >= quota:
            break
    return picked


def _public_node(node: dict) -> dict:
    return {
        "title": node["title"],
        "body": node["body"],
        "tags": node["tags"],
        "concepts": node["concepts"],
    }


def build(count: int = 100, seed: int = 20261023) -> dict:
    nodes = _load_nodes() + _load_current_state_nodes()
    random.Random(seed).shuffle(nodes)
    candidates = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            candidates.append(_candidate(a, b))

    quotas = {"duplicate": 12, "extends": 25, "conflicts": 0,
              "new": 38, "uncertain": 25}
    used: set[tuple[str, str]] = set()
    selected = []
    for label, quota in quotas.items():
        selected.extend(_pick(candidates, label, quota, used))
    if len(selected) < count:
        selected.extend(_pick(candidates, "uncertain", count - len(selected), used))
    selected = selected[:count]
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} candidates available; requested {count}")

    items = []
    for index, item in enumerate(selected, 1):
        items.append({
            "id": f"real-{index:03d}",
            "new_note": _public_node(item["a"]),
            "candidates": [_public_node(item["b"])],
            "expect": item["label"],
            "rationale": item["reason"],
            "label_source": "deterministic-real-data-v1",
            "label_confidence": round(min(1.0, item["score"]), 4),
            "source_ids": [item["a"]["id"], item["b"]["id"]],
            "source_kinds": [item["a"]["kind"], item["b"]["kind"]],
        })
    counts = defaultdict(int)
    for item in items:
        counts[item["expect"]] += 1
    return {
        "version": 1,
        "private": True,
        "source_policy": "real-local-memory-only",
        "label_method": "deterministic-rules-v1",
        "review_status": "rule-labeled-not-human-reviewed",
        "operation": "kb_novelty",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "labels": ["duplicate", "extends", "conflicts", "new", "uncertain"],
        "counts": dict(counts),
        "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build private decision gold set")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--output", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = build(max(1, args.count), args.seed)
    out = Path(args.output) if args.output else HERMES_HOME / "evals" / "decision_gold.real.json"
    resolved = out.resolve()
    if REPO.resolve() in resolved.parents:
        raise SystemExit("refusing to write private real data inside the public repository")
    if args.dry_run:
        print(json.dumps({"ok": True, "dry_run": True, "counts": data["counts"],
                          "output": str(resolved)}, ensure_ascii=False, indent=2))
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(out)
    print(json.dumps({"ok": True, "output": str(out), "count": len(data["items"]),
                      "counts": data["counts"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
