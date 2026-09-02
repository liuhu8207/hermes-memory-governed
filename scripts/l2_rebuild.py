#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild L2 semantic index from L3 (source of truth).

Use cases:
- L2 table corrupted or out of sync
- embedding model/dim changed (config.vector.model / vector.dim)
- extraction logic upgraded

⚠️ 维度一致性：本脚本通过 :class:`EmbeddingService` 解析 embedding 后端
（API → 本地 → 文本，优先级与生产读写路径完全一致），建表维度取后端**实际**维度
（API 由返回向量长度自动检测，如 1024；本地模型用 model 输出维度，如 512）。
因此只要本机 config 与生产一致，重建出的 L2 表维度必然与生产写入向量对齐，
无需手动指定维度。切换后端后跑一次本脚本即可重建。

Usage:
    python scripts/l2_rebuild.py --dry-run     # preview facts without writing
    python scripts/l2_rebuild.py               # rebuild (drops memories table)
    python scripts/l2_rebuild.py --yes         # skip confirmation
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from hermes_env import bootstrap, env_path, env_int, setup_logging

logger = setup_logging("l2_rebuild")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# 项目根目录：让 scripts 独立运行也能 import plugin.memory_governed
sys.path.insert(0, str(SCRIPT_DIR.parent))

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")
L2_DB_PATH = env_path("L2_DB_PATH", MEMORY_DIR / "l2")
DAYS_BACK = env_int("L2_REBUILD_DAYS", 0)  # 0 = all history


def _load_backend():
    """Resolve the embedding backend via EmbeddingService (same as runtime).

    优先级：API（embedding.provider + base_url + api_key_env）→ 本地模型
    （vector.model）→ 不可用。返回 (service, dim)；不可用时 service 为 None。
    dim 是后端**实际**维度（API 由返回向量长度探测，本地由模型输出维度探测）。
    """
    try:
        from plugin.memory_governed._config import load_governed_config
        from plugin.memory_governed._embedding import EmbeddingService
    except ImportError as e:
        logger.error("cannot import plugin modules (run from project root): %s", e)
        return None, 0

    cfg = load_governed_config(HERMES_HOME)
    service = EmbeddingService.get(cfg)
    if not service.available:
        logger.error(
            "no embedding backend available (%s) — configure API "
            "(embedding.provider/base_url/api_key_env) or a local model "
            "(vector.model), then retry", service.last_error)
        return None, service.dim
    logger.info("embedding backend: %s (dim=%d)", service.backend_name, service.dim)
    return service, service.dim


def load_l3_messages() -> list[dict[str, Any]]:
    """Read all messages from L3 (source of truth)."""
    import sqlite3

    if not L3_DB_PATH.exists():
        logger.warning("L3 database not found: %s", L3_DB_PATH)
        return []

    conn = sqlite3.connect(f"file:{L3_DB_PATH}?mode=ro", uri=True)
    cutoff_ts = 0.0
    if DAYS_BACK > 0:
        cutoff_ts = datetime.now().timestamp() - DAYS_BACK * 86400
    rows = conn.execute(
        """SELECT id, session_id, role, content, timestamp FROM messages
           WHERE timestamp > ? AND role IN ('user', 'assistant') AND content != ''
           ORDER BY timestamp ASC""",
        (cutoff_ts,),
    ).fetchall()
    conn.close()

    messages = [
        {"role": role, "content": content, "timestamp": ts, "_l3_rowid": rid, "_session_id": sid}
        for rid, sid, role, content, ts in rows
    ]
    logger.info("loaded %d messages from L3", len(messages))
    return messages


def extract_facts(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract atomic facts (same heuristics as WriteQueue._extract_atomic_facts)."""
    import re

    facts: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        for sentence in re.split(r"[.!?。！？\n]", content):
            sentence = sentence.strip()
            if len(sentence) < 10 or len(sentence) > 500:
                continue
            if sentence.endswith(("?", "？")):
                continue
            if sentence.startswith(("Please", "Do ", "Don't ", "请", "不要")):
                continue
            facts.append({
                "content": sentence,
                "category": "rebuild",
                "source": "l2-rebuild",
                "timestamp": datetime.fromtimestamp(msg["timestamp"]).isoformat()
                             if msg.get("timestamp") else datetime.now().isoformat(),
                "source_rowid": msg.get("_l3_rowid"),
            })
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild L2 from L3")
    parser.add_argument("--dry-run", action="store_true", help="Preview facts without writing")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation")
    args = parser.parse_args()

    messages = load_l3_messages()
    facts = extract_facts(messages)
    logger.info("extracted %d facts from %d messages", len(facts), len(messages))

    # dry-run 只预览 facts，不解析 embedding 后端、不碰 lancedb —— 这样在没有
    # vector 依赖（无 fastembed/lancedb/API）的环境也能跑（test_dry_run 依赖此行为）。
    if args.dry_run:
        for f in facts[:20]:
            print(f"  - [{f['source_rowid']}] {f['content'][:80]}")
        print(f"... total {len(facts)} facts")
        return 0

    if not facts:
        logger.warning("no facts extracted — nothing to rebuild")
        return 0

    # 真正重建才需要 embedding 后端（API → 本地 → 文本，维度取实际值）
    service, dim = _load_backend()
    if service is None:
        return 1

    if not args.yes:
        answer = input(f"Drop and rebuild L2 table with {len(facts)} facts (dim={dim})? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            logger.info("aborted")
            return 1

    # Generate embeddings via the SAME backend the runtime uses.
    # embed_batch 返回与输入对齐的列表，失败项为 None（该 fact 保留 null 向量，
    # 与生产 _sync._index_l2 行为一致，后续可被 null-vector 迁移补齐）。
    vectors = service.embed_batch([f["content"] for f in facts])
    embedded = 0
    for fact, vec in zip(facts, vectors):
        if vec:
            fact["vector"] = list(vec)
            embedded += 1
    if embedded == 0:
        logger.error("embedding failed for all facts — check backend config / API key / network")
        return 1
    logger.info("embedded %d/%d facts", embedded, len(facts))

    # Drop and rebuild the table
    import lancedb
    import pyarrow as pa

    L2_DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(L2_DB_PATH))
    if "memories" in db.list_tables().tables:
        db.drop_table("memories")
        logger.info("dropped old memories table")

    schema = pa.schema([
        pa.field("content", pa.string()),
        pa.field("category", pa.string()),
        pa.field("source", pa.string()),
        pa.field("timestamp", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("source_rowid", pa.int64()),
    ])
    table = db.create_table("memories", schema=schema)
    table.add(facts)
    logger.info("L2 rebuild complete: %d rows", table.count_rows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
