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

⚠️ 署名（provenance）必须活过重建：

重建先 DROP 再 CREATE。若 CREATE 的 schema 少了 ``role`` / ``agent`` /
``project``，灾难恢复跑一次就会抹掉所有行的归属证据，而
``l2_apply_gate._row_roles`` 在 ``role`` 列缺失时**一律回退按 user 判定** ——
外部 agent 靠结构证据（不是强信号）进来的行，被 user 的尺子一量就是不合格，
下一次跑门禁会被整批删掉。也就是：一次重建 = 先抹掉归属证据，再据此把不属
于判据的行删光。

所以本脚本做三件事：

1. schema 复用 ``_sync.l2_provenance_fields()``（与建表 / 迁移同一份定义）；
2. DROP 之前先备份（整目录拷贝，表读不出来也能拷），并把备份路径显著打出来；
3. 外部 agent 写进来的行（``source_rowid`` 为 NULL，无法从 L3 重新抽取）
   原样回填，连它们的 ``agent`` / ``project`` 一起。

Usage:
    python scripts/l2_rebuild.py --dry-run     # preview facts without writing
    python scripts/l2_rebuild.py               # rebuild (drops memories table)
    python scripts/l2_rebuild.py --yes         # skip the interactive confirmation
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Backups live OUTSIDE the lancedb directory on purpose: lancedb lists every
# subdirectory of its root as a table, so ``<l2>/memories.bak`` would show up
# as a table named "memories.bak" and confuse every later listing.
BACKUP_DIR = env_path("L2_BACKUP_DIR", MEMORY_DIR / "l2_backups")

def _rebuilt_agent() -> str:
    """Agent name stamped on rows reconstructed from L3 dialogue.

    Read from ``_sync._LOCAL_AGENT`` rather than restated: those rows came from
    the in-process writer, and a second copy of its name is exactly the kind of
    drift that already cost this codebase a column. External agents keep their
    own name because their rows are carried over, not reconstructed.
    """
    try:
        from plugin.memory_governed._sync import _LOCAL_AGENT
        return _LOCAL_AGENT
    except Exception:  # noqa: BLE001 — dry-run must work without the plugin
        return "hermes"

#: Width of the warning banner printed before a drop.
_BANNER = 78


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
    """Extract atomic facts (same heuristics as WriteQueue._extract_atomic_facts).

    Each fact carries its provenance, because a rebuild that drops the columns
    also drops the evidence of who wrote the row — see the module docstring.
    ``project`` stays NULL: a sentence lifted out of a conversation has no
    reliable project attribution, and guessing one would be worse than
    admitting we do not know.
    """
    import re

    facts: list[dict[str, Any]] = []
    agent = _rebuilt_agent()
    for msg in messages:
        content = msg.get("content", "")
        if not content:
            continue
        role = msg.get("role") or "user"
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
                "role": role,
                "agent": agent,
                "project": None,
            })
    return facts


def _table_names(db) -> List[str]:
    """List table names across the two lancedb listing APIs."""
    listing = db.list_tables().tables if hasattr(db, "list_tables") else db.table_names()
    return [t.name if hasattr(t, "name") else str(t) for t in listing]


def _snapshot_rows(db, table: str) -> List[Dict[str, Any]]:
    """Read every row of ``table`` into plain dicts (best effort).

    Returns ``[]`` when the table cannot be read — which is exactly the case a
    rebuild is invoked for, so failing to snapshot must not abort the run. The
    filesystem backup below still protects the data.
    """
    try:
        return db.open_table(table).to_arrow().to_pylist()
    except Exception as e:  # noqa: BLE001 — corruption is the reason we are here
        logger.warning("cannot read existing table for carry-over (%s) — "
                       "rows will not be preserved", e)
        return []


def _table_dir(l2_db_path: Path, table: str) -> Optional[Path]:
    """Locate the on-disk directory holding ``table``.

    lancedb lays a table down as ``<name>.lance`` (current) or ``<name>``
    (older), and the two spellings have both been seen in the wild — guessing
    one is how a "backup" silently copies nothing.
    """
    for candidate in (l2_db_path / f"{table}.lance", l2_db_path / table):
        if candidate.is_dir():
            return candidate
    return None


def _backup_table(l2_db_path: Path, table: str) -> Optional[Path]:
    """Copy the table directory aside before it is dropped.

    A logical export (``to_arrow``) is useless on a corrupted table, which is
    the one case that matters here, so the backup is a byte-level directory
    copy: it succeeds whenever the files are still on disk.
    """
    src = _table_dir(l2_db_path, table)
    if src is None:
        logger.error("cannot locate the directory of table '%s' under %s — "
                     "refusing to drop it", table, l2_db_path)
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"{src.name}_{stamp}"
    try:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    except OSError as e:
        logger.error("BACKUP FAILED (%s) — refusing to drop %s", e, src)
        return None
    logger.info("backed up %s -> %s", src, dest)
    return dest


def _announce_drop(table: str, row_count: int, backup: Optional[Path],
                   carried: int) -> None:
    """Print the drop as the destructive operation it is. Never silent."""
    line = "=" * _BANNER
    logger.warning(line)
    logger.warning("DESTRUCTIVE: about to DROP table '%s' (%d rows)", table, row_count)
    if backup is not None:
        logger.warning("backup: %s", backup)
    else:
        logger.warning("backup: NONE — the table directory could not be copied")
    logger.warning("provenance: role / agent / project columns are preserved; "
                   "%d external-agent row(s) will be carried over", carried)
    logger.warning(line)


def _external_rows(rows: List[Dict[str, Any]],
                   rebuilt_contents: set) -> List[Dict[str, Any]]:
    """Rows that cannot be re-derived from L3, kept as they were.

    An external agent's write has ``source_rowid`` NULL (it never came from a
    dialogue turn), so no amount of re-reading L3 will bring it back. Dropping
    it would silently discard another agent's memory — and, because those are
    exactly the rows whose admission rests on structural evidence rather than a
    strong signal, they could never get back in.
    """
    carried: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("source_rowid") is not None:
            continue
        content = (row.get("content") or "").strip()
        if not content or content in rebuilt_contents:
            continue
        if not row.get("vector"):
            # A row without a vector is unrecallable anyway; re-embedding it
            # here would attribute someone else's text to this script's run.
            continue
        carried.append(row)
    return carried


def _project_to_schema(rows: List[Dict[str, Any]],
                       field_names: List[str]) -> List[Dict[str, Any]]:
    """Drop keys the target schema does not declare (lancedb rejects them)."""
    return [{k: v for k, v in row.items() if k in field_names} for row in rows]


def _confirm(prompt: str) -> bool:
    """Ask for confirmation; a non-interactive stdin counts as 'no'."""
    try:
        answer = input(prompt)
    except (EOFError, OSError):
        logger.error("no interactive stdin — pass --yes to rebuild non-interactively")
        return False
    return answer.strip().lower() in ("y", "yes")


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

    import lancedb
    import pyarrow as pa

    from plugin.memory_governed._sync import l2_provenance_fields

    L2_DB_PATH.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(L2_DB_PATH))
    table = "memories"

    # --- 备份 + 显著提示（drop 之前，绝不静默） -----------------------------
    existing_rows: List[Dict[str, Any]] = []
    backup: Optional[Path] = None
    if table in _table_names(db):
        existing_rows = _snapshot_rows(db, table)
        backup = _backup_table(L2_DB_PATH, table)
        if backup is None:
            logger.error("aborting: refusing to drop '%s' without a backup", table)
            return 1
        carried = _external_rows(existing_rows, {(f.get("content") or "").strip()
                                                 for f in facts})
        _announce_drop(table, len(existing_rows), backup, len(carried))
    else:
        logger.info("no existing '%s' table — nothing to back up", table)
        carried = []

    if not args.yes:
        if not _confirm(f"Drop and rebuild L2 table with {len(facts)} facts (dim={dim})? [y/N] "):
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

    # --- drop and rebuild ---------------------------------------------------
    if table in _table_names(db):
        db.drop_table(table)
        logger.info("dropped old memories table")

    schema = pa.schema([
        pa.field("content", pa.string()),
        pa.field("category", pa.string()),
        pa.field("source", pa.string()),
        pa.field("timestamp", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        # One definition, three consumers: create-table, the plugin's legacy
        # backfill, and the CLI's backfill. Restating the list here is how
        # `project` went missing from fresh tables in the first place.
        *l2_provenance_fields(),
    ])
    field_names = [f.name for f in schema]
    table_obj = db.create_table(table, schema=schema)
    table_obj.add(_project_to_schema(facts, field_names))
    logger.info("L2 rebuilt: %d rows", table_obj.count_rows())

    if carried:
        table_obj.add(_project_to_schema(carried, field_names))
        logger.info("carried over %d external-agent row(s) with their provenance",
                    len(carried))

    final = db.open_table(table)
    logger.info("L2 rebuild complete: %d rows", final.count_rows())
    return 0


if __name__ == "__main__":
    sys.exit(main())
