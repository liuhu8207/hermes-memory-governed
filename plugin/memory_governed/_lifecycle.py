# -*- coding: utf-8 -*-
"""L2 记忆生命周期的共享原语（Phase 3 / WeKnora 借鉴）。

状态机（记录落位即状态，不在 LanceDB 行上加列 —— 避免 schema 迁移）：

* **active** —— 在 ``memories`` 表里，可被召回；
* **superseded** —— 被 ``retract`` 撤回：正文移入 archive JSONL（带撤回
  reason），内容指纹进墓碑，**后续自动抽取/remember 不得复活**；
* **archived** —— 容量淘汰（``sync.l2_max_items``）：最少使用的行移入
  archive JSONL（reason=capacity-demoted），**不写墓碑** —— 被挤掉的事实
  没有做错什么，重新写入是合法操作；只有明确撤回的事实才禁止复活。

三个存储都在 ``<l2_db_path>/..``（即 ``memory/``）下：

* ``l2_tombstones.json`` —— ``{fp: {content_prefix, reason, agent, ts}}``
* ``l2_archive.jsonl``   —— 逐行 JSON 的归档记录（append-only）
* ``l2_usage.db``        —— SQLite ``{fp, hits, last_used}``，召回命中即
  touch；容量淘汰按 ``last_used`` 选受害者（最少使用者先走）

CLI（``memory_cli``）与插件（``_recall``）共用本模块 —— 生命周期逻辑有且
只有一份实现，否则两个入口会给出两种答案（本仓库的老问题）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._text import normalize_for_match

logger = logging.getLogger(__name__)

_TOMBSTONES_NAME = "l2_tombstones.json"
_ARCHIVE_NAME = "l2_archive.jsonl"
_USAGE_NAME = "l2_usage.db"


def content_fp(text: str) -> str:
    """内容指纹：归一化（小写 + 折叠空白）后 sha256。

    归一化走 ``_text.normalize_for_match`` —— **同一份实现**，因为 KB 查重也
    用它；重新抽取生成的**同句**事实必然命中同一指纹，改写（paraphrase）不命中
    是可接受的边界（墓碑记的是「这句话被明确撤回过」，不是语义等价检测）。
    """
    norm = normalize_for_match(text)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def tombstones_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / _TOMBSTONES_NAME


def archive_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / _ARCHIVE_NAME


def usage_db_path(memory_dir: Path) -> Path:
    return Path(memory_dir) / _USAGE_NAME


def load_tombstones(memory_dir: Path) -> Dict[str, Dict[str, Any]]:
    """``{fp: record}``；文件缺失/损坏时返回空表（读路径绝不抛错）。

    损坏时打 warning 而不是静默：墓碑丢了意味着被撤回的事实可能复活，
    这是需要被人看见的事件，不是可以吞掉的异常。
    """
    path = tombstones_path(memory_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
        logger.warning("l2 tombstones file is not a dict: %s", path)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("l2 tombstones unreadable (%s): %s", path, e)
    return {}


def add_tombstone(memory_dir: Path, fp: str,
                  record: Dict[str, Any]) -> None:
    """追加一条墓碑（原子写；已存在同指纹时不覆盖 —— 最早的 reason 为准）。"""
    current = load_tombstones(memory_dir)
    if fp in current:
        return
    current[fp] = record
    path = tombstones_path(memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(path)


def archive_append(memory_dir: Path, record: Dict[str, Any]) -> None:
    """归档记录 append 一行 JSON（状态机的落盘处）。失败向上抛 —— 归档
    是删除动作的一部分，静默丢归档 = 数据丢失。"""
    path = archive_path(memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def archive_count(memory_dir: Path) -> int:
    path = archive_path(memory_dir)
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# -- 使用度（容量淘汰的受害者选择依据） -------------------------------------

def backup_table(l2_db_path, *, tag: str,
                 memory_dir: Optional[Path] = None,
                 table: str = "memories") -> Optional[Path]:
    """破坏性删除前，把表目录**字节级**复制到 ``memory/l2_backups/``。

    为什么不是 ``to_arrow()`` 导出：**损坏的表根本导不出来**，而「损坏」恰恰
    是备份最要紧的那一个情形（同 ``scripts/l2_rebuild.py::_backup_table``）。

    找不到表目录或复制失败 ⇒ 返回 ``None``，**调用方必须据此中止删除** ——
    本仓库对破坏性操作一贯「拿不到备份就拒绝动手」（见 ``l2_rebuild`` 的
    drop 守卫）。归档 JSONL 不是还原点：它**不含 ``vector``**，要还原得重新
    嵌入；这一点此前全仓没有写明。
    """
    base = Path(l2_db_path)
    src: Optional[Path] = None
    for candidate in (base / f"{table}.lance", base / table):
        if candidate.is_dir():
            src = candidate
            break
    if src is None:
        logger.error("找不到表目录（%s 下没有 %s.lance）—— 拒绝在没有备份的情况下删除",
                     base, table)
        return None
    dest_root = Path(memory_dir) if memory_dir is not None else base.parent
    dest = (dest_root / "l2_backups"
            / f"{src.name}_{tag}_{datetime.now():%Y%m%d_%H%M%S}")
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    except OSError as e:
        logger.error("表备份失败（%s）—— 拒绝删除 %s", e, src)
        return None
    logger.info("破坏性操作前已备份 %s -> %s", src, dest)
    return dest


def _usage_connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=1.0)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS l2_usage ("
        "fp TEXT PRIMARY KEY,"
        "hits INTEGER NOT NULL DEFAULT 0,"
        "last_used REAL NOT NULL DEFAULT 0)"
    )
    return conn


def usage_touch(memory_dir: Path, contents: List[str]) -> None:
    """召回命中即计数；失败静默（统计绝不影响读路径）。"""
    contents = [c for c in (contents or []) if c]
    if not contents:
        return
    conn = None
    try:
        conn = _usage_connect(usage_db_path(memory_dir))
        now = time.time()
        for c in contents:
            conn.execute(
                "INSERT INTO l2_usage(fp, hits, last_used) VALUES(?, 1, ?)"
                " ON CONFLICT(fp) DO UPDATE SET"
                " hits = hits + 1, last_used = excluded.last_used",
                (content_fp(c), now),
            )
        conn.commit()
    except Exception as e:  # noqa: BLE001
        logger.debug("l2 usage touch failed: %s", e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("l2 usage conn close failed: %s", e)


def usage_last_used(memory_dir: Path) -> Dict[str, float]:
    """``{fp: last_used}``；失败返回空表（容量淘汰退回时间序）。"""
    conn = None
    try:
        conn = _usage_connect(usage_db_path(memory_dir))
        rows = conn.execute("SELECT fp, last_used FROM l2_usage").fetchall()
        return {str(fp): float(lu or 0.0) for fp, lu in rows}
    except Exception as e:  # noqa: BLE001
        logger.debug("l2 usage read failed: %s", e)
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("l2 usage conn close failed: %s", e)


def usage_count(memory_dir: Path) -> int:
    conn = None
    try:
        conn = _usage_connect(usage_db_path(memory_dir))
        row = conn.execute("SELECT COUNT(*) FROM l2_usage").fetchone()
        return int(row[0]) if row else 0
    except Exception:  # noqa: BLE001
        return 0
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("l2 usage conn close failed: %s", e)


def sql_quote(value: str) -> str:
    """LanceDB delete 谓词用的字符串字面量（单引号翻倍）。"""
    return "'" + str(value).replace("'", "''") + "'"


def lifecycle_counts(memory_dir: Path) -> Dict[str, int]:
    """health 报告用的生命周期计数（全离线文件，不碰表）。"""
    return {
        "tombstones": len(load_tombstones(memory_dir)),
        "archived": archive_count(memory_dir),
        "usage_rows": usage_count(memory_dir),
    }


def resolve_memory_dir(cfg) -> Optional[Path]:
    """从插件配置推导 ``memory/`` 目录；``l2_db_path`` 缺失时返回 ``None``。

    与 ``_kb``/``_index_db`` 同一推导（``l2_db_path`` 的父目录），但显式
    返回 ``None`` 而不是退到 ``"."`` —— 生命周期存储写错地方比读不到更糟。
    """
    raw = str(getattr(cfg, "l2_db_path", "") or "").strip()
    if not raw:
        return None
    return Path(raw).parent
