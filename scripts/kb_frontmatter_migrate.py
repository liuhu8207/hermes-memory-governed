#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""为存量笔记回填 frontmatter ``status``（双库分工契约的存量迁移）。

status 契约（与 ``plugin._kb._status_for_section`` 同一规则，由
``tests/test_kb_governance.py`` 钉住两侧一致）：

* ``inbox`` / ``knowledge`` → ``draft``（自动区）
* ``archive`` → ``archived``
* 其余（含根级笔记）→ ``curated``（人工区）

行为约定：

- 默认 **dry-run**：只报告将写入什么，一个文件都不碰（批量操作先报 scope）。
- ``--apply`` 前先落 **vault 外快照**（``$HERMES_HOME/memory/vault_backups/<ts>/``）：
  vault 是人工策展的正本、通常不在版本控制下，**拿不到快照就拒绝写入**。
- ``--apply`` 才落盘：原子写（tmp + ``os.replace``），保留原正文与全部既有
  字段，只补 ``status``。
- **幂等**：已有 ``status`` 的笔记跳过 —— 重复执行是零写入。

用法::

    python scripts/kb_frontmatter_migrate.py            # dry-run 报告
    python scripts/kb_frontmatter_migrate.py --apply
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

# 与 hgm_mcp / wb_*_hook 同一惯例：repo root 进 sys.path 后再 import CLI，
# 保持脚本可以从任意 cwd 执行。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import memory_cli


def _plan(config: dict):
    """扫描全库，返回 ``(would_update, already, errors)``。

    ``would_update`` 每项是 ``(path, meta, body, status, section)``；
    ``already`` 是已有 ``status`` 的笔记数（幂等计数）。
    """
    would = []
    already = 0
    errors = []
    vault = memory_cli.wiki_dir(config)
    for p in memory_cli.iter_notes(config):
        try:
            meta, body = memory_cli.parse_frontmatter(
                p.read_text(encoding="utf-8", errors="replace"))
        except OSError as e:
            errors.append(f"{p}: {e}")
            continue
        if str(meta.get("status") or "").strip():
            already += 1
            continue
        try:
            rel = p.relative_to(vault)
            section = rel.parts[0] if len(rel.parts) > 1 else ""
        except ValueError:
            section = ""
        would.append((p, meta, body,
                      memory_cli._status_for_section(section),
                      section or "(root)"))
    return would, already, errors


def _apply_one(path: Path, meta: dict, body: str, status: str) -> None:
    """补 ``status`` 后原子写回 —— 与 ``cmd_kb_add`` 同一写法。"""
    new_meta = dict(meta)
    new_meta["status"] = status
    text = memory_cli.dump_frontmatter(new_meta) + "\n\n" + body.rstrip() + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _snapshot(paths, vault: Path, dest_root: Path) -> Path | None:
    """把将要改写的笔记**整份复制到 vault 之外**，成功返回快照目录。

    vault 是人工策展的正本、通常不在版本控制下 —— 原地批量改写一旦出错就
    无从还原。宁可多占一份磁盘，不可没有退路（同 ``l2_rebuild`` 的
    「拿不到备份就拒绝 drop」）。
    """
    dest = dest_root / time.strftime("%Y%m%d_%H%M%S")
    try:
        for p in paths:
            target = dest / p.relative_to(vault)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
    except (OSError, ValueError) as e:
        print(json.dumps({"ok": False, "error": f"snapshot failed: {e}"},
                         ensure_ascii=False), file=sys.stderr)
        return None
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill frontmatter status for existing vault notes "
                    "(dry-run by default)")
    parser.add_argument("--apply", action="store_true",
                        help="Actually write the status field (default: report only)")
    args = parser.parse_args()

    config = memory_cli.load_config()
    would, already, errors = _plan(config)

    out = {
        "ok": True,
        "applied": bool(args.apply),
        "vault": str(memory_cli.wiki_dir(config)),
        "would_update": len(would),
        "already_status": already,
        "by_status": dict(Counter(item[3] for item in would)),
        "by_section": dict(Counter(item[4] for item in would)),
        "errors": errors,
        "updated": [],
    }

    if args.apply:
        if would:
            # 先落 vault 外快照，拿不到就**拒绝写入**（一条都不碰）。
            dest_root = (Path(memory_cli.hermes_home())
                         / "memory" / "vault_backups")
            snap = _snapshot([p for p, *_ in would],
                             memory_cli.wiki_dir(config), dest_root)
            if snap is None:
                out.update({"ok": False, "applied": False,
                            "error": "refused: could not snapshot the notes "
                                     "before writing — nothing was touched"})
                print(json.dumps(out, ensure_ascii=False, indent=2))
                return 1
            out["backup_dir"] = str(snap)
        for path, meta, body, status, _section in would:
            try:
                _apply_one(path, meta, body, status)
                out["updated"].append(str(path))
            except OSError as e:
                errors.append(f"{path}: {e}")
        out["updated_count"] = len(out["updated"])

    # 批量操作先报 scope：无论 dry-run 还是 apply，计数都在最前面可见。
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
