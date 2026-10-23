# -*- coding: utf-8 -*-
"""Backup retention standard and its wiring into cron."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "backup_retention.py"


def _load():
    spec = importlib.util.spec_from_file_location("backup_retention_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _dir_with_mb(path: Path, mb: int = 1):
    path.mkdir(parents=True, exist_ok=True)
    (path / "data.bin").write_bytes(b"x" * (mb * 1024 * 1024))


def test_stamp_comes_from_the_name_not_mtime(tmp_path):
    mod = _load()
    p = tmp_path / "memories.lance_retract_20261021_095138"
    p.mkdir()
    os.utime(p, (datetime(2020, 1, 1).timestamp(),
                 datetime(2020, 1, 1).timestamp()))
    assert mod._stamp(p) == datetime(2026, 10, 21, 9, 51, 38)


def test_l2_keeps_recent_months_and_milestones(tmp_path):
    mod = _load()
    paths = []
    for day in range(1, 20):
        p = tmp_path / ("memories.lance_retract_202610%02d_120000" % day)
        p.mkdir()
        paths.append(p)
    milestone = tmp_path / "memories.lance_cleanup-20260901_20260901_000000"
    milestone.mkdir()
    paths.append(milestone)

    keep, drop = mod._select_l2(paths)

    assert len(keep) >= mod.L2_KEEP_RECENT
    assert milestone in keep, "milestone backups must never be pruned"
    assert any("20261001" in p.name for p in drop)
    assert all(p not in drop for p in keep)


def test_apply_deletes_only_outside_the_keep_set(tmp_path, monkeypatch, capsys):
    mod = _load()
    home = tmp_path / "home"
    memory = home / "memory"
    wiki = tmp_path / "wiki"
    (memory / "l2_backups").mkdir(parents=True)
    (memory / "l3_backups").mkdir()
    (memory / "kb_index").mkdir()
    wiki.mkdir()
    (home / "governed_memory.json").write_text(
        json.dumps({"wiki_dir": str(wiki), "l2_db_path": str(memory / "l2")}),
        encoding="utf-8")

    for day in range(1, 12):
        _dir_with_mb(memory / "l2_backups" /
                     ("memories.lance_retract_202610%02d_120000" % day), 1)

    monkeypatch.setattr(mod, "HERMES_HOME", home)
    monkeypatch.setattr(mod, "MEMORY_DIR", memory)
    monkeypatch.setattr(sys, "argv", ["backup_retention.py", "--dry-run"])
    assert mod.main() == 0
    assert len(list((memory / "l2_backups").iterdir())) == 11, "dry-run must not delete"

    monkeypatch.setattr(sys, "argv",
                        ["backup_retention.py", "--apply", "--yes"])
    assert mod.main() == 0
    assert len(list((memory / "l2_backups").iterdir())) <= 10


def test_cron_runs_backup_retention():
    path = Path(__file__).resolve().parent.parent / "scripts" / "cron_memory_daily.py"
    spec = importlib.util.spec_from_file_location("cron_backup_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    steps = {s.label: s for s in mod.STEPS}
    assert "backup-retention" in steps
    assert "--apply" in steps["backup-retention"].args


def test_all_vault_backups_move_outside_obsidian(tmp_path, monkeypatch, capsys):
    mod = _load()
    home = tmp_path / "home"
    memory = home / "memory"
    wiki = tmp_path / "wiki"
    memory.mkdir(parents=True)
    wiki.mkdir()
    (home / "governed_memory.json").write_text(
        json.dumps({"wiki_dir": str(wiki), "l2_db_path": str(memory / "l2")}),
        encoding="utf-8")
    for stamp in ("20261021_120000", "20261022_120000"):
        backup = wiki / f".kb-backups-{stamp}"
        backup.mkdir()
        (backup / "a.md").write_text("body", encoding="utf-8")

    monkeypatch.setattr(mod, "HERMES_HOME", home)
    monkeypatch.setattr(mod, "MEMORY_DIR", memory)
    monkeypatch.setattr(sys, "argv",
                        ["backup_retention.py", "--apply", "--yes"])
    assert mod.main() == 0
    capsys.readouterr()
    assert not list(wiki.glob(".kb-backups-*"))
    assert len(list((memory / "vault_backups").glob(".kb-backups-*"))) == 2
