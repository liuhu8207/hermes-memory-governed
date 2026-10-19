"""Retracted facts must stay retracted on every write path."""

from __future__ import annotations

from pathlib import Path

from plugin.memory_governed._lifecycle import add_tombstone, content_fp, is_tombstoned
from plugin.memory_governed._sync import filter_tombstoned_facts

ORIGINAL = (
    "HGM 的 L2 表 memories 新增 project 列，NULL 表示全局事实，"
    "recall 加 --project 时保留本项目行与全局行、屏蔽其他项目。"
)


def _tombstones(memory_dir: Path) -> dict:
    add_tombstone(
        memory_dir,
        content_fp(ORIGINAL),
        {
            "content_prefix": ORIGINAL[:80],
            "reason": "test retraction",
            "retracted_by": "tester",
            "ts": "2026-10-19T00:00:00",
        },
    )
    from plugin.memory_governed._lifecycle import load_tombstones

    return load_tombstones(memory_dir)


def test_is_tombstoned_matches_exact_and_extended_variant(tmp_path):
    tombstones = _tombstones(tmp_path)
    assert is_tombstoned(ORIGINAL, tombstones) is not None
    assert is_tombstoned(ORIGINAL + "另：写入前先备份。", tombstones) is not None
    assert is_tombstoned("完全无关的另一条事实，长度足够但不共享前缀。", tombstones) is None


def test_filter_tombstoned_facts_drops_exact_and_variant(tmp_path):
    _tombstones(tmp_path)
    facts = [
        {"content": ORIGINAL},
        {"content": ORIGINAL + "补充一句"},
        {"content": "Hermes Gateway 重启必须用 schtasks /run /tn Hermes_Gateway。"},
    ]
    kept = filter_tombstoned_facts(facts, tmp_path)
    assert len(kept) == 1
    assert kept[0]["content"].startswith("Hermes Gateway")


def test_cmd_remember_refuses_tombstoned_variant(hermes_home, monkeypatch):
    memory_dir = hermes_home / "memory"
    _tombstones(memory_dir)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    import memory_cli

    out = memory_cli.cmd_remember(
        memory_cli.load_config(),
        ORIGINAL + "另：写入前先备份。",
        "codex",
        project="",
    )
    assert out["ok"] is False
    assert out["reason"] == "tombstoned"
