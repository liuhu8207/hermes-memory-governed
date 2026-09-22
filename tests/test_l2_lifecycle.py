# -*- coding: utf-8 -*-
# ruff: noqa: F811 — pytest fixture 跨文件复用：`store` 由 import 引入，
# 各测试以参数消费它时被静态分析视为重绑定；这是 pytest 的标准模式。
"""L2 记忆生命周期测试（Phase 3 / WeKnora 借鉴）。

覆盖五件事：

1. **retract（supersede）**：归档 JSONL（state=superseded + reason）→
   内容指纹墓碑 → 从表中删除；``--dry-run`` 只列不动；
2. **防复活**：被撤回的事实再次 ``remember`` 一律拒绝并点名原 reason；
3. **容量淘汰**：``sync.l2_max_items``（默认 0=关）开启后，将超限时把
   **最少使用**的同 agent 事实降级归档（state=archived，**不写墓碑** ——
   被挤掉的可以再写回来）；
4. **净化漏斗**：凭据 fail-closed 拒绝、长事实包含性查重、召回命中 touch
   （仅容量开启时落表）；
5. **health** 报告 ``l2_lifecycle`` 状态计数。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import memory_cli as cli
from plugin.memory_governed import _lifecycle as life

# 复用同文件里的 HERMES_HOME store fixture 与假嵌入（绝不外呼）
from tests.test_cli_security_and_recall import (  # noqa: F401 — pytest fixture
    _install_fake_embedding,
    store,
)

# 这两句在既有测试里被 external_write_verdict 放行 —— 用同一对，避免
# 测试失败被误读成门禁变化；第三句沿用同样的「X 而不是 Y」形状。
_FACT_A = "家用NAS 192.0.2.62 上重度使用 Docker 部署服务"
_FACT_B = "SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE"
_FACT_C = "构建机 build-07 使用 Ubuntu 24.04 而不是 Debian 12 编译内核模块"
_FACT_LONG = (
    _FACT_A + "，必须在周五之前完成迁移并且迁移窗口固定在凌晨两点到四点之间，"
    "超过窗口自动回滚，所有卷在迁移前都要做一次快照备份，"
    "迁移完成后由值班同学在工单系统里确认并关闭变更单，"
    "过程中任何异常都要先止损再排查根因。"
)


def _table_contents(store) -> list:
    lancedb = pytest.importorskip("lancedb")
    db = lancedb.connect(str(store.home / "memory" / "l2"))
    return db.open_table("memories").to_arrow()["content"].to_pylist()


def _memory_dir(store) -> Path:
    return store.home / "memory"


def _set_config(store, extra: dict) -> None:
    cfg = {"wiki_dir": str(store.vault),
           "kb": {"confidence_threshold": 0.7}}
    cfg.update(extra)
    (store.home / "governed_memory.json").write_text(
        json.dumps(cfg, ensure_ascii=False), encoding="utf-8")


def _archive_records(store) -> list:
    path = life.archive_path(_memory_dir(store))
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# 1) retract + 墓碑防复活
# ---------------------------------------------------------------------------

class TestRetract:
    def test_retract_archives_tombstones_and_deletes(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        assert cli.cmd_remember({}, _FACT_A, "dsh")["ok"] is True
        assert cli.cmd_remember({}, _FACT_B, "dsh")["ok"] is True

        out = cli.cmd_retract({}, "ROCKET_TLS", reason="outdated",
                              agent="tester")
        assert out["ok"] is True, out
        assert out["retracted"] == 1 and out["tombstones"] == 1

        remaining = _table_contents(store)
        assert _FACT_A in remaining and _FACT_B not in remaining

        records = _archive_records(store)
        assert len(records) == 1
        assert records[0]["state"] == "superseded"
        assert records[0]["reason"] == "outdated"
        assert records[0]["content"] == _FACT_B

        tombs = life.load_tombstones(_memory_dir(store))
        assert life.content_fp(_FACT_B) in tombs
        assert tombs[life.content_fp(_FACT_B)]["reason"] == "outdated"

    def test_remember_refuses_tombstoned_fact(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_B, "dsh")
        cli.cmd_retract({}, "ROCKET_TLS", reason="wrong info", agent="tester")

        out = cli.cmd_remember({}, _FACT_B, "dsh")
        assert out["ok"] is False
        # 拒绝理由必须点名这是墓碑拦的，而不是别的什么门禁
        assert "tombstone" in str(out).lower()
        assert "wrong info" in str(out)
        # 没有复活：表里依然没有它
        assert _FACT_B not in _table_contents(store)

    def test_dry_run_lists_matches_without_touching_anything(
            self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_B, "dsh")

        out = cli.cmd_retract({}, "ROCKET_TLS", dry_run=True)
        assert out["ok"] is True and out["dry_run"] is True
        assert out["matched"] == 1
        assert _FACT_B in _table_contents(store)
        assert life.load_tombstones(_memory_dir(store)) == {}
        assert _archive_records(store) == []

    def test_no_match_is_success_with_zero(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_A, "dsh")
        out = cli.cmd_retract({}, "不存在的内容片段")
        assert out["ok"] is True and out["retracted"] == 0
        assert out.get("note") == "no fact matched"

    def test_empty_match_is_refused(self, store):
        out = cli.cmd_retract({}, "  ")
        assert out["ok"] is False


# ---------------------------------------------------------------------------
# 2) 容量淘汰（demote-not-delete）
# ---------------------------------------------------------------------------

class TestCapacity:
    def test_disabled_by_default(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        for fact in (_FACT_A, _FACT_B, _FACT_C):
            out = cli.cmd_remember({}, fact, "dsh")
            assert out["ok"] is True, out
            assert "demoted" not in out
        assert len(_table_contents(store)) == 3

    def test_overflow_demotes_least_recently_used(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _set_config(store, {"sync": {"l2_max_items": 2}})

        cli.cmd_remember({}, _FACT_A, "dsh")
        cli.cmd_remember({}, _FACT_B, "dsh")
        out = cli.cmd_remember({}, _FACT_C, "dsh")
        assert out["ok"] is True, out
        # 两条都没被召回过 → 平手按写入顺序，最早写的 A 先走
        assert len(out.get("demoted", [])) == 1
        assert out["demoted"][0].startswith("家用NAS")

        remaining = _table_contents(store)
        assert _FACT_A not in remaining
        assert _FACT_B in remaining and _FACT_C in remaining

        records = _archive_records(store)
        assert len(records) == 1
        assert records[0]["state"] == "archived"
        assert records[0]["reason"] == "capacity-demoted"
        # 容量淘汰不写墓碑：被挤掉的事实没有做错什么
        assert life.load_tombstones(_memory_dir(store)) == {}

    def test_demoted_fact_can_be_remembered_again(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _set_config(store, {"sync": {"l2_max_items": 1}})
        cli.cmd_remember({}, _FACT_A, "dsh")
        cli.cmd_remember({}, _FACT_B, "dsh")  # A 被挤出
        assert _FACT_A not in _table_contents(store)
        again = cli.cmd_remember({}, _FACT_A, "dsh")
        assert again["ok"] is True, again
        assert _FACT_A in _table_contents(store)

    def test_recall_touch_is_gated_by_capacity(self, store, monkeypatch):
        """容量关 → touch 零副作用；容量开 → touch 落 l2_usage 表。"""
        _install_fake_embedding(monkeypatch)
        usage_db = _memory_dir(store) / "l2_usage.db"

        cli._touch_l2_usage([_FACT_A])
        assert not usage_db.exists()

        _set_config(store, {"sync": {"l2_max_items": 2}})
        cli._touch_l2_usage([_FACT_A, _FACT_A])
        entries = life.usage_last_used(_memory_dir(store))
        assert entries[life.content_fp(_FACT_A)] > 0

    def test_demote_uses_usage_order_when_available(self, store, monkeypatch):
        """B 被召回过、A 没有 → 淘汰的是 A（last_used 优先于写入顺序）。"""
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        _set_config(store, {"sync": {"l2_max_items": 2}})
        cli.cmd_remember({}, _FACT_A, "dsh")
        cli.cmd_remember({}, _FACT_B, "dsh")
        # 反转天平：只有 A 被 touch 过
        life.usage_touch(_memory_dir(store), [_FACT_A])
        out = cli.cmd_remember({}, _FACT_C, "dsh")
        assert out["demoted"][0].startswith("SecretStore")
        remaining = _table_contents(store)
        assert _FACT_A in remaining and _FACT_B not in remaining


# ---------------------------------------------------------------------------
# 3) 净化漏斗：凭据 + 包含性查重
# ---------------------------------------------------------------------------

class TestRememberFunnel:
    def test_secret_fact_is_refused(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        out = cli.cmd_remember(
            {}, "构建机的 token = sk-AbCdEf123456789XY", "dsh")
        assert out["ok"] is False
        assert "secret" in str(out).lower()

    def test_long_containment_is_a_duplicate(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        first = cli.cmd_remember({}, _FACT_LONG, "dsh")
        assert first["ok"] is True, first
        # 超集句（≥120 字归一化）→ duplicate_of，不再插一行
        superset = _FACT_LONG + "另外所有机器的时钟必须统一指向内网 NTP 服务器。"
        second = cli.cmd_remember({}, superset, "dsh")
        assert second["ok"] is True and second.get("duplicate") is True
        assert second.get("duplicate_of")
        assert len(_table_contents(store)) == 1

    def test_short_containment_is_not_flagged(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_A, "dsh")
        # 短句：包含判断不适用（只有精确重复才算 duplicate）
        out = cli.cmd_remember({}, _FACT_A, "dsh")
        assert out.get("duplicate") is True
        assert "duplicate_of" not in out


# ---------------------------------------------------------------------------
# 4) health 生命周期计数
# ---------------------------------------------------------------------------

class TestHealthLifecycle:
    def test_health_reports_lifecycle_counts(self, store, monkeypatch):
        pytest.importorskip("lancedb")
        _install_fake_embedding(monkeypatch)
        cli.cmd_remember({}, _FACT_B, "dsh")
        cli.cmd_retract({}, "ROCKET_TLS", reason="r", agent="tester")

        out = cli.cmd_health(store.cfg)
        lc = out.get("l2_lifecycle")
        assert isinstance(lc, dict), out
        assert lc.get("tombstones") == 1
        assert lc.get("archived") == 1
        assert "usage_rows" in lc and "max_items" in lc
        # active 是表行数（撤回后剩 0）或 unknown（表读不到）
        assert lc.get("active") in (0, "unknown")
