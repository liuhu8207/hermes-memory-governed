# -*- coding: utf-8 -*-
"""CLI 手动写入路径的容量淘汰：与每日 sweep 同一套安全规则。"""

from __future__ import annotations

import pyarrow as pa

import memory_cli
from plugin.memory_governed import _lifecycle as life


class _Table:
    def __init__(self, rows):
        self.rows = list(rows)
        self.predicates = []

    def to_arrow(self):
        cols = {k: [r.get(k) for r in self.rows] for k in ("content", "agent", "timestamp")}
        return pa.table({k: pa.array(v) for k, v in cols.items()})

    def delete(self, predicate: str):
        self.predicates.append(predicate)
        # 测试只关心「谓词是否比 content 更精确」，按内容+时间戳模拟
        import re
        cm = re.search(r"content = '((?:[^']|'')*)'", predicate)
        content = cm.group(1).replace("''", "'") if cm else None
        tm = re.search(r"timestamp = '((?:[^']|'')*)'", predicate)
        stamp = tm.group(1).replace("''", "'") if tm else None
        self.rows = [r for r in self.rows
                     if not (r.get("content") == content
                             and (stamp is None or str(r.get("timestamp")) == stamp))]


class _Cfg:
    def __init__(self, tmp_path):
        self.l2_db_path = str(tmp_path / "l2")


def _patch(monkeypatch, tmp_path):
    class _Life:
        usage_last_used = staticmethod(lambda _d: {})
        content_fp = staticmethod(life.content_fp)
        sql_quote = staticmethod(life.sql_quote)
        backup_table = staticmethod(lambda *a, **k: tmp_path / "backup")
        archive_append = staticmethod(lambda *a, **k: None)
        resolve_memory_dir = staticmethod(lambda cfg: tmp_path)

    monkeypatch.setattr(memory_cli, "plugin_module", lambda name: _Life)
    return _Life


def test_cli_capacity_refuses_to_evict_without_usage_data(monkeypatch, tmp_path):
    rows = [{"content": "old but recalled", "agent": "hermes", "timestamp": "1"},
            {"content": "new", "agent": "hermes", "timestamp": "2"}]
    table = _Table(rows)
    _patch(monkeypatch, tmp_path)

    demoted = memory_cli._l2_enforce_capacity(table, _Cfg(tmp_path), "hermes", 1)

    assert demoted == []
    assert len(table.rows) == 2, "无使用度数据时不得删除"
    assert table.predicates == [], "不得发出任何 delete"


def test_cli_capacity_predicate_is_more_specific_than_content(monkeypatch, tmp_path):
    rows = [
        {"content": "same", "agent": "hermes", "timestamp": "1"},
        {"content": "same", "agent": "hermes", "timestamp": "2"},
        {"content": "other", "agent": "hermes", "timestamp": "3"},
    ]
    table = _Table(rows)
    _Life = _patch(monkeypatch, tmp_path)
    _Life.usage_last_used = staticmethod(
        lambda _d: {life.content_fp("same"): 0.0, life.content_fp("other"): 9.0})
    _Life.archive_append = staticmethod(lambda *a, **k: None)

    demoted = memory_cli._l2_enforce_capacity(table, _Cfg(tmp_path), "hermes", 2)

    assert len(demoted) == 1
    assert table.predicates, "应当发出 delete"
    assert "timestamp =" in table.predicates[0], (
        "只按 content 删会误删同内容的其它行：%r" % table.predicates[0])
    assert len(table.rows) == 2, "只应删掉最早的那一行"
