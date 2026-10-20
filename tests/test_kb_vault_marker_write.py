# -*- coding: utf-8 -*-
"""KB 索引 ↔ vault 配对 sidecar（``kb_index/vault_root.json``）**必须写得出来**。

漏洞（QA 复评合成实测）
----------------------
``_record_vault_root()`` 写入前没有 ``mkdir``，且 ``except OSError`` 只
``logger.debug`` ⇒ ``kb_index/`` 目录还不存在时：

* marker ``exists after: False`` —— sidecar **没写出来**；
* ``sync_index`` 仍然报 ``ok: True`` —— **没有任何人知道**。

这个 sidecar 是"索引属于哪个 vault"的**唯一**证据。它静默缺失 ⇒ 配对判定退化
（``index_vault_unknown`` 会长期为真）⇒ 那套 fail-closed 删除闸门的判定依据被
架空。生产上通常踩不到（真 ``KBIndex._init()`` 会建目录），但这是一次**不可见的
失败** —— 正是本项目反复在清的故障族。

两个方向都钉住
--------------
1. **目录不存在也能写出**（写入前建目录）；
2. **写不出去必须被计数**：``_diag.stats()["degraded"]`` 里能数到
   ``kb_index::vault_marker_unwritten``，且打 WARNING —— **不再是静默**。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from plugin.memory_governed import _diag
from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed._kb import KnowledgeBase

#: 期望被计数的降级键。
DEGRADED_KEY = "kb_index::vault_marker_unwritten"


def _make_kb(tmp_path) -> KnowledgeBase:
    """一个最小可用的 KB（真实 vault 目录 + 真实索引路径，不启动向量后端）。"""
    wiki = tmp_path / "wiki"
    (wiki / "notes").mkdir(parents=True)
    (wiki / "notes" / "n00.md").write_text(
        "标题 0\n\n正文 0\n", encoding="utf-8")

    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(wiki)
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    return KnowledgeBase(cfg)


class TestMarkerIsWrittenEvenWhenTheDirectoryIsMissing:
    def test_marker_is_written_when_kb_index_does_not_exist(self, tmp_path):
        """``kb_index/`` **还不存在** ⇒ 照样写出 sidecar（写前建目录）。"""
        kb = _make_kb(tmp_path)
        assert not Path(kb._index_db).exists(), "前提：kb_index/ 还不存在"

        kb._record_vault_root()

        marker = kb._index_vault_marker()
        assert marker.is_file(), "sidecar 没被写出来（不可见的失败）"
        payload = json.loads(marker.read_text(encoding="utf-8"))
        assert payload["vault_root"] == str(Path(kb._vault).resolve())
        assert kb._recorded_vault_root() == str(Path(kb._vault).resolve()), (
            "写出来了却读不出来 ⇒ 配对证据仍然缺失")

    def test_marker_is_written_when_only_the_parent_is_missing(self, tmp_path):
        """连 ``kb_index`` 的**父目录**都不存在 ⇒ 也写出来（``parents=True``）。"""
        kb = _make_kb(tmp_path)
        assert not Path(kb._index_db).parent.exists(), "前提：父目录也不存在"

        kb._record_vault_root()

        assert kb._index_vault_marker().is_file()


class TestFailureIsCountedNotSilent:
    def test_unwritable_marker_is_counted_as_degraded(self, tmp_path):
        """写不出去 ⇒ 计进 ``degraded``（有计数 = 有人知道），而不是 debug 日志。

        把 ``kb_index`` 变成一个**占位文件**：``mkdir`` 与 ``write_text`` 都会抛
        ``OSError`` —— 这是"父目录不可写"的最小复现。
        """
        kb = _make_kb(tmp_path)
        Path(kb._index_db).parent.mkdir(parents=True, exist_ok=True)
        Path(kb._index_db).write_text("占位：让 mkdir / write 都失败",
                                      encoding="utf-8")

        _diag.reset()
        kb._record_vault_root()

        counts = _diag.stats()["degraded"]
        assert counts.get(DEGRADED_KEY) == 1, (
            "sidecar 写失败没有被计数 ⇒ 又是静默：%s" % counts)
        assert not kb._index_vault_marker().is_file()

    def test_unwritable_marker_emits_a_warning(self, tmp_path, caplog):
        """写不出去 ⇒ 打 WARNING（**看得见**），不是 ``debug``。"""
        kb = _make_kb(tmp_path)
        Path(kb._index_db).parent.mkdir(parents=True, exist_ok=True)
        Path(kb._index_db).write_text("占位：让 mkdir / write 都失败",
                                      encoding="utf-8")

        _diag.reset()
        with caplog.at_level(logging.WARNING, logger="hermes_governed.diag"):
            kb._record_vault_root()

        warnings = [record.getMessage() for record in caplog.records
                    if record.name == "hermes_governed.diag"
                    and record.levelno >= logging.WARNING]
        assert warnings, "sidecar 写失败必须看得见（WARNING 级）"
        assert any(DEGRADED_KEY.replace("::", ": ") in message
                   or "vault_marker_unwritten" in message
                   for message in warnings), warnings

    def test_sync_still_returns_after_a_failed_marker(self, tmp_path):
        """写 sidecar 失败**不阻断**同步本身（降级，不是崩溃）。"""
        kb = _make_kb(tmp_path)
        Path(kb._index_db).parent.mkdir(parents=True, exist_ok=True)
        Path(kb._index_db).write_text("占位：让 mkdir / write 都失败",
                                      encoding="utf-8")

        _diag.reset()
        kb._record_vault_root()  # 不抛

        assert _diag.stats()["degraded"].get(DEGRADED_KEY) == 1
