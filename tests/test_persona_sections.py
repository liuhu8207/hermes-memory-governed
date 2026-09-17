# -*- coding: utf-8 -*-
"""The two copies of the "derived persona sections" list must not drift.

There are two implementations of the same rule on purpose:

* ``plugin/memory_governed/_persona.py`` — used by ``_recall.get_l4``, which
  feeds both ``system_prompt_block`` and the ``score=1.0`` L4 recall result;
* ``scripts/wb_session_hook.py`` — a host adapter, which cannot import the
  plugin package without dragging in ``_recall`` and ``_kb``, and must keep
  running under a bare interpreter in under a second.

Duplication that cannot be removed is duplication that must be pinned. If these
two lists diverge, one surface silently starts injecting a dump of every L2 fact
again — and the symptom is not an error, it is "the memory tools are never
used", which took a full round of investigation to trace the first time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from plugin.memory_governed._persona import (         # noqa: E402
    DERIVED_SECTIONS,
    persona_only,
)

PERSONA = ("# User Profile\n_Generated: 2026-01-01T00:00:00_\n"
           "## User\n喜欢结构化输出\n\n"
           "## Knowledge Areas\n- tech: 2 facts\n\n"
           "## Known Facts\n- ⚠️ **必须同步 `rsa.key`**\n- 内网地址是 10.0.0.1\n\n"
           "## Stats\n- Conversations archived: 329\n")


def _hook():
    spec = importlib.util.spec_from_file_location(
        "wb_session_hook_under_test", REPO / "scripts" / "wb_session_hook.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheTwoListsAgree:
    def test_section_lists_are_identical(self):
        assert tuple(DERIVED_SECTIONS) == tuple(_hook()._DERIVED_PERSONA_SECTIONS), (
            "两份『派生段落』名单不一致 —— 有一边会重新注入 L2 全量转储")

    def test_both_implementations_produce_the_same_text(self):
        assert persona_only(PERSONA) == _hook()._persona_only(PERSONA)


class TestPersonaOnly:
    def test_the_dump_is_removed(self):
        out = persona_only(PERSONA)
        for marker in ("Knowledge Areas", "Known Facts", "Stats", "rsa.key", "10.0.0.1"):
            assert marker not in out, marker

    def test_the_profile_survives(self):
        assert "喜欢结构化输出" in persona_only(PERSONA)

    def test_a_hand_written_persona_passes_through(self):
        plain = "# User Profile\n只看这一句"
        assert persona_only(plain) == plain

    @pytest.mark.parametrize("raw", ["", None, "   ", 123])
    def test_empty_or_non_string_input(self, raw):
        assert persona_only(raw) == ""

    def test_a_later_section_alone_still_truncates(self):
        """Stats can appear without the fact listing; it must still be cut."""
        text = "# User Profile\n简介\n## Stats\n- Conversations archived: 1\n"
        assert persona_only(text) == "# User Profile\n简介"


class TestTheSingleReadPoint:
    """``get_l4`` is where both injection paths converge — so it is where to fix."""

    def test_get_l4_strips_the_dump(self, tmp_path, monkeypatch):
        from plugin.memory_governed._recall import RecallEngine

        persona = tmp_path / "persona.md"
        persona.write_text(PERSONA, encoding="utf-8")

        engine = RecallEngine.__new__(RecallEngine)     # no heavy construction
        engine._l4_cache = None
        engine._l4_cache_time = 0.0

        class Cfg:
            l4_persona_path = str(persona)

        engine._config = Cfg()
        out = engine.get_l4()
        assert "喜欢结构化输出" in out
        assert "rsa.key" not in out, "get_l4 仍在返回 L2 转储 —— 两条注入路都会中招"

    def test_the_cache_holds_the_stripped_text(self, tmp_path):
        from plugin.memory_governed._recall import RecallEngine

        persona = tmp_path / "persona.md"
        persona.write_text(PERSONA, encoding="utf-8")
        engine = RecallEngine.__new__(RecallEngine)
        engine._l4_cache = None
        engine._l4_cache_time = 0.0

        class Cfg:
            l4_persona_path = str(persona)

        engine._config = Cfg()
        engine.get_l4()
        assert "rsa.key" not in engine._l4_cache
