# -*- coding: utf-8 -*-
"""The PreToolUse guard that keeps agents out of the hand-written layers.

Why it exists: measured 2026-09-17, an agent asked to "记一下" a rule did not
call ``hgm_remember`` — it edited ``memory/MEMORY.md`` directly (twice, zero
memory-tool calls) and grew L1 from 1374 to 2241 bytes. The contract said L1 is
not agent-writable; nothing enforced it.

The tests below care about two failure directions, and the second matters more:

* **under-blocking** — the edit that started all this must be refused;
* **over-blocking** — a guard that also refuses legitimate work gets switched
  off, and then nothing is guarded. So "an edit whose *text* mentions
  MEMORY.md" must pass, as must every read, and any malformed input must
  answer "allow".
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK = REPO / "scripts" / "wb_pretool_hook.py"


def _load():
    spec = importlib.util.spec_from_file_location("wb_pretool_hook", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load()


@pytest.fixture(autouse=True)
def _no_real_log(monkeypatch):
    """Never let a test append to the real guard log."""
    monkeypatch.setenv("HGM_GUARD_LOG", "0")


def edit(path, new_text="x"):
    return {"tool_name": "Edit",
            "tool_input": {"file_path": path, "old_string": "a", "new_string": new_text}}


def bash(command):
    return {"tool_name": "Bash", "tool_input": {"command": command}}


class TestItBlocksTheRealCase:
    """The write that this guard exists because of."""

    def test_editing_l1_memory_is_refused(self):
        for path in (
            r"C:\Users\example\AppData\Local\hermes\memory\MEMORY.md",
            "C:/Users/example/AppData/Local/hermes/memory/MEMORY.md",
            "~/AppData/Local/hermes/memory/MEMORY.md",
        ):
            decision, reason, _, hit = hook.decide(edit(path))
            assert decision == "deny", f"{path} 没被拦住"
            assert hit == "MEMORY.md"
            assert "hgm_remember" in reason, "拒绝理由没有告诉它该用什么"

    @pytest.mark.parametrize("name", ["USER.md", "persona.md", "persona_meta.json"])
    def test_the_other_protected_files_too(self, name):
        decision, _, _, hit = hook.decide(edit(rf"C:\x\hermes\memory\{name}"))
        assert decision == "deny" and hit == name

    def test_writing_a_protected_file_is_refused(self):
        payload = {"tool_name": "Write",
                   "tool_input": {"file_path": "C:/x/memory/persona.md",
                                  "content": "覆盖它"}}
        assert hook.decide(payload)[0] == "deny"


class TestItDoesNotBlockRealWork:
    """Over-blocking is the failure that gets a guard switched off."""

    def test_a_document_that_merely_mentions_the_file_is_allowed(self):
        """Writing *about* the memory system must keep working.

        This is the guard's own documentation, the project notes, and half the
        reports written while building it.
        """
        payload = edit(r"D:\proj\docs\notes.md",
                       new_text="不要把 `MEMORY.md` 手工改掉，要用 hgm_remember。")
        assert hook.decide(payload)[0] == "allow"

    def test_a_path_that_merely_contains_the_name_is_allowed(self):
        assert hook.decide(edit(r"D:\proj\MEMORY.md.bak"))[0] == "allow"
        assert hook.decide(edit(r"D:\proj\not-MEMORY.md"))[0] == "allow"

    def test_ordinary_files_are_allowed(self, tmp_path):
        for path in (r"D:\proj\README.md", "/tmp/x/main.py", r"C:\x\hermes\config.yaml"):
            assert hook.decide(edit(path))[0] == "allow", path

    @pytest.mark.parametrize("command", [
        "cat C:/Users/x/hermes/memory/MEMORY.md",
        "grep -n schtasks ~/AppData/Local/hermes/memory/MEMORY.md",
        "ls -l C:/Users/x/hermes/memory/",
    ])
    def test_reading_through_the_shell_is_allowed(self, command):
        assert hook.decide(bash(command))[0] == "allow", command

    def test_an_unknown_tool_is_allowed(self):
        assert hook.decide({"tool_name": "DeferExecuteTool",
                            "tool_input": {"name": "x"}})[0] == "allow"

    def test_a_write_elsewhere_through_the_shell_is_allowed(self):
        assert hook.decide(bash("echo hi >> /tmp/notes.txt"))[0] == "allow"


class TestTheShellPath:
    def test_a_redirect_into_l1_is_refused(self):
        decision, _, _, hit = hook.decide(
            bash('echo "新规则" >> C:/Users/x/hermes/memory/MEMORY.md'))
        assert decision == "deny" and hit == "MEMORY.md"

    def test_naming_a_protected_file_without_writing_is_allowed(self):
        """Both conditions, or the guard blocks its own diagnostics."""
        assert hook.decide(bash("wc -l ./memory/MEMORY.md"))[0] == "allow"


class TestTheContract:
    def _run(self, raw, capsys):
        import sys
        monkeypatch_stdin = io.StringIO(raw)
        old = sys.stdin
        sys.stdin = monkeypatch_stdin
        try:
            rc = hook.main()
        finally:
            sys.stdin = old
        return rc, capsys.readouterr().out

    def test_a_denial_is_valid_json_with_the_decision(self, capsys):
        rc, out = self._run(json.dumps(
            edit("C:/x/hermes/memory/MEMORY.md")), capsys)
        assert rc == 0
        data = json.loads(out)
        assert data["continue"] is True
        specific = data["hookSpecificOutput"]
        assert specific["hookEventName"] == "PreToolUse"
        assert specific["permissionDecision"] == "deny"
        assert "hgm_remember" in specific["permissionDecisionReason"]

    def test_an_allow_says_nothing_specific(self, capsys):
        """An explicit "allow" would override a decision the user's own settings
        may have made, so an allowed call carries no hookSpecificOutput."""
        rc, out = self._run(json.dumps(edit(r"D:\proj\a.md")), capsys)
        assert rc == 0
        assert "hookSpecificOutput" not in json.loads(out)

    @pytest.mark.parametrize("raw", ["", "   ", "{not json", "[]", "null"])
    def test_malformed_input_allows_and_stays_valid(self, capsys, raw):
        rc, out = self._run(raw, capsys)
        assert rc == 0
        assert json.loads(out)["continue"] is True, "坏输入不该拦住任何东西"

    def test_it_logs_the_denial_when_logging_is_on(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HGM_GUARD_LOG", "1")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._run(json.dumps(edit(str(tmp_path / "memory" / "MEMORY.md"))), capsys)
        text = (tmp_path / "memory" / "guard_log.txt").read_text(encoding="utf-8")
        assert "deny" in text and "MEMORY.md" in text

    def test_it_can_be_switched_off(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HGM_GUARD_LOG", "0")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        self._run(json.dumps(edit(str(tmp_path / "memory" / "MEMORY.md"))), capsys)
        assert not (tmp_path / "memory" / "guard_log.txt").exists()
