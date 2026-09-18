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


#: The guarded directory is decided by HERMES_HOME, so the tests fix it
#: rather than hard-coding a path. Before the guard was scoped to that directory
#: every test used a made-up path and still passed — because the guard matched on
#: the file name alone, which is the bug this suite now pins.
FAKE_HERMES_HOME = "C:/fake-hermes"
L1_DIR = FAKE_HERMES_HOME + "/memory"


@pytest.fixture(autouse=True)
def _fake_hermes_home(monkeypatch):
    """Point the guard at a predictable directory, and never log for real."""
    monkeypatch.setenv("HERMES_HOME", FAKE_HERMES_HOME)
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
            r"C:\fake-hermes\memory\MEMORY.md",
            "C:/fake-hermes/memory/MEMORY.md",
        ):
            decision, reason, _, hit = hook.decide(edit(path))
            assert decision == "deny", f"{path} 没被拦住"
            assert hit == "MEMORY.md"
            assert "hgm_remember" in reason, "拒绝理由没有告诉它该用什么"

    @pytest.mark.parametrize("name", ["USER.md", "persona.md", "persona_meta.json"])
    def test_the_other_protected_files_too(self, name):
        decision, _, _, hit = hook.decide(edit(rf"C:\fake-hermes\memory\{name}"))
        assert decision == "deny" and hit == name

    def test_writing_a_protected_file_is_refused(self):
        payload = {"tool_name": "Write",
                   "tool_input": {"file_path": "C:/fake-hermes/memory/persona.md",
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
        "cat C:/fake-hermes/memory/MEMORY.md",
        "grep -n schtasks ~/AppData/Local/hermes/memory/MEMORY.md",
        "ls -l C:/fake-hermes/memory/",
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
            bash('echo "新规则" >> C:/fake-hermes/memory/MEMORY.md'))
        assert decision == "deny" and hit == "MEMORY.md"

    def test_naming_a_protected_file_without_writing_is_allowed(self):
        """Both conditions, or the guard blocks its own diagnostics."""
        assert hook.decide(bash("wc -l ./memory/MEMORY.md"))[0] == "allow"


class TestTheFilesystemSpelling:
    """The guard must match the *file*, not the bytes.

    Measured 2026-09-17, the literal-text match let two spellings of the same
    file through: ``memory.md`` / ``Memory.MD`` (this filesystem is
    case-insensitive) and ``MEMORY.md.`` (Windows strips a trailing dot). A rule
    that can be sidestepped by holding Shift is not being enforced.
    """

    @pytest.mark.parametrize("path", [
        r"C:\fake-hermes\memory\memory.md",
        "C:/fake-hermes/memory/Memory.MD",
        "C:/fake-hermes/memory/mEmOrY.mD",
    ])
    def test_another_case_is_the_same_file(self, path):
        decision, _, _, hit = hook.decide(edit(path))
        assert decision == "deny" and hit == "MEMORY.md", path

    @pytest.mark.parametrize("path", [
        "C:/fake-hermes/memory/MEMORY.md.",
        r"C:\fake-hermes\memory\USER.md...",
        "C:/fake-hermes/memory/persona.md.",
    ])
    def test_a_trailing_dot_is_the_same_file(self, path):
        decision, _, _, hit = hook.decide(edit(path))
        assert decision == "deny" and hit, path

    def test_the_shell_path_is_normalised_too(self):
        assert hook.decide(
            bash("echo x >> C:/fake-hermes/memory/memory.md"))[0] == "deny"

    def test_a_near_miss_is_still_a_different_file(self):
        """Normalising must not swallow names that are *not* the protected file."""
        for path in (r"D:\proj\MEMORY.md.bak", r"D:\proj\not-MEMORY.md",
                     "D:/proj/MEMORY.md.old", r"D:\proj\MY-MEMORY.md.txt"):
            assert hook.decide(edit(path))[0] == "allow", path


class TestBothToolNamingStyles:
    """The hook reference warns the received ``tool_name`` depends on the host.

    The CLI uses ``Edit`` / ``Write`` / ``Bash``; the IDE uses
    ``replace_in_file`` / ``write_to_file`` / ``execute_command``. Measured
    2026-09-17 this environment sends CLI-style names, so the guard worked — but
    under the other style it would have answered "allow" to everything, with no
    error anywhere. A rule that silently stops being enforced is this project's
    recurring failure, so both spellings are accepted.
    """

    @pytest.mark.parametrize("tool", ["replace_in_file", "write_to_file",
                                      "multi_replace_in_file", "insert_content"])
    def test_ide_style_file_writers_are_recognised(self, tool):
        payload = {"tool_name": tool,
                   "tool_input": {"filePath": "C:/fake-hermes/memory/MEMORY.md",
                                  "content": "x"}}
        decision, reason, _, hit = hook.decide(payload)
        assert decision == "deny" and hit == "MEMORY.md", (
            f"{tool} 是 IDE 风格名，没被识别 —— 守卫会静默失效")

    def test_ide_style_shell_is_recognised(self):
        payload = {"tool_name": "execute_command",
                   "tool_input": {"command": "echo x >> C:/fake-hermes/memory/USER.md"}}
        assert hook.decide(payload)[0] == "deny"

    def test_the_camel_case_path_key_is_read(self):
        """The IDE example in the reference uses ``filePath``, not ``file_path``."""
        assert "filePath" in hook.PATH_KEYS

    def test_cli_style_still_works(self):
        assert hook.decide(edit("C:/fake-hermes/memory/MEMORY.md"))[0] == "deny"
        assert hook.decide(bash("echo x >> C:/fake-hermes/memory/MEMORY.md"))[0] == "deny"


class TestItIsScopedToTheL1Directory:
    """The guard must not fire on a file that merely *shares a name*.

    Measured 2026-09-18: this project keeps its own workspace memory at
    ``.workbuddy/memory/MEMORY.md``, and the host's instructions require agents
    to maintain it. The first version of the guard matched on the base name
    alone, so it refused its own author's attempt to update that file. That is
    the over-blocking failure worth avoiding: a guard that stops legitimate work
    gets switched off, and then nothing is guarded.
    """

    L1 = "C:/fake-hermes/memory"
    WORKSPACE = "D:/Sync/Drive/proj/.workbuddy/memory"

    @pytest.mark.parametrize("path", [
        L1 + "/MEMORY.md",
        L1 + "/memory.md",                      # case-insensitive filesystem
        L1 + "/MEMORY.md.",                     # trailing dot is stripped by FS
        L1.replace("/", "\\") + "\\MEMORY.md",  # Windows separator
        L1 + "/USER.md",
        L1 + "/persona.md",
    ])
    def test_the_real_l1_files_are_still_refused(self, path):
        assert hook.protected_hit(path), f"{path} 没被拦住 —— 洞还在"

    @pytest.mark.parametrize("path", [
        WORKSPACE + "/MEMORY.md",
        "./.workbuddy/memory/MEMORY.md",
        "D:/proj/MEMORY.md",
        str(REPO / ".workbuddy/memory/MEMORY.md"),
    ])
    def test_a_same_named_file_elsewhere_is_allowed(self, path):
        assert not hook.protected_hit(path), (
            f"{path} 被误拦 —— 它不在 L1 目录里，只是重名")

    def test_the_l1_directory_itself_is_not_a_file_hit(self):
        assert not hook.protected_hit(self.L1 + "/memory")

    def test_a_bare_name_is_refused(self):
        """The hook cannot see the working directory, so assume the worst."""
        assert hook.protected_hit("MEMORY.md")

    def test_the_default_dir_is_used_when_hermes_home_is_unset(self, monkeypatch):
        """A hook's environment need not carry HERMES_HOME.

        With it unset the guarded directory is derived from the user's home, and
        a ``~``-spelled path must compare equal to the absolute one — otherwise
        the L1 file is protected only when it is written the long way round.
        """
        monkeypatch.delenv("HERMES_HOME", raising=False)
        assert hook.protected_hit("~/AppData/Local/hermes/memory/MEMORY.md")
        assert hook.protected_hit(str(Path.home() / "AppData" / "Local"
                                     / "hermes" / "memory" / "USER.md"))
        assert not hook.protected_hit("~/somewhere/else/MEMORY.md")

    def test_the_protected_dir_follows_hermes_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        assert hook.protected_hit(str(tmp_path / "memory" / "MEMORY.md"))
        assert not hook.protected_hit(self.L1 + "/MEMORY.md"), (
            "换了 HERMES_HOME 之后，别处的同名文件不该还算受保护")

    def test_a_shell_redirect_into_l1_is_still_refused(self):
        assert hook.decide({"tool_name": "Bash",
                            "tool_input": {"command":
                                           f'echo "x" >> {self.L1}/MEMORY.md'}})[0] == "deny"

    def test_a_shell_write_to_the_workspace_copy_is_allowed(self):
        assert hook.decide({"tool_name": "Bash",
                            "tool_input": {"command":
                                           f'echo "x" >> {self.WORKSPACE}/MEMORY.md'}})[0] == "allow"

    def test_editing_the_workspace_copy_is_allowed(self):
        for tool in ("Edit", "Write"):
            payload = {"tool_name": tool,
                       "tool_input": {"file_path": self.WORKSPACE + "/MEMORY.md"}}
            assert hook.decide(payload)[0] == "allow", tool


class TestNamingIsNotWriting:
    """Mentioning a guarded file is not the same as writing to it.

    Measured 2026-09-18, two false positives in a row made the guard block its own
    author: a command that merely *printed* a line naming MEMORY.md (because bare
    ``echo`` counted as a write indicator), and a command that tried to *back the
    file up* (because the name appeared anywhere in the string). Both are the
    over-blocking failure — a guard that stops real work gets switched off, and
    then nothing is guarded.
    """

    L1 = "C:/fake-hermes/memory"

    @pytest.mark.parametrize("command", [
        'echo "MANUAL 说明：不要手改 MEMORY.md"',
        "grep -n TODO " + "C:/fake-hermes/memory/MEMORY.md",
        "wc -c < C:/fake-hermes/memory/MEMORY.md",
        "diff C:/fake-hermes/memory/MEMORY.md /tmp/other",
    ])
    def test_reading_or_printing_is_allowed(self, command):
        assert hook.decide(bash(command))[0] == "allow", command

    def test_backing_the_file_up_is_allowed(self):
        """The destination decides, not the mention."""
        assert hook.decide(bash(
            "cp C:/fake-hermes/memory/MEMORY.md D:/backups/MEMORY.md.bak"))[0] == "allow"

    def test_copying_into_the_l1_file_is_refused(self):
        assert hook.decide(bash(
            "cp D:/other/MEMORY.md C:/fake-hermes/memory/MEMORY.md"))[0] == "deny"

    @pytest.mark.parametrize("command", [
        "rm C:/fake-hermes/memory/MEMORY.md",
        "mv D:/x " + "C:/fake-hermes/memory/USER.md",
        "sed -i s/a/b/ C:/fake-hermes/memory/persona.md",
    ])
    def test_writing_with_the_target_last_is_refused(self, command):
        assert hook.decide(bash(command))[0] == "deny", command

    def test_bare_echo_is_not_a_write_indicator(self):
        assert "echo " not in hook.WRITE_TOKENS and "printf " not in hook.WRITE_TOKENS, (
            "echo/printf 不带重定向不是写 —— 它们会把只打印一行的命令也拦下")


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
            edit("C:/fake-hermes/memory/MEMORY.md")), capsys)
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
