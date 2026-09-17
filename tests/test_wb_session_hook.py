# -*- coding: utf-8 -*-
"""The WorkBuddy ``SessionStart`` hook — contract and lookup order.

Why this file exists
--------------------
The hook is the only surface through which WorkBuddy reaches the governed
store, and until 2026-09-17 it had **no tests at all**. That is how it shipped
reading the working directory from a single field: the IDE hook reference
documents ``cwd`` on the SessionStart payload, the CLI reference does not (it
lists ``permission_mode`` in its place), and an instrumented desktop session
was reported to send only ``session_id / source / transcript_path``.

When the bet lost, the project block disappeared **silently** — and that block
is the only thing that tells an agent its writes are invisible to every other
agent. The regression test for that is
:meth:`TestTheRegression.test_project_block_survives_a_payload_without_cwd`.
"""
from __future__ import annotations

import importlib.util
import io
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK_PY = REPO / "scripts" / "wb_session_hook.py"


def _load_hook():
    """Import the hook by path — it lives under ``scripts/``, not a package."""
    spec = importlib.util.spec_from_file_location("wb_session_hook", HOOK_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load_hook()

#: This checkout, in the two shapes the hook deals in. **Derived, never
#: hardcoded** — the repo is public, so the suite must not carry the operator's
#: path, and a derived value also happens to be correct on any other machine.
REPO_WIN = str(REPO).replace("\\", "/")
REPO_MSYS = ("/" + REPO_WIN[0].lower() + REPO_WIN[2:]
             if REPO_WIN[1:2] == ":" else REPO_WIN)

#: A neutral pair for the pure string-rewriting tests. Those only check the
#: separators and the drive letter, so the path need not exist — and must not
#: name a real machine. Tests that need the project walk to *resolve* use
#: ``REPO_WIN`` instead.
MSYS_REPO = "/d/repos/hermes-memory-governed"
WIN_REPO = "D:/repos/hermes-memory-governed"


# -- to_windows_path --------------------------------------------------------
class TestToWindowsPath:
    @pytest.mark.parametrize("raw, expected", [
        ("/d/repos/x", "D:/repos/x"),
        ("/c/Users/a", "C:/Users/a"),
        ("/e/", "E:/"),
        ("D:/already", "D:/already"),
        ("relative/path", "relative/path"),
        ("", ""),
        ("   ", ""),
    ])
    def test_conversion(self, raw, expected):
        assert hook.to_windows_path(raw) == expected

    def test_a_posix_path_would_be_unusable_verbatim(self):
        """The reason the conversion exists, stated as an assertion."""
        assert hook.to_windows_path(MSYS_REPO).startswith("D:/")
        assert not hook.to_windows_path(MSYS_REPO).startswith("/")


# -- resolve_cwd ------------------------------------------------------------
class TestResolveCwd:
    def test_payload_wins_over_everything(self, monkeypatch):
        monkeypatch.setenv("CODEBUDDY_PROJECT_DIR", "/c/elsewhere")
        cwd, source = hook.resolve_cwd({"cwd": WIN_REPO})
        assert (cwd, source) == (WIN_REPO, "payload.cwd")

    def test_falls_back_to_the_documented_env_var(self, monkeypatch):
        monkeypatch.setenv("CODEBUDDY_PROJECT_DIR", MSYS_REPO)
        cwd, source = hook.resolve_cwd({})
        assert cwd == WIN_REPO
        assert source == "CODEBUDDY_PROJECT_DIR"

    def test_falls_back_to_the_sibling_runtime_var(self, monkeypatch):
        monkeypatch.delenv("CODEBUDDY_PROJECT_DIR", raising=False)
        monkeypatch.setenv("CLAUDE_PROJECT_DIR", MSYS_REPO)
        cwd, source = hook.resolve_cwd({})
        assert cwd == WIN_REPO
        assert source == "CLAUDE_PROJECT_DIR"

    def test_working_directory_is_the_last_resort(self, monkeypatch):
        monkeypatch.delenv("CODEBUDDY_PROJECT_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(hook.os, "getcwd", lambda: WIN_REPO)
        cwd, source = hook.resolve_cwd({})
        assert (cwd, source) == (WIN_REPO, "os.getcwd()")

    def test_unusable_getcwd_degrades_to_empty_not_raise(self, monkeypatch):
        monkeypatch.delenv("CODEBUDDY_PROJECT_DIR", raising=False)
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)

        def boom():
            raise OSError("cwd is gone")

        monkeypatch.setattr(hook.os, "getcwd", boom)
        assert hook.resolve_cwd({}) == ("", "")

    def test_blank_values_are_not_treated_as_answers(self, monkeypatch):
        monkeypatch.setenv("CODEBUDDY_PROJECT_DIR", "   ")
        monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
        monkeypatch.setattr(hook.os, "getcwd", lambda: WIN_REPO)
        cwd, source = hook.resolve_cwd({"cwd": "  "})
        assert (cwd, source) == (WIN_REPO, "os.getcwd()")

    def test_source_is_always_reported(self, monkeypatch):
        """A fallback that cannot name itself cannot be debugged."""
        monkeypatch.setenv("CODEBUDDY_PROJECT_DIR", MSYS_REPO)
        for payload in ({}, {"cwd": WIN_REPO}):
            cwd, source = hook.resolve_cwd(payload)
            assert cwd and source


# -- build_context ----------------------------------------------------------
L1 = {"memory_rules_md": "# Memory Rules\n必须用 schtasks 重启 gateway",
      "user_profile_md": "# User\n某市",
      "persona_md": ""}


class TestPersonaDumpIsNotInjected:
    """L4 appends a dump of every L2 fact; injecting it defeats the tool surface.

    Measured 2026-09-17: 20 of 23 L2 facts were already present in the session
    injection, because ``persona.md`` ends with ``## Known Facts`` — a listing
    the plugin's own ``_sync.py`` calls a dump. So an agent had no reason ever to
    call ``hgm_recall``, and since that listing is not filtered by ``project``,
    facts scoped to one project reached every session — the exact leak the scope
    exists to prevent.
    """

    PERSONA = ("# User Profile\n_Generated: 2026-01-01T00:00:00_\n"
               "## User\n喜欢结构化输出\n\n"
               "## Knowledge Areas\n- tech: 2 facts\n\n"
               "## Known Facts\n- ⚠️ **必须同步 `rsa.key`**\n- 内网地址是 10.0.0.1\n\n"
               "## Stats\n- Conversations archived: 329\n")

    def test_the_facts_listing_is_stripped(self):
        out = hook._persona_only(self.PERSONA)
        assert "Known Facts" not in out
        assert "rsa.key" not in out
        assert "Knowledge Areas" not in out
        assert "Stats" not in out

    def test_the_real_persona_survives(self):
        out = hook._persona_only(self.PERSONA)
        assert "喜欢结构化输出" in out

    def test_it_does_not_invent_sections_that_are_absent(self):
        plain = "# User Profile\n只看这一句"
        assert hook._persona_only(plain) == plain

    @pytest.mark.parametrize("raw", ["", None, "   "])
    def test_empty_input(self, raw):
        assert hook._persona_only(raw) == ""

    def test_build_context_drops_the_dump(self):
        ctx = hook.build_context({"persona_md": self.PERSONA}, "")
        assert "用户画像" in ctx          # the section is still there…
        assert "rsa.key" not in ctx      # …but the dump is not
        assert "喜欢结构化输出" in ctx

    def test_a_fact_scoped_to_a_project_no_longer_leaks(self):
        """The isolation half of the same bug: the dump ignored ``project``."""
        scoped = {"persona_md": self.PERSONA + "\n## Known Facts\n- 只属于某项目的内部地址 10.9.9.9\n"}
        assert "10.9.9.9" not in hook.build_context(scoped, "")


class TestBuildContext:
    def test_project_block_appears_for_a_known_project(self):
        ctx = hook.build_context(L1, REPO_WIN)
        assert "### 当前项目" in ctx
        assert "hermes-memory-governed" in ctx

    def test_project_block_is_omitted_without_a_directory(self):
        """No cwd means no guess — an empty answer beats a wrong one."""
        ctx = hook.build_context(L1, "")
        assert "### 当前项目" not in ctx
        assert "手写规则" in ctx

    def test_a_directory_without_project_markers_is_not_invented(self):
        ctx = hook.build_context(L1, "C:/Windows")
        assert "### 当前项目" not in ctx

    def test_cap_is_stated_not_silent(self):
        fat = {"memory_rules_md": "x" * (hook.MAX_CHARS * 2)}
        ctx = hook.build_context(fat, WIN_REPO)
        assert len(ctx) <= hook.MAX_CHARS + 200
        assert "已截断" in ctx

    def test_placeholder_only_sections_are_dropped(self):
        ctx = hook.build_context({"memory_rules_md": "<!-- 直接编辑此文件 -->"}, "")
        assert "<!--" not in ctx

    def test_the_block_still_tells_the_agent_about_the_store(self):
        ctx = hook.build_context(L1, WIN_REPO)
        assert "hgm_recall" in ctx         # the tool surface is named…
        assert "hgm_kb_search" in ctx      # …including the knowledge base
        assert "不可写" in ctx             # the boundary is stated…
        assert "互不感知" in ctx           # WorkBuddy's own memory is separate

    def test_the_boundary_says_it_is_enforced_and_where_to_go(self):
        """Stating a rule the agent can break is what failed once already.

        Measured 2026-09-17: an agent asked to remember a rule edited L1 directly
        instead of calling the tool. The wording now names the enforcement and
        the two sanctioned routes, so a reader cannot conclude that hand-editing
        is merely discouraged.
        """
        ctx = hook.build_context(L1, WIN_REPO)
        assert "PreToolUse" in ctx, "没有说明会被钩子拦"
        assert "会被 PreToolUse 钩子直接拒绝" in ctx or "会被拒绝" in ctx
        assert "hgm_remember" in ctx, "只说了不许，没说该用什么"
        assert "告诉用户" in ctx, "立规则这条路没有出口"

    def test_it_names_the_tools_before_the_cli(self):
        """The instruction is what the agent follows — measured, not assumed.

        The first version of this block told the agent to run
        ``python memory_cli.py recall`` and never mentioned the MCP tools. A turn
        later the host had listed the tools 13 times and called them **zero**
        times: the agent did exactly what it was told. So the tools must be the
        first thing named, and the CLI must be framed as the fallback.

        Compared *inside the instruction block*, not across the whole payload:
        L1's own text happens to mention ``memory_cli.py`` in a stored rule, and
        a whole-payload comparison would have been measuring that instead. The
        first version of this test passed for exactly that wrong reason.
        """
        section = hook.build_context(L1, "").split("### 这个存储里还有什么", 1)[1]
        assert "hgm_recall" in section and "memory_cli.py" in section
        assert section.index("hgm_recall") < section.index("memory_cli.py"), (
            "CLI 出现在 MCP 工具之前 —— agent 会照着先出现的那个做")
        assert "没有 MCP 工具时" in section, "CLI 没有被写成后备路径"

    def test_the_project_block_also_leads_with_the_tools(self):
        hint = hook.build_context(L1, REPO_WIN).split("### 当前项目", 1)[1]
        assert hint.index("hgm_recall") < hint.index("memory_cli.py")


# -- the contract the host relies on ---------------------------------------
class TestHookContract:
    def _run_main(self, monkeypatch, capsys, raw: str, env: dict):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr(hook, "fetch_l1", lambda: dict(L1))
        monkeypatch.setattr(hook.sys, "stdin", io.StringIO(raw))
        rc = hook.main()
        captured = capsys.readouterr()
        return rc, captured.out, captured.err

    @pytest.mark.parametrize("raw", ["", "   ", "{not json", "[]", "null"])
    def test_malformed_input_still_yields_a_valid_payload(self, monkeypatch, capsys, raw):
        rc, out, _ = self._run_main(monkeypatch, capsys, raw,
                                    {"CODEBUDDY_PROJECT_DIR": MSYS_REPO})
        assert rc == 0
        data = json.loads(out)                     # must parse, whatever arrived
        assert data["continue"] is True
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_normal_payload_shape(self, monkeypatch, capsys):
        rc, out, _ = self._run_main(
            monkeypatch, capsys,
            json.dumps({"session_id": "s1", "cwd": WIN_REPO, "source": "startup"}),
            {})
        assert rc == 0
        assert len(out.strip().splitlines()) == 1   # stdout is JSON and nothing else
        data = json.loads(out)
        ctx = data["hookSpecificOutput"]["additionalContext"]
        assert ctx

    def test_a_debug_line_never_lands_in_stdout(self, monkeypatch, capsys):
        """stderr is opt-in because coalescing streams would corrupt the JSON."""
        rc, out, err = self._run_main(monkeypatch, capsys,
                                      json.dumps({"session_id": "s2"}),
                                      {"HGM_HOOK_DEBUG": "1"})
        json.loads(out)                              # strict: no stray text
        assert "[hgm-hook]" in err
        assert "已注入" in err


class TestTheRegression:
    """The bug this file was written for."""

    def test_project_block_survives_a_payload_without_cwd(self, monkeypatch, capsys):
        """The CLI hook reference documents no ``cwd`` on SessionStart.

        Betting on that one field meant the project block vanished silently
        whenever the payload matched the CLI shape. It must now survive.
        """
        monkeypatch.setenv("CODEBUDDY_PROJECT_DIR", REPO_MSYS)
        monkeypatch.setattr(hook, "fetch_l1", lambda: dict(L1))
        cli_shaped = json.dumps({
            "session_id": "cli-shaped",
            "transcript_path": "/x/y.jsonl",
            "permission_mode": "default",
            "hook_event_name": "SessionStart",
            "source": "startup",
        })
        monkeypatch.setattr(hook.sys, "stdin", io.StringIO(cli_shaped))
        assert hook.main() == 0
        ctx = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
        assert "### 当前项目" in ctx, "项目块在 CLI 形态的 payload 下消失了"
        assert "hermes-memory-governed" in ctx
        if REPO_MSYS != REPO_WIN:      # Windows only: git-bash form must convert
            assert REPO_MSYS not in ctx, "MSYS 路径没有转成 Windows 形态"


@pytest.mark.skipif(shutil.which("sh") is None, reason="needs Git Bash")
class TestEndToEnd:
    """Runs the real shell entry point the way the host does."""

    def test_the_wrapper_emits_json_and_ignores_a_hostile_path(self):
        proc = subprocess.run(
            ["sh", str(REPO / "scripts" / "wb_session_hook.sh")],
            input=json.dumps({"session_id": "e2e", "cwd": REPO_WIN}),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(REPO), timeout=180,
            env={**__import__("os").environ, "PATH": "/usr/bin:/bin"},
        )
        assert proc.returncode == 0
        data = json.loads(proc.stdout)
        ctx = data["hookSpecificOutput"]["additionalContext"]
        assert len(ctx) > 500, "注入退化了（L1 没读到？）"
        assert "### 当前项目" in ctx
