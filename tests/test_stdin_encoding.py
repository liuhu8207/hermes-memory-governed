# -*- coding: utf-8 -*-
"""Every script that reads stdin must say which encoding to read it in.

Why this file exists
--------------------
The MCP server decoded its stdin with the platform default. On 2026-09-18 that
turned out to be cp936 under one host's environment, and the failure had two
faces: either the bytes decoded into mojibake that got written to the store, or
they hit a byte GBK cannot decode and the process died on `for line in sys.stdin:`
with an empty stdout.

The three WorkBuddy hooks have the **same** structure and the **same** exposure —
`raw = sys.stdin.read()` — and their failure mode is worse, because it is quiet:
the read sits inside `except Exception: return {}`, so a decode error produces an
empty payload and the hook carries on as though the host had said nothing. All
three would stop working at once — memory injection, per-turn fact injection, and
the L1 guard — with no error anywhere to notice.

Whether a host hands over a GBK stdin is not something these scripts can control,
and it differs per host: the same server failed under one and not another. So the
rule is not "it works today", it is "say what you mean". Hence a source guard:
a script that reads stdin without reconfiguring it fails this file.
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PY = sys.executable

#: The reconfigure spelling these scripts use.
_RECONFIGURE = re.compile(r"for\s+\w+\s+in\s*\(([^)]*)\)")


def reads_stdin(source: str) -> bool:
    return "sys.stdin" in source


def reconfigures_stdin(source: str) -> bool:
    """True when some stream tuple passed to ``reconfigure`` includes stdin.

    Checked structurally rather than by substring so that a comment mentioning
    ``sys.stdin`` is not mistaken for handling, and so that the tuple may hold the
    streams in any order.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        names = [n.attr if isinstance(n, ast.Attribute) else getattr(n, "id", "")
                 for n in ast.walk(node.iter)]
        if "stdin" not in names:
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Attribute) and inner.attr == "reconfigure"):
                return True
    return False


class TestEveryStdinReaderSaysItsEncoding:
    def _scripts(self):
        return sorted(p for p in SCRIPTS.glob("*.py"))

    def test_no_script_reads_stdin_without_reconfiguring_it(self):
        offenders = [p.name for p in self._scripts()
                     if reads_stdin(p.read_text(encoding="utf-8"))
                     and not reconfigures_stdin(p.read_text(encoding="utf-8"))]
        assert not offenders, (
            f"这些脚本读了 stdin 却没声明编码: {offenders} —— 宿主若给 GBK，"
            f"它们会静默拿到空 payload（§docstring）")

    def test_the_guard_would_catch_the_original_mistake(self):
        """The guard must fail on the pre-fix code, or it guards nothing."""
        pre_fix = (
            "import sys\n"
            "for _stream in (sys.stdout, sys.stderr):\n"
            "    if hasattr(_stream, 'reconfigure'):\n"
            "        _stream.reconfigure(encoding='utf-8')\n"
            "def read_payload():\n"
            "    return sys.stdin.read()\n")
        assert reads_stdin(pre_fix) and not reconfigures_stdin(pre_fix)

    def test_it_accepts_the_fixed_spelling(self):
        fixed = (
            "import sys\n"
            "for _stream in (sys.stdin, sys.stdout, sys.stderr):\n"
            "    if hasattr(_stream, 'reconfigure'):\n"
            "        _stream.reconfigure(encoding='utf-8')\n"
            "def read_payload():\n"
            "    return sys.stdin.read()\n")
        assert reconfigures_stdin(fixed)

    @pytest.mark.parametrize("name", ["wb_prompt_hook.py", "wb_session_hook.py",
                                      "wb_pretool_hook.py", "hgm_mcp.py"])
    def test_the_known_stdin_readers_comply(self, name):
        source = (SCRIPTS / name).read_text(encoding="utf-8")
        assert reads_stdin(source), f"{name} 不再读 stdin —— 测试该跟着改"
        assert reconfigures_stdin(source), f"{name} 没声明 stdin 编码"


class TestItActuallyDecodesUtf8UnderAHostileEnvironment:
    """Source guards catch the spelling; this catches the behaviour.

    ``PYTHONIOENCODING=gbk`` reproduces a host that hands over a GBK stdin. With
    the fix the hook decodes the UTF-8 payload correctly; the log records the
    character count, so a mis-decode or a swallowed decode error is visible as the
    wrong number rather than as a silent empty payload.
    """

    def _run_hook(self, tmp_path, io_encoding):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HERMES_HOME"] = str(tmp_path)
        env["PYTHONIOENCODING"] = io_encoding
        env.pop("HGM_HOOK_LOG", None)
        # The hook logs to ``$HERMES_HOME/memory/hook_log.txt`` and swallows a
        # write failure, so the directory has to exist or the test would read an
        # empty file and blame the encoding for it.
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"session_id": "enc-test",
                              "prompt": "配置在哪里放着呢"}, ensure_ascii=False)
        proc = subprocess.run([PY, str(SCRIPTS / "wb_prompt_hook.py")],
                              input=payload + "\n", capture_output=True, text=True,
                              encoding="utf-8", env=env, cwd=str(REPO), timeout=60)
        log = tmp_path / "memory" / "hook_log.txt"
        row = log.read_text(encoding="utf-8") if log.exists() else ""
        return proc, row

    def test_a_chinese_payload_is_counted_correctly(self, tmp_path):
        proc, row = self._run_hook(tmp_path, "gbk")
        assert proc.returncode == 0, proc.stderr[-400:]
        match = re.search(r"raw_len=(\d+)", row)
        assert match, f"钩子没有记录输入长度（说明它静默返回了空 payload）: {row!r}"
        assert match.group(1) == "8", (
            f"中文提示「配置在哪里放着呢」是 8 个字符，日志记成 {match.group(1)} —— "
            f"说明 stdin 没按 UTF-8 解码")

    def test_the_response_is_still_valid_json(self, tmp_path):
        proc, _ = self._run_hook(tmp_path, "gbk")
        json.loads(proc.stdout)          # a hook must never emit anything else
