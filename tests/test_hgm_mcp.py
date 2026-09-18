# -*- coding: utf-8 -*-
"""The HGM MCP server: protocol contract, tool surface, and failure behaviour.

Why this file exists
--------------------
The store was previously reachable from WorkBuddy only through a
``UserPromptSubmit`` hook — an undocumented payload. Measured 2026-09-17: a
6-character message arrived as ``prompt`` of length 9, a 21-character one as
length 28, other runs carried 991, and the session transcript shows the host
wrapping a turn as ~2300 characters of reminders. Brute-forcing the payload's
digest found nothing. MCP replaces that with a schema both sides agree on.

So the tests here care about two things the hook could never offer:

* **the protocol** — a host must be able to handshake, list, and call, and must
  get a well-formed answer to every malformed thing it might send;
* **survival** — an MCP server is long-lived, and a server that dies on one bad
  frame looks to the user like "memory stopped working" rather than like an
  error.

Expensive calls hit the real store on purpose (that is the integration being
tested); the suite keeps them to a minimum.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "scripts" / "hgm_mcp.py"
PY = sys.executable


class Client:
    """A minimal MCP host: one JSON-RPC frame per line, on stdio."""

    def __init__(self, agent: str = "pytest-mcp"):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HGM_AGENT"] = agent
        env["HGM_MCP_LOG"] = "0"          # never write into the real log
        self.proc = subprocess.Popen(
            [PY, str(SERVER)], cwd=str(REPO), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)

    def request(self, method, params=None, msg_id=1):
        frame = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            frame["params"] = params
        self.proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def raw(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def notify(self, method):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def call(self, name, arguments=None, msg_id=99):
        return self.request("tools/call",
                            {"name": name, "arguments": arguments or {}}, msg_id)

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.wait(timeout=30)
        return self.proc.stderr.read()


@pytest.fixture(scope="module")
def client():
    c = Client()
    try:
        yield c
    finally:
        c.close()


class TestHandshake:
    def test_initialize_answers_the_protocol_version_and_capabilities(self, client):
        r = client.request("initialize", {"protocolVersion": "2024-11-05",
                                          "capabilities": {},
                                          "clientInfo": {"name": "pytest", "version": "1"}})
        assert "result" in r, r
        assert r["result"]["serverInfo"]["name"] == "hgm-governed-memory"
        assert "tools" in r["result"]["capabilities"]

    def test_initialized_is_a_notification_and_gets_no_reply(self, client):
        """If the server answered notifications, the host's stream would desync."""
        client.notify("notifications/initialized")
        r = client.request("ping", msg_id=2)
        assert r["id"] == 2 and "result" in r, "通知被回了话，流会错位"

    def test_ping(self, client):
        assert "result" in client.request("ping", msg_id=3)


class TestToolSurface:
    def test_every_tool_is_described_for_a_model_to_choose_from(self, client):
        r = client.request("tools/list", msg_id=4)
        tools = r["result"]["tools"]
        names = {t["name"] for t in tools}
        assert names == {"hgm_recall", "hgm_remember", "hgm_kb_search",
                         "hgm_kb_add", "hgm_transcribe", "hgm_agents"}
        for t in tools:
            assert t["description"].strip(), f"{t['name']} 没有描述，模型无法判断何时用"
            assert t["inputSchema"]["type"] == "object"
            assert "properties" in t["inputSchema"]

    def test_remember_declares_its_arguments(self, client):
        tools = {t["name"]: t for t in client.request("tools/list", msg_id=5)["result"]["tools"]}
        schema = tools["hgm_remember"]["inputSchema"]
        assert "fact" in schema["properties"]
        assert schema["required"] == ["fact"]


class TestErrorPaths:
    """A host may send anything. None of it may take the server down."""

    def test_unknown_method(self, client):
        r = client.request("no/such/method", msg_id=6)
        assert r["error"]["code"] == -32601

    def test_unknown_tool(self, client):
        r = client.call("nope", msg_id=7)
        assert r["error"]["code"] == -32602

    def test_unparseable_frame(self, client):
        assert client.raw("{not json", )["error"]["code"] == -32700

    def test_a_tool_that_raises_becomes_isError_not_a_protocol_error(self, client):
        """The call was well formed; the *work* failed.

        Reporting it as a JSON-RPC error would tell the caller to fix its
        request shape rather than read the reason.
        """
        r = client.call("hgm_recall", {}, msg_id=8)
        assert r["result"]["isError"] is True
        assert "query is required" in r["result"]["content"][0]["text"]

    def test_the_server_keeps_serving_after_all_of_that(self, client):
        r = client.request("tools/list", msg_id=9)
        assert len(r["result"]["tools"]) == 6


class TestRealCalls:
    """The integration itself — kept to a minimum because each call is real."""

    def test_recall_returns_attributed_facts(self, client):
        """Every hit must carry its writer; zero hits must say so.

        This reads the real store, which this machine has and CI does not. The
        first version asserted that a *particular* fact was present — true here,
        never true on CI, so it failed there for a reason that had nothing to do
        with the code. The sandboxed equivalent (seed a fact, read it back) is in
        ``test_hgm_mcp_write.py``; this one checks the shape of the reply against
        whatever the store actually holds.
        """
        r = client.call("hgm_recall", {"query": "怎么免密登录", "top_k": 3}, msg_id=10)
        text = r["result"]["content"][0]["text"]
        assert not r["result"].get("isError"), text
        assert "L2 facts:" in text

        hits = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("- (")]
        if hits:
            assert all("<" in h and ">" in h for h in hits), (
                f"命中没有署名，调用方无法判断是谁写的: {hits[:2]}")
        else:
            assert "0 hit" in text, f"零命中必须明说，而不是留白: {text}"

    def test_a_query_with_no_answer_says_so_instead_of_going_quiet(self, client):
        """'Nothing matched' and 'could not read' must not look identical."""
        r = client.call("hgm_recall", {"query": "红烧肉怎么做才好吃"}, msg_id=11)
        text = r["result"]["content"][0]["text"]
        assert "0 hit" in text
        assert "(" in text and ")" in text, f"零命中没有给原因: {text}"

    def test_remember_dry_run_reports_the_gate_verdict(self, client):
        r = client.call("hgm_remember",
                        {"fact": "MCP 冒烟测试的机器地址是 10.0.0.9", "dry_run": True},
                        msg_id=12)
        payload = json.loads(r["result"]["content"][0]["text"])
        assert "admitted" in payload
        assert payload.get("agent") == "pytest-mcp", "身份没有从 HGM_AGENT 传进来"

    def test_a_vague_fact_is_refused_with_a_reason(self, client):
        r = client.call("hgm_remember", {"fact": "我想要一个更好的方案"}, msg_id=13)
        payload = json.loads(r["result"]["content"][0]["text"])
        assert payload.get("ok") is False
        assert payload.get("reason") or payload.get("error")

    def test_agents_reports_provenance(self, client):
        r = client.call("hgm_agents", {}, msg_id=14)
        payload = json.loads(r["result"]["content"][0]["text"])
        assert "agents" in payload and "unattributed" in payload

    def test_kb_search_reaches_the_vault(self, client):
        r = client.call("hgm_kb_search", {"query": "共享记忆系统"}, msg_id=15)
        assert "note" in r["result"]["content"][0]["text"].lower()


class TestEconomy:
    def test_a_long_lived_server_does_not_repay_startup_per_call(self, client):
        """The reason this exists at all: the CLI pays ~2.8s per invocation.

        In a resident process LanceDB is imported once, so a repeat call should
        be markedly cheaper than shelling out. This asserts the shape of that
        win, not a precise number — the machine's load varies.
        """
        client.call("hgm_kb_search", {"query": "预热查询"}, msg_id=16)   # warm
        started = time.time()
        client.call("hgm_kb_search", {"query": "预热查询"}, msg_id=17)
        elapsed = time.time() - started
        assert elapsed < 2.8, f"常驻进程里单次调用仍要 {elapsed:.2f}s，没享受到常驻的好处"


class TestObservability:
    """The server must be able to say whether the host got as far as the tools.

    Added after a real launch left *nothing* behind: the log showed a start and
    no stop — terminated outside the read loop — and the server had recorded no
    handshake at all, so "the host read the tool list" and "the host gave up
    before saying anything" were indistinguishable. Same lesson as the WorkBuddy
    hook: an invisible surface is an undebuggable one.
    """

    def test_the_handshake_is_recorded(self, tmp_path):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HGM_AGENT"] = "obs-test"
        env["HERMES_HOME"] = str(tmp_path)      # log lands here, not in the real home
        env.pop("HGM_MCP_LOG", None)
        proc = subprocess.Popen([PY, str(SERVER)], cwd=str(REPO), env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        proc.stdin.write(json.dumps({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                       "clientInfo": {"name": "obs", "version": "1"}}}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2,
                                     "method": "tools/list"}) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
        proc.stdout.readline()
        proc.stdin.close()
        proc.wait(timeout=30)
        text = (tmp_path / "memory" / "mcp_log.txt").read_text(encoding="utf-8")
        assert "first_frame" in text
        assert "initialize" in text and "client=obs/1" in text, text
        assert "tools/list" in text and "count=6" in text, text

    def test_the_first_frame_shape_is_recorded_without_its_content(self, tmp_path):
        """Framing is the one mismatch that shows up only as the host giving up."""
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("HGM_MCP_LOG", None)
        env["HGM_AGENT"] = "obs-test"
        env["HERMES_HOME"] = str(tmp_path)
        proc = subprocess.Popen([PY, str(SERVER)], cwd=str(REPO), env=env,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "ping",
                                     "params": {"secret": "不该出现在日志里"}}) + "\n")
        proc.stdin.flush()
        proc.stdout.readline()
        proc.stdin.close()
        proc.wait(timeout=30)
        text = (tmp_path / "memory" / "mcp_log.txt").read_text(encoding="utf-8")
        assert "content_length_header=False" in text, text
        assert "不该出现在日志里" not in text, "日志里出现了请求内容"


class TestInterpreterBootstrap:
    """The server calls memory_cli in-process, so *this* interpreter needs LanceDB.

    A host may launch it with any Python, so the server relocates — but the first
    attempt at that used ``os.execve`` into another interpreter's ``python.exe``
    and **segfaulted** (exit 139, empty stderr) before the handshake. Silent
    death is the worst outcome for a surface the host expects to keep answering,
    so the mechanism is now asserted rather than trusted.
    """

    @staticmethod
    def _hook_module():
        import importlib.util
        spec = importlib.util.spec_from_file_location("hgm_mcp_under_test", SERVER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_it_does_not_execve(self):
        """Guards the exact construct that segfaulted, by *call site* not by text.

        Behavioural tests cannot catch this — the failure is an OS-level crash
        with no Python-level signal, and it only happens on an interpreter that
        is not the current one. So the construct itself is banned — and banned
        via AST, because the function's docstring explains the segfault in prose
        and a substring search would flag the explanation as the offence.
        """
        import ast
        tree = ast.parse(SERVER.read_text(encoding="utf-8"))
        offenders = [
            (node.lineno, node.func.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"execv", "execve", "execl", "execvp"}
        ]
        assert not offenders, (
            f"自举又用回了 os.exec*（第 {offenders} 行）—— 它在本机实测会段错误"
            f"（退出码 139，stderr 为空）")

    def test_the_relaunch_inherits_stdio(self):
        """The child must talk to the host directly, or the session breaks."""
        import ast
        tree = ast.parse(SERVER.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "run"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "subprocess"):
                kwargs = {kw.arg for kw in node.keywords}
                assert not ({"capture_output", "stdout", "stdin"} & kwargs), (
                    "捕获了子进程输出，宿主就看不到会话了 —— 必须让它继承 stdio")

    def test_it_returns_immediately_when_nothing_is_missing(self, monkeypatch):
        mod = self._hook_module()
        import memory_cli
        called = []
        monkeypatch.setattr(memory_cli, "_missing_modules", lambda: [])
        monkeypatch.setattr(memory_cli, "_candidate_interpreters",
                            lambda: called.append("looked") or [])
        mod.ensure_usable_interpreter()
        assert called == [], "没有缺依赖时不应去找别的解释器"

    def test_missing_modules_are_checked_before_candidates(self, monkeypatch):
        """Order matters: the normal case must not build a candidate list."""
        mod = self._hook_module()
        import memory_cli
        order = []
        monkeypatch.delenv("HGM_MCP_BOOTSTRAPPED", raising=False)
        monkeypatch.setattr(memory_cli, "_missing_modules",
                            lambda: order.append("missing") or ["lancedb"])
        monkeypatch.setattr(memory_cli, "_candidate_interpreters",
                            lambda: order.append("candidates") or [])
        mod.ensure_usable_interpreter()
        assert order == ["missing", "candidates"], order

    def test_it_gives_up_visibly_when_no_candidate_works(self, monkeypatch, capsys):
        mod = self._hook_module()
        import memory_cli
        monkeypatch.delenv("HGM_MCP_BOOTSTRAPPED", raising=False)
        monkeypatch.setattr(memory_cli, "_missing_modules", lambda: ["lancedb"])
        monkeypatch.setattr(memory_cli, "_candidate_interpreters", lambda: [])
        mod.ensure_usable_interpreter()          # must not raise
        err = capsys.readouterr().err
        assert "lancedb" in err, "放弃时要在 stderr 说清楚缺什么，不能静默"

    def test_it_does_not_relaunch_twice(self, monkeypatch):
        """A broken venv must not become an infinite relaunch loop."""
        mod = self._hook_module()
        import memory_cli
        monkeypatch.setenv("HGM_MCP_BOOTSTRAPPED", "1")
        monkeypatch.setattr(memory_cli, "_missing_modules", lambda: ["lancedb"])
        monkeypatch.setattr(memory_cli, "_candidate_interpreters",
                            lambda: (_ for _ in ()).throw(AssertionError("不该再找候选者")))
        mod.ensure_usable_interpreter()
