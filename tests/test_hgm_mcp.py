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
                         "hgm_kb_add", "hgm_agents"}
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
        assert len(r["result"]["tools"]) == 5


class TestRealCalls:
    """The integration itself — kept to a minimum because each call is real."""

    def test_recall_returns_attributed_facts(self, client):
        r = client.call("hgm_recall", {"query": "怎么免密登录", "top_k": 3}, msg_id=10)
        text = r["result"]["content"][0]["text"]
        assert not r["result"].get("isError"), text
        assert "L2 facts:" in text
        assert "能不能做成免密" in text, text            # a fact known to exist
        assert "<" in text and ">" in text, "命中没有署名，调用方无法判断是谁写的"

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
