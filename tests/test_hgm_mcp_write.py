# -*- coding: utf-8 -*-
"""The MCP **write** tools, which the read tests never touched.

Why this file exists
--------------------
Everything proven end-to-end so far was a read: the host called
``hgm_kb_search`` / ``hgm_recall`` / ``hgm_agents`` and got answers. The write
tools were in a worse state than "untested" —

* ``hgm_remember`` had only ever been called with ``dry_run`` (no write) or with
  input the gate refuses (no write), so the *writing* half of it had never run;
* ``hgm_kb_add`` had **never been called at all**, by anyone.

Writes also carry risk that reads do not: a read that misbehaves wastes a turn,
a write that misbehaves puts something wrong into a store other agents read.
So these tests run against a **sandbox** HERMES_HOME with its own vault and
never touch the real store, and they assert the *store after the call*, not the
shape of the reply.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "scripts" / "hgm_mcp.py"
REAL_CONFIG = Path.home() / "AppData" / "Local" / "hermes" / "governed_memory.json"


class Sandbox:
    """An MCP client wired to a throwaway store and vault."""

    def __init__(self, root: Path, agent: str = "write-test"):
        self.root = root
        self.home = root / "home"
        self.vault = root / "vault"
        (self.home / "memory").mkdir(parents=True, exist_ok=True)
        (self.vault / "notes").mkdir(parents=True, exist_ok=True)
        # Start from the real config (so the embedding backend works) but point
        # the vault at the sandbox — kb-add must have somewhere harmless to write.
        cfg = {}
        if REAL_CONFIG.exists():
            cfg = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
        cfg["wiki_dir"] = str(self.vault)
        (self.home / "governed_memory.json").write_text(
            json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HERMES_HOME"] = str(self.home)
        env["HGM_AGENT"] = agent
        env["HGM_MCP_LOG"] = "0"
        self.proc = subprocess.Popen(
            [sys.executable, str(SERVER)], cwd=str(REPO), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.n = 0

    def call(self, name, arguments=None, timeout=120):
        self.n += 1
        frame = {"jsonrpc": "2.0", "id": self.n, "method": "tools/call",
                 "params": {"name": name, "arguments": arguments or {}}}
        self.proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        assert line, f"{name} 没有返回 —— server 可能死了: {self.proc.stderr.read()[:400]}"
        result = json.loads(line)["result"]
        return result["content"][0]["text"], result.get("isError", False)

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    root = tmp_path_factory.mktemp("hgm-mcp-write")
    s = Sandbox(root)
    try:
        yield s
    finally:
        s.close()


def _rows(box):
    """Read the sandbox store directly — the reply is not the evidence."""
    lancedb = pytest.importorskip("lancedb")
    db = lancedb.connect(str(box.home / "memory" / "l2"))
    lister = getattr(db, "list_tables", None)
    listing = lister() if callable(lister) else db.table_names()
    names = getattr(listing, "tables", None) or listing
    names = [t.name if hasattr(t, "name") else str(t) for t in names]
    if "memories" not in names:
        return []
    # ``to_arrow().to_pylist()``, not ``to_pandas()``: the deployed vector extras
    # are lancedb + pyarrow + fastembed, and **pandas is not among them**. The
    # product code avoids ``to_pandas`` for exactly this reason (see the notes in
    # ``test_l4_persona.py``); this helper had reintroduced it.
    return db.open_table("memories").to_arrow().to_pylist()


class TestRememberWrites:
    def test_a_real_write_lands_in_the_store(self, box):
        """The half that dry_run never exercised."""
        text, is_error = box.call("hgm_remember",
                                  {"fact": "写入路径测试的机器地址是 10.7.7.7"})
        payload = json.loads(text)
        assert not is_error, text
        assert payload.get("ok") is True, payload
        assert payload.get("dry_run") is None, "dry_run 不该出现在真实写入里"

        rows = _rows(box)
        contents = [str(r.get("content")) for r in rows]
        assert any("10.7.7.7" in c for c in contents), (
            f"调用报告成功，但库里没有这条事实 —— 回复说 ok 不等于写进去了: {contents}")

    def test_the_row_carries_its_writer(self, box):
        rows = _rows(box)
        mine = [r for r in rows if "10.7.7.7" in str(r.get("content"))]
        assert mine, "上一条测试的写入没找到"
        assert mine[0].get("agent") == "write-test", (
            "署名不对 —— 共享存储里这条事实会被算到别的 agent 头上")

    def test_a_duplicate_is_not_written_twice(self, box):
        before = len(_rows(box))
        text, _ = box.call("hgm_remember",
                           {"fact": "写入路径测试的机器地址是 10.7.7.7"})
        payload = json.loads(text)
        assert payload.get("duplicate") is True, payload
        assert len(_rows(box)) == before, "重复写入又加了一行"

    def test_a_project_scope_is_recorded(self, box):
        text, _ = box.call("hgm_remember",
                           {"fact": "写入路径测试的项目专属地址是 10.7.7.8",
                            "project": "write-probe"})
        payload = json.loads(text)
        assert payload.get("ok") is True, payload
        rows = [r for r in _rows(box) if "10.7.7.8" in str(r.get("content"))]
        assert rows and rows[0].get("project") == "write-probe", rows

    def test_a_global_fact_is_stored_as_null_not_empty_string(self, box):
        """``NULL`` and ``""`` mean different things to a scoped recall."""
        text, _ = box.call("hgm_remember",
                           {"fact": "写入路径测试的全局地址是 10.7.7.9",
                            "project": ""})
        assert json.loads(text).get("ok") is True
        rows = [r for r in _rows(box) if "10.7.7.9" in str(r.get("content"))]
        assert rows, "全局事实没写进去"
        value = rows[0].get("project")
        assert value is None or (isinstance(value, float) and value != value), (
            f"全局事实的 project 应为 NULL，实际是 {value!r}")

    def test_a_refusal_writes_nothing(self, box):
        before = len(_rows(box))
        text, _ = box.call("hgm_remember", {"fact": "我想要一个更好的方案"})
        payload = json.loads(text)
        assert payload.get("ok") is False
        assert len(_rows(box)) == before, "被拒的写入仍然改了库"


class TestKbAddWrites:
    """``hgm_kb_add`` had never been called by anyone before this file."""

    def test_a_note_is_created_and_findable(self, box):
        text, is_error = box.call("hgm_kb_add",
                                  {"title": "MCP 写入测试笔记",
                                   "body": "这条笔记用于验证 hgm_kb_add 真的落盘。"})
        payload = json.loads(text)
        assert not is_error, text
        assert payload.get("ok") is True, payload
        written = Path(str(payload.get("path")))
        assert written.exists(), f"回复说写成功，磁盘上没有这个文件: {written}"
        assert "hgm_kb_add" in written.read_text(encoding="utf-8")

        found, _ = box.call("hgm_kb_search", {"query": "MCP 写入测试笔记"})
        assert "MCP 写入测试笔记" in found, f"写进去却搜不出来: {found}"

    def test_the_note_is_attributed_to_its_writer(self, box):
        text, _ = box.call("hgm_kb_add", {"title": "MCP 署名测试笔记",
                                          "body": "验证署名。"})
        payload = json.loads(text)
        assert payload.get("agent") == "write-test", payload

    def test_a_traversing_section_is_refused_and_writes_nothing(self, box):
        outside = box.root / "escaped"
        text, _ = box.call("hgm_kb_add", {"title": "越界笔记",
                                          "body": "不该被写到 vault 外面",
                                          "section": "../escaped"})
        payload = json.loads(text)
        assert payload.get("ok") is False, payload
        assert "refused" in str(payload.get("error", "")).lower(), payload
        assert not outside.exists(), "目录穿越真的写出去了"


class TestSurrogateInput:
    """A lone surrogate must not reach the store, and must not kill the server.

    This is the regression for the 2026-09-18 report. Verified by hand first,
    then pinned here: the embedding call used to raise on it (so the semantic
    channel silently fell back to lexical), and Arrow cannot encode it (so the
    row was unwritable).
    """

    def test_a_lone_surrogate_does_not_break_the_write(self, box):
        box.n += 1
        frame = {"jsonrpc": "2.0", "id": box.n, "method": "tools/call",
                 "params": {"name": "hgm_remember",
                            "arguments": {"fact": "代理对回归测试的地址是 10.9.9.9 \udcae"}}}
        # ``ensure_ascii`` defaults to True, which escapes the surrogate as
        # \udcae — the only way to transport it at all. With ensure_ascii=False
        # the *sender* raises UnicodeEncodeError and nothing is ever sent, so a
        # test written that way would prove nothing about the server.
        box.proc.stdin.write(json.dumps(frame) + "\n")
        box.proc.stdin.flush()
        line = box.proc.stdout.readline()
        assert line, f"代理对把 server 打崩了: {box.proc.stderr.read()[-300:]}"
        payload = json.loads(json.loads(line)["result"]["content"][0]["text"])
        assert payload.get("ok") is True, payload

        rows = [r for r in _rows(box) if "10.9.9.9" in str(r.get("content"))]
        assert rows, "工具说写成功了，但库里没有这条"
        stored = str(rows[0]["content"])
        assert not any(0xD800 <= ord(c) <= 0xDFFF for c in stored), (
            f"库里存进了代理对 —— Arrow 之后会读不出来: {stored!r}")
        stored.encode("utf-8")                       # must not raise
        assert "代理对回归测试" in stored, "好字符不该被一起丢掉"

    def test_a_surrogate_in_a_query_does_not_break_recall(self, box):
        """The read side hits the same encoder, and used to fail the same way."""
        box.n += 1
        frame = {"jsonrpc": "2.0", "id": box.n, "method": "tools/call",
                 "params": {"name": "hgm_recall",
                            "arguments": {"query": "代理对回归测试 \udcae", "top_k": 3}}}
        box.proc.stdin.write(json.dumps(frame) + "\n")
        box.proc.stdin.flush()
        line = box.proc.stdout.readline()
        assert line, "查询里的代理对把 server 打崩了"
        result = json.loads(line)["result"]
        assert not result.get("isError"), result["content"][0]["text"][:200]
        assert "L2 facts" in result["content"][0]["text"]


class TestTheRoundTrip:
    """One agent writes, a *separate process* running as another agent reads.

    This is the claim the whole system exists to make, and it is the one thing a
    single-process test cannot show: two clients, two identities, one store.
    """

    def test_what_one_agent_writes_another_agent_finds(self, box):
        text, _ = box.call("hgm_remember",
                           {"fact": "跨 agent 写入测试的地址是 10.8.8.8"})
        assert json.loads(text).get("ok") is True, text

        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["HERMES_HOME"] = str(box.home)          # the same store…
        env["HGM_AGENT"] = "agent-two"              # …a different identity
        env["HGM_MCP_LOG"] = "0"
        proc = subprocess.Popen([sys.executable, str(SERVER)], cwd=str(REPO),
                                env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding="utf-8", bufsize=1)
        try:
            frame = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                     "params": {"name": "hgm_recall",
                                "arguments": {"query": "跨 agent 写入测试", "top_k": 5}}}
            proc.stdin.write(json.dumps(frame, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            out = json.loads(proc.stdout.readline())["result"]["content"][0]["text"]
        finally:
            proc.stdin.close()
            proc.wait(timeout=30)
        assert "10.8.8.8" in out, f"另一个 agent 读不到刚写的: {out}"
        assert "write-test" in out, f"读到内容但署名丢了: {out}"
