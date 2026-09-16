# -*- coding: utf-8 -*-
"""Shared-store contract: agent identity, provenance, and external writes.

The store used to have exactly one writer (Hermes, in-process), so "who wrote
this" was never a question and the CLI could hardcode an author. Opening the
store to other agents turns both of those into correctness requirements:

* a write must be attributable  -> ``resolve_agent`` / ``note_agent`` / agent col
* a write must be governed      -> ``external_write_verdict``
* a write must not clobber      -> same-title protection in ``cmd_kb_add``

The gate thresholds below are the measured result of a 10-positive /
10-negative calibration (see the floors block in ``_sync.py``). They are pinned
here so that relaxing them later fails a test rather than silently degrading
every agent's memory.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pyarrow as pa
import pytest

import memory_cli as cli
from plugin.memory_governed._sync import (
    _EXTERNAL_MIN_SIGNAL,
    _EXTERNAL_MIN_STRUCTURE,
    _EXTERNAL_MIN_WEIGHTED_LEN,
    external_structural_evidence,
    external_write_verdict,
)

# --- samples used for calibration ------------------------------------------
POSITIVE = [
    "SecretStore 使用 `ROCKET_TLS` 而不是 `SSL_CERT_FILE`",
    "我在NAS有装secretstore，但本地布署好像不能同步给其它地方装的secretstore",
    "我家里用的是虚拟机软路由，我打算换回硬件路由器，还要装tailscale进行组网",
    "mosdns client_proxy_mode=whitelist 已启用: 白名单 rule/client_ip_whitelist.txt",
    "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号",
    "必须同步 `rsa.key`，否则两边加密的数据不兼容",
    "家用NAS 192.0.2.62 上重度使用 Docker 部署服务",
    "Hermes Gateway 必须用 schtasks /run 重启，direct spawn 会被回收",
    "L2 召回的 distance 必须用 cosine 度量再映射成 score",
]
NEGATIVE = [
    "看起来 governed 工具有一些问题",
    "好的，我明白了",
    "刚才试了一下，好像可以了",
    "今天的天气不错",
    "帮我看看这个文件",
    "嗯嗯",
    "那你继续吧",
    "这个方案我再想想",
]


class TestExternalWriteGate:
    def test_keeps_concrete_facts(self):
        """Every sample here is the kind of thing an agent is asked to remember."""
        rejected = [s for s in POSITIVE if not external_write_verdict(s)[0]]
        assert not rejected, f"gate wrongly rejected: {rejected}"

    def test_leaks_no_chatter(self):
        """A leaked row is worse than a missed one — it pollutes every recall."""
        leaked = [s for s in NEGATIVE if external_write_verdict(s)[0]]
        assert not leaked, f"gate admitted noise: {leaked}"

    @pytest.mark.parametrize("text, reason", [
        ("我儿子的准考证，考试前一天记得提醒我", "ephemeral"),
        ("> 手写用户信息。直接编辑此文件。", "template"),
        ("[Image] 为什么你每次要弹这个", "media"),
        (r"C:\Users\example\AppData\Local\hermes\config.yaml", "abs_path"),
        ("", "empty"),
    ])
    def test_rejections_name_a_reason(self, text, reason):
        """A silent drop is indistinguishable from success, so reasons are mandatory."""
        admitted, got = external_write_verdict(text)
        assert not admitted
        assert got == reason

    def test_structural_evidence_is_what_saves_noun_facts(self):
        """The word lists score these 0; structure is the only thing admitting them."""
        noun_fact = "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号"
        assert external_structural_evidence(noun_fact) >= _EXTERNAL_MIN_STRUCTURE
        assert external_write_verdict(noun_fact)[0]

    def test_thresholds_are_the_calibrated_ones(self):
        """Pinned so a future edit has to be deliberate, not incidental."""
        assert (_EXTERNAL_MIN_SIGNAL, _EXTERNAL_MIN_WEIGHTED_LEN,
                _EXTERNAL_MIN_STRUCTURE) == (2, 40, 2)

    def test_role_user_must_not_be_used(self):
        """Why the gate scores with an empty role.

        Scoring external text as a user turn (baseline +2) admits every short
        acknowledgement — measured 8/10 leaks. This asserts the reason the
        design looks odd, so nobody "fixes" it back.
        """
        from plugin.memory_governed._sync import _fact_signal_score
        chatter = "好的，我明白了"
        assert _fact_signal_score(chatter, "user", strong_only=True) >= _EXTERNAL_MIN_SIGNAL
        assert _fact_signal_score(chatter, "", strong_only=True) < _EXTERNAL_MIN_SIGNAL


class TestAgentIdentity:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv(cli.AGENT_ENV_VAR, "fromenv")
        assert cli.resolve_agent("explicit") == "explicit"

    def test_env_used_when_no_flag(self, monkeypatch):
        monkeypatch.setenv(cli.AGENT_ENV_VAR, "fromenv")
        assert cli.resolve_agent("") == "fromenv"

    def test_detection_when_nothing_declared(self, monkeypatch):
        monkeypatch.delenv(cli.AGENT_ENV_VAR, raising=False)
        for marker in ("DSH_WORKSPACE", "DSH_PYTHON", "DSH_SESSION"):
            monkeypatch.delenv(marker, raising=False)
        monkeypatch.setenv("DSH_WORKSPACE", "/tmp/x")
        assert cli.resolve_agent("") == "dsh"

    def test_unknown_caller_stays_honest(self, monkeypatch):
        """Never guess a real agent's name — a wrong attribution corrupts provenance."""
        monkeypatch.delenv(cli.AGENT_ENV_VAR, raising=False)
        for _, markers in cli._AGENT_ENV_MARKERS:
            for m in markers:
                monkeypatch.delenv(m, raising=False)
        assert cli.resolve_agent("") == cli.DEFAULT_AGENT

    @pytest.mark.parametrize("raw, expected", [
        ("DSH", "dsh"),
        ("  AutoClaw  ", "autoclaw"),
        ("weird/name\\with:junk", "weirdnamewithjunk"),
        ("x" * 80, "x" * cli._AGENT_NAME_MAX),
        ("", ""),
        (None, ""),
    ])
    def test_names_are_normalized(self, raw, expected):
        assert cli.normalize_agent(raw) == expected


class TestNoteAgent:
    def test_explicit_agent_field_wins(self):
        assert cli.note_agent({"agent": "dsh", "source": "l3"}) == "dsh"

    def test_legacy_source_is_honoured_for_known_agents(self):
        """The one hand-written `source: dsh` note predates the agent field."""
        assert cli.note_agent({"source": "dsh"}) == "dsh"

    @pytest.mark.parametrize("source", ["l3", "wiki-backup", "20260507_005134_8947f3", ""])
    def test_legacy_source_is_not_treated_as_an_author(self, source):
        """`source` means provenance TYPE in the existing vault, not writer.

        Believing it blindly invents agents called "l3" and "wiki-backup".
        """
        assert cli.note_agent({"source": source}) == ""


# --- write paths ------------------------------------------------------------
@pytest.fixture
def shared(tmp_path, monkeypatch):
    """A HERMES_HOME with a vault and an L2 table, wired into the CLI."""
    home = tmp_path / ".hermes"
    (home / "memory" / "l2").mkdir(parents=True)
    vault = tmp_path / "vault"
    for sub in ("notes", "inbox"):
        (vault / sub).mkdir(parents=True)
    (home / "governed_memory.json").write_text(
        json.dumps({"wiki_dir": str(vault), "kb": {"confidence_threshold": 0.7}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return types.SimpleNamespace(home=home, vault=vault)


class TestKbAddProvenance:
    def test_writer_and_source_are_separate_fields(self, shared):
        out = cli.cmd_kb_add({"wiki_dir": str(shared.vault)}, "T", "body", "notes",
                             [], [], None, agent="dsh")
        assert out["ok"] is True
        meta, _ = cli.parse_frontmatter(Path(out["path"]).read_text(encoding="utf-8"))
        assert meta["agent"] == "dsh"
        # `source` describes how the note got here, NOT who wrote it — conflating
        # the two is what made the old vault unreadable ("source: l3").
        assert meta["source"] == "agent-write"

    def test_another_agent_cannot_silently_clobber(self, shared):
        cfg = {"wiki_dir": str(shared.vault)}
        cli.cmd_kb_add(cfg, "T", "first", "notes", [], [], None, agent="dsh")
        out = cli.cmd_kb_add(cfg, "T", "second", "notes", [], [], None, agent="autoclaw")
        assert out["ok"] is False
        assert "dsh" in out["error"]
        meta, body = cli.parse_frontmatter(
            (shared.vault / "notes" / "T.md").read_text(encoding="utf-8"))
        assert "first" in body, "original note must survive a refused overwrite"

    def test_same_agent_may_update_its_own_note(self, shared):
        cfg = {"wiki_dir": str(shared.vault)}
        cli.cmd_kb_add(cfg, "T", "first", "notes", [], [], None, agent="dsh")
        out = cli.cmd_kb_add(cfg, "T", "second", "notes", [], [], None, agent="dsh")
        assert out["ok"] is True
        assert out["updated"] is True
        _, body = cli.parse_frontmatter(
            (shared.vault / "notes" / "T.md").read_text(encoding="utf-8"))
        assert "second" in body

    def test_overwrite_is_available_but_explicit(self, shared):
        cfg = {"wiki_dir": str(shared.vault)}
        cli.cmd_kb_add(cfg, "T", "first", "notes", [], [], None, agent="dsh")
        out = cli.cmd_kb_add(cfg, "T", "second", "notes", [], [], None,
                             agent="autoclaw", overwrite=True)
        assert out["ok"] is True
        meta, _ = cli.parse_frontmatter(
            (shared.vault / "notes" / "T.md").read_text(encoding="utf-8"))
        assert meta["agent"] == "autoclaw"

    def test_secrets_are_refused(self, shared):
        out = cli.cmd_kb_add({"wiki_dir": str(shared.vault)}, "T",
                             "api_key = sk-abcdefghijklmnopqrst", "notes",
                             [], [], None, agent="dsh")
        assert out["ok"] is False


class TestRemember:
    """`remember` writes L2 and therefore needs embeddings; they are faked here
    so the suite stays offline and deterministic."""

    @staticmethod
    def _install_fake_embedding(monkeypatch, dim=8):
        real = cli.plugin_module

        class _FakeService:
            available = True
            last_error = ""
            backend_name = "fake:test"

            def embed_one(self, text):
                return [0.1] * dim

        def dispatch(name):
            if name == "_embedding":
                return types.SimpleNamespace(
                    EmbeddingService=types.SimpleNamespace(
                        get=lambda cfg: _FakeService()))
            return real(name)

        monkeypatch.setattr(cli, "plugin_module", dispatch)

    def _make_table(self, shared, dim=8):
        import lancedb
        db = lancedb.connect(str(shared.home / "memory" / "l2"))
        schema = pa.schema([
            pa.field("content", pa.string()),
            pa.field("category", pa.string()),
            pa.field("source", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("source_rowid", pa.int64()),
            pa.field("role", pa.string()),
        ])
        return db.create_table("memories", schema=schema)

    def test_writes_attributed_row(self, shared, monkeypatch):
        self._make_table(shared)
        self._install_fake_embedding(monkeypatch)

        out = cli.cmd_remember({}, "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号",
                               "dsh", category="ops")
        assert out["ok"] is True, out

        import lancedb
        table = lancedb.connect(str(shared.home / "memory" / "l2")).open_table("memories")
        arrow = table.to_arrow()
        assert arrow.column("agent").to_pylist() == ["dsh"]
        assert arrow.column("role").to_pylist() == ["agent"]

    def test_adds_agent_column_to_legacy_table(self, shared, monkeypatch):
        """Migration must work: `add_columns` takes a pa.Field, not a dict of arrays."""
        self._make_table(shared)
        self._install_fake_embedding(monkeypatch)

        cli.cmd_remember({}, "家用NAS 192.0.2.62 上重度使用 Docker 部署服务", "dsh")

        import lancedb
        table = lancedb.connect(str(shared.home / "memory" / "l2")).open_table("memories")
        assert "agent" in [f.name for f in table.schema]

    def test_rejected_fact_is_not_written(self, shared, monkeypatch):
        self._make_table(shared)
        self._install_fake_embedding(monkeypatch)

        out = cli.cmd_remember({}, "嗯嗯", "dsh")
        assert out["ok"] is False
        assert out["reason"] == "weak_signal"

        import lancedb
        table = lancedb.connect(str(shared.home / "memory" / "l2")).open_table("memories")
        assert table.count_rows() == 0

    def test_duplicate_is_not_written_twice(self, shared, monkeypatch):
        self._make_table(shared)
        self._install_fake_embedding(monkeypatch)
        fact = "必须同步 `rsa.key`，否则两边加密的数据不兼容"

        assert cli.cmd_remember({}, fact, "dsh")["ok"] is True
        second = cli.cmd_remember({}, fact, "dsh")
        assert second["ok"] is True and second["duplicate"] is True

        import lancedb
        table = lancedb.connect(str(shared.home / "memory" / "l2")).open_table("memories")
        assert table.count_rows() == 1

    def test_missing_embedding_refuses_instead_of_writing_unrecallable_row(
            self, shared, monkeypatch):
        self._make_table(shared)
        real = cli.plugin_module

        class _Unavailable:
            available = False
            last_error = "no api key"
            backend_name = "none"

        def dispatch(name):
            if name == "_embedding":
                return types.SimpleNamespace(
                    EmbeddingService=types.SimpleNamespace(
                        get=lambda cfg: _Unavailable()))
            return real(name)

        monkeypatch.setattr(cli, "plugin_module", dispatch)
        out = cli.cmd_remember({}, "家用NAS 192.0.2.62 上重度使用 Docker", "dsh")
        assert out["ok"] is False
        assert out["reason"] == "embedding_unavailable"

    def test_dry_run_touches_nothing(self, shared, monkeypatch):
        self._make_table(shared)
        self._install_fake_embedding(monkeypatch)
        out = cli.cmd_remember({}, "家用NAS 192.0.2.62 上重度使用 Docker", "dsh",
                               dry_run=True)
        assert out["ok"] is True and out["dry_run"] is True

        import lancedb
        table = lancedb.connect(str(shared.home / "memory" / "l2")).open_table("memories")
        assert table.count_rows() == 0


class TestReplayConsistency:
    """Writes and replays must use the same ruler.

    ``l2_apply_gate`` replays the gate over stored rows to drop those a tightened
    rule would no longer admit. If it scored agent rows with the dialogue-role
    floor instead of the external gate, every row that got in on structural
    evidence would be deleted the next time anyone ran it.
    """

    def test_agent_rows_survive_a_replay(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import l2_apply_gate

        noun_fact = "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号"
        assert l2_apply_gate.admits(noun_fact, "agent") is True

    def test_agent_replay_rejects_what_the_write_gate_rejects(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import l2_apply_gate

        assert l2_apply_gate.admits("好的，我明白了", "agent") is False
