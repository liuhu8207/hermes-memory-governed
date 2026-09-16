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
    L2_PROVENANCE_COLUMNS,
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
    # 认知性否定：陈述的是说话人的无知，不是关于世界的事实。'我不' 曾以权重 2
    # 待在强信号池里，正好等于 _EXTERNAL_MIN_SIGNAL，而强信号分支没有长度/结构
    # 要求 —— 这三条曾全部被放行。见 tmp/calibrate_gate_negation.py。
    "我不知道",
    "我不太确定",
    "我不这么认为",
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

    @pytest.mark.parametrize("text", ["我不知道", "我不太确定", "我不这么认为"])
    def test_epistemic_negation_is_refused_with_a_reason(self, text):
        """A hedge reports the speaker's ignorance; it is not a fact about anything.

        These are the three samples the negation bypass was found with. They
        must be refused *and* say why — a bare False looks like success.
        """
        admitted, reason = external_write_verdict(text)
        assert not admitted, f"gate admitted a hedge: {text}"
        assert reason == "weak_signal"

    def test_the_negation_token_is_not_in_the_strong_pool(self):
        """Pin the fix at the source, not only at the verdict.

        The behaviour test above would go green for any reason; this one fails
        if somebody puts '我不' back and re-tunes something else to compensate.
        """
        from plugin.memory_governed._sync import _FACT_SIGNALS_USER_ZH
        assert "我不" not in _FACT_SIGNALS_USER_ZH

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


class TestWorkBuddyIdentity:
    """Detection must be pinned to *observed* marker names.

    The first version guessed plausible ones (``WORKBUDDY_SESSION``,
    ``WORKBUDDY_HOME``, ``CODEBUDDY_SESSION``). None of them exist, so detection
    fell through to ``external`` and every WorkBuddy write would have been
    misattributed — the same class of bug as the hardcoded ``"dsh"`` this whole
    subsystem was built to remove. These tests exist so a future edit cannot
    quietly reintroduce a guess.
    """

    @staticmethod
    def _clear(monkeypatch):
        monkeypatch.delenv(cli.AGENT_ENV_VAR, raising=False)
        for _, markers in cli._AGENT_ENV_MARKERS:
            for m in markers:
                monkeypatch.delenv(m, raising=False)

    @pytest.mark.parametrize("marker", [
        "WORKBUDDY_APP_NAME", "WORKBUDDY_CONFIG_DIR",
        "WORKBUDDY_USER_DATA_DIR", "CODEBUDDY_SESSION_ID",
    ])
    def test_observed_markers_identify_workbuddy(self, monkeypatch, marker):
        self._clear(monkeypatch)
        monkeypatch.setenv(marker, "1")
        assert cli.resolve_agent("") == "workbuddy"

    def test_the_guessed_names_are_not_relied_on(self, monkeypatch):
        self._clear(monkeypatch)
        for bogus in ("WORKBUDDY_SESSION", "WORKBUDDY_HOME", "CODEBUDDY_SESSION"):
            monkeypatch.setenv(bogus, "1")
        assert cli.resolve_agent("") == cli.DEFAULT_AGENT


class TestVaultScanning:
    """A write must be readable back, whichever section it was filed under.

    ``kb-add --section X`` creates ``X/`` on demand, but the scanner walked a
    fixed list of sections. A note filed under any other heading was written
    successfully and then never found again — invisible to ``kb-search``,
    ``kb-get`` and the provenance report alike. Nothing signalled the loss,
    which is what makes it worth a test rather than a comment.
    """

    def test_note_in_a_nonstandard_section_is_retrievable(self, shared):
        cfg = {"wiki_dir": str(shared.vault)}
        out = cli.cmd_kb_add(cfg, "运维笔记", "示例主路由 8022", "运维",
                             [], [], None, agent="workbuddy")
        assert out["ok"] is True

        titles = [r["title"] for r in cli.cmd_kb_search(cfg, "运维笔记", 5, "")]
        assert titles == ["运维笔记"]
        assert cli.cmd_kb_get(cfg, "运维笔记") is not None

    def test_provenance_sees_notes_outside_the_standard_sections(self, shared):
        cfg = {"wiki_dir": str(shared.vault)}
        cli.cmd_kb_add(cfg, "运维笔记", "body", "运维", [], [], None, agent="workbuddy")
        assert cli.cmd_agents(cfg)["agents"]["workbuddy"]["kb_notes"] == 1

    def test_hidden_and_scaffold_files_are_not_notes(self, shared):
        vault = shared.vault
        (vault / ".obsidian").mkdir()
        (vault / ".obsidian" / "workspace.md").write_text("x", encoding="utf-8")
        (vault / "index.md").write_text("# index", encoding="utf-8")
        cli.cmd_kb_add({"wiki_dir": str(vault)}, "真笔记", "body", "notes",
                       [], [], None, agent="dsh")

        names = {p.name for p in cli.iter_notes({"wiki_dir": str(vault)})}
        assert "真笔记.md" in names
        assert "workspace.md" not in names
        assert "index.md" not in names


class TestRuntimeReport:
    """``runtime`` must describe the environment *after* relocation.

    It used to run before bootstrap, so it reported the invoked interpreter and
    printed ``full_store_visible: false`` while ``recall`` was in fact working.
    A diagnostic that contradicts reality is worse than no diagnostic.
    """

    def test_runtime_participates_in_bootstrap(self):
        assert "runtime" in cli._L2_COMMANDS

    def test_relocation_is_stated_not_hidden(self, monkeypatch):
        monkeypatch.setenv(cli._ORIGINAL_PYTHON_FLAG, "/usr/bin/python3")
        report = cli.runtime_report()
        assert report["requested_interpreter"] == "/usr/bin/python3"
        assert "note" in report

    def test_no_relocation_means_no_extra_fields(self, monkeypatch):
        monkeypatch.delenv(cli._ORIGINAL_PYTHON_FLAG, raising=False)
        report = cli.runtime_report()
        assert "requested_interpreter" not in report
        assert "interpreter" in report


# -- project attribution ----------------------------------------------------
class TestProjectLabelling:
    """Folding a working directory into a project label."""

    def test_normalize_keeps_cjk(self):
        # Checkouts on this machine live under D:\repos\Drive\项目\... . Agent
        # names drop non-ASCII, but doing that here would collapse several
        # distinct projects onto one empty label.
        assert cli.normalize_project("我的项目-v2") == "我的项目-v2"

    def test_normalize_strips_separators_and_whitespace(self):
        assert cli.normalize_project("a/b\\c d") == "abcd"

    def test_normalize_tolerates_none(self):
        assert cli.normalize_project(None) == ""

    def test_infer_walks_up_to_the_marker(self, tmp_path):
        # A hook fires with the cwd the user opened, which is often a subdir.
        repo = tmp_path / "myrepo"
        (repo / ".git").mkdir(parents=True)
        deep = repo / "src" / "pkg"
        deep.mkdir(parents=True)
        assert cli.infer_project(str(deep)) == "myrepo"

    def test_infer_reports_global_when_there_is_no_marker(self, tmp_path):
        # The leaf name used to be the fallback, which invented a project out
        # of a directory that merely exists: `~/.workbuddy` became the project
        # ".workbuddy" and `~` became the user's name. Facts written from
        # anywhere under HOME were then invisible to a real project scope.
        leaf = tmp_path / "somewhere"
        leaf.mkdir()
        assert cli.infer_project(str(leaf)) == ""

    def test_infer_refuses_a_user_level_directory(self, tmp_path, monkeypatch):
        # `.workbuddy` is in _PROJECT_MARKERS, but the HOME-adjacent copy is
        # WorkBuddy's user config, not a checkout.
        home = tmp_path / "home"
        (home / ".workbuddy").mkdir(parents=True)
        monkeypatch.setenv("USERPROFILE", str(home))
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        assert cli.infer_project(str(home / ".workbuddy")) == ""
        assert cli.infer_project(str(home)) == ""

    def test_infer_refuses_a_nonexistent_directory(self, tmp_path):
        # Attributing facts to a directory that is not there is worse than
        # admitting we cannot tell where we are.
        assert cli.infer_project(str(tmp_path / "nope" / "deeper")) == ""

    def test_infer_treats_a_file_path_as_its_directory(self, tmp_path):
        repo = tmp_path / "repo2"
        (repo / ".git").mkdir(parents=True)
        f = repo / "notes.md"
        f.write_text("x", encoding="utf-8")
        assert cli.infer_project(str(f)) == "repo2"

    def test_infer_prefers_the_nearest_marker(self, tmp_path):
        # A nested checkout belongs to the inner project, not the outer one.
        outer = tmp_path / "outer"
        (outer / ".git").mkdir(parents=True)
        inner = outer / "vendor" / "inner"
        (inner / ".git").mkdir(parents=True)
        assert cli.infer_project(str(inner)) == "inner"


class TestRememberAttribution:
    """The three intents of ``--project``, which must stay distinguishable."""

    def test_unspecified_infers_from_cwd(self, shared, monkeypatch, tmp_path):
        repo = tmp_path / "inferred-proj"
        (repo / ".git").mkdir(parents=True)
        monkeypatch.chdir(repo)
        out = cli.cmd_remember({}, "任意文本", "workbuddy", dry_run=True)
        assert out["project"] == "inferred-proj"

    def test_empty_string_means_global(self, shared, monkeypatch, tmp_path):
        # Explicitly global must NOT fall back to the cwd: that is the whole
        # point of distinguishing "" from "not supplied".
        repo = tmp_path / "cwdproj"
        (repo / ".git").mkdir(parents=True)
        monkeypatch.chdir(repo)
        out = cli.cmd_remember({}, "任意文本", "workbuddy", dry_run=True,
                               project="")
        assert out["project"] == ""

    def test_explicit_project_beats_the_cwd(self, shared, monkeypatch, tmp_path):
        repo = tmp_path / "cwdproj"
        (repo / ".git").mkdir(parents=True)
        monkeypatch.chdir(repo)
        out = cli.cmd_remember({}, "任意文本", "workbuddy", dry_run=True,
                               project="other-proj")
        assert out["project"] == "other-proj"


class TestProjectScopedRecall:
    """A scoped query sees its own project plus globals — and nothing else."""

    @staticmethod
    def _seed(home, rows):
        import lancedb

        l2 = home / "memory" / "l2"
        l2.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(l2))
        if "memories" in db.table_names():
            db.drop_table("memories")
        db.create_table("memories", data=rows)

    @staticmethod
    def _row(text, project):
        return {
            "content": text, "category": "other", "source": "test",
            "timestamp": "2026-09-16T00:00:00", "vector": [0.0, 0.0],
            "source_rowid": None, "role": "agent", "agent": "tester",
            "project": project,
        }

    def _seeded(self, shared):
        self._seed(shared.home, [
            self._row("alpha 事项属于 p1", "p1"),
            self._row("alpha 事项属于 p2", "p2"),
            self._row("alpha 是全局事实", None),
        ])

    def test_scoped_query_keeps_own_project_and_globals(self, shared):
        self._seeded(shared)
        hits = cli.search_l2("alpha", 10, [], project="p1")
        body = " | ".join(h["content"] for h in hits)
        assert "属于 p1" in body
        assert "全局事实" in body
        assert "属于 p2" not in body

    def test_scoped_query_hides_other_projects(self, shared):
        self._seeded(shared)
        hits = cli.search_l2("alpha", 10, [], project="p2")
        body = " | ".join(h["content"] for h in hits)
        assert "属于 p2" in body
        assert "属于 p1" not in body

    def test_unscoped_query_sees_everything(self, shared):
        self._seeded(shared)
        hits = cli.search_l2("alpha", 10, [])
        body = " | ".join(h["content"] for h in hits)
        assert "属于 p1" in body and "属于 p2" in body

    def test_unknown_project_still_sees_globals(self, shared):
        # A project with no facts of its own must not be blinded to the
        # infrastructure knowledge that applies everywhere.
        self._seeded(shared)
        hits = cli.search_l2("alpha", 10, [], project="never-heard-of-it")
        body = " | ".join(h["content"] for h in hits)
        assert "全局事实" in body
        assert "属于 p1" not in body and "属于 p2" not in body

    def test_hits_carry_their_project(self, shared):
        self._seeded(shared)
        hits = cli.search_l2("alpha", 10, [])
        by_content = {h["content"]: h for h in hits}
        assert by_content["alpha 事项属于 p1"]["project"] == "p1"
        # A global fact has no project key at all — absent, not empty.
        assert "project" not in by_content["alpha 是全局事实"]


class TestWikiDirIsolation:
    """``wiki_dir`` must follow HERMES_HOME, not one developer's machine.

    It used to fall back to ``D:\\Sync\\Drive\\项目\\github\\wiki``. Setting
    ``HERMES_HOME`` to a sandbox therefore did not isolate anything: ``health``
    still reported the real vault's notes, so every write rehearsal touched
    real data and the only workaround was to hand-write a
    ``governed_memory.json``. A sandbox that is not a sandbox is worse than no
    sandbox, because it looks isolated while it is not.
    """

    def test_unconfigured_home_owns_its_own_vault(self, tmp_path, monkeypatch):
        home = tmp_path / "sandbox"
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert cli.wiki_dir({}) == home / "wiki"

    def test_default_never_escapes_hermes_home(self, tmp_path, monkeypatch):
        """No configuration, no environment: the answer is still inside HOME."""
        home = tmp_path / "sandbox"
        monkeypatch.setenv("HERMES_HOME", str(home))
        resolved = cli.wiki_dir(cli.load_config())
        assert str(resolved).startswith(str(home)), \
            f"unconfigured wiki_dir escaped the sandbox: {resolved}"

    def test_configured_value_still_wins(self, tmp_path, monkeypatch):
        """The real deployment sets wiki_dir explicitly; that must keep working."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "sandbox"))
        explicit = r"C:\Users\example\wiki"
        assert str(cli.wiki_dir({"wiki_dir": explicit})) == explicit

    def test_no_hardcoded_development_path(self):
        """Source-level pin: the default is derived, never written down.

        Checked positively as well as negatively — a future edit that swaps one
        hardcoded path for another would satisfy only the negative half.
        """
        import inspect

        src = inspect.getsource(cli.wiki_dir)
        assert "hermes_home()" in src, "default must be derived from HERMES_HOME"
        # The old fallback was D:\repos\Drive\项目\github\wiki. Neither a drive
        # letter nor the repo's parent directory may appear as a literal.
        assert "Drive" not in src and "github" not in src


class TestL2SchemaDeclaresProject:
    """Guards a regression that actually happened.

    The ``project`` column was added to the backfill list but *not* to the
    table schema, so an existing table gained the column while a freshly
    created one never would. Behaviour tests cannot see that: they all run
    against a table that already has the column. Only the schema declaration
    itself catches it.
    Both consumers now read ``L2_PROVENANCE_COLUMNS`` instead of restating the
    list, so the guard is that neither has drifted back to a private copy.
    """

    @staticmethod
    def _source():
        path = Path(__file__).resolve().parents[1] / "plugin" / "memory_governed" / "_sync.py"
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _cli_source():
        path = Path(__file__).resolve().parents[1] / "memory_cli.py"
        return path.read_text(encoding="utf-8")

    def test_schema_is_built_from_the_shared_definition(self):
        assert "*l2_provenance_fields()," in self._source()

    def test_backfill_reads_the_shared_definition(self):
        assert "for col, typ in L2_PROVENANCE_COLUMNS:" in self._source()

    def test_cli_backfill_reads_the_same_definition(self):
        # The CLI is the third consumer and the one that missed `role`.
        assert "L2_PROVENANCE_COLUMNS" in self._cli_source()
        assert 'for col in ("agent", "project")' not in self._cli_source()

    def test_definition_covers_every_provenance_column(self):
        names = [name for name, _ in L2_PROVENANCE_COLUMNS]
        assert {"source_rowid", "role", "agent", "project"} <= set(names)

    def test_open_l2_table_backfills_project_on_a_legacy_table(self, shared):
        import lancedb

        l2 = shared.home / "memory" / "l2"
        db = lancedb.connect(str(l2))
        if "memories" in db.table_names():
            db.drop_table("memories")
        db.create_table("memories", data=[{
            "content": "legacy row", "category": "other", "source": "test",
            "timestamp": "2026-09-16T00:00:00", "vector": [0.0, 0.0],
        }])
        cfg = types.SimpleNamespace(l2_db_path=str(l2))
        table = cli.open_l2_table(cfg)
        assert table is not None
        assert "project" in [f.name for f in table.schema]

