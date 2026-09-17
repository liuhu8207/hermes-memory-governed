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
    _FACT_SIGNALS_USER_ZH,
    _INTERROGATIVE_MODAL_FORMS,
    _ROLE_WEIGHTS,
    _SELF_SUFFICIENT_SIGNALS,
    _USER_MARKERS,
    _fact_signal_score,
    _has_self_sufficient_modal,
    dialogue_fact_admits,
    external_named_evidence,
    external_signal_pools,
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
    # 单个通用虚词就能过关（2026-09-17）：权重 2 == _EXTERNAL_MIN_SIGNAL，
    # 所以任何一个词条单独出现就够。它们单独出现时都不作事实承诺。
    "我的天哪",
    "我在干嘛呢",
    "太麻烦了",
    "我想要不还是算了",
    "因为这样吧",
    "可能是因为吧",
    "应该用的是这个",
    # 长文本分支的洞：数字 + 任意小写英文串 就凑够「结构证据 2」。
    "这个项目 2024 年吧，随便什么 whatever 都行",
    "好像是 2024 那个吧，随便 docker 什么的",
    "一共有 3 个方案吧，随便 pick 一个",
]

#: The payloads QA measured as ADMIT before the corroboration rule. Kept as
#: their own list so the test names them as the regression they are.
GENERIC_TOKEN_PAYLOADS = [
    "我的天哪",
    "我在干嘛呢",
    "太麻烦了",
    "我想要不还是算了",
    "因为这样吧",
    "可能是因为吧",
    "应该用的是这个",
]

#: The same weakness in the long-text branch: two structure slots, both of which
#: a year and an English filler word satisfy.
STRUCTURE_SLOT_ABUSE = [
    "这个项目 2024 年吧，随便什么 whatever 都行",
    "好像是 2024 那个吧，随便 docker 什么的",
    "一共有 3 个方案吧，随便 pick 一个",
]

#: Real Chinese constraints with NO IP / path / identifier / digit / latin at
#: all. They are why a small set of modal signals may still admit alone —
#: requiring corroboration from them would delete rules that have nothing to be
#: corroborated with.
CHINESE_CONSTRAINTS = [
    "不能明文存密码",
    "我需要每天都备份",
    "不能把密钥提交进仓库",
    "任何对外接口都必须走鉴权",
]

#: Defect A (2026-09-17) — a POLAR QUESTION *mentions* a modal without asserting
#: it. Chinese marks a yes/no question with an A-不-A frame ("能不能") or 否
#: ("能否"/"可否"), so `能不能…` literally contains `不能`. A bare substring test
#: therefore admitted every one of these as if it stated a constraint. Measured:
#: the first four all ADMIT before the fix.
POLAR_QUESTIONS = [
    "能不能帮我改一下配置",
    "能不能做成免密",
    "这个能不能行",
    "我能不能先看看",
    "是否能用 root 账号登录",
    "可不可以每周自动备份",
    "行不行这样配一次就好",
    "能否直接覆盖原来的配置",
]

#: Defect C (2026-09-17) — high-frequency first-person FUNCTION words that used
#: to sit in the admission table. Each is common enough that, once one token was
#: worth the whole threshold, it became a skeleton key ("我的天哪" / "我在干嘛呢" /
#: "太麻烦了" were admitted as facts). Evicted to the ranking-only list.
GENERIC_WORDS_EVICTED = ["我的", "我在", "太麻烦"]


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

    @pytest.mark.parametrize("text", GENERIC_TOKEN_PAYLOADS)
    def test_single_generic_token_is_not_evidence(self, text):
        """One pool entry used to weigh exactly _EXTERNAL_MIN_SIGNAL, so a single
        token admitted a row. "我的" is a possessive particle, "我在" an aspect
        marker, "因为" a conjunction — none of them asserts anything alone.

        Same shape as the '我不' bypass, and fixed the same way: the rule now
        needs two independent pieces of evidence, not one token's weight.
        """
        admitted, reason = external_write_verdict(text)
        assert not admitted, f"gate admitted a token, not a fact: {text}"
        assert reason == "weak_signal"

    @pytest.mark.parametrize("text", STRUCTURE_SLOT_ABUSE)
    def test_long_text_branch_needs_named_evidence(self, text):
        """`digit` and a bare lowercase latin run are trivially satisfied.

        The old long-text branch asked only for two structure slots, so a year
        plus any English word qualified. At least one slot must now name
        something: an IP, a path, or an identifier.
        """
        admitted, reason = external_write_verdict(text)
        assert not admitted, f"structure slots were mistaken for evidence: {text}"
        assert reason == "weak_signal"

    @pytest.mark.parametrize("text", CHINESE_CONSTRAINTS)
    def test_chinese_constraints_without_any_structure_still_admit(self, text):
        """The cost side of the corroboration rule, pinned so it stays zero.

        These carry no IP, no path, no identifier, no digit and no latin — the
        only thing standing behind them is a modal commitment. They are the
        reason 我需要 / 必须 / 不能 may admit alone.
        """
        admitted, reason = external_write_verdict(text)
        assert admitted, f"a real constraint was dropped: {text} ({reason})"

    def test_corroboration_is_what_admits_a_single_generic_token(self):
        """Same token, plus a named thing, is a fact — the token was never the
        problem, the missing corroboration was."""
        # "我家里" alone is a place, not a fact; with the hardware named it is one.
        assert not external_write_verdict("我家里")[0]
        assert external_write_verdict("我家里用的是软路由+策略服务A，打算换回硬件路由器")[0]

    def test_the_negation_token_is_not_in_the_strong_pool(self):
        """Pin the fix at the source, not only at the verdict.

        The behaviour test above would go green for any reason; this one fails
        if somebody puts '我不' back and re-tunes something else to compensate.
        """
        from plugin.memory_governed._sync import _FACT_SIGNALS_USER_ZH
        assert "我不" not in _FACT_SIGNALS_USER_ZH

    def test_the_evicted_generic_words_are_not_in_the_strong_pool(self):
        """Defect C, pinned at the source (2026-09-17).

        Same shape as the '我不' guard above: 我的 / 我在 / 太麻烦 are among the
        commonest tokens in Chinese prose and must not decide admission. They
        may still MARK a line as user-authored for the *shape* rules, which only
        ever loosens a heading/length check and can never admit a row.
        """
        for word in GENERIC_WORDS_EVICTED:
            assert word not in _FACT_SIGNALS_USER_ZH, word
            assert word in _USER_MARKERS, word

    def test_self_sufficient_set_is_exactly_the_modal_commitments(self):
        """Only modal commitments may admit a row on their own.

        Widening this set is how the original hole would come back: every entry
        added here becomes a skeleton key for a whole family of sentences that
        merely *contain* it.
        """
        assert set(_SELF_SUFFICIENT_SIGNALS) == {"我需要", "必须", "不能"}

    def test_one_pool_alone_is_exactly_the_old_threshold(self):
        """Documents why one token used to suffice, so the rule reads honestly."""
        assert _EXTERNAL_MIN_SIGNAL == 2
        # One pool is still worth exactly _EXTERNAL_MIN_SIGNAL — which is why a
        # single token used to clear the bar on its own.
        assert external_signal_pools("我想要一个能自动同步的方案") == 1
        # "因为这样吧" hits the TECH pool; a bare conjunction is still one pool.
        assert external_signal_pools("因为这样吧") == 1
        # "我的天哪" no longer hits ANY pool (2026-09-17): 我的/我在/太麻烦 were
        # evicted, so the canonical "one token == one pool == admission" example
        # is now a zero-pool input. See `GENERIC_WORDS_EVICTED`.
        assert external_signal_pools("我的天哪") == 0
        assert external_structural_evidence("我的天哪") == 0
        assert not external_named_evidence("我的天哪")

    def test_the_two_pools_are_counted_independently(self):
        """Two *kinds* of evidence admit; repetition inside one list does not."""
        assert external_signal_pools("必须同步 否则不兼容") == 1
        assert external_signal_pools("应该用 cosine 而不是点积") == 1
        assert external_signal_pools("报错是因为 total 未定义，必须改成 dict") == 2

    def test_named_evidence_excludes_bare_digits_and_lowercase_latin(self):
        assert external_named_evidence("示例主路由 192.0.2.1") is True
        assert external_named_evidence("`rsa.key` 必须同步") is True
        assert external_named_evidence("mosdns client_proxy_mode=whitelist") is True
        assert external_named_evidence("2024 年吧 whatever") is False
        assert external_named_evidence("docker 那个 config") is False

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


# ---------------------------------------------------------------------------
# Defect A (2026-09-17) — a question is not a commitment
# ---------------------------------------------------------------------------

class TestInterrogativeIsNotACommitment:
    """`能不能…` contains `不能`, so a substring test read the question as a rule."""

    @pytest.mark.parametrize("text", POLAR_QUESTIONS)
    def test_polar_questions_are_refused_with_a_reason(self, text):
        admitted, reason = external_write_verdict(text)
        assert not admitted, f"a question was read as a commitment: {text}"
        assert reason == "weak_signal"

    def test_every_interrogative_frame_is_stripped(self):
        for form in _INTERROGATIVE_MODAL_FORMS:
            assert not _has_self_sufficient_modal(f"你{form}帮我处理一下"), form

    def test_a_real_constraint_is_not_mistaken_for_a_question(self):
        """The cost side: 不能 inside an ASSERTION is a rule, not an A-不-A frame."""
        text = "不能同时写入 — 同一时间只在一个实例操作"
        assert _has_self_sufficient_modal(text)
        assert external_write_verdict(text)[0]


# ---------------------------------------------------------------------------
# Defect B (2026-09-17) — the role prior ranks, it does not gate
# ---------------------------------------------------------------------------

class TestRolePriorDoesNotGate:
    """`_ROLE_WEIGHTS["user"] == +2 == _MIN_SIGNAL_USER`, so when one call feeds
    both the ranking key and the admission gate the prior is silently promoted
    into EVIDENCE: one generic word plus the baseline cleared the bar because of
    WHO said it. Admission callers now pass ``include_role=False``."""

    def test_admission_score_ignores_the_role_prior(self):
        text = "部署方案确定用 Docker Compose，因为要支持多服务编排"
        assert _fact_signal_score(text, "user", strong_only=True,
                                  include_role=False) == (
            _fact_signal_score(text, "", strong_only=True, include_role=False))

    def test_default_still_applies_the_role_prior_for_ranking(self):
        """Ranking keeps the prior — a prior belongs in a sort key."""
        text = "部署方案确定用 Docker Compose，因为要支持多服务编排"
        base = _fact_signal_score(text)
        assert _fact_signal_score(text, "user") == base + _ROLE_WEIGHTS["user"]
        assert _fact_signal_score(text, "assistant") == (
            base + _ROLE_WEIGHTS["assistant"])

    @pytest.mark.parametrize("text", ["我想要不还是算了", "我的一般做法吧",
                                      "我家里呢，你懂的"])
    def test_a_bare_generic_word_no_longer_rides_the_user_baseline(self, text):
        assert not dialogue_fact_admits(text, "user")

    @pytest.mark.parametrize("text", POLAR_QUESTIONS)
    def test_polar_questions_do_not_ride_the_user_baseline_either(self, text):
        assert not dialogue_fact_admits(text, "user")

    def test_one_signal_with_a_named_thing_is_still_corroborated_evidence(self):
        """The cost side, pinned: single-signal facts must NOT be lost."""
        assert dialogue_fact_admits("我们决定用 PostgreSQL 因为它更稳定", "user")
        assert dialogue_fact_admits(
            "我家里用的是虚拟机软路由，我打算换回硬件路由器", "user")

    def test_a_modal_commitment_is_admitted_alone(self):
        for text in ("我需要每天都备份", "不能明文存密码",
                     "不能同时写入 — 同一时间只在一个实例操作"):
            assert dialogue_fact_admits(text, "user"), text

    def test_a_decorated_run_log_still_fails_the_assistant_floor(self):
        """Removing the role prior must not loosen the assistant side."""
        assert not dialogue_fact_admits(
            "备份在 `config.yaml.bak.before-deepseek-curation`", "assistant")


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
        hits = cli.recall_l2("alpha", 10, [], project="p1",
                             lexical_only=True)["hits"]
        body = " | ".join(h["content"] for h in hits)
        assert "属于 p1" in body
        assert "全局事实" in body
        assert "属于 p2" not in body

    def test_scoped_query_hides_other_projects(self, shared):
        self._seeded(shared)
        hits = cli.recall_l2("alpha", 10, [], project="p2",
                             lexical_only=True)["hits"]
        body = " | ".join(h["content"] for h in hits)
        assert "属于 p2" in body
        assert "属于 p1" not in body

    def test_unscoped_query_sees_everything(self, shared):
        self._seeded(shared)
        hits = cli.recall_l2("alpha", 10, [], lexical_only=True)["hits"]
        body = " | ".join(h["content"] for h in hits)
        assert "属于 p1" in body and "属于 p2" in body

    def test_unknown_project_still_sees_globals(self, shared):
        # A project with no facts of its own must not be blinded to the
        # infrastructure knowledge that applies everywhere.
        self._seeded(shared)
        hits = cli.recall_l2("alpha", 10, [], project="never-heard-of-it",
                             lexical_only=True)["hits"]
        body = " | ".join(h["content"] for h in hits)
        assert "全局事实" in body
        assert "属于 p1" not in body and "属于 p2" not in body

    def test_hits_carry_their_project(self, shared):
        self._seeded(shared)
        hits = cli.recall_l2("alpha", 10, [], lexical_only=True)["hits"]
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

