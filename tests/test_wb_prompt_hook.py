# -*- coding: utf-8 -*-
"""The WorkBuddy ``UserPromptSubmit`` hook — gate, snapshot and contract.

The gate is the part worth testing hardest. It was calibrated 2026-09-17 and the
calibration *rejected* the obvious design: re-using ``l2_lexical_score`` with a
floor could not separate the two groups, because the tokenizer treated junk as
evidence —

    "1+1 等于几"            -> 0.5000, off the token "1" in "1. **DHCP ..."
    "帮我把这个表格转成 CSV"   -> 0.3333, off the pronoun "这个"

so the hook requires a *strong term* instead. Those two strings are therefore
regression cases, not illustrations:
:meth:`TestTheCalibrationFoundThese.test_the_false_positives_that_killed_the_old_gate`.
"""
from __future__ import annotations

import importlib.util
import io
import json
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
HOOK_PATH = REPO / "scripts" / "wb_prompt_hook.py"


def _load():
    spec = importlib.util.spec_from_file_location("wb_prompt_hook", HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


hook = _load()


@pytest.fixture(autouse=True)
def _never_touch_the_real_log(monkeypatch):
    """The hook appends to a run log; a test run must not write into it.

    Autouse because it is a side effect, not a subject: one forgotten
    ``main()`` call would otherwise append to the user's real log file.
    :class:`TestRunLog` turns it back on deliberately, with a patched path.
    """
    monkeypatch.setenv("HGM_HOOK_LOG", "0")

FACTS = [
    {"content": "能不能做成免密", "agent": "dsh", "timestamp": 100},
    {"content": "SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE",
     "agent": "dsh", "timestamp": 200},
    {"content": "1. **DHCP DNS 必须指向 策略服务B** — 否则设备无法走代理",
     "agent": "workbuddy", "timestamp": 300},
    {"content": "msf你现在ssh连接还是老弹密码窗，能不能把这个解决掉",
     "agent": "external", "timestamp": 400},
    # Both added because the real store contains them and they produced
    # production false positives — see TestTheCalibrationFoundThese.
    {"content": "- ❌ AI 不能读密码 → 没有 CLI/API 接口",
     "agent": "unattributed", "timestamp": 500},
    {"content": "key太麻烦了，我希望有一个能让你读取，但又安全的办法",
     "agent": "unattributed", "timestamp": 600},
]


def _snapshot(tmp_path, facts=None, **overrides) -> Path:
    payload = {"schema": 1, "generated_at": _iso(time.time()),
               "truncated": False, "count": len(facts or FACTS),
               "facts": FACTS if facts is None else facts}
    payload.update(overrides)
    path = tmp_path / "l2_snapshot.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _iso(epoch):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(epoch, tz=timezone.utc).astimezone().isoformat()


# -- strong terms -----------------------------------------------------------
class TestStrongTerms:
    @pytest.mark.parametrize("text, expected", [
        ("怎么免密登录", "免密"),
        ("LanceDB 的表结构是什么", "lancedb"),
        ("SecretStore 跑在哪台机器上", "secretstore"),
    ])
    def test_keeps_evidence(self, text, expected):
        assert expected in hook.strong_terms(text)

    @pytest.mark.parametrize("text", ["1+1 等于几", "123", "2026"])
    def test_digits_are_not_evidence(self, text):
        assert not any(t.isdigit() for t in hook.strong_terms(text))

    @pytest.mark.parametrize("text", ["ok 一下", "a 和 b", "去 IO"])
    def test_short_ascii_is_not_evidence(self, text):
        assert not [t for t in hook.strong_terms(text)
                    if t.isascii() and len(t) < hook.MIN_ASCII_TERM]

    def test_function_bigrams_are_dropped(self):
        """Two grammar chars straddling a boundary — the 把这 false positive."""
        assert hook._is_function_bigram("把这")
        assert "把这" not in hook.strong_terms("帮我把这个表格转成 CSV")

    @pytest.mark.parametrize("term", ["的办", "再把", "在有", "看有", "没什"])
    def test_a_bigram_that_LEADS_with_grammar_is_dropped(self, term):
        """Both chars being grammatical was the first rule, and it was too weak.

        Measured in production on 2026-09-17: 有没有更好的办法 produced the
        bigram 的办 (的 is grammar, 办 is a real word), which survived the
        both-chars rule and matched a fact about a SecretStore workaround.
        Leading with a particle is the reliable signal.
        """
        assert hook._is_function_bigram(term)

    @pytest.mark.parametrize("term", ["免密", "门禁", "阈值", "密码", "代理", "同步", "反向", "实测"])
    def test_content_bigrams_survive_the_function_filter(self, term):
        assert not hook._is_function_bigram(term)

    @pytest.mark.parametrize("term", ["办法", "方法", "方式", "状态", "结果"])
    def test_low_information_nouns_are_dropped(self, term):
        """Real words that every topic has, so sharing one says nothing."""
        assert term in hook.GENERIC_TERMS

    def test_no_tokenizer_means_no_terms(self, monkeypatch):
        monkeypatch.setattr(hook.sys, "path", [])
        monkeypatch.setitem(__import__("sys").modules, "memory_cli", None)

        # Even if the import path is broken, the hook must not raise.
        assert isinstance(hook.strong_terms("怎么免密登录"), set)


# -- matching ---------------------------------------------------------------
class TestMatch:
    def test_a_related_prompt_matches(self):
        assert hook.match("怎么免密登录", FACTS)

    def test_unrelated_prompt_matches_nothing(self):
        assert hook.match("今天天气怎么样", FACTS) == []

    def test_ranking_prefers_more_shared_terms(self):
        top = hook.match("SecretStore ROCKET_TLS 证书", FACTS)[0][2]["content"]
        assert "ROCKET_TLS" in top

    def test_top_n_is_respected(self):
        assert len(hook.match("免密 secretstore 代理 密码", FACTS, top_n=1)) <= 1

    def test_empty_prompt_matches_nothing(self):
        assert hook.match("", FACTS) == []

    def test_facts_without_content_are_skipped(self):
        assert hook.match("免密", [{"content": None}, {"content": "  "}]) == []

    def test_it_ranks_by_shared_count_not_by_coverage(self):
        """Coverage divides by prompt length, punishing long questions.

        That is the dimension error that emptied the CLI's lexical channel once;
        the gate must not reintroduce it.
        """
        long_prompt = "免密 " + "无关词 " * 30
        assert hook.match(long_prompt, FACTS) == hook.match("免密", FACTS)


class TestTheCalibrationFoundThese:
    """Every entry here is a measured false positive, not a hypothetical."""

    @pytest.mark.parametrize("prompt", ["1+1 等于几", "帮我把这个表格转成 CSV"])
    def test_the_false_positives_that_killed_the_old_gate(self, prompt):
        """The score-based gate let these through via '1' and 把这."""
        assert hook.match(prompt, FACTS) == [], (
            f"{prompt!r} 曾经靠 '1' / '把这' 命中；强词项判据必须挡住它")

    def test_the_one_that_shipped_yesterday(self):
        """Found in production, on a prompt with no topic at all.

        ``你再实测看有什么问题没有？`` matched "- ❌ AI 不能读密码 → 没有
        CLI/API 接口" through the bigram 没有, because 没 was missing from the
        function-character list. One absent character defeated the scheme.
        """
        assert hook.match("你再实测看有什么问题没有？", FACTS) == []

    @pytest.mark.parametrize("prompt", [
        "有没有更好的办法", "这样做可以吗", "有什么问题没有", "这个对不对",
        "你觉得呢", "帮我看看这段代码", "现在几点了", "谢谢", "继续",
        "好的没问题", "还有别的吗", "今天天气怎么样", "1+1 等于几",
        "帮我写一个 Python 冒泡排序", "这个 HTML 页面颜色改深一点",
    ])
    def test_ordinary_prompts_stay_silent(self, prompt):
        assert hook.match(prompt, FACTS) == [], f"{prompt!r} 不该命中"

    @pytest.mark.parametrize("prompt", [
        "怎么免密登录", "SecretStore 跑在哪台机器上", "secretstore 两边同步要注意什么",
        "AI 能不能读密码",
    ])
    def test_true_positives_still_survive(self, prompt):
        assert hook.match(prompt, FACTS), f"{prompt!r} 应当命中却没命中"

    def test_one_known_false_positive_is_accepted_on_purpose(self):
        """Shared content words are not shared meaning, and this is the proof.

        解释一下什么是反向代理 shares 代理 with a fact about routing traffic
        through a proxy. A lexical matcher cannot tell 走代理 from 反向代理, so
        the choice is a short irrelevant line or an extra heuristic. Recorded in
        the module docstring as the price of the design.
        """
        assert hook.match("解释一下什么是反向代理", FACTS) != []


class TestRunLog:
    """The hook is invisible by default; this is how a bad run is seen."""

    def test_a_run_is_recorded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HGM_HOOK_LOG", "1")
        path = _snapshot(tmp_path)
        monkeypatch.setattr(hook, "snapshot_file", lambda: path)
        monkeypatch.setattr(hook.sys, "stdin",
                            io.StringIO(json.dumps({"prompt": "怎么免密登录",
                                                    "session_id": "s1"})))
        monkeypatch.setattr(hook.sys, "stdout", io.StringIO())
        hook.main()
        text = hook.log_path().read_text(encoding="utf-8")
        assert "session=s1" in text
        assert "matched=1" in text
        assert "免密" not in text, "日志不得记录提问内容本身"

    def test_the_log_is_capped(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HGM_HOOK_LOG", "1")
        path = _snapshot(tmp_path)
        monkeypatch.setattr(hook, "snapshot_file", lambda: path)
        monkeypatch.setattr(hook, "MAX_LOG_LINES", 5)
        for i in range(12):
            hook.log_run("prompt", f"s{i}", "note")
        lines = hook.log_path().read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5

    def test_logging_never_raises(self, monkeypatch):
        monkeypatch.setenv("HGM_HOOK_LOG", "1")
        monkeypatch.setattr(hook, "snapshot_file",
                            lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        hook.log_run("prompt", "s", "note")          # must not raise

    def test_it_can_be_switched_off(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HGM_HOOK_LOG", "0")
        path = _snapshot(tmp_path)
        monkeypatch.setattr(hook, "snapshot_file", lambda: path)
        hook.log_run("prompt", "s", "note")
        assert not hook.log_path().exists()


# -- context ----------------------------------------------------------------
class TestBuildContext:
    def test_empty_matches_produce_no_context(self):
        assert hook.build_context([]) == ""

    def test_context_names_the_source_agent(self):
        text = hook.build_context(hook.match("怎么免密登录", FACTS))
        assert "dsh" in text and "能不能做成免密" in text

    def test_context_is_capped(self, monkeypatch):
        monkeypatch.setattr(hook, "MAX_CHARS", 60)
        text = hook.build_context(hook.match("免密 secretstore 代理 密码", FACTS))
        assert len(text) <= 61

    def test_context_says_it_is_automatic_and_fallible(self):
        """An injected claim the agent cannot audit is worse than none."""
        text = hook.build_context(hook.match("怎么免密登录", FACTS))
        assert "自动" in text and "可能不相关" in text


# -- snapshot loading -------------------------------------------------------
class TestSnapshot:
    def test_reads_a_good_snapshot(self, tmp_path):
        facts, problem = hook.load_snapshot(_snapshot(tmp_path))
        assert problem == "" and len(facts) == len(FACTS)

    def test_missing_file_is_not_an_error(self, tmp_path):
        facts, problem = hook.load_snapshot(tmp_path / "nope.json")
        assert facts == [] and problem

    def test_corrupt_json_is_not_an_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        facts, problem = hook.load_snapshot(path)
        assert facts == [] and problem

    def test_unknown_schema_is_refused(self, tmp_path):
        facts, problem = hook.load_snapshot(_snapshot(tmp_path, schema=99))
        assert facts == [] and "schema" in problem

    def test_a_truncated_snapshot_stands_down(self, tmp_path):
        """A partial file would make the matcher confidently blind."""
        facts, problem = hook.load_snapshot(_snapshot(tmp_path, truncated=True))
        assert facts == [] and "truncated" in problem

    def test_a_stale_snapshot_is_not_trusted(self, tmp_path):
        old = _snapshot(tmp_path, generated_at=_iso(time.time() - 48 * 3600))
        facts, problem = hook.load_snapshot(old)
        assert facts == [] and "stale" in problem

    def test_a_stamp_that_cannot_be_parsed_counts_as_stale(self, tmp_path):
        facts, problem = hook.load_snapshot(_snapshot(tmp_path, generated_at="wat"))
        assert facts == [] and problem


# -- the host contract ------------------------------------------------------
class TestContract:
    def _run(self, monkeypatch, capsys, raw, snapshot_path=None):
        if snapshot_path is not None:
            monkeypatch.setattr(hook, "snapshot_file", lambda: snapshot_path)
        monkeypatch.setattr(hook.sys, "stdin", io.StringIO(raw))
        rc = hook.main()
        cap = capsys.readouterr()
        return rc, cap.out, cap.err

    def test_a_related_prompt_injects(self, monkeypatch, capsys, tmp_path):
        rc, out, _ = self._run(monkeypatch, capsys,
                              json.dumps({"prompt": "怎么免密登录"}),
                              _snapshot(tmp_path))
        assert rc == 0
        data = json.loads(out)
        assert data["continue"] is True
        assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
        assert "能不能做成免密" in data["hookSpecificOutput"]["additionalContext"]

    @pytest.mark.parametrize("raw", ["", "   ", "{not json", "[]", "null"])
    def test_malformed_input_still_yields_valid_json(self, monkeypatch, capsys, raw):
        rc, out, _ = self._run(monkeypatch, capsys, raw)
        assert rc == 0
        assert json.loads(out)["continue"] is True

    def test_no_snapshot_means_no_injection_but_no_failure(self, monkeypatch, capsys, tmp_path):
        rc, out, _ = self._run(monkeypatch, capsys,
                              json.dumps({"prompt": "怎么免密登录"}),
                              tmp_path / "missing.json")
        data = json.loads(out)
        assert rc == 0 and data["continue"] is True
        assert data["hookSpecificOutput"]["additionalContext"] == ""

    def test_an_unrelated_prompt_costs_nothing(self, monkeypatch, capsys, tmp_path):
        rc, out, _ = self._run(monkeypatch, capsys,
                              json.dumps({"prompt": "帮我写个快速排序"}),
                              _snapshot(tmp_path))
        assert json.loads(out)["hookSpecificOutput"]["additionalContext"] == ""

    def test_stdout_is_one_json_line_and_stderr_is_opt_in(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("HGM_HOOK_DEBUG", "1")
        rc, out, err = self._run(monkeypatch, capsys,
                                 json.dumps({"prompt": "怎么免密登录", "session_id": "s"}),
                                 _snapshot(tmp_path))
        assert len(out.strip().splitlines()) == 1
        json.loads(out)
        assert "[hgm-prompt]" in err
