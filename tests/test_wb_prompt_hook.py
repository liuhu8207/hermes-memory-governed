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

FACTS = [
    {"content": "能不能做成免密", "agent": "dsh", "timestamp": 100},
    {"content": "SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE",
     "agent": "dsh", "timestamp": 200},
    {"content": "1. **DHCP DNS 必须指向 策略服务B** — 否则设备无法走代理",
     "agent": "workbuddy", "timestamp": 300},
    {"content": "msf你现在ssh连接还是老弹密码窗，能不能把这个解决掉",
     "agent": "external", "timestamp": 400},
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
        """The 把这 false positive: two grammar chars straddling a boundary."""
        assert hook._is_function_bigram("把这")
        assert "把这" not in hook.strong_terms("帮我把这个表格转成 CSV")

    def test_content_bigrams_survive_the_function_filter(self):
        for term in ("免密", "门禁", "阈值", "密码", "代理"):
            assert not hook._is_function_bigram(term)

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
    """Measured false positives that the score-based gate let through."""

    @pytest.mark.parametrize("prompt", ["1+1 等于几", "帮我把这个表格转成 CSV"])
    def test_the_false_positives_that_killed_the_old_gate(self, prompt):
        assert hook.match(prompt, FACTS) == [], (
            f"{prompt!r} 曾经靠 '1' / '把这' 命中；强词项判据必须挡住它")

    @pytest.mark.parametrize("prompt", [
        "怎么免密登录", "SecretStore 跑在哪台机器上", "secretstore 两边同步要注意什么",
    ])
    def test_true_positives_still_survive(self, prompt):
        assert hook.match(prompt, FACTS), f"{prompt!r} 应当命中却没命中"


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
