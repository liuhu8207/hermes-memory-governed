# -*- coding: utf-8 -*-
"""Offline tests for the optional Jev decision layer."""

from __future__ import annotations

import json
import types
from pathlib import Path

from plugin.memory_governed._config import (
    DecisionConfig,
    GovernedMemoryConfig,
    load_governed_config,
)
from plugin.memory_governed._decision import DecisionEngine, DecisionResult
from plugin.memory_governed._jev_policy import JevDecisionPolicy
from plugin.memory_governed._kb import KnowledgeBase
from scripts.decision_audit import summarize


class _DuplicatePolicy:
    available = True

    def __init__(self, confidence: float = 0.97) -> None:
        self.confidence = confidence
        self.calls = 0

    def evaluate(self, operation, state):
        self.calls += 1
        return DecisionResult(
            operation=operation,
            source="test",
            chosen="duplicate",
            probabilities={"duplicate": self.confidence,
                           "new": 1.0 - self.confidence},
            confidence=self.confidence,
        )


class _UnavailablePolicy:
    available = False
    last_error = "offline"

    def evaluate(self, operation, state):
        raise AssertionError("unavailable policy must not be called")


def _kb(tmp_path: Path, *, mode: str = "shadow") -> KnowledgeBase:
    cfg = GovernedMemoryConfig()
    cfg.wiki_dir = str(tmp_path / "wiki")
    cfg.l2_db_path = str(tmp_path / "memory" / "l2")
    cfg.vector.backend = "none"
    cfg.decision = DecisionConfig(
        enabled=True,
        mode=mode,
        background_shadow=False,
        audit_path=str(tmp_path / "decision.jsonl"),
        cache_size=8,
    )
    kb = KnowledgeBase(cfg)
    kb.ensure()
    return kb


def test_config_loads_nested_decision_section(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "governed_memory.json").write_text(json.dumps({
        "decision": {
            "enabled": True,
            "mode": "bad-mode",
            "max_candidates": 50,
            "review_confidence": 2.0,
        }
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    cfg = load_governed_config(home)
    assert cfg.decision.enabled is True
    assert cfg.decision.mode == "shadow"
    assert cfg.decision.max_candidates == 20
    assert cfg.decision.review_confidence == 1.0


def test_shadow_engine_audits_without_raw_note_text(tmp_path):
    cfg = DecisionConfig(
        enabled=True,
        mode="shadow",
        background_shadow=True,
        audit_path=str(tmp_path / "audit.jsonl"),
        cache_size=8,
    )
    policy = _DuplicatePolicy()
    engine = DecisionEngine(cfg, policy=policy)
    report = engine.observe_kb_novelty(
        {"title": "same fact", "body": "private-note-body-token",
         "tags": ["api"], "concepts": ["endpoint"]},
        lambda: [{"title": "canonical", "path": "notes/canonical.md",
                  "snippet": "existing", "score": 0.9}],
    )
    assert report == {"status": "queued", "mode": "shadow",
                      "operation": "kb_novelty"}
    assert engine.flush(2.0) is True
    row = json.loads((tmp_path / "audit.jsonl").read_text(
        encoding="utf-8").strip())
    assert row["chosen"] == "duplicate"
    assert row["action"] == "warn"
    assert row["candidate_count"] == 1
    assert len(row["input_hash"]) == 64
    raw = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "private-note-body-token" not in raw
    assert "same fact" not in raw
    engine.shutdown()


def test_unavailable_policy_skips_expensive_candidate_lookup(tmp_path):
    cfg = DecisionConfig(enabled=True, mode="active")
    engine = DecisionEngine(cfg, policy=_UnavailablePolicy(),
                            audit_path=tmp_path / "audit.jsonl")
    called = False

    def candidates():
        nonlocal called
        called = True
        return []

    report = engine.observe_kb_novelty({"title": "x", "body": "y"}, candidates)
    assert report["fallback"] is True
    assert report["action"] == "review"
    assert called is False


def test_shadow_kb_write_is_unchanged(tmp_path):
    kb = _kb(tmp_path, mode="shadow")
    kb._decision = DecisionEngine(kb._config.decision,
                                  policy=_DuplicatePolicy())
    kb.add(title="Canonical endpoint", body="MiMo supports OpenAI protocol.",
           section="notes", tags=["api"], concepts=["mimo"])
    result = kb.add(title="Duplicate endpoint",
                    body="MiMo API uses the OpenAI protocol.",
                    section="notes", tags=["api"], concepts=["mimo"])
    note = kb.get("Duplicate endpoint")
    assert result["ok"] is True
    assert result["review_required"] is False
    assert result["decision"]["chosen"] == "duplicate"
    assert note["meta"].get("review_required") is None
    assert note["meta"].get("decision_action") is None


def test_active_kb_write_surfaces_review_flag_without_rejecting(tmp_path):
    kb = _kb(tmp_path, mode="active")
    kb._decision = DecisionEngine(kb._config.decision,
                                  policy=_DuplicatePolicy())
    kb.add(title="Canonical endpoint", body="MiMo supports OpenAI protocol.",
           section="notes", tags=["api"], concepts=["mimo"])
    result = kb.add(title="Duplicate endpoint",
                    body="MiMo API uses the OpenAI protocol.",
                    section="notes", tags=["api"], concepts=["mimo"])
    note = kb.get("Duplicate endpoint")
    assert result["ok"] is True
    assert result["review_required"] is True
    assert note["meta"]["review_required"] is True
    assert note["meta"]["decision_action"] == "warn"
    assert note["meta"]["decision_confidence"] == 0.97


def test_jev_policy_without_key_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    cfg = DecisionConfig(enabled=True, mode="shadow",
                         api_key_env="TYPESAFE_API_KEY")
    policy = JevDecisionPolicy(cfg)
    assert policy.available is False
    assert policy.evaluate("kb_novelty", {}) is None


def test_jev_policy_parses_choice_response_without_network(monkeypatch, tmp_path):
    class _Response:
        def model_dump(self):
            return {
                "model": "jev-test",
                "usage": {"input_tokens": 42},
                "answers": {
                    "relation": {
                        "type": "choice",
                        "choice": "duplicate",
                        "probabilities": {"duplicate": 0.9, "extends": 0.03,
                                          "conflicts": 0.02, "new": 0.03,
                                          "uncertain": 0.02},
                        "confidence": 0.9,
                    }
                },
            }

    class _Client:
        def system_one(self, **kwargs):
            assert kwargs["state"]["new_note"]["title"] == "same"
            return _Response()

    fake_sdk = types.SimpleNamespace(
        Choice=lambda **kwargs: kwargs,
        RetryPolicy=lambda **kwargs: kwargs,
        TypeSafeClient=lambda **kwargs: _Client(),
    )
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty-home"))
    cfg = DecisionConfig(enabled=True, mode="shadow")
    policy = JevDecisionPolicy(cfg)
    policy._sdk = fake_sdk
    policy._api_key = "test-key"
    result = policy.evaluate("kb_novelty", {
        "new_note": {"title": "same"},
        "candidates": [],
    })
    assert result is not None
    assert result.chosen == "duplicate"
    assert result.confidence == 0.9
    assert result.model == "jev-test"


def test_decision_audit_summary(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps({"ts": "t1", "operation": "kb_novelty", "source": "jev",
                    "chosen": "duplicate", "action": "warn",
                    "confidence": 0.9, "fallback": False, "latency_ms": 100})
        + "\n"
        + json.dumps({"ts": "t2", "operation": "kb_novelty",
                      "source": "fallback", "chosen": "uncertain",
                      "action": "review", "confidence": 0.0,
                      "fallback": True, "latency_ms": 200})
        + "\nmalformed\n",
        encoding="utf-8",
    )
    result = summarize(path, decision_enabled=True)
    assert result["ok"] is True
    assert result["status"] == "ready"
    assert result["enabled"] is True
    assert result["rows"] == 2
    assert result["malformed"] == 1
    assert result["fallback_rate"] == 0.5
    assert result["actions"]["warn"] == 1
    assert result["latency_ms_p95"] == 200.0


def test_decision_audit_missing_file_is_not_a_failure(tmp_path):
    missing = tmp_path / "missing.jsonl"
    disabled = summarize(missing, decision_enabled=False)
    assert disabled["ok"] is True
    assert disabled["status"] == "disabled"
    assert disabled["rows"] == 0

    enabled = summarize(missing, decision_enabled=True)
    assert enabled["ok"] is True
    assert enabled["status"] == "enabled_no_records"
    assert enabled["rows"] == 0
