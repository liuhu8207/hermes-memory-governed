# -*- coding: utf-8 -*-
"""Session 归纳（_synthesize.py）回归测试。

Scope:
- 未启用 / 未配置 / 缺 key → 返回 []（优雅降级，绝不抛异常）。
- 对话拼接（_build_transcript）过滤非 user/assistant、截断上限。
- JSON 解析（_parse_candidates）清洗坏条目、围栏、前后噪声。
- 真实调用链路（mock httpx）：endpoint/model/payload 正确，结果清洗后返回。

全部在 bare interpreter 上可跑：httpx 用 monkeypatch 注入假对象。
"""

from __future__ import annotations

import sys
import types

import pytest

from plugin.memory_governed._config import GovernedMemoryConfig
from plugin.memory_governed import _synthesize


# ---------------------------------------------------------------------------
# 未启用 / 未配置 降级
# ---------------------------------------------------------------------------

class TestDisabled:
    def test_disabled_returns_empty(self):
        cfg = GovernedMemoryConfig()
        assert cfg.synthesis.enabled is False
        assert _synthesize.synthesize_notes([{"role": "user", "content": "x"}], cfg) == []

    def test_enabled_but_no_base_url(self):
        cfg = GovernedMemoryConfig()
        cfg.synthesis.enabled = True
        cfg.synthesis.base_url = ""
        assert _synthesize.synthesize_notes([{"role": "user", "content": "x"}], cfg) == []

    def test_enabled_but_no_key_env(self, monkeypatch):
        cfg = GovernedMemoryConfig()
        cfg.synthesis.enabled = True
        cfg.synthesis.base_url = "https://example.com/v1"
        cfg.synthesis.api_key_env = "SYNTH_KEY_MISSING"
        monkeypatch.delenv("SYNTH_KEY_MISSING", raising=False)
        assert _synthesize.synthesize_notes([{"role": "user", "content": "x"}], cfg) == []


# ---------------------------------------------------------------------------
# 对话拼接
# ---------------------------------------------------------------------------

class TestTranscript:
    def test_filters_non_user_assistant(self):
        msgs = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好呀"},
            {"role": "user", "content": ""},
        ]
        t = _synthesize._build_transcript(msgs)
        assert t == "user: 你好\nassistant: 你好呀"

    def test_truncates_to_limit(self, monkeypatch):
        monkeypatch.setattr(_synthesize, "_MAX_TRANSCRIPT_CHARS", 20)
        msgs = [
            {"role": "user", "content": "这是一段很长的对话内容"},
            {"role": "assistant", "content": "后续内容被截断"},
        ]
        t = _synthesize._build_transcript(msgs)
        assert len(t) <= 20

    def test_multimodal_list_content_extracts_text(self):
        # hermes 真实 transcript 里 content 可能是 OpenAI 多模态 parts（list）
        msgs = [
            {"role": "user", "content": [
                {"type": "text", "text": "帮我看看这张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "图上是一个 Linux Mint 桌面"},
            ]},
        ]
        t = _synthesize._build_transcript(msgs)
        assert "帮我看看这张图" in t
        assert "Linux Mint" in t
        assert "base64" not in t  # 图片 data URL 不被当成文本
        assert "image_url" not in t

    def test_multimodal_list_content_skips_non_text_parts(self):
        msgs = [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
                {"type": "text", "text": "正文在这里"},
            ]},
        ]
        t = _synthesize._build_transcript(msgs)
        assert t == "user: 正文在这里"

    def test_content_to_text_handles_str_and_none(self):
        assert _synthesize._content_to_text(None) == ""
        assert _synthesize._content_to_text("plain") == "plain"
        assert _synthesize._content_to_text(["a", "b"]) == "a\nb"
        assert _synthesize._content_to_text(123) == "123"


# ---------------------------------------------------------------------------
# JSON 解析
# ---------------------------------------------------------------------------

class TestParseCandidates:
    def test_parses_clean_array(self):
        text = '[{"title":"A","body":"b","tags":["t"],"concepts":["c"],"confidence":0.9}]'
        out = _synthesize._parse_candidates(text)
        assert len(out) == 1
        assert out[0]["title"] == "A"
        assert out[0]["confidence"] == 0.9

    def test_strips_markdown_fence_and_noise(self):
        text = 'Here you go:\n```json\n[{"title":"A","body":"b","confidence":0.8}]\n```\nDone.'
        out = _synthesize._parse_candidates(text)
        assert len(out) == 1
        assert out[0]["title"] == "A"

    def test_salvages_truncated_array(self):
        # 截断的 JSON：第一个对象完整，第二个被截断
        text = '[{"title":"A","body":"b","confidence":0.9}, {"title":"B","body":"被截断'
        out = _synthesize._parse_candidates(text)
        assert len(out) == 1
        assert out[0]["title"] == "A"

    def test_salvages_missing_bracket(self):
        # 缺右括号的截断数组
        text = '[{"title":"A","body":"b","confidence":0.9}, {"title":"B","body":"c","confidence":0.8}'
        out = _synthesize._parse_candidates(text)
        assert len(out) == 2
        assert [o["title"] for o in out] == ["A", "B"]

    def test_salvage_returns_empty_on_garbage(self):
        assert _synthesize._parse_candidates('not json at all') == []

    def test_drops_bad_items(self):
        text = '[{"title":"ok","body":"b","confidence":0.8}, {"body":"no title"}, "junk", 42]'
        out = _synthesize._parse_candidates(text)
        assert len(out) == 1
        assert out[0]["title"] == "ok"

    def test_clamps_confidence(self):
        text = '[{"title":"A","body":"b","confidence":5.0}]'
        out = _synthesize._parse_candidates(text)
        assert out[0]["confidence"] == 1.0

    def test_no_array_returns_empty(self):
        assert _synthesize._parse_candidates("not json at all") == []


# ---------------------------------------------------------------------------
# 真实调用链路（mock httpx）
# ---------------------------------------------------------------------------

class TestSynthesizeCall:
    def _cfg(self):
        cfg = GovernedMemoryConfig()
        cfg.synthesis.enabled = True
        cfg.synthesis.base_url = "https://example.com/v1"
        cfg.synthesis.api_key_env = "SYNTH_KEY"
        cfg.synthesis.model = "mimo-v2.5"
        return cfg

    def test_full_chain(self, monkeypatch):
        cfg = self._cfg()
        monkeypatch.setenv("SYNTH_KEY", "sk-test")

        captured = {}

        class _Resp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": '[{"title":"T","body":"B","confidence":0.8}]'}}]}

        def _fake_post(endpoint, **kwargs):
            captured["endpoint"] = endpoint
            captured["kwargs"] = kwargs
            return _Resp()

        monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=_fake_post))

        out = _synthesize.synthesize_notes(
            [{"role": "user", "content": "我们以后用 Postgres"}], cfg
        )
        assert len(out) == 1
        assert out[0]["title"] == "T"
        # 端点 + 模型 + 鉴权正确
        assert captured["endpoint"] == "https://example.com/v1/chat/completions"
        body = captured["kwargs"]["json"]
        assert body["model"] == "mimo-v2.5"
        assert captured["kwargs"]["headers"]["Authorization"] == "Bearer sk-test"
        assert "Postgres" in body["messages"][1]["content"]

    def test_http_error_returns_empty(self, monkeypatch):
        cfg = self._cfg()
        monkeypatch.setenv("SYNTH_KEY", "sk-test")

        class _Resp:
            status_code = 500
            text = "boom"

            def json(self):
                return {}

        monkeypatch.setitem(
            sys.modules, "httpx", types.SimpleNamespace(post=lambda *a, **k: _Resp())
        )
        assert _synthesize.synthesize_notes([{"role": "user", "content": "x"}], cfg) == []

    def test_bad_json_response_returns_empty(self, monkeypatch):
        cfg = self._cfg()
        monkeypatch.setenv("SYNTH_KEY", "sk-test")

        class _Resp:
            status_code = 200
            text = ""

            def json(self):
                return {"choices": [{"message": {"content": "not json"}}]}

        monkeypatch.setitem(
            sys.modules, "httpx", types.SimpleNamespace(post=lambda *a, **k: _Resp())
        )
        assert _synthesize.synthesize_notes([{"role": "user", "content": "x"}], cfg) == []
