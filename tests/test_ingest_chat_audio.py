# -*- coding: utf-8 -*-
"""``chat_audio`` 风格 ASR（chat-completions + ``input_audio``）的测试。

MiMo（``mimo-v2.5-asr``）这类端点**不是** OpenAI audio 兼容的：``/audio/
transcriptions`` 返回 404，转写必须走 ``/chat/completions``，音频以 base64
data URL 放进 user 消息的 ``input_audio`` 内容块，文本在
``choices[0].message.content``。

全部离线：``httpx`` 用 ``sys.modules`` 注入假模块（与 ``test_ingest.py`` 同款），
ffmpeg 转码被 monkeypatch —— 没有任何测试碰网络或真实 API。
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

from plugin.memory_governed import _config, _ingest
from plugin.memory_governed._config import GovernedMemoryConfig


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _chat_cfg(**kw) -> GovernedMemoryConfig:
    cfg = GovernedMemoryConfig()
    cfg.asr.api_style = "chat_audio"
    cfg.asr.base_url = kw.get("base_url", "https://api.example.com/v1")
    cfg.asr.model = kw.get("model", "mimo-v2.5-asr")
    cfg.asr.api_key_env = kw.get("api_key_env", "MIMO_KEY")
    cfg.asr.language = kw.get("language", "")
    if "max_encoded_bytes" in kw:
        cfg.asr.max_encoded_bytes = kw["max_encoded_bytes"]
    return cfg


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _install_httpx(monkeypatch, post):
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=post))


def _fake_mp3(monkeypatch, payload=b"fake-mp3-bytes"):
    """把 ``_transcode_to_mp3`` 换成「写一个真文件并记录调用」，返回调用记录列表。"""
    calls = []

    def fake(src, out_dir):
        calls.append(str(src))
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "converted.mp3"
        p.write_bytes(payload)
        return p

    monkeypatch.setattr(_ingest, "_transcode_to_mp3", fake)
    return calls


# ---------------------------------------------------------------------------
# 分派：默认风格不受影响
# ---------------------------------------------------------------------------

def test_default_api_style_is_transcriptions():
    assert GovernedMemoryConfig().asr.api_style == "transcriptions"


def test_default_style_never_touches_chat_audio(tmp_path, monkeypatch):
    cfg = GovernedMemoryConfig()
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 5.0)
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0:
                        {"ok": True, "text": "hi", "path": str(path)})

    def _boom(*a, **k):
        raise AssertionError("默认风格绝不能走 chat_audio")

    monkeypatch.setattr(_ingest, "_transcribe_chat_audio", _boom)
    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert r["text"] == "hi"


def test_unknown_api_style_falls_back_to_transcriptions(tmp_path, monkeypatch):
    cfg = GovernedMemoryConfig()
    cfg.asr.api_style = "banana"
    audio = tmp_path / "a.m4a"
    audio.write_bytes(b"x")
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 5.0)
    seen = {}

    def _trans(path, config, timeout=180.0):
        seen["used"] = "transcriptions"
        return {"ok": True, "text": "t"}

    def _chat(*a, **k):
        seen["used"] = "chat_audio"
        return {"ok": True, "text": "c"}

    monkeypatch.setattr(_ingest, "transcribe_audio", _trans)
    monkeypatch.setattr(_ingest, "_transcribe_chat_audio", _chat)
    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert seen["used"] == "transcriptions"


def test_unknown_api_style_dispatcher_defaults_to_transcriptions(monkeypatch):
    cfg = GovernedMemoryConfig()
    cfg.asr.api_style = ""
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0: {"ok": True, "text": "x"})

    def _boom(*a, **k):
        raise AssertionError("空风格应退回 transcriptions")

    monkeypatch.setattr(_ingest, "_transcribe_chat_audio", _boom)
    r = _ingest._transcribe_one("p.mp3", cfg, 5.0)
    assert r["text"] == "x"


def test_config_validator_resets_unknown_api_style_without_raising():
    cfg = GovernedMemoryConfig()
    cfg.asr.api_style = "weird"
    _config._validate_api_style(cfg)          # 绝不抛
    assert cfg.asr.api_style == "transcriptions"

    good = GovernedMemoryConfig()
    good.asr.api_style = "chat_audio"
    _config._validate_api_style(good)
    assert good.asr.api_style == "chat_audio"


def test_chat_audio_dispatches_to_the_chat_primitive(monkeypatch):
    cfg = _chat_cfg()
    seen = {}

    def _chat(path, config, timeout):
        seen["hit"] = (path, timeout)
        return {"ok": True, "text": "c"}

    monkeypatch.setattr(_ingest, "_transcribe_chat_audio", _chat)

    def _boom(*a, **k):
        raise AssertionError("chat_audio 不应走 transcriptions 原语")

    monkeypatch.setattr(_ingest, "transcribe_audio", _boom)
    r = _ingest._transcribe_one("p.wav", cfg, 7.0)
    assert r["text"] == "c"
    assert seen["hit"] == ("p.wav", 7.0)


# ---------------------------------------------------------------------------
# 请求体形状
# ---------------------------------------------------------------------------

def test_body_is_json_with_data_url_and_no_asr_options_when_language_empty(
        tmp_path, monkeypatch):
    cfg = _chat_cfg(language="")
    monkeypatch.setenv("MIMO_KEY", "sk-x")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    captured = {}

    def fake_post(endpoint, **kw):
        captured["endpoint"] = endpoint
        captured["kw"] = kw
        return _Resp(200, {"choices": [{"message": {"content": "hello"}}]})

    _install_httpx(monkeypatch, fake_post)

    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is True
    assert r["text"] == "hello"
    assert captured["endpoint"] == "https://api.example.com/v1/chat/completions"

    kw = captured["kw"]
    # JSON，不是 multipart
    assert "json" in kw and "files" not in kw and "data" not in kw
    assert kw["headers"]["Content-Type"] == "application/json"
    assert kw["headers"]["Authorization"] == "Bearer sk-x"

    block = kw["json"]["messages"][0]["content"][0]
    assert block["type"] == "input_audio"
    data_url = block["input_audio"]["data"]
    assert data_url.startswith("data:audio/mpeg;base64,")
    assert base64.b64decode(data_url.split(",", 1)[1]) == b"fake-mp3-bytes"
    # language 留空 → 不给 asr_options
    assert "asr_options" not in kw["json"]


def test_body_includes_asr_options_when_language_set(tmp_path, monkeypatch):
    cfg = _chat_cfg(language="zh")
    monkeypatch.setenv("MIMO_KEY", "sk-x")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    captured = {}

    def fake_post(endpoint, **kw):
        captured["json"] = kw["json"]
        return _Resp(200, {"choices": [{"message": {"content": "你好"}}]})

    _install_httpx(monkeypatch, fake_post)
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is True
    assert captured["json"]["asr_options"] == {"language": "zh"}


# ---------------------------------------------------------------------------
# 读取文本的位置
# ---------------------------------------------------------------------------

def test_reads_transcript_from_choices_message_content(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(
        200, {"choices": [{"message": {"content": "来自 message"}}]}))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is True
    assert r["text"] == "来自 message"


def test_falls_back_to_choices_text_key(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(
        200, {"choices": [{"text": "来自顶层 text"}]}))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is True
    assert r["text"] == "来自顶层 text"


def test_neither_content_nor_text_names_the_observed_shape(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(200, {"choices": [
        {"message": {"content": ""}, "audio": {}, "audio_tokens": 3,
         "tool_calls": []}]}))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    # 要报出**实际看到的形状**，而不是笼统的 no text
    assert "choices[0] keys" in r["error"]
    assert "audio" in r["error"] and "tool_calls" in r["error"]


def test_no_choices_names_the_top_level_shape(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(
        200, {"id": "x", "usage": {}}))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert "no choices" in r["error"]
    assert "usage" in r["error"]


# ---------------------------------------------------------------------------
# mp3 归一化：无条件
# ---------------------------------------------------------------------------

def test_transcodes_even_when_input_is_already_mp3(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "already.mp3"
    audio.write_bytes(b"x")
    calls = _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(
        200, {"choices": [{"message": {"content": "t"}}]}))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is True
    assert calls == [str(audio)], "mp3 输入也必须先转码"


def test_undecodable_input_is_refused(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "voice.silk"
    audio.write_bytes(b"\x02#!SILK_V3 nope")

    def _bad(src, out_dir):
        raise RuntimeError("ffmpeg failed transcoding voice.silk: Invalid data found")

    monkeypatch.setattr(_ingest, "_transcode_to_mp3", _bad)

    def _boom(*a, **k):
        raise AssertionError("无法解码时绝不能发请求")

    _install_httpx(monkeypatch, _boom)
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert ".silk" in r["error"]


# ---------------------------------------------------------------------------
# base64 上限
# ---------------------------------------------------------------------------

def test_oversized_payload_is_refused_by_name(tmp_path, monkeypatch):
    cfg = _chat_cfg(max_encoded_bytes=4)          # 故意极小
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch, payload=b"12345678")   # base64 → 12 字节 > 4

    def _boom(*a, **k):
        raise AssertionError("超限时不应发请求")

    _install_httpx(monkeypatch, _boom)
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert "max_encoded_bytes" in r["error"]
    assert "4" in r["error"]          # 上限
    assert "chunk_minutes" in r["error"]


# ---------------------------------------------------------------------------
# 错误与密钥
# ---------------------------------------------------------------------------

def test_http_error_is_a_structured_envelope(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(401, None, text="unauthorized"))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert "401" in r["error"]


def test_non_json_response_is_a_structured_envelope(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    monkeypatch.setenv("MIMO_KEY", "sk")
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    _fake_mp3(monkeypatch)
    _install_httpx(monkeypatch, lambda e, **kw: _Resp(200, None, text="<html>"))
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert "non-JSON" in r["error"]


def test_missing_api_key_is_reported(tmp_path, monkeypatch):
    cfg = _chat_cfg(api_key_env="MIMO_MISSING")
    monkeypatch.delenv("MIMO_MISSING", raising=False)
    audio = tmp_path / "v.wav"
    audio.write_bytes(b"x")
    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert "api key" in r["error"]


def test_missing_file_is_reported(tmp_path):
    cfg = _chat_cfg()
    r = _ingest._transcribe_chat_audio(str(tmp_path / "nope.wav"), cfg, 30.0)
    assert r["ok"] is False
    assert "not found" in r["error"]


# ---------------------------------------------------------------------------
# 长音频：拆段 / 拼接 / 部分失败 —— 机制对两种风格共享
# ---------------------------------------------------------------------------

def test_long_chat_audio_splits_joins_and_reports_partial(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    cfg.asr.chunk_minutes = 1.0
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 150.0)
    chunks = [tmp_path / f"chunk_{i:04d}.mp3" for i in range(3)]
    for c in chunks:
        c.write_bytes(b"c")
    monkeypatch.setattr(_ingest, "split_audio", lambda *a, **k: list(chunks))

    counter = {"n": 0}

    def fake_chat(path, config, timeout):
        counter["n"] += 1
        n = counter["n"]
        if n == 2:
            return {"ok": False, "error": "ASR HTTP 502: bad gateway"}
        return {"ok": True, "path": str(path), "text": f"part-{n}",
                "model": "m", "truncated": False}

    monkeypatch.setattr(_ingest, "_transcribe_chat_audio", fake_chat)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["chunks"] == 3
    assert r["ok"] is False
    assert r["partial"] is True
    assert r["chunks_failed"] == [2]                 # 1-based
    assert "part-1" in r["text"] and "part-3" in r["text"]
    assert "part-2" not in r["text"]
    assert "chunk 2/3 failed" in r["error"]


def test_long_chat_audio_all_chunks_succeed(tmp_path, monkeypatch):
    cfg = _chat_cfg()
    cfg.asr.chunk_minutes = 1.0
    audio = tmp_path / "long.wav"
    audio.write_bytes(b"x")
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 150.0)
    chunks = [tmp_path / f"chunk_{i:04d}.mp3" for i in range(3)]
    monkeypatch.setattr(_ingest, "split_audio", lambda *a, **k: list(chunks))
    monkeypatch.setattr(_ingest, "_transcribe_chat_audio",
                        lambda path, config, timeout:
                        {"ok": True, "path": str(path), "text": f"p{Path(path).stem[-1]}",
                         "model": "m", "truncated": False})

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert r["chunks"] == 3
    assert r["text"] == "p0\n\np1\n\np2"


# ---------------------------------------------------------------------------
# 原语未改动
# ---------------------------------------------------------------------------

def test_transcribe_audio_primitive_is_untouched():
    assert _ingest.transcribe_audio.__defaults__ == (180.0,)
