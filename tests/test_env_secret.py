# -*- coding: utf-8 -*-
"""``env_secret``：进程环境变量优先，回退 ``$HERMES_HOME/.env`` 的共享密钥读取。

背景：密钥只装在 ``$HERMES_HOME/.env`` 里（部署约定），而 Hermes 插件 / MCP 服务 /
CLI 的进程环境里**没有**这些变量。于是 L2 语义通道直接判为不可用，退化成关键词
检索（``[lexical-dis-max]``）。本模块为四类调用点统一补上「环境优先、``.env``
兜底」的读取：``_ingest``（transcriptions + chat_audio）、``_embedding``、``_synthesize``。

覆盖：
* 优先级（进程环境压过 ``.env``）；
* 环境缺失时用 ``.env``；
* 两处都没有 → 返回空，调用方**沿用原有报错文案**；
* ``.env`` 缺失 / 畸形 / 带引号 / 空行 / 注释都能容忍；
* 值绝不进入日志或错误；
* 四个调用点确实走这条回退。

全部离线：``httpx`` 用 ``sys.modules`` / ``setattr`` 注入假实现，不碰网络，
测试用唯一命名的 env 变量名，不会误命中机器上真实的 ``.env``。
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

from plugin.memory_governed import _config, _embedding, _ingest, _synthesize
from plugin.memory_governed._config import GovernedMemoryConfig, env_secret


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dotenv_home(tmp_path, monkeypatch):
    """把 ``HERMES_HOME`` 指向一个隔离目录，调用方自行往里写 ``.env``。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_env(home: Path, text: str) -> None:
    (home / ".env").write_text(text, encoding="utf-8")


def _install_httpx_post(monkeypatch, captured, response):
    """把 ``httpx.post`` 换成记录调用并返回固定响应的假实现。"""

    def _post(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        captured["json"] = kwargs.get("json")
        return response

    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(post=_post))


class _TextResp:
    """``/audio/transcriptions`` 的最小响应。"""

    status_code = 200
    text = ""

    def json(self):
        return {"text": "转写内容"}


class _ChatResp:
    """``/chat/completions``（chat_audio）的最小响应。"""

    status_code = 200
    text = ""

    def json(self):
        return {"choices": [{"message": {"content": "转写内容"}}]}


class _EmbedResp:
    """``/embeddings`` 的最小响应（含 ``raise_for_status``）。"""

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# 共享 helper 本体
# ---------------------------------------------------------------------------

def test_process_env_wins_over_dotenv(dotenv_home, monkeypatch):
    _write_env(dotenv_home, "K_PRECEDENCE=from_file\n")
    monkeypatch.setenv("K_PRECEDENCE", "from_env")
    assert env_secret("K_PRECEDENCE") == "from_env"


def test_dotenv_used_when_env_absent(dotenv_home, monkeypatch):
    monkeypatch.delenv("K_FALLBACK", raising=False)
    _write_env(dotenv_home, "K_FALLBACK=from_file\n")
    assert env_secret("K_FALLBACK") == "from_file"


def test_missing_everywhere_is_empty(dotenv_home, monkeypatch):
    monkeypatch.delenv("K_ABSENT", raising=False)
    _write_env(dotenv_home, "SOMETHING_ELSE=x\n")
    assert env_secret("K_ABSENT") == ""


def test_missing_dotenv_file_is_not_an_error(dotenv_home, monkeypatch):
    monkeypatch.delenv("K_NOHOME", raising=False)
    # 完全没有 .env 文件
    assert env_secret("K_NOHOME") == ""


def test_blank_or_empty_name_is_empty(dotenv_home):
    assert env_secret("") == ""
    assert env_secret("   ") == ""
    assert env_secret(None) == ""


def test_dotenv_tolerates_comments_blanks_malformed_and_quotes(dotenv_home, monkeypatch):
    for name in ("DQ", "SQ", "PLAIN", "AFTER_MALFORMED"):
        monkeypatch.delenv(name, raising=False)
    _write_env(dotenv_home, (
        "# a comment line\n"
        "\n"
        "this line has no equals sign\n"
        'DQ="double quoted"\n'
        "SQ='single quoted'\n"
        "PLAIN=plain value   \n"
        "=nameless key\n"
        "AFTER_MALFORMED=still parsed\n"
    ))
    assert env_secret("DQ") == "double quoted"
    assert env_secret("SQ") == "single quoted"
    assert env_secret("PLAIN") == "plain value"
    assert env_secret("AFTER_MALFORMED") == "still parsed"
    assert env_secret("nameless key") == ""


def test_hermes_home_env_points_the_lookup(dotenv_home, monkeypatch):
    monkeypatch.delenv("K_LOCATED", raising=False)
    _write_env(dotenv_home, "K_LOCATED=here\n")
    assert _config._hermes_home_path() == Path(dotenv_home)
    assert env_secret("K_LOCATED") == "here"


def test_value_never_appears_in_logs(dotenv_home, monkeypatch, caplog):
    monkeypatch.delenv("K_SECRET", raising=False)
    _write_env(dotenv_home, "K_SECRET=super-secret-value-xyz\n")
    with caplog.at_level(logging.DEBUG):
        assert env_secret("K_SECRET") == "super-secret-value-xyz"
    assert "super-secret-value-xyz" not in caplog.text


# ---------------------------------------------------------------------------
# 调用点：ASR（transcriptions 风格）
# ---------------------------------------------------------------------------

def _asr_cfg(api_key_env: str) -> GovernedMemoryConfig:
    cfg = GovernedMemoryConfig()
    cfg.asr.base_url = "https://api.example.com/v1"
    cfg.asr.model = "m"
    cfg.asr.api_key_env = api_key_env
    return cfg


def test_transcribe_audio_reads_key_from_dotenv(dotenv_home, tmp_path, monkeypatch):
    monkeypatch.delenv("ASR_DOTENV_KEY", raising=False)
    _write_env(dotenv_home, "ASR_DOTENV_KEY=sk-from-dotenv\n")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake")

    captured: dict = {}
    _install_httpx_post(monkeypatch, captured, _TextResp())

    r = _ingest.transcribe_audio(str(audio), _asr_cfg("ASR_DOTENV_KEY"))
    assert r["ok"] is True
    assert captured["headers"]["Authorization"] == "Bearer sk-from-dotenv"


def test_transcribe_audio_process_env_overrides_dotenv(dotenv_home, tmp_path, monkeypatch):
    _write_env(dotenv_home, "ASR_DOTENV_KEY=sk-from-dotenv\n")
    monkeypatch.setenv("ASR_DOTENV_KEY", "sk-from-env")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake")

    captured: dict = {}
    _install_httpx_post(monkeypatch, captured, _TextResp())

    r = _ingest.transcribe_audio(str(audio), _asr_cfg("ASR_DOTENV_KEY"))
    assert r["ok"] is True
    assert captured["headers"]["Authorization"] == "Bearer sk-from-env"


def test_transcribe_audio_missing_key_keeps_original_error(dotenv_home, tmp_path, monkeypatch):
    monkeypatch.delenv("ASR_DOTENV_MISSING", raising=False)
    _write_env(dotenv_home, "UNRELATED=1\n")
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"fake")

    r = _ingest.transcribe_audio(str(audio), _asr_cfg("ASR_DOTENV_MISSING"))
    assert r["ok"] is False
    assert r["error"] == "ASR api key env not set: ASR_DOTENV_MISSING"


# ---------------------------------------------------------------------------
# 调用点：ASR（chat_audio 风格）
# ---------------------------------------------------------------------------

def test_chat_audio_reads_key_from_dotenv(dotenv_home, tmp_path, monkeypatch):
    monkeypatch.delenv("MIMO_DOTENV_KEY", raising=False)
    _write_env(dotenv_home, "MIMO_DOTENV_KEY=sk-chat-dotenv\n")
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    def fake_transcode(src, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "c.mp3"
        p.write_bytes(b"mp3")
        return p

    monkeypatch.setattr(_ingest, "_transcode_to_mp3", fake_transcode)

    captured: dict = {}
    _install_httpx_post(monkeypatch, captured, _ChatResp())

    cfg = _asr_cfg("MIMO_DOTENV_KEY")
    cfg.asr.api_style = "chat_audio"
    cfg.asr.model = "mimo-v2.5-asr"

    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is True
    assert captured["headers"]["Authorization"] == "Bearer sk-chat-dotenv"


def test_chat_audio_missing_key_keeps_original_error(dotenv_home, tmp_path, monkeypatch):
    monkeypatch.delenv("MIMO_DOTENV_MISSING", raising=False)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"fake")

    cfg = _asr_cfg("MIMO_DOTENV_MISSING")
    cfg.asr.api_style = "chat_audio"

    r = _ingest._transcribe_chat_audio(str(audio), cfg, 30.0)
    assert r["ok"] is False
    assert r["error"] == "ASR api key env not set: MIMO_DOTENV_MISSING"


# ---------------------------------------------------------------------------
# 调用点：embedding
# ---------------------------------------------------------------------------

def test_embedding_signature_reads_key_from_dotenv(dotenv_home, monkeypatch):
    monkeypatch.delenv("EMB_DOTENV_SIG", raising=False)
    _write_env(dotenv_home, "EMB_DOTENV_SIG=emb-from-dotenv\n")
    cfg = GovernedMemoryConfig()
    cfg.embedding.api_key_env = "EMB_DOTENV_SIG"
    assert "emb-from-dotenv" in _embedding.EmbeddingService._build_signature(cfg)


def test_embedding_api_probe_uses_dotenv_key(dotenv_home, monkeypatch):
    monkeypatch.delenv("EMB_DOTENV_PROBE", raising=False)
    _write_env(dotenv_home, "EMB_DOTENV_PROBE=sk-emb-dotenv\n")
    _embedding.EmbeddingService.reset()

    captured: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["headers"] = headers
        texts = json["input"]
        texts = texts if isinstance(texts, list) else [texts]
        return _EmbedResp({
            "data": [{"index": i, "embedding": [0.25] * 1024}
                     for i in range(len(texts))],
        })

    monkeypatch.setattr("httpx.post", fake_post)

    cfg = GovernedMemoryConfig()
    cfg.embedding.provider = "openai"
    cfg.embedding.base_url = "https://api.example.com/v1"
    cfg.embedding.model = "text-embedding-3-small"
    cfg.embedding.api_key_env = "EMB_DOTENV_PROBE"

    try:
        service = _embedding.EmbeddingService.get(cfg)
        assert service.available is True
        assert captured["headers"]["Authorization"] == "Bearer sk-emb-dotenv"
    finally:
        _embedding.EmbeddingService.reset()


# ---------------------------------------------------------------------------
# 调用点：synthesis
# ---------------------------------------------------------------------------

def _synth_cfg(api_key_env: str) -> GovernedMemoryConfig:
    cfg = GovernedMemoryConfig()
    cfg.synthesis.enabled = True
    cfg.synthesis.base_url = "https://api.example.com/v1"
    cfg.synthesis.model = "m"
    cfg.synthesis.api_key_env = api_key_env
    return cfg


def test_synthesis_reads_key_from_dotenv(dotenv_home, monkeypatch):
    monkeypatch.delenv("SYN_DOTENV_KEY", raising=False)
    _write_env(dotenv_home, "SYN_DOTENV_KEY=sk-syn-dotenv\n")

    captured: dict = {}

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "[]"}}]}

    _install_httpx_post(monkeypatch, captured, _Resp())

    messages = [{"role": "user", "content": "记住：部署在 10.0.0.1 的 8080 端口"}]
    _synthesize.synthesize_notes(messages, _synth_cfg("SYN_DOTENV_KEY"))
    assert captured["headers"]["Authorization"] == "Bearer sk-syn-dotenv"


def test_synthesis_missing_key_keeps_original_warning(dotenv_home, monkeypatch, caplog):
    monkeypatch.delenv("SYN_DOTENV_MISSING", raising=False)
    _write_env(dotenv_home, "UNRELATED=1\n")

    messages = [{"role": "user", "content": "记住：部署在 10.0.0.1 的 8080 端口"}]
    with caplog.at_level(logging.WARNING):
        out = _synthesize.synthesize_notes(messages, _synth_cfg("SYN_DOTENV_MISSING"))

    assert out == []
    assert "synthesis api key env not set: SYN_DOTENV_MISSING" in caplog.text
