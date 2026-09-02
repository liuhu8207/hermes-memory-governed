# -*- coding: utf-8 -*-
"""Ingestion layer tests (Phase 2): fetch_url / read_file / transcribe_audio.

全部不依赖真实网络 / 外部 API / 重依赖：fetch 用 mock 的 ``_http_get``，
pdf/docx 用 monkeypatch 强制 ImportError 验证降级提示，transcribe 用
``sys.modules`` 注入 fake httpx。
"""

from __future__ import annotations

import sys
import types

from plugin.memory_governed import _ingest
from plugin.memory_governed._config import GovernedMemoryConfig


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

def _mock_get(status, body, ctype="text/html"):
    def _get(url, timeout):
        return status, body, {"content-type": ctype}
    return _get


def test_fetch_html_extracts_title_and_body(monkeypatch):
    html = (
        "<html><head><title>我的文章</title></head>"
        "<body><script>alert('x')</script><p>这是正文第一段。</p>"
        "<p>这是正文第二段。</p></body></html>"
    )
    monkeypatch.setattr(_ingest, "_http_get", _mock_get(200, html))
    r = _ingest.fetch_url("https://example.com/a")
    assert r["ok"] is True
    assert r["title"] == "我的文章"
    assert "正文第一段" in r["content"]
    assert "alert" not in r["content"]  # script 被剥掉


def test_fetch_plain_text(monkeypatch):
    monkeypatch.setattr(_ingest, "_http_get", _mock_get(200, "plain text body", "text/plain"))
    r = _ingest.fetch_url("https://example.com/notes.txt")
    assert r["ok"] is True
    assert r["content"] == "plain text body"


def test_fetch_rejects_non_http():
    r = _ingest.fetch_url("file:///etc/passwd")
    assert r["ok"] is False
    assert "scheme" in r["error"]


def test_fetch_http_error(monkeypatch):
    monkeypatch.setattr(_ingest, "_http_get", _mock_get(404, "not found"))
    r = _ingest.fetch_url("https://example.com/missing")
    assert r["ok"] is False
    assert "404" in r["error"]


def test_fetch_empty_url():
    r = _ingest.fetch_url("")
    assert r["ok"] is False
    assert "Missing url" in r["error"]


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def test_read_text_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# 标题\n\n内容", encoding="utf-8")
    r = _ingest.read_file(str(f))
    assert r["ok"] is True
    assert r["title"] == "note"
    assert "标题" in r["content"]


def test_read_gbk_encoded(tmp_path):
    f = tmp_path / "gbk.txt"
    f.write_bytes("中文内容".encode("gbk"))
    r = _ingest.read_file(str(f))
    assert r["ok"] is True
    assert "中文内容" in r["content"]


def test_read_missing_file(tmp_path):
    r = _ingest.read_file(str(tmp_path / "nope.txt"))
    assert r["ok"] is False
    assert "not found" in r["error"]


def test_read_unsupported_type(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"\x00\x01")
    r = _ingest.read_file(str(f))
    assert r["ok"] is False
    assert "unsupported" in r["error"]


def test_read_pdf_missing_dep(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pypdf":
            raise ImportError("no pypdf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    r = _ingest.read_file(str(pdf))
    assert r["ok"] is False
    assert "pypdf" in r["error"]


def test_read_docx_missing_dep(tmp_path, monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "docx":
            raise ImportError("no docx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    d = tmp_path / "a.docx"
    d.write_bytes(b"PK fake")
    r = _ingest.read_file(str(d))
    assert r["ok"] is False
    assert "python-docx" in r["error"]


# ---------------------------------------------------------------------------
# transcribe_audio
# ---------------------------------------------------------------------------

def test_transcribe_not_configured(tmp_path):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    r = _ingest.transcribe_audio(str(audio), cfg)
    assert r["ok"] is False
    assert "not configured" in r["error"]


def test_transcribe_missing_key(tmp_path, monkeypatch):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.base_url = "https://api.example.com/v1"
    cfg.asr.model = "m"
    cfg.asr.api_key_env = "ASR_KEY_MISSING"
    monkeypatch.delenv("ASR_KEY_MISSING", raising=False)
    r = _ingest.transcribe_audio(str(audio), cfg)
    assert r["ok"] is False
    assert "api key" in r["error"]


def test_transcribe_success(tmp_path, monkeypatch):
    audio = tmp_path / "rec.m4a"
    audio.write_bytes(b"fake-audio")
    cfg = GovernedMemoryConfig()
    cfg.asr.base_url = "https://api.siliconflow.cn/v1"
    cfg.asr.model = "XingChenAGI/XingChenASR-V3.2-Ultra"
    cfg.asr.api_key_env = "ASR_KEY"
    monkeypatch.setenv("ASR_KEY", "sk-test")

    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return {"text": "会议纪要内容"}

    def _fake_post(endpoint, **kwargs):
        captured["endpoint"] = endpoint
        captured["kwargs"] = kwargs
        return _Resp()

    fake_httpx = types.SimpleNamespace(post=_fake_post)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    r = _ingest.transcribe_audio(str(audio), cfg)
    assert r["ok"] is True
    assert r["text"] == "会议纪要内容"
    assert captured["endpoint"] == "https://api.siliconflow.cn/v1/audio/transcriptions"
    assert captured["kwargs"]["data"]["model"] == "XingChenAGI/XingChenASR-V3.2-Ultra"
    # multipart 上传了音频文件
    assert captured["kwargs"]["files"]["file"][0] == "rec.m4a"


def test_transcribe_http_error(tmp_path, monkeypatch):
    audio = tmp_path / "a.mp3"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.base_url = "https://api.example.com/v1"
    cfg.asr.model = "m"
    cfg.asr.api_key_env = "ASR_KEY"
    monkeypatch.setenv("ASR_KEY", "sk")

    class _Resp:
        status_code = 401
        text = "unauthorized"

    fake_httpx = types.SimpleNamespace(post=lambda *a, **kw: _Resp())
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    r = _ingest.transcribe_audio(str(audio), cfg)
    assert r["ok"] is False
    assert "401" in r["error"]
