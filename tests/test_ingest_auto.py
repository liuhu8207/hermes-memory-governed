# -*- coding: utf-8 -*-
"""Long-audio transcription tests: probe / split / ``transcribe_audio_auto``.

Why this file exists
--------------------
The ASR endpoint (SiliconFlow XingChenASR-V3.2-Ultra) measures 6.7~8.3 seconds
per audio-minute, so the old hardcoded 180s request ceiling failed every
recording past ~25 minutes — a one-hour meeting never transcribed. The fix
splits long audio into re-encoded mp3 chunks and joins the transcripts.

Nothing here touches the network or the real API: ``httpx`` is never imported,
``transcribe_audio`` is replaced, and the ffmpeg/ffprobe tools are stubbed. The
one real end-to-end run lives outside the suite (the CLI probe).
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from plugin.memory_governed import _ingest
from plugin.memory_governed._config import GovernedMemoryConfig


def _ok(text: str) -> dict:
    return {"ok": True, "path": "x", "text": text, "model": "m", "truncated": False}


# ---------------------------------------------------------------------------
# probe_audio_duration
# ---------------------------------------------------------------------------

def test_unknown_duration_is_none_without_raising(tmp_path, monkeypatch):
    # 文件不存在：直接 None
    assert _ingest.probe_audio_duration(str(tmp_path / "nope.m4a")) is None
    # 非音频文件 + ffprobe 不可用：也 None，而不是抛异常
    junk = tmp_path / "data.bin"
    junk.write_bytes(b"\x00\x01\x02")
    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: None)
    assert _ingest.probe_audio_duration(str(junk)) is None


def test_wav_duration_falls_back_to_stdlib_when_ffprobe_missing(tmp_path, monkeypatch):
    wav = tmp_path / "clip.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 8000)  # 8000 帧 @ 8kHz = 1.0s

    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: None)
    dur = _ingest.probe_audio_duration(str(wav))
    assert dur is not None
    assert abs(dur - 1.0) < 0.05


# ---------------------------------------------------------------------------
# transcribe_audio_auto — short audio
# ---------------------------------------------------------------------------

def test_short_audio_does_one_call_and_never_splits(tmp_path, monkeypatch):
    audio = tmp_path / "short.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.timeout_seconds = 600.0
    cfg.asr.chunk_minutes = 10.0

    calls = []

    def _fake_transcribe(path, config, timeout=180.0):
        calls.append((path, timeout))
        return _ok("hello")

    monkeypatch.setattr(_ingest, "transcribe_audio", _fake_transcribe)
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 30.0)

    def _boom(*a, **k):
        raise AssertionError("短音频不应触发 split_audio")

    monkeypatch.setattr(_ingest, "split_audio", _boom)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert r["text"] == "hello"
    assert len(calls) == 1
    assert calls[0][1] == 600.0  # 超时来自 config.asr.timeout_seconds


def test_unknown_duration_falls_back_to_single_call(tmp_path, monkeypatch):
    # 用「已接受」的容器，专测「探测不到时长 → 单次」这条路，
    # 避免被容器归一化分支抢先（那条另有专门的测试）。
    audio = tmp_path / "weird.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()

    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: None)
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0: _ok("single"))

    def _boom(*a, **k):
        raise AssertionError("时长未知时不应拆段")

    monkeypatch.setattr(_ingest, "split_audio", _boom)
    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert r["text"] == "single"


# ---------------------------------------------------------------------------
# transcribe_audio_auto — long audio
# ---------------------------------------------------------------------------

def _long_setup(tmp_path, monkeypatch, duration, chunk_minutes, n_chunks=None):
    audio = tmp_path / "long.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.chunk_minutes = chunk_minutes
    cfg.asr.timeout_seconds = 12.0
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: duration)
    chunk_paths = [tmp_path / f"chunk_{i:04d}.mp3"
                   for i in range(n_chunks if n_chunks else 3)]
    monkeypatch.setattr(_ingest, "split_audio", lambda *a, **k: list(chunk_paths))
    return audio, cfg


def test_long_audio_splits_and_joins_in_order(tmp_path, monkeypatch):
    audio, cfg = _long_setup(tmp_path, monkeypatch, duration=150.0,
                             chunk_minutes=1.0, n_chunks=3)
    seen = []

    def _fake_transcribe(path, config, timeout=180.0):
        seen.append((Path(path).stem, timeout))
        return _ok(f"part-{Path(path).stem}")

    monkeypatch.setattr(_ingest, "transcribe_audio", _fake_transcribe)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert r["chunks"] == 3
    assert r["duration_seconds"] == 150.0
    assert r["text"] == "part-chunk_0000\n\npart-chunk_0001\n\npart-chunk_0002"
    assert [t for _, t in seen] == [12.0, 12.0, 12.0]


def test_one_chunk_fails_keeps_the_rest_and_reports_the_index(tmp_path, monkeypatch):
    audio, cfg = _long_setup(tmp_path, monkeypatch, duration=150.0,
                             chunk_minutes=1.0, n_chunks=3)

    def _fake_transcribe(path, config, timeout=180.0):
        i = int(Path(path).stem.split("_")[1])
        if i == 1:
            return {"ok": False, "error": "ASR HTTP 502: bad gateway"}
        return _ok(f"part-{i}")

    monkeypatch.setattr(_ingest, "transcribe_audio", _fake_transcribe)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is False
    assert r["partial"] is True
    # 1-based：chunk_0001 是第 2 段
    assert r["chunks_failed"] == [2]
    assert "part-0" in r["text"] and "part-2" in r["text"]
    assert "part-1" not in r["text"]
    assert "chunk 2/3 failed" in r["error"] and "502" in r["error"]
    assert r["chunks"] == 3


def test_all_chunks_fail_yields_empty_text(tmp_path, monkeypatch):
    audio, cfg = _long_setup(tmp_path, monkeypatch, duration=150.0,
                             chunk_minutes=1.0, n_chunks=3)
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0:
                        {"ok": False, "error": "ASR HTTP 500: boom"})

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is False
    assert r["partial"] is True
    assert r["text"] == ""
    assert r["chunks_failed"] == [1, 2, 3]  # 1-based


def test_split_producing_no_chunks_is_not_a_success(tmp_path, monkeypatch):
    """一个没有内容的成功报告正是本项目反复被坑的那类静默失败。"""
    audio, cfg = _long_setup(tmp_path, monkeypatch, duration=418.0,
                             chunk_minutes=1.0, n_chunks=3)
    monkeypatch.setattr(_ingest, "split_audio", lambda *a, **k: [])
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0: _ok("x"))

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is False
    assert r["chunks"] == 0
    assert r["text"] == ""
    assert r["error"].strip(), "空片段必须带非空 error"


def test_ffmpeg_missing_and_long_audio_names_ffmpeg_not_timeout(tmp_path, monkeypatch):
    audio = tmp_path / "long.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.chunk_minutes = 1.0
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 300.0)
    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: None)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is False
    assert "ffmpeg" in r["error"].lower()
    assert "300" in r["error"]
    assert "timeout" not in r["error"].lower()


def test_temp_directory_is_removed_after_the_call(tmp_path, monkeypatch):
    audio = tmp_path / "long.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.chunk_minutes = 1.0
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 150.0)
    monkeypatch.setattr(_ingest, "split_audio",
                        lambda *a, **k: [tmp_path / "chunk_0000.mp3"])
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0: _ok("t"))

    created = {}
    real_mkdtemp = _ingest.tempfile.mkdtemp

    def _fake_mkdtemp(*a, **k):
        d = real_mkdtemp()
        created["dir"] = d
        return d

    monkeypatch.setattr(_ingest.tempfile, "mkdtemp", _fake_mkdtemp)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert "dir" in created
    assert not Path(created["dir"]).exists(), "临时目录没有被清理"


def test_temp_directory_is_removed_even_when_splitting_fails(tmp_path, monkeypatch):
    audio = tmp_path / "long.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.chunk_minutes = 1.0
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 150.0)
    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: "ffmpeg")

    def _boom_split(*a, **k):
        raise RuntimeError("ffmpeg failed splitting x: boom")

    monkeypatch.setattr(_ingest, "split_audio", _boom_split)

    created = {}
    real_mkdtemp = _ingest.tempfile.mkdtemp

    def _fake_mkdtemp(*a, **k):
        d = real_mkdtemp()
        created["dir"] = d
        return d

    monkeypatch.setattr(_ingest.tempfile, "mkdtemp", _fake_mkdtemp)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is False
    assert "split failed" in r["error"]
    assert "dir" in created
    assert not Path(created["dir"]).exists()


# ---------------------------------------------------------------------------
# container normalisation (_AUDIO_MIME 之外的容器，无论长短)
# ---------------------------------------------------------------------------

def test_empty_path_is_a_structured_error(tmp_path):
    cfg = GovernedMemoryConfig()
    for blank in ("", "   ", "\t\n"):
        r = _ingest.transcribe_audio_auto(blank, cfg)
        assert r["ok"] is False
        assert "no path given" in r["error"]


def test_non_native_container_is_transcoded_even_when_short(tmp_path, monkeypatch):
    """短 .amr 也必须归一化 —— 端点解不了 amr，原文上传只会 HTTP 400。"""
    amr = tmp_path / "voice.amr"
    amr.write_bytes(b"#!AMR\n fake")
    cfg = GovernedMemoryConfig()
    cfg.asr.chunk_minutes = 10.0

    converted = tmp_path / "voice.mp3"
    converted.write_bytes(b"fake-mp3")
    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: "ffmpeg")

    transcode_calls = []

    def _fake_transcode(src, out_dir):
        transcode_calls.append(str(src))
        return converted

    monkeypatch.setattr(_ingest, "_transcode_to_mp3", _fake_transcode)
    # 转出来的 mp3 是「已接受」容器；短 → 单次
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 11.4)

    seen = []

    def _fake_transcribe(path, config, timeout=180.0):
        seen.append(str(path))
        return _ok("hello from amr")

    monkeypatch.setattr(_ingest, "transcribe_audio", _fake_transcribe)

    def _boom(*a, **k):
        raise AssertionError("短录音不应触发拆段")

    monkeypatch.setattr(_ingest, "split_audio", _boom)

    r = _ingest.transcribe_audio_auto(str(amr), cfg)
    assert r["ok"] is True
    assert r["text"] == "hello from amr"
    assert transcode_calls == [str(amr)]
    # 转写用的是转出来的 mp3，不是原始 amr
    assert seen == [str(converted)]
    # 但信封里的 path 必须指调用方传入的 amr —— 临时 mp3 返回前已被删除
    assert r["path"] == str(amr)


def test_non_native_container_without_ffmpeg_is_refused(tmp_path, monkeypatch):
    amr = tmp_path / "voice.amr"
    amr.write_bytes(b"#!AMR\n fake")
    cfg = GovernedMemoryConfig()
    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: None)

    def _boom(*a, **k):
        raise AssertionError("ffmpeg 缺失时绝不能上传原文")

    monkeypatch.setattr(_ingest, "transcribe_audio", _boom)

    r = _ingest.transcribe_audio_auto(str(amr), cfg)
    assert r["ok"] is False
    assert ".amr" in r["error"]
    assert "ffmpeg" in r["error"].lower()


def test_non_native_container_that_ffmpeg_cannot_decode_is_refused(tmp_path, monkeypatch):
    """这正是一个 .silk 的结果：结构化错误，绝不是裸 HTTP 400 或静默成功。"""
    silk = tmp_path / "voice.silk"
    silk.write_bytes(b"\x02#!SILK_V3 fake")
    cfg = GovernedMemoryConfig()
    monkeypatch.setattr(_ingest, "_ffmpeg_tool", lambda name: "ffmpeg")

    def _fake_transcode(src, out_dir):
        raise RuntimeError("ffmpeg failed transcoding voice.silk: Invalid data found")

    monkeypatch.setattr(_ingest, "_transcode_to_mp3", _fake_transcode)

    def _boom(*a, **k):
        raise AssertionError("无法解码时绝不能上传原文")

    monkeypatch.setattr(_ingest, "transcribe_audio", _boom)

    r = _ingest.transcribe_audio_auto(str(silk), cfg)
    assert r["ok"] is False
    assert ".silk" in r["error"]
    assert r["error"].strip()


# ---------------------------------------------------------------------------
# timeout resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("junk", [0, -5, "x", None, float("nan")])
def test_bad_timeout_falls_back_to_default(tmp_path, monkeypatch, junk):
    audio = tmp_path / "s.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.timeout_seconds = junk  # type: ignore[assignment]

    seen = []

    def _fake_transcribe(path, config, timeout=180.0):
        seen.append(timeout)
        return _ok("t")

    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 5.0)
    monkeypatch.setattr(_ingest, "transcribe_audio", _fake_transcribe)

    r = _ingest.transcribe_audio_auto(str(audio), cfg)
    assert r["ok"] is True
    assert seen == [600.0]


def test_good_timeout_is_used_verbatim(tmp_path, monkeypatch):
    audio = tmp_path / "s.m4a"
    audio.write_bytes(b"x")
    cfg = GovernedMemoryConfig()
    cfg.asr.timeout_seconds = 42.0

    seen = []
    monkeypatch.setattr(_ingest, "probe_audio_duration", lambda p: 5.0)
    monkeypatch.setattr(_ingest, "transcribe_audio",
                        lambda path, config, timeout=180.0:
                        (seen.append(timeout), _ok("t"))[1])

    _ingest.transcribe_audio_auto(str(audio), cfg)
    assert seen == [42.0]


# ---------------------------------------------------------------------------
# _ffmpeg_tool resolution
# ---------------------------------------------------------------------------

def test_ffmpeg_tool_prefers_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HGM_FFMPEG", "/custom/ffmpeg")
    assert _ingest._ffmpeg_tool("ffmpeg") == "/custom/ffmpeg"
    monkeypatch.setenv("HGM_FFPROBE", "/custom/ffprobe")
    assert _ingest._ffmpeg_tool("ffprobe") == "/custom/ffprobe"


def test_ffmpeg_tool_returns_none_when_absent(monkeypatch):
    monkeypatch.delenv("HGM_FFMPEG", raising=False)
    monkeypatch.delenv("HGM_FFPROBE", raising=False)
    monkeypatch.setattr(_ingest.shutil, "which", lambda name: None)
    assert _ingest._ffmpeg_tool("ffmpeg") is None
    assert _ingest._ffmpeg_tool("ffprobe") is None
