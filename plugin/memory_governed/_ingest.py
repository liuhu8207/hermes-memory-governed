# -*- coding: utf-8 -*-
"""Ingestion helpers for the governed knowledge base.

把「外部资料 → 结构化文本」的机械活收在这里，供 agent 的摄入工具调用。

**职责边界**（重要）：本模块只负责**抓取/读取/转录**，不做「归纳总结」。
归纳由 agent 的 LLM 完成（决策：归纳用 agent 同款 LLM），落库用
``governed_kb_add``。这样任何 agent 都能走同一条链路：

    ingest 工具（拿内容） → agent 归纳 → governed_kb_add（落库）

三类摄入，全部**可选依赖降级**，绝不抛异常（返回 ``{ok, error}``）：

- ``fetch_url``: URL → 正文（HTML 提取 / 纯文本直读）
- ``read_file``: 本地文件 → 文本（txt/md/pdf/docx；pdf/docx 依赖可选）
- ``transcribe_audio``: 音频 → 转写文本（ASR API，OpenAI 兼容
  ``/audio/transcriptions``，默认硅基流动 XingChenASR）

统一返回结构 ``{ok: bool, ..., error?: str}``，方便工具层直接透传。
"""

from __future__ import annotations

import html as _html
import logging
import os
import re as _re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

#: 摄入内容的最大字符数（超长截断，避免爆 agent context；归纳用原文更优，
#: 但不可控的网页/文件需要上限）。
MAX_CONTENT_CHARS: int = 50000

_TAG_RE = _re.compile(r"<[^>]+>")
#: 正文无关块（脚本/样式/导航/页眉页脚），连同内容整体删除
_NOISE_BLOCK_RE = _re.compile(
    r"<(script|style|nav|header|footer|aside|noscript|form)[^>]*>.*?</\1>",
    _re.S | _re.I,
)
#: 块级标签 → 换行（保留段落结构）
_BLOCK_RE = _re.compile(
    r"</?(p|div|li|h[1-6]|tr|br|section|article|blockquote|pre)[^>]*>",
    _re.I,
)
_TITLE_RE = _re.compile(r"<title[^>]*>(.*?)</title>", _re.S | _re.I)

_AUDIO_MIME = {
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: float) -> tuple[int, str, Dict[str, str]]:
    """GET 抓取。优先 httpx，降级 urllib（标准库）。返回 (status, text, headers)。"""
    headers = {"User-Agent": "hermes-memory-governed/0.1 (+knowledge-base)"}
    try:
        import httpx  # type: ignore

        r = httpx.get(url, timeout=timeout, follow_redirects=True, headers=headers)
        return r.status_code, r.text, {k.lower(): v for k, v in r.headers.items()}
    except ImportError:
        pass
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, {k.lower(): v for k, v in resp.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, "", {}
    except Exception as e:  # noqa: BLE001
        logger.debug("http get failed for %s: %s", url, e)
        return 0, "", {}


def _html_to_text(html_str: str) -> str:
    """HTML → 纯文本（内置轻量提取器，无第三方依赖）。"""
    text = _NOISE_BLOCK_RE.sub(" ", html_str)
    text = _BLOCK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    text = _re.sub(r"[ \t\r]+", " ", text)
    text = _re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _truncate(content: str, limit: int = MAX_CONTENT_CHARS) -> tuple[str, bool]:
    if len(content) <= limit:
        return content, False
    return content[:limit], True


def _is_html(content_type: str) -> bool:
    return "html" in (content_type or "").lower()


def _read_text(path: Path) -> str:
    """读文本文件，utf-8 → gbk → 兜底 errors=replace。"""
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 摄入：URL
# ---------------------------------------------------------------------------

def fetch_url(url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """抓取 URL 并提取正文。返回 ``{ok, url, title, content, content_type, truncated}``。"""
    url = (url or "").strip()
    if not url:
        return {"ok": False, "error": "Missing url"}
    if not url.lower().startswith(("http://", "https://")):
        return {"ok": False, "error": f"Unsupported scheme (http/https only): {url}"}

    try:
        status, body, headers = _http_get(url, timeout)
    except Exception as e:  # noqa: BLE001 — 摄入工具绝不抛给 agent
        return {"ok": False, "error": f"fetch failed: {e}"}
    if status == 0:
        return {"ok": False, "error": f"fetch failed (network error): {url}"}
    if status >= 400:
        return {"ok": False, "error": f"HTTP {status}: {url}"}

    content_type = headers.get("content-type", "")
    if _is_html(content_type):
        title_m = _TITLE_RE.search(body)
        title = _html.unescape(title_m.group(1)).strip() if title_m else url.rstrip("/").split("/")[-1] or url
        content = _html_to_text(body)
    else:
        # 纯文本 / JSON / CSV 等直接返回原文（JSON 保持可读）
        title = url.rstrip("/").split("/")[-1] or url
        content = body

    if not content.strip():
        return {"ok": False, "error": f"no extractable content from {url}"}
    content, truncated = _truncate(content)
    return {
        "ok": True,
        "url": url,
        "title": title,
        "content": content,
        "content_type": content_type.split(";")[0].strip() or "text",
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# 摄入：本地文件
# ---------------------------------------------------------------------------

_TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".log",
    ".py", ".js", ".ts", ".html", ".htm", ".xml", ".yaml", ".yml",
    ".rst", ".tex", ".org",
}


def _read_pdf(path: Path) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return {"ok": False, "error": "reading PDF requires pypdf: pip install pypdf"}
    try:
        reader = PdfReader(str(path))
        parts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(parts)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"pdf extract failed: {e}"}
    if not text.strip():
        return {"ok": False, "error": f"no extractable text (scanned PDF?): {path.name}"}
    return {"ok": True, "content": text}


def _read_docx(path: Path) -> Dict[str, Any]:
    try:
        import docx  # type: ignore
    except ImportError:
        return {"ok": False, "error": "reading docx requires python-docx: pip install python-docx"}
    try:
        d = docx.Document(str(path))
        paras = [p.text for p in d.paragraphs if p.text.strip()]
        # 表格也纳入
        for table in d.tables:
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    paras.append(" | ".join(cells))
        text = "\n".join(paras)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"docx extract failed: {e}"}
    if not text.strip():
        return {"ok": False, "error": f"no extractable text: {path.name}"}
    return {"ok": True, "content": text}


def read_file(path: str) -> Dict[str, Any]:
    """读本地文件提取文本。按扩展名分派（txt/md 直读、pdf/docx 可选依赖）。

    返回 ``{ok, path, title, content, content_type, truncated}``。
    """
    p = Path((path or "").strip())
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}
    if not p.is_file():
        return {"ok": False, "error": f"not a file: {path}"}

    suffix = p.suffix.lower()
    title = p.stem

    if suffix in _TEXT_SUFFIXES:
        try:
            content = _read_text(p)
        except OSError as e:
            return {"ok": False, "error": f"read failed: {e}"}
    elif suffix == ".pdf":
        r = _read_pdf(p)
        if not r.get("ok"):
            return r
        content = r["content"]
    elif suffix == ".docx":
        r = _read_docx(p)
        if not r.get("ok"):
            return r
        content = r["content"]
    else:
        return {"ok": False, "error": f"unsupported file type: {suffix or '(none)'}"}

    if not content.strip():
        return {"ok": False, "error": f"empty content: {path}"}
    content, truncated = _truncate(content)
    return {
        "ok": True,
        "path": str(p),
        "title": title,
        "content": content,
        "content_type": suffix.lstrip(".") or "text",
        "truncated": truncated,
    }


# ---------------------------------------------------------------------------
# 摄入：音频转写（ASR）
# ---------------------------------------------------------------------------

def transcribe_audio(path: str, config: Any, timeout: float = 180.0) -> Dict[str, Any]:
    """音频 → 转写文本（OpenAI 兼容 ``/audio/transcriptions``）。

    ASR 配置在 ``config.asr``（provider/api_key_env/base_url/model/language）。
    依赖 httpx（multipart 上传）；未装或未配置时返回清晰的 ``error``。
    """
    p = Path((path or "").strip())
    if not p.exists():
        return {"ok": False, "error": f"file not found: {path}"}

    asr = getattr(config, "asr", None)
    base_url = str(getattr(asr, "base_url", "") or "").strip()
    model = str(getattr(asr, "model", "") or "").strip()
    if not base_url or not model:
        return {"ok": False, "error": "ASR not configured (asr.base_url / asr.model)"}

    api_key_env = str(getattr(asr, "api_key_env", "") or "")
    api_key = os.environ.get(api_key_env, "") if api_key_env else ""
    if not api_key:
        return {"ok": False, "error": f"ASR api key env not set: {api_key_env}"}

    try:
        import httpx  # type: ignore
    except ImportError:
        return {"ok": False, "error": "transcription requires httpx: pip install httpx"}

    mime = _AUDIO_MIME.get(p.suffix.lower(), "application/octet-stream")
    endpoint = base_url.rstrip("/") + "/audio/transcriptions"
    data: Dict[str, Any] = {"model": model}
    language = str(getattr(asr, "language", "") or "").strip()
    if language:
        data["language"] = language

    try:
        with p.open("rb") as f:
            audio_bytes = f.read()
        r = httpx.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (p.name, audio_bytes, mime)},
            data=data,
            timeout=timeout,
        )
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"transcription request failed: {e}"}

    if r.status_code >= 400:
        return {"ok": False, "error": f"ASR HTTP {r.status_code}: {r.text[:300]}"}
    try:
        payload = r.json()
        text = payload.get("text", "") or ""
    except Exception:  # noqa: BLE001
        return {"ok": False, "error": "ASR returned non-JSON response"}

    if not text.strip():
        return {"ok": False, "error": "ASR returned empty transcript"}
    text, truncated = _truncate(text)
    return {
        "ok": True,
        "path": str(p),
        "text": text,
        "model": model,
        "truncated": truncated,
    }
