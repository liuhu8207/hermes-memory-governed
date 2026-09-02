#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-source knowledge collector for Obsidian integration."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from hermes_env import bootstrap, env_path, setup_logging
from wiki_utils import StateManager

logger = setup_logging("knowledge_collector")

HERMES_HOME = bootstrap(__file__)
WIKI_DIR = env_path("WIKI_DIR", HERMES_HOME / "wiki")
KNOWLEDGE_DIR = WIKI_DIR / "knowledge"

SUPPORTED_DOC_EXTENSIONS = {".md", ".txt", ".pdf", ".docx", ".doc"}
SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}


def ensure_dirs() -> None:
    for subdir in ["documents", "voices", "web", "chats", "inductions", "concepts", "categories"]:
        (KNOWLEDGE_DIR / subdir).mkdir(parents=True, exist_ok=True)


def get_state() -> StateManager:
    return StateManager(KNOWLEDGE_DIR, "collector")


def file_hash(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.md5(content).hexdigest()


def collect_documents(source_dirs: list[Path] | None = None, include_wiki: bool = True) -> list[dict[str, Any]]:
    if source_dirs is None:
        source_dirs = [HERMES_HOME / "documents", Path.home() / "Documents"]

    state = get_state()
    items: list[dict[str, Any]] = []
    exclude_dirs = {"knowledge", "memory", ".git", "cron", "scripts", "__pycache__"}

    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for ext in SUPPORTED_DOC_EXTENSIONS:
            for path in source_dir.rglob(f"*{ext}"):
                parts = {p.lower() for p in path.parts}
                if parts & exclude_dirs:
                    continue
                fhash = file_hash(path)
                if state.is_processed(fhash):
                    continue
                items.append({
                    "type": "document",
                    "path": str(path),
                    "hash": fhash,
                    "name": path.stem,
                    "extension": path.suffix,
                    "mtime": path.stat().st_mtime,
                })

    if include_wiki and WIKI_DIR.exists():
        for ext in SUPPORTED_DOC_EXTENSIONS:
            for path in WIKI_DIR.rglob(f"*{ext}"):
                parts = {p.lower() for p in path.relative_to(WIKI_DIR).parts}
                if parts & exclude_dirs:
                    continue
                if path.parent == WIKI_DIR:
                    continue
                fhash = file_hash(path)
                if state.is_processed(fhash):
                    continue
                items.append({
                    "type": "web",
                    "path": str(path),
                    "hash": fhash,
                    "name": path.stem,
                    "extension": path.suffix,
                    "mtime": path.stat().st_mtime,
                })

    logger.info("found %d new documents", len(items))
    return items


def collect_web_page(url: str) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=30, follow_redirects=True) as client:
            response = client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            html = response.text
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            title = title_match.group(1).strip() if title_match else url
            text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
            text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return {
                "type": "web",
                "url": url,
                "title": title,
                "content": text[:5000],
                "collected_at": datetime.now().isoformat(),
            }
    except Exception as exc:
        logger.warning("failed to collect web page %s: %s", url, exc)
        return None


def collect_clipboard(content: str, source: str = "clipboard") -> dict[str, Any] | None:
    if not content or not content.strip():
        return None
    return {
        "type": "clip",
        "content": content.strip(),
        "source": source,
        "collected_at": datetime.now().isoformat(),
    }


def collect_file(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None
    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return {
            "type": "file",
            "path": str(file_path),
            "name": file_path.stem,
            "content": content[:10000],
            "collected_at": datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.warning("failed to read file %s: %s", file_path, exc)
        return None


def transcribe_with_whisper_cli(audio_path: Path) -> str | None:
    try:
        import subprocess
        result = subprocess.run(
            ["whisper", str(audio_path), "--model", "base", "--language", "zh", "--output_format", "txt"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if result.returncode == 0:
            txt_path = audio_path.with_suffix(".txt")
            if txt_path.exists():
                text = txt_path.read_text(encoding="utf-8", errors="replace")
                txt_path.unlink()
                return text.strip()
        return None
    except Exception as exc:
        logger.debug("whisper CLI failed: %s", exc)
        return None


def transcribe_with_whisper_api(audio_path: Path) -> str | None:
    api_key = os.getenv("WHISPER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("WHISPER_BASE_URL", "https://api.openai.com/v1")
    if not api_key:
        return None
    try:
        import httpx
        with httpx.Client(timeout=300) as client:
            with open(audio_path, "rb") as f:
                response = client.post(
                    f"{base_url.rstrip('/')}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"file": (audio_path.name, f, "audio/mpeg")},
                    data={"model": "whisper-1", "language": "zh"},
                )
            response.raise_for_status()
            return response.json().get("text", "").strip()
    except Exception as exc:
        logger.debug("whisper API failed: %s", exc)
        return None


def transcribe_audio(audio_path: Path) -> str | None:
    text = transcribe_with_whisper_cli(audio_path)
    if text:
        logger.info("transcribed with whisper CLI: %s", audio_path.name)
        return text
    text = transcribe_with_whisper_api(audio_path)
    if text:
        logger.info("transcribed with whisper API: %s", audio_path.name)
        return text
    logger.warning("could not transcribe: %s", audio_path.name)
    return None


def collect_audio(file_path: Path) -> dict[str, Any] | None:
    if not file_path.exists():
        return None
    text = transcribe_audio(file_path)
    if not text:
        text = f"[音频文件: {file_path.name}]"
    return {
        "type": "voice",
        "path": str(file_path),
        "name": file_path.stem,
        "content": text[:10000],
        "duration_estimate": len(text) // 10,
        "collected_at": datetime.now().isoformat(),
    }


def collect_audio_dir(audio_dir: Path | None = None) -> list[dict[str, Any]]:
    if audio_dir is None:
        audio_dir = HERMES_HOME / "audio"
    items: list[dict[str, Any]] = []
    if not audio_dir.exists():
        return items
    for ext in SUPPORTED_AUDIO_EXTENSIONS:
        for path in audio_dir.rglob(f"*{ext}"):
            item = collect_audio(path)
            if item:
                items.append(item)
    logger.info("collected %d audio files", len(items))
    return items


def collect_batch(urls: list[str] | None = None, files: list[Path] | None = None, clips: list[str] | None = None, audio_dir: Path | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    items.extend(collect_documents())
    items.extend(collect_audio_dir(audio_dir))
    if urls:
        for url in urls:
            item = collect_web_page(url)
            if item:
                items.append(item)
    if files:
        for file_path in files:
            if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                item = collect_audio(file_path)
            else:
                item = collect_file(file_path)
            if item:
                items.append(item)
    if clips:
        for clip in clips:
            item = collect_clipboard(clip)
            if item:
                items.append(item)
    logger.info("total items collected: %d", len(items))
    return items


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge collector")
    parser.add_argument("--url", action="append", help="URL to collect")
    parser.add_argument("--file", action="append", help="File path to collect")
    parser.add_argument("--clip", action="append", help="Clipboard text to collect")
    parser.add_argument("--audio-dir", help="Directory to scan for audio files")
    args = parser.parse_args()

    ensure_dirs()
    items = collect_batch(
        urls=args.url,
        files=[Path(f) for f in args.file] if args.file else None,
        clips=args.clip,
        audio_dir=Path(args.audio_dir) if args.audio_dir else None,
    )

    state = get_state()
    state.last_run = datetime.now().isoformat()

    for item in items:
        print(json.dumps(item, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
