#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared utilities for wiki/knowledge scripts (lightweight subset).

Migrated from hermes-memory-system/scripts/wiki_utils.py (2026-08-28).
Contains only the helpers used by the migrated knowledge scripts:
- atomic_write_json: crash-safe JSON writes
- StateManager: simple JSON-backed key-value state with processed-set support
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

logger = logging.getLogger("wiki_utils")

T = TypeVar("T")


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically (temp file + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class StateManager:
    """Simple JSON-backed state store with processed-set helpers."""

    def __init__(self, state_dir: Path, name: str = "global") -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = state_dir / f".{name}_state.json"
        self._state: dict[str, Any] | None = None

    def load(self) -> dict[str, Any]:
        if self._state is None:
            if self.state_file.exists():
                try:
                    self._state = json.loads(self.state_file.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    self._state = {}
            else:
                self._state = {}
        return self._state

    def save(self) -> None:
        if self._state is not None:
            atomic_write_json(self.state_file, self._state)

    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.load()
        self._state[key] = value
        self.save()

    def add_to_set(self, key: str, value: str) -> None:
        self.load()
        items = self._state.get(key, [])
        if value not in items:
            items.append(value)
            self._state[key] = items
            self.save()

    def has(self, key: str, value: str) -> bool:
        return value in self.load().get(key, [])

    def mark_processed(self, identifier: str, category: str = "processed") -> None:
        self.add_to_set(category, identifier)

    def is_processed(self, identifier: str, category: str = "processed") -> bool:
        return identifier in self.load().get(category, [])


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        sm = StateManager(Path(td), "test")
        sm.set("a", 1)
        sm.mark_processed("x")
        assert sm.get("a") == 1
        assert sm.is_processed("x")
        print("StateManager smoke OK")


def retry(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry decorator with exponential backoff (from hermes-memory-system)."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(f"{func.__name__} attempt {attempt}/{max_attempts} failed: {exc}, retry in {wait:.1f}s")
                        time.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


def load_log(log_path: Path) -> dict[str, float]:
    """Load a JSON log file, returning {} if missing/corrupt."""
    if log_path.exists():
        return json.loads(log_path.read_text(encoding="utf-8", errors="replace"))
    return {}


def save_log(log_path: Path, data: dict[str, Any]) -> None:
    """Persist a JSON log atomically (mirror of :func:`load_log`)."""
    atomic_write_json(log_path, data)


# ---------------------------------------------------------------------------
# Generic text helpers
# ---------------------------------------------------------------------------

def slugify(text: str, max_len: int = 80) -> str:
    """Convert arbitrary text into a filesystem-safe slug (CJK preserved)."""
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", "-", text, flags=re.UNICODE)
    text = text.strip("-")
    return (text[:max_len].rstrip("-") or "untitled")


def extract_summary(content: str, max_len: int = 500) -> str:
    """Extract a short summary from a generated Markdown report.

    Prefers an explicit 摘要/Summary heading, otherwise falls back to the first
    non-heading paragraph. Used only by the NotebookLM scripts to fill the
    ``summary`` field — never required for core knowledge-library operation.
    """
    if not content:
        return ""
    for heading in ("执行摘要", "摘要", "Summary", "Executive Summary"):
        m = re.search(
            rf"^#+\s*{re.escape(heading)}\s*\n+(.+?)(?=\n#+|\Z)",
            content, flags=re.S | re.M | re.I,
        )
        if m:
            block = m.group(1).strip()
            if block:
                return block[:max_len]
    paras = [
        p.strip() for p in re.split(r"\n\s*\n", content)
        if p.strip() and not p.lstrip().startswith("#")
    ]
    if paras:
        return paras[0][:max_len]
    return content.strip()[:max_len]


# ---------------------------------------------------------------------------
# Async retry (mirror of the sync ``retry`` above)
# ---------------------------------------------------------------------------

def async_retry(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable:
    """Async retry decorator with exponential backoff."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        wait = delay * (backoff ** (attempt - 1))
                        logger.warning(
                            "%s attempt %d/%d failed: %s, retry in %.1fs",
                            getattr(func, "__name__", "coroutine"),
                            attempt, max_attempts, exc, wait,
                        )
                        await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Wiki page collection + candidate discovery (NotebookLM induction inputs)
# ---------------------------------------------------------------------------

_SKIP_PAGE_NAMES = {"index.md", "log.md", "schema.md", "readme.md"}


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return (meta, body) from a Markdown doc, tolerant of missing frontmatter."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    block = text[3:end]
    meta: dict[str, Any] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key] = _split_list(value[1:-1])
        else:
            meta[key] = value.strip("'\"")
    return meta, text[end + 4:]


def _split_list(raw: str) -> list[str]:
    """Split a comma/space separated list, stripping # / [[]] / quotes."""
    raw = raw.replace("[[", "").replace("]]", "")
    return [p.strip("# '\"") for p in re.split(r"[,，\s]+", raw) if p.strip("# '\"")]


def collect_pages(wiki_dir: Path, max_body_length: int = 12000) -> list[dict[str, Any]]:
    """Collect wiki Markdown pages as {title, path, body, mtime, tags, concepts}."""
    pages: list[dict[str, Any]] = []
    if not wiki_dir.exists():
        return pages
    for md in sorted(wiki_dir.rglob("*.md")):
        if md.name.lower() in _SKIP_PAGE_NAMES:
            continue
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        meta, body = _parse_frontmatter(text)
        pages.append({
            "title": meta.get("title") or md.stem,
            "path": str(md),
            "body": body[:max_body_length],
            "mtime": md.stat().st_mtime,
            "tags": meta.get("tags", []),
            "concepts": meta.get("concepts", []),
        })
    return pages


def find_candidates(
    pages: list[dict[str, Any]],
    thresholds: dict[str, int],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Group pages by tag/concept/entity, returning topics that clear thresholds.

    Returns ``(kind, topic, pages)`` tuples. ``kind`` is "tag" / "concept" /
    "entity"; ``entity`` is derived from ``[[wikilink]]`` mentions in the body
    (there is no dedicated frontmatter field). Topics below their threshold are
    dropped, so with no wiki data this is an empty list and the NotebookLM
    induction scripts naturally no-op.
    """
    def _group(attr: str) -> list[tuple[str, list[dict[str, Any]]]]:
        buckets: dict[str, list[dict[str, Any]]] = {}
        for page in pages:
            for value in page.get(attr) or []:
                buckets.setdefault(value, []).append(page)
        return list(buckets.items())

    candidates: list[tuple[str, str, list[dict[str, Any]]]] = []
    for kind in ("tag", "concept"):
        threshold = int(thresholds.get(kind, 0))
        for topic, matched in _group(kind):
            if len(matched) >= threshold:
                candidates.append((kind, topic, matched))

    entity_threshold = int(thresholds.get("entity", 0))
    entity_buckets: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        for ent in re.findall(r"\[\[([^\]]+)\]\]", page.get("body", "")):
            entity_buckets.setdefault(ent, []).append(page)
    for topic, matched in entity_buckets.items():
        if len(matched) >= entity_threshold:
            candidates.append(("entity", topic, matched))
    return candidates
