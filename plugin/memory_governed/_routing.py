# -*- coding: utf-8 -*-
"""Deterministic query-entry routing for hybrid memory recall.

The router is local, cheap, and advisory to the retrieval implementation. It
identifies exact identifiers, file/navigation queries, episodic questions, and
conceptual questions. Callers may use those hints to prefer an entry channel or
skip an expensive rerank when a title/identifier match is already unambiguous.

It never removes a fallback channel by itself and never calls an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Tuple


@dataclass(frozen=True)
class QueryRoute:
    enabled: bool
    mode: str
    reasons: Tuple[str, ...]
    exact_terms: Tuple[str, ...]
    preferred_layers: Tuple[str, ...]
    fast_keyword: bool


_QUOTED_RE = re.compile(r"[“\"']([^”\"']{2,120})[”\"']")
_URL_RE = re.compile(r"https?://[^\s，。！？；]+", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"(?<!\w)[A-Za-z]:[^\s，。！？；]+")
_POSIX_PATH_RE = re.compile(r"(?:^|\s)(?:\./|\.\./|/)[^\s，。！？；]+")
_FLAG_RE = re.compile(r"(?<!\w)--[A-Za-z0-9][A-Za-z0-9_-]*")
_VERSION_RE = re.compile(
    r"(?<![A-Za-z0-9])v?\d+\.\d+(?:\.\d+)*(?:[-+][A-Za-z0-9.-]+)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_ERROR_RE = re.compile(
    r"(?<!\w)(?:0x[0-9a-f]{2,}|E\d{2,}|ERR(?:OR)?[-_]?\d{2,}|HTTP[-_ ]?\d{3})(?!\w)",
    re.IGNORECASE,
)
_FILE_RE = re.compile(
    r"(?<!\w)[^\s/]+\.(?:md|py|json|jsonl|yaml|yml|toml|ini|log|sql|db|sh|ps1|bat|exe|dll|js|ts|tsx|jsx|csv)(?!\w)",
    re.IGNORECASE,
)
_IP_PORT_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?::\d{1,5})?(?!\d)")
_DOTTED_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){1,}\b")
_MIXED_ID_RE = re.compile(
    r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z][A-Za-z0-9_-]{2,}\b"
)

_TEMPORAL = (
    "最近", "上次", "昨天", "今天", "当时", "之前", "以前", "后来", "最新",
    "当前", "现在", "刚刚", "刚才", "最后一次", "哪次", "几天前", "历史",
    "改动", "变更", "最近一次", "last time", "recent", "yesterday", "previous",
)
_MEMORY = (
    "我说过", "你记得", "还记得", "我们决定", "约定", "偏好", "我喜欢", "我习惯",
    "聊过", "对话", "会话", "上次说", "提过", "告诉过你",
)
_NAVIGATION = (
    "知识库", "笔记", "文档", "文件", "哪篇", "标题", "路径", "目录", "索引",
    "分类", "标签", "链接",
)
_CONCEPTUAL = (
    "怎么", "如何", "为什么", "原理", "区别", "差异", "优缺点", "方案", "设计",
    "关系", "是什么", "怎么做", "支持什么", "有哪些", "是否", "能不能",
    "how", "why", "difference", "design", "architecture",
)


def _contains_any(text: str, needles: Iterable[str]) -> list[str]:
    lower = text.lower()
    return [needle for needle in needles if needle.lower() in lower]


def _unique_terms(values: Iterable[str], limit: int = 12) -> Tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = str(value or "").strip().strip("“”\"'，。！？；:：()[]{}<>")
        if not term:
            continue
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= limit:
            break
    return tuple(out)


def _matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return [m.group(0).strip() for m in pattern.finditer(text)]


class QueryRouter:
    """Classify a query into a cheap, explainable entry route."""

    def __init__(self, config: Any = None) -> None:
        self.enabled = bool(getattr(config, "enabled", False))
        self._limit = max(1, int(getattr(config, "max_exact_terms", 12) or 12))

    def route(self, query: str) -> QueryRoute:
        text = re.sub(r"\s+", " ", str(query or "")).strip()
        if not self.enabled or not text:
            return QueryRoute(False, "general", (), (), (), False)

        exact: list[str] = []
        for pattern in (_QUOTED_RE, _URL_RE, _WIN_PATH_RE, _POSIX_PATH_RE,
                        _FLAG_RE, _VERSION_RE, _ERROR_RE, _FILE_RE,
                        _IP_PORT_RE, _DOTTED_RE, _MIXED_ID_RE):
            if pattern is _QUOTED_RE:
                exact.extend(m.group(1) for m in pattern.finditer(text))
            else:
                exact.extend(_matches(pattern, text))
        exact_terms = _unique_terms(exact, self._limit)

        temporal = _contains_any(text, _TEMPORAL)
        memory = _contains_any(text, _MEMORY)
        navigation = _contains_any(text, _NAVIGATION)
        conceptual = _contains_any(text, _CONCEPTUAL)

        reasons: list[str] = []
        if exact_terms:
            reasons.append("exact_identifier")
        if navigation:
            reasons.extend("navigation:" + x for x in navigation[:2])
        if memory:
            reasons.extend("memory:" + x for x in memory[:2])
        if temporal:
            reasons.extend("temporal:" + x for x in temporal[:2])
        if conceptual:
            reasons.extend("conceptual:" + x for x in conceptual[:2])

        if exact_terms:
            mode = "exact"
            preferred = ("kb", "l2", "l3")
        elif navigation:
            mode = "navigational"
            preferred = ("kb", "l2")
        elif memory or temporal:
            mode = "episodic"
            preferred = ("l3", "l2", "kb")
        elif conceptual:
            mode = "conceptual"
            preferred = ("kb", "l2", "l3")
        else:
            mode = "general"
            preferred = ("l2", "l3", "kb")

        return QueryRoute(
            enabled=True,
            mode=mode,
            reasons=tuple(reasons[:8]),
            exact_terms=exact_terms,
            preferred_layers=preferred,
            fast_keyword=mode in ("exact", "navigational"),
        )
