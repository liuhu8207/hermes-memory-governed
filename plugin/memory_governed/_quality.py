# -*- coding: utf-8 -*-
"""High-precision quality classification for L2 facts.

Admission gates catch many weak facts, but two classes still slipped through
and had to be cleaned by hand:

* **test residue** — ``自动归档测试`` / ``方案N上线`` written while testing a
  feature. They are long and structured, so the generic gate admits them.
* **assistant fragments** — ``最后追问…`` / ``指出这是…`` lifted from a reply.
  They have no subject and cannot stand alone as a fact.

This module is deliberately **high precision**: only patterns that are
unambiguously not durable knowledge are rejected automatically. Stale or
duplicate facts are *reported* by the sweep, not silently deleted.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

#: Unambiguous test residue.
_TEST_PATTERNS = (
    re.compile(r"^自动归档测试"),
    re.compile(r"^方案\d+.*测试"),
    re.compile(r"^方案\d+上线"),
    re.compile(r"自动归档测试[：:]agent "),
)

#: Assistant-side meta fragments (no subject, not a standalone fact).
_FRAGMENT_PREFIXES = (
    "最后追问",
    "指出这是",
    "同时提醒",
    "补齐实操",
    "GUI 流程",
    "文件系统必须",
    "常见问题排查",
    "四步流程",
    "6) 常见问题",
    "4) 四步流程",
)

#: Markers of a fact that records an *open* problem. Reported, never auto-deleted:
#: "X 尚未修复" can still be a true statement about the world.
_STALE_MARKERS = ("尚未修复", "已确认随 origin/main")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def classify_fact(text: str) -> Optional[Dict[str, Any]]:
    """Return ``{kind, reason}`` for a non-durable fact, else ``None``.

    Kinds:
      * ``test_residue`` — safe to auto-retract;
      * ``assistant_fragment`` — safe to reject at admission / auto-retract;
      * ``stale_marker`` — report only.
    """
    cleaned = _clean(text)
    if not cleaned:
        return {"kind": "empty", "reason": "empty fact"}
    for pattern in _TEST_PATTERNS:
        if pattern.search(cleaned):
            return {"kind": "test_residue",
                    "reason": "test residue, not durable knowledge"}
    for prefix in _FRAGMENT_PREFIXES:
        if cleaned.startswith(prefix):
            return {"kind": "assistant_fragment",
                    "reason": "assistant meta fragment without a subject"}
    for marker in _STALE_MARKERS:
        if marker in cleaned:
            return {"kind": "stale_marker",
                    "reason": "records an open problem; needs review"}
    return None


def admission_rejection(text: str) -> Optional[str]:
    """Reason tag for the write gate, or ``None`` when the fact may be admitted."""
    verdict = classify_fact(text)
    if verdict and verdict["kind"] in ("test_residue", "assistant_fragment"):
        return verdict["kind"]
    return None
