# -*- coding: utf-8 -*-
"""High-precision quality classification for L2 facts.

Admission gates catch many weak facts, but two classes still slipped through
and had to be cleaned by hand:

* **test residue** — ``自动归档测试`` / ``方案N上线`` written while testing a
  feature. They are long and structured, so the generic gate admits them.
* **assistant fragments** — ``最后追问…`` / ``指出这是…`` lifted from a reply.
  They have no subject and cannot stand alone as a fact.
* **assistant status summaries** — report-style prose such as
  ``本轮三个问题全部修复…``. They are valuable chat history (L3) but not
  durable atomic facts. This rule is context-gated to assistant/auto-extract
  rows so the same wording from an external fact writer is not rejected.

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

#: A report summary can use ordinary fact-like vocabulary, so text alone is a
#: weak signal. It is classified only when the caller supplies assistant or
#: auto-extract provenance. Two independent report markers are required.
_STATUS_REPORT_START_RE = re.compile(
    r"^(?:第?[一二三四五六七八九十\d]+轮)?(?:复评|评估|审计|检查|评测)完成"
)
_STATUS_REPORT_SIGNALS = (
    "全部修复", "均为根因修复", "本轮", "测试", "全绿", "通过", "黄金集",
    "failure_streak", "reports/", "passed", "failed", "P0", "P1",
)

#: Markers of a fact that records an *open* problem. Reported, never auto-deleted:
#: "X 尚未修复" can still be a true statement about the world.
_STALE_MARKERS = ("尚未修复", "已确认随 origin/main")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _looks_like_assistant_status_report(cleaned: str, role: str,
                                        source: str) -> bool:
    if role != "assistant" and source != "auto-extract":
        return False
    if _STATUS_REPORT_START_RE.search(cleaned):
        return True
    if not cleaned.startswith(("本轮", "本轮工作", "复评", "评估", "审计")):
        return False
    return sum(1 for marker in _STATUS_REPORT_SIGNALS if marker in cleaned) >= 2


def classify_fact(text: str, *, role: str = "",
                  source: str = "") -> Optional[Dict[str, Any]]:
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
    if _looks_like_assistant_status_report(cleaned, role, source):
        return {"kind": "assistant_fragment",
                "reason": "assistant status summary, not an atomic fact"}
    for prefix in _FRAGMENT_PREFIXES:
        if cleaned.startswith(prefix):
            return {"kind": "assistant_fragment",
                    "reason": "assistant meta fragment without a subject"}
    for marker in _STALE_MARKERS:
        if marker in cleaned:
            return {"kind": "stale_marker",
                    "reason": "records an open problem; needs review"}
    return None


def admission_rejection(text: str, *, role: str = "",
                        source: str = "") -> Optional[str]:
    """Reason tag for the write gate, or ``None`` when the fact may be admitted."""
    verdict = classify_fact(text, role=role, source=source)
    if verdict and verdict["kind"] in ("test_residue", "assistant_fragment"):
        return verdict["kind"]
    return None
