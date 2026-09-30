# -*- coding: utf-8 -*-
"""Diagnostics: the single place where degradation and data loss are reported.

This module exists so that "optional dependency missing" (normal, expected) and
"data was lost / indexes diverged" (always a bug worth shouting about) are never
again mixed into a silent ``logger.debug("...")``.

Two event classes:

``log_degraded()``
    WARNING. Something fell back to a reduced mode (lancedb not installed,
    embedding backend unavailable, FTS schema incomplete...). These fire on
    every turn in a degraded deployment, so they are THROTTLED: the same
    ``(component, reason)`` pair is only emitted once per 60 seconds. The
    suppressed count is visible in :func:`stats`.

``log_data_loss()``
    ERROR. Something was written incompletely, skipped, or diverged (a row
    landed in ``messages`` but not in ``messages_fts``, a LanceDB delete+add
    migration failed halfway...). NEVER throttled, NEVER silent — a deployment
    with an ERROR-level log handler must see these.

Counters are process-global and thread-safe; :func:`stats` returns a
JSON-serializable dict that the ``governed_health`` tool can consume directly.

Note on ``stats()`` keys: it returns exactly ``{"degraded", "data_loss",
"suppressed"}``. Per-turn extraction metrics are kept separate (see
:func:`record_metric` / :func:`metrics`) so that consumers relying on the
three-key contract keep working.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes_governed.diag")

# Same (component, reason) is reported at most once per this many seconds.
THROTTLE_SECONDS: float = 60.0

_lock = threading.RLock()

# (component, reason) -> monotonic timestamp of the last emitted record
_last_emitted: Dict[tuple, float] = {}

# "component::reason" -> how many times the event was recorded
_degraded_counts: Dict[str, int] = {}
_data_loss_counts: Dict[str, int] = {}

# name -> [count, samples]
_metrics: Dict[str, int] = {}

# How many degraded records were dropped by throttling.
_suppressed = 0


def _safe_str(value: Any) -> str:
    """Stringify ``value`` **without ever raising**.

    Exception (and other) objects are UNTRUSTED input: a custom ``__str__``
    may itself raise. Diagnostics run inside ``except`` blocks, so a raise
    here would turn a graceful degradation into a crash — precisely the bug
    this module exists to prevent. Any failure falls back to a safe
    placeholder instead.
    """
    try:
        return str(value)
    except Exception as e:  # noqa: BLE001 — 被格式化的对象 __str__ 本身可能抛
        # silent-ok: value-fallback — _safe_str 的终极兜底，本函数就是"永不抛"的实现
        try:
            return f"<unrepr {type(value).__name__}: {type(e).__name__}>"
        except Exception:  # noqa: BLE001 — type()/__name__ 理论上安全，仍兜底
            # silent-ok: value-fallback — _safe_str 的终极兜底，本函数就是"永不抛"的实现
            return "<unrepr>"


def _format(component: str, reason: str, detail: str, exc: Optional[BaseException]) -> str:
    """Build the log line for an event (English: log text is English).

    Every untrusted value is stringified through :func:`_safe_str`, so a broken
    ``__str__`` on ``exc``/``detail`` can never propagate (see the never-throw
    contract on :func:`log_degraded`).
    """
    message = f"{component}: {reason}"
    if detail:
        message += f" | {_safe_str(detail)}"
    if exc is not None:
        message += f" | {type(exc).__name__}: {_safe_str(exc)}"
    return message


def log_degraded(component: str, reason: str, *, detail: str = "",
                 exc: BaseException | None = None) -> None:
    """Record a graceful-degradation event (WARNING, throttled per 60s).

    **Contract — this function never raises.** Every call site lives inside an
    ``except`` block; if it raised, a graceful degradation would become a
    crash. All formatting of untrusted values (``detail``/``exc``) is guarded
    via :func:`_safe_str`, including a ``__str__`` that itself raises.

    Args:
        component: Subsystem that degraded, e.g. ``"l2_write"``, ``"l3_fts"``.
        reason: Machine-friendly reason code, e.g. ``"lancedb_missing"``.
        detail: Optional human-readable context (paths, ids, counts). Pass
            **static/context** text here, not eagerly-formatted exception text
            (``exc`` already carries the exception; the diagnostic layer
            produces its text safely).
        exc: Optional exception that caused the degradation.
    """
    global _suppressed
    key = (component, reason)
    counter_key = f"{component}::{reason}"

    with _lock:
        _degraded_counts[counter_key] = _degraded_counts.get(counter_key, 0) + 1
        now = time.monotonic()
        last = _last_emitted.get(key)
        if last is not None and (now - last) < THROTTLE_SECONDS:
            _suppressed += 1
            return
        _last_emitted[key] = now

    logger.warning("degraded — %s", _format(component, reason, detail, exc))


def log_data_loss(component: str, reason: str, *, detail: str = "",
                  exc: BaseException | None = None) -> None:
    """Record a data-loss / divergence event (ERROR, never throttled).

    Use for every path where a write is partially applied: main table written
    but index not, row deleted but not re-added, candidate marked imported but
    not landed in L1, etc.

    **Contract — this function never raises** (same as :func:`log_degraded`):
    it is called from ``except`` blocks, and its formatting is guarded via
    :func:`_safe_str`.
    """
    counter_key = f"{component}::{reason}"
    with _lock:
        _data_loss_counts[counter_key] = _data_loss_counts.get(counter_key, 0) + 1

    logger.error("DATA LOSS — %s", _format(component, reason, detail, exc))


def record_metric(name: str, value: int = 1) -> None:
    """Accumulate a numeric metric (e.g. facts extracted per turn).

    Kept out of :func:`stats` so the health-tool contract stays stable; read
    back with :func:`metrics`.
    """
    with _lock:
        _metrics[name] = _metrics.get(name, 0) + int(value)


def metrics() -> Dict[str, int]:
    """Return a copy of the accumulated metrics (JSON-serializable)."""
    with _lock:
        return dict(_metrics)


def stats() -> Dict[str, Any]:
    """Return diagnostic counters for the health tool.

    Returns:
        ``{"degraded": {counter_key: count}, "data_loss": {counter_key: count},
        "suppressed": int}`` — plain JSON-serializable types. Counter keys are
        ``"component::reason"``.
    """
    with _lock:
        return {
            "degraded": dict(_degraded_counts),
            "data_loss": dict(_data_loss_counts),
            "suppressed": _suppressed,
        }


def reset() -> None:
    """Clear all counters and throttle state (tests / explicit re-arm)."""
    global _suppressed
    with _lock:
        _last_emitted.clear()
        _degraded_counts.clear()
        _data_loss_counts.clear()
        _metrics.clear()
        _suppressed = 0
