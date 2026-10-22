# -*- coding: utf-8 -*-
"""Optional fast-decision layer for governed memory.

The layer is intentionally small and replaceable. It never owns persistence and
never becomes a required part of the read/write path. Every caller must be able
to continue with the existing deterministic rules when this module is disabled,
unavailable, or returns an uncertain answer.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol

from ._diag import log_degraded

logger = logging.getLogger(__name__)


@dataclass
class DecisionResult:
    """A side-effect-free decision returned by a policy."""

    operation: str
    source: str = "fallback"
    chosen: str = "uncertain"
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    action: str = "review"
    reason_codes: list[str] = field(default_factory=list)
    latency_ms: int = 0
    fallback: bool = False
    error: str = ""
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)

    def public_dict(self) -> Dict[str, Any]:
        """Return JSON-safe metadata; final memory content is never included."""
        return {
            "operation": self.operation,
            "source": self.source,
            "chosen": self.chosen,
            "probabilities": {
                str(k): round(float(v), 6)
                for k, v in self.probabilities.items()
            },
            "confidence": round(float(self.confidence), 6),
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "latency_ms": int(self.latency_ms),
            "fallback": bool(self.fallback),
            "error": str(self.error or "")[:160],
            "model": str(self.model or "")[:120],
        }


class DecisionPolicy(Protocol):
    """Minimal interface implemented by Jev and deterministic test doubles."""

    available: bool

    def evaluate(
        self,
        operation: str,
        state: Dict[str, Any],
    ) -> Optional[DecisionResult]: ...


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False,
                     separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest()


class DecisionAuditLog:
    """Append-only JSONL audit with no raw memory text by default."""

    def __init__(self, path: str | os.PathLike[str] | None):
        self.path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve() \
            if path else None
        self._lock = threading.Lock()

    def emit(self, record: Dict[str, Any]) -> None:
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        line = json.dumps(row, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), default=str)
        logger.info("decision %s", line)
        if self.path is None:
            return
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(line + "\n")
        except Exception as e:  # noqa: BLE001 - audit failure must not break callers
            log_degraded("decision", "audit_write_failed",
                         detail=str(self.path), exc=e)


_EXECUTORS: list[ThreadPoolExecutor] = []
_EXECUTOR_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _shutdown_executors() -> None:
    with _EXECUTOR_LOCK:
        executors = list(_EXECUTORS)
        _EXECUTORS.clear()
    for executor in executors:
        try:
            executor.shutdown(wait=False, cancel_futures=True)
        except Exception as e:  # noqa: BLE001 - interpreter-exit cleanup is best effort
            log_degraded("decision", "executor_shutdown_failed", exc=e)


def _new_executor() -> ThreadPoolExecutor:
    global _ATEXIT_REGISTERED
    executor = ThreadPoolExecutor(max_workers=2,
                                  thread_name_prefix="hgm-decision")
    with _EXECUTOR_LOCK:
        _EXECUTORS.append(executor)
        if not _ATEXIT_REGISTERED:
            atexit.register(_shutdown_executors)
            _ATEXIT_REGISTERED = True
    return executor


class DecisionEngine:
    """Coordinate one policy, bounded caching, shadow execution, and audit."""

    def __init__(
        self,
        config: Any,
        *,
        audit_path: str | os.PathLike[str] | None = None,
        policy: Optional[DecisionPolicy] = None,
    ) -> None:
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))
        self.mode = str(getattr(config, "mode", "shadow") or "shadow")
        self.audit = DecisionAuditLog(
            audit_path or getattr(config, "audit_path", "") or None
        )
        self.policy = policy
        self._policy_error = ""
        if self.enabled and self.policy is None:
            try:
                from ._jev_policy import JevDecisionPolicy
                self.policy = JevDecisionPolicy(config)
            except Exception as e:  # noqa: BLE001 - optional dependency
                self._policy_error = type(e).__name__
                log_degraded("decision", "policy_init_failed", exc=e)
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: set[Future] = set()
        self._futures_lock = threading.Lock()
        self._cache: "OrderedDict[str, DecisionResult]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._cache_cap = max(0, int(getattr(config, "cache_size", 0) or 0))

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.policy is not None
                    and getattr(self.policy, "available", False))

    @property
    def policy_error(self) -> str:
        return self._policy_error

    def observe_kb_novelty(
        self,
        new_note: Dict[str, Any],
        candidate_provider: Callable[[], list[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate one KB write using bounded, sanitized note summaries.

        Shadow mode queues the work and returns immediately. Active mode is
        synchronous so a caller can surface an advisory review flag. Neither
        mode rejects the note in this version.
        """
        if not self.enabled or not bool(getattr(self.config,
                                                "kb_novelty_enabled", True)):
            return None
        state = self._kb_state(new_note)
        operation = "kb_novelty"
        if self.mode == "shadow" and bool(
                getattr(self.config, "background_shadow", True)):
            self._submit(self._run_kb, operation, state, candidate_provider)
            return {"status": "queued", "mode": "shadow",
                    "operation": operation}
        return self._run_kb(operation, state, candidate_provider).public_dict()

    def _kb_state(self, new_note: Dict[str, Any]) -> Dict[str, Any]:
        limit = max(200, int(getattr(self.config, "max_content_chars", 1600)))
        return {
            "new_note": {
                "title": str(new_note.get("title") or "")[:300],
                "body": str(new_note.get("body") or "")[:limit],
                "tags": [str(x)[:80] for x in (new_note.get("tags") or [])[:20]],
                "concepts": [str(x)[:80]
                             for x in (new_note.get("concepts") or [])[:20]],
                "section": str(new_note.get("section") or "")[:80],
            }
        }

    @staticmethod
    def _candidate_state(candidates: list[Dict[str, Any]],
                         limit: int) -> list[Dict[str, Any]]:
        out: list[Dict[str, Any]] = []
        for item in candidates[:max(1, limit)]:
            out.append({
                "title": str(item.get("title") or "")[:300],
                "snippet": str(item.get("snippet") or item.get("body") or "")[:800],
                "tags": [str(x)[:80] for x in (item.get("tags") or [])[:20]],
                "concepts": [str(x)[:80]
                             for x in (item.get("concepts") or [])[:20]],
                "path": str(item.get("path") or "")[:500],
                "score": round(float(item.get("score", 0.0) or 0.0), 6),
            })
        return out

    def _run_kb(
        self,
        operation: str,
        base_state: Dict[str, Any],
        candidate_provider: Callable[[], list[Dict[str, Any]]],
    ) -> DecisionResult:
        started = time.monotonic()
        candidates: list[Dict[str, Any]] = []
        if self.available:
            try:
                candidates = candidate_provider()
                if not isinstance(candidates, list):
                    candidates = []
            except Exception as e:  # noqa: BLE001 - candidate lookup is best effort
                log_degraded("decision", "candidate_lookup_failed", exc=e)
                candidates = []
        limit = max(1, int(getattr(self.config, "max_candidates", 5) or 5))
        state = {
            **base_state,
            "candidates": self._candidate_state(candidates, limit),
        }
        result = self._evaluate_cached(operation, state)
        result.latency_ms = max(0, int((time.monotonic() - started) * 1000))
        result = self._apply_action_policy(operation, result)
        self.audit.emit({
            "event": "decision",
            "operation": operation,
            "mode": self.mode,
            "source_repo": "hermes-memory-governed",
            "policy_version": "jev-decision-v1",
            "input_hash": _canonical_hash(state),
            "candidate_count": len(state["candidates"]),
            **result.public_dict(),
        })
        return result

    def _evaluate_cached(self, operation: str,
                         state: Dict[str, Any]) -> DecisionResult:
        if not self.enabled or self.policy is None:
            reason = self._policy_error or "disabled_or_unavailable"
            return DecisionResult(operation=operation, error=reason,
                                  fallback=True)
        key = _canonical_hash({"operation": operation, "state": state,
                               "model": getattr(self.config, "model", "")})
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return DecisionResult(**vars(cached))
        try:
            result = self.policy.evaluate(operation, state)
        except Exception as e:  # noqa: BLE001 - policy failures are advisory
            log_degraded("decision", "policy_evaluate_failed", exc=e)
            return DecisionResult(operation=operation,
                                  error=type(e).__name__, fallback=True)
        if result is None:
            return DecisionResult(
                operation=operation,
                error=getattr(self.policy, "last_error", "")
                or "policy_unavailable",
                fallback=True,
            )
        if not isinstance(result, DecisionResult):
            return DecisionResult(operation=operation,
                                  error="invalid_policy_result",
                                  fallback=True)
        if self._cache_cap:
            with self._cache_lock:
                self._cache[key] = DecisionResult(**vars(result))
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_cap:
                    self._cache.popitem(last=False)
        return result

    def _apply_action_policy(self, operation: str,
                             result: DecisionResult) -> DecisionResult:
        review_floor = float(getattr(self.config,
                                     "review_confidence", 0.65) or 0.0)
        warn_floor = float(getattr(self.config,
                                   "warn_confidence", 0.85) or 0.0)
        if result.fallback:
            result.action = "review"
            return result
        if result.confidence < review_floor:
            result.action = "review"
            return result
        if operation == "kb_novelty":
            if result.chosen == "duplicate" and result.confidence >= warn_floor:
                result.action = "warn"
            elif result.chosen in ("conflicts", "uncertain"):
                result.action = "review"
            else:
                result.action = "allow"
        return result

    def _submit(
        self,
        fn: Callable[..., DecisionResult],
        operation: str,
        state: Dict[str, Any],
        candidate_provider: Callable[[], list[Dict[str, Any]]],
    ) -> None:
        if self._executor is None:
            self._executor = _new_executor()
        future = self._executor.submit(fn, operation, state, candidate_provider)
        with self._futures_lock:
            self._futures.add(future)

        def _done(completed: Future,
                  *, _set: set[Future] = self._futures,
                  _lock: threading.Lock = self._futures_lock) -> None:
            with _lock:
                _set.discard(completed)
            try:
                completed.result()
            except Exception as e:  # noqa: BLE001 - background must not raise
                log_degraded("decision", "background_failed", exc=e)

        future.add_done_callback(_done)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for queued shadow decisions. Returns False on timeout."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._futures_lock:
                pending = list(self._futures)
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            finished = [future for future in pending if future.done()]
            if not finished:
                time.sleep(min(0.05, remaining))
                continue
            for future in finished:
                try:
                    future.result()
                except Exception as e:  # noqa: BLE001 - background failure was audited
                    log_degraded("decision", "background_result_failed", exc=e)
                with self._futures_lock:
                    self._futures.discard(future)

    def shutdown(self, wait: bool = False) -> None:
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        executor.shutdown(wait=wait, cancel_futures=not wait)
