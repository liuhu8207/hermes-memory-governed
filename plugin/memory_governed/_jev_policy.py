# -*- coding: utf-8 -*-
"""Thin, optional adapter for TypeSafe's Jev System One API.

Only the official SDK is imported, lazily and behind the decision feature flag.
No Jev-Mem graph/database package is required or imported here.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from ._config import env_secret
from ._decision import DecisionResult
from ._diag import log_degraded

logger = logging.getLogger(__name__)


_KB_CRITERIA = {
    "duplicate": (
        "The candidate already contains every material fact in new_note. "
        "A translation or paraphrase still counts as duplicate; wording does not."
    ),
    "extends": (
        "The candidate covers the same canonical topic, but new_note adds a "
        "specific new detail that should remain discoverable."
    ),
    "conflicts": (
        "The candidate and new_note make incompatible claims about the same "
        "thing, or both cannot be true without a temporal explanation."
    ),
    "new": (
        "No candidate already represents the main fact in new_note; it is a "
        "genuinely new topic or entity."
    ),
    "uncertain": (
        "The supplied evidence is insufficient for a safe duplicate, extension, "
        "or conflict decision."
    ),
}


class JevDecisionPolicy:
    """Call Jev with bounded typed questions and no retries in the hot path."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.last_error = ""
        self._client = None
        self._client_lock = threading.Lock()
        self._sdk = None
        self._sdk_error = ""
        self._api_key = ""
        try:
            import typesafe_sdk as sdk  # type: ignore[import-not-found]
            self._sdk = sdk
        except Exception as e:  # noqa: BLE001 - optional dependency
            self._sdk_error = type(e).__name__
            log_degraded("decision", "sdk_unavailable", exc=e)
            return
        key_env = str(getattr(config, "api_key_env", "") or "").strip()
        self._api_key = env_secret(key_env) if key_env else ""
        if not self._api_key:
            self._sdk_error = "missing_api_key"

    @property
    def available(self) -> bool:
        return bool(self._sdk is not None and self._api_key
                    and str(getattr(self.config, "model", "") or "").strip())

    def _get_client(self):
        if not self.available:
            return None
        with self._client_lock:
            if self._client is None:
                kwargs: Dict[str, Any] = {
                    "api_key": self._api_key,
                    "model": str(getattr(self.config, "model", "") or ""),
                }
                base_url = str(getattr(self.config, "base_url", "") or "").strip()
                if base_url:
                    kwargs["base_url"] = base_url
                retry_cls = getattr(self._sdk, "RetryPolicy", None)
                if retry_cls is not None:
                    try:
                        kwargs["retry"] = retry_cls(max_retries=0, timeout=None)
                    except TypeError as e:
                        log_degraded("decision", "retry_policy_unsupported", exc=e)
                self._client = self._sdk.TypeSafeClient(**kwargs)
            return self._client

    def _question(self, operation: str):
        if operation != "kb_novelty":
            raise ValueError("unsupported decision operation: " + operation)
        return self._sdk.Choice(
            instructions=(
                "Compare `state.new_note` with each item in `state.candidates`. "
                "Choose exactly one relation. Shared topic alone is not duplicate; "
                "new details beat duplicate; contradictory facts beat extends."
            ),
            criteria=dict(_KB_CRITERIA),
        )

    def evaluate(
        self,
        operation: str,
        state: Dict[str, Any],
    ) -> Optional[DecisionResult]:
        client = self._get_client()
        if client is None:
            self.last_error = self._sdk_error or "unavailable"
            return None
        try:
            questions = {"relation": self._question(operation)}
            response = client.system_one(
                state=state,
                questions=questions,
                timeout=max(0.2, float(getattr(self.config,
                                               "timeout_seconds", 1.5) or 1.5)),
            )
            raw = response.model_dump() if hasattr(response, "model_dump") \
                else response
            item = (raw.get("answers") or {}).get("relation") or {}
            if item.get("type") != "choice":
                raise ValueError("unexpected Jev answer type")
            chosen = str(item.get("choice") or "")
            if chosen not in _KB_CRITERIA:
                raise ValueError("unexpected Jev choice")
            probabilities = {
                str(k): float(v) for k, v in (item.get("probabilities") or {}).items()
            }
            if not probabilities or any(not 0.0 <= v <= 1.0
                                         for v in probabilities.values()):
                raise ValueError("invalid Jev probabilities")
            if set(probabilities) != set(_KB_CRITERIA):
                raise ValueError("Jev probabilities do not cover the requested choices")
            if abs(sum(probabilities.values()) - 1.0) > 1e-3:
                raise ValueError("Jev probabilities do not sum to one")
            if probabilities[chosen] + 1e-6 < max(probabilities.values()):
                raise ValueError("Jev selected choice is not the highest probability")
            confidence = float(item.get("confidence", probabilities.get(chosen, 0.0)))
            return DecisionResult(
                operation=operation,
                source="jev",
                chosen=chosen,
                probabilities=probabilities,
                confidence=max(0.0, min(1.0, confidence)),
                model=str(raw.get("model") or getattr(self.config, "model", "")),
                usage=dict(raw.get("usage") or {}),
            )
        except Exception as e:  # noqa: BLE001 - advisory failure falls through
            self.last_error = type(e).__name__
            log_degraded("decision", "jev_call_failed", exc=e)
            return None
