"""Bridge: reviewed durable-memory candidates for Scope Recall.

This module prepares, validates, and exports reviewed memory candidates
that can be imported into Scope Recall or other runtime providers.

The bridge is intentionally export-first and review-gated:
- Auto-extracted facts with high confidence → direct L2 write
- Medium confidence → Bridge candidate (JSONL, reviewable)
- Low confidence → discarded with log
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._config import GovernedMemoryConfig
from ._diag import log_data_loss

logger = logging.getLogger(__name__)

# Quarantine file for lines that cannot be parsed (never silently discarded).
CORRUPT_FILENAME = "candidates.corrupt.jsonl"
# Quarantine file for candidates that carry secret-like content (fail closed).
QUARANTINE_FILENAME = "quarantine.jsonl"

# Only export to these Scope Recall targets
SAFE_TARGETS = {"user", "memory", "project", "ops"}

# Secret patterns to redact / quarantine. Keep this list HIGH PRECISION:
# under fail-closed every false positive silently drops a legitimate memory,
# so bare high-entropy strings (base64, JWTs, uuids) are deliberately NOT
# matched — a future rule must require surrounding context keywords.
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{15,}"),
    # key=value：值必须是"像密钥"的裸 token（字母数字开头、>=16 位，只含
    # 字母数字/下划线/连字符）。排除代码表达式（含括号/引号）、属性访问
    # （settings.x 含点）、占位符（<>）、中文描述——这些是误报主源。
    re.compile(r"(?i)(api[_-]?key|app[_-]?secret|token|password|passwd|pwd|secret)\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9_\-]{15,})"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{20,}"),
]

# Human-readable names, index-aligned with SECRET_PATTERNS.
SECRET_PATTERN_NAMES = (
    "openai_style_key",
    "github_token",
    "key_value_secret",
    "pem_private_key",
    "bearer_token",
)


def redact_sensitive(text: str) -> str:
    """Redact API keys and secrets from text.

    Used for logs, the quarantine preview and ``last_export.json`` display —
    never to smuggle secret-bearing content into the review feed.
    """
    redacted = text or ""
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(
            lambda m: f"{m.group(1)}=[REDACTED]" if m.lastindex == 2 else "[REDACTED]",
            redacted,
        )
    return redacted


def has_secret_like_text(text: str) -> bool:
    """Check if text contains secret-like patterns."""
    return redact_sensitive(text) != (text or "")


def matched_secret_patterns(text: str) -> List[str]:
    """Names of the secret patterns that match ``text`` (empty list = clean)."""
    if not text:
        return []
    names: List[str] = []
    for index, pattern in enumerate(SECRET_PATTERNS):
        if pattern.search(text):
            if index < len(SECRET_PATTERN_NAMES):
                names.append(SECRET_PATTERN_NAMES[index])
            else:
                names.append(f"pattern_{index}")
    return names


def stable_id(prefix: str, content: str, source_path: str) -> str:
    """Generate a stable ID for a bridge candidate."""
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(source_path.encode("utf-8", errors="replace"))
    digest.update(content.encode("utf-8", errors="replace"))
    return f"{prefix}-{digest.hexdigest()[:24]}"


class BridgeExporter:
    """Export reviewed durable-memory candidates as JSONL.

    Candidates are tagged with 'review-required' and can be reviewed
    before importing into Scope Recall.

    Secrets are FAIL CLOSED: a candidate whose content matches a secret
    pattern never reaches candidates.jsonl (and therefore never reaches L1).
    It is appended to quarantine.jsonl (redacted preview + source + pattern
    names + timestamp) so an auditor can see that a secret existed without
    the secret itself being stored. ``config.allow_secret_candidates = True``
    is the explicit escape hatch: such candidates are stored REDACTED and
    flagged — the L1 hard gate in :meth:`import_approved` still blocks them.

    Archives: when candidates.jsonl exceeds max_jsonl_lines (default 500),
    the active file is rotated to archive/candidates_YYYYMMDD_HHMMSS.jsonl
    and a fresh candidates.jsonl starts.
    """

    # Maximum lines in candidates.jsonl before auto-archiving
    MAX_JSONL_LINES = 500

    def __init__(self, config: GovernedMemoryConfig):
        self._config = config
        self._bridge_dir = Path(config.bridge_dir)
        self._bridge_dir.mkdir(parents=True, exist_ok=True)
        # Serialises "read existing ids -> validate -> append" so concurrent
        # exporters cannot interleave and duplicate entries.
        self._export_lock = threading.Lock()

    def export_candidates(self, candidates: List[Dict[str, Any]]) -> dict:
        """Append candidates to JSONL (deduplicating by id) and return export stats.

        Secret-bearing candidates are quarantined (see class docstring) and
        reported under ``quarantined``; ``written`` counts only clean rows.
        """
        jsonl_path = self._bridge_dir / "candidates.jsonl"
        report_path = self._bridge_dir / "last_export.json"

        written = 0
        skipped = 0
        quarantined = 0
        errors = 0
        quarantine_preview: Optional[str] = None

        with self._export_lock:
            # Load existing candidate IDs to avoid duplicates
            existing_ids: set = set()
            if jsonl_path.exists():
                try:
                    with open(jsonl_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                existing = json.loads(line)
                                cid = existing.get("id")
                                if cid:
                                    existing_ids.add(cid)
                            except json.JSONDecodeError:
                                pass
                except Exception:  # noqa: BLE001 - dedup is best-effort
                    pass

            # Append mode inside the lock: concurrent callers serialise here.
            with open(jsonl_path, "a", encoding="utf-8") as f:
                for candidate in candidates:
                    # Validate
                    if not self._validate_candidate(candidate):
                        skipped += 1
                        continue

                    # Fail closed on secrets: quarantine, never store.
                    patterns = matched_secret_patterns(candidate.get("content", ""))
                    if patterns:
                        if self._allow_secret_candidates():
                            # Explicit escape hatch: store REDACTED + flagged.
                            candidate["content"] = redact_sensitive(
                                candidate.get("content", "")
                            )
                            candidate["secret_redacted"] = True
                            logger.warning(
                                "Storing REDACTED secret-like candidate "
                                "(allow_secret_candidates=true): %s",
                                patterns,
                            )
                        else:
                            self._quarantine_candidate(candidate, patterns)
                            quarantined += 1
                            quarantine_preview = redact_sensitive(
                                str(candidate.get("content", ""))
                            )[:120]
                            log_data_loss(
                                "bridge_secret",
                                "candidate_quarantined",
                                detail=(
                                    f"id={candidate.get('id', '?')} "
                                    f"source={candidate.get('source_path', '?')} "
                                    f"patterns={patterns}"
                                ),
                            )
                            continue

                    # Ensure required fields
                    candidate.setdefault("id", stable_id(
                        "hermes",
                        candidate.get("content", ""),
                        candidate.get("source_path", ""),
                    ))
                    candidate.setdefault("target", "memory")
                    candidate.setdefault("memory_type", "memory")
                    candidate.setdefault("tags", ["review-required"])
                    candidate.setdefault("source", "hermes-memory-governed")
                    candidate.setdefault("updated_at", datetime.now().isoformat())
                    candidate.setdefault("metadata", {})

                    # Deduplicate by id
                    if candidate["id"] in existing_ids:
                        skipped += 1
                        continue

                    existing_ids.add(candidate["id"])
                    f.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    written += 1

            # Write export report
            report = {
                "timestamp": datetime.now().isoformat(),
                "candidates_total": len(candidates),
                "written": written,
                "skipped": skipped,
                "quarantined": quarantined,
                "quarantine_preview": quarantine_preview,
                "errors": errors,
                "jsonl_path": str(jsonl_path),
            }
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info("Bridge export: %d written, %d skipped, %d quarantined",
                        written, skipped, quarantined)

            # Auto-archive if the file is getting too large
            self._archive_if_needed(jsonl_path)

        return report

    def _allow_secret_candidates(self) -> bool:
        """Escape hatch from fail-closed (default OFF)."""
        return bool(getattr(self._config, "allow_secret_candidates", False))

    def _quarantine_candidate(self, candidate: dict, patterns: List[str]) -> Optional[Path]:
        """Append a secret-bearing candidate to quarantine.jsonl (redacted)."""
        path = self._bridge_dir / QUARANTINE_FILENAME
        entry = {
            "id": candidate.get("id") or stable_id(
                "hermes",
                str(candidate.get("content", "")),
                str(candidate.get("source_path", "")),
            ),
            # The secret itself is masked; the SHAPE of what was said survives.
            "content": redact_sensitive(str(candidate.get("content", ""))),
            "source": candidate.get("source") or candidate.get("source_path") or "unknown",
            "target": candidate.get("target", "memory"),
            "patterns": patterns,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.warning(
                "Bridge quarantined secret-like candidate (patterns=%s): %s",
                patterns, entry["id"],
            )
            return path
        except OSError as e:
            log_data_loss(
                "bridge_secret",
                "quarantine_write_failed",
                detail=f"content dropped entirely: {entry['id']} patterns={patterns}",
                exc=e,
            )
            return None

    def _archive_if_needed(self, jsonl_path: Path) -> None:
        """Rotate candidates.jsonl to archive/ if it exceeds the line limit.

        The archived file is named candidates_YYYYMMDD_HHMMSS.jsonl and a
        fresh candidates.jsonl is left in place for subsequent writes.
        """
        if not jsonl_path.exists():
            return
        try:
            line_count = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
        except Exception:
            return
        if line_count <= self.MAX_JSONL_LINES:
            return

        archive_dir = self._bridge_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = archive_dir / f"candidates_{ts}.jsonl"

        try:
            jsonl_path.rename(archive_path)
            logger.info("Bridge archived %d lines to %s", line_count, archive_path)
        except OSError as e:
            logger.debug("Bridge archive failed: %s", e)

    def _validate_candidate(self, candidate: dict) -> bool:
        """Validate a bridge candidate."""
        if not candidate.get("content"):
            return False
        target = candidate.get("target", "memory")
        if target not in SAFE_TARGETS:
            logger.debug("Unsafe target: %s", target)
            return False
        if len(candidate["content"]) < 10:
            return False
        return True

    def get_status(self) -> dict:
        """Get bridge status."""
        jsonl_path = self._bridge_dir / "candidates.jsonl"
        report_path = self._bridge_dir / "last_export.json"

        status = {
            "bridge_dir": str(self._bridge_dir),
            "jsonl_exists": jsonl_path.exists(),
            "jsonl_size": jsonl_path.stat().st_size if jsonl_path.exists() else 0,
            "candidate_count": 0,
            "quarantine_count": self._count_lines(self._bridge_dir / QUARANTINE_FILENAME),
            "corrupt_count": self._count_lines(self._bridge_dir / CORRUPT_FILENAME),
            "last_export": None,
        }

        if jsonl_path.exists():
            try:
                with open(jsonl_path, encoding="utf-8") as f:
                    status["candidate_count"] = sum(1 for _ in f)
            except Exception:
                pass

        if report_path.exists():
            try:
                status["last_export"] = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        return status

    def import_approved(self, l1_path: str | Path, auto_approve: bool = False) -> dict:
        """Land reviewed candidates into L1 (MEMORY.md) — closes the review loop.

        Flow: candidates.jsonl → filter by review state → append to L1 with a
        bridge provenance comment → mark rows as imported in the JSONL.

        Review gate:
        - auto_approve=True: import everything (including review-required)
        - auto_approve=False: only import candidates explicitly tagged 'approved'
          (or without 'review-required').

        SECRET HARD GATE (the real trust boundary): any row that carries a
        secret — flagged ``secret_redacted``, tagged ``needs-careful-review``,
        or whose text still matches a secret pattern — is blocked
        UNCONDITIONALLY. ``auto_approve=True`` does NOT bypass it, and neither
        does an 'approved' tag. This is deliberately independent of the export
        check so that entry-point regressions and historical rows already in
        the feed can never reach L1.

        Already-imported rows (imported=true) are skipped.

        Ordering (crash-safety): rows are marked imported and written back
        ATOMICALLY (temp file + os.replace) BEFORE anything is appended to L1.
        A failure during the L1 append can therefore only lose an import
        (reported as data loss), never duplicate one.

        Unparsable lines are never deleted: they are kept verbatim in
        candidates.jsonl and copied to candidates.corrupt.jsonl for triage.
        """
        jsonl_path = self._bridge_dir / "candidates.jsonl"
        if not jsonl_path.exists():
            return {"imported": 0, "skipped": 0, "corrupt_lines": 0,
                    "blocked_secrets": 0, "reason": "no candidates file"}

        l1_file = Path(l1_path)
        imported = 0
        skipped = 0
        blocked_secrets = 0
        corrupt_lines: List[str] = []

        # (raw_line, parsed_row or None) — the raw line is kept so corrupt
        # entries survive a rewrite.
        rows: List[Tuple[str, Optional[dict]]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n")
                if not raw.strip():
                    continue
                try:
                    rows.append((raw, json.loads(raw)))
                except json.JSONDecodeError:
                    rows.append((raw, None))
                    corrupt_lines.append(raw)
                    skipped += 1

        to_import: list[dict] = []
        for _raw, row in rows:
            if row is None:
                continue

            # Hard gate: runs BEFORE the review gate and ignores auto_approve.
            if self._row_has_secret(row):
                blocked_secrets += 1
                log_data_loss(
                    "bridge_secret",
                    "row_blocked_at_l1_gate",
                    detail=(
                        f"id={row.get('id', '?')} "
                        f"source={row.get('source_path') or row.get('source', '?')} "
                        "— a secret reached the review feed; it must not enter L1"
                    ),
                )
                continue

            if row.get("imported"):
                skipped += 1
                continue
            tags = row.get("tags", [])
            if not auto_approve:
                if "review-required" in tags or "approved" not in tags:
                    skipped += 1
                    continue
            to_import.append(row)

        if corrupt_lines:
            self._quarantine_corrupt(corrupt_lines)

        if not to_import:
            return {"imported": 0, "skipped": skipped, "corrupt_lines": len(corrupt_lines),
                    "blocked_secrets": blocked_secrets}

        # 1) Mark imported + atomic rewrite FIRST (no duplicate imports).
        if not self._mark_imported(jsonl_path, rows, {r.get("id") for r in to_import}):
            return {"imported": 0, "skipped": skipped,
                    "corrupt_lines": len(corrupt_lines),
                    "blocked_secrets": blocked_secrets, "errors": 1}

        # 2) Then land them in L1 (a failure here only loses an import).
        try:
            l1_file.parent.mkdir(parents=True, exist_ok=True)
            with open(l1_file, "a", encoding="utf-8") as f:
                for row in to_import:
                    content = str(row.get("content", "")).strip()
                    if not content:
                        continue
                    source = row.get("source", "bridge")
                    f.write(f"\n\n<!-- bridge:imported source={source} date={datetime.now().date()} -->\n")
                    f.write(f"- {content}\n")
                    imported += 1
        except Exception as e:  # noqa: BLE001 - must be reported, not swallowed
            not_landed = [r.get("id") for r in to_import[imported:]]
            log_data_loss(
                "bridge",
                "l1_append_failed",
                detail=f"{imported}/{len(to_import)} landed; not_landed={not_landed}",
                exc=e,
            )
            skipped += len(to_import) - imported

        logger.info(
            "Bridge import_approved: %d imported, %d skipped, %d corrupt, %d secret-blocked",
            imported, skipped, len(corrupt_lines), blocked_secrets,
        )
        return {"imported": imported, "skipped": skipped,
                "corrupt_lines": len(corrupt_lines),
                "blocked_secrets": blocked_secrets}

    @staticmethod
    def _row_has_secret(row: dict) -> bool:
        """True when a candidate row must never reach L1.

        Three independent triggers, so the gate holds even if the export-side
        detection is later weakened or a row predates it:
        1. explicit ``secret_redacted`` flag;
        2. ``needs-careful-review`` tag (legacy redact-and-store builds);
        3. the content STILL matches a secret pattern (historical rows).
        """
        if row.get("secret_redacted"):
            return True
        if "needs-careful-review" in (row.get("tags") or []):
            return True
        return has_secret_like_text(str(row.get("content", "")))

    @staticmethod
    def _count_lines(path: Path) -> int:
        """Count non-empty lines of ``path`` (0 when it does not exist)."""
        if not path.exists():
            return 0
        try:
            return sum(1 for line in path.open(encoding="utf-8") if line.strip())
        except OSError:
            return 0

    def _quarantine_corrupt(self, corrupt_lines: List[str]) -> Optional[Path]:
        """Append unparsable lines to candidates.corrupt.jsonl (triage copy)."""
        path = self._bridge_dir / CORRUPT_FILENAME
        try:
            with open(path, "a", encoding="utf-8") as f:
                for raw in corrupt_lines:
                    f.write(raw + "\n")
            logger.warning("Bridge quarantined %d corrupt line(s) to %s",
                           len(corrupt_lines), path)
            return path
        except OSError as e:
            log_data_loss("bridge", "corrupt_quarantine_failed", detail=str(path), exc=e)
            return None

    @staticmethod
    def _mark_imported(jsonl_path: Path,
                       rows: List[Tuple[str, Optional[dict]]],
                       imported_ids: set) -> bool:
        """Atomically rewrite the JSONL, marking ``imported_ids`` as imported.

        Returns True on success, False (with a data-loss record) if the write
        failed — in which case the original file is left untouched.
        """
        tmp_path = jsonl_path.with_name(jsonl_path.name + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for raw, row in rows:
                    if row is None:
                        # Never destroy data we cannot parse.
                        f.write(raw + "\n")
                        continue
                    if row.get("id") in imported_ids:
                        row["imported"] = True
                        row["imported_at"] = datetime.now().isoformat()
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp_path, jsonl_path)
            return True
        except Exception as e:  # noqa: BLE001
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            log_data_loss("bridge", "jsonl_rewrite_failed", detail=str(jsonl_path), exc=e)
            return False
