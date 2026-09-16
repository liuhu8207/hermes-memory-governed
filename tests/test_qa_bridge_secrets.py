"""Bridge secret handling — ACCEPTANCE SPEC for the fail-closed (option A) fix.

STATUS: PENDING IMPLEMENTATION (engineer A). These tests are intentionally red
until the fail-closed behaviour lands. Do not "fix" these tests to pass — they
encode the agreed acceptance criteria (architect verdict, section 六).

Agreed design:
  - Detected secrets are NEVER written to candidates.jsonl (fail closed).
    They go to bridge/quarantine.jsonl (redacted preview + source + timestamp).
  - import_approved() has a HARD GATE at the L1 write boundary: any row that
    carries a secret (flag, needs-careful-review tag, or still-detectable
    secret text) is blocked unconditionally — auto_approve=True does NOT
    bypass it.
  - SECRET_PATTERNS gains PEM and Bearer. Bare base64 is explicitly rejected
    (precision > recall under fail-closed: a false positive silently drops a
    legitimate memory).

Acceptance criteria (architect, 六):
  1. test_rejects_secrets returns green (written == 0)
  2. secret candidate lands in quarantine.jsonl, NOT candidates.jsonl;
     get_status()["quarantine_count"] increments
  3. hard gate: approved+secret -> imported 0; auto_approve=True -> imported 0;
     historical secret row pre-seeded in candidates.jsonl -> still blocked
  4. PEM / Bearer are detected and quarantined
  5. reverse check: sk-xxx is not written as a legitimate memory
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugin.memory_governed._bridge import BridgeExporter, has_secret_like_text

SECRET = "The API key: sk-abc123def456ghi789jkl0 must be kept secret"
PEM = "-----BEGIN PRIVATE KEY----- MIIEvQIBADANBgkqhkiG9w0BAQEFAASC"
BEARER = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"


@pytest.fixture
def bridge(config, tmp_path):
    return BridgeExporter(config)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# --- 1 + 5: fail closed ------------------------------------------------------

class TestFailClosed:
    def test_secret_is_rejected_not_stored(self, bridge, config):
        """Acceptance 1 + 5: written == 0, nothing lands in candidates.jsonl."""
        report = bridge.export_candidates(
            [{"content": SECRET, "target": "memory", "source_path": "ops.md"}]
        )
        assert report["written"] == 0, (
            f"secret candidate was stored (written={report['written']}) — "
            "fail-closed requires rejection"
        )
        candidates = _read_jsonl(Path(config.bridge_dir) / "candidates.jsonl")
        assert candidates == [], (
            f"candidates.jsonl must not contain secret candidates, got {candidates}"
        )

    def test_pem_and_bearer_detected(self, bridge, config):
        """Acceptance 4: the two newly approved patterns are recognised."""
        assert has_secret_like_text(PEM), "PEM block is not detected"
        assert has_secret_like_text(BEARER), "Bearer token is not detected"
        report = bridge.export_candidates([
            {"content": PEM, "target": "memory", "source_path": "a.md"},
            {"content": BEARER, "target": "memory", "source_path": "b.md"},
        ])
        assert report["written"] == 0, f"PEM/Bearer leaked into candidates: {report}"


# --- 2: quarantine -----------------------------------------------------------

class TestQuarantine:
    def test_secret_goes_to_quarantine_not_candidates(self, bridge, config):
        """Acceptance 2: audit value preserved at zero L1 risk."""
        bridge.export_candidates(
            [{"content": SECRET, "target": "memory", "source_path": "ops.md"}]
        )
        quarantine = _read_jsonl(Path(config.bridge_dir) / "quarantine.jsonl")
        assert len(quarantine) == 1, (
            f"expected the secret candidate in quarantine.jsonl, got {quarantine}"
        )
        assert "[REDACTED]" in quarantine[0].get("content", ""), (
            "quarantine entry should carry a redacted preview"
        )
        assert quarantine[0].get("source") == "ops.md"
        candidates = _read_jsonl(Path(config.bridge_dir) / "candidates.jsonl")
        assert candidates == [], "quarantined rows must never reach candidates.jsonl"

    def test_status_reports_quarantine_count(self, bridge, config):
        bridge.export_candidates(
            [{"content": SECRET, "target": "memory", "source_path": "ops.md"}]
        )
        status = bridge.get_status()
        assert status.get("quarantine_count", 0) == 1, (
            f"get_status() must expose quarantine_count, got {status}"
        )


# --- 3: hard gate at the L1 boundary ----------------------------------------

class TestL1HardGate:
    """Acceptance 3 — the actual security substance of this fix.

    The gate must live at import_approved (the real trust boundary), not only
    at export_candidates, so that entry-point regressions and historical rows
    cannot reach L1.
    """

    def _approved_secret_row(self) -> dict:
        return {
            "id": "hermes-hardgate",
            "content": SECRET,
            "target": "memory",
            "tags": ["approved"],          # reviewer rubber-stamped it
            "secret_redacted": True,       # written by the current (redact-and-store) build
            "source": "hermes-memory-governed",
            "source_path": "ops.md",
        }

    def test_approved_secret_is_blocked(self, bridge, config):
        """Approved tag must NOT override the secret gate."""
        jsonl = Path(config.bridge_dir) / "candidates.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl.write_text(json.dumps(self._approved_secret_row()) + "\n", encoding="utf-8")

        report = bridge.import_approved(config.l1_memory_path)
        assert report["imported"] == 0, (
            f"an approved secret row reached L1: {report}"
        )

    def test_auto_approve_cannot_bypass_the_gate(self, bridge, config):
        """The most important case: auto_approve=True must not bypass it."""
        jsonl = Path(config.bridge_dir) / "candidates.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl.write_text(json.dumps(self._approved_secret_row()) + "\n", encoding="utf-8")

        report = bridge.import_approved(config.l1_memory_path, auto_approve=True)
        assert report["imported"] == 0, (
            f"auto_approve=True bypassed the secret gate: {report}"
        )

    def test_historical_secret_row_is_blocked(self, bridge, config):
        """A legacy row (no secret_redacted flag, no tag) must still be caught."""
        jsonl = Path(config.bridge_dir) / "candidates.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        legacy = self._approved_secret_row()
        legacy.pop("secret_redacted")
        legacy["tags"] = ["approved"]
        jsonl.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        report = bridge.import_approved(config.l1_memory_path, auto_approve=True)
        assert report["imported"] == 0, (
            f"historical undetected-flagged secret row reached L1: {report}"
        )

    def test_clean_candidates_still_import(self, bridge, config):
        """Control: the gate must not block legitimate memories."""
        jsonl = Path(config.bridge_dir) / "candidates.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        clean = {
            "id": "hermes-clean",
            "content": "We decided to use Postgres instead of MySQL",
            "target": "memory",
            "tags": ["approved"],
            "source": "hermes-memory-governed",
            "source_path": "decisions.md",
        }
        jsonl.write_text(json.dumps(clean) + "\n", encoding="utf-8")

        report = bridge.import_approved(config.l1_memory_path)
        assert report["imported"] == 1, f"legitimate memory was blocked: {report}"


# --- 6: content gate at the L1 boundary (2026-09-16) -------------------------
#
# The export-time screen only guards rows being CREATED. Rows already sitting in
# candidates.jsonl — written before a rule existed, or by an entry point that
# regressed — would otherwise still reach L1, and L1 is re-read by the next
# persona build, so one bad row recirculates forever. Measured case: the
# 2026-09-16 pool held a whole rendered persona document.

class TestContentGateAtL1Boundary:
    _PERSONA_DUMP = (
        "# User Profile\n_Generated: 2026-09-16T00:54:43.681897_\n\n"
        "> 手写用户信息。直接编辑此文件。\n\n"
        "## Known Facts\n- 刚才在修复 `governed_health` 的 bug 时网关重启了\n\n"
        "## Stats\n- Conversations archived: 261\n"
    )

    def _seed(self, bridge, config, content: str) -> Path:
        jsonl = Path(config.bridge_dir) / "candidates.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl.write_text(json.dumps({
            "id": "hermes-legacy-dump",
            "content": content,
            "target": "user",
            "tags": ["approved"],
            "source": "hermes-memory-governed",
            "source_path": "persona.md",
        }) + "\n", encoding="utf-8")
        return jsonl

    def test_historical_dump_cannot_reach_l1(self, bridge, config):
        self._seed(bridge, config, self._PERSONA_DUMP)
        report = bridge.import_approved(config.l1_memory_path)
        assert report["imported"] == 0, f"persona dump reached L1: {report}"
        assert report["blocked_content"] == 1, report
        body = Path(config.l1_memory_path).read_text(encoding="utf-8")
        assert "手写用户信息" not in body
        assert "Conversations archived" not in body

    def test_auto_approve_cannot_bypass_the_content_gate(self, bridge, config):
        self._seed(bridge, config, self._PERSONA_DUMP)
        report = bridge.import_approved(config.l1_memory_path, auto_approve=True)
        assert report["imported"] == 0, (
            f"auto_approve=True bypassed the content gate: {report}"
        )
        assert report["blocked_content"] == 1, report

    def test_clean_row_is_unaffected(self, bridge, config):
        """Control: the new gate must not narrow what legitimately imports."""
        jsonl = Path(config.bridge_dir) / "candidates.jsonl"
        jsonl.parent.mkdir(parents=True, exist_ok=True)
        jsonl.write_text(json.dumps({
            "id": "hermes-clean-2",
            "content": "SecretStore 使用 `ROCKET_TLS` 而不是 `SSL_CERT_FILE`",
            "target": "memory",
            "tags": ["approved"],
            "source": "hermes-memory-governed",
        }) + "\n", encoding="utf-8")

        report = bridge.import_approved(config.l1_memory_path)
        assert report["imported"] == 1, report
        assert report["blocked_content"] == 0, report
