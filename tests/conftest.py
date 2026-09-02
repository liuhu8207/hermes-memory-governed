"""Shared test infrastructure for the governed memory test suite.

Ensures the project root is importable so ``plugin.memory_governed`` can be
imported regardless of how pytest is invoked, and provides the common
HERMES_HOME / config / provider fixtures used across test modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugin.memory_governed import GovernedMemoryProvider  # noqa: E402
from plugin.memory_governed._config import load_governed_config  # noqa: E402


@pytest.fixture
def hermes_home(tmp_path):
    """Minimal but complete HERMES_HOME with L1-L4 + bridge."""
    home = tmp_path / ".hermes"
    mem = home / "memory"
    mem.mkdir(parents=True)
    (mem / "MEMORY.md").write_text(
        "# Rules\n- Don't refactor old modules\n- Mobile still uses them",
        encoding="utf-8",
    )
    (mem / "USER.md").write_text(
        "# User\n- Name: Test User\n- Prefers concise replies", encoding="utf-8"
    )
    (mem / "l2").mkdir()
    (mem / "l3").mkdir()
    (mem / "persona.md").write_text(
        "# Profile\n- Language: Chinese/English", encoding="utf-8"
    )
    bridge = home / "cron" / "output" / "scope_recall_bridge"
    bridge.mkdir(parents=True)

    cfg = {
        "l1_memory_path": str(mem / "MEMORY.md"),
        "l1_user_path": str(mem / "USER.md"),
        "l2_db_path": str(mem / "l2"),
        "l3_db_path": str(mem / "l3" / "l3.db"),
        "l4_persona_path": str(mem / "persona.md"),
        "l4_meta_path": str(mem / "persona_meta.json"),
        "bridge_dir": str(bridge),
    }
    (home / "governed_memory.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )
    return home


@pytest.fixture
def config(hermes_home):
    return load_governed_config(str(hermes_home))


@pytest.fixture
def provider(hermes_home):
    p = GovernedMemoryProvider()
    p.initialize("sess", hermes_home=str(hermes_home))
    yield p
    p.shutdown()
