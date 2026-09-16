# -*- coding: utf-8 -*-
"""Vault containment (P0) and CLI L2 ranking / schema (P1) regression tests.

Two unrelated-looking failures share a cause, which is why they share a file:
in both cases the *shared* entry point quietly implemented a weaker version of
a contract the plugin already honoured.

* ``kb-add --section`` joined attacker-controlled text straight onto the vault
  path, so ``--section ../memory`` wrote MEMORY.md — the human-authored L1
  rulebook the CLI documents as not writable by any agent.
* The CLI's L2 channel scored every keyword hit ``0.9``, so the calibrated L2
  threshold could never reject anything, and it returned hits in write order
  without saying it had truncated them.

Both were invisible to the suite because the tests that existed exercised the
paths that already worked.
"""

from __future__ import annotations

import json
import os
import subprocess
import types
from pathlib import Path

import pyarrow as pa
import pytest

import memory_cli as cli
from plugin.memory_governed._sync import (
    L2_PROVENANCE_COLUMNS,
    l2_provenance_fields,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """HERMES_HOME with a vault and an L2 directory, wired into the CLI."""
    home = tmp_path / ".hermes"
    (home / "memory" / "l2").mkdir(parents=True)
    vault = tmp_path / "vault"
    for sub in ("notes", "inbox"):
        (vault / sub).mkdir(parents=True)
    (home / "governed_memory.json").write_text(
        json.dumps({"wiki_dir": str(vault), "kb": {"confidence_threshold": 0.7}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    return types.SimpleNamespace(home=home, vault=vault,
                                 cfg={"wiki_dir": str(vault)})


def _make_link(link: Path, target: Path) -> bool:
    """Point ``link`` at ``target`` as a symlink, or a junction as a fallback.

    Junctions matter and are the realistic case on this machine (both
    ``hermes-home`` and ``wiki`` are junctions): creating a symlink on Windows
    needs Developer Mode or elevation, while a junction needs neither, so a
    symlink-only helper would silently stop testing the interesting case.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        if cli._is_link(link):
            return True
        # Seen on this machine: symlink_to returns without error and creates
        # nothing (no Developer Mode), which would make the test below pass for
        # the wrong reason — it would be checking a path that is not a link.
        if link.exists():
            link.rmdir()
    except (OSError, NotImplementedError, ValueError):
        pass
    if os.name != "nt" or link.exists():
        return False
    proc = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0 and cli._is_link(link)


# -- P0: vault containment --------------------------------------------------
class TestKbAddSectionContainment:
    """``--section`` is caller-controlled text and was joined unchecked.

    The refusal has to be loud. The pre-fix behaviour returned ``ok: true``
    with a path outside the vault, so an agent could rewrite L1 while its own
    health check kept reporting success.
    """

    @staticmethod
    def _sections():
        out = ["../memory", "../../memory", "notes/../../memory"]
        if os.altsep == "\\":  # Windows: the backslash really is a separator
            out.append("..\\memory")
        return out

    @pytest.mark.parametrize("section", _sections.__func__())
    def test_traversal_is_refused_not_rewritten(self, store, section):
        out = cli.cmd_kb_add(store.cfg, "MEMORY", "pwned", section,
                             [], [], None, agent="dsh")
        assert out["ok"] is False
        assert "outside the vault" in out["error"]
        # The write must not land anywhere: a silently relocated note is worse
        # than a refused one, because nothing signals the loss.
        assert not list(store.home.parent.rglob("MEMORY.md"))

    def test_absolute_section_is_refused(self, store, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        out = cli.cmd_kb_add(store.cfg, "USER", "pwned", str(outside),
                             [], [], None, agent="dsh")
        assert out["ok"] is False
        assert not (outside / "USER.md").exists()

    def test_absolute_section_within_the_home_tree_is_refused(self, store):
        # The QA reproduction pointed at <sandbox>/memory, i.e. the sibling of
        # the vault's own parent — inside HERMES_HOME but outside the vault.
        out = cli.cmd_kb_add(store.cfg, "MEMORY", "pwned",
                             str(store.home / "memory"), [], [], None)
        assert out["ok"] is False
        assert not (store.home / "memory" / "MEMORY.md").exists()

    def test_link_inside_the_vault_to_outside_is_refused(self, store, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        link = store.vault / "leak"
        if not _make_link(link, outside):
            pytest.skip("cannot create a symlink or junction in this environment")
        out = cli.cmd_kb_add(store.cfg, "MEMORY", "pwned", "leak",
                             [], [], None, agent="dsh")
        assert out["ok"] is False
        assert "link" in out["error"]
        assert not (outside / "MEMORY.md").exists()

    def test_the_naive_join_would_have_escaped(self, store):
        # Pins the mechanism rather than one exploit: if the containment check
        # is ever swapped back for `vault / section`, this fails even though
        # every other assertion here still passes.
        naive = Path(os.path.normpath(str(store.vault / "../memory" / "MEMORY.md")))
        assert not cli._is_within(naive, Path(os.path.abspath(str(store.vault))))

    def test_legitimate_sections_still_work(self, store):
        out = cli.cmd_kb_add(store.cfg, "运维笔记", "body", "areas/ops",
                             [], [], None, agent="dsh")
        assert out["ok"] is True, out
        assert cli._is_within(Path(out["path"]).resolve(),
                              Path(os.path.realpath(str(store.vault))))

    def test_title_cannot_climb_out_either(self, store):
        # --title was already covered by slugify, but containment is now
        # asserted on the final path, not on each argument in isolation.
        out = cli.cmd_kb_add(store.cfg, "../../../MEMORY", "x", "notes",
                             [], [], None, agent="dsh")
        assert out["ok"] is True, out
        assert cli._is_within(Path(out["path"]).resolve(),
                              Path(os.path.realpath(str(store.vault))))


class TestKbSearchSectionContainment:
    """The same escape on the read side: ``--section`` fed a ``rglob`` root."""

    def test_escaping_section_is_refused_and_named(self, store):
        refused: list = []
        hits = cli.cmd_kb_search(store.cfg, "anything", 5, "../memory", refused)
        assert hits == []
        assert refused and "outside the vault" in refused[0]

    def test_legitimate_section_still_searches(self, store):
        cli.cmd_kb_add(store.cfg, "运维笔记", "示例主路由 8022", "notes",
                       [], [], None, agent="dsh")
        refused: list = []
        hits = cli.cmd_kb_search(store.cfg, "运维", 5, "notes", refused)
        assert [h["title"] for h in hits] == ["运维笔记"]
        assert refused == []


# -- P1: CLI L2 ranking -----------------------------------------------------
_ON_TOPIC = "SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE"
_PARTIAL = "SecretStore 部署在 NAS 上"
_OFF_TOPIC = "今天的天气不错"


def _seed(home, texts) -> None:
    import lancedb

    l2 = home / "memory" / "l2"
    l2.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(l2))
    if "memories" in db.table_names():
        db.drop_table("memories")
    db.create_table("memories", data=[{
        "content": t, "category": "other", "source": "test",
        "timestamp": "2026-09-16T00:00:00", "vector": [0.0, 0.0],
        "source_rowid": None, "role": "agent", "agent": "tester",
        "project": None,
    } for t in texts])


class TestCliL2Ranking:
    """The CLI must not be the weaker of the two L2 implementations.

    The threshold is calibrated at 0.76 (relevant 8/8 measured 0.8458~0.9095,
    irrelevant 8/8 measured 0.6755~0.7515). A hardcoded 0.9 cleared it
    unconditionally, so an external agent got an unfiltered, unordered list
    that looked like it had passed the same gate the plugin applies.
    """

    def test_hits_are_ordered_by_relevance_not_write_order(self, store):
        # Write order is the worst possible ranking: it hands back whichever
        # facts happened to be stored first.
        _seed(store.home, [_PARTIAL, _OFF_TOPIC, _ON_TOPIC])
        hits = cli.search_l2("SecretStore ROCKET_TLS", 10, [])
        assert [h["content"] for h in hits] == [_ON_TOPIC, _PARTIAL]

    def test_score_is_coverage_not_a_constant(self, store):
        _seed(store.home, [_PARTIAL, _OFF_TOPIC, _ON_TOPIC])
        hits = cli.search_l2("SecretStore ROCKET_TLS", 10, [])
        scores = [h["score"] for h in hits]
        assert scores == [1.0, 0.5]
        # The bug, pinned: every hit used to carry 0.9 regardless of overlap.
        assert 0.9 not in scores
        assert all(h["score_basis"] == "keyword-coverage" for h in hits)

    def test_irrelevant_query_is_filtered_by_the_floor(self, store):
        _seed(store.home, ["只有 alpha 这一个词是重合的"])
        # 1 of 4 query terms -> 0.25 coverage, far below the calibrated 0.76.
        assert cli.search_l2("alpha beta gamma delta", 10, [], l2_floor=0.76) == []
        # ... and reported as filtered rather than as an empty vault.
        ranking = cli.recall_l2("alpha beta gamma delta", 10, [],
                                l2_floor=0.76)["ranking"]
        assert ranking["filtered_out"] == 1
        # Without a floor the same hit is returned, but never as a high score.
        unfiltered = cli.search_l2("alpha beta gamma delta", 10, [], l2_floor=0.0)
        assert [h["score"] for h in unfiltered] == [0.25]

    def test_relevant_query_survives_the_calibrated_floor(self, store):
        _seed(store.home, [_ON_TOPIC, _PARTIAL])
        hits = cli.search_l2("SecretStore ROCKET_TLS", 10, [], l2_floor=0.76)
        # Full coverage clears 0.76; the half-match does not. That is the
        # behaviour the constant 0.9 made impossible.
        assert [h["content"] for h in hits] == [_ON_TOPIC]

    def test_floor_is_read_through_the_plugin_function(self, store, monkeypatch):
        monkeypatch.setattr(
            cli, "plugin_config",
            lambda: types.SimpleNamespace(
                recall=types.SimpleNamespace(l2_min_score=0.76)))
        _seed(store.home, [_PARTIAL])
        ranking = cli.recall_l2("SecretStore ROCKET_TLS", 10, [])["ranking"]
        assert ranking["floor"] == 0.76
        assert ranking["floor_resolved"] is True
        assert cli.recall_l2("SecretStore ROCKET_TLS", 10, [])["hits"] == []

    def test_ranking_is_reported_honestly(self, store):
        _seed(store.home, [_ON_TOPIC])
        ranking = cli.recall_l2("SecretStore", 10, [], l2_floor=0.5)["ranking"]
        assert ranking["ranked"] is True
        assert ranking["basis"] == "keyword-coverage"
        assert ranking["floor"] == 0.5

    def test_an_unresolvable_floor_is_admitted_not_implied(self, store,
                                                           monkeypatch):
        # If the threshold cannot be read, saying so beats returning unfiltered
        # hits that look like they cleared a gate.
        def _boom(_name):
            raise RuntimeError("plugin unavailable")

        monkeypatch.setattr(cli, "plugin_module", _boom)
        ranking = cli.recall_l2("SecretStore", 10, [])["ranking"]
        assert ranking["floor_resolved"] is False
        assert ranking["floor"] == 0.0

    def test_truncation_is_reported(self, store):
        _seed(store.home, [_ON_TOPIC, _PARTIAL])
        ranking = cli.recall_l2("SecretStore", 1, [], l2_floor=0.1)["ranking"]
        assert ranking["truncated"] == 1


# -- P1: L2 provenance columns ---------------------------------------------
class TestL2ProvenanceColumns:
    """``role`` existed in the schema and in cmd_remember but not in the
    CLI's backfill, so ``table.add()`` raised "field 'role' does not exist" on
    any legacy table and every external agent's ``remember`` failed at once.

    Guards the *shape* of the bug, not just this instance: three places must
    agree (create-table schema, plugin backfill, CLI backfill) and the tests
    all run against tables that already have the columns.
    """

    def test_definition_covers_all_provenance_columns(self):
        names = {name for name, _ in L2_PROVENANCE_COLUMNS}
        assert {"source_rowid", "role", "agent", "project"} <= names

    def test_fields_carry_the_declared_types(self):
        fields = {f.name: str(f.type) for f in l2_provenance_fields()}
        assert fields["role"] == "string"
        assert fields["agent"] == "string"
        assert fields["project"] == "string"
        assert fields["source_rowid"] == "int64"

    def test_remember_succeeds_on_a_table_without_the_role_column(
            self, store, monkeypatch, dim=8):
        import lancedb

        real = cli.plugin_module

        class _FakeService:
            available = True
            last_error = ""
            backend_name = "fake:test"

            def embed_one(self, text):
                return [0.1] * dim

        def dispatch(name):
            if name == "_embedding":
                return types.SimpleNamespace(
                    EmbeddingService=types.SimpleNamespace(
                        get=lambda cfg: _FakeService()))
            return real(name)

        monkeypatch.setattr(cli, "plugin_module", dispatch)

        # A table created before provenance was tracked: no role/agent/project
        # and no source_rowid either.
        db = lancedb.connect(str(store.home / "memory" / "l2"))
        db.create_table("memories", schema=pa.schema([
            pa.field("content", pa.string()),
            pa.field("category", pa.string()),
            pa.field("source", pa.string()),
            pa.field("timestamp", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
        ]))

        out = cli.cmd_remember({}, "家用NAS 192.0.2.62 上重度使用 Docker 部署服务",
                               "dsh")
        assert out["ok"] is True, out

        table = db.open_table("memories")
        cols = [f.name for f in table.schema]
        assert "role" in cols and "agent" in cols and "project" in cols
        assert table.to_arrow().column("role").to_pylist() == ["agent"]
