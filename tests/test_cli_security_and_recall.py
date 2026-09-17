# -*- coding: utf-8 -*-
"""Vault containment (P0), CLI L2 recall calibration (P0), and structured
refusals (P1) — regression tests.

Three failures with one shared cause: the *shared* entry point quietly
implemented a weaker contract than the plugin behind it.

* ``kb-add --section`` was joined straight onto the vault path, so
  ``--section ../memory`` wrote MEMORY.md — the human-authored L1 rulebook the
  CLI documents as unwritable by any agent.
* The CLI's L2 channel scored every keyword hit ``0.9``, so no threshold could
  reject anything; the fix then applied a *cosine*-calibrated threshold to a
  coverage fraction, whose ceiling is ``1/len(query terms)``. Measured on the
  real store that emptied L2 for 15 of 15 natural-language queries.
* ``--section 'memory:evil'`` is lexically inside the vault, so containment
  passed it and ``mkdir`` raised an uncaught ``NotADirectoryError``: no JSON, a
  bare traceback, and a caller that cannot tell "refused" from "broken".

Every scenario below failed before its fix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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

    Junctions matter and are the realistic case on this machine (the production
    ``wiki`` and ``hermes-home`` are both junctions): creating a symlink on
    Windows needs Developer Mode or elevation, while a junction needs neither,
    so a symlink-only helper would silently stop testing the interesting case.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
        if cli._is_link(link):
            return True
        # Seen on this machine: symlink_to returns without error and creates
        # nothing, which would make a test below pass for the wrong reason — it
        # would be checking a path that is not a link at all.
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

    The refusal has to be loud: the pre-fix behaviour returned ``ok: true`` with
    a path outside the vault, so an agent could rewrite L1 while its own health
    check kept reporting success.
    """

    @staticmethod
    def _sections():
        out = ["../memory", "../../memory", "notes/../../memory"]
        if os.name == "nt":
            out.append("..\\memory")
        return out

    @pytest.mark.parametrize("section", _sections.__func__())
    def test_traversal_is_refused_not_rewritten(self, store, section):
        out = cli.cmd_kb_add(store.cfg, "MEMORY", "pwned", section,
                             [], [], None, agent="dsh")
        assert out["ok"] is False
        assert out["error"].startswith("refused:")
        # The write must not land anywhere: a silently relocated note is worse
        # than a refused one, because nothing signals the loss.
        assert not list(store.home.parent.rglob("MEMORY.md"))

    def test_safe_vault_path_itself_refuses_parent_traversal(self, store):
        # Kept as a unit-level guard: the section validator rejects '..' before
        # this is reached, so without this test the containment check itself
        # could rot unnoticed.
        with pytest.raises(cli.VaultPathEscape):
            cli.safe_vault_path(store.vault, "../memory", "MEMORY.md")

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
        # Pins the mechanism rather than one exploit: if containment is ever
        # swapped back for `vault / section`, this fails even though every other
        # assertion here still passes.
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
        # Was caught by containment ("outside the vault"); it is now refused
        # first by the shared section validator, so read and write agree on the
        # same input instead of the reader accepting what the writer rejected.
        refused: list = []
        hits = cli.cmd_kb_search(store.cfg, "anything", 5, "../memory", refused)
        assert hits == []
        assert refused and refused[0].startswith("refused:")
        assert "../memory" in refused[0]

    def test_legitimate_section_still_searches(self, store):
        cli.cmd_kb_add(store.cfg, "运维笔记", "示例主路由 8022", "notes",
                       [], [], None, agent="dsh")
        refused: list = []
        hits = cli.cmd_kb_search(store.cfg, "运维", 5, "notes", refused)
        assert [h["title"] for h in hits] == ["运维笔记"]
        assert refused == []


# -- P1: the read path follows links too ------------------------------------
class TestVaultWalkContainment:
    """``rglob`` follows links, so the walk cannot be trusted to stay inside.

    The old code did the opposite of checking: ``except ValueError:
    out.append(p)`` kept whatever the walk found outside, so ``kb-search`` and
    ``recall`` returned the contents of whatever a link pointed at — including
    files the vault has no business exposing.
    """

    def test_an_out_of_vault_path_from_the_walk_is_refused_and_named(
            self, store, monkeypatch):
        outside = store.home.parent / "outside"
        outside.mkdir(parents=True, exist_ok=True)
        stray = outside / "stray.md"
        stray.write_text("---\ntitle: stray\n---\nZIGZAG_TOKEN_91827364\n",
                         encoding="utf-8")

        real_rglob = Path.rglob

        def fake_rglob(self, pattern, **kwargs):
            # Emulates a junction under the vault: the walk hands back a path
            # whose real location is outside. Deterministic on every platform,
            # unlike depending on whether rglob follows links here.
            yield from real_rglob(self, pattern, **kwargs)
            yield stray

        monkeypatch.setattr(Path, "rglob", fake_rglob)
        errors: list = []
        notes = cli.iter_notes(store.cfg, None, errors)
        assert stray not in notes
        assert any("outside the vault" in e for e in errors), errors
        # And nothing from outside reaches the search results.
        hits = cli.cmd_kb_search(store.cfg, "ZIGZAG_TOKEN_91827364", 5, "")
        assert hits == []

    def test_a_junction_under_the_vault_is_not_read_through(self, store,
                                                            tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text(
            "---\ntitle: secret\n---\nZIGZAG_TOKEN_91827364\n", encoding="utf-8")
        link = store.vault / "notes" / "leak"
        if not _make_link(link, outside):
            pytest.skip("cannot create a symlink or junction in this environment")

        hits = cli.cmd_kb_search(store.cfg, "ZIGZAG_TOKEN_91827364", 5, "")
        assert hits == []
        vault_real = Path(os.path.realpath(str(store.vault)))
        for note in cli.iter_notes(store.cfg, None, []):
            assert cli._is_within(Path(os.path.realpath(str(note))), vault_real)

    def test_a_link_chain_is_resolved_not_just_one_hop(self, store, tmp_path):
        # Two hops: vault/notes/hop -> vault/notes/hop2 -> outside.
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.md").write_text(
            "---\ntitle: secret\n---\nZIGZAG_TOKEN_91827364\n", encoding="utf-8")
        hop2 = store.vault / "notes" / "hop2"
        if not _make_link(hop2, outside):
            pytest.skip("cannot create a symlink or junction in this environment")
        hop = store.vault / "notes" / "hop"
        if not _make_link(hop, hop2):
            pytest.skip("cannot create a chain in this environment")
        hits = cli.cmd_kb_search(store.cfg, "ZIGZAG_TOKEN_91827364", 5, "")
        assert hits == []


# -- P1: refusals must be verdicts, not tracebacks --------------------------
class TestSectionRefusalIsStructuredJson:
    """A caller that is an agent cannot act on a traceback."""

    @pytest.mark.parametrize("section", [
        "memory:evil",          # NTFS alternate data stream
        "notes:evil",
        "notes*", "no?tes", "no|tes", 'no"tes', "no\x01tes",
        ".", "./",
    ])
    def test_bad_section_names_are_refused_with_a_verdict(self, store, section):
        out = cli.cmd_kb_add(store.cfg, "T", "body", section, [], [], None)
        assert isinstance(out, dict)
        assert out["ok"] is False
        assert out["error"].startswith("refused:")

    def test_an_omitted_section_still_defaults_to_notes(self, store):
        # Pre-existing behaviour, pinned so the validation above cannot be
        # tightened into breaking the default.
        out = cli.cmd_kb_add(store.cfg, "T", "body", "", [], [], None)
        assert out["ok"] is True, out
        assert out["section"] == "notes"

    def test_section_naming_an_existing_file_is_refused(self, store):
        (store.vault / "notes" / "seed.md").write_text("x", encoding="utf-8")
        out = cli.cmd_kb_add(store.cfg, "T", "body", "notes/seed.md",
                             [], [], None)
        assert out["ok"] is False
        assert out["error"].startswith("refused:")
        assert "not a directory" in out["error"]

    def test_reserved_device_name_is_refused(self, store):
        # "CON.md" is rejected by the filesystem, so it must be refused by us —
        # otherwise the caller gets an OSError instead of a verdict, and on the
        # read side path.exists() can resolve to the console device.
        out = cli.cmd_kb_add(store.cfg, "CON", "body", "notes", [], [], None)
        assert out["ok"] is False
        assert "reserved" in out["error"]

    def test_cli_emits_json_and_exit_code_1_not_a_traceback(self, store,
                                                            monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", [
            "memory_cli.py", "kb-add", "T", "body", "--section", "memory:evil"])
        rc = cli.main()
        captured = capsys.readouterr()
        assert rc == 1
        payload = json.loads(captured.out)  # would raise on a traceback
        assert payload["ok"] is False
        assert payload["error"].startswith("refused:")


# -- P2: read and write must validate a --section the same way ---------------
class TestKbSearchUsesTheSectionValidator:
    """``iter_notes`` called ``safe_vault_path`` but not the section validator.

    So every malformed ``--section`` that ``kb-add`` refuses with a verdict was
    accepted by ``kb-search`` and returned as an empty result: rc=0, no
    ``refused`` field, indistinguishable from "no notes matched".
    """

    MALFORMED = [
        "memory:evil",   # NTFS alternate data stream
        "CON",           # reserved device name
        "NUL",
        "no\x01tes",     # a control character
        "a" * 300,       # longer than any filesystem component
        "   ",           # whitespace only
    ]

    @pytest.mark.parametrize("section", MALFORMED)
    def test_read_path_refuses_what_the_write_path_refuses(self, store, section):
        refused: list = []
        hits = cli.cmd_kb_search(store.cfg, "anything", 5, section, refused)
        assert hits == []
        assert refused, "the read path swallowed a malformed section"
        assert refused[0].startswith("refused:")


class TestReadWriteSectionValidationIsSymmetric:
    """Both paths must run the SAME validator: same input, same verdict.

    Guards the *shape* of the bug, not one payload: if the read path ever stops
    calling ``split_vault_section``, these fail even while every other assertion
    still passes.
    """

    @pytest.mark.parametrize("section", [
        "../memory", "memory:evil", "CON", "NUL", "no\x01tes", "a" * 300,
        "   ", ".git", ".obsidian",
    ])
    def test_the_same_section_is_refused_identically_by_both_paths(
            self, store, section):
        add = cli.cmd_kb_add(store.cfg, "T", "body", section, [], [], None)
        refused: list = []
        cli.cmd_kb_search(store.cfg, "T", 5, section, refused)
        assert add["ok"] is False
        assert add["error"].startswith("refused:")
        assert refused and refused[0].startswith("refused:")
        # Byte-for-byte the same verdict: one validator, not two.
        assert add["error"] == refused[0]


class TestKbSearchRefusalIsALoudFailure:
    """A refused read must not exit 0 — a script keying on rc would read it as
    an empty vault."""

    REFUSED = ["memory:evil", "CON", "NUL", "no\x01tes", "a" * 300, "   ",
               "../memory"]

    @pytest.mark.parametrize("section", REFUSED)
    def test_cli_exits_1_with_json_and_a_refused_field(
            self, store, monkeypatch, capsys, section):
        monkeypatch.setattr(sys, "argv", [
            "memory_cli.py", "kb-search", "x", "--section", section])
        rc = cli.main()
        payload = json.loads(capsys.readouterr().out)  # raises on a traceback
        assert rc == 1
        assert payload["refused"], payload
        assert all(r.startswith("refused:") for r in payload["refused"])

    def test_a_legitimate_section_still_exits_0(self, store, monkeypatch,
                                                capsys):
        cli.cmd_kb_add(store.cfg, "运维笔记", "示例主路由 8022", "notes",
                       [], [], None, agent="dsh")
        monkeypatch.setattr(sys, "argv", [
            "memory_cli.py", "kb-search", "运维", "--section", "notes"])
        rc = cli.main()
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert [h["title"] for h in payload["results"]] == ["运维笔记"]
        assert "refused" not in payload


# -- P2: a dot-directory is not writable ------------------------------------
class TestDotDirectoryIsNotWritable:
    """``kb-add --section .git`` was accepted and wrote into ``.git``, but
    ``iter_notes`` skips dot-directories — a write that can never be read back,
    and the failure is silent. The write rule is now the read rule."""

    def test_a_dot_section_is_refused_and_writes_nothing(self, store):
        out = cli.cmd_kb_add(store.cfg, "EVIL", "body", ".git", [], [], None)
        assert out["ok"] is False
        assert out["error"].startswith("refused:")
        assert "hidden directory" in out["error"]
        # Nothing was created, anywhere: a refused write must not leave a trace.
        assert not (store.vault / ".git").exists()
        assert not list(store.vault.rglob("EVIL.md"))

    def test_a_note_already_buried_in_a_dot_dir_is_unreadable(self, store):
        # Pins *why* the write is refused: the vault walk never returns it, so
        # accepting the write would have produced an invisible note.
        hidden = store.vault / ".obsidian"
        hidden.mkdir()
        (hidden / "buried.md").write_text(
            "---\ntitle: buried\n---\nZIGZAG_TOKEN_91827364\n",
            encoding="utf-8")
        assert cli.cmd_kb_search(store.cfg, "ZIGZAG_TOKEN_91827364", 5, "") == []
        out = cli.cmd_kb_add(store.cfg, "buried2", "body", ".obsidian",
                             [], [], None)
        assert out["ok"] is False
        assert out["error"].startswith("refused:")

    def test_a_legitimate_section_is_still_readable_after_write(self, store):
        # The pre-existing invariant, pinned so tightening the rule cannot have
        # broken it: write it, read it straight back.
        out = cli.cmd_kb_add(store.cfg, "运维笔记", "示例主路由 8022", "notes",
                             [], [], None, agent="dsh")
        assert out["ok"] is True, out
        hits = cli.cmd_kb_search(store.cfg, "运维", 5, "notes", [])
        assert [h["title"] for h in hits] == ["运维笔记"]


# -- P0: CLI L2 recall calibration ------------------------------------------
_ON_TOPIC = "SecretStore 使用 ROCKET_TLS 而不是 SSL_CERT_FILE"
_PARTIAL = "SecretStore 部署在 NAS 上"
_OFF_TOPIC = "今天的天气不错"
_OFF_TOPIC_QUERY = "推荐一本讲罗马历史的书"


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


class TestL2ScoreShape:
    """The score must not be capped by how long the question is.

    The shipped version divided by ``len(query terms)``, so the ceiling was
    ``1/len(terms)``: a five-term question could not exceed 0.2 no matter how
    well it matched, and no multi-word query could reach any sane floor.
    """

    def test_a_multi_term_query_clears_the_floor(self):
        score = cli.l2_lexical_score(
            "SecretStore ROCKET_TLS 端口 密码 重启", _ON_TOPIC)
        # 2 of 5 terms present; the old formula gave 2/5 = 0.4 and the old
        # threshold was 0.76, so this could never have survived.
        assert score == pytest.approx(2 / 3)
        assert score > cli.DEFAULT_L2_LEXICAL_FLOOR

    def test_score_rises_with_evidence(self):
        assert cli.l2_lexical_score("SecretStore", _ON_TOPIC) == pytest.approx(1.0)
        assert cli.l2_lexical_score("SecretStore 端口", _ON_TOPIC) == pytest.approx(0.5)
        assert cli.l2_lexical_score("端口 密码", _ON_TOPIC) == pytest.approx(0.0)

    def test_word_order_does_not_decide_the_match(self):
        # The phrase matcher cannot see this; the bigram matcher must.
        # 数据同步 vs 同步数据 is the recall hole QA hit.
        doc = "两边同步数据要注意"
        assert cli.l2_lexical_score("数据同步", doc) >= cli.DEFAULT_L2_LEXICAL_FLOOR

    def test_ascii_and_cjk_are_not_glued_into_one_term(self):
        # `[\w\u4e00-\u9fff]+` made "本地IP通过https访问secretstore" a single
        # term, which can then never match anything except itself.
        terms = cli.l2_query_terms("我想用本地IP通过https访问secretstore")
        assert "secretstore" in terms
        assert "https" in terms
        assert "通过" in terms

    def test_stopwords_carry_no_signal(self):
        assert cli.l2_lexical_score("的 了 是", _ON_TOPIC) == 0.0

    def test_default_floor_sits_in_the_calibrated_window(self):
        # Calibrated on the project's real store (21 facts, 8 relevant +
        # 8 irrelevant queries): relevant top1 in [0.667, 1.0], irrelevant
        # top1 in [0.0, 0.333]. The window is therefore (1/3, 1/2] — one
        # matched term must keep a two-term query but not a three-term one.
        floor = cli.DEFAULT_L2_LEXICAL_FLOOR
        assert 1 / 3 < floor <= 1 / 2

    def test_one_of_two_terms_survives_but_one_of_three_does_not(self):
        assert (cli.l2_lexical_score("SecretStore 端口", _ON_TOPIC)
                >= cli.DEFAULT_L2_LEXICAL_FLOOR)
        assert (cli.l2_lexical_score("SecretStore 端口 密码", _ON_TOPIC)
                < cli.DEFAULT_L2_LEXICAL_FLOOR)


class TestCliL2Ranking:
    def test_hits_are_ordered_by_relevance_not_write_order(self, store):
        # Write order is the worst possible ranking: it hands back whichever
        # facts happened to be stored first.
        _seed(store.home, [_PARTIAL, _OFF_TOPIC, _ON_TOPIC])
        hits = cli.search_l2("SecretStore ROCKET_TLS", 10, [])
        assert [h["content"] for h in hits] == [_ON_TOPIC, _PARTIAL]

    def test_score_is_evidence_not_a_constant(self, store):
        _seed(store.home, [_PARTIAL, _OFF_TOPIC, _ON_TOPIC])
        hits = cli.search_l2("SecretStore ROCKET_TLS", 10, [])
        scores = [h["score"] for h in hits]
        assert scores == [1.0, 0.5]
        # The bug, pinned: every hit used to carry 0.9 regardless of overlap.
        assert 0.9 not in scores
        assert all(h["score_basis"] == "lexical-dis-max" for h in hits)

    def test_a_natural_language_question_still_finds_the_fact(self, store):
        # QA's exact reproduction through the CLI: floor on -> 0 hits with
        # filtered_out 3. The question shares exactly one entity with the fact,
        # which is what a lexical channel has to work with.
        _seed(store.home, [_ON_TOPIC, _PARTIAL, _OFF_TOPIC])
        hits = cli.search_l2("SecretStore 跑在哪台机器上", 10, [])
        assert hits, "the QA reproduction still returns nothing through the CLI"
        assert _ON_TOPIC in [h["content"] for h in hits]

    def test_irrelevant_query_is_filtered_by_the_default_floor(self, store):
        _seed(store.home, ["只有 alpha 这一个词是重合的"])
        assert cli.search_l2(_OFF_TOPIC_QUERY, 10, []) == []
        assert cli.search_l2("alpha beta gamma delta", 10, []) == []
        # ... and reported as filtered rather than as an empty vault: this count
        # is what made the regression visible in the first place.
        ranking = cli.recall_l2("alpha beta gamma delta", 10, [],
                                lexical_only=True)["ranking"]
        assert ranking["filtered_out"] == 1

    def test_relevant_query_survives_the_default_floor(self, store):
        _seed(store.home, [_ON_TOPIC, _PARTIAL])
        hits = cli.search_l2("SecretStore ROCKET_TLS", 10, [])
        assert [h["content"] for h in hits] == [_ON_TOPIC, _PARTIAL]

    def test_the_cosine_floor_is_not_applied_to_this_channel(self, store):
        # The regression, end to end: recall.l2_min_score is calibrated on
        # cosine similarity and lives in the deployed config at 0.76. Applying
        # it to a lexical score is the defect.
        (store.home / "governed_memory.json").write_text(json.dumps({
            "wiki_dir": str(store.vault),
            "recall": {"l2_min_score": 0.99},
        }), encoding="utf-8")
        _seed(store.home, [_ON_TOPIC])
        result = cli.recall_l2("SecretStore ROCKET_TLS", 10, [],
                               lexical_only=True)
        assert len(result["hits"]) == 1
        assert result["ranking"]["floor"] == cli.DEFAULT_L2_LEXICAL_FLOOR
        assert result["ranking"]["floor_source"] == "default"

    def test_floor_is_overridable_from_the_config_file(self, store):
        (store.home / "governed_memory.json").write_text(json.dumps({
            "wiki_dir": str(store.vault),
            "recall": {"l2_lexical_min_score": 0.9},
        }), encoding="utf-8")
        _seed(store.home, [_ON_TOPIC, _PARTIAL])
        result = cli.recall_l2("SecretStore ROCKET_TLS", 10, [],
                               lexical_only=True)
        assert result["ranking"]["floor"] == 0.9
        assert result["ranking"]["floor_source"] == "config-file"
        assert [h["content"] for h in result["hits"]] == [_ON_TOPIC]
        assert result["ranking"]["filtered_out"] == 1

    def test_ranking_is_reported_honestly(self, store):
        _seed(store.home, [_ON_TOPIC])
        ranking = cli.recall_l2("SecretStore", 10, [], l2_floor=0.5,
                                lexical_only=True)["ranking"]
        assert ranking["ranked"] is True
        assert ranking["basis"] == "lexical-dis-max"
        assert ranking["floor"] == 0.5
        assert ranking["floor_source"] == "explicit"

    def test_truncation_is_reported(self, store):
        _seed(store.home, [_ON_TOPIC, _PARTIAL])
        ranking = cli.recall_l2("SecretStore", 1, [], l2_floor=0.1,
                                lexical_only=True)["ranking"]
        assert ranking["truncated"] == 1


# -- P1: the CLI's semantic L2 channel --------------------------------------
#: A deterministic stand-in for the embedding backend. ``car`` / ``车子`` /
#: ``automobile`` share one axis, so the fake model treats them as one meaning
#: with no shared wording — which is precisely the case no lexical matcher can
#: solve, and therefore the case that proves the vector channel is doing
#: something the keyword channel cannot.
_FAKE_DIM = 4
_FAKE_AXES = {"car": 0, "automobile": 0, "车子": 0, "alpha": 1, "beta": 2}
_CAR_DOC = "I drive an automobile to work"


def _fake_vec(text: str) -> list:
    v = [0.0] * _FAKE_DIM
    low = str(text or "").lower()
    for tok, axis in _FAKE_AXES.items():
        if tok in low:
            v[axis] = 1.0
    if not any(v):
        v[_FAKE_DIM - 1] = 1.0
    return v


def _seed_vectors(home, texts) -> None:
    import lancedb

    l2 = home / "memory" / "l2"
    l2.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(l2))
    if "memories" in db.table_names():
        db.drop_table("memories")
    db.create_table("memories", data=[{
        "content": t, "category": "other", "source": "test",
        "timestamp": "2026-09-16T00:00:00", "vector": _fake_vec(t),
        "source_rowid": None, "role": "agent", "agent": "tester",
        "project": None,
    } for t in texts])


def _install_fake_embedding(monkeypatch, available: bool = True,
                            last_error: str = "") -> None:
    """Point ``plugin_module("_embedding")`` at a fake, so tests never call out."""
    real = cli.plugin_module

    class _Service:
        def __init__(self):
            self.available = available
            self.last_error = last_error
            self.backend_name = "fake:test"

        def embed_one(self, text):
            return _fake_vec(text) if available else None

    def dispatch(name):
        if name == "_embedding":
            return types.SimpleNamespace(
                EmbeddingService=types.SimpleNamespace(
                    get=lambda cfg: _Service()))
        return real(name)

    monkeypatch.setattr(cli, "plugin_module", dispatch)


class TestL2SemanticChannel:
    """The CLI had no vector channel at all: every hit was keyword-scored.

    So a query that shared no wording with a fact could not find it, no matter
    how close the meaning — measured on the real store, ``openssl`` returned 0
    hits while the fact "SecretStore uses ROCKET_TLS, not SSL_CERT_FILE" sat
    right there. Multi-agent sharing is worth little if a rephrasing loses the
    memory.
    """

    def test_meaning_finds_a_fact_that_shares_no_wording(self, store, monkeypatch):
        # The whole point of the channel, and impossible lexically.
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_CAR_DOC])
        # Lexical, i.e. what the CLI could do before: no shared token, no hit.
        assert cli.recall_l2("car", 5, [], lexical_only=True)["hits"] == []
        # Semantic: "car" and "automobile" are one axis in the fake model.
        hits = cli.recall_l2("car", 5, [])["hits"]
        assert len(hits) == 1
        assert hits[0]["score"] == pytest.approx(1.0)
        assert hits[0]["score_basis"] == "semantic-cosine"

    def test_the_semantic_channel_replaces_not_merges(self, store, monkeypatch):
        # Ranking a cosine score and a coverage fraction in one list is the same
        # dimension error as thresholding one with the other's floor.
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_CAR_DOC, "alpha one", "beta two"])
        result = cli.recall_l2("car", 10, [])
        assert result["ranking"]["basis"] == "semantic-cosine"
        assert result["hits"]
        assert all(h["score_basis"] == "semantic-cosine" for h in result["hits"])

    def test_the_semantic_floor_uses_its_own_key(self, store, monkeypatch):
        # The red line: recall.l2_min_score is the in-plugin channel's cosine
        # threshold and must NOT govern this one, nor may the lexical floor.
        (store.home / "governed_memory.json").write_text(json.dumps({
            "wiki_dir": str(store.vault),
            "recall": {"l2_min_score": 0.99, "l2_lexical_min_score": 0.99,
                       "l2_semantic_min_score": 0.5},
        }), encoding="utf-8")
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_CAR_DOC])
        result = cli.recall_l2("car", 5, [])
        assert result["ranking"]["floor"] == 0.5
        assert result["ranking"]["floor_source"] == "config-file"
        assert result["hits"]

    def test_the_default_semantic_floor_sits_in_the_calibrated_window(
            self, store, monkeypatch):
        # Calibrated on the real store (21 facts, bge-m3): relevant top1 in
        # [0.7843, 0.9640], irrelevant top1 in [0.6693, 0.7586].
        floor = cli.DEFAULT_L2_SEMANTIC_FLOOR
        assert 0.7586 < floor <= 0.7843

    def test_an_unavailable_backend_degrades_to_lexical_and_says_why(
            self, store, monkeypatch):
        # Must not raise, must not silently return an empty L2.
        _install_fake_embedding(monkeypatch, available=False,
                                last_error="simulated: no api key")
        _seed_vectors(store.home, ["alpha one"])
        errors: list = []
        result = cli.recall_l2("alpha", 5, errors)
        assert result["ranking"]["basis"] == "lexical-dis-max"
        assert result["ranking"]["semantic"]["available"] is False
        assert "simulated: no api key" in result["ranking"]["semantic"]["reason"]
        assert any("l2 semantic" in e for e in errors), errors
        # ... and the lexical channel still answers.
        assert [h["content"] for h in result["hits"]] == ["alpha one"]

    def test_lexical_only_skips_a_live_backend(self, store, monkeypatch):
        _install_fake_embedding(monkeypatch)
        _seed_vectors(store.home, [_CAR_DOC, "alpha one"])
        result = cli.recall_l2("car", 5, [], lexical_only=True)
        assert result["ranking"]["basis"] == "lexical-dis-max"
        assert result["hits"] == []
        # "Not attempted" and "attempted and unavailable" are different facts and
        # are reported differently: no 'semantic' block at all here, versus one
        # carrying available=False and a reason in the degradation test above.
        assert "semantic" not in result["ranking"]

    def test_the_cli_exposes_lexical_only(self):
        parser = cli.build_parser()
        args = parser.parse_args(["recall", "openssl", "--lexical-only"])
        assert args.lexical_only is True
        assert parser.parse_args(["recall", "openssl"]).lexical_only is False


# -- P1: L2 provenance columns ---------------------------------------------
class TestL2ProvenanceColumns:
    """``role`` existed in the schema and in cmd_remember but not in the CLI's
    backfill, so ``table.add()`` raised "field 'role' does not exist" on any
    legacy table and every external agent's ``remember`` failed at once.

    Guards the *shape* of the bug, not just this instance: three places must
    agree (create-table schema, plugin backfill, CLI backfill) and every test
    runs against a table that already has the columns.
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
