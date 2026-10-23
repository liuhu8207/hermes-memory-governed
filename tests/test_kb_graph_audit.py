# -*- coding: utf-8 -*-
"""KB graph audit distinguishes concept placeholders from truly missing notes."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "kb_graph_audit.py"


def _load():
    spec = importlib.util.spec_from_file_location("kb_graph_audit_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_graph_audit_separates_concepts_and_orphans(tmp_path, monkeypatch):
    mod = _load()
    home = tmp_path / "home"
    home.mkdir()
    wiki = tmp_path / "wiki"
    (wiki / "notes").mkdir(parents=True)
    monkeypatch.setattr(mod, "HERMES_HOME", home)
    (wiki / "notes" / "A.md").write_text(
        "---\ntitle: A\nstatus: curated\nconcepts: [Concept]\n---\n"
        "[[B]] [[Concept]]\n", encoding="utf-8")
    (wiki / "notes" / "B.md").write_text(
        "---\ntitle: B\nstatus: curated\n---\n[[A]]\n", encoding="utf-8")
    (wiki / "notes" / "C.md").write_text(
        "---\ntitle: C\nstatus: curated\n---\nbody\n", encoding="utf-8")
    result = mod.audit(wiki)
    assert result["orphans"] == ["C"]
    assert result["concept_links"] == 1
    assert result["unknown_links"] == 0
