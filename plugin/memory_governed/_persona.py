"""Which parts of ``persona.md`` are a profile, and which are a derived dump.

Why this module exists
----------------------
``persona.md`` is not only a persona. The L4 synthesiser appends three generated
sections, and ``_sync.py`` says so in as many words — ``## Known Facts`` is
described there as "L2 事实的转储". That listing is the whole of L2, rewritten
into a file that is injected into **every** session.

Measured 2026-09-17, the consequence was not subtle:

* **20 of 23 L2 facts were already present in the session injection**, so an
  agent had no reason ever to call the memory tools. The host listed the tools
  thirteen times in one session and called them zero times — the tools were
  working perfectly and were simply unnecessary.
* The dump is **not filtered by ``project``**, so a fact scoped to one project
  reached every session — precisely the leak the scope exists to prevent.
* About 1.1KB of every session's context was this duplication.

Stripping happens at **read** time rather than at generation time on purpose.
The sections are load-bearing elsewhere: ``_bridge.py`` reads persona.md and
skips exactly these headings when collecting promotion candidates. Removing them
from the file would change what the bridge sees; removing them from the
*injected* text changes only what an agent is told.

Consumers
---------
* ``_recall.get_l4`` — the single read point, which feeds both
  ``system_prompt_block`` and the ``score=1.0`` L4 recall result.
* ``scripts/wb_session_hook.py`` keeps its own copy, because importing this
  package from a host adapter drags in the whole plugin (``_recall``, ``_kb``,
  …) and that hook must keep running under a bare interpreter in under a second.
  ``tests/test_persona_sections.py`` pins the two lists equal so they cannot
  drift.
"""
from __future__ import annotations

#: Generated listings, in the order the synthesiser appends them. Everything from
#: the first one found to the end of the file is dropped.
DERIVED_SECTIONS = ("## Knowledge Areas", "## Known Facts", "## Stats")


def persona_only(text) -> str:
    """Return ``text`` with the generated listings removed.

    Returns ``""`` for empty input. Sections absent from the text do not add
    anything — a hand-written persona passes through untouched.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    cut = len(text)
    for header in DERIVED_SECTIONS:
        at = text.find(header)
        if at != -1:
            cut = min(cut, at)
    return text[:cut].strip()
