"""Encoding hygiene for text that arrives from outside the plugin.

The problem
-----------
Python strings can hold lone surrogates (``\\udcae``) that UTF-8 cannot encode.
They arrive in practice: a host's prompt injection occasionally carries an
escaped surrogate from a terminal or a log, and JSON transports it fine because
``json.dumps`` escapes it as ``\\udcae`` — the string is representable, just not
encodable.

Every layer below then fails in its own way, and none of them say why:

* the embedding call raises ``'utf-8' codec can't encode character '\\udcae'``,
  which ``embed_one`` turns into a ``None`` return — so the semantic channel
  quietly gives up and recall drops to lexical;
* LanceDB stores Arrow strings, and Arrow demands strict UTF-8, so the row
  cannot be written at all;
* SQLite encodes its text parameters as UTF-8 too, so the L3 archive and its FTS
  mirror raise on the same input.

Measured 2026-09-18: the first of those had been silently degrading the semantic
channel on the DSH side.

Why one function, called from one place
---------------------------------------
This logic had three copies — ``EmbeddingService._sanitize``,
``hgm_mcp._ensure_utf8`` and an inline block in ``memory_cli.cmd_remember`` — all
identical, and none of them covered the plugin's own extraction path, which is
where a fact actually lands in L2. Three copies of a guard is three chances to
drift, and the fourth entry point was the one that mattered.

So: one definition here, and the *write* path calls it once at its ingress
(``sync_turn``) rather than every layer defending itself. A surrogate that does
not enter the pipeline cannot fail anywhere downstream.

``errors="replace"`` is a deliberate choice: the affected character becomes
``?`` and the rest survives. Rejecting the whole write would be the stricter
option, and for a recall query it would be the worse one — a question with one
bad byte is still a question.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def sanitize_utf8(text: Any) -> str:
    """Return ``text`` as a string that is guaranteed encodable as UTF-8.

    Clean text is returned untouched, and the common path pays only one
    ``encode`` that succeeds. Text containing lone surrogates comes back with
    those characters replaced by ``?``.
    """
    if text is None:
        return ""
    value = text if isinstance(text, str) else str(text)
    try:
        value.encode("utf-8")
        return value
    except UnicodeEncodeError:
        return value.encode("utf-8", errors="replace").decode("utf-8")


def sanitize_messages(messages: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Sanitize the ``content`` of each message, leaving everything else alone.

    Used at the ingress of the write pipeline, so that the archive, its FTS
    mirror and the extraction that produces L2 facts all see the same text —
    one sanitization covering four consumers instead of four guards.
    """
    out: List[Dict[str, Any]] = []
    for message in messages or []:
        if isinstance(message, dict) and "content" in message:
            copy = dict(message)
            copy["content"] = sanitize_utf8(copy["content"])
            out.append(copy)
        else:
            out.append(message)
    return out
