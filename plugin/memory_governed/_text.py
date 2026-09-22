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

import re
from typing import Any, Dict, Iterable, List, Optional

#: 包含性查重 / 指纹比对的最小正文长度（归一化后）。比这短的正文参与
#: "互相包含"判断会误判（短句几乎必然互相包含）。
DEDUP_MIN_CHARS = 120


def normalize_for_match(text: Any) -> str:
    """**唯一的**文本归一化规则：小写 + 折叠全部空白。

    两个地方必须给出同一个答案，所以这里只留一份实现：

    * **L2 内容指纹**（``_lifecycle.content_fp`` → 墓碑）：撤回一条事实时记的
      指纹，与再次写入时算的指纹必须一致，否则"撤回过的事实"会被重新抽取
      **复活**；
    * **KB 包含性查重**（``_kb``/``memory_cli``）：两个写入路径判断"这条是不是
      已存在"必须同规则，否则同一段正文在插件侧算重复、在 CLI 侧算新笔记。

    这个仓库已经因为"同一规则两份实现"付过一次代价（L2 provenance 列的
    schema 有、回填没有）。所以调用方一律别名到本函数，别再抄一份 ——
    漂移的后果是**静默**的：要么撤回失效，要么查重失效。
    """
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


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
