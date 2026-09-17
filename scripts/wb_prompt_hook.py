#!/usr/bin/env python
"""WorkBuddy ``UserPromptSubmit`` hook — surface related facts, off a snapshot.

Why the snapshot, and not the CLI
---------------------------------
Measured 2026-09-17 on the real store:

    sh + python start .............. 0.71s
    recall --lexical-only .......... 2.16s
    recall (semantic) .............. 2.83s

Most of that is interpreter start plus LanceDB load and a full table scan. Paying
it once per session is fine; paying it before *every prompt* is not — and the
prompt is exactly when a related fact is worth having. The store holds 21 facts
(~5KB of JSON), so the question "does the store already say something about
this?" is answerable from a file: no spawn, no LanceDB, no network.

Why the gate is not ``l2_lexical_score >= floor``
------------------------------------------------
Calibrated 2026-09-17 against 6 prompts that do have a matching fact and 6 that
do not. Scoring with the CLI's own lexical matcher **cannot separate them**:

    relevant   best score in [0.0000, 1.0000]     (two scored a true 0)
    irrelevant best score in [0.0000, 0.5000]

and the false positives were not near-misses — they came from the tokenizer
treating junk as evidence:

    "1+1 等于几"            -> 0.5000, off the token "1" inside "1. **DHCP ..."
    "帮我把这个表格转成 CSV"   -> 0.3333, off the pronoun "这个"

So this hook does not re-use that score as a gate. It requires a **strong term**
to be shared — the same tokenizer's terms, minus the ones that are not evidence:
pure digits, single characters, and a short list of function words. A fact is
eligible only if it shares at least one such term, and the block is capped hard.

Known limit, measured rather than assumed: a shared content word is not meaning.
``解释一下什么是反向代理`` still shares 代理 with a fact about routing traffic
through a proxy, and will inject it. The cost is one short line the agent can
ignore; the alternative — noticing a related fact only after answering — is the
cost this hook exists to avoid.

Contract
--------
* JSON on stdin (``prompt`` / ``cwd`` / ``session_id``); JSON on stdout.
* **Never blocks, never fails loudly.** Every error path emits
  ``{"continue": true}`` with no context. A memory layer that can stall a
  conversation is worse than one that is quiet for a turn.
* No context when nothing matches: ordinary prompts cost zero tokens.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

REPO = Path(__file__).resolve().parent.parent

#: How many facts may be injected at once.
TOP_N = 2

#: Hard ceiling on the injected block, header included.
MAX_CHARS = 400

#: A snapshot older than this is not trusted for *injection* — the store may
#: have moved on, and an injected claim looks authoritative. A session start
#: rewrites it, so only very long-lived sessions can hit this.
MAX_AGE_SECONDS = 24 * 3600

#: ASCII terms shorter than this are not evidence. Kills ``1+1`` (the token
#: "1" matched the "1." of a numbered list) without touching real markers like
#: ``msf``, ``csv``, ``secretstore``.
MIN_ASCII_TERM = 3

#: CJK bigrams that carry no topic. The tokenizer emits bigrams, so a phrase
#: like 这个 / 怎么 / 可以 is *always* produced and will match any text that
#: happens to contain it. Measured false positive: 帮我把这个表格转成 CSV
#: scored 0.3333 purely off 这个.
GENERIC_TERMS = {
    "这个", "那个", "什么", "怎么", "为什", "可以", "不能", "能不能", "是否",
    "一下", "帮我", "我们", "你们", "他们", "现在", "已经", "还是", "就是",
    "这些", "那些", "如果", "然后", "因为", "所以", "但是", "而且", "或者",
    "时候", "问题", "东西", "地方", "怎么", "如何", "多少", "哪个",
}

#: Characters that carry grammar rather than topic. A bigram built **entirely**
#: from these is as uninformative as either of its halves, and the tokenizer
#: will happily produce them by straddling a word boundary — measured:
#: 帮我把这个表格转成 CSV matched a fact about an SSH password prompt through
#: the bigram 把这. Filtering whole phrases one by one does not scale; filtering
#: by composition does.
FUNCTION_CHARS = set(
    "的了和是在有就不也都还吗呢把被给与让使对于之其而或所以因从此那这"
    "什么怎么你我们他它们个上下中前后里外时候能不能可以是否一二三四五六"
    "七八九十多少你要想做会得着过到为很更最再又只才"
)


def _is_function_bigram(term: str) -> bool:
    """True for CJK bigrams made only of grammatical characters."""
    if term.isascii() or len(term) != 2:
        return False
    return all(ch in FUNCTION_CHARS for ch in term)


def snapshot_file() -> Path:
    """Same default the CLI writes to — one definition, two processes."""
    home = os.environ.get("HERMES_HOME") or os.path.join(
        os.path.expanduser("~"), "AppData", "Local", "hermes")
    return Path(home) / "memory" / "l2_snapshot.json"


def read_payload() -> dict:
    try:
        raw = sys.stdin.read()
    except Exception:  # noqa: BLE001 — a hook must not die on its input
        return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def load_snapshot(path: Path, now: float = None) -> tuple:
    """Return ``(facts, problem)``. Empty facts means "inject nothing"."""
    import time

    now = time.time() if now is None else now
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except OSError:
        return [], "no snapshot"
    except (ValueError, TypeError):
        return [], "snapshot unreadable"
    if not isinstance(data, dict):
        return [], "snapshot malformed"
    if data.get("schema") != 1:
        return [], f"snapshot schema {data.get('schema')!r} not understood"
    if data.get("truncated"):
        # A partial snapshot would make this hook confidently blind. Say so and
        # stand down; the caller can still pull through the CLI.
        return [], "snapshot truncated"
    age = now - _epoch(data.get("generated_at"))
    if age > MAX_AGE_SECONDS:
        return [], f"snapshot stale ({age / 3600:.1f}h)"
    facts = data.get("facts")
    return (facts if isinstance(facts, list) else []), ""


def _epoch(stamp) -> float:
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except (TypeError, ValueError):
        return 0.0


def strong_terms(text: str) -> set:
    """Terms that are evidence — the CLI's tokenizer, minus the noise.

    Re-uses ``memory_cli.l2_query_terms`` rather than re-tokenising: a second
    tokenizer would drift from the one the recall path uses, and then this hook
    and the store would disagree about what counts as a match.
    """
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import memory_cli  # noqa: PLC0415 — deliberate, keeps the hook light
        terms = memory_cli.l2_query_terms(text)
    except Exception:  # noqa: BLE001 — no tokenizer, no injection
        return set()

    out = set()
    for term in terms:
        t = str(term).strip()
        if not t or t in GENERIC_TERMS:
            continue
        if _is_function_bigram(t):
            continue                       # 把这 / 值是 — grammar, not topic
        if t.isdigit():
            continue                       # "1", "2026" are not topics
        if t.isascii() and len(t) < MIN_ASCII_TERM:
            continue
        if not t.isascii() and len(t) < 2:
            continue                       # unigram CJK is never evidence
        out.add(t)
    return out


def match(prompt: str, facts: list, top_n: int = TOP_N) -> list:
    """Facts sharing at least one strong term with the prompt, best first.

    Ranked by *how many* distinct strong terms overlap, not by a coverage
    fraction: coverage divides by the prompt's length, so a long question is
    punished for being long — the same dimension error that emptied the CLI's
    lexical channel once already.
    """
    wanted = strong_terms(prompt)
    if not wanted:
        return []
    scored = []
    for fact in facts:
        content = str((fact or {}).get("content") or "").strip()
        if not content:
            continue
        shared = wanted & strong_terms(content)
        if shared:
            scored.append((len(shared), sorted(shared), fact))
    scored.sort(key=lambda row: (-row[0], row[2].get("timestamp") or 0))
    return scored[:top_n]


def build_context(matches: list) -> str:
    if not matches:
        return ""
    lines = ["### 共享记忆里的相关事实（HGM · L2）", ""]
    for _, shared, fact in matches:
        who = fact.get("agent") or "unattributed"
        lines.append(f"- {str(fact.get('content')).strip()}  _({who}; {'/'.join(shared)})_")
    lines.append("")
    lines.append("> 自动匹配，可能不相关。要完整召回请跑 "
                 "`python memory_cli.py recall \"<查询>\"`。")
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rstrip() + "…"
    return text


def main() -> int:
    payload = read_payload()
    prompt = str(payload.get("prompt") or "")
    facts, problem = load_snapshot(snapshot_file())
    context = build_context(match(prompt, facts)) if facts else ""

    out = {
        "continue": True,
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        },
    }
    if os.environ.get("HGM_HOOK_DEBUG") == "1":
        sys.stderr.write(
            f"[hgm-prompt] {len(context)} 字符 "
            f"({'命中' if context else ('快照不可用: ' + problem if problem else '无相关事实')}; "
            f"facts={len(facts)}; session={payload.get('session_id', '?')})\n"
        )
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
