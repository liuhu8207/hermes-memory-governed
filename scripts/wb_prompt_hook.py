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

import hashlib
import json
import os
import re
import sys
from pathlib import Path

for _stream in (sys.stdin, sys.stdout, sys.stderr):
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
    # Low-information nouns. They are real words, so the composition filter above
    # cannot catch them — but every topic has them, so sharing one says nothing.
    # Measured: 有没有更好的办法 matched a fact about a SecretStore reading
    # workaround purely through 办法. Removing such a term only costs a match
    # when it was the *only* overlap, which is exactly the case where the match
    # was noise.
    "办法", "方法", "方式", "事情", "情况", "内容", "状态", "结果", "样子",
}

#: Characters that carry grammar rather than topic. A bigram built **entirely**
#: from these is as uninformative as either of its halves, and the tokenizer
#: will happily produce them by straddling a word boundary — measured:
#: 帮我把这个表格转成 CSV matched a fact about an SSH password prompt through
#: the bigram 把这. Filtering whole phrases one by one does not scale; filtering
#: by composition does.
#:
#: It only works if the list is complete, and the first version was not: 没 was
#: missing, so 没有 survived as a "strong term" and the ordinary question
#: 你再实测看有什么问题没有？ matched a fact reading "❌ AI 不能读密码 → 没有
#: CLI/API 接口". One absent character defeated the whole scheme, which is why
#: the negatives below are regression cases rather than illustrations.
FUNCTION_CHARS = set(
    "的了和是在有就不也都还吗呢把被给与让使对于之其而或所以因从此那这"
    "什么怎么你我们他它们个上下中前后里外时候能不能可以是否一二三四五六"
    "七八九十多少你要想做会得着过到为很更最再又只才"
    # Second pass, added after that miss: negations, modals and light verbs that
    # only ever glue a real term to the sentence.
    "没未别该需看试知道行好太挺常等等样"
)


def _is_function_bigram(term: str) -> bool:
    """True for a CJK bigram that leads with a grammatical character.

    The tokenizer slices fixed-width bigrams out of a run of CJK, so it cannot
    see word boundaries: 有没有更好的办法 yields 的办, and 帮我把这个表格转成
    CSV yields 把这. Both straddle a boundary and neither is a word.

    Requiring **both** characters to be grammatical was the first attempt and it
    missed 的办 — the trailing character there is 办, a real word. Leading with a
    particle is the reliable signal: a bigram that starts with 的 / 把 / 在 / 有
    is glued to whatever preceded it, whatever follows.

    Cost, measured rather than assumed: bigrams like 有关 or 在于 are dropped
    too. The positives in the calibration set (免密 / 门禁 / 阈值 / 密码 / 代理 /
    同步 / 反向 …) start with content characters and are untouched.
    """
    if term.isascii() or len(term) != 2:
        return False
    return term[0] in FUNCTION_CHARS


#: Blocks the host wraps its own additions in. Matching must not see them: they
#: are not what the user asked, and one of them is this hook's **own output**.
#:
#: Measured 2026-09-17 from the run log, which is why the log exists. A message
#: whose visible text was 11 characters arrived with ``prompt`` of length 17; a
#: touch earlier, a run carried length 991. Two consequences, both observed:
#: the matcher judged text the user never wrote, and — because a previous
#: injection is part of that text — it kept matching the same two facts every
#: turn off terms like ``governed`` / ``hgm`` / ``memory`` that came from its
#: own last answer. A matcher that reads its own output is a feedback loop.
_HOST_BLOCK_RE = re.compile(r"<[a-zA-Z_][\w:-]*(?:\s[^>]*)?>.*?</[a-zA-Z_][\w:-]*>",
                            re.S)

#: This hook's own injection, in case the host returns it inside ``prompt``
#: rather than as a separate block. Matched on the header and the trailing
#: disclaimer so the whole thing goes, not just its first line.
_OWN_BLOCK_RE = re.compile(
    r"###\s*共享记忆里的相关事实[^\n]*\n(?:.*\n)*?>[^\n]*自动匹配[^\n]*", re.S)

#: This hook's own stdout. The host appends it verbatim — observed twice in one
#: context — so an *empty* answer arrives as a bare ``{"continue": true, ...}``
#: line with no header for :data:`_OWN_BLOCK_RE` to anchor on. It is emitted on
#: one line by design, which is what makes a line-anchored pattern safe.
_OWN_JSON_RE = re.compile(r'^[ \t]*\{[ \t]*"continue".*$', re.M)

#: Any tag, paired or not. Removing only balanced pairs leaves the closers of
#: nested blocks behind (``</additional_data> </system-reminder>``) — measured —
#: and those names then enter the term pool: the residual matched facts through
#: ``system`` / ``reminder`` / ``additional_data``, i.e. pure markup.
_TAG_RE = re.compile(r"</?[a-zA-Z_][\w:.-]*[^>]*>")


#: Where the user's own words live when the host wraps the message. Measured
#: from the session transcript: the recorded user turn is a ~2300-character
#: block of reminders ending in ``<user_query>好，继续测试</user_query>``.
_USER_QUERY_RE = re.compile(r"<user_query>(.*?)</user_query>", re.S)


def clean_prompt(prompt: str) -> str:
    """The part of ``prompt`` that the user actually typed.

    Returns ``""`` when nothing is left, which the caller treats as "no match"
    — the right answer for a turn that carried only host scaffolding.

    Order matters, and the first branch is the important one. Stripping host
    blocks wholesale is right for scaffolding but catastrophic for a *wrapped
    message*: the probe showed a wrapped payload cleaning down to length 0 —
    the question itself deleted. That is a confident silence, the exact failure
    this store exists to eliminate. So when the wrapper is present, keep what is
    inside it and discard the rest, rather than the other way round.

    The remaining passes answer observed problems: balanced host blocks (their
    content must go, not just their tags), this hook's own injection, this
    hook's own stdout, and finally any tag left over from a nested block.
    """
    text = str(prompt or "")
    if not text:
        return ""
    found = _USER_QUERY_RE.search(text)
    if found:
        text = found.group(1)
    else:
        text = _HOST_BLOCK_RE.sub(" ", text)
        text = _OWN_BLOCK_RE.sub(" ", text)
        text = _OWN_JSON_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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


#: A run log exists because this hook is otherwise invisible. It runs before
#: every prompt, its output is merged into the conversation, and when it
#: misbehaves the symptom is "the agent said something odd" — far from the
#: cause. Measured 2026-09-17: the host's rendered context contained the whole
#: context block *and* a separate raw stdout line whose ``additionalContext``
#: was empty, which is only explicable if the hook ran twice for one prompt.
#: That is 2x the latency and no way to see it without a log.
#:
#: Only sizes are recorded — never the prompt itself. A diagnostic that quietly
#: accumulates the user's questions would be a worse bug than the one it hunts.
MAX_LOG_LINES = 200


def log_path() -> Path:
    """Beside the snapshot, inside HERMES_HOME."""
    return snapshot_file().with_name("hook_log.txt")


def log_run(event: str, session: str, note: str) -> None:
    """Append one line. Never raises — logging must not be able to break a turn."""
    if os.environ.get("HGM_HOOK_LOG") == "0":
        return
    try:
        from datetime import datetime

        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        lines.append(f"{datetime.now().isoformat(timespec='seconds')}\t{event}\t"
                     f"session={session or '?'}\t{note}")
        if len(lines) > MAX_LOG_LINES:
            lines = lines[-MAX_LOG_LINES:]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:  # noqa: BLE001 — a log is not worth a broken turn
        pass


def payload_shape(payload: dict) -> str:
    """Field names and value sizes of the hook payload — never the values.

    Logging the prompt's *length* was not enough to understand this input. A
    21-character question arrived as ``prompt`` of length 28 and matched nothing,
    while the very same text matched two facts when run by hand; elsewhere the
    field carried 991 characters, of which cleaning removed 487. So ``prompt``
    does not hold what the user typed, and the way to find out which field does
    is to look at the payload's shape rather than at its text.

    Sizes and type names only: a diagnostic that quietly accumulates the user's
    words would be a worse bug than the one it is hunting.
    """
    parts = []
    for key in sorted(payload):
        value = payload[key]
        if isinstance(value, str):
            parts.append(f"{key}:str{len(value)}")
        elif isinstance(value, (list, tuple, dict, set)):
            parts.append(f"{key}:{type(value).__name__}{len(value)}")
        else:
            parts.append(f"{key}:{type(value).__name__}")
    return ",".join(parts) or "(empty)"


#: Wrapper names the host is known to add around a user's message. Logging
#: *which* of these appear is diagnostic and leaks nothing: the vocabulary is
#: fixed and belongs to the protocol, not to the user.
_HARNESS_MARKERS = (
    "user_query", "system-reminder", "memory_and_skills_reminder",
    "additional_data", "current_time", "task-notification", "identity_context",
)


def prompt_shape(text: str) -> str:
    """Character classes and protocol markers present — never the text.

    The hash was supposed to settle what ``prompt`` contains: log a short digest,
    then hash candidates locally until one matches. It did not — a 6-character
    message arrived as ``prompt`` of length 9, and no combination of that message
    with whitespace (including U+3000 and zero-width characters) reproduced the
    digest. Brute force cannot close that gap, so the log now reports enough
    *structure* to identify the input by inspection.

    Counts by class, plus which protocol wrappers are present. A Chinese
    sentence and a Chinese sentence with three trailing newlines are
    indistinguishable by length but not by class; a message the host re-wrapped
    is identified by the marker list.
    """
    classes = {"cjk": 0, "ascii": 0, "digit": 0, "space": 0, "other": 0}
    for ch in text:
        if ch.isascii() and ch.isalpha():
            classes["ascii"] += 1
        elif ch.isdigit():
            classes["digit"] += 1
        elif ch.isspace():
            classes["space"] += 1
        elif "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf":
            classes["cjk"] += 1
        else:
            classes["other"] += 1
    counts = ",".join(f"{k}{v}" for k, v in classes.items())
    found = ",".join(m for m in _HARNESS_MARKERS if m in text) or "-"
    return f"cls=[{counts}] markers=[{found}]"


def main() -> int:
    payload = read_payload()
    raw_prompt = str(payload.get("prompt") or "")
    # Judge only what the user typed — see clean_prompt(). Feeding the host's
    # scaffolding back in is what made every turn match the same two facts.
    prompt = clean_prompt(raw_prompt)
    facts, problem = load_snapshot(snapshot_file())
    matches = match(prompt, facts) if facts else []
    context = build_context(matches)

    # Both lengths are logged: the gap between them is the host scaffolding, and
    # watching that gap is how the feedback loop was found. If raw_len keeps
    # growing while clean_len stays flat, the scaffolding is accumulating.
    # ``shape`` is here because lengths alone could not answer which field
    # actually carries the user's sentence; ``clean_sha`` is here because they
    # could not answer *what* that field contains either. A short hash of the
    # cleaned text lets a candidate string be identified by hashing it locally —
    # no content in the log, and unlike a length it is exact.
    log_run("prompt", str(payload.get("session_id") or ""),
            f"raw_len={len(raw_prompt)} clean_len={len(prompt)} "
            f"clean_sha={hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:12]} "
            f"facts={len(facts)} matched={len(matches)} "
            f"injected={len(context)} problem={problem or '-'} "
            f"{prompt_shape(raw_prompt)} "
            f"shape=[{payload_shape(payload)}]")

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
