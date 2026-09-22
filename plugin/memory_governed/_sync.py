"""Async write pipeline for the governed memory provider.

Write path: sync_turn() writes L3 synchronously (fast, <10ms),
queues L2 + extraction for background processing.

Reliability contract enforced here:
- L3 writes are thread-safe (single connection guarded by an RLock,
  ``check_same_thread=False``, ``busy_timeout``) and indexed, so the dedup
  lookup is an index SEARCH instead of a full table SCAN.
- Every L3 row is mirrored into ``messages_fts``; if that mirror cannot be
  written the divergence is reported through :mod:`._diag` (never silent).
- ``WriteQueue.stop()`` joins its worker and drains the queue with per-item
  error isolation, so queued turns are not silently lost at interpreter exit.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import queue
import re
import sqlite3
import threading
import time
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._config import GovernedMemoryConfig
from ._diag import log_data_loss, log_degraded, record_metric
from ._synthesize import _content_to_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# L2 provenance columns — ONE definition, three consumers
# ---------------------------------------------------------------------------
#: The L2 columns that carry *who/where* a row came from, as
#: ``(column name, pyarrow type constructor name)``.
#:
#: Single source of truth, and deliberately so. This is the shape of an
#: incident that already happened: ``project`` was added to the backfill list
#: but not to the create-table schema, so an *existing* table gained the column
#: while a *freshly created* one never would. No behaviour test can see that —
#: every test runs against a table that already has the column — and the whole
#: suite stayed green while the bug shipped. The three places that must agree
#: are:
#:
#: 1. the create-table schema (see :meth:`WriteQueue._init_l2`),
#: 2. the legacy backfill loop in the same method,
#: 3. the CLI's backfill (:func:`memory_cli.open_l2_table`), which writes the
#:    same rows through a different door.
#:
#: All three now iterate this constant. Types are named, not instantiated:
#: pyarrow is imported lazily so the CLI's read-only commands keep working on
#: an interpreter that has nothing installed.
L2_PROVENANCE_COLUMNS: Tuple[Tuple[str, str], ...] = (
    # L3 message rowid the fact was lifted from, for audit.
    ("source_rowid", "int64"),
    # Originating turn role, so the quality gate can be replayed against the
    # real rule instead of assuming "user" (see _extract_atomic_facts).
    ("role", "string"),
    # Writing agent identity — single-writer deployments never needed it, a
    # shared store cannot do without it.
    ("agent", "string"),
    # Owning project, or NULL for a fact that holds everywhere. Deliberately
    # left NULL by the local extractor: a fact lifted out of a conversation has
    # no reliable project attribution, and guessing one would be worse than
    # admitting we do not know. Callers that *do* know — an external agent
    # working inside a checkout — declare it via
    # ``memory_cli.py remember --project``.
    ("project", "string"),
)


def l2_provenance_fields() -> list:
    """The provenance columns as ``pa.Field`` objects, for a table schema.

    Kept as a function rather than a module constant because instantiating the
    fields requires pyarrow, which this module imports lazily.
    """
    import pyarrow as pa

    return [pa.field(name, getattr(pa, typ)()) for name, typ in L2_PROVENANCE_COLUMNS]


# ---------------------------------------------------------------------------
# Module-level helpers (shared by WriteQueue and L3Writer)
# ---------------------------------------------------------------------------

# Sentence splitting that does NOT shred decimals, versions or URLs.
#   - CJK terminators (。！？；) always end a sentence.
#   - ASCII terminators only end a sentence when followed by whitespace AND a
#     character that can start a new sentence (uppercase letter, digit, CJK).
#     "version 3.14" -> '.' followed by '1' (no whitespace)    -> keep
#     "example.com/docs" -> '.' followed by 'c' (no whitespace) -> keep
#     "Friday. The"  -> '.' + space + 'T'                      -> split
#   - Newlines always split.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？；])"
    r"|(?<=[.!?])(?=\s+[A-Z0-9\u4e00-\u9fff])"
    r"|\n+"
)

# Trailing terminators removed from a sentence. '?' / '？' are deliberately NOT
# stripped: `_looks_like_fact` uses them to reject questions.
_TRAILING_TERMINATOR_RE = re.compile(r"[.!。！？；]+\s*$")

# rowid_map keys are truncated to this length to bound memory.
_ROWID_KEY_MAX = 500

# Prefix length used for the "message was truncated" fallback match.
_ROWID_PREFIX_LEN = 200

# Columns L3Writer wants to mirror into messages_fts (in insertion order).
_FTS_COLUMNS = ("content", "role", "session_id", "timestamp")

# L2 DLQ file name (recoverable rows from the NULL-vector migration).
_L2_DLQ_NAME = "l2_null_vector_migration.jsonl"


def _strip_trailing_terminators(text: str) -> str:
    """Drop trailing '。！？；.!'-style terminators (keeps '?' for question detection)."""
    return _TRAILING_TERMINATOR_RE.sub("", text).strip()


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences without breaking decimals, URLs or versions.

    Examples:
        "We are shipping version 3.14 today. See https://example.com/docs for info."
            -> ["We are shipping version 3.14 today",
                "See https://example.com/docs for info"]
        "The p99 latency is 12.5 ms in production."
            -> ["The p99 latency is 12.5 ms in production"]
    """
    if not text:
        return []
    sentences: List[str] = []
    for raw in _SENTENCE_SPLIT_RE.split(text):
        piece = _strip_trailing_terminators(raw).strip()
        if piece:
            sentences.append(piece)
    return sentences


def _rowid_key(content: str) -> str:
    """Build a bounded-size rowid_map key."""
    return content[:_ROWID_KEY_MAX]


def _resolve_source_rowid(fact_content: str, rowid_map: dict) -> Optional[int]:
    """Resolve an extracted fact back to the L3 message rowid it came from.

    ``rowid_map`` is keyed by message content (whole messages) *and* by the
    sentences of those messages, while ``fact_content`` is a single sentence.
    Resolution order:

    1. exact match on the fact (O(1) — the common case);
    2. exact match on the truncated key;
    3. substring match: the fact is a sentence inside a longer message
       (O(facts x messages), acceptable for a single turn);
    4. prefix match for truncated message keys.

    Args:
        fact_content: The extracted fact text.
        rowid_map: Mapping of message content (or sentence) -> L3 rowid.

    Returns:
        The L3 ``messages.id`` the fact came from, or None if unresolvable.
    """
    if not fact_content or not rowid_map:
        return None

    ref = rowid_map.get(fact_content)
    if ref is not None:
        return ref

    ref = rowid_map.get(_rowid_key(fact_content))
    if ref is not None:
        return ref

    prefix = fact_content[:_ROWID_PREFIX_LEN]
    for msg_content, rowid in rowid_map.items():
        if not msg_content:
            continue
        if fact_content in msg_content:
            return rowid
        # Message key was truncated: fall back to a prefix comparison.
        if len(msg_content) >= _ROWID_KEY_MAX and msg_content.startswith(prefix):
            return rowid
    return None


def _weighted_len(text: str) -> int:
    """Length weighted for CJK density (a CJK char carries ~3x an ASCII char)."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return (len(text) - cjk) + cjk * 3


def _py(value: Any) -> Any:
    """Convert a pyarrow scalar to a plain Python value (passthrough otherwise)."""
    return value.as_py() if hasattr(value, "as_py") else value


# ---------------------------------------------------------------------------
# Fact-extraction heuristics
# ---------------------------------------------------------------------------

# Positive signals (used to rank facts when the per-turn cap bites).
_FACT_SIGNALS_ZH = (
    "决定", "确定", "采用", "使用", "选择", "改为", "方案", "原因", "结论",
    "约定", "规则", "注意", "记住", "以后", "计划", "打算", "目标是",
    "必须", "不能", "禁止", "偏好", "喜欢", "讨厌", "命名", "规范",
    # Moved here from the STRONG table 2026-09-17 (see the rationale on
    # `_FACT_SIGNALS_USER_ZH`). Ranking-only by construction: this list is read
    # solely in the `strong_only=False` branch, so the generic word list ranks
    # and never decides admission — the project invariant.
    "我的", "我在", "太麻烦",
)
_FACT_SIGNALS_EN = (
    "prefer", "always", "never", "decided", "decide", "will use", "we use",
    "we'll", "must", "should", "because", "requirement", "deadline",
    "convention", "rule", "remember", "from now on", "let's", "agreed",
    "the reason", "conclusion", "migrate", "deprecat",
)

# Named constants for the signal weights below. Keeping them explicit makes the
# ranking tunable without hunting magic numbers through the scoring function.
_USER_SIGNAL_WEIGHT = 2
_SOFT_SIGNAL_WEIGHT = 1

# First-person user statements that COMMIT to something — the core payload of a
# memory system. "我需要/必须/不能/否则" encode constraints and preferences that
# stay true across sessions, so they carry a full signal weight and may decide
# admission on their own.
#
# High-frequency first-person FUNCTION words moved OUT of this table
# (2026-09-17): 我的 / 我在 / 太麻烦. They are among the commonest tokens in
# Chinese prose, and once one token was worth the whole external threshold each
# became a skeleton key — "我的天哪" / "我在干嘛呢" / "太麻烦了" were admitted as
# facts. Same category error as the earlier "我不" bypass: an admission table
# must hold COMMITMENTS, not tokens that merely happen to be frequent. They now
# live in the ranking-only `_FACT_SIGNALS_ZH` list, where frequency is exactly
# what you want — they lift a sentence when the per-turn cap bites and decide
# nothing.
#
# Kept here after review: each still needs corroboration to pass the external
# gate (see the tiers below), but each asserts *whose* or *how* something is
# configured rather than merely that the speaker exists —
#    我想要 / 我一般 / 我家里 / 我装 / 我用的
# "我家里用的是软路由+策略服务A" is a deployment fact; "我在" (an aspect marker) is not.
_FACT_SIGNALS_USER_ZH = (
    "我需要", "我想要", "我一般", "我家里", "我装", "我用的",
    "必须", "不能", "否则",
)

# First-person MARKERS for the SHAPE rules (2026-09-17).
#
# `_FACT_SIGNALS_USER_ZH` used to serve two masters: the strong admission pool
# AND a "this line carries a first-person marker" test inside
# `WriteQueue._looks_like_fact` — used to spare a short CJK fragment from the
# heading rule ("记住我的偏好是深色主题") and to exempt a short assistant reply
# from the narrator-length rule. Those two jobs pull in opposite directions: the
# admission pool must hold COMMITMENTS only, while the shape test legitimately
# wants every first-person marker, including the function words just evicted.
#
# So the marker set is kept whole here — identical to the pool as it stood
# before 2026-09-17, i.e. `_FACT_SIGNALS_USER_ZH + the three evicted words`. It
# only ever loosens a *shape* check; it can never admit a row, so breadth costs
# nothing.
_USER_MARKERS = _FACT_SIGNALS_USER_ZH + ("我的", "我在", "太麻烦")

# Why "我不" is NOT in the pool above (removed 2026-09-16):
#
# It weighed 2 — exactly `_EXTERNAL_MIN_SIGNAL` — and the strong-signal branch
# imposes no length or structure requirement, so *any* sentence starting with it
# walked straight through the external write gate. Measured: "我不知道",
# "我不太确定", "我不这么认为" were all admitted. Those are epistemic hedges:
# they report the speaker's ignorance, not a fact about the world, and a row
# that says nothing is pure competition for recall slots.
#
# The other entries are all *commitment* shapes (我需要 / 我用的 / 必须…). "我不"
# was the only one that can be purely non-committal, which makes it a category
# error in a gate rather than a matter of tuning.
#
# A blanket minimum length on the strong branch was the alternative and was
# rejected: it would have to clear "我不这么认为" (weighted len 18) while keeping
# real short facts like "必须用 cosine" (weighted len 16), i.e. there is no
# length that separates them. Removing the token is targeted instead.
#
# Calibration (8 relevant / 8 irrelevant, tmp/calibrate_gate_negation.py):
#   before  relevant 7/8, irrelevant leaked 3/8
#   after   relevant 7/8, irrelevant leaked 0/8   (24 historical negatives: 0 leaked)

# Hedged / procedural wording. Real but weakly committed, so rank it below a
# hard user constraint. (Never blocks extraction on its own — ranking only.)
_FACT_SIGNALS_SOFT = ("可能", "建议", "试试", "或许", "大概", "一般来说")

# Technical conclusions — causal / diagnostic phrasing.
#
# Assistant turns split into two kinds that otherwise look identical to the
# scorer: "I just restarted the gateway" (a run log) and "the tool errored
# because `total` was undefined" (knowledge). The second kind is what makes
# an assistant turn worth remembering, so it gets a real weight.
_FACT_SIGNALS_TECH = (
    "因为", "是因为", "原因是", "根因", "根本原因", "解决办法", "导致", "报错是因为",
    "而不是", "应该用", "正确做法",
    "instead of", "because", "root cause", "workaround",
)
_TECH_SIGNAL_WEIGHT = 2

# Inline code identifiers (`` `like_this` ``) mark a sentence as being about
# concrete named entities rather than chatter. Weighted BELOW a real signal on
# purpose: a decorated run log ("备份在 `config.yaml.bak.x`") must still fail
# the floor.
_CODE_IDENT_RE = re.compile(r"`[^`\n]{2,60}`")
_CODE_IDENT_WEIGHT = 1

# Durable knowledge comes overwhelmingly from USER turns: preferences,
# constraints and requirements. Assistant turns mostly narrate execution
# ("让我检查一下…"), which is a run log rather than knowledge.
_ROLE_WEIGHTS: Dict[str, int] = {"user": 2, "assistant": -1}

# Admission floors for the STRONG-signal score (see `_fact_signal_score`).
# The score is a GATE, not merely a ranking hint.
#
# Why strong-only (P0, 2026-09-16): the first version of the role weighting
# only fed `facts.sort()`, so assistant narration (baseline -1) was admitted
# anyway — 79 of the day's 99 rows scored <= 0 under the assistant rule, e.g.
# "网关已重启，飞书在线", "备份在 config.yaml.bak.xxx".
#
# The second problem showed up immediately after wiring a naive total-score
# floor: the generic word list (`_FACT_SIGNALS_ZH`: 方案 / 以后 / 使用 / 注意…)
# is so easy to hit that bare chatter cleared the bar —
# "还没想好，我网上找找类似的方案" scored 3 (user baseline 2 + 方案 1).
# So the gate counts ONLY strong signals:
#   * user constraints   (_FACT_SIGNALS_USER_ZH: 我需要 / 必须 / 不能 / 否则…)
#   * technical conclusions (_FACT_SIGNALS_TECH: 是因为 / 而不是 / 根因…)
#   * inline code identifiers (`like_this`)
# The generic word list still contributes to RANKING, just not to admission.
#
# These are CONTENT floors (2026-09-17). They used to be expressed *relative to
# the role baseline* ("user baseline +2 -> must clear 2"), which silently made
# the baseline a piece of evidence: a user sentence carrying one generic word
# scored 2 + 2 = 4 and cleared the bar on identity alone. The role prior no
# longer reaches the gate (see `_fact_signal_score(include_role=...)`), so the
# floors are now read directly against the sentence's own signals:
#   * user      -> more than ONE strong unit. A bare generic word scores 2, so
#                  `> 2` demands real corroboration; a self-sufficient modal
#                  (我需要 / 必须 / 不能) is admitted separately as a rule in its
#                  own right. Equivalent to the old intent, minus the padding.
#   * assistant -> more than a lone decoration (floor 1). A run log decorated
#                  with one inline identifier scores 1 and must still fail —
#                  this is the old `content - 1 > 0` restated without the prior.
_MIN_SIGNAL_USER = 2
_MIN_SIGNAL_ASSISTANT = 1

#: Agent name recorded on rows this plugin writes itself. Hermes is the only
#: in-process writer; external agents go through ``memory_cli.py`` and carry
#: their own identity in the same column. Without this a shared store cannot
#: answer "who wrote this row", which single-writer deployments never needed.
_LOCAL_AGENT = "hermes"

# ---------------------------------------------------------------------------
# Floors for EXTERNAL writes (multi-agent sharing)
# ---------------------------------------------------------------------------
# An external agent submits a fact with NO dialogue context: the text is a
# knowledge claim, not a turn in a conversation. The role weights above
# therefore do not apply. Measured 2026-09-16 against 10 positive / 10 negative
# real samples, `role="user"` kept 10/10 but LEAKED 8/10 — its +2 baseline lifts
# every short acknowledgement ("嗯嗯", "好的，我明白了") over the bar. Empty role
# is the only honest choice when no dialogue context is known.
#
# The strong-signal word lists alone keep just 6/10: they are tuned for user
# constraints, so a noun-shaped fact scores ZERO. Measured examples that score 0
# yet are exactly what an agent is asked to remember:
#     "示例主路由 192.0.2.1 的 SSH 端口是 8022，走 PPPoE 拨号"
#     "家用NAS 192.0.2.62 上重度使用 Docker 部署服务"
#     "mosdns client_proxy_mode=whitelist 已启用"
# Structural evidence is the complement: an IP, a path, an inline identifier, a
# digit, a latin run. Combining the two:
#
#     admit  <=>  signal >= 2  OR  (weighted_len >= 40 AND structure >= 2)
#
# re-measured with the REAL _weighted_len(): 9/10 kept, 0/10 leaked. Separation
# is clean — no negative sample exceeds structure 1, and every positive that
# misses the word lists carries structure >= 3. The single miss ("我计划 55 岁
# 提前退休，需要筹备孩子的教育基金") is a life-plan preference, which belongs in
# L1/KB rather than the fact layer.
#
# Deliberately NOT screened here: credentials. They stay on the existing secret
# channel (matched_secret_patterns + quarantine) so the security event keeps its
# redacted audit record instead of collapsing into a bare counter.
_EXTERNAL_MIN_SIGNAL = 2
_EXTERNAL_MIN_WEIGHTED_LEN = 40
_EXTERNAL_MIN_STRUCTURE = 2

# ---------------------------------------------------------------------------
# Corroboration: one token is not evidence (2026-09-17)
# ---------------------------------------------------------------------------
# The rule used to be "weighted strong-signal sum >= _EXTERNAL_MIN_SIGNAL", and
# every pool entry weighs exactly that. So a SINGLE token admitted a row on its
# own, which turned tokens that commit to nothing into skeleton keys. Measured,
# all admitted:
#
#     我的天哪 / 我在干嘛呢 / 太麻烦了 / 我想要不还是算了
#     因为这样吧 / 可能是因为吧 / 应该用的是这个
#
# Same shape as the '我不' bypass: tokens that assert nothing when they stand
# alone ("我的" is a possessive particle, "我在" an aspect marker, "因为" a
# conjunction). Hunting them one at a time does not scale, because the defect is
# in the rule, not the list — ONE TOKEN IS NOT EVIDENCE.
#
# A writer with no dialogue context to vouch for it must therefore show two
# independent things: two strong signals, or one strong signal corroborated by a
# named thing. The exceptions are genuine modal commitments, which are a rule in
# their own right.
#
# Per-token disposition — why a tier split rather than deleting:
#
#   * SELF-SUFFICIENT (may admit alone)  我需要 / 必须 / 不能
#       "不能明文存密码" is a rule in seven characters; there is nothing else in
#       the sentence to corroborate it with, and requiring corroboration would
#       lose it.
#   * NEEDS CORROBORATION  我的 / 我在 / 我用的 / 我装 / 我家里 / 我一般 /
#                          我想要 / 太麻烦 / 否则
#       Possessive particles, aspect markers, moods and a conjunction. They say
#       *whose*, *where* or *how one feels*, and only become a fact once the
#       object is named ("我家里用的是软路由+策略服务A"). Kept in the pool because they
#       still carry real ranking value on dialogue turns — deleting them would
#       pay a cost there for no extra safety here.
#   * NEEDS CORROBORATION (tech)  因为 / 应该用
#       Conjunctions. "因为 X 所以 Y" always carries X and "应该用 A 而不是 B"
#       always carries B, so corroboration costs a real technical conclusion
#       nothing; "因为这样吧" and "应该用的是这个" carry nothing.
#
# The long-text branch gets the same treatment. Two structure slots used to be
# enough, but `digit` and a bare lowercase latin run are both trivially
# satisfied — "这个项目 2024 年吧，随便什么 whatever 都行" qualified on a year
# and one English word. At least one slot must now be NAMED evidence: an IP, a
# path, or an identifier.
#
# Calibration (8 relevant / 6 Chinese constraints / 8 payload / 4 structure-hole
# / 24 historical negatives, tmp/calibrate_gate_commitment.py):
#   before  relevant 7/8, constraints 6/6, payload leaked 8/8, structure-hole 3/4
#   after   relevant 7/8, constraints 6/6, payload leaked 0/8, structure-hole 0/4
_SELF_SUFFICIENT_SIGNALS = ("我需要", "必须", "不能")

# Polar interrogative frames that MENTION a modal without committing to it
# (2026-09-17). Chinese marks a yes/no question by embedding the modal in an
# A-不-A frame ("能不能" = "can, or cannot?") or with 否 ("能否" / "可否"). A bare
# substring match on _SELF_SUFFICIENT_SIGNALS therefore fired on the QUESTION
# rather than the commitment: "能不能帮我改一下配置" *contains* "不能" and was
# admitted as if it asserted a constraint. Measured, all ADMIT before the fix:
#     能不能帮我改一下配置 / 能不能做成免密 / 这个能不能行 / 我能不能先看看
# None of them asserts anything — each merely asks. Stripping the frame first
# restores the distinction the substring test threw away.
#
# Ordered longest-first so "可不可以" is consumed before the shorter prefix scan,
# and so no partial strip can leave a stray "不能" behind.
_INTERROGATIVE_MODAL_FORMS = (
    "可不可以", "能不能", "是否能", "行不行", "能否", "可否",
)


def _has_self_sufficient_modal(text: str) -> bool:
    """True when ``text`` ASSERTS a self-sufficient modal commitment.

    A modal commitment (我需要 / 必须 / 不能) is a rule in its own right — "不能
    明文存密码" is a rule in seven characters, with nothing else in the sentence
    to corroborate it. The same characters, however, also appear inside POLAR
    QUESTIONS, where they commit to nothing: "能不能帮我改一下配置" asks for
    permission, it does not state a constraint. A plain substring test cannot
    tell the two apart, so the question frames in
    :data:`_INTERROGATIVE_MODAL_FORMS` are removed before the modals are sought.

    A genuine constraint never sits inside one of those frames, so stripping is
    lossless on the positive side: "不能同时写入 — 同一时间只在一个实例操作"
    keeps its 不能 and is still admitted.

    Args:
        text: Candidate sentence text (already undecorated by the caller).

    Returns:
        True when a modal commitment survives the removal of any question frame.
    """
    if not text:
        return False
    asserted = text
    for form in _INTERROGATIVE_MODAL_FORMS:
        asserted = asserted.replace(form, "")
    return any(s in asserted for s in _SELF_SUFFICIENT_SIGNALS)


_EXTERNAL_IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_EXTERNAL_PATH_RE = re.compile(r"[A-Za-z]:[\\/]|/[\w.-]+/[\w.-]+|`[^`]*\.\w{1,4}`")
_EXTERNAL_IDENT_RE = re.compile(
    r"`[^`]+`|\b[A-Za-z_][A-Za-z0-9_]{2,}(?:[._-][A-Za-z0-9_]+)+\b"
)
_EXTERNAL_DIGIT_RE = re.compile(r"\d")
_EXTERNAL_LATIN_RE = re.compile(r"[A-Za-z]{2,}")

# Pure chatter: never a durable fact.
_GREETINGS = {
    "你好", "您好", "嗨", "哈喽", "谢谢", "多谢", "感谢", "好的", "好", "嗯",
    "嗯嗯", "收到", "明白", "ok", "okay", "hi", "hello", "hey", "thanks",
    "thank you", "thx", "yes", "no", "yep", "nope", "bye", "再见", "哈哈",
    "哈哈哈", "辛苦了", "没问题", "可以", "行",
}

# Imperative / request openers.
_COMMAND_PREFIXES = (
    "please", "pls", "do ", "don't", "dont", "can you", "could you",
    "would you", "帮我", "请", "不要", "别", "看一下", "看看", "试试",
    "试一下", "麻烦", "帮我看", "帮我改", "运行", "执行",
)

# Code / log lines are never durable knowledge.
_CODE_PREFIXES = (
    ">>>", "...", "$ ", "#!", "//", "/*", "```", "traceback", "select ",
    "insert ", "update ", "delete ", "create table", "def ", "class ",
    "import ", "return ", "if __name__", "http://", "https://",
    "file \"", "  file \"", "at 0x", "{", "}", ");",
)
_CODE_MARKERS = ("traceback (most recent call last)", ">>> ", "{", "}", ");")

# "from os import path" is code, but "From now on we use pnpm" is a durable
# preference — so import lines need a pattern, not a bare "from " prefix.
_IMPORT_LINE_RE = re.compile(
    r"^from\s+[\w.]+\s+import\s|^import\s+[\w.]+(\s*,\s*[\w.]+)*\s*$", re.IGNORECASE
)

# ---------------------------------------------------------------------------
# L2 quality gate
#
# L2 is a VECTOR store, so a junk row is worse than wasted disk: it competes
# for recall slots and wins them. Once a row is in L2 it is already past the
# Bridge review gate (which sits between L2 and L1), so nothing downstream can
# clean it up — this gate has to reject noise at extraction time.
#
# Categories rejected below (each observed in the real L2 corpus):
#   1. assistant process narration      "让我检查一下记忆系统的状态"
#   2. markdown structure               "============", "| 组件 | 状态 |", "## 方案一：…"
#   3. system-injected notices          "[System note: …]"
#   4. status-report metadata rows      "Provider: governed", "大小: 236KB"
#   5. self-check boilerplate           "记忆系统自检报告", "⏳ 待积累"
# ---------------------------------------------------------------------------

# Assistant narration about what it is about to do. A run log of the current
# turn, not knowledge worth carrying into the next one. ("让我", which the
# older `_COMMAND_PREFIXES` misses, is the single biggest offender.)
_ASSISTANT_PROCESS_PREFIXES = (
    "让我", "我来", "我先", "让我来", "让我先", "现在让我", "好的，让我",
    "接下来", "下面", "咱们", "来看一下", "来总结一下", "来测试一下",
    "我这就", "好，让我",
)

# Politeness / filler offers. Never durable knowledge.
_POLITE_MARKERS = (
    "有什么我能帮", "有什么可以帮", "能帮到你吗", "我可以帮你", "我可以为您",
    "要我帮你", "需要我帮你", "要不要我",
)

# Assistant outcome reports about THIS turn ("Bug 修复成功", "现在正常工作了").
# A result is a fact about one run, not durable knowledge.
_ASSISTANT_OUTCOME_MARKERS = (
    "修复成功", "修复完成", "正常工作了", "已正常工作", "已正常运行",
    "对接正常", "现在可以直接使用", "现在可以这样使用", "就不需要",
)

# Notices injected by the runtime rather than written by either participant.
_SYSTEM_NOISE_MARKERS = (
    "[system note", "system note", "operation interrupted",
    "gateway shutdown", "gateway is no longer running", "gateway is now back online",
    "session was restored", "has already run", "any restart/shutdown command",
)

# Self-check / health-report boilerplate (`governed_health` output was the
# single largest noise source in the corpus).
_SELF_CHECK_MARKERS = (
    "自检报告", "健康报告", "自检一下", "全面自检", "自检完毕", "状态报告",
    "待积累", "尚未创建", "尚未生成",
)

# Markdown structure. These carry formatting, not content.
_MD_RULE_RE = re.compile(r"^(?:={3,}|-{3,}|_{3,}|\*{3,}|~{3,})$")
_MD_TABLE_ROW_RE = re.compile(r"^\|.*\|$")
# NOTE: `\S.*` rather than a bare `\S+` — real headings contain spaces and
# emoji ("## 方案三：链接分享 + 密码", "## ✅ 对接状态"), which `\S+` misses.
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+\S.*$")
_MD_BOLD_LABEL_RE = re.compile(r"^\*\*[^*]{1,40}\*\*")
# Sections written without '#' ("L2 语义记忆 (向量搜索)").
_MD_PAREN_TITLE_RE = re.compile(r"^[^\n:：。！？]{1,30}[（(][^）)]{1,30}[）)]\s*$")
_MD_BRACKET_LABEL_RE = re.compile(r"^【[^】]{0,24}】")

# Tree listings and "path - count" directory enumerations.
_MD_TREE_RE = re.compile(r"^[├└│─\s]*[├└│]")
_DIR_COUNT_RE = re.compile(r"^[A-Za-z0-9_./\-]+\s+[-–—]\s+\S.{0,40}$")

# --- content-type hard rejects (2026-09-16 evening) ------------------------
# These three classes were absent from the first version of the gate and cost
# a day of L2 pollution (12 → 99 rows, ~2/3 noise).

# Multimodal placeholders: the text stand-in for an image/attachment, never
# knowledge. The short form `[Image]` was already caught by the length rule,
# but the LONG variant passes every other check because it looks like an
# ordinary sentence:
#     "[Image attached at: C:\Users\...\cache\images\img_c68df9eac1b1.jpg"
# 10 such rows observed in one day of traffic.
_MEDIA_PLACEHOLDER_RE = re.compile(
    r"^\[(?:image|img|screenshot|图片|截图|附件|file|photo|video|audio|voice)"
    r"[\s\]#._-]",
    re.IGNORECASE,
)

# Credentials (passwords / tokens / private keys). Hard reject, because L2 is
# a VECTOR store: a credential row gets recalled and injected straight into the
# model context, which is a far wider blast radius than the L3 archive (which
# is expected to hold raw conversation). Observed: one plaintext password
# (`Liu@****96`) extracted from a user message.
_CREDENTIAL_RE = re.compile(
    # "LETTERS<sep>digits" — the shape of most hand-made passwords
    r"(?<![A-Za-z0-9])[A-Za-z]{2,12}[@#$!]\d{6,}"
    # explicit key/value credential rows
    r"|(?:pass(?:word|wd)?|pwd|密码|口令|token|secret|密钥)\s*[:=：]\s*\S{3,}"
    # API tokens
    r"|\b(?:sk|ak|ghp|xoxb|glpat)-[A-Za-z0-9_\-]{16,}"
    # PEM private key blocks
    r"|BEGIN\s+(?:RSA|OPENSSH|EC|DSA|PGP)?\s*PRIVATE\s+KEY",
    re.IGNORECASE,
)

# A line that is nothing but a local absolute path (Windows drive / UNC). This
# is machine state, not durable knowledge, and it goes stale the moment the
# file moves ("C:\Users\...\plugins\model-providers\deepseek\__init__.py").
_ABS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)\S+$")


# ---------------------------------------------------------------------------
# Ephemeral (one-off, time-bound) intents — Bridge-only gate
# ---------------------------------------------------------------------------
# Why this exists and why it is NOT part of `_looks_like_fact`:
#
# L2 is a RECALL layer — a wrong row costs one bad hint. Bridge candidates,
# by contrast, get **promoted into L1**, which is injected into every single
# turn as the "rules" layer. A one-off todo that lands there becomes a
# permanent rule: the live candidate pool held
#     「我儿子的准考证，考试前一天记得提醒我」
# which, once promoted, means Hermes reminds about an exam forever — long
# after the exam ended. So ephemeral content must be stopped **before it
# enters the pool**, not filtered at promotion time.
#
# The rule is a conjunction (time expression AND reminder verb) minus a
# periodic-preference veto. That asymmetry matters: 「每次要输密码太麻烦」
# contains a reminder-ish shape but is a durable preference and must survive;
# 「考试前一天记得提醒我」 has no periodic marker and is genuinely one-off.
_EPHEMERAL_TIME_RE = re.compile(
    r"(明天|后天|大后天|今晚|今早|今天|当天|"
    r"本周[一二三四五六日天]|这周|这星期|这个星期|"
    r"下周[一二三四五六日天]?|下星期|下个星期|下个月|这个月|下季度|"
    r"\d+\s*(?:天|日|周|个?月|小时|分钟)(?:后|之后|以内|内)|"
    r"前(?:一天|两天|一晚|三天|一周)|"
    r"tomorrow|tonight|next\s+(?:week|month|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
    r"in\s+\d+\s+(?:days?|weeks?|months?)|"
    r"day\s+before)",
    re.IGNORECASE,
)

_EPHEMERAL_ACTION_RE = re.compile(
    r"(提醒我|记得提醒|别忘了|别忘记|到时候|记得|叫我|通知我|"
    r"remind\s+me|don'?t\s+forget|remember\s+to)",
    re.IGNORECASE,
)

#: 周期性/长期语汇 —— 它们描述的正是 L1 该记住的东西，构成否决票。
_DURABLE_PERIODIC_RE = re.compile(
    r"(每次|总是|以后|一直|永远|从来不|从不|每天|每周|每月|定期|习惯|"
    r"every\s+time|always|never|usually|from\s+now\s+on)",
    re.IGNORECASE,
)


def is_ephemeral_content(text: str) -> bool:
    """Detect one-off, time-bound intents that must never reach L1.

    A conjunction of a time expression and a reminder verb, vetoed by any
    periodic/durable marker. Returns False for anything ambiguous — the cost
    of letting a todo through is a stale permanent rule, but the cost of
    dropping a real preference is far higher.

    Examples:
        >>> is_ephemeral_content("我儿子的准考证，考试前一天记得提醒我")
        True
        >>> is_ephemeral_content("每次要输密码太麻烦了，能不能做成免密")
        False
    """
    s = (text or "").strip()
    if not s:
        return False
    if _DURABLE_PERIODIC_RE.search(s):
        return False
    return bool(_EPHEMERAL_TIME_RE.search(s) and _EPHEMERAL_ACTION_RE.search(s))


# ---------------------------------------------------------------------------
# Template skeletons — Bridge-only gate
# ---------------------------------------------------------------------------
# The vault/memory files ship with commented-out scaffolding. `source_record`
# in scripts/scope_recall_bridge.py treats a whole FILE as one candidate, so a
# file that was never edited becomes a candidate whose only content is the
# skeleton. Promoting it copies the scaffold into L1, and the next persona
# build reads L1 back into persona.md — a closed loop that keeps re-injecting
# the scaffold. Strip first, then judge on what is left.
#
# Two consumers, deliberately different in strictness:
#   * strip_template_fragments() — used by PRODUCERS, before a candidate exists;
#   * the predicates below — used by the GATE, as the last line before L1.
# A producer that forgets to strip is exactly what the gate must catch, so the
# gate cannot treat "there is real content after the scaffold" as a pass.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

#: 强痕迹：**只有脚手架才会出现**，正常用户陈述/知识笔记里不可能有。
#: 命中任意一条即判定「脚手架残留」—— 语句可以引用"user profile"这个词，
#: 但不会说出"直接编辑此文件"。
_TEMPLATE_MARKERS_STRONG = (
    "直接编辑此文件",
    "在此添加",
    "手写用户信息",
    "手写规则层",
    "_generated:",
    "your name here",
)

#: 弱痕迹：模板的 H1/章节标题。它们**同时可能出现在真实内容里**
#: （笔记标题、正文引用），所以只用来判「整条是否只剩骨架」，
#: 不参与「脚手架残留」的硬拦。
_TEMPLATE_MARKERS_WEAK = (
    "手写规则",
    # 模板的 H1 标题本身也算痕迹：USER.md 的 "# User Profile" 与
    # MEMORY.md 的 "# Memory Rules" 会被 persona 构建原样吞进去，
    # 只靠 "手写…" 那几行清不干净（实测 summary 里仍残留
    # "User Profile User User Profile 身份 偏好 当前项目"）。
    "user profile",
    "memory rules",
)

#: 全部痕迹（净化时使用；判据上按强弱分别处理）。
_TEMPLATE_MARKERS = _TEMPLATE_MARKERS_STRONG + _TEMPLATE_MARKERS_WEAK

#: 记忆系统**自身聚合产物**的指纹。persona.md 末尾会把「知识领域计数」
#: 「已知事实转储」「归档统计」整段渲染出来，这些是某一时刻的**派生快照**：
#: 一方面立刻过时，另一方面它们是内部诊断对话的搬运（实测混进了
#: "L1手写规则层空的" "刚才在修复 governed_health 的 bug 时网关重启了"
#: 这类过程噪声）。这类内容进 L1 没有任何价值，只会变成永久垃圾规则。
_MEMORY_AGGREGATE_MARKERS = (
    "conversations archived",
    "messages archived",
    "facts\n- ",
)

#: 「## Stats」这类统计小节标题（小写匹配）。
_MEMORY_AGGREGATE_HEADINGS = ("stats", "knowledge areas")

#: 净化后至少要有这么多实质字符，才不算「只是模板骨架」。
_TEMPLATE_MIN_SUBSTANTIVE_CHARS = 40


def strip_template_fragments(text: str) -> str:
    """去掉模板骨架：HTML 注释、生成时间戳行、使用说明行、**空章节标题**。

    最后一步是必要的：模板里的 ``## 身份`` / ``## 偏好`` 这些标题本身
    不在 ``_TEMPLATE_MARKERS`` 里（它们太通用，直接拉黑会误伤真实笔记），
    但它们**后面没有任何内容**，所以按「空章节」处理更准确 —— 一个有内容
    的 ``## 身份`` 会被保留，模板里的空壳则被丢掉。
    """
    if not text:
        return ""
    out = _HTML_COMMENT_RE.sub("", text)
    kept = []
    for line in out.splitlines():
        s = line.strip()
        if not s:
            kept.append(line)
            continue
        low = s.lower()
        if any(m in s or m in low for m in _TEMPLATE_MARKERS):
            continue
        kept.append(line)
    return _drop_empty_sections("\n".join(kept)).strip()


def _drop_empty_sections(text: str) -> str:
    """删掉「后面没有任何正文」的标题行（模板遗留的空章节）。"""
    lines = text.splitlines()
    out: List[str] = []
    for i, line in enumerate(lines):
        s = line.strip()
        if s and _MD_HEADING_RE.match(s):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines) or _MD_HEADING_RE.match(lines[j].strip()):
                continue  # 空章节标题，丢弃
        out.append(line)
    return "\n".join(out)


def is_template_placeholder(text: str) -> bool:
    """整条内容是否只是模板骨架（净化后没有实质内容）。

    先决条件：**必须含模板痕迹**（``_TEMPLATE_MARKERS``）。否则短句会被
    误判 —— 「都配吧，免得以后每次都弹窗」只有 13 个字符，任何"字符数
    下限"的判据都会把它当成空模板。

    满足先决条件后，再看「非标题行的有效字符数」：USER.md 模板净化后只剩
    若干个空标题（``## 身份`` / ``## 偏好`` …），实质字符数为 0，判为模板；
    而 persona.md 这类「真实内容 + 夹带模板片段」净化后仍有大量正文，放行。
    """
    raw = (text or "").strip()
    if not raw:
        return True
    low = raw.lower()
    if not any(m in raw or m in low for m in _TEMPLATE_MARKERS):
        return False
    stripped = strip_template_fragments(raw)
    if not stripped:
        return True
    substantive = 0
    for line in stripped.splitlines():
        s = line.strip()
        if not s or _MD_HEADING_RE.match(s):
            continue
        substantive += len(s)
    return substantive < _TEMPLATE_MIN_SUBSTANTIVE_CHARS


def has_template_scaffold(text: str) -> bool:
    """内容里是否**残留**脚手架文字（强痕迹）。

    与 :func:`is_template_placeholder` 的分工：

    - ``is_template_placeholder`` 问「这条**是不是只有**骨架」——用于生产端
      判断「剥掉骨架后还有没有东西」，所以夹带真实内容的放行；
    - 本函数问「骨架**还在不在**」——用于闸门（进 L1 前的最后一道）。

    为什么闸门要更严：生产端本该先 :func:`strip_template_fragments` 再建候选。
    如果到了闸门还能看到 ``直接编辑此文件`` / ``_Generated:`` 这种强痕迹，
    说明**上游某条路径漏了净化**（历史遗留行、新加的写入者、或插件不可用时的
    降级分支）。此时整条丢弃比"边净化边放行"更安全 —— 候选是给人复核的，
    丢掉可以手工补，而脚手架一旦进 L1 就会被下一轮 persona 构建读回去，
    形成闭环污染。
    """
    raw = (text or "")
    if not raw.strip():
        return False
    low = raw.lower()
    return any(m in raw or m in low for m in _TEMPLATE_MARKERS_STRONG)


def is_memory_aggregate(text: str) -> bool:
    """是否是记忆系统自身的聚合快照（persona 渲染产物 / 统计块）。

    实测 ``persona.md`` 被整文件当作一个 ``target=user`` 候选导出，内容里
    除了模板骨架，还跟着：

    - ``## Knowledge Areas`` → ``- other: 24 facts`` 这类计数
    - ``## Known Facts``     → 24 条 L2 事实的**转储**（含过程噪声）
    - ``## Stats``           → ``Conversations archived: 261``

    这些是 L2/L3 的**派生视图**，不是新信息；把它们 promote 进 L1 等于让
    派生数据回流成规则。真正的做法是在生产端拆成单条事实（见
    ``scripts/scope_recall_bridge.py`` 的 ``persona_fact_records``），
    走到闸门还整块出现就只能整条拒绝。
    """
    raw = (text or "")
    if not raw.strip():
        return False
    low = raw.lower()
    if any(m in low for m in _MEMORY_AGGREGATE_MARKERS):
        return True
    for line in raw.splitlines():
        s = line.strip()
        if not _MD_HEADING_RE.match(s):
            continue
        # 去掉前导 '#' 取标题正文（_MD_HEADING_RE 无捕获组）
        heading = s.lstrip("#").strip().lower()
        if heading in _MEMORY_AGGREGATE_HEADINGS:
            return True
    return False

# Metadata "key: value" rows out of status reports. Matched against the
# UNDECORATED copy so that "- **Provider**: `governed`", "**Provider**: …"
# and "Provider: …" all hit the same rule. Deliberately a closed list: a
# generic short-key rule would also eat prose headings such as
# "方案二：文件夹权限控制".
_REPORT_KV_KEYS = (
    # English keys
    "provider", "status", "state", "size", "path", "storage", "store", "dir",
    "embedding", "reranking", "reranker", "file", "files", "content", "total",
    "total files", "version", "available", "latency", "cache", "mode",
    "index", "vector index", "kb", "bridge", "write queue", "queue", "model",
    "dim", "timeout", "ttl", "decay", "health", "component",
    # Chinese keys
    "状态", "大小", "路径", "存储", "内容", "文件", "总文件", "总行数",
    "索引", "向量索引", "索引状态", "时间衰减", "可用", "延迟", "版本",
    "模式", "容量", "地址", "组件", "健康",
)
_REPORT_KV_RE = re.compile(
    r"^(?:"
    + "|".join(re.escape(k) for k in sorted(_REPORT_KV_KEYS, key=len, reverse=True))
    + r")\s*[:：]",
    re.IGNORECASE,
)

# "L1（手写规则层）: …" — a parenthesis-labelled metadata row.
_PAREN_LABEL_KV_RE = re.compile(
    r"^[^:：（）()]{0,24}[（(][^）)]{1,30}[）)][^:：]{0,16}[:：]"
)

# Leading decoration that must be removed BEFORE prefix/key checks: bullets,
# ordinal list marks and status symbols ("- ⚠️ **Provider**: governed").
_LEADING_BULLET_RE = re.compile(r"^(?:[-*•>]|\d+[.)、])\s+")
_LEADING_SYMBOL_RE = re.compile(r"^[^\w\u4e00-\u9fff*`(【\[\"'“”]+")
_BOLD_STRIP_RE = re.compile(r"\*\*([^*]{1,200}?)\*\*")

# An assistant sentence this short cannot carry durable knowledge unless it
# echoes a user constraint. Only applied when the role is known.
_ASSISTANT_MIN_WEIGHTED_LEN = 24

# Section titles written as bare fragments ("项目规则和行为约束", "记忆系统怎么样").
# A CJK-only run with no punctuation and no user voice is a heading, not a
# sentence. English is exempt: its spaces are word separators, so it has no
# equivalent of this shape.
_TITLE_FRAGMENT_MAX_WEIGHT = 36
_TITLE_FRAGMENT_RE = re.compile(r"^\S+$")

# Assistant lead-in lines that end on a colon ("来看一下当前记忆系统的状态：").
_COLON_LEADIN_PREFIXES = (
    "以下是", "如下", "来看一下", "来总结一下", "总结一下", "可以有几个",
    "但有几个", "有几个方案", "可以这样配合", "现在可以直接", "现在可以这样",
)

# Max times _undecorate() peels one layer of leading decoration.
_UNDECORATE_PEELS = 4


def _undecorate(text: str) -> str:
    """Peel leading bullets, list marks and status symbols off a sentence.

    ``"- ⚠️ **Provider**: governed"`` -> ``"**Provider**: governed"``.
    Reference-style structure needs to survive (``**`` is preserved) because
    the bold-label rule matches on it, so only symbols that are NOT part of a
    markdown token are stripped.

    Args:
        text: Raw sentence text.

    Returns:
        The sentence with leading decoration removed.
    """
    body = text.strip()
    for _ in range(_UNDECORATE_PEELS):
        candidate = _LEADING_BULLET_RE.sub("", body, count=1).lstrip()
        candidate = _LEADING_SYMBOL_RE.sub("", candidate, count=1)
        if candidate == body:
            break
        body = candidate
    return body


def _fact_signal_score(sentence: str, role: str = "",
                       *, strong_only: bool = False,
                       include_role: bool = True) -> int:
    """Score how much durable value a sentence carries.

    Higher is better. Two consumers:

    * ``strong_only=False`` (default) — a RANKING key, used to decide which
      facts survive when the per-turn cap in
      :meth:`WriteQueue._extract_atomic_facts` bites.
    * ``strong_only=True`` — the ADMISSION score, compared against
      :data:`_MIN_SIGNAL_USER` / :data:`_MIN_SIGNAL_ASSISTANT`. Only the
      strong pools count (user constraints, technical conclusions, inline code
      identifiers); the generic word list is excluded because it is far too
      easy to hit to serve as a gate.

    Role prior vs. evidence — why ``include_role`` exists (2026-09-17)
    ---------------------------------------------------------------
    ``_ROLE_WEIGHTS["user"]`` is ``+2``, exactly ``_MIN_SIGNAL_USER``. When the
    same call feeds both the ranking key AND the admission gate, that prior is
    silently promoted into EVIDENCE: a sentence carrying one strong signal
    scores ``2 + 2 = 4 > 2`` and passes, so a single word decides admission
    because of WHO said it, not WHAT was said. Measured, a bare
    "我的天哪" / "我想要不还是算了" cleared the gate on the role padding alone.

    Admission callers therefore pass ``include_role=False`` so the gate sees the
    content score only — one piece of evidence is not evidence. Ranking callers
    keep the default ``True``: there the prior is doing its legitimate job
    (prefer user turns when the per-turn cap bites), and a prior belongs in a
    sort key.

    Args:
        sentence: Candidate fact text.
        role: Role of the originating message (``"user"`` / ``"assistant"``).
            User turns state the preferences and constraints this system
            exists to remember, so they are weighted UP; assistant turns are
            mostly narration about how the request was fulfilled, so they are
            weighted DOWN. Unknown/empty roles are neutral.
        strong_only: Count only strong signals (see above).
        include_role: Add the role prior to the score. Leave ``True`` for
            ranking; pass ``False`` for admission (see above).

    Returns:
        An integer that may be negative.
    """
    lowered = sentence.lower()
    score = 0
    if not strong_only:
        for signal in _FACT_SIGNALS_EN:
            if signal in lowered:
                score += 1
        for signal in _FACT_SIGNALS_ZH:
            if signal in sentence:
                score += 1
        for signal in _FACT_SIGNALS_SOFT:
            if signal in sentence:
                score -= _SOFT_SIGNAL_WEIGHT
    for signal in _FACT_SIGNALS_USER_ZH:
        if signal in sentence:
            score += _USER_SIGNAL_WEIGHT
    for signal in _FACT_SIGNALS_TECH:
        if signal in sentence or signal in lowered:
            score += _TECH_SIGNAL_WEIGHT
            break
    if _CODE_IDENT_RE.search(sentence):
        score += _CODE_IDENT_WEIGHT
    if include_role:
        score += _ROLE_WEIGHTS.get(role, 0)
    return score


def dialogue_fact_admits(text: str, role: str = "") -> bool:
    """The dialogue-path admission gate — ONE definition, THREE consumers.

    Consumers:
      * :meth:`WriteQueue._extract_atomic_facts` — the write path (into L2);
      * ``scripts/l2_apply_gate.py`` — the replay of already-written rows;
      * :meth:`GovernedMemoryProvider._extract_session_candidates` — the Bridge
        candidate collector (its output is promoted into **L1**).

    Writes, replays and the Bridge *must* use the same ruler, so the rule lives
    here instead of in any caller. The Bridge site was the last hold-out: it
    hand-rolled ``_fact_signal_score(..., strong_only=True) <= _MIN_SIGNAL_USER``
    (2026-09-17), which — unlike this gate — kept the role prior AND never
    resolved interrogative frames, so every ``能不能…`` question was admitted as a
    standing rule. The caller still owns the shape checks
    (:meth:`WriteQueue._looks_like_fact` / ``_passes_structural_filter``); this
    only scores and thresholds.

    Rule (2026-09-17): a self-sufficient modal commitment (我需要 / 必须 / 不能)
    may be admitted alone — it is a rule in its own right. Otherwise the
    sentence's own CONTENT must clear the floor for its role, or present ONE
    strong unit CORROBORATED by a named/structural thing. The role prior is
    deliberately absent: it ranks (see ``_fact_signal_score``'s ``include_role``)
    but never decides admission.

    The corroboration clause is the external gate's rule restated for dialogue
    text, and it is load-bearing: without it, removing the role prior would have
    deleted a whole family of real one-signal facts —
    "我们决定用 PostgreSQL 因为它更稳定" (one conclusion signal + one identifier),
    "我家里用的是虚拟机软路由…" (one marker + named hardware), and every
    "…because…" conclusion. Those carry exactly one strong unit; the old gate let
    them in on the +2 role baseline, so the baseline was doing the corroboration's
    job. Now the sentence has to.

    Args:
        text: Candidate sentence, already known to look like a fact.
        role: ``"user"`` / ``"assistant"``; anything else is treated as user.

    Returns:
        True when the sentence may enter L2.
    """
    if _has_self_sufficient_modal(text):
        return True
    floor = _MIN_SIGNAL_ASSISTANT if role == "assistant" else _MIN_SIGNAL_USER
    score = _fact_signal_score(text, role, strong_only=True, include_role=False)
    if score > floor:
        return True
    # One strong unit (score == _USER_SIGNAL_WEIGHT) is not evidence by itself;
    # one unit PLUS a named/structural thing is. `score >= _USER_SIGNAL_WEIGHT`
    # is implied by the floor for assistants, so this only ever adds the
    # one-unit case for users — it never loosens the assistant rule.
    if score >= _USER_SIGNAL_WEIGHT and external_structural_evidence(text) >= 1:
        return True
    return False


def external_structural_evidence(text: str) -> int:
    """Count independent structural markers that a factual statement carries.

    Complements :func:`_fact_signal_score` for external writes, whose word lists
    do not recognise noun-shaped facts (see the floors block above). Each marker
    is independent evidence that the text *names* something concrete rather than
    expressing a mood or a request.
    """
    if not text:
        return 0
    return sum((
        bool(_EXTERNAL_IP_RE.search(text)),
        bool(_EXTERNAL_PATH_RE.search(text)),
        bool(_EXTERNAL_IDENT_RE.search(text)),
        bool(_EXTERNAL_DIGIT_RE.search(text)),
        bool(_EXTERNAL_LATIN_RE.search(text)),
    ))


def external_named_evidence(text: str) -> bool:
    """True when ``text`` names something concrete: an IP, a path, an identifier.

    The three markers here are the ones that cannot be hit by accident, unlike
    the ``digit`` and ``latin`` slots above — a year and a lowercase English word
    are both satisfied by "2024 年吧，随便什么 whatever 都行". Requiring one of
    *these* is what stops the long-text branch from admitting chatter.
    """
    if not text:
        return False
    return any(rx.search(text) for rx in
               (_EXTERNAL_IP_RE, _EXTERNAL_PATH_RE, _EXTERNAL_IDENT_RE))


def external_signal_pools(text: str) -> int:
    """Count the independent strong-signal POOLS ``text`` hits (0..2).

    One per pool, mirroring :func:`_fact_signal_score`'s ``break`` semantics for
    the technical list: the question the gate asks is "how many *independent*
    kinds of evidence are present", and two synonyms from the same list are one
    kind. Counting matches instead would let "因为…所以…" style repetition stand
    in for corroboration.
    """
    if not text:
        return 0
    lowered = text.lower()
    pools = 1 if any(s in text for s in _FACT_SIGNALS_USER_ZH) else 0
    pools += 1 if any(s in text or s in lowered for s in _FACT_SIGNALS_TECH) else 0
    return pools


def external_write_verdict(text: str) -> Tuple[bool, str]:
    """Decide whether an EXTERNAL agent may write ``text`` into L2.

    The gate is deliberately *looser* than :func:`screen_bridge_content`: Bridge
    candidates are promoted into L1, where a wrong row becomes a standing rule
    injected every turn, whereas L2 is only a recall layer where a wrong row
    costs one misleading hint. It is still far stricter than the word-list-only
    floor, because an external writer has no dialogue context to vouch for it.

    Admission (2026-09-17 onward) — see :data:`_SELF_SUFFICIENT_SIGNALS`::

        admit  <=>  self-sufficient modal signal (我需要 / 必须 / 不能)
                OR  two strong-signal pools
                OR  one strong-signal pool corroborated by a named thing
                OR  weighted_len >= 40 AND struct >= 2 AND named evidence

    The middle two clauses exist because the previous rule admitted on a single
    token's weight, which made "我的天哪" as good as a fact.

    Returns:
        ``(admitted, reason)``. ``reason`` is ``"ok"`` when admitted; otherwise a
        short machine-readable tag (``empty`` / ``ephemeral`` / ``template`` /
        ``media`` / ``abs_path`` / ``weak_signal``) that the caller **must**
        surface to the user. A silent drop here would look identical to success.
    """
    raw = (text or "").strip()
    if not raw:
        return False, "empty"

    cleaned = _undecorate(raw)

    # Same content-shape checks the Bridge gate applies. Kept in sync by living
    # in this module, which is the single definition site for all of them.
    if is_ephemeral_content(cleaned):
        return False, "ephemeral"
    if is_template_placeholder(cleaned):
        return False, "template"
    if _MEDIA_PLACEHOLDER_RE.match(cleaned):
        return False, "media"
    if _ABS_PATH_RE.match(cleaned):
        return False, "abs_path"

    # --- admission: one token is not evidence (see _SELF_SUFFICIENT_SIGNALS) --
    # A genuine modal commitment is a rule in its own right. Checked through
    # `_has_self_sufficient_modal` rather than a bare `in`, so that a POLAR
    # QUESTION ("能不能…") is not mistaken for the constraint it quotes.
    if _has_self_sufficient_modal(cleaned):
        return True, "ok"

    # Otherwise the text must present _EXTERNAL_MIN_SIGNAL worth of evidence
    # TWICE OVER: either two independent pools, or one pool corroborated by a
    # named thing. `pools * _EXTERNAL_MIN_SIGNAL` is what one pool is worth, so
    # a single pool alone no longer clears the bar.
    pools = external_signal_pools(cleaned)
    if pools * _EXTERNAL_MIN_SIGNAL >= _EXTERNAL_MIN_SIGNAL * 2:
        return True, "ok"
    if (pools * _EXTERNAL_MIN_SIGNAL >= _EXTERNAL_MIN_SIGNAL
            and external_structural_evidence(cleaned) >= 1):
        return True, "ok"

    # Long + structured, but `digit` and a bare latin run are both trivially
    # satisfied, so at least one slot must be a NAMED thing.
    if (_weighted_len(cleaned) >= _EXTERNAL_MIN_WEIGHTED_LEN
            and external_structural_evidence(cleaned) >= _EXTERNAL_MIN_STRUCTURE
            and external_named_evidence(cleaned)):
        return True, "ok"

    return False, "weak_signal"


# ---------------------------------------------------------------------------
# Write queue
# ---------------------------------------------------------------------------

# Sentinel pushed into the queue to wake a worker sleeping in get().
_STOP_SENTINEL = object()

# Extra grace period after `stop(timeout)` expires before declaring the
# worker stuck (an item finishing exactly at the deadline should not be
# reported as a failure).
_JOIN_GRACE_SECONDS = 1.0

# Live queues tracked for the atexit best-effort flush (weakly referenced so
# that short-lived test queues are still garbage collected).
_LIVE_QUEUES: "weakref.WeakSet[WriteQueue]" = weakref.WeakSet()
_LIVE_QUEUES_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _flush_all_queues() -> None:
    """Best-effort flush at interpreter exit (daemon workers would be killed)."""
    with _LIVE_QUEUES_LOCK:
        pending = list(_LIVE_QUEUES)
    for q in pending:
        try:
            q.stop(timeout=1.0)
        except Exception as e:  # noqa: BLE001 - atexit must never raise
            logger.debug("atexit write-queue flush failed: %s", e)


def _register_exit_flush(instance: "WriteQueue") -> None:
    """Track ``instance`` for the atexit flush (handler registered once)."""
    global _ATEXIT_REGISTERED
    with _LIVE_QUEUES_LOCK:
        _LIVE_QUEUES.add(instance)
        if not _ATEXIT_REGISTERED:
            atexit.register(_flush_all_queues)
            _ATEXIT_REGISTERED = True


class WriteQueue:
    """Background worker that processes L2 indexing and fact extraction.

    L3 writes are done synchronously in sync_turn() for safety.
    L2 + extraction go through this queue for non-blocking writes.
    """

    def __init__(self, config: GovernedMemoryConfig):
        self._config = config
        self._queue: queue.Queue = queue.Queue(maxsize=config.sync.write_queue_maxsize)
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._l2_store = None
        # Both are ALWAYS defined: `_index_l2` reads `_embed_fn` and an
        # AttributeError there was swallowed by a bare `except Exception`.
        self._embed_model = None   # EmbeddingService instance (or None)
        self._embed_fn = None      # callable(text) -> list[float] | None
        self._stats_lock = threading.Lock()
        self._stats = {"processed": 0, "errors": 0}

    def start(self) -> None:
        """Start the background writer thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="memory-write-worker",
        )
        self._worker_thread.start()
        _register_exit_flush(self)

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the worker, drain remaining items and join the thread.

        Draining is error-isolated: an item that raises must not discard the
        rest of the queue (it used to propagate out of stop() and kill the
        remaining turns). The worker thread is then joined so that an
        in-flight item is finished before the interpreter can kill a daemon
        thread.

        Args:
            timeout: Seconds to wait for the worker thread to finish. A short
                grace period is added for an item finishing right at the
                deadline.
        """
        self._running = False

        # Drain: bounded so a pathological producer cannot block shutdown
        # forever, and one item's failure never aborts the drain.
        # (The stop sentinel is pushed AFTER the drain, otherwise the drain
        # would consume it and the worker would sleep out its 1s poll.)
        max_items = max(int(getattr(self._config.sync, "write_queue_maxsize", 100) or 100), 1)
        drained = 0
        while drained < max_items:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _STOP_SENTINEL:
                continue
            try:
                self._process_item(item)
                with self._stats_lock:
                    self._stats["processed"] += 1
            except Exception as e:  # noqa: BLE001 - isolation is the point
                with self._stats_lock:
                    self._stats["errors"] += 1
                logger.error("Write queue drain error: %s", e)
            finally:
                drained += 1

        # Wake a worker sleeping in get(timeout=1.0) so the join is immediate.
        try:
            self._queue.put_nowait(_STOP_SENTINEL)
        except queue.Full:
            pass

        thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Grace period: the item may be finishing exactly at the
                # deadline; only report if it is genuinely stuck.
                thread.join(timeout=_JOIN_GRACE_SECONDS)
                if thread.is_alive():
                    log_degraded(
                        "write_queue",
                        "stop_join_timeout",
                        detail=f"worker still alive after {timeout}s + "
                               f"{_JOIN_GRACE_SECONDS}s grace",
                    )

    def enqueue(self, messages: List[Dict[str, Any]], session_id: str = "",
                rowid_map: Optional[dict] = None,
                provenance_missing: bool = False) -> None:
        """Queue messages for background L2 indexing + extraction.

        Args:
            messages: The turn's messages.
            session_id: Session the messages belong to.
            rowid_map: content -> L3 rowid map returned by :class:`L3Writer`.
            provenance_missing: True when the L3 write failed, so the facts
                still get indexed but carry no ``source_rowid``. The flag is
                propagated to the queue item for observability.
        """
        item = {
            "messages": messages,
            "session_id": session_id,
            "rowid_map": rowid_map or {},
            "provenance_missing": provenance_missing,
            "timestamp": time.time(),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            log_degraded(
                "write_queue",
                "queue_full_dropped_oldest",
                detail=f"maxsize={self._config.sync.write_queue_maxsize}",
            )
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                log_data_loss(
                    "write_queue",
                    "enqueue_dropped",
                    detail=f"session_id={session_id!r} messages={len(messages)}",
                )

    def _worker_loop(self) -> None:
        """Background worker that processes write items."""
        while self._running:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is _STOP_SENTINEL:
                break
            try:
                self._process_item(item)
                with self._stats_lock:
                    self._stats["processed"] += 1
            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001 - worker must not die
                with self._stats_lock:
                    self._stats["errors"] += 1
                logger.error("Write worker error: %s", e)

    def _process_item(self, item: dict) -> None:
        """Process a single write item: L2 indexing + extraction."""
        messages = item.get("messages", [])
        if not messages:
            return

        # L2: Extract atomic facts and index
        if self._config.sync.l2_write == "async":
            self._index_l2(messages, item.get("rowid_map") or {})

        # Tencent extraction (if enabled)
        if self._config.tencent_extract.enabled:
            self._extract_facts(messages)

    def _index_l2(self, messages: List[Dict[str, Any]], rowid_map: Optional[dict] = None) -> None:
        """Index messages into L2 LanceDB with embeddings and L3 references."""
        try:
            if self._l2_store is None:
                self._init_l2()
            if self._l2_store is None:
                return

            # Extract facts from the latest turn
            facts = self._extract_atomic_facts(messages)
            if not facts:
                return

            # Deduplicate (vs. the store + within the batch) BEFORE embedding:
            # a long-lived session re-feeds its history every turn, so without
            # this the same sentences were re-embedded and re-appended (~19x
            # redundancy observed in production).
            facts = self._drop_existing_facts(facts)
            if not facts:
                return

            # Attach L3 source reference (rowid) for audit drill-down.
            # Facts are SENTENCES while rowid_map is keyed by MESSAGE, so the
            # lookup must fall back to substring matching.
            has_ref_col = "source_rowid" in [f.name for f in self._l2_store.schema]
            if has_ref_col:
                rmap = rowid_map or {}
                for fact in facts:
                    ref = _resolve_source_rowid(fact["content"], rmap)
                    if ref is not None:
                        fact["source_rowid"] = ref

            self._attach_vectors(facts)

            # Dimension consistency check: verify vector dimensions match
            # the table schema. This prevents writing 512-dim vectors from
            # a local backend into a 1024-dim table (API backend) or vice versa.
            expected_dim = self._expected_vector_dim()
            if expected_dim > 0:
                dim_mismatch_count = 0
                for fact in facts:
                    vector = fact.get("vector")
                    if vector is not None and len(vector) != expected_dim:
                        dim_mismatch_count += 1
                        # Remove mismatched vector, keep the fact (text-only)
                        del fact["vector"]
                if dim_mismatch_count > 0:
                    log_data_loss(
                        "l2_write",
                        "vector_dim_mismatch",
                        detail=(
                            f"{dim_mismatch_count} facts had vectors with wrong "
                            f"dimension (expected {expected_dim}). "
                            "Vectors removed, facts stored as text-only."
                        ),
                    )

            # Project onto the table's real columns. `facts` also carries
            # internal keys (`_signals`), and `agent` is absent on a table that
            # predates the provenance column — filtering here makes the write
            # safe whether or not the migration has run yet.
            cols = {f.name for f in self._l2_store.schema}
            facts = [{k: v for k, v in f.items() if k in cols} for f in facts]

            self._l2_store.add(facts)
        except Exception as e:  # noqa: BLE001 - L2 failure must not kill the turn
            log_degraded("l2_write", "index_failed", exc=e)

    def _expected_vector_dim(self) -> int:
        """Return the expected vector dimension from the L2 table schema.

        Returns 0 if the table has no vector column or cannot be inspected.
        This is used to prevent dimension mismatches when the embedding backend
        changes (e.g., API 1024-dim → local 512-dim fallback).
        """
        store = self._l2_store
        if store is None:
            return 0
        try:
            for field in store.schema:
                if field.name == "vector":
                    # field.type is fixed_size_list<item: float>[N]
                    # Extract N from the type string
                    type_str = str(field.type)
                    if "fixed_size_list" in type_str:
                        # Parse "fixed_size_list<item: float>[1024]"
                        import re
                        match = re.search(r'\[(\d+)\]', type_str)
                        if match:
                            return int(match.group(1))
            return 0
        except Exception:
            return 0

    def _existing_contents(self) -> set:
        """Return the set of ``content`` values already present in L2.

        Reads ONLY the content column where possible: pulling the 1024-dim
        vector column for thousands of rows is needlessly expensive. Best-effort
        by contract — a failure here degrades to an empty set, which can only
        cause a duplicate re-add, never a loss.

        STRATEGY ORDER MATTERS (see STRATEGY 1 comment): pyarrow is a hard
        dependency of lancedb, so the Arrow projection scan must be tried first.
        The pandas-based strategies are kept only as later fallbacks for
        environments that happen to have pandas installed.
        """
        store = self._l2_store
        if store is None:
            return set()

        # STRATEGY 1 (must be first): pyarrow is a HARD dependency of lancedb,
        # so an Arrow scan is the only path guaranteed to exist on every
        # deployment. A scan query with `.select(["content"])` projects ONLY the
        # content column, so the 1024-dim vector column is never materialised.
        # `limit()` must be >= row count (the default is 10).
        #
        # LESSON LEARNED (2026-09): the original ordering tried `to_pandas()`
        # first. pandas is NOT a lancedb hard dependency — the hermes deploy
        # runtime (`hermes-agent/venv`) ships lancedb + pyarrow + numpy but NO
        # pandas and NO pylance. Every pandas-based strategy therefore raised
        # ModuleNotFoundError there, `_existing_contents()` returned an empty
        # set, and store-level L2 dedup was silently inert (every turn re-added
        # the same facts). The project dev venv happened to have pandas, so the
        # tests passed and the bug escaped to production. Arrow must come first.
        try:
            n = store.count_rows()
            if n == 0:
                return set()
            table = store.search().select(["content"]).limit(n).to_arrow()
            return {str(c) for c in table.column("content").to_pylist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through to the next strategy
            pass

        # STRATEGY 2: full-table Arrow scan. No column projection, so it drags
        # the whole 1024-dim vector column along — used only when the projected
        # scan above is unavailable. Still pyarrow-only, hence still dependency-free.
        try:
            table = store.to_arrow()
            return {str(c) for c in table.column("content").to_pylist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through
            pass

        # STRATEGY 3: pandas-based projection over a scan query. Works ONLY
        # where pandas is installed (project dev venv) — NOT the deploy runtime.
        try:
            n = store.count_rows()
            if n == 0:
                return set()
            df = store.search().select(["content"]).limit(n).to_pandas()
            return {str(c) for c in df["content"].tolist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through to the next strategy
            pass

        # STRATEGY 4: some LanceDB builds expose column projection on to_pandas().
        try:
            df = store.to_pandas(columns=["content"])
            return {str(c) for c in df["content"].tolist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through
            pass

        # STRATEGY 5: older LanceDB builds project through the Lance dataset
        # (needs pylance, which is usually absent).
        try:
            table = store.to_lance().to_table(columns=["content"])
            return {str(c) for c in table.column("content").to_pylist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through
            pass

        # STRATEGY 6: in-memory doubles (e.g. test stores) expose their rows
        # directly via a plain list attribute.
        rows = getattr(store, "rows", None)
        if isinstance(rows, list):
            return {
                str(r["content"]) for r in rows
                if isinstance(r, dict) and r.get("content") is not None
            }

        logger.debug("L2 dedup: could not read existing content; skipping store-level dedup")
        return set()

    def _drop_existing_facts(self, facts: List[dict]) -> List[dict]:
        """Drop facts whose content is already in L2, or repeated in this batch.

        Two layers, both required:

        - store-level: a re-fed long session re-extracts the same sentences
          every turn — without this they are re-embedded and re-appended;
        - batch-level: the same sentence can be extracted twice within one
          batch (duplicate messages in the turn).

        First-seen order is preserved. Call BEFORE ``_attach_vectors`` so no
        embedding work is spent on rows that will be dropped.
        """
        if not facts:
            return facts
        seen = self._existing_contents()
        kept: List[dict] = []
        dropped = 0
        for fact in facts:
            key = str(fact.get("content", ""))
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(fact)
        if dropped:
            record_metric("facts_deduped", dropped)
            logger.info("L2 dedup: dropped %d/%d fact(s) already present",
                        dropped, len(facts))
        return kept

    def _attach_vectors(self, facts: List[dict]) -> None:
        """Fill ``fact["vector"]`` where an embedding backend produced one.

        ``embed_batch`` may return None for individual texts; those facts are
        stored without a vector rather than being dropped (LanceDB keeps the
        row searchable via the text fallback).
        """
        if not facts:
            return
        texts = [f["content"] for f in facts]
        vectors = self._embed_texts(texts)
        if not vectors:
            return
        for fact, vector in zip(facts, vectors):
            if vector:
                fact["vector"] = vector

    def _embed_texts(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed ``texts``, returning a list ALIGNED with the input.

        ``None`` marks a text that could not be embedded. Returns an empty list
        when no backend is available.
        """
        if not texts:
            return []

        svc = self._embed_model
        # Preferred path: the shared EmbeddingService singleton.
        if svc is not None and hasattr(svc, "embed_batch"):
            try:
                results = svc.embed_batch(texts) or []
                return [list(v) if v else None for v in results]
            except Exception as e:  # noqa: BLE001 - degrade, never crash the turn
                log_degraded("l2_embedding", "embed_batch_failed", exc=e)
                return []

        # Legacy backend objects exposing .encode() (numpy-style).
        if svc is not None and hasattr(svc, "encode"):
            try:
                return [list(v) for v in svc.encode(texts).tolist()]
            except Exception as e:  # noqa: BLE001
                log_degraded("l2_embedding", "encode_failed", exc=e)
                return []

        # API-style single-text embedding function.
        if self._embed_fn is not None:
            out: List[Optional[List[float]]] = []
            for text in texts:
                try:
                    vector = self._embed_fn(text)
                except Exception as e:  # noqa: BLE001
                    log_degraded("l2_embedding", "embed_one_failed", exc=e)
                    vector = None
                out.append(list(vector) if vector else None)
            return out

        return []

    def _init_l2(self) -> None:
        """Lazy-init LanceDB for writing (vector backend via EmbeddingService)."""
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as e:
            log_degraded("l2_write", "lancedb_missing", exc=e)
            return

        # 先解析 embedding 后端：探测会用实际维度覆盖配置（API 场景覆盖
        # embedding.dimensions、本地场景覆盖 vector.dim）。建表维度取实际
        # 探测值，否则 API 场景会拿 config.vector.dim（本地默认 512）建表
        # 而嵌入向量是另一维度 → 写入错配。
        self._init_embedding_service()

        try:
            db_path = self._config.l2_db_path
            Path(db_path).mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(db_path)
            tables = db.list_tables().tables
            if "memories" in tables:
                self._l2_store = db.open_table("memories")
            else:
                # Create with proper PyArrow schema INCLUDING vector column.
                # 维度取实际探测值（service.dim）；API 场景下 config.vector.dim
                # 是本地模型默认值（512），与 API 实际维度无关。
                dim = getattr(self._embed_model, "dim", 0) or self._config.vector.dim
                # Provenance columns come from L2_PROVENANCE_COLUMNS so the
                # schema and the legacy backfill cannot drift apart (see the
                # comment there for the incident that motivated it).
                schema = pa.schema([
                    pa.field("content", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("timestamp", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    *l2_provenance_fields(),
                ])
                self._l2_store = db.create_table("memories", schema=schema)

            # Backfill legacy columns (best-effort). `role` carries the
            # originating turn's role so the quality gate can be replayed with
            # the real rule instead of assuming "user" (see _extract_atomic_facts).
            # `agent` carries who wrote the row, which single-writer deployments
            # never needed but a shared store cannot do without.
            #
            # NOTE the argument form: lancedb's add_columns takes a pa.Field /
            # pa.Schema (new columns initialised to null) or a {name: SQL}
            # mapping — NOT a {name: pa.array} mapping. The array form raises
            # TypeError, and because this whole block is best-effort the failure
            # was swallowed into a debug line, so the migration silently never
            # ran. Existing rows correctly stay NULL: "written before provenance
            # was tracked" is the truth, and guessing would be worse.
            try:
                col_names = [f.name for f in self._l2_store.schema]
                for col, typ in L2_PROVENANCE_COLUMNS:
                    if col not in col_names:
                        self._l2_store.add_columns(pa.field(col, getattr(pa, typ)()))
                        logger.info("L2 legacy table backfilled with %s column", col)
            except Exception as e:  # noqa: BLE001
                log_degraded("l2_write", "column_backfill_failed", exc=e)
        except Exception as e:  # noqa: BLE001
            log_degraded("l2_write", "store_init_failed", exc=e)
            return

        # 建表/开表后，为之前无向量的旧行补向量（此时 _l2_store 已就绪）
        self._migrate_null_vectors()

    def _init_embedding_service(self) -> None:
        """Resolve the embedding backend through the shared EmbeddingService.

        The selection ladder (API → fastembed → sentence-transformers) lives in
        :class:`._embedding.EmbeddingService` so the read and write paths share
        exactly one model instance (and one download).
        """
        self._embed_model = None
        self._embed_fn = None

        try:
            from ._embedding import EmbeddingService
        except ImportError as e:
            log_degraded("l2_embedding", "service_import_failed", exc=e)
            return

        try:
            service = EmbeddingService.get(self._config)
        except Exception as e:  # noqa: BLE001
            log_degraded("l2_embedding", "service_init_failed", exc=e)
            return

        if not getattr(service, "available", False):
            log_degraded(
                "l2_embedding",
                "backend_unavailable",
                detail=str(getattr(service, "last_error", "") or ""),
            )
            return

        self._embed_model = service
        self._embed_fn = service.embed_one
        logger.info("L2 embedding service loaded for writing (dim=%s)",
                    getattr(service, "dim", self._config.vector.dim))

    def _dlq_path(self) -> Path:
        """Directory-less-safe path of the L2 dead-letter file."""
        l2_dir = Path(self._config.l2_db_path)
        return l2_dir.parent / "dlq" / _L2_DLQ_NAME

    def _write_dlq(self, rows: List[dict], reason: str) -> Optional[Path]:
        """Append rows to the L2 DLQ so a failed migration stays recoverable."""
        if not rows:
            return None
        path = self._dlq_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(
                        {"ts": time.time(), "reason": reason, "row": row},
                        ensure_ascii=False,
                    ) + "\n")
            return path
        except Exception as e:  # noqa: BLE001
            log_data_loss("l2_write", "dlq_write_failed", detail=str(path), exc=e)
            return None

    def _migrate_null_vectors(self) -> None:
        """Re-embed existing rows where the vector column is NULL.

        When the embedding model first becomes available, older rows
        (inserted without a vector column or with nulls) need vectors
        so that vector search can find them.

        Strategy: read all rows, identify NULL-vector rows, re-embed
        them, then delete + re-add (LanceDB has no in-place vector
        update via SQL).

        LanceDB offers no transaction across delete+add, so this is
        best-effort: rows are written to the DLQ BEFORE deletion, and a
        failure is reported via :func:`._diag.log_data_loss` instead of
        being swallowed.
        """
        if self._l2_store is None or self._embed_model is None:
            return
        try:
            table = self._l2_store.to_arrow()
            if table.num_rows == 0:
                return

            # Find rows with null vector
            vec_col = table.column("vector")
            content_col = table.column("content")
            null_indices = [
                i for i in range(table.num_rows)
                if _py(vec_col[i]) is None
            ]
            if not null_indices:
                return

            logger.info("L2 migration: re-embedding %d rows with NULL vectors", len(null_indices))

            # Collect content for re-embedding
            contents = [str(_py(content_col[i])) for i in null_indices]
            vectors = self._embed_texts(contents)
            if not any(vectors):
                log_degraded("l2_write", "null_vector_migration_no_vectors",
                             detail=f"rows={len(null_indices)}")
                return

            # Build new rows with vectors filled (source_rowid preserved).
            category_col = table.column("category")
            source_col = table.column("source")
            ts_col = table.column("timestamp")
            has_ref = "source_rowid" in table.column_names
            ref_col = table.column("source_rowid") if has_ref else None

            new_rows = []
            for idx, vector in zip(null_indices, vectors):
                if not vector:
                    continue
                ref_value = _py(ref_col[idx]) if ref_col is not None else None
                row = {
                    "content": str(_py(content_col[idx])),
                    "category": str(_py(category_col[idx]) or ""),
                    "source": str(_py(source_col[idx]) or ""),
                    "timestamp": str(_py(ts_col[idx]) or ""),
                    "vector": vector,
                    # Provenance must survive the migration (P2-1).
                    "source_rowid": int(ref_value) if ref_value is not None else None,
                }
                new_rows.append(row)

            if not new_rows:
                return

            # Persist a recoverable copy BEFORE deleting anything.
            dlq_path = self._write_dlq(new_rows, "null_vector_migration")

            # Delete NULL-vector rows then re-add with vectors.
            try:
                rowids = []
                rid_col = table.column("rowid") if "rowid" in table.column_names else None
                if rid_col is not None:
                    # _py(): a pyarrow Scalar's str() is its repr, which would
                    # produce invalid SQL ("rowid IN (<pyarrow.Int64Scalar: 3>)").
                    rowids = [_py(rid_col[i]) for i in null_indices]

                if rowids:
                    id_list = ", ".join(str(r) for r in rowids)
                    self._l2_store.delete(f"rowid IN ({id_list})")
                else:
                    # Fallback: delete by content match
                    for content in contents:
                        safe = content.replace("'", "''")
                        self._l2_store.delete(f"content = '{safe}'")

                self._l2_store.add(new_rows)
                logger.info("L2 migration complete: %d vectors filled", len(new_rows))
            except Exception as e:  # noqa: BLE001
                log_data_loss(
                    "l2_write",
                    "null_vector_migration_failed",
                    detail=f"rows={len(new_rows)} dlq={dlq_path}",
                    exc=e,
                )
        except Exception as e:  # noqa: BLE001
            log_degraded("l2_write", "null_vector_migration_skipped", exc=e)

    def _resolve_source_rowid(self, fact_content: str, rowid_map: dict,
                              messages: Optional[List[Dict[str, Any]]] = None) -> Optional[int]:
        """Resolve a fact back to its L3 rowid (instance wrapper).

        Delegates to the module-level :func:`_resolve_source_rowid`. ``messages``
        is optional and used as a last-resort fallback: locate the message that
        contains the fact and map that message to its rowid.
        """
        ref = _resolve_source_rowid(fact_content, rowid_map)
        if ref is not None or not messages:
            return ref

        for msg in messages:
            # Normalise first: a multimodal (list) message would make
            # `fact_content in content` raise instead of matching.
            content = _content_to_text(msg.get("content"))
            if content and fact_content in content:
                ref = rowid_map.get(_rowid_key(content))
                if ref is not None:
                    return ref
        return None

    def _extract_atomic_facts(self, messages: List[Dict[str, Any]]) -> List[dict]:
        """Extract atomic facts from messages (rule-based, no LLM)."""
        facts = []
        for msg in messages:
            role = msg.get("role", "")
            # Normalise multimodal content (list of OpenAI parts) to text so that
            # sentence splitting never sees a `list` (which raised
            # "expected string or bytes-like object, got 'list'").
            content = _content_to_text(msg.get("content"))
            if not content or role not in ("user", "assistant"):
                continue

            # Simple extraction: sentences that look like facts
            for sentence in self._split_sentences(content):
                # Pass the role so the gate can be stricter on assistant
                # narration, and so ranking prefers user-stated constraints.
                if not self._looks_like_fact(sentence, role):
                    continue
                # The score is a GATE before it is a ranking key: a sentence
                # that carries no durable signal must not enter L2 at all
                # (assistant run logs flooded the store — see
                # _MIN_SIGNAL_ASSISTANT for the measurements). The gate itself
                # lives in `dialogue_fact_admits`, shared verbatim with the
                # replay script so the write path and the replay cannot drift.
                if not dialogue_fact_admits(sentence, role):
                    continue
                facts.append({
                    "content": sentence.strip(),
                    "category": self._categorize(sentence),
                    "source": "auto-extract",
                    "timestamp": datetime.now().isoformat(),
                    # 落库真实角色（2026-09-16）：质量门回放只能按真实 role
                    # 复算分数，而过去 L2 表里根本没有这个字段，只好一律按
                    # user 判定（偏松，会把该拦的助手叙述放过去）。
                    "role": role,
                    # 写入者身份（2026-09-16 多 agent 共享）：共享存储必须能回答
                    # 「这行是谁写的」，单写者时代不需要这个字段。
                    "agent": _LOCAL_AGENT,
                    "_signals": _fact_signal_score(sentence, role),
                })

        # Ranking: keep the strongest facts when the per-turn cap bites.
        # (Stable sort — equal-scored facts keep their original order.)
        facts.sort(key=lambda f: f["_signals"], reverse=True)

        capped = facts[:self._config.sync.l2_max_facts_per_turn]
        for fact in capped:
            del fact["_signals"]

        # Observability: how much do we actually extract per turn?
        # Feeds the threshold tuning planned for next week.
        record_metric("fact_turns", 1)
        record_metric("facts_extracted", len(capped))
        record_metric("facts_dropped_by_cap", max(len(facts) - len(capped), 0))
        logger.info("facts_extracted=%d (of %d candidates, cap=%d)",
                    len(capped), len(facts), self._config.sync.l2_max_facts_per_turn)

        return capped

    def _extract_facts(self, messages: List[Dict[str, Any]]) -> None:
        """Tencent-style LLM extraction (optional, async)."""
        if not self._config.tencent_extract.enabled:
            return
        if not self._config.tencent_extract.llm_api_key:
            return

        # TODO: Implement Tencent L0→L1 extraction via LLM
        # This would call the configured LLM to extract atomic facts
        # and write them to the Bridge for review
        logger.debug("Tencent extraction not yet implemented")

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences (see module-level :func:`_split_sentences`)."""
        return _split_sentences(text)

    @staticmethod
    def _looks_like_fact(sentence: str, role: str = "") -> bool:
        """Heuristic: does this sentence look like a durable fact?

        Rejects content that is never durable knowledge: questions, greetings,
        imperatives, code/log lines, fragments that are too short or too long,
        markdown structure, system-injected notices, status-report metadata
        rows, self-check boilerplate and assistant process narration.

        Anything that survives is still *lenient* on purpose — admission is
        cheap, but every admitted row becomes a permanent competitor in L2
        vector space, which is why the noise classes above are hard rejects.

        Args:
            sentence: Candidate sentence text.
            role: Optional originating message role. When it is ``"assistant"``
                an extra minimum-length rule applies, because an assistant's
                short replies ("Bug 修复成功", "现在可以直接使用：") are outcome
                reports rather than knowledge.

        Returns:
            True when the sentence should be extracted into L2.
        """
        text = (sentence or "").strip()
        if not text:
            return False

        # Questions: '?' is preserved by _split_sentences on purpose.
        if text.endswith("?") or text.endswith("？"):
            return False

        # Skip very short or very long
        if len(text) < 6 or len(text) > 500 or _weighted_len(text) < 6:
            return False

        lowered = text.lower().rstrip(".!。！；;")

        # --- 1.5 content-type hard rejects ------------------------------
        # Checked before everything else: none of these can ever become a
        # durable fact, and each one is cheap to test.
        if _MEDIA_PLACEHOLDER_RE.match(text):
            return False
        if _CREDENTIAL_RE.search(text):
            return False
        if _ABS_PATH_RE.match(text):
            return False

        # Pure greetings / acknowledgements
        if lowered in _GREETINGS:
            return False

        # Commands / imperatives
        if lowered.startswith(_COMMAND_PREFIXES):
            return False

        # Code or log lines
        if lowered.startswith(_CODE_PREFIXES):
            return False
        if any(marker in lowered for marker in _CODE_MARKERS):
            return False
        if _IMPORT_LINE_RE.search(text):
            return False

        # --- 1. markdown structure -------------------------------------
        if _MD_RULE_RE.match(text):
            return False
        if _MD_TABLE_ROW_RE.match(text):
            return False
        if _MD_HEADING_RE.match(text):
            return False
        if _MD_BRACKET_LABEL_RE.match(text):
            return False
        if _MD_TREE_RE.match(text) or _DIR_COUNT_RE.match(text):
            return False
        if _MD_PAREN_TITLE_RE.match(text):
            return False
        # Bare CJK heading fragments ("项目规则和行为约束", "记忆系统怎么样").
        # A colon means the line already has a value half ("方案二：文件夹权限
        # 控制"), so those are exempt — they are judged by the KV rule instead.
        if (any("\u4e00" <= ch <= "\u9fff" for ch in text)
                and _TITLE_FRAGMENT_RE.match(text)
                and ":" not in text and "：" not in text
                and _weighted_len(text) < _TITLE_FRAGMENT_MAX_WEIGHT
                and not any(marker in text for marker in _USER_MARKERS)):
            return False

        # --- 2. system noise / politeness / outcome reports -------------
        if any(marker in text for marker in _SELF_CHECK_MARKERS):
            return False
        if any(marker in lowered for marker in _SYSTEM_NOISE_MARKERS):
            return False
        if any(marker in text for marker in _POLITE_MARKERS):
            return False
        if any(marker in text for marker in _ASSISTANT_OUTCOME_MARKERS):
            return False

        # Everything from here on operates on the undecorated body so that
        # bullets and status symbols cannot smuggle a row past the rules
        # ("- ⚠️ **Provider**: governed" is still a metadata row).
        undecorated = _undecorate(text)

        # --- 3. assistant process narration -----------------------------
        if undecorated.lower().startswith(_ASSISTANT_PROCESS_PREFIXES):
            return False

        # --- 4. markdown bold used as a structural label ----------------
        # "**✅ 正常的部分：**" is a label; "**必须同步 rsa.key**，否则…" keeps a
        # real body past the label and therefore survives.
        bold_match = _MD_BOLD_LABEL_RE.match(undecorated)
        if bold_match:
            body = undecorated[bold_match.end():].strip().strip("：:，-–—")
            if not body or text.endswith(("：", ":")) or _fact_signal_score(text) == 0:
                return False
        elif text.startswith("*"):
            # Bold decoration with no real content ("** 现在可以这样使用：").
            return False

        # --- 5. status-report metadata rows -----------------------------
        plain = _BOLD_STRIP_RE.sub(r"\1", undecorated).strip()
        if _REPORT_KV_RE.match(plain) or _PAREN_LABEL_KV_RE.match(plain):
            return False

        # --- 6. lead-in lines ending on a colon -------------------------
        # "来看一下当前记忆系统的状态：" introduces content it does not contain;
        # "不管哪个方案，Hermes 都可以通过 Bitwarden CLI 或 API 读取：" states it.
        if text.endswith(("：", ":")):
            if _fact_signal_score(text) == 0:
                return False
            if undecorated.startswith(_COLON_LEADIN_PREFIXES):
                return False

        # --- 7. assistant outcome reports -------------------------------
        if (role == "assistant"
                and _weighted_len(text) < _ASSISTANT_MIN_WEIGHTED_LEN
                and not any(marker in text for marker in _USER_MARKERS)):
            return False

        return True

    @staticmethod
    def _categorize(sentence: str) -> str:
        """Simple category assignment."""
        lower = sentence.lower()
        tech_kw = ["python", "docker", "linux", "api", "code", "database", "server",
                    "git", "script", "代码", "函数", "配置", "部署"]
        work_kw = ["project", "deadline", "meeting", "task", "项目", "需求", "进度"]
        life_kw = ["health", "exercise", "travel", "run", "park", "walk",
                    "健康", "运动", "旅行", "跑步"]

        if any(kw in lower for kw in tech_kw):
            return "tech"
        if any(kw in lower for kw in work_kw):
            return "work"
        if any(kw in lower for kw in life_kw):
            return "life"
        return "other"

    @property
    def stats(self) -> dict:
        with self._stats_lock:
            return dict(self._stats)

    @property
    def pending(self) -> int:
        """Approximate number of items still queued."""
        return self._queue.qsize()


class L3Writer:
    """Synchronous L3 SQLite writer. Fast (<10ms) due to WAL mode + indexes.

    Thread-safe: one connection shared by all callers, guarded by an RLock,
    created with ``check_same_thread=False`` and a 30s busy timeout. Without
    this, a second thread either raises
    ``ProgrammingError: SQLite objects created in a thread...`` or fails with
    ``database is locked``.
    """

    def __init__(self, config: GovernedMemoryConfig):
        self._config = config
        self._conn: Optional[sqlite3.Connection] = None
        self._conn_lock = threading.RLock()
        # Columns actually present in messages_fts (None until resolved).
        self._fts_cols: Optional[List[str]] = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the shared SQLite connection (WAL + indexes + FTS)."""
        with self._conn_lock:
            if self._conn is not None:
                return self._conn

            db_path = self._config.l3_db_path
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False: one connection shared across threads.
            # timeout: wait instead of raising "database is locked".
            self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
            self._conn.execute("PRAGMA journal_mode=WAL")
            # WAL + NORMAL: durable across process crashes, may lose only the
            # last few transactions on an OS/power failure (accepted trade-off
            # for the write path).
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp REAL,
                    metadata TEXT,
                    hash TEXT
                )
            """)
            # Create FTS5 table if not exists
            try:
                self._conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                    USING fts5(content, role, session_id, timestamp)
                """)
            except Exception as e:  # noqa: BLE001 - FTS5 is optional at build time
                log_degraded("l3_fts", "fts_table_unavailable", detail=db_path, exc=e)
            # Legacy tables: backfill hash column for idempotent writes
            try:
                self._conn.execute("ALTER TABLE messages ADD COLUMN hash TEXT")
            except Exception:  # noqa: BLE001 - column already exists
                pass
            # Dedup + retention indexes (new databases never run migration 003).
            self._ensure_indexes(self._conn)
            self._conn.commit()

            # Resolve the FTS layout once per connection.
            self._fts_cols = self._detect_fts_columns(self._conn)
            return self._conn

    @staticmethod
    def _ensure_indexes(conn: sqlite3.Connection) -> None:
        """Create the dedup + retention indexes (idempotent).

        Without ``idx_messages_dedup`` the per-message lookup
        ``SELECT 1 FROM messages WHERE session_id = ? AND hash = ?`` is a full
        table scan (877 ms for a 50-message turn at 80k rows; 2.6 ms with the
        index — 337x).
        """
        try:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_dedup ON messages (session_id, hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages (timestamp)"
            )
        except Exception as e:  # noqa: BLE001
            log_degraded("l3_write", "index_create_failed", exc=e)

    @staticmethod
    def _detect_fts_columns(conn: sqlite3.Connection) -> List[str]:
        """Return the columns of ``messages_fts``, or [] when FTS is unusable.

        Legacy databases can carry an FTS table with fewer columns than we
        write (e.g. no ``timestamp``). Writing the full 4-tuple into it fails
        for every row, which silently diverges the index from the main table —
        so the layout is detected and the insert is adapted to it.
        """
        try:
            rows = conn.execute("PRAGMA table_info(messages_fts)").fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [r[1] for r in rows]

    def _fts_insert_columns(self, conn: sqlite3.Connection) -> Optional[List[str]]:
        """Columns to write into messages_fts (None = FTS unavailable)."""
        if self._fts_cols is None:
            self._fts_cols = self._detect_fts_columns(conn)
        cols = self._fts_cols
        if not cols:
            log_degraded(
                "l3_fts",
                "fts_unavailable",
                detail="messages_fts missing or unreadable — L3 full-text search disabled",
            )
            return None

        usable = [c for c in _FTS_COLUMNS if c in cols]
        if "content" not in usable:
            log_data_loss(
                "l3_fts",
                "fts_schema_incompatible",
                detail=f"messages_fts columns={cols} — no 'content' column to index",
            )
            return None

        if len(usable) != len(_FTS_COLUMNS):
            # Recoverable: we still index the row, just with fewer columns.
            logger.warning(
                "L3 FTS schema mismatch: messages_fts has columns %s (expected %s) "
                "— indexing with %s. Run scripts/l3_reindex_fts.py to rebuild.",
                cols, list(_FTS_COLUMNS), usable,
            )
        return usable

    def write(self, messages: List[Dict[str, Any]], session_id: str = "") -> dict:
        """Write messages to L3 (and mirror them into FTS).

        Synchronous, thread-safe, idempotent per (session_id, content hash).

        Returns:
            Mapping of message content (and each of its sentences) to the L3
            rowid, used by the L2 writer to attach ``source_rowid``.
        """
        rowid_map: dict = {}
        if not messages:
            return rowid_map

        with self._conn_lock:
            conn = self._get_conn()
            now = time.time()
            fts_cols = self._fts_insert_columns(conn)

            try:
                for msg in messages:
                    role = msg.get("role", "")
                    # hermes transcripts may carry multimodal content (a list of
                    # OpenAI parts). Normalise to text BEFORE hashing/splitting,
                    # otherwise `str + list` raises TypeError and the whole batch
                    # is rolled back (real data loss).
                    content = _content_to_text(msg.get("content"))
                    if not content:
                        continue

                    # Idempotent writes: skip if (session_id, content_hash) is archived
                    content_hash = hashlib.sha256(
                        (session_id + "|" + role + "|" + content).encode("utf-8")
                    ).hexdigest()
                    exists = conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? AND hash = ?",
                        (session_id, content_hash),
                    ).fetchone()
                    if exists:
                        continue

                    # Insert into main table
                    cur = conn.execute(
                        "INSERT INTO messages (session_id, role, content, timestamp, hash) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (session_id, role, content, now, content_hash),
                    )
                    rowid = cur.lastrowid

                    # Key by the whole message AND by each sentence: extracted
                    # facts are sentences, so this makes L2 -> L3 drill-down an
                    # O(1) lookup in the common case.
                    rowid_map[_rowid_key(content)] = rowid
                    for sentence in _split_sentences(content):
                        rowid_map.setdefault(_rowid_key(sentence), rowid)

                    self._write_fts(conn, fts_cols, content, role, session_id, now)
            except Exception:
                try:
                    conn.rollback()
                except Exception:  # noqa: BLE001
                    pass
                raise

            conn.commit()
        return rowid_map

    def _write_fts(self, conn: sqlite3.Connection, fts_cols: Optional[List[str]],
                   content: str, role: str, session_id: str, now: float) -> None:
        """Mirror one row into messages_fts; report (never swallow) failures.

        A row that lands in ``messages`` but not in ``messages_fts`` is
        archived yet permanently unsearchable — that is data loss, not a
        debug-level curiosity.
        """
        if not fts_cols:
            return

        values = {
            "content": content,
            "role": role,
            "session_id": session_id,
            "timestamp": now,
        }
        sql = "INSERT INTO messages_fts ({}) VALUES ({})".format(
            ", ".join(fts_cols), ", ".join("?" * len(fts_cols))
        )
        try:
            conn.execute(sql, [values[c] for c in fts_cols])
        except Exception as e:  # noqa: BLE001
            log_data_loss(
                "l3_fts",
                "fts_insert_failed",
                detail=f"session_id={session_id!r} content={content[:50]!r}",
                exc=e,
            )

    def shutdown(self) -> None:
        with self._conn_lock:
            if self._conn:
                self._conn.close()
                self._conn = None
                self._fts_cols = None
