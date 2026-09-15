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
from typing import Any, Dict, List, Optional

from ._config import GovernedMemoryConfig
from ._diag import log_data_loss, log_degraded, record_metric
from ._synthesize import _content_to_text

logger = logging.getLogger(__name__)

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
)
_FACT_SIGNALS_EN = (
    "prefer", "always", "never", "decided", "decide", "will use", "we use",
    "we'll", "must", "should", "because", "requirement", "deadline",
    "convention", "rule", "remember", "from now on", "let's", "agreed",
    "the reason", "conclusion", "migrate", "deprecat",
)

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


def _fact_signal_score(sentence: str) -> int:
    """Count how many preference/decision signals a sentence contains."""
    lowered = sentence.lower()
    score = 0
    for signal in _FACT_SIGNALS_EN:
        if signal in lowered:
            score += 1
    for signal in _FACT_SIGNALS_ZH:
        if signal in sentence:
            score += 1
    return score


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

            self._l2_store.add(facts)
        except Exception as e:  # noqa: BLE001 - L2 failure must not kill the turn
            log_degraded("l2_write", "index_failed", exc=e)

    def _existing_contents(self) -> set:
        """Return the set of ``content`` values already present in L2.

        Reads ONLY the content column: pulling the 1024-dim vector column for
        thousands of rows is needlessly expensive. Best-effort by contract — a
        failure here degrades to an empty set, which can only cause a duplicate
        re-add, never a loss.
        """
        store = self._l2_store
        if store is None:
            return set()

        # Preferred (lancedb >= 0.17): a scan query with `.select(["content"])`
        # projects ONLY the content column, so the 1024-dim vector column is
        # never materialised. `limit()` must be >= row count (the default is 10).
        try:
            n = store.count_rows()
            if n == 0:
                return set()
            df = store.search().select(["content"]).limit(n).to_pandas()
            return {str(c) for c in df["content"].tolist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through to the next strategy
            pass

        # Some LanceDB builds expose column projection on to_pandas().
        try:
            df = store.to_pandas(columns=["content"])
            return {str(c) for c in df["content"].tolist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through
            pass

        # Older LanceDB builds: project through the Lance dataset (needs pylance).
        try:
            table = store.to_lance().to_table(columns=["content"])
            return {str(c) for c in table.column("content").to_pylist() if c is not None}
        except Exception:  # noqa: BLE001 - fall through
            pass

        # In-memory doubles (e.g. test stores) expose their rows directly.
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
                schema = pa.schema([
                    pa.field("content", pa.string()),
                    pa.field("category", pa.string()),
                    pa.field("source", pa.string()),
                    pa.field("timestamp", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                    pa.field("source_rowid", pa.int64()),  # L3 message rowid for audit
                ])
                self._l2_store = db.create_table("memories", schema=schema)

            # Backfill source_rowid column on legacy tables (best-effort)
            try:
                col_names = [f.name for f in self._l2_store.schema]
                if "source_rowid" not in col_names:
                    self._l2_store.add_columns({"source_rowid": pa.array([], type=pa.int64())})
                    logger.info("L2 legacy table backfilled with source_rowid column")
            except Exception as e:  # noqa: BLE001
                logger.debug("L2 source_rowid column backfill skipped: %s", e)
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
                if self._looks_like_fact(sentence):
                    facts.append({
                        "content": sentence.strip(),
                        "category": self._categorize(sentence),
                        "source": "auto-extract",
                        "timestamp": datetime.now().isoformat(),
                        "_signals": _fact_signal_score(sentence),
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
    def _looks_like_fact(sentence: str) -> bool:
        """Heuristic: does this sentence look like a durable fact?

        Deliberately LENIENT (recall first — the Bridge review gate filters
        noise later), but with explicit exclusions for content that is never
        durable knowledge: greetings, imperatives, code/log lines, questions
        and fragments that are too short or too long.
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
