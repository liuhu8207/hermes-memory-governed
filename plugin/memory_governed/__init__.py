"""Hermes Memory Governed — MemoryProvider plugin.

Integrates:
- L1 handwritten rules (MEMORY.md / USER.md)
- L2 semantic memory (LanceDB)
- L3 conversation archive (SQLite FTS5)
- L4 persona (persona.md)
- Async recall via queue_prefetch / prefetch
- Non-blocking sync via WriteQueue
- Optional: Mermaid compression, Tencent extraction, Bridge review gate

Read path:  queue_prefetch() → parallel L2+L3+L4 → cache → prefetch() <1ms
Write path: sync_turn() → L3 sync + L2/extraction async queue
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ._config import GovernedMemoryConfig, load_governed_config
from ._migrations import MigrationRunner
from ._recall import RecallEngine
from ._sync import (
    WriteQueue,
    L3Writer,
    _content_to_text,
    dialogue_fact_admits,
    is_ephemeral_content,
)
from ._compress import MermaidCompressor
from ._bridge import BridgeExporter, screen_bridge_content
from ._kb import KnowledgeBase
from ._text import sanitize_messages
from . import _ingest
from . import _synthesize

try:  # pragma: no cover - _diag.py 由另一位工程师并行新建
    from ._diag import log_data_loss, log_degraded
except Exception:  # noqa: BLE001 — 文件尚未落地时的兼容回退
    def log_data_loss(component: str, reason: str, *, detail: str = "",
                      exc: BaseException | None = None) -> None:
        """Fallback: ERROR 级数据丢失日志（真实 _diag.py 落地后即被替换）。"""
        logger.error("[data-loss] %s: %s %s", component, reason, detail, exc_info=exc)

    def log_degraded(component: str, reason: str, *, detail: str = "",
                     exc: BaseException | None = None) -> None:
        """Fallback: WARNING 级优雅降级日志（真实 _diag.py 落地后即被替换）。"""
        logger.warning("[degraded] %s: %s %s", component, reason, detail, exc_info=exc)

logger = logging.getLogger(__name__)

__all__ = ["GovernedMemoryProvider", "register"]

# Tool schemas for this provider
SEARCH_SCHEMA = {
    "name": "governed_search",
    "description": (
        "Search across all memory layers (L1 rules, L2 facts, L3 conversations). "
        "Returns relevant memories ranked by relevance. Use when you need to recall "
        "something from past interactions or stored knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

AUDIT_SCHEMA = {
    "name": "governed_audit",
    "description": (
        "Audit a specific memory: show its layer, source, confidence, and "
        "drill-down path. Use to verify where a memory came from."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string", "description": "Memory ID or content fragment to audit."},
        },
        "required": ["memory_id"],
    },
}

HEALTH_SCHEMA = {
    "name": "governed_health",
    "description": (
        "Check the health of the memory system: L2 capacity, L4 freshness, "
        "Bridge status, write queue stats. Use for diagnostics."
    ),
    "parameters": {"type": "object", "properties": {}},
}

KB_SEARCH_SCHEMA = {
    "name": "governed_kb_search",
    "description": (
        "Search the knowledge base (Obsidian vault of durable, long-term notes). "
        "Returns ranked notes with semantic + keyword relevance and backlinks. "
        "Use to recall stored knowledge, methods, decisions, or reference material "
        "that lives beyond the current conversation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to find in the knowledge base."},
            "top_k": {"type": "integer", "description": "Max results (default: 10)."},
            "section": {"type": "string", "description": "Optional: inbox/notes/projects/areas/resources/archive."},
        },
        "required": ["query"],
    },
}

KB_ADD_SCHEMA = {
    "name": "governed_kb_add",
    "description": (
        "Write a durable note into the knowledge base (Obsidian vault). "
        "Adds frontmatter (title/tags/concepts/source) and auto-links concepts "
        "as [[wikilinks]]. Use to persist knowledge worth keeping long-term. "
        "Lower-confidence material should go to section 'inbox' for human review."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title (also the filename)."},
            "body": {"type": "string", "description": "Markdown body of the note."},
            "section": {"type": "string", "description": "Vault section, default 'notes' (or 'inbox' for pending review)."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
            "concepts": {"type": "array", "items": {"type": "string"}, "description": "Optional concepts to auto-link as [[wikilinks]]."},
            "source": {"type": "string", "description": "Optional source (URL / file / conversation id)."},
            "confidence": {"type": "number", "description": "Optional 0-1 confidence. If set, high (>= threshold) goes to 'notes', low goes to 'inbox' for review. Omit to use explicit section."},
        },
        "required": ["title", "body"],
    },
}

KB_REVIEW_SCHEMA = {
    "name": "governed_kb_review",
    "description": (
        "Review inbox notes pending confirmation (semi-automatic ingestion). "
        "action 'list' shows pending notes; 'approve' moves a note from inbox "
        "to notes (promoting it to the main library); 'reject' moves it to "
        "archive. Use 'list' to see what awaits review, then approve/reject by title."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "approve", "reject"], "description": "Review action."},
            "title": {"type": "string", "description": "Note title (required for approve/reject)."},
            "limit": {"type": "integer", "description": "Max notes to list (default 50)."},
        },
        "required": ["action"],
    },
}

KB_GET_SCHEMA = {
    "name": "governed_kb_get",
    "description": (
        "Read one full note from the knowledge base by title. "
        "Use to retrieve the complete body of a note found via governed_kb_search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title."},
        },
        "required": ["title"],
    },
}

KB_FETCH_SCHEMA = {
    "name": "governed_kb_fetch",
    "description": (
        "Fetch a URL and extract its readable text (article, doc, JSON, etc.). "
        "Returns the raw content; you (the agent) should then summarize it and "
        "persist the distilled knowledge via governed_kb_add. Use for ingesting "
        "articles, links, or web resources into the knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "HTTP(S) URL to fetch."},
        },
        "required": ["url"],
    },
}

KB_READ_FILE_SCHEMA = {
    "name": "governed_kb_read_file",
    "description": (
        "Read a local file and extract its text (txt/md/json/csv/pdf/docx). "
        "Returns the raw content; you (the agent) should then summarize it and "
        "persist the distilled knowledge via governed_kb_add. Use for ingesting "
        "work files, meeting notes, or documents into the knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to the file."},
        },
        "required": ["path"],
    },
}

KB_TRANSCRIBE_SCHEMA = {
    "name": "governed_kb_transcribe",
    "description": (
        "Transcribe an audio file to text via the configured ASR service "
        "(e.g. SiliconFlow XingChenASR). Long recordings are split automatically "
        "and transcribed chunk by chunk, so an hour-long meeting works; recordings "
        "ffmpeg can decode (.amr/.silk included) are normalised to mp3 first. "
        "Returns the transcript; you (the agent) should then summarize it and "
        "persist via governed_kb_add. Use for meeting recordings or voice memos."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to the audio file."},
        },
        "required": ["path"],
    },
}


# ---------------------------------------------------------------------------
# Prefetch infrastructure — 模块级共享，所有 provider 实例复用
# ---------------------------------------------------------------------------

#: 后台预取共享线程池。取代"每次调用起一个裸线程"。
_PREFETCH_EXECUTOR: Optional[ThreadPoolExecutor] = None
_PREFETCH_EXECUTOR_LOCK = threading.Lock()
#: 正在召回中的 query —— 同一 query 的重复请求直接丢弃（单飞去重）
_inflight: Set[str] = set()
_inflight_lock = threading.Lock()
#: WriteQueue.enqueue 是否支持 provenance_missing 关键字（探测结果缓存）
_ENQUEUE_SUPPORTS_PROVENANCE: Optional[bool] = None


def _get_prefetch_executor() -> ThreadPoolExecutor:
    """惰性创建（或在被关闭后重建）共享预取线程池，最多 2 个线程。"""
    global _PREFETCH_EXECUTOR
    pool = _PREFETCH_EXECUTOR
    if pool is not None:
        return pool
    with _PREFETCH_EXECUTOR_LOCK:
        if _PREFETCH_EXECUTOR is None:
            _PREFETCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="prefetch",
            )
        return _PREFETCH_EXECUTOR


def _shutdown_prefetch_executor() -> None:
    """关闭共享执行器（wait=False，绝不阻塞退出）。下次 queue_prefetch 惰性重建。"""
    global _PREFETCH_EXECUTOR
    with _PREFETCH_EXECUTOR_LOCK:
        pool, _PREFETCH_EXECUTOR = _PREFETCH_EXECUTOR, None
    # 被 wait=False 丢弃的排队任务永远不会执行，它们的 finally 也永远不会跑，
    # 必须清空 _inflight，否则这些 query 会永久失去召回能力。
    with _inflight_lock:
        _inflight.clear()
    if pool is not None:
        pool.shutdown(wait=False)


class _PrefetchCache(OrderedDict):
    """有容量上限 + 机会式 TTL 清扫的 LRU 缓存。

    继承 ``OrderedDict`` 是为了让 ``cache[key] = value`` 这种直接赋值（历史
    代码和测试都在用）也受容量上限约束 —— 光在调用点写 LRU 逻辑会漏掉它。
    """

    def __init__(self, max_size: int = 64, ttl_seconds: float = 30.0,
                 sweep_limit: int = 8) -> None:
        super().__init__()
        self.max_size = max(1, int(max_size))
        self.ttl_seconds = float(ttl_seconds)
        self.sweep_limit = max(1, int(sweep_limit))

    def __setitem__(self, key, value) -> None:
        """写入即置顶，然后清扫过期项、按 LRU 淘汰超容项。"""
        super().__setitem__(key, value)
        self.move_to_end(key, last=True)
        self.sweep_expired()
        while len(self) > self.max_size:
            self.popitem(last=False)

    def sweep_expired(self) -> int:
        """删除最多 ``sweep_limit`` 个已过期条目（从最旧的开始）。

        每次写入顺带清扫一小批，避免全量遍历，也避免脏条目永远赖在缓存里。
        """
        now = time.time()
        removed = 0
        for key in list(self.keys()):  # 插入顺序：最旧在前
            if removed >= self.sweep_limit:
                break
            try:
                entry = OrderedDict.__getitem__(self, key)
                _, timestamp = entry
                expired = (now - float(timestamp)) >= self.ttl_seconds
            except (KeyError, TypeError, ValueError):
                expired = True
            if expired:
                try:
                    del self[key]
                except KeyError:
                    continue
                removed += 1
        return removed


class _RecallStatus:
    """recall_status() 的返回快照。

    命中率在构造时按同一份 hits/misses 计算并 clamp 到 [0, 1]。
    """

    __slots__ = ("count", "provider", "cache_hits", "cache_misses", "cache_hit_rate")

    def __init__(self, count: int, provider: str, hits: int, misses: int) -> None:
        total = hits + misses
        rate = (hits / total) if total > 0 else 0.0
        self.count = count
        self.provider = provider
        self.cache_hits = hits
        self.cache_misses = misses
        self.cache_hit_rate = min(1.0, max(0.0, rate))


class GovernedMemoryProvider:
    """Governed memory provider with L1-L4 layers and async recall."""

    #: 预取缓存容量上限（LRU）。万轮会话下这是内存上界。
    _PREFETCH_CACHE_MAX: int = 64
    #: 每次写入顺带清扫的过期条目上限（避免全量遍历）
    _PREFETCH_CACHE_SWEEP: int = 8

    def __init__(self):
        self._config: Optional[GovernedMemoryConfig] = None
        self._recall: Optional[RecallEngine] = None
        self._l3_writer: Optional[L3Writer] = None
        self._write_queue: Optional[WriteQueue] = None
        self._compressor: Optional[MermaidCompressor] = None
        self._bridge: Optional[BridgeExporter] = None
        self._kb: Optional[KnowledgeBase] = None

        # Prefetch cache: query -> (result_text, timestamp)。有界 LRU + TTL 清扫。
        self._prefetch_cache: _PrefetchCache = _PrefetchCache(
            max_size=self._PREFETCH_CACHE_MAX,
            ttl_seconds=30.0,
            sweep_limit=self._PREFETCH_CACHE_SWEEP,
        )
        self._prefetch_lock = threading.Lock()

        # Recall status for UI
        self._last_recall_count = 0
        self._last_recall_time = 0.0
        # Prefetch cache observability
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def name(self) -> str:
        return "governed"

    def is_available(self) -> bool:
        """Check if the provider is configured and ready."""
        try:
            from hermes_constants import get_hermes_home
            hermes_home = get_hermes_home()
        except Exception:
            hermes_home = Path.home() / ".hermes"

        config_path = hermes_home / "governed_memory.json"
        if config_path.exists():
            return True
        l1_path = hermes_home / "memory" / "MEMORY.md"
        l1_user = hermes_home / "memory" / "USER.md"
        return l1_path.exists() or l1_user.exists()

    def unavailable_reason(self) -> str:
        return (
            "Governed memory not configured. Create ~/.hermes/memory/MEMORY.md "
            "or run the installer. See docs/install.md."
        )

    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize all components."""
        hermes_home = kwargs.get("hermes_home", "")
        if not hermes_home:
            try:
                from hermes_constants import get_hermes_home
                hermes_home = str(get_hermes_home())
            except Exception:
                hermes_home = str(Path.home() / ".hermes")

        self._config = load_governed_config(hermes_home)

        # Run pending schema migrations (idempotent, version-tracked)
        MigrationRunner(Path(hermes_home)).run_pending(self, self._config)

        # Ensure memory directories exist
        for subdir in ["memory", "memory/l2", "memory/l3", "cron/output/scope_recall_bridge"]:
            Path(hermes_home, subdir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        # KB 先于 RecallEngine 构造：召回引擎需要一个知识库句柄来产出
        # 「有相关笔记」提示（KB 通道）。此处顺序有依赖，勿调换。
        self._kb = KnowledgeBase(self._config)
        self._kb.ensure()
        self._recall = RecallEngine(self._config, kb=self._kb)
        self._l3_writer = L3Writer(self._config)
        self._write_queue = WriteQueue(self._config)
        self._compressor = MermaidCompressor(self._config)
        self._bridge = BridgeExporter(self._config)

        # Start write queue worker
        self._write_queue.start()

        # 预取缓存的 TTL 由配置决定
        self._prefetch_cache.ttl_seconds = float(
            self._config.recall.prefetch_ttl_seconds or 30.0
        )

        # Pre-load L1 and L4
        self._recall.get_l1()
        self._recall.get_l4()

        logger.info(
            "GovernedMemory initialized: l1=%s l2=%s l3=%s l4=%s",
            Path(self._config.l1_memory_path).exists(),
            Path(self._config.l2_db_path).exists(),
            Path(self._config.l3_db_path).exists(),
            Path(self._config.l4_persona_path).exists(),
        )

    def system_prompt_block(self) -> str:
        """Inject L4 persona into system prompt."""
        if not self._recall:
            return ""
        l4 = self._recall.get_l4()
        if not l4:
            return ""
        return f"\n\n## User Profile\n{l4}"

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return cached recall results. <1ms if cache hit.

        未命中时不再静默降级：在返回 L1 兜底的同时触发一次自愈式后台召回，
        下一轮就能拿到完整的 L2/L3 结果 —— README 承诺的"内容完整"不再依赖
        host 恰好提前调用了 queue_prefetch。
        """
        if not self._recall:
            return ""

        with self._prefetch_lock:
            cached = self._prefetch_cache.get(query)
            if cached is not None:
                result_text, timestamp = cached
                if (time.time() - timestamp) < self._prefetch_cache.ttl_seconds:
                    self._cache_hits += 1
                    return result_text
                # 过期条目顺手清掉，别让它留在缓存里占地方
                try:
                    del self._prefetch_cache[query]
                except KeyError:
                    pass
            self._cache_misses += 1

        # 自愈：后台补一次召回，调用方不等待
        self._submit_prefetch(query)

        l1 = self._recall.get_l1()
        if l1:
            return f"[User Rules]\n{l1}"
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Background parallel recall for the NEXT turn.

        提交到模块级共享线程池（最多 2 线程），并对同一 query 做单飞去重：
        重复请求不会产生重复的召回工作。
        """
        self._submit_prefetch(query)

    def _submit_prefetch(self, query: str) -> bool:
        """Submit a background recall for ``query`` unless one is already fresh/pending.

        Returns True when a task was actually submitted.
        """
        if not self._recall or not query:
            return False

        # 缓存里已有未过期结果 —— 无需再召回
        with self._prefetch_lock:
            cached = self._prefetch_cache.get(query)
            if cached is not None and (time.time() - cached[1]) < self._prefetch_cache.ttl_seconds:
                return False

        # 同一 query 已在召回中 —— 单飞去重
        with _inflight_lock:
            if query in _inflight:
                return False
            _inflight.add(query)

        try:
            _get_prefetch_executor().submit(self._run_prefetch, query)
        except Exception as e:  # noqa: BLE001 — 线程池不可用时不能影响主流程
            with _inflight_lock:
                _inflight.discard(query)
            logger.debug("Prefetch submit failed: %s", e)
            return False
        return True

    def _run_prefetch(self, query: str) -> None:
        """Background recall body. 无论成败都在 finally 里摘掉 in-flight 标记。"""
        try:
            config = self._config
            recall = self._recall
            if recall is None or config is None:
                return

            results = recall.parallel_recall(query)
            l1 = recall.get_l1()
            formatted = recall.format_recall(
                results, l1,
                l1_budget=config.recall.l1_budget_tokens,
                l23_budget=config.recall.l23_budget_tokens,
            )

            with self._prefetch_lock:
                self._prefetch_cache[query] = (formatted, time.time())

            self._last_recall_count = len(results)
            self._last_recall_time = time.time()

        except Exception as e:
            logger.debug("Prefetch failed: %s", e)
        finally:
            with _inflight_lock:
                _inflight.discard(query)

    def recall_status(self):
        """Report what the last prefetch injected.

        hits/misses/count 在锁内一次性读取，保证三者同源（否则并发下命中率
        可能算出 >1 的值）。
        """
        with self._prefetch_lock:
            hits = self._cache_hits
            misses = self._cache_misses
            count = self._last_recall_count

        if count == 0 and (hits + misses) == 0:
            return None
        return _RecallStatus(count=count, provider=self.name, hits=hits, misses=misses)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Non-blocking write: L3 sync + L2/extraction async.

        ``user_content`` / ``assistant_content`` 不再是死参数：当调用方只传这
        两个参数（没传 ``messages``）时，按 (user, assistant) 顺序构造出同构的
        消息列表再落库 —— 过去这种情况是 0 行归档且调用方完全无感知。
        """
        if not messages:
            messages = self._messages_from_contents(user_content, assistant_content)
        if not messages:
            return

        # Sanitize once, here, rather than at each layer that would otherwise
        # fail on a lone surrogate: SQLite encodes text parameters as UTF-8 (the
        # archive and its FTS mirror), Arrow demands strict UTF-8 (the L2 row),
        # and the embedding call raises before either. Measured 2026-09-18, the
        # last of those had been silently degrading the semantic channel —
        # embed_one turned the exception into a None and recall fell back to
        # lexical without saying so. One sanitization at the ingress covers all
        # four consumers, which is also why the three duplicate copies that used
        # to live in _embedding / hgm_mcp / memory_cli are gone.
        messages = sanitize_messages(messages)

        # L3: synchronous write (<10ms, WAL mode); returns content→rowid map
        rowid_map: dict = {}
        provenance_missing = False
        if self._l3_writer is None:
            # 未初始化：这一轮无法溯源，但内容本身仍然值得进 L2
            provenance_missing = True
            logger.debug("sync_turn before initialize(): L3 skipped, no provenance")
        else:
            try:
                rowid_map = self._l3_writer.write(messages, session_id) or {}
            except Exception as e:
                rowid_map = {}
                provenance_missing = True
                # 数据丢失：L2 事实将永久无法下钻回 L3，必须可见
                log_data_loss(
                    "l3",
                    "l3_write_failed",
                    detail=f"session={session_id!r}, messages={len(messages)}",
                    exc=e,
                )
        if not rowid_map:
            provenance_missing = True

        # L2 + extraction: async via write queue (with L3 references)
        self._enqueue_for_l2(messages, session_id, rowid_map, provenance_missing)

    @staticmethod
    def _messages_from_contents(user_content: str,
                                assistant_content: str) -> List[Dict[str, Any]]:
        """Build a (user, assistant) message pair from the plain-content parameters.

        过滤掉空内容的条目，保持与真实消息同构（不加额外字段）。
        """
        messages: List[Dict[str, Any]] = []
        for role, content in (("user", user_content), ("assistant", assistant_content)):
            if content and str(content).strip():
                messages.append({"role": role, "content": content})
        if messages:
            logger.debug(
                "sync_turn: synthesized %d message(s) from content parameters",
                len(messages),
            )
        return messages

    def _enqueue_for_l2(self, messages: List[Dict[str, Any]], session_id: str,
                        rowid_map: dict, provenance_missing: bool) -> None:
        """Queue messages for L2 indexing, marking missing L3 provenance.

        L3 写失败时**仍然**推给 L2：L2 事实本身有价值（可检索、可用），只是
        缺溯源。丢掉它才是真正的丢数据。这里显式打上 provenance_missing 标记，
        让 governed_audit 的下钻明确报"无溯源"，而不是假装没这回事。
        """
        if not self._write_queue:
            return
        if not provenance_missing or not self._enqueue_supports_provenance_flag():
            self._write_queue.enqueue(messages, session_id, rowid_map=rowid_map)
            return
        self._write_queue.enqueue(
            messages, session_id, rowid_map=rowid_map, provenance_missing=True,
        )

    @staticmethod
    def _enqueue_supports_provenance_flag() -> bool:
        """WriteQueue.enqueue 是否接受 provenance_missing（结果缓存一次）。

        用签名探测而不是 try/except TypeError —— 后者一旦 enqueue 内部抛出
        TypeError 就会造成重复入队。
        """
        global _ENQUEUE_SUPPORTS_PROVENANCE
        if _ENQUEUE_SUPPORTS_PROVENANCE is None:
            try:
                import inspect

                from ._sync import WriteQueue

                params = inspect.signature(WriteQueue.enqueue).parameters
                _ENQUEUE_SUPPORTS_PROVENANCE = "provenance_missing" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except (TypeError, ValueError, ImportError):
                _ENQUEUE_SUPPORTS_PROVENANCE = False
        return bool(_ENQUEUE_SUPPORTS_PROVENANCE)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Compress messages into Mermaid canvas before context compression."""
        if not self._compressor:
            return ""
        return self._compressor.compress(messages)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Session end: trigger Bridge export, session synthesis, and L4 refresh."""
        if self._bridge:
            candidates = self._extract_session_candidates(messages)
            if candidates:
                self._bridge.export_candidates(candidates)

        # session 级自动归纳（插件侧调 LLM，模型对齐 agent 对话模型）
        self._synthesize_session(messages)

        if self._recall:
            self._recall.refresh_l4_background()

    def _synthesize_session(self, messages: List[Dict[str, Any]]) -> None:
        """Session 结束自动归纳 → 治理落库。

        归纳用 agent 同款模型（``config.synthesis``），落库走 ``_kb.add`` 的
        治理（密钥闸门 / 置信分流 / 链接补全）。失败静默降级，绝不抛给宿主。

        归纳前先做零成本过滤（``_should_synthesize``）：session 里没有 durable
        信号（偏好/决策/配置意图）就跳过，省一次 LLM 调用。
        """
        if not self._config or not self._kb:
            return
        # 零成本前置过滤：无 durable 信号则跳过（reasoning 模型归纳 ~5-6k token）
        if (getattr(self._config.synthesis, "require_durable_signal", True)
                and not self._should_synthesize(messages)):
            logger.info("session synthesis skipped: no durable signal")
            return
        try:
            notes = _synthesize.synthesize_notes(messages, self._config)
        except Exception as e:  # noqa: BLE001
            logger.warning("session synthesis failed: %s", e)
            return
        if not notes:
            return

        added = 0
        for n in notes:
            try:
                r = self._kb.add(
                    title=str(n.get("title", "")),
                    body=str(n.get("body", "")),
                    tags=n.get("tags") or [],
                    concepts=n.get("concepts") or [],
                    source="session-synthesis",
                    confidence=n.get("confidence"),
                )
                if r.get("ok"):
                    added += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("synthesis add failed: %s", e)
        logger.info("session synthesis: %d candidates, %d added", len(notes), added)

    def _should_synthesize(self, messages: List[Dict[str, Any]]) -> bool:
        """零成本启发式：session 里有没有值得沉淀的 durable 信号。

        扫 user 消息（多模态 content 只取 text parts），任一命中 durable 意图词
        即返回 True。宁缺毋滥：只认明确的偏好/决策/变更/配置意图，闲聊、查询、
        纯执行任务不命中 → 归纳被跳过，省 token。
        """
        import re

        signals = [
            # 明确偏好/决策（对齐 _looks_like_durable_fact）
            r"\bprefer\b", r"\balways\b", r"\bnever\b", r"\bwill use\b",
            r"偏好", r"习惯", r"记住", r"记得", r"决定", r"默认",
            r"要求", r"必须", r"禁止", r"不允许", r"绝不",
            r"不要", r"别再", r"以后", r"每次", r"总是",
            # 变更/选型/配置决策
            r"改用", r"换成", r"换回", r"选型", r"采用", r"迁移", r"重构",
            r"配好", r"已配", r"保存好", r"设置好",
            # 否定决策（决定不用/不换/不切）
            r"不用切", r"不用换", r"不换", r"不改", r"不切",
            # 收敛决策（"够了/先这样/暂时先"）
            r"够用", r"够了", r"先这样", r"暂时先",
        ]
        for msg in messages or []:
            if str(msg.get("role", "") or "") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                parts = [
                    p.get("text") for p in content
                    if isinstance(p, dict) and isinstance(p.get("text"), str)
                ]
                content = "\n".join(parts)
            content = str(content or "").strip()
            if not content:
                continue
            # 纯提问不算 durable 表达（"你以后都用什么？"）
            if content.endswith(("?", "？")):
                continue
            content_lower = content.lower()
            if any(re.search(sig, content_lower) for sig in signals):
                return True
        return False

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror built-in memory writes to L1 files."""
        if not self._config:
            return

        path = self._config.l1_user_path if target == "user" else self._config.l1_memory_path
        if target not in ("user", "memory"):
            return

        if action == "add":
            self._append_to_file(path, content)
        elif action == "replace":
            # Metadata may carry old_text for targeted replacement
            old_text = (metadata or {}).get("old_text", "")
            if old_text:
                self._replace_in_file(path, old_text, content)
            else:
                # Without old_text, append as new entry
                self._append_to_file(path, content)
        elif action == "remove":
            old_text = (metadata or {}).get("old_text", content)
            self._remove_from_file(path, old_text)

        # Invalidate L1 cache
        if self._recall:
            self._recall._l1_cache = None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, AUDIT_SCHEMA, HEALTH_SCHEMA,
                KB_SEARCH_SCHEMA, KB_ADD_SCHEMA, KB_GET_SCHEMA,
                KB_REVIEW_SCHEMA,
                KB_FETCH_SCHEMA, KB_READ_FILE_SCHEMA, KB_TRANSCRIBE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle tool calls for governed_search/audit/health/kb_*."""
        if tool_name == "governed_search":
            return self._handle_search(args)
        elif tool_name == "governed_audit":
            return self._handle_audit(args)
        elif tool_name == "governed_health":
            return self._handle_health()
        elif tool_name == "governed_kb_search":
            return self._handle_kb_search(args)
        elif tool_name == "governed_kb_add":
            return self._handle_kb_add(args)
        elif tool_name == "governed_kb_get":
            return self._handle_kb_get(args)
        elif tool_name == "governed_kb_review":
            return self._handle_kb_review(args)
        elif tool_name == "governed_kb_fetch":
            return self._handle_kb_fetch(args)
        elif tool_name == "governed_kb_read_file":
            return self._handle_kb_read_file(args)
        elif tool_name == "governed_kb_transcribe":
            return self._handle_kb_transcribe(args)
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._recall:
            self._recall.shutdown()
        if self._write_queue:
            self._write_queue.stop()
        if self._l3_writer:
            self._l3_writer.shutdown()
        # wait=False：绝不阻塞退出；下次 queue_prefetch 会惰性重建线程池
        _shutdown_prefetch_executor()

    # -- Tool handlers -----------------------------------------------------

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "Missing query parameter"})

        top_k = min(int(args.get("top_k", 10)), 50)
        results = self._recall.parallel_recall(query) if self._recall else []
        items = [
            {"layer": r.layer, "content": r.content, "score": round(r.score, 3)}
            for r in results[:top_k]
        ]
        return json.dumps({"results": items, "count": len(items)})

    def _handle_audit(self, args: dict) -> str:
        memory_id = args.get("memory_id", "")
        if not memory_id:
            return json.dumps({"error": "Missing memory_id parameter"})

        audit_result = {
            "memory_id": memory_id,
            "l1_found": False,
            "l2_found": False,
            "l3_found": False,
            "l4_found": False,
        }

        import re as _re

        # Check L1 — use word-boundary regex for precise matching
        l1 = self._recall.get_l1() if self._recall else ""
        if l1:
            l1_lower = l1.lower()
            mid_lower = memory_id.lower()
            if mid_lower in l1_lower or _re.search(r'\b' + _re.escape(mid_lower) + r'\b', l1_lower):
                audit_result["l1_found"] = True
                audit_result["l1_source"] = "handwritten"

        # Check L2 (with L3 drill-down via source_rowid)
        if self._recall:
            l2_results = self._recall._search_l2(memory_id)
            if l2_results:
                audit_result["l2_found"] = True
                audit_result["l2_score"] = l2_results[0].score
                src_ref = (l2_results[0].metadata or {}).get("source_rowid")
                if src_ref:
                    audit_result["l2_source_rowid"] = src_ref
                    drill = self._drill_l3_rowid(src_ref)
                    if drill:
                        audit_result["l3_message"] = drill

        # Check L3
        if self._recall:
            l3_results = self._recall._search_l3(memory_id)
            if l3_results:
                audit_result["l3_found"] = True
                audit_result["l3_matches"] = len(l3_results)

        # Check L4
        l4 = self._recall.get_l4() if self._recall else ""
        if l4:
            l4_lower = l4.lower()
            mid_lower = memory_id.lower()
            if mid_lower in l4_lower or _re.search(r'\b' + _re.escape(mid_lower) + r'\b', l4_lower):
                audit_result["l4_found"] = True

        return json.dumps(audit_result, ensure_ascii=False)

    def _drill_l3_rowid(self, rowid) -> Optional[dict]:
        """Read one L3 message by rowid (audit drill-down, L2 → L3)."""
        if not self._config or rowid is None:
            return None
        try:
            import sqlite3
            conn = sqlite3.connect(f"file:{self._config.l3_db_path}?mode=ro", uri=True)
            row = conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE id = ?",
                (int(rowid),),
            ).fetchone()
            conn.close()
            if row:
                return {"role": row[0], "content": (row[1] or "")[:200], "timestamp": row[2]}
        except Exception as e:
            logger.debug("L3 drill-down failed for rowid %s: %s", rowid, e)
        return None

    def _handle_health(self) -> str:
        cache_hits = getattr(self, '_cache_hits', 0)
        cache_misses = getattr(self, '_cache_misses', 0)
        total = cache_hits + cache_misses
        if not self._config:
            return json.dumps({"error": "Provider not initialized", "available": self.is_available()}, ensure_ascii=False)
        health = {
            "provider": self.name,
            "available": self.is_available(),
            "l1_memory_exists": Path(self._config.l1_memory_path).exists(),
            "l1_user_exists": Path(self._config.l1_user_path).exists(),
            "l2_exists": Path(self._config.l2_db_path).exists(),
            "l3_exists": Path(self._config.l3_db_path).exists(),
            "l4_exists": Path(self._config.l4_persona_path).exists(),
            "write_queue": self._write_queue.stats if self._write_queue else {},
            "bridge": self._bridge.get_status() if self._bridge else {},
            "kb": self._kb.stats() if self._kb else {},
            "last_recall_count": self._last_recall_count,
            "prefetch_cache": {
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": round(cache_hits / total, 3) if total else 0.0,
            },
        }
        return json.dumps(health, ensure_ascii=False)

    def _handle_kb_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return json.dumps({"error": "Missing query parameter"}, ensure_ascii=False)
        if not self._kb:
            return json.dumps({"error": "Knowledge base not initialized"}, ensure_ascii=False)
        top_k = int(args.get("top_k", self._config.kb.top_k))
        section = str(args.get("section", "") or "")
        results = self._kb.search(query, top_k=top_k, section=section)
        return json.dumps({"results": results, "count": len(results)}, ensure_ascii=False)

    def _handle_kb_add(self, args: dict) -> str:
        title = args.get("title", "")
        body = args.get("body", "")
        if not title or not body:
            return json.dumps({"error": "Missing title or body parameter"}, ensure_ascii=False)
        if not self._kb:
            return json.dumps({"error": "Knowledge base not initialized"}, ensure_ascii=False)
        try:
            result = self._kb.add(
                title=title,
                body=body,
                section=str(args.get("section", "notes") or "notes"),
                tags=args.get("tags") or [],
                concepts=args.get("concepts") or [],
                source=str(args.get("source", "") or ""),
                update=bool(args.get("update", False)),
                confidence=args.get("confidence"),
            )
        except Exception as e:  # noqa: BLE001 — 工具绝不抛给 agent
            logger.warning("governed_kb_add failed: %s", e)
            return json.dumps({"error": f"Failed to add note: {e}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _handle_kb_get(self, args: dict) -> str:
        title = args.get("title", "")
        if not title:
            return json.dumps({"error": "Missing title parameter"}, ensure_ascii=False)
        if not self._kb:
            return json.dumps({"error": "Knowledge base not initialized"}, ensure_ascii=False)
        note = self._kb.get(title)
        if note is None:
            return json.dumps({"error": f"Note not found: {title}"}, ensure_ascii=False)
        return json.dumps(note, ensure_ascii=False)

    def _handle_kb_review(self, args: dict) -> str:
        action = str(args.get("action", "list") or "list")
        if not self._kb:
            return json.dumps({"error": "Knowledge base not initialized"}, ensure_ascii=False)
        try:
            if action == "list":
                try:
                    limit = int(args.get("limit", 50) or 50)
                except (TypeError, ValueError):
                    limit = 50
                result = {"ok": True, "pending": self._kb.list_review(limit=limit)}
            elif action == "approve":
                title = str(args.get("title", "") or "")
                if not title:
                    result = {"ok": False, "error": "Missing title for approve"}
                else:
                    result = self._kb.approve(title)
            elif action == "reject":
                title = str(args.get("title", "") or "")
                if not title:
                    result = {"ok": False, "error": "Missing title for reject"}
                else:
                    result = self._kb.reject(title)
            else:
                result = {"ok": False, "error": f"Unknown action: {action}"}
        except Exception as e:  # noqa: BLE001 — 工具绝不抛给 agent
            logger.warning("governed_kb_review failed: %s", e)
            return json.dumps({"error": f"Review failed: {e}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _handle_kb_fetch(self, args: dict) -> str:
        url = args.get("url", "")
        if not url:
            return json.dumps({"error": "Missing url parameter"}, ensure_ascii=False)
        try:
            result = _ingest.fetch_url(url)
        except Exception as e:  # noqa: BLE001
            logger.warning("governed_kb_fetch failed: %s", e)
            return json.dumps({"error": f"fetch failed: {e}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _handle_kb_read_file(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return json.dumps({"error": "Missing path parameter"}, ensure_ascii=False)
        try:
            result = _ingest.read_file(path)
        except Exception as e:  # noqa: BLE001
            logger.warning("governed_kb_read_file failed: %s", e)
            return json.dumps({"error": f"read_file failed: {e}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    def _handle_kb_transcribe(self, args: dict) -> str:
        path = args.get("path", "")
        if not path:
            return json.dumps({"error": "Missing path parameter"}, ensure_ascii=False)
        try:
            result = _ingest.transcribe_audio_auto(path, self._config)
        except Exception as e:  # noqa: BLE001
            logger.warning("governed_kb_transcribe failed: %s", e)
            return json.dumps({"error": f"transcribe failed: {e}"}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False)

    # -- Helpers ------------------------------------------------------------

    def _append_to_file(self, path_str: str, content: str) -> None:
        """Append content to a file, creating if needed."""
        try:
            p = Path(path_str)
            p.parent.mkdir(parents=True, exist_ok=True)
            existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            if content.strip() not in existing:
                with open(p, "a", encoding="utf-8") as f:
                    f.write(f"\n\n{content.strip()}\n")
        except Exception as e:
            logger.debug("Failed to append to %s: %s", path_str, e)

    def _replace_in_file(self, path_str: str, old_text: str, new_text: str) -> None:
        """Replace old_text with new_text in a file. If old_text not found, append."""
        try:
            p = Path(path_str)
            if not p.exists():
                self._append_to_file(path_str, new_text)
                return
            content = p.read_text(encoding="utf-8", errors="replace")
            if old_text in content:
                content = content.replace(old_text, new_text, 1)
                p.write_text(content, encoding="utf-8")
            else:
                # Old text not found — append new
                self._append_to_file(path_str, new_text)
        except Exception as e:
            logger.debug("Failed to replace in %s: %s", path_str, e)

    def _remove_from_file(self, path_str: str, old_text: str) -> None:
        """Remove old_text from a file."""
        try:
            p = Path(path_str)
            if not p.exists():
                return
            content = p.read_text(encoding="utf-8", errors="replace")
            if old_text in content:
                content = content.replace(old_text, "", 1)
                # Clean up double newlines
                while "\n\n\n" in content:
                    content = content.replace("\n\n\n", "\n\n")
                p.write_text(content, encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to remove from %s: %s", path_str, e)

    def _extract_session_candidates(self, messages: List[Dict[str, Any]]) -> List[dict]:
        """Extract Bridge candidates from a session's messages.

        三道关（2026-09-16 与 L2 统一；2026-09-17 第三关并入共享尺子）：

        1. ``screen_bridge_content`` —— 整条硬拒（时效性待办 / 模板骨架 /
           多模态占位符 / 凭据 / 纯路径）。时效性尤其关键：候选会被 promote
           进 L1 成为**永久规则**，一条「考试前一天记得提醒我」会永远生效。
        2. ``_passes_structural_filter`` —— 结构性排除（太短 / 问句 / 列表 /
           空白）。
        3. ``dialogue_fact_admits`` —— **与写入路径、回放脚本完全相同的那把
           尺子**（``_sync.dialogue_fact_admits``）。这里曾经内联
           ``_fact_signal_score(content, "user", strong_only=True)``，既不传
           ``include_role=False``、也不消解疑问句式，于是「能不能…」因为字面
           含「不能」被整类放行 —— 而本入口的产物会 promote 进 L1，代价比 L2
           还重。改为调用共享定义后，三个站点共用一条规则。

        另外用 ``_content_to_text`` 归一化 content：多模态消息的 content 是
        list，直接做字符串运算会崩（这正是 9-15 那次 P0 的成因）。
        """
        candidates = []
        for msg in messages:
            if msg.get("role", "") != "user":
                continue
            content = _content_to_text(msg.get("content", ""))
            if not content or not content.strip():
                continue

            reason = screen_bridge_content(content)
            if reason is not None:
                logger.debug("Session candidate rejected (%s): %s", reason, content[:60])
                continue
            if not self._passes_structural_filter(content):
                continue
            # 统一尺子（与写入 _sync._extract_atomic_facts / 回放 l2_apply_gate
            # 共用同一个 dialogue_fact_admits）：去掉角色先验、消解「能不能…」
            # 疑问句式、并要求单条信号必须带佐证 —— 一条证据不构成证据。
            if not dialogue_fact_admits(content, "user"):
                logger.debug("Session candidate below signal floor: %s", content[:60])
                continue

            candidates.append({
                "content": content.strip()[:600],
                "target": "memory",
                "memory_type": "memory",
                "tags": ["session-extract", "review-required"],
                "source": "hermes-memory-governed",
                "source_path": "session",
            })

        return candidates

    @staticmethod
    def _passes_structural_filter(content: str) -> bool:
        """结构性排除：太短 / 问句 / 列表 / 空白。

        与 :meth:`_looks_like_durable_fact` 的关系：那个方法还带一套**独立的
        偏好词表**，在 Bridge 路径上与统一的强信号门槛（``_fact_signal_score``）
        重复且更窄 —— 实测「我在NAS有装secretstore，但本地布署不能同步」
        强信号 6 分（命中「我在」「不能」）却被旧词表拦下。Bridge 路径因此
        只保留结构检查，值不值得记交给那把统一的尺子。
        """
        stripped = (content or "").strip()
        if not stripped:
            return False
        if stripped.endswith(("?", "？")):
            return False
        cjk_count = sum(1 for c in stripped if "\u4e00" <= c <= "\u9fff")
        non_cjk_count = len(stripped) - cjk_count
        if non_cjk_count + cjk_count * 3 < 15:
            return False
        if stripped.count(",") > 3 or stripped.count("、") > 3:
            return False
        return True

    @staticmethod
    def _looks_like_durable_fact(content: str) -> bool:
        """Heuristic: does this look like a durable preference or decision?

        DEAD CODE as of 2026-09-17 — no production caller. The Bridge path
        (:meth:`_extract_session_candidates`) stopped consulting it on 2026-09-16
        when it moved to the single strong-signal ruler, and grepping ``plugin/``
        finds only this definition and docstring mentions. It is kept (not
        deleted) solely because two regression suites still pin its historical
        behaviour — ``tests/test_memory_governed.py::TestBuild`` and
        ``tests/test_qa_concurrency.py::TestDurableFactHeuristic``. Do NOT wire
        it back into any admission path: it carries an independent, narrower word
        list and is exactly the kind of second ruler this module is consolidating
        away (see ``_sync.dialogue_fact_admits``).

        Requires a preference/decision keyword AND that the content
        is a statement (not a question, shopping list, or short request).
        """
        import re

        # Must contain at least one preference/decision signal
        # CJK 部分按"意图动词"覆盖：偏好/习惯/记住/记得/决定/默认/要求/
        # 必须/禁止/不允许/绝不/不要/别再/以后/每次/总是。
        # 早先只写了 偏好/记得/以后(都|请|要)/不要(再|去)，连"不要动 X"
        # "我们决定用 X""绝不允许改 X" 都漏掉。
        pref_patterns = [
            r"\bprefer\b", r"\balways\b", r"\bnever\b", r"\bwill use\b",
            r"\bdon'?t\b.*(?:refactor|change|remove|delete|use|touch)",
            r"偏好", r"习惯", r"记住", r"记得", r"决定", r"默认",
            r"要求", r"必须", r"禁止", r"不允许", r"绝不",
            r"不要", r"别再", r"以后", r"每次", r"总是",
        ]
        content_lower = content.lower()
        if not any(re.search(p, content_lower) for p in pref_patterns):
            return False

        # Exclude: questions, very short messages, shopping/todo lists
        if content.strip().endswith(("?", "？")):
            return False

        # Minimum length: CJK chars carry more info per char, so use a
        # weighted count (CJK ~3 chars carries the info of ~1 English word).
        stripped = content.strip()
        cjk_count = sum(1 for c in stripped if "\u4e00" <= c <= "\u9fff")
        non_cjk_count = len(stripped) - cjk_count
        weighted_len = non_cjk_count + cjk_count * 3
        if weighted_len < 15:
            return False

        # Exclude messages that look like multiple comma/slash-separated items (list-like)
        if content.count(",") > 3 or content.count("、") > 3:
            return False

        return True


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the governed memory provider with Hermes."""
    ctx.register_memory_provider(GovernedMemoryProvider())
