"""Parallel recall engine for the governed memory provider.

Read path: queue_prefetch() runs L2+L3+L4 in parallel threads,
prefetch() returns cached results in <1ms.
"""

from __future__ import annotations

import logging
import math
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._config import GovernedMemoryConfig
from ._embedding import EmbeddingService

try:  # pragma: no cover - _diag.py 由另一位工程师并行新建
    from ._diag import log_degraded, log_data_loss
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

#: LanceDB ``cosine`` metric 返回的是**余弦距离** d ∈ [0, 2]：
#: 0 = 完全同向，1 = 正交，2 = 完全反向。
COSINE_DISTANCE_MAX: float = 2.0

#: ``_distance`` 字段缺失 / 非数值时的兜底值，按"正交"处理 → score = 0.5。
#: 不能取 0.0（会把脏行误判成完美命中），也不能取 2.0（等于静默丢弃）。
DEFAULT_COSINE_DISTANCE: float = 1.0

#: :meth:`RecallEngine.format_recall` 丢弃低相关项的阈值（P0 用户可见症状的闸门）。
MIN_SCORE: float = 0.1

#: L3 召回只保留"用户 / 助手"对话，过滤 ``tool`` 角色。
#: 实测 tool 消息（JSON 结果 / 终端输出 / 文件列表）占 L3 语料约 60%，
#: 且会以 score=1.0 挤进注入上下文 —— 它们是工具过程输出，不是可复用的
#: 会话记忆，必须挡在召回之外。提为模块级常量便于以后调整白名单。
L3_RECALL_ROLES: tuple = ("user", "assistant")


def layer_score_floor(layer: str, recall_cfg=None, *,
                      l2_min_score: "float | None" = None) -> float:
    """返回指定记忆层的最低相关分门槛（``>= floor`` 才会被注入）。

    L2 与 L3 的分数分布不同源，必须分开设阈值：

    * **L3**（FTS5 关键词 + 时间衰减）：有效命中实测 0.3~0.7，门槛沿用
      :data:`MIN_SCORE`（0.1）。
    * **L2**（1024 维余弦映射 ``1 - d/2``）：无关查询的地板分就有 0.73，
      门槛取 ``recall.l2_min_score``；未配置（``0.0``）时**回退**到
      :data:`MIN_SCORE`，保证老配置行为不变。

    Args:
        layer: ``"l2"`` / ``"l3"`` 等记忆层标识。
        recall_cfg: ``RecallConfig`` 实例（可为 ``None``）。
        l2_min_score: 显式覆盖值，用于测试或调用方临时收紧；``None`` 时读配置。

    Returns:
        该层的绝对分数门槛。任何异常都返回 :data:`MIN_SCORE`，绝不抛错——
        召回是读路径，宁可放行也不能因为配置脏值把记忆全砍光。
    """
    if layer != "l2":
        return MIN_SCORE
    value = l2_min_score
    if value is None:
        value = getattr(recall_cfg, "l2_min_score", 0.0)
    try:
        value = float(value or 0.0)
    except (TypeError, ValueError):
        return MIN_SCORE
    if value <= 0.0 or math.isnan(value):
        return MIN_SCORE
    return value


def distance_to_score(distance: float) -> float:
    """把 LanceDB 的余弦距离换算成 [0, 1] 的相关性分数。

    召回链路上 L3 的分数已归一化到 [0, 1]，``MIN_SCORE`` 也是 [0, 1] 上的
    绝对阈值，因此 L2 的距离必须先做 ``1 - d/2`` 的线性映射再 clamp。

    P0 背景：旧实现写成 ``1.0 - min(d, 1.0)``（假设返回的是 [0, 1] 的余弦
    相似度），而 LanceDB 默认 metric 是 L2 欧氏距离 —— 384/512 维向量的典型
    距离是 1.0~1.5，``min(d, 1.0)`` 恒为 1.0 → score 恒为 0 → ``MIN_SCORE``
    把 **100%** 的 L2 命中丢掉，用户侧表现为"L2 记忆完全不召回"。

    Args:
        distance: 余弦距离。``None`` / 非数值 / NaN / inf 时按
            :data:`DEFAULT_COSINE_DISTANCE` 兜底（score = 0.5），不抛异常。

    Returns:
        [0, 1] 区间的相关性分数（越大越相关），越界输入被 clamp 到边界。
    """
    try:
        value = float(distance)
    except (TypeError, ValueError):
        value = DEFAULT_COSINE_DISTANCE
    # NaN / ±inf 会让后续的比较与排序失去意义，统一按缺省距离处理
    if math.isnan(value) or math.isinf(value):
        value = DEFAULT_COSINE_DISTANCE
    return max(0.0, min(1.0, 1.0 - value / COSINE_DISTANCE_MAX))


def row_to_score(row: Any) -> float:
    """从 LanceDB 结果行里取 ``_distance`` 并换算成 score。

    字段缺失或行对象不是 Mapping 时走 :data:`DEFAULT_COSINE_DISTANCE` 兜底，
    保证一行脏数据不会让整个读路径抛异常。
    """
    getter = getattr(row, "get", None)
    if not callable(getter):
        return distance_to_score(DEFAULT_COSINE_DISTANCE)
    try:
        distance = getter("_distance", DEFAULT_COSINE_DISTANCE)
    except Exception as e:  # noqa: BLE001 — 行的 get 可能抛任意异常
        logger.debug("L2 row _distance lookup failed, using default: %s", e)
        distance = DEFAULT_COSINE_DISTANCE
    return distance_to_score(distance)


@dataclass
class RecallResult:
    """A single recalled memory item."""
    layer: str           # "l1", "l2", "l3", "l4", "kb"
    content: str
    score: float
    source: str = ""     # file path or "handwritten"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RecallEngine:
    """Parallel recall across L1/L2/L3/L4 layers.

    Design:
    - L1: always included, loaded into memory at init, refreshed on file change
    - L2: LanceDB vector search, async via thread pool
    - L3: SQLite FTS5 search with time decay, async via thread pool
    - L4: persona.md, cached in memory, background refresh
    """

    #: L1 缓存的 TTL（秒）。mtime 变化时不等 TTL 立即重载。
    L1_TTL_SECONDS: float = 60.0
    #: 一次 LIKE 查询最多用多少个片段做 OR（防止超长查询生成爆炸式 SQL）
    LIKE_SEGMENT_LIMIT: int = 8
    #: CJK 连续段 / 非 CJK 连续段
    SEGMENT_PATTERN = re.compile(
        r"[\u4e00-\u9fff\u3000-\u303f]+|[^\u4e00-\u9fff\u3000-\u303f\s]+"
    )

    def __init__(self, config: GovernedMemoryConfig, kb: Any = None):
        self._config = config
        #: 知识库门面（``_kb.KnowledgeBase``）。可选注入：为 None 时 KB 通道
        #: 静默关闭，其余三层不受影响。用 Any 避免 _recall ↔ _kb 循环导入。
        self._kb = kb
        self._l1_cache: Optional[str] = None
        self._l1_cache_time: float = 0
        #: 缓存时所读 L1 文件的最大 mtime（秒，float）。用于廉价感知外部改动。
        self._l1_cache_mtime: float = 0.0
        self._l4_cache: Optional[str] = None
        self._l4_cache_time: float = 0
        self._l2_store = None  # Lazy-loaded LanceDB
        self._embed_fn = None  # Lazy-loaded embedding function
        self._embed_model = None  # Lazy-loaded SentenceTransformer
        self._embed_service: Optional[EmbeddingService] = None
        # L2 懒初始化：无锁会导致冷启动并发重复加载嵌入模型（内存翻倍）
        self._l2_lock = threading.Lock()
        # 已尝试过初始化（无论成败）。失败后不再重试，避免每次查询都 import lancedb
        self._l2_init_attempted: bool = False
        self._shutdown: bool = False

    # -- L1: Handwritten (always in memory) --------------------------------

    def get_l1(self) -> str:
        """L1 hand-written rules. Loaded once, refreshed every 60s if file changed.

        TTL 只是兜底：每次命中缓存时先做一次廉价的 ``stat()``，只要任一 L1
        文件的 mtime 变了就立刻重载 —— 脚本或其它进程改动 MEMORY.md / USER.md
        后最长 60s 不可见的问题由此解决。``stat`` 失败时安全降级（保留旧缓存）。
        """
        now = time.time()
        if self._l1_cache is not None and (now - self._l1_cache_time) < self.L1_TTL_SECONDS:
            if not self._l1_files_changed():
                return self._l1_cache

        parts = []
        for path_str in [self._config.l1_memory_path, self._config.l1_user_path]:
            p = Path(path_str)
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        parts.append(text)
                except Exception:
                    pass

        self._l1_cache = "\n\n".join(parts) if parts else ""
        self._l1_cache_time = now
        self._l1_cache_mtime = self._l1_mtime_signature()
        return self._l1_cache

    def _l1_mtime_signature(self) -> float:
        """返回两个 L1 文件中最大的 mtime（秒）。文件不存在时记为 0。"""
        max_mtime = 0.0
        for path_str in (self._config.l1_memory_path, self._config.l1_user_path):
            try:
                stat_result = Path(path_str).stat()
            except (OSError, ValueError):
                continue
            max_mtime = max(max_mtime, stat_result.st_mtime)
        return max_mtime

    def _l1_files_changed(self) -> bool:
        """廉价检查 L1 文件是否被外部改动。stat 失败时返回 False（保留旧缓存）。"""
        try:
            return self._l1_mtime_signature() != self._l1_cache_mtime
        except Exception as e:  # noqa: BLE001 — 任何异常都不能影响读路径
            logger.debug("L1 mtime check failed, keeping cache: %s", e)
            return False

    # -- L4: Persona (cached, background refresh) ---------------------------

    def get_l4(self) -> str:
        """L4 persona. Cached in memory, refreshed every 30 minutes."""
        now = time.time()
        if self._l4_cache is not None and (now - self._l4_cache_time) < 1800:
            return self._l4_cache

        p = Path(self._config.l4_persona_path)
        if p.exists():
            try:
                self._l4_cache = p.read_text(encoding="utf-8", errors="replace").strip()
            except Exception:
                self._l4_cache = ""
        else:
            self._l4_cache = ""
        self._l4_cache_time = now
        return self._l4_cache

    def refresh_l4_background(self) -> None:
        """Trigger background L4 refresh."""
        threading.Thread(target=self.get_l4, daemon=True, name="l4-refresh").start()

    # -- L2: Semantic search (LanceDB) -------------------------------------

    def _search_l2(self, query: str) -> List[RecallResult]:
        """Search L2 semantic memory.

        Strategy:
        - If LanceDB + embedding model available: vector search
        - If LanceDB available but no embedding: FTS-like text scan
        - If LanceDB not available: return empty (L2 disabled)
        """
        try:
            # 双重检查锁懒初始化：冷启动并发时只加载一次嵌入模型。
            # _l2_init_attempted 保证失败后不再重试（否则每次查询都要 import
            # lancedb + 尝试加载模型，是默认部署形态下的固定开销）。
            if not self._l2_init_attempted:
                with self._l2_lock:
                    if not self._l2_init_attempted:
                        self._init_l2()
            if self._l2_store is None:
                return []

            # Try vector search first (requires embedding model)
            if self._embed_fn is not None:
                return self._search_l2_vector(query)

            # Fallback: scan table for text matches (slow but works)
            return self._search_l2_text(query)

        except Exception as e:
            logger.debug("L2 search failed: %s", e)
            return []

    def _search_l2_vector(self, query: str) -> List[RecallResult]:
        """L2 vector search using embedding model."""
        try:
            query_vec = self._embed_fn(query)
            # metric 必须显式指定 cosine：LanceDB 默认是 L2 欧氏距离，其量纲与
            # distance_to_score() 的 [0, 2] 假设不符（旧 P0 的根因之一）。
            results = (
                self._l2_store.search(query_vec)
                .metric("cosine")
                .limit(self._config.recall.l2_max_results)
                .to_list()
            )
            return [
                RecallResult(
                    layer="l2",
                    content=r.get("content", ""),
                    # 余弦距离 → 相关性分数的唯一换算入口（见 distance_to_score
                    # 的 P0 说明）。_distance 缺失时按正交兜底（score = 0.5）。
                    score=row_to_score(r),
                    source="lance",
                    metadata={"category": r.get("category", ""),
                              "source_rowid": r.get("source_rowid")},
                )
                for r in results
            ]
        except Exception as e:
            logger.debug("L2 vector search failed: %s", e)
            return []

    def _search_l2_text(self, query: str) -> List[RecallResult]:
        """L2 text fallback: scan all rows for substring matches."""
        try:
            table = self._l2_store.to_arrow()
            if table.num_rows == 0:
                return []

            content_col = table.column("content").to_pylist()
            has_ref = "source_rowid" in table.column_names
            ref_col = table.column("source_rowid").to_pylist() if has_ref else [None] * table.num_rows
            query_lower = query.lower()
            results = []
            for i, content in enumerate(content_col):
                if query_lower in content.lower():
                    # Score: more matches = higher score
                    match_count = content.lower().count(query_lower)
                    score = min(0.5 + match_count * 0.1, 1.0)
                    results.append(RecallResult(
                        layer="l2",
                        content=content,
                        score=score,
                        source="lance",
                        metadata={"row": i, "source_rowid": ref_col[i]},
                    ))
            # Sort descending by score (more matches first)
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:self._config.recall.l2_max_results]
        except Exception as e:
            logger.debug("L2 text search failed: %s", e)
            return []

    def _init_l2(self) -> None:
        """Lazy-initialize LanceDB connection and embedding model.

        Embedding priority (统一由 :class:`EmbeddingService` 决定，避免读写两端
        各自维护一份降级阶梯):
        1. SiliconFlow/OpenAI API (if configured in governed_memory.json)
        2. fastembed / sentence-transformers (local, per config.vector.backend)
        3. Text fallback (no embedding)

        无论成败都先把 ``_l2_init_attempted`` 置 True：初始化失败后后续调用直接
        返回 []，不再重复 import / 加载模型。

        唯一例外是"L2 目录还不存在"（写路径尚未建表）：这不是失败，只是时机未
        到，重置标记以便以后重试 —— 重试成本只有一次 ``Path.exists()``。
        """
        self._l2_init_attempted = True

        # 只有尚未拿到嵌入函数时才重新探测后端 —— 不覆盖已加载的模型
        if self._embed_fn is None:
            service = EmbeddingService.get(self._config)
            self._embed_service = service
            self._embed_model = None
            if service.available:
                self._embed_fn = service.embed_one
            else:
                self._embed_fn = None
                log_degraded(
                    "l2",
                    "embedding_unavailable",
                    detail=service.last_error or "no embedding backend",
                )

        try:
            import lancedb
        except ImportError:
            logger.debug("lancedb not installed, L2 disabled")
            log_degraded("l2", "lancedb_missing", detail="L2 falls back to L1+L3")
            return

        try:
            db_path = self._config.l2_db_path
            if not Path(db_path).exists():
                # 数据库目录尚未创建 —— 允许以后重试（成本仅一次 Path.exists）
                self._l2_init_attempted = False
                return
            db = lancedb.connect(db_path)
            tables = db.list_tables().tables
            if "memories" not in tables:
                logger.debug("L2 table 'memories' not found, L2 disabled")
                return

            self._l2_store = db.open_table("memories")
            schema = self._l2_store.schema
            has_vector = any(f.name == "vector" for f in schema)
            if not has_vector:
                # 表里没有 vector 列，只能走文本扫描
                logger.debug("L2 table has no vector column, text search only")
                self._embed_fn = None

        except Exception as e:
            logger.debug("L2 init failed: %s", e)
            log_degraded("l2", "init_failed", detail=str(e), exc=e)

    # -- L3: Conversation archive (SQLite FTS5) ----------------------------

    def _search_l3(self, query: str) -> List[RecallResult]:
        """Search L3 conversation archive via SQLite FTS5 with time decay.

        整个调用只开一个只读连接，并复用给 CJK 的 LIKE 回退分支 —— 混合查询
        "python 今天 天气 北京 部署" 过去会因为逐段调用 ``_search_l3_like`` 而
        开 5 个连接、做 5 次全表扫描。
        """
        db_path = self._config.l3_db_path
        if not Path(db_path).exists():
            return []

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        except Exception as e:
            logger.debug("L3 connect failed: %s", e)
            return []

        try:
            conn.row_factory = sqlite3.Row

            # Check if FTS5 table exists
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]

            if "messages_fts" not in tables:
                return []

            # Escape special FTS5 characters and build query
            fts_query = self._build_fts_query(query)

            results: List[RecallResult] = []

            # FTS5 covers the non-CJK terms; CJK 走 LIKE 回退
            if fts_query:
                results.extend(self._search_l3_fts(conn, fts_query))

            # CJK: 整句交给 _search_l3_like，由它分段后一次 OR 匹配
            if self._has_cjk(query):
                seen = {r.content for r in results}
                for item in self._search_l3_like(query, conn=conn):
                    if item.content not in seen:
                        results.append(item)
                        seen.add(item.content)

            # Normalize scores to 0~1 (max -> 1.0) so L3 scores are
            # comparable with L2 scores and survive MIN_SCORE filtering.
            # FTS5 rank absolute values are tiny (1e-6 scale) which would
            # otherwise be wiped out by any absolute threshold.
            if results:
                max_score = max(r.score for r in results)
                if max_score > 0:
                    for r in results:
                        r.score = r.score / max_score

            return results
        except Exception as e:
            logger.debug("L3 search failed: %s", e)
            return []
        finally:
            self._close_quietly(conn)

    def _search_l3_fts(self, conn: sqlite3.Connection, fts_query: str) -> List[RecallResult]:
        """Run the FTS5 branch on an existing connection (English / non-CJK terms).

        ``messages_fts`` 自带 ``role`` 列，直接加 ``AND role IN (...)`` 过滤 tool
        角色即可，无需 join 回 ``messages``。
        """
        results: List[RecallResult] = []
        role_placeholders = ",".join("?" * len(L3_RECALL_ROLES))
        try:
            rows = conn.execute(
                "SELECT rowid, content, rank, timestamp FROM messages_fts "
                f"WHERE messages_fts MATCH ? AND role IN ({role_placeholders}) "
                "ORDER BY rank LIMIT ?",
                (fts_query, *L3_RECALL_ROLES, self._config.recall.l3_max_results),
            ).fetchall()
        except Exception as e:
            logger.debug("L3 FTS5 search failed: %s", e)
            return results

        now = time.time()
        half_life_seconds = self._config.recall.l3_time_decay_hours * 3600

        # Collect raw ranks first for min-max normalization
        raw_ranks = [abs(row["rank"]) if row["rank"] else 0.0 for row in rows]
        min_rank = min(raw_ranks) if raw_ranks else 0.0
        max_rank = max(raw_ranks) if raw_ranks else 1.0
        rank_range = max_rank - min_rank

        for row, raw_rank in zip(rows, raw_ranks):
            # FTS5 bm25 rank: negative, LARGER |rank| = more relevant.
            # Min-max normalize: best result (largest |rank|) → score 1.0.
            if rank_range > 0:
                score = (raw_rank - min_rank) / rank_range  # [0, 1], best=1.0
            else:
                score = 0.8  # All same rank → equal score
            try:
                ts = float(row["timestamp"]) if row["timestamp"] else now
                age_seconds = max(0, now - ts)
                age_factor = math.exp(-0.693 * age_seconds / half_life_seconds)
            except (ValueError, TypeError):
                age_factor = 1.0
            results.append(RecallResult(
                layer="l3",
                content=row["content"],
                score=score * age_factor,
                source="sqlite",
            ))
        return results

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Build a safe FTS5 query from user input.

        Returns FTS5 query for English terms only.
        CJK terms are stripped — caller should use LIKE fallback for them.
        """
        if not query or not query.strip():
            return ""

        special_chars = set('*"<>+-^()~')

        words = query.split()
        if not words:
            return ""

        # Separate CJK and non-CJK terms
        fts_terms = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            # Check if this word contains CJK
            has_cjk = any('\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' for c in word)
            if has_cjk:
                continue  # Skip CJK — caller uses LIKE
            escaped = "".join("\\" + c if c in special_chars else c for c in word)
            fts_terms.append(f'"{escaped}"')

        if not fts_terms:
            return ""  # All CJK — signal caller to use LIKE

        return " OR ".join(fts_terms)

    @staticmethod
    def _has_cjk(text: str) -> bool:
        """Check if text contains CJK characters."""
        return any('\u4e00' <= c <= '\u9fff' or '\u3000' <= c <= '\u303f' for c in text)

    @staticmethod
    def _escape_like(text: str) -> str:
        """Escape LIKE wildcards (%, _) in text."""
        return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

    def _search_l3_like(self, query: str,
                        conn: Optional[sqlite3.Connection] = None) -> List[RecallResult]:
        """L3 LIKE fallback for CJK queries that FTS5 can't handle.

        不再盲目去掉空格（"天气 北京" → "天气北京" 要求两词在原文中相邻，实测
        对 "今天天气不错，北京的空气质量…" 返回 0 条）。改为把查询切成片段后
        **逐段 OR 匹配**：先按空白分词，再把每个词切成 CJK / 非 CJK 连续段。
        单个段内部保持原样匹配，不去空格。

        ``conn`` 允许调用方（``_search_l3``）复用同一个只读连接；谁创建谁关闭。
        """
        db_path = self._config.l3_db_path
        if not Path(db_path).exists():
            return []

        segments = self._split_like_segments(query)
        if not segments:
            return []

        owns_conn = conn is None
        if owns_conn:
            conn = self._open_l3_readonly()
        if conn is None:
            return []

        try:
            patterns = [f"%{self._escape_like(segment)}%" for segment in segments]
            where_clause = " OR ".join(["content LIKE ? ESCAPE '\\'"] * len(patterns))
            role_placeholders = ",".join("?" * len(L3_RECALL_ROLES))
            # 括号保证语义是 (片段1 OR 片段2 ...) AND role IN (...)，
            # 而不是把 role 条件并进 OR 链（那样会让任意 tool 行都被命中）。
            rows = conn.execute(
                "SELECT rowid, content, timestamp FROM messages "
                f"WHERE ({where_clause}) AND role IN ({role_placeholders}) "
                "ORDER BY timestamp DESC LIMIT ?",
                (*patterns, *L3_RECALL_ROLES, self._config.recall.l3_max_results),
            ).fetchall()
        except Exception as e:
            logger.debug("L3 LIKE search failed: %s", e)
            return []
        finally:
            if owns_conn:
                self._close_quietly(conn)

        now = time.time()
        half_life_seconds = self._config.recall.l3_time_decay_hours * 3600
        results: List[RecallResult] = []
        for row in rows:
            content = row["content"] or ""
            try:
                ts = float(row["timestamp"]) if row["timestamp"] else now
                age_seconds = max(0, now - ts)
                age_factor = math.exp(-0.693 * age_seconds / half_life_seconds)
            except (ValueError, TypeError):
                age_factor = 1.0
            # LIKE 没有 rank 信号：命中片段越多分越高（0.5~0.8）。
            # 单片段全命中 = 0.8 × 时间衰减，与历史行为一致。
            matched = sum(1 for segment in segments if segment in content)
            ratio = min(1.0, matched / len(segments))
            results.append(RecallResult(
                layer="l3",
                content=content,
                score=(0.5 + 0.3 * ratio) * age_factor,
                source="sqlite",
            ))

        return results

    def _split_like_segments(self, query: str) -> List[str]:
        """把查询拆成可 OR 匹配的片段。

        - 先按空白分词："天气 北京" → ["天气", "北京"]
        - 每个词再按 CJK / 非 CJK 切成连续段："python3.11发布" → ["python3.11", "发布"]
        - 单片段查询保留原样（段内不去空格）
        - 多片段时丢弃单字段（"的"/"了"/"a" 这类噪声会淹没 LIMIT）
        """
        if not query or not query.strip():
            return []

        segments: List[str] = []
        for token in query.split():
            segments.extend(run for run in self.SEGMENT_PATTERN.findall(token) if run)

        # 去重保序
        unique: List[str] = []
        seen: set = set()
        for segment in segments:
            if segment not in seen:
                seen.add(segment)
                unique.append(segment)

        if len(unique) > 1:
            filtered = [s for s in unique if len(s) >= 2]
            if filtered:
                unique = filtered

        return unique[:self.LIKE_SEGMENT_LIMIT]

    def _open_l3_readonly(self) -> Optional[sqlite3.Connection]:
        """打开一个 L3 只读连接；失败返回 None（不抛异常）。"""
        try:
            conn = sqlite3.connect(f"file:{self._config.l3_db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            logger.debug("L3 LIKE connect failed: %s", e)
            return None

    @staticmethod
    def _close_quietly(conn: Optional[sqlite3.Connection]) -> None:
        """关闭连接并吞掉所有异常（连接关闭失败不该影响读路径）。"""
        if conn is None:
            return
        try:
            conn.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("L3 connection close failed: %s", e)

    # -- Merge and rank ----------------------------------------------------

    def parallel_recall(self, query: str) -> List[RecallResult]:
        """Run L2 + L3 + L4 in parallel, merge results by score.

        Each call creates its own temporary pool so concurrent callers
        never compete for worker slots.  The pool is explicitly shut
        down with wait=False so orphan threads die with the process
        and the caller is never blocked by slow tasks.
        Falls back to sequential execution if shutdown has been called.
        """
        if self._shutdown:
            return self._sequential_recall(query)

        results: List[RecallResult] = []
        collected: set = set()
        timeout = self._config.recall.parallel_timeout_seconds

        pool = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="recall",
        )
        try:
            futures = {
                pool.submit(self._search_l2, query): "l2",
                pool.submit(self._search_l3, query): "l3",
                pool.submit(self._get_l4_result): "l4",
                pool.submit(self._search_kb, query): "kb",
            }

            try:
                for future in as_completed(futures, timeout=timeout):
                    layer = futures[future]
                    try:
                        layer_results = future.result(timeout=0)
                        if isinstance(layer_results, list):
                            results.extend(layer_results)
                        elif isinstance(layer_results, RecallResult):
                            results.append(layer_results)
                        collected.add(id(future))
                    except Exception as e:
                        logger.debug("Recall layer %s failed: %s", layer, e)
                        collected.add(id(future))
            # concurrent.futures.TimeoutError 在 Python 3.10 不是内建 TimeoutError
            # 的子类（3.11 才合并为别名），只写 `except TimeoutError` 在 3.10 上是
            # 死代码：超时会逃逸到外层被 debug 吞掉，静默返回空结果。
            except (TimeoutError, FuturesTimeoutError):
                logger.debug("Recall timeout after %.1fs, collected %d results", timeout, len(results))
                for future, layer in futures.items():
                    if id(future) not in collected and future.done():
                        try:
                            layer_results = future.result(timeout=0)
                            if isinstance(layer_results, list):
                                results.extend(layer_results)
                            elif isinstance(layer_results, RecallResult):
                                results.append(layer_results)
                        except Exception:
                            pass
        finally:
            # shutdown(wait=False): never block the caller.
            # Orphan threads die when the process exits.
            pool.shutdown(wait=False)

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _sequential_recall(self, query: str) -> List[RecallResult]:
        """Fallback sequential recall when thread pool is unavailable."""
        results: List[RecallResult] = []
        for fn in (self._search_l2, self._search_l3, self._get_l4_result, self._search_kb):
            try:
                r = fn(query) if fn != self._get_l4_result else fn()
                if isinstance(r, list):
                    results.extend(r)
                elif isinstance(r, RecallResult):
                    results.append(r)
            except Exception as e:
                logger.debug("Sequential recall layer failed: %s", e)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _search_kb(self, query: str) -> List[RecallResult]:
        """知识库提示通道（layer="kb"）。

        只产出**提示**（笔记标题 + 路径），**不注入正文** —— 这是标定后的
        刻意取舍：vault 语料同质化（26 篇里 16 篇是会议记录），实测相关/无关
        的融合分分离带只有 +0.039（无关最高 0.4487 / 相关最低 0.4870），
        任何绝对阈值都脆弱；而"注入全文"一旦误报就要付出几百 token 和
        上下文污染。改为注入一行提示后，误报成本降到 ~40 token，于是可以
        用宽松阈值换高覆盖率，把"要不要深读"的决定权交给 agent
        （``governed_kb_get`` / ``governed_kb_search``）。

        未注入 KB（``kb=None``）或 ``kb.recall_min_score <= 0`` 时静默返回空，
        保持旧行为不变。
        """
        kb = getattr(self, "_kb", None)
        if kb is None:
            return []
        kb_cfg = getattr(self._config, "kb", None)
        if kb_cfg is None or not bool(getattr(kb_cfg, "recall_hint_enabled", True)):
            return []
        min_score = float(getattr(kb_cfg, "recall_min_score", 0.0) or 0.0)
        if min_score <= 0.0:
            return []
        limit = max(1, int(getattr(kb_cfg, "recall_max_notes", 3) or 3))
        try:
            hits = kb.search(query, top_k=limit)
        except Exception as e:  # noqa: BLE001 — 提示通道失败绝不能影响主召回
            logger.debug("KB recall failed: %s", e)
            return []

        out: List[RecallResult] = []
        for h in hits:
            score = float(h.get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            title = str(h.get("title") or "").strip()
            path = str(h.get("path") or "").strip()
            if not title and not path:
                continue
            out.append(RecallResult(
                layer="kb",
                content=title or Path(path).stem,
                score=score,
                source=path,
                metadata={
                    "kind": h.get("kind", ""),
                    "section": h.get("section", ""),
                    "keyword_score": h.get("keyword_score", 0.0),
                    "semantic_score": h.get("semantic_score", 0.0),
                },
            ))
        return out

    def _get_l4_result(self) -> Optional[RecallResult]:
        """Get L4 persona as a recall result."""
        l4 = self.get_l4()
        if not l4:
            return None
        return RecallResult(
            layer="l4",
            content=l4,
            score=1.0,  # L4 always has highest priority
            source="persona",
        )

    def format_recall(self, results: List[RecallResult], l1_text: str,
                      l1_budget: int, l23_budget: int,
                      l2_min_score: "float | None" = None) -> str:
        """Format recall results into injectable context text.

        L1 gets its own fixed budget (never displaced).
        L2/L3 share the remaining budget.
        L4 is injected via system_prompt_block, not here.

        Args:
            l2_min_score: 可选的 L2 门槛覆盖值；``None`` 时读
                ``config.recall.l2_min_score``（见 :func:`layer_score_floor`）。
        """
        parts = []

        # L1: always included, fixed budget
        if l1_text:
            truncated_l1 = self._truncate_to_tokens(l1_text, l1_budget)
            parts.append(f"[User Rules]\n{truncated_l1}")

        # KB: 独立小预算的**提示**段（不占 L2/L3 预算）。
        # 只给标题+路径，不注入正文 —— 理由见 RecallEngine._search_kb。
        # 这一段是「wiki 被真正用起来」的入口：此前 25 篇笔记躺在 vault 里，
        # 召回链路完全不碰 KB，agent 既不知道它们存在也无从索取。
        kb_items = [r for r in results if r.layer == "kb"]
        if kb_items:
            kb_items.sort(key=lambda r: r.score, reverse=True)
            kb_limit = 3
            kb_cfg = getattr(getattr(self, "_config", None), "kb", None)
            if kb_cfg is not None:
                kb_limit = max(1, int(getattr(kb_cfg, "recall_max_notes", 3) or 3))
            lines = []
            for item in kb_items[:kb_limit]:
                src = f" ({item.source})" if item.source else ""
                lines.append(f"- 《{item.content}》{src}")
            parts.append(
                "[Knowledge] 知识库中有相关笔记，需要细节时用 governed_kb_get 读取全文：\n"
                + "\n".join(lines)
            )

        # L2/L3: shared budget, skip low-relevance items, then dedup by content.
        # 两层门槛分开取：L2 用 recall.l2_min_score（未配置回退 MIN_SCORE），
        # L3 恒用 MIN_SCORE —— L3 的 FTS5 分数本就偏低（0.3~0.7），套用 L2
        # 的高门槛会把它整层砍空（实测 82/91 条被误删）。
        recall_cfg = getattr(getattr(self, "_config", None), "recall", None)
        l2_floor = layer_score_floor("l2", recall_cfg, l2_min_score=l2_min_score)
        l3_floor = layer_score_floor("l3", recall_cfg)
        l23_items = [
            r for r in results
            if r.layer in ("l2", "l3")
            and r.score >= (l2_floor if r.layer == "l2" else l3_floor)
        ]

        # 按 content 去重，只保留最高分那条：实测 L2 重复率 94.8%，
        # 同一事实的副本会彼此抢占 l23_budget，把真正多样的记忆挤出去。
        best_by_content: Dict[str, RecallResult] = {}
        for item in l23_items:
            current = best_by_content.get(item.content)
            if current is None or item.score > current.score:
                best_by_content[item.content] = item

        # 显式按分数降序，不依赖调用方：parallel_recall 已经排好序，但
        # format_recall 也会被直接调用（测试 / 其它入口），必须先排序再进预算
        # 循环，否则低分条目会先占满预算、把高分条目 break 掉。
        l23_items = sorted(best_by_content.values(), key=lambda r: r.score, reverse=True)

        if l23_items:
            budget_chars = l23_budget * 4  # rough token→char
            used = 0
            selected = []
            for item in l23_items:
                item_chars = len(item.content)
                if used + item_chars > budget_chars:
                    break
                selected.append(item)
                used += item_chars

            if selected:
                lines = []
                for item in selected:
                    tag = "Fact" if item.layer == "l2" else "History"
                    lines.append(f"[{tag}] {item.content}")
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def shutdown(self) -> None:
        """Mark as shut down so parallel_recall falls back to sequential."""
        self._shutdown = True

    @staticmethod
    def _truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Rough truncation: 1 token ≈ 4 chars for English, ~2 chars for CJK."""
        max_chars = max_tokens * 3  # conservative estimate
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n...[truncated]"
