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
from typing import Any, Dict, List, Optional, Tuple

from ._persona import persona_only

from ._config import GovernedMemoryConfig
from ._embedding import EmbeddingService
from ._lifecycle import resolve_memory_dir, usage_touch

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

#: KB 提示通道向 :meth:`KnowledgeBase.search` 申请的候选池大小。
#:
#: 必须显著大于 ``recall_max_notes``：候选先按融合分（0.4*kw + 0.6*sem）排序，
#: 而纯关键词命中的融合分被 0.4 的权重压到 <= 0.4，在普遍 0.5~0.9 的候选里排
#: 不进前 3 —— 不放开候选池，关键词通道就算放行了也取不到（实测那篇漏掉的
#: 笔记排在全库第 27 位）。放大候选池**不增加 embedding 调用**（每条查询仍然
#: 只嵌一次），代价只是 LanceDB 多返回几行。
_RECALL_KB_POOL_FACTOR: int = 10
_RECALL_KB_POOL_MIN: int = 30

#: 项目作用域生效时，向量检索的候选放大倍数（仅在 `where` 预过滤不可用时使用）。
#:
#: LanceDB 的 `search().where(..., prefilter=True)` 是**先过滤再取 top-N**，
#: 与 CLI 的全表扫描语义完全一致，所以正常情况下不需要放大。只有在 prefilter
#: 不可用（旧版 LanceDB）时，才退化为「多取候选 → Python 侧过滤 → 截断」：
#: 此时放大倍数越小，越可能因为其它项目的行占满名额而漏掉本项目的行。
_L2_SCOPE_OVERFETCH: int = 5


def _project_of(row: Any) -> str:
    """行里的 project 归一成 ``str``；NULL / 空 → ``""``（= 全局事实）。"""
    value = row.get("project") if hasattr(row, "get") else None
    return str(value) if value else ""


def l2_project_allows(scope: Optional[str], row_project: Any) -> bool:
    """行级项目过滤，语义与 ``memory_cli.recall_l2`` 一致。

    CLI 的原文是 ``if project and pj and pj != project: continue`` —— 也就是
    「保留本项目 **+** 全局（``project IS NULL``）」：笔记本里的通用知识
    （"SecretStore 跑在 NAS 上"）在任何项目下问都成立，而**别的项目**的事实
    才是必须挡住的。这里刻意复用同一套语义而不是另发明一套，否则同一个问题
    在 CLI 与插件两条读路径上又会得出两个答案 —— 那正是本次要修的缺陷本身。

    三态：
        * ``scope is None`` —— 不过滤（向后兼容，行为与修复前一致）
        * ``scope == ""`` —— 只看全局（CLI 读路径表达不出的那个取值）
        * 其它 —— 该项目 + 全局
    """
    if scope is None:
        return True
    pj = str(row_project) if row_project else ""
    if not scope:
        return not pj
    return not pj or pj == scope


def l2_project_where(scope: str) -> str:
    """把作用域翻成 LanceDB 的 ``where`` 谓词（配 ``prefilter=True`` 用）。

    单引号按 SQL 字面量转义（``'`` → ``''``），与 ``_kb._sql_literal`` 同一套
    规矩：项目名来自目录名，虽然通常很干净，但不能让一个含引号的目录名把谓词
    拼坏。
    """
    if not scope:
        return "project IS NULL"
    literal = "'" + str(scope).replace("'", "''") + "'"
    return f"project IS NULL OR project = {literal}"


def layer_score_floor(layer: str, recall_cfg=None, *,
                      l2_min_score: "float | None" = None) -> float:
    """返回指定记忆层的最低相关分门槛（``>= floor`` 才会被注入）。

    L2 与 L3 的分数分布不同源，必须分开设阈值：

    * **L3**（FTS5 关键词 + 时间衰减）：有效命中实测 0.3~0.7，门槛沿用
      :data:`MIN_SCORE`（0.1）。
    * **L2**（1024 维，经 :func:`distance_to_score` 映射 ``1 - d/2``，
      **不是**原始余弦相似度）：无关查询的地板分就有 0.73（score 标尺；
      对应原始 cos ≈ 0.46）。门槛取 ``recall.l2_min_score``；未配置
      （``0.0``）时**回退**到 :data:`MIN_SCORE`，保证老配置行为不变。

    ⚠️ 0.73 是 2026-09-16 在 **21 条精选**语料上的旧标定，别当普适数。
    2026-09-22 用独立构造的查询集重测（每条相关查询的正确行都指名在库里）：
    噪声上沿在 33 行精选集上 0.7686、在 1632 行（含 rebuild 碎片）上
    **0.8914**，而相关下沿只有 0.7427 / 0.7506 ⇒ **两带始终重叠**，不存在
    能全挡噪声又全保相关的阈值。0.76 只作粗筛，其数值本身不要动；完整测量
    见 ``_config.RecallConfig.l2_min_score`` 与
    ``memory_cli.DEFAULT_L2_SEMANTIC_FLOOR`` 的注释。

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
    except Exception as e:  # noqa: BLE001 - 行的 get 可能抛任意异常
        logger.debug("L2 row _distance lookup failed, using default: %s", e)
        distance = DEFAULT_COSINE_DISTANCE
    return distance_to_score(distance)


#: L2 双路融合的 RRF 衰减常数（与 ``_kb._RRF_K`` 同值；两模块互不 import
#: —— ``_kb`` 已经依赖本模块，反向引入会成环）。
_L2_RRF_K = 60

#: RRF 两路等权：向量与词法各 0.5。没有先验说明哪路更可信 —— 向量长于
#: 同义改写，词法长于精确术语，让排名说话。
_L2_RRF_W_VEC = 0.5
_L2_RRF_W_LEX = 0.5


def _fuse_l2_channels(vec: List["RecallResult"],
                      lex: List["RecallResult"]) -> List["RecallResult"]:
    """向量 + 词法两路按加权 RRF 融合（``recall.l2_fusion`` 开启时）。

    分数语义（与 ``_kb._rrf_fuse`` 的单路上限约定不同 —— 这里按**实际出现
    的通道**归一化）：单通道第一名 = 1.0，双通道第一名 = 1.0。``score``
    表达「在能看到的通道里排第几」，是排名量纲。

    原生相似度另存 ``metadata["native_score"]``（向量 =
    :func:`distance_to_score` 的 score 标尺 ``(1+cos)/2``，**不是**原始余弦；
    词法 = 子串分；双路 = 取大）—— ``recall.l2_min_score``（0.76，同一 score
    标尺）在 ``format_recall`` 里对它生效。拿这个标尺去量 RRF 排名分会把整层
    砍空，这是把门槛留在原生分上的原因：**排序用融合分，准入看原生分**。

    同一事实（``source_rowid`` 相同，缺失时退化为内容相同）在两路中合并为
    一条，``metadata["trace"]`` 记下两路名次 —— 「双通道确认」是可诊断的。
    """

    def _key(r: "RecallResult") -> tuple:
        rid = (r.metadata or {}).get("source_rowid")
        return ("rid", rid) if rid is not None else ("content", r.content)

    def _ranks(results: List["RecallResult"]) -> Dict[tuple, int]:
        ordered = sorted(results, key=lambda r: r.score, reverse=True)
        return {_key(r): i + 1 for i, r in enumerate(ordered)}

    vec_ranks = _ranks(vec)
    lex_ranks = _ranks(lex)
    vec_by_key = {_key(r): r for r in vec}
    lex_by_key = {_key(r): r for r in lex}

    out: List[RecallResult] = []
    for key in set(vec_by_key) | set(lex_by_key):
        in_vec, in_lex = key in vec_by_key, key in lex_by_key
        # 按实际出现的通道归一化：分母 = 出现通道的权重和在第一名处的 RRF。
        raw = 0.0
        present_w = 0.0
        v_rank = vec_ranks.get(key)
        l_rank = lex_ranks.get(key)
        if in_vec:
            raw += _L2_RRF_W_VEC / (_L2_RRF_K + v_rank)
            present_w += _L2_RRF_W_VEC
        if in_lex:
            raw += _L2_RRF_W_LEX / (_L2_RRF_K + l_rank)
            present_w += _L2_RRF_W_LEX
        fused = min(1.0, raw / (present_w / (_L2_RRF_K + 1)))

        # 原生分 = 各**出现**通道的原生分取大（任一通道证据足够强即过门槛）。
        # 注意不能写成先取字典值再按 present 过滤 —— 缺席通道的键访问会在
        # 判断之前先抛 KeyError。
        natives = []
        if in_vec:
            natives.append(vec_by_key[key].score)
        if in_lex:
            natives.append(lex_by_key[key].score)
        native = max(natives)
        # 主记录优先取向量侧（多带 category 等字段），内容两路同源一致。
        primary = vec_by_key[key] if in_vec else lex_by_key[key]
        meta = dict(primary.metadata or {})
        meta["native_score"] = round(native, 4)
        meta["trace"] = {
            "fusion": "rrf",
            "vector_rank": v_rank,
            "lexical_rank": l_rank,
            "channels": ([c for c, on in (("vector", in_vec), ("lexical", in_lex))
                          if on]),
        }
        out.append(RecallResult(
            layer="l2",
            content=primary.content,
            score=round(fused, 4),
            source=primary.source,
            metadata=meta,
        ))
    # 分数降序；同分按内容字典序稳定排序 —— 上游按 set 并集迭代，
    # 不加 tiebreak 同分顺序会跨进程漂移（测试与 trace 都要求可复现）。
    out.sort(key=lambda r: (-r.score, r.content))
    return out


def _l2_admitted(r, l2_floor: float) -> bool:
    """L2 单条准入：融合命中按**原生分**，未融合的按 ``score``。

    ``recall.l2_fusion`` 开启后 L2 的 ``score`` 是 RRF 排名量纲，拿 score 标尺
    （``(1+cos)/2``，部署值 0.76）去量它会把整层砍空 —— 所以带 ``native_score``
    的融合命中按原生分（向量 = ``(1+cos)/2`` / 词法 = 子串）过门槛，排序仍按
    融合分：
    **排序用融合分、准入看原生分**。未融合的历史命中没有 ``native_score``，
    仍按 ``score``，行为不变。

    **只有这一份实现**：``format_recall``（实际注入）与 ``admitted_l2``
    （评测门）都调它 —— 门槛一旦有两份，评测门与实际召回就会给出两个答案。
    """
    native = (r.metadata or {}).get("native_score")
    effective = float(native) if native is not None else r.score
    return effective >= l2_floor


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
        #: 最近一次 L2 检索实际生效的项目作用域（``None`` = 未过滤）。
        #: 供调用方/诊断读取：作用域生效这件事必须可观测，不能只是静默过滤。
        self.last_l2_scope: Optional[str] = None
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
        """L4 persona. Cached in memory, refreshed every 30 minutes.

        The generated listings are stripped here, at the single read point, so
        that **both** consumers are covered at once: ``system_prompt_block`` and
        the ``score=1.0`` L4 recall result, which is part of every parallel
        recall. Measured 2026-09-17: ``persona.md`` ends with a dump of all 23 L2
        facts, so every session opened with 20 of them already in context and
        nothing for the memory tools to be needed for. See ``_persona.py``.

        Stripping on read, not on write: ``_bridge.py`` skips these same
        headings when collecting promotion candidates, so the file must keep
        them — only what an agent is *told* changes here.
        """
        now = time.time()
        if self._l4_cache is not None and (now - self._l4_cache_time) < 1800:
            return self._l4_cache

        p = Path(self._config.l4_persona_path)
        if p.exists():
            try:
                raw = p.read_text(encoding="utf-8", errors="replace").strip()
                self._l4_cache = persona_only(raw)
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

    def _search_l2(self, query: str, project: Optional[str] = None) -> List[RecallResult]:
        """Search L2 semantic memory.

        Strategy:
        - If LanceDB + embedding model available: vector search
        - If LanceDB available but no embedding: FTS-like text scan
        - If LanceDB not available: return empty (L2 disabled)

        Args:
            project: 项目作用域，三态见 :func:`l2_project_allows`
                （``None`` = 不过滤 / ``""`` = 只看全局 / ``"名字"`` = 该项目+全局）。
                调用方通常传 ``config.recall.project_scope``；默认 ``None``
                保持修复前的行为不变。

        作用域生效时会留下可观测痕迹（缺了这一条，就只是把「两套答案」换成
        「一套答案但没人知道被过滤过」）：
        - ``self.last_l2_scope`` 记下这次实际生效的作用域；
        - 每条命中的 ``metadata["project"]`` 带上它自己的归属（全局行不带）；
        - 有过滤时打一条 ``logger.info``。
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

            self.last_l2_scope = project
            if project is not None:
                logger.info("L2 recall narrowed to project scope %r "
                            "(globals always included)", project)

            # 双路融合（recall.l2_fusion 开启且嵌入可用）：向量与词法都跑。
            # 关闭时保持历史链路：向量优先，不可用才退词法（二选一）。
            if bool(getattr(self._config.recall, "l2_fusion", False)) \
                    and self._embed_fn is not None:
                return self._search_l2_fused(query, project)

            # Try vector search first (requires embedding model)
            if self._embed_fn is not None:
                return self._search_l2_vector(query, project)

            # Fallback: scan table for text matches (slow but works)
            return self._search_l2_text(query, project)

        except Exception as e:
            logger.debug("L2 search failed: %s", e)
            return []

    def _search_l2_fused(self, query: str,
                         project: Optional[str] = None) -> List[RecallResult]:
        """双路 RRF 融合检索（``recall.l2_fusion`` 开启时的 L2 主链路）。

        两路都跑、按排名融合（见 :func:`_fuse_l2_channels`）：精确子串命中
        的事实不再因为向量通道可用而被整体跳过 —— 旧链路是二选一。

        任一路为空时直接返回另一路（不值得为单路开融合），此时 ``score``
        仍是原生分、无 ``native_score``，门槛语义与历史行为一致。
        """
        try:
            vec = self._search_l2_vector(query, project)
            lex = self._search_l2_text(query, project)
            if not vec:
                return lex
            if not lex:
                return vec
            return _fuse_l2_channels(vec, lex)
        except Exception as e:
            logger.debug("L2 fused search failed: %s", e)
            return []

    def _search_l2_vector(self, query: str,
                          project: Optional[str] = None) -> List[RecallResult]:
        """L2 vector search using embedding model.

        ``project`` 非 ``None`` 时按项目过滤，语义与 CLI 一致（本项目 + 全局）。
        优先用 ``where(..., prefilter=True)``：LanceDB 在**取 top-N 之前**过滤，
        因此「先过滤再排名」与 CLI 的全表扫描完全等价。若当前 LanceDB 版本
        不支持（抛异常），退回「多取候选 → Python 侧过滤」，并记一条 degraded
        说明 —— 静默降级成一个可能漏结果的近似实现，比不降级更糟。
        """
        try:
            query_vec = self._embed_fn(query)
            search = (self._l2_store.search(query_vec, vector_column_name="vector")
                      .metric("cosine"))
            limit = int(self._config.recall.l2_max_results)
            prefilt = None
            if project is not None:
                try:
                    search = search.where(l2_project_where(project), prefilter=True)
                    prefilt = "where"
                except Exception as e:  # noqa: BLE001 — 旧版 LanceDB 没有 prefilter
                    prefilt = "post"
                    limit = limit * _L2_SCOPE_OVERFETCH
                    log_degraded(
                        "l2",
                        "project_prefilter_unsupported",
                        detail=f"{type(e).__name__}: {e}; "
                               f"falling back to over-fetch x{_L2_SCOPE_OVERFETCH} "
                               f"then filtering in Python",
                    )
            # metric 必须显式指定 cosine：LanceDB 默认是 L2 欧氏距离，其量纲与
            # distance_to_score() 的 [0, 2] 假设不符（旧 P0 的根因之一）。
            results = search.limit(limit).to_list()

            out: List[RecallResult] = []
            for r in results:
                if project is not None and prefilt == "post" \
                        and not l2_project_allows(project, r.get("project")):
                    continue
                meta = {"category": r.get("category", ""),
                        "source_rowid": r.get("source_rowid")}
                pj = _project_of(r)
                if pj:
                    meta["project"] = pj
                if project is not None:
                    meta["project_scope"] = project
                out.append(RecallResult(
                    layer="l2",
                    content=r.get("content", ""),
                    # 余弦距离 → 相关性分数的唯一换算入口（见 distance_to_score
                    # 的 P0 说明）。_distance 缺失时按正交兜底（score = 0.5）。
                    score=row_to_score(r),
                    source="lance",
                    metadata=meta,
                ))
                if len(out) >= int(self._config.recall.l2_max_results):
                    break
            return out
        except Exception as e:
            logger.debug("L2 vector search failed: %s", e)
            return []

    def _search_l2_text(self, query: str,
                        project: Optional[str] = None) -> List[RecallResult]:
        """L2 text fallback: scan all rows for substring matches.

        ``project`` 非 ``None`` 时在同一趟扫描里按项目过滤 —— 这里本来就是
        全表扫描，过滤不增加任何成本，语义与向量路径（以及 CLI）一致。
        """
        try:
            table = self._l2_store.to_arrow()
            if table.num_rows == 0:
                return []

            content_col = table.column("content").to_pylist()
            has_ref = "source_rowid" in table.column_names
            ref_col = table.column("source_rowid").to_pylist() if has_ref else [None] * table.num_rows
            has_pj = "project" in table.column_names
            pj_col = table.column("project").to_pylist() if has_pj else [None] * table.num_rows
            query_lower = query.lower()
            results = []
            for i, content in enumerate(content_col):
                if project is not None and not l2_project_allows(project, pj_col[i]):
                    continue
                if query_lower in content.lower():
                    # Score: more matches = higher score
                    match_count = content.lower().count(query_lower)
                    score = min(0.5 + match_count * 0.1, 1.0)
                    meta = {"row": i, "source_rowid": ref_col[i]}
                    pj = _project_of({"project": pj_col[i]}) if has_pj else ""
                    if pj:
                        meta["project"] = pj
                    if project is not None:
                        meta["project_scope"] = project
                    results.append(RecallResult(
                        layer="l2",
                        content=content,
                        score=score,
                        source="lance",
                        metadata=meta,
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
                pool.submit(self._search_l2, query, self.l2_scope()): "l2",
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
        scope = self.l2_scope()
        for fn in (self._search_l2, self._search_l3, self._get_l4_result, self._search_kb):
            try:
                if fn == self._get_l4_result:
                    r = fn()
                elif fn == self._search_l2:
                    r = fn(query, scope)
                else:
                    r = fn(query)
                if isinstance(r, list):
                    results.extend(r)
                elif isinstance(r, RecallResult):
                    results.append(r)
            except Exception as e:
                logger.debug("Sequential recall layer failed: %s", e)
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def l2_scope(self) -> Optional[str]:
        """当前生效的 L2 项目作用域（``config.recall.project_scope``）。

        抽成方法而不是各处直接读属性：配置段缺失 / 属性不存在时都退回 ``None``
        （不过滤），保证旧配置与替身对象的行为与修复前一致。
        """
        recall_cfg = getattr(self._config, "recall", None)
        if recall_cfg is None:
            return None
        return getattr(recall_cfg, "project_scope", None)

    def _search_kb(self, query: str) -> List[RecallResult]:
        """知识库提示通道（layer="kb"）。

        只产出**提示**（笔记标题 + 路径），**不注入正文** —— 这是标定后的
        刻意取舍：vault 语料同质化（26 篇里 16 篇是会议记录），实测相关/无关
        的融合分分离带只有 +0.039（无关最高 0.4487 / 相关最低 0.4870），
        任何绝对阈值都脆弱；而"注入全文"一旦误报就要付出几百 token 和
        上下文污染。改为注入一行提示后，误报成本降到 ~40 token，于是可以
        用宽松阈值换高覆盖率，把"要不要深读"的决定权交给 agent
        （``governed_kb_get`` / ``governed_kb_search``）。

        未注入 KB（``kb=None``）时静默返回空，保持旧行为不变。
        门槛需满足：``recall_min_score > 0`` 或 ``recall_min_kw_score > 0``
        （两者都 =0 表示通道关闭，与旧版一致）。

        **关键词通道为什么此前是死的**（P0，2026-09-17）：融合分是
        ``0.4*kw + 0.6*sem``（见 ``_kb._KB_KW_WEIGHT`` / ``_KB_SEM_WEIGHT``），
        而部署的 ``recall_min_score = 0.45`` 直接套在这个融合分上。于是**纯
        关键词命中的理论上限就是 0.4**（``kw_norm`` 满格也只有 ``0.4*1.0``），
        永远够不到 0.45 —— 关键词这一路被结构性地关死，只剩语义通道。
        现在改成**逐通道判定**：``kw >= recall_min_kw_score`` OR
        ``融合分 >= recall_min_score``。
        """
        kb = getattr(self, "_kb", None)
        if kb is None:
            return []
        kb_cfg = getattr(self._config, "kb", None)
        if kb_cfg is None or not bool(getattr(kb_cfg, "recall_hint_enabled", True)):
            return []
        min_score = float(getattr(kb_cfg, "recall_min_score", 0.0) or 0.0)
        min_kw = float(getattr(kb_cfg, "recall_min_kw_score", 0.0) or 0.0)
        if min_score <= 0.0 and min_kw <= 0.0:
            return []
        limit = max(1, int(getattr(kb_cfg, "recall_max_notes", 3) or 3))

        # 候选池必须大于 limit：融合分普遍 0.5~0.9，而纯关键词命中被 0.4 权
        # 重压到 <=0.4，先按融合分截到 limit 条就等于把它提前淘汰了（实测漏
        # 掉的那篇排在全库第 27 位）。多取候选不增加 embedding 调用 —— 每条
        # 查询仍然只嵌一次，代价只是 LanceDB 多返回几行。
        pool = max(limit * _RECALL_KB_POOL_FACTOR, _RECALL_KB_POOL_MIN)
        try:
            hits = kb.search(query, top_k=pool)
        except Exception as e:  # noqa: BLE001 — 提示通道失败绝不能影响主召回
            logger.debug("KB recall failed: %s", e)
            return []

        admitted: List[Tuple[float, str, Dict[str, Any]]] = []
        for h in hits:
            fused = float(h.get("score", 0.0) or 0.0)
            kw = float(h.get("keyword_score", 0.0) or 0.0)
            # 走哪一进门是诊断依据：出错时能直接看出是语义给高了还是关键词
            # 给高了，而不是只能看到一个融合分。
            via = ""
            if fused >= min_score > 0.0:
                via = "fused"
            elif kw >= min_kw > 0.0:
                via = "keyword"
            if not via:
                continue
            title = str(h.get("title") or "").strip()
            path = str(h.get("path") or "").strip()
            if not title and not path:
                continue
            admitted.append((max(fused, kw), via, {
                "title": title, "path": path, "hit": h, "fused": fused, "kw": kw,
            }))

        # 放行后按**放行依据**重排再截断：不重排的话，被关键词放行的笔记虽然
        # 进了候选、还是会因为融合分低而排在末尾被 trim 掉 —— 修复等于没生效。
        # 用的是 max(fused, kw)：两路都命中时取更强的那一路作为强度指标。
        admitted.sort(key=lambda row: row[0], reverse=True)

        out: List[RecallResult] = []
        for score, via, row in admitted[:limit]:
            h = row["hit"]
            out.append(RecallResult(
                layer="kb",
                content=row["title"] or Path(row["path"]).stem,
                score=score,
                source=row["path"],
                metadata={
                    "kind": h.get("kind", ""),
                    "section": h.get("section", ""),
                    "keyword_score": row["kw"],
                    "semantic_score": h.get("semantic_score", 0.0),
                    "fused_score": row["fused"],
                    "admit_via": via,
                    "index_status": h.get("index_status", {}),
                    # 排序轨迹（融合模式/两路名次/亲和度），随提示一起可见
                    "trace": h.get("trace", {}),
                },
            ))

        # 命中即计数（亲和度数据源）：只记真正注入提示的条目 —— 搜索不等于
        # 消费。touch 按 kb.affinity_enabled 门禁、失败静默，提示通道绝不因
        # 统计挂掉。
        touch = getattr(kb, "touch", None)
        if callable(touch):
            try:
                touch([r.source for r in out])
            except Exception as e:  # noqa: BLE001
                logger.debug("kb usage touch failed: %s", e)
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

        def _passes_floor(r: RecallResult) -> bool:
            if r.layer == "l3":
                return r.score >= l3_floor
            return _l2_admitted(r, l2_floor)

        l23_items = [
            r for r in results
            if r.layer in ("l2", "l3") and _passes_floor(r)
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

            # 命中即计数（容量淘汰的受害者选择依据，Phase 3）：只记真正注入
            # 上下文的事实。仅 ``sync.l2_max_items`` 开启时落表 —— 默认关闭
            # = 读路径零副作用；失败静默，统计绝不影响召回。
            if int(getattr(getattr(self._config, "sync", None),
                           "l2_max_items", 0) or 0) > 0:
                mem_dir = resolve_memory_dir(self._config)
                if mem_dir is not None:
                    usage_touch(mem_dir, [i.content for i in selected
                                          if i.layer == "l2"])

            if selected:
                lines = []
                for item in selected:
                    tag = "Fact" if item.layer == "l2" else "History"
                    lines.append(f"[{tag}] {item.content}")
                parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def admitted_l2(self, query: str, top_k: int = 5) -> List[RecallResult]:
        """L2 命中，**按本引擎的准入规则**返回（评测门用）。

        为什么不让调用方自己筛：``format_recall`` 把准入、去重、预算与排版揉
        在一起、返回的是注入文本；调用方若重算一遍门槛，就成了"一个存储两个
        答案"。这里只做「检索 → 准入 → 按 content 去重（留最高分）→ top_k」，
        不含预算裁剪，且与 ``format_recall`` 共用 ``_l2_admitted`` 同一份实现。
        """
        try:
            results = self.parallel_recall(query)
        except Exception:  # noqa: BLE001 — 评测门不该因单条查询崩掉
            return []
        recall_cfg = getattr(self._config, "recall", None)
        floor = layer_score_floor("l2", recall_cfg)
        best: Dict[str, RecallResult] = {}
        for r in results:
            if r.layer != "l2" or not _l2_admitted(r, floor):
                continue
            cur = best.get(r.content)
            if cur is None or r.score > cur.score:
                best[r.content] = r
        ranked = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return ranked[:max(1, int(top_k or 1))]

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
