# -*- coding: utf-8 -*-
"""Knowledge base over the Obsidian vault (检索 / 写入 / 索引投影).

正本 = vault（``_vault.py`` 的 Markdown + frontmatter + ``[[wikilink]]``）。
本模块提供两层：

- :class:`KBIndex` —— 向量索引**投影**（LanceDB + ``EmbeddingService``）。
  可选、可重建；lancedb / embedding 任一不可用时优雅降级，绝不抛异常。
- :class:`KnowledgeBase` —— 门面：检索（语义 + 关键词 + 反链融合）、写入、
  更新、读取、列举、重建索引。

设计要点：

- **关键词检索始终可用**（纯文件扫描，零第三方依赖）——即使向量后端缺失，
  ``search`` 也能返回结果；语义检索是增强而非必需。
- 中文查询没有空格分词，关键词打分用 bigram 重叠做兜底（title/tags/concepts
  短文本上做），整串命中在 body 上做 ``in`` 匹配。
- 写入时对 ``concepts`` 里的词在正文首次出现处自动加 ``[[词]]`` 双向链接，
  这是「Obsidian 双向链接利用起来」的核心动作。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ._config import GovernedMemoryConfig
from ._embedding import EmbeddingService
from ._bridge import has_secret_like_text
from ._llm import chat_completion
from ._recall import row_to_score
from . import _vault

logger = logging.getLogger(__name__)

try:  # pragma: no cover - _diag.py 尚未落地时的兼容回退（与 _recall.py 同一套）
    from ._diag import log_degraded
except Exception:  # noqa: BLE001
    def log_degraded(component: str, reason: str, *, detail: str = "",
                     exc: BaseException | None = None) -> None:
        """Fallback: 无 _diag.py 时的 WARNING 级降级日志。"""
        logger.warning("[degraded] %s: %s %s", component, reason, detail,
                       exc_info=exc)

#: [[Title]] / [[Title|alias]] / [[dir/Title]] / [[Title#heading]]
_LINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:\|[^\]]*)?\]\]")

#: 索引表名（LanceDB）
_TABLE_NAME = "kb_notes"

#: 关键词分归一化分母（见 ``_keyword_score_norm``）。
_KEYWORD_SCORE_DENOM = 5.0

#: 关键词分 / 语义分在融合时的权重。
#:
#: 为什么需要融合而不是取 max（P0，2026-09-16）：旧实现把关键词原始分
#: （0~12）和语义分（0~1）直接丢进同一个 dict、用 ``max()`` 合并再统一
#: 排序，结果是**关键词永远碾压语义** —— 实测 ``hermes 记忆系统`` 返回
#: 1.8/1.6/1.4 全是关键词分，向量检索这一路等于白建。而 vault 里全是
#: 中文长笔记，查询词往往不在标题而在正文，恰恰是语义检索该发力的场景。
#: 归一化后按权重线性融合，语义为主（vault 以长文为主，同义改写多）、
#: 关键词为辅（精确术语命中依然是强证据）。
_KB_KW_WEIGHT = 0.4
_KB_SEM_WEIGHT = 0.6

#: RRF 融合的衰减常数 k（``score += w/(k+rank)``，rank 从 1 起）。
#:
#: k=60 是 Reciprocal Rank Fusion 文献与 WeKnora 的通用取值：第 1 名与第 2
#: 名的差距是 ``w/61`` vs ``w/62`` —— 足以让排名靠前者胜出，又不至于让
#: 第 1 名垄断（第一名对总分的贡献被压到与相邻名次同一量级）。
_RRF_K = 60

#: 亲和度封顶系数：实际乘数 = ``1 + 0.15·log1p(hits)/log1p(8)`` ∈ [1, 1.15]。
#:
#: 上限刻意压在 ×1.15：这是「缓慢复利的微推」—— 只扰动相近名次的取舍，
#: 不可能让一篇陈年旧文因为历史热度压过当前强相关的新命中。
#: ⚠️ 它是**振幅系数**（hits 恰为 ``_AFFINITY_SATURATION`` 时取到 ``1 + 它``），
#: 不是上限本身 —— 少了 ``_AFFINITY_MAX`` 的夹取，乘数会随 hits **无界增长**
#: （hits=1000 → ×1.47），与上面这句承诺相反。
_AFFINITY_CAP = 0.15

#: 亲和度乘数的**硬上限**：``1 + _AFFINITY_CAP``，在 hits == ``_AFFINITY_SATURATION``
#: 处取到，之后再涨就夹住。没有它，「封顶 ×1.15」只是注释里的一句话。
_AFFINITY_MAX = 1.0 + _AFFINITY_CAP

#: 亲和度的饱和参考命中数：``log1p(8)`` 归一化，命中 8 次即拿满封顶。
_AFFINITY_SATURATION = 8

#: MMR 候选池：``top_k * factor``，钳在 [min, max] 内（个人 vault 规模足够）。
_MMR_POOL_FACTOR = 5
_MMR_POOL_MIN = 20
_MMR_POOL_MAX = 60


def _rrf_fuse(kw_scores: Dict[str, float], sem_scores: Dict[str, float],
              kw_weight: float, sem_weight: float) -> Dict[str, Dict[str, Any]]:
    """加权 RRF：按**排名**（而非原始分）融合关键词/语义两路命中。

    归一化分母取两路第一名的理论分 ``(w_kw + w_sem)/(k+1) = 1/(k+1)``
    （启用通道的权重和恒为 1），于是：

    * 单路上限 = 该路权重（关键词 0.4 / 语义 0.6）——与线性融合的天花板
      完全一致，``recall_min_score`` / ``recall_min_kw_score`` 两条准入走廊
      在两种模式下语义不变（纯关键词依然够不到 0.45，必须走关键词走廊）；
    * 双路都排第 1 → 恰好 1.0。

    Returns:
        ``{rel_path: {"rrf": float, "kw_rank": int|None, "sem_rank": int|None}}``
    """

    def _ranks(scores: Dict[str, float]) -> Dict[str, int]:
        # 分数降序；同分按路径字典序稳定排序，保证 trace 可复现。
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return {rel: i + 1 for i, (rel, _s) in enumerate(ordered)}

    kw_ranks = _ranks(kw_scores) if kw_weight > 0.0 else {}
    sem_ranks = _ranks(sem_scores) if sem_weight > 0.0 else {}

    out: Dict[str, Dict[str, Any]] = {}
    for rel in set(kw_ranks) | set(sem_ranks):
        raw = 0.0
        if rel in kw_ranks:
            raw += kw_weight / (_RRF_K + kw_ranks[rel])
        if rel in sem_ranks:
            raw += sem_weight / (_RRF_K + sem_ranks[rel])
        out[rel] = {
            "rrf": min(1.0, raw * (_RRF_K + 1)),
            "kw_rank": kw_ranks.get(rel),
            "sem_rank": sem_ranks.get(rel),
        }
    return out

#: 删除旧笔记的重试参数（Windows 文件占用，见 :func:`_unlink_with_retry`）。
#:
#: WinError 32（``PermissionError`` / ``errno 13``）在 Windows 上是**瞬时**
#: 占用于是多半靠一次退避就能解开：杀软扫盘、Windows Search 索引器、另一个
#: 线程的读句柄都会短时间占住文件。旧实现 unlink 失败只 ``logger.warning``
#: 然后照常返回 ``ok=True`` —— 于是笔记同时留在 inbox 和 notes 里（重复数据），
#: 而 ``approve`` 告诉调用方「批准成功」。
_UNLINK_RETRIES = 4

#: 指数退避基数（秒）：实际等待 0.02 / 0.04 / 0.08，累计约 0.14s。
#: 相对一次真实向量检索（≈273ms）可以忽略，却刚好覆盖常见的瞬时占用窗口。
_UNLINK_BACKOFF_BASE = 0.02

#: mtime 比对容差（秒）：差值不超过它就认为「索引与正本一致」。
#:
#: 不能取 0：文件系统/同步盘的时间戳精度不一致（FAT/exFAT 1s、某些网盘 2s），
#: 严格相等会造成同一批笔记反复被判定为「陈旧」，每次检索都白白重嵌一遍。
_INDEX_MTIME_TOLERANCE = 0.5


def _as_list(value: Any) -> List[str]:
    """归一化 frontmatter 的 tags/concepts 到 List[str]。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _sql_literal(s: str) -> str:
    """SQL 字符串字面量（单引号转义为双单引号）。"""
    return "'" + s.replace("'", "''") + "'"


def _file_mtime(path: Path) -> float:
    """文件的 mtime（epoch 秒）；取不到时返回 0.0。

    写路径统一用它给索引打时间戳：只要写索引时 mtime 与正本一致，
    下一次 ``sync_index`` 就会判定「未变化」而跳过重新嵌入。
    """
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def _unlink_with_retry(
    path: Path,
    *,
    attempts: int = _UNLINK_RETRIES,
    base_delay: float = _UNLINK_BACKOFF_BASE,
) -> Optional[str]:
    """删除 ``path``；瞬时占用（Windows WinError 32）时退避重试。

    Windows 上另一个进程/线程只要持有该文件的句柄，``unlink`` 就会抛
    ``PermissionError``（errno 13 / WinError 32）。这种占用绝大多数是
    **瞬时**的（杀软扫盘、Windows Search 索引器、并发读），退避重试即可解开。
    直接放弃则会把「移动」变成「复制」，把这一层留给调用方去假装成功就是本次
    要消灭的缺陷。

    ``FileNotFoundError`` 视为**成功**：目标状态（文件不存在）已经达成 ——
    「早就没了」不是错误，把它报成失败反而会让调用方拒绝一次实际成功的移动。

    Args:
        path: 要删除的文件。
        attempts: 最大尝试次数，``>= 1``。
        base_delay: 退避基数，第 i 次失败后等待 ``base_delay * 2**i`` 秒。

    Returns:
        ``None`` 表示删除成功；否则返回含 errno 的失败原因字符串，供调用方
        **如实上报** —— 调用方必须据此判定本次移动失败。
    """
    last_reason = "unknown error"
    for i in range(max(1, attempts)):
        try:
            path.unlink()
            return None
        except FileNotFoundError:
            return None          # 已经不存在 —— 目的已经达到
        except OSError as e:     # noqa: BLE001 - 需要 errno，不吞
            last_reason = "errno=%s: %s" % (getattr(e, "errno", None), e)
            if i < max(1, attempts) - 1:
                time.sleep(base_delay * (2 ** i))
    return last_reason


def _bigrams(s: str) -> set:
    """字符串的 2-gram 集合（中文连续查询的关键词兜底）。"""
    s = s.lower()
    return {s[i:i + 2] for i in range(len(s) - 1)}


#: 双库分工的 frontmatter ``status`` 契约（Phase 2）：
#:
#: * 人工区（notes/projects/areas/resources/运维/根级笔记）→ ``curated``
#: * 自动区（inbox/knowledge）→ ``draft``（inbox 的待审标记仍是既有
#:   ``review_required`` 布尔，两个字段各管一件事）
#: * archive → ``archived``
#:
#: ``memory_cli._status_for_section`` 是同一规则的 CLI 侧拷贝（CLI 不能依赖
#: 插件加载）； ``tests/test_kb_governance.py`` 钉住两侧逐 section 一致 ——
#: 这是本仓库「一个存储只能有一个答案」约定在状态字段上的落法。
def _status_for_section(section: str) -> str:
    if section in ("inbox", "knowledge"):
        return "draft"
    if section == "archive":
        return "archived"
    return "curated"


#: 包含性查重的最小正文长度（归一化后）。低于它的正文（"好的""收到"）互相
#: 包含是常态而不是重复 —— 门槛太低会把琐碎确认当重复拒掉。
_DEDUP_MIN_CHARS = 120


def _normalize_for_dedup(text: str) -> str:
    """查重用的正文归一化：小写 + 折叠全部空白。"""
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _enrich_note(title: str, body: str, config: Any) -> Dict[str, Any]:
    """入库富化（``kb.enrich`` 开启时）：一行摘要 + 2~3 个自问。

    产出存进 frontmatter（``summary`` / ``questions``）喂给关键词召回路 ——
    纯向量对精确术语的漏召回由这些显式词补上（WeKnora 的 summary /
    generated questions）。**尽力而为**：LLM 不可用 / 超时 / 坏 JSON 一律
    返回 ``{}`` —— 富化是加分项，绝不阻断写入。
    """
    resp = chat_completion(
        [
            {"role": "system", "content": (
                "You enrich one knowledge note. Output ONLY JSON (no markdown, "
                "no prose): {\"summary\": string (<=120 chars, one line), "
                "\"questions\": [string] (2-3 questions this note answers)}.")},
            {"role": "user", "content": f"# {title}\n{body[:4000]}"},
        ],
        config, temperature=0.0, max_tokens=400, timeout=30.0,
    )
    if not resp:
        return {}
    try:
        text = str(resp).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?|\```$", "", text).strip()
        data = json.loads(text)
        out: Dict[str, Any] = {}
        summary = str(data.get("summary") or "").strip()
        if summary:
            out["summary"] = summary[:200]
        questions = [str(q).strip()
                     for q in (data.get("questions") or []) if str(q).strip()]
        if questions:
            out["questions"] = questions[:3]
        return out
    except Exception as e:  # noqa: BLE001 - 富化失败不阻断写入
        logger.debug("kb enrich parse failed: %s", e)
        return {}


class _Note:
    """一篇笔记的内存表示。"""

    __slots__ = ("path", "title", "meta", "body")

    def __init__(self, path: Path, title: str, meta: Dict[str, Any], body: str):
        self.path = path
        self.title = title
        self.meta = meta
        self.body = body

    @property
    def section(self) -> str:
        """相对 vault 的第一段目录名（inbox/notes/projects/...）。"""
        rel = self.path.parent
        return rel.name if rel.name else ""

    @property
    def tags(self) -> List[str]:
        return _as_list(self.meta.get("tags"))

    @property
    def concepts(self) -> List[str]:
        return _as_list(self.meta.get("concepts"))

    @property
    def short_text(self) -> str:
        """用于 bigram 命中的短文本（标题 + 标签 + 概念）。"""
        return " ".join([self.title] + self.tags + self.concepts)

    @property
    def full_text(self) -> str:
        """用于整串命中与向量索引的全文。"""
        return "\n".join([self.title, self.short_text, self.body])


class KBIndex:
    """向量索引投影层（可选）。不可用时 ``available=False`` 并记录原因。

    可用性**不是一次判定定终生**（P2，2026-09-17）：``__init__`` 那一刻的
    瞬时失败（网络抖动 / API 限流 / key 临时不可用）会把整条语义通道锁死到
    进程结束、且悄无声息 —— 这与「索引陈旧不自愈」同族，都是
    「一次性判定 + 无自愈 + 不报错」。因此当判定为「不可用」时，
    :attr:`available` 会按 :data:`REPROBE_TTL_SECONDS` **懒重探**：后端一旦
    恢复就自愈，并把次数 / 结论写进状态字段（见
    :meth:`KnowledgeBase._probe_state`）。

    语义边界（别混）：本类的 ``available=False`` 严格表示「后端真的用不了」，
    与「可用性未知」（``_probe_state`` 的 ``probe="unsupported"`` /
    ``fresh=None``）是两个概念。重探失败只会维持 ``available=False``，
    **绝不触碰** ``fresh`` 语义。
    """

    #: 「不可用」判定的懒重探间隔（秒）。
    #:
    #: 不能设 0：每次调用都重探会让每个检索多一次 lancedb connect，把廉价
    #: 探测变成新的性能问题。也不能太大：恢复太慢。30s 是折中，且只有
    #: **当前不可用**时才会走到这里 —— 可用状态下 :attr:`available` 零成本。
    REPROBE_TTL_SECONDS: float = 30.0

    def __init__(self, config: GovernedMemoryConfig, db_path: Path):
        self._config = config
        self._db_path = db_path
        self._service = None
        self._store = None
        self._available = False
        self._last_error = ""
        # 懒重探状态：锁保证并发检索时「TTL 内只探一次」
        self._reprobe_lock = threading.Lock()
        self._reprobe_count = 0
        self._next_reprobe_at = 0.0
        self._init()
        self._schedule_next_reprobe()

    def _schedule_next_reprobe(self) -> None:
        """把下一次懒重探排到 TTL 之后（可用状态下这个值用不到）。"""
        self._next_reprobe_at = time.monotonic() + self.REPROBE_TTL_SECONDS

    def _maybe_reprobe(self) -> None:
        """当前不可用时按 TTL 懒重探一次；恢复则自愈并留下可观测痕迹。

        只在 ``_available is False`` 时才被 :attr:`available` 调用，所以
        可用状态下这条路径的成本是 0（一次属性判断）。
        """
        if self._available:
            return
        if time.monotonic() < self._next_reprobe_at:
            return
        with self._reprobe_lock:
            # 双重检查：并发检索下可能已被另一个线程探过
            if self._available or time.monotonic() < self._next_reprobe_at:
                return
            self._reprobe_count += 1
            self._schedule_next_reprobe()
            self._init(reprobe=True)
            if self._available:
                log_degraded(
                    "kb_index", "recovered",
                    detail="vector backend available again after "
                           f"{self._reprobe_count} lazy re-probe attempt(s)",
                )
            else:
                # 仍是「不可用」——如实上报；**不改** fresh 语义（那是
                # _probe_state 的 probe="unsupported" 管的另一件事）。
                log_degraded(
                    "kb_index", "reprobe_still_unavailable",
                    detail=f"attempt #{self._reprobe_count}: "
                           f"{self._last_error or 'unknown'}",
                )

    # -- 初始化 ------------------------------------------------------

    def _init(self, *, reprobe: bool = False) -> None:
        """初始化（``reprobe=False``）或懒重探（``reprobe=True``）向量后端。

        Args:
            reprobe: 懒重探路径为 True。此时若拿到的是「启动那一刻就探测失败、
                且被 :class:`EmbeddingService` 单例缓存住」的服务，必须先
                ``reset()`` 丢弃它再重建 —— 否则 ``get()`` 会一直把同一个
                不可用实例还给我们，重探等于空转，瞬时故障永远恢复不了。
        """
        try:
            self._service = EmbeddingService.get(self._config)
            if reprobe and not getattr(self._service, "available", False):
                # reset() 只清空单例引用，已在用旧实例的调用方不受影响；
                # 重建会真正重跑一次后端探针（API 限流解除后即可恢复）。
                EmbeddingService.reset()
                self._service = EmbeddingService.get(self._config)
        except Exception as e:  # noqa: BLE001 — 索引只是投影，失败不能影响主流程
            self._last_error = f"embedding: {e}"
            return
        if self._service is None or not getattr(self._service, "available", False):
            self._last_error = "embedding unavailable"
            return
        try:
            import lancedb  # noqa: F401
            import pyarrow as pa  # noqa: F401
        except ImportError as e:
            self._last_error = f"lancedb missing: {e.name}"
            return
        try:
            self._db_path.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(str(self._db_path))
            names = db.table_names()
            if _TABLE_NAME in names:
                self._store = db.open_table(_TABLE_NAME)
                # 旧 schema（无 mtime）无法做增量同步，且索引只是**投影**、
                # 正本是 vault，可直接重建。此处就地迁移，重建交给
                # KnowledgeBase.sync_index()（会发现索引为空 → 全量补齐）。
                if "mtime" not in [f.name for f in self._store.schema]:
                    logger.info("kb index schema lacks 'mtime' — rebuilding")
                    db.drop_table(_TABLE_NAME)
                    self._store = None
            if self._store is None:
                dim = int(getattr(self._service, "dim", 0) or self._config.vector.dim)
                schema = pa.schema([
                    pa.field("path", pa.string()),
                    pa.field("text", pa.string()),
                    # 文件 mtime（epoch 秒）。增量同步靠它判断「是否需要重新
                    # 嵌入」；没有它就只能全量重建（26 篇 ≈ 26 次 API 调用）。
                    pa.field("mtime", pa.float64()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ])
                self._store = db.create_table(_TABLE_NAME, schema=schema)
            self._available = True
        except Exception as e:  # noqa: BLE001
            self._last_error = f"lancedb: {e}"
            self._store = None
            self._available = False

    @property
    def available(self) -> bool:
        """向量后端是否可用；当前不可用时按 TTL 懒重探一次（见类文档）。

        ``available=False`` 的含义严格是「后端真的用不了」，与「可用性未知」
        （``KnowledgeBase._probe_state`` 的 ``probe="unsupported"`` /
        ``fresh=None``）是两个概念 —— 重探逻辑不会把它们混回去。
        """
        if not self._available:
            self._maybe_reprobe()
        return self._available

    @property
    def reprobe_count(self) -> int:
        """进程启动以来懒重探的次数（诊断用，进 ``index_status``）。"""
        return self._reprobe_count

    @property
    def last_error(self) -> str:
        return self._last_error

    # -- 写 ----------------------------------------------------------

    def upsert(self, path: str, text: str, mtime: float = 0.0) -> None:
        """写入/更新一条向量。失败仅 debug 记录，不抛。

        Args:
            mtime: 正本文件的 mtime（epoch 秒）。增量同步据此判断陈旧。
        """
        if not self._available or self._service is None:
            return
        vec = self._service.embed_one(text)
        if vec is None:
            return
        try:
            self.delete(path)
        except Exception:  # noqa: BLE001 — 删不掉就靠搜索侧去重兜底
            pass
        try:
            self._store.add([{"path": path, "text": text,
                              "mtime": float(mtime or 0.0), "vector": vec}])
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index upsert failed for %s: %s", path, e)

    def path_mtimes(self) -> Dict[str, float]:
        """索引里 ``path -> mtime`` 的映射（供增量同步比对）。

        失败时返回空 dict —— 调用方会把空映射理解为「索引里什么都没有」，
        从而触发一次全量补齐；宁可多嵌一次，也不能漏掉正本里的新笔记。
        """
        if not self._available or self._store is None:
            return {}
        try:
            n = self._store.count_rows()
            if n <= 0:
                return {}
            arrow = (
                self._store.search()
                .select(["path", "mtime"])
                .limit(n)
                .to_arrow()
            )
            paths = arrow.column("path").to_pylist()
            mtimes = arrow.column("mtime").to_pylist()
            return {p: float(m or 0.0) for p, m in zip(paths, mtimes)}
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index path_mtimes failed: %s", e)
            return {}

    def delete(self, path: str) -> None:
        if not self._available or self._store is None:
            return
        self._store.delete(where=f"path = {_sql_literal(path)}")

    def drop(self) -> None:
        """清空并重建索引表（reindex 用）。"""
        if not self._available:
            return
        import lancedb  # noqa: F401
        try:
            db = lancedb.connect(str(self._db_path))
            if _TABLE_NAME in db.table_names():
                db.drop_table(_TABLE_NAME)
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index drop failed: %s", e)
        self._available = False
        self._init()
        self._schedule_next_reprobe()

    # -- 读 ----------------------------------------------------------

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """语义检索；返回 [{path, text, score, kind='semantic'}]。失败返回 []。"""
        if not self._available or self._service is None:
            return []
        vec = self._service.embed_one(query)
        if vec is None:
            return []
        try:
            rows = (
                self._store.search(vec)
                .metric("cosine")
                .limit(max(1, k))
                .to_list()
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index search failed: %s", e)
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "path": r.get("path", ""),
                "text": r.get("text", ""),
                # 复用 _recall 的换算，避免两份实现漂移（cosine distance ∈ [0,2]）
                "score": row_to_score(r),
                "kind": "semantic",
            })
        return out


# -- vault 遍历（单一实现） ------------------------------------------------
#
# 这一段存在的全部理由：本模块曾经有**两份**各自手写的 ``rglob("*.md")``
# （``scan_vault_paths`` 喂索引同步、``_iter_notes`` 喂检索打分），而 CLI 侧
# ``memory_cli.iter_notes`` 是第三份。三份规则各不相同 —— 于是同一个仓库，
# 插件和 CLI 对「库里有哪些笔记」给出不同答案；共享存储只能有一个答案。
# 现在插件侧的两条路都走 :func:`iter_vault_notes`，规则与 CLI 侧对齐
# （跳过所有点目录 + realpath 包含性检查）。
def _is_link(path: Path) -> bool:
    """True for a symlink **or** a Windows junction.

    Junctions are the easy one to miss: ``os.path.islink`` reports ``False``
    for them because they carry a different reparse tag, yet they redirect
    just as effectively — and on this machine the production vault *is* a
    junction, so a symlink-only check steps straight over the construct it
    exists to catch. ``os.path.isjunction`` is Python 3.12+, hence the
    ``getattr`` fallback.
    """
    try:
        if path.is_symlink():
            return True
    except OSError:
        return False
    isjunction = getattr(os.path, "isjunction", None)
    if callable(isjunction):
        try:
            return bool(isjunction(str(path)))
        except OSError:
            return False
    return False


def _is_within(child: Path, parent: Path) -> bool:
    """True when ``child`` is ``parent`` or sits below it."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def iter_vault_notes(vault: Path, *,
                     errors: "Optional[List[str]]" = None) -> Iterator[Path]:
    """Yield every note under ``vault``, refusing anything that leaves it.

    This is the **only** vault walk in this module; both the index-sync side
    (:meth:`KnowledgeBase.scan_vault_paths`) and the retrieval side
    (:meth:`KnowledgeBase._iter_notes`) consume it, so they cannot drift
    apart. Three rules, applied in this order:

    1. **realpath containment** — ``rglob`` *follows* links, so a junction or
       symlink under the vault walks out of it and a chain walks out further.
       Every hit is re-judged on its ``realpath``; escapes are **refused and
       named**, never silently dropped.
    2. **all dot-directories are skipped** (not just ``.obsidian``) —
       ``.git`` / ``.trash`` / ``.obsidian`` hold tooling, not notes, and the
       CLI side already skips them.
    3. ``index.md`` is navigation scaffolding (MOC), not a note.

    Args:
        vault: Vault root. May itself be a junction — both sides of every
            comparison are resolved through the same rule, so that stays in.
        errors: Optional sink. Every refusal is appended here *and* logged,
            because "the vault is empty" and "the walk was refused" look
            identical to a caller otherwise — and silently returning empty is
            the failure family this project keeps hunting down.

    Yields:
        Note paths, sorted for deterministic ordering.
    """
    if not vault.exists():
        return
    vault_real = Path(os.path.realpath(str(vault)))
    #: Paths already named in a refusal, so the link pass below does not repeat
    #: what the loop already reported (on Windows a junction is walked into, so
    #: its escaping files are named individually there).
    seen_refusals: List[str] = []
    for p in sorted(vault.rglob("*.md")):
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        # rglob 已经跟着链接走出去了，所以这里必须用 realpath 复核「还在不在库里」。
        real = Path(os.path.realpath(str(p)))
        if not _is_within(real, vault_real):
            msg = ("refused: '%s' resolves to '%s', outside the vault '%s'"
                   % (p, real, vault_real))
            logger.warning("[kb] vault walk %s", msg)
            seen_refusals.append(str(p))
            if errors is not None:
                errors.append(msg)
            continue
        try:
            rel = p.relative_to(vault)
        except ValueError:
            # 只可能经由「留在库内」的链接到达；此时真实身份才是可用的那个
            rel = real.relative_to(vault_real)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        if p.name == "index.md":  # MOC 导航不参与检索
            continue
        yield p

    # rglob descends in a platform-dependent way, and that difference is exactly
    # the one that hid a silent failure: a Windows **junction** is transparent to
    # ``scandir`` so the loop above walks into it and records a refusal per file,
    # while ``pathlib`` refuses to descend a directory **symlink** (cycle guard).
    # On POSIX an out-of-vault link therefore produced no hits and, with them, no
    # refusal — the escape was excluded **silently**, so "that link is ignored"
    # looked identical to "there is nothing there", which is the pair this
    # function's ``errors`` sink exists to keep apart.
    #
    # So links are checked explicitly. Pruning them from ``dirnames`` also
    # guarantees the walk never leaves the vault: ``os.path.islink`` reports
    # False for a junction, so ``os.walk`` alone would happily descend one.
    for dirpath, dirnames, _files in os.walk(vault, followlinks=False):
        for name in list(dirnames):
            candidate = Path(dirpath) / name
            if not _is_link(candidate):
                continue
            dirnames.remove(name)
            try:
                real = Path(os.path.realpath(str(candidate)))
            except OSError:  # pragma: no cover — unresolvable link
                continue
            if _is_within(real, vault_real):
                continue
            if any(str(candidate) in seen for seen in seen_refusals):
                continue  # already named by the loop above (Windows junction)
            msg = ("refused: '%s' is a link to '%s', outside the vault '%s'"
                   % (candidate, real, vault_real))
            logger.warning("[kb] vault walk %s", msg)
            seen_refusals.append(str(candidate))
            if errors is not None:
                errors.append(msg)


class _UsageStore:
    """命中使用度（亲和度数据源）：``rel_path → (hits, last_used)``。

    SQLite 单表、每次操作独立连接 —— 召回跑在线程池里，短连接比持有跨线程
    句柄简单且足够快（单行 upsert 亚毫秒）。**所有失败都降级为「本次没有
    统计」**：使用度是排序的加分项，绝不允许它把读路径搞挂。
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), timeout=1.0)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS kb_usage ("
            "path TEXT PRIMARY KEY,"
            "hits INTEGER NOT NULL DEFAULT 0,"
            "last_used REAL NOT NULL DEFAULT 0)"
        )
        return conn

    def hits(self) -> Dict[str, int]:
        """全量命中计数；失败返回空表（= 无亲和度加成）。"""
        conn = None
        try:
            conn = self._connect()
            rows = conn.execute("SELECT path, hits FROM kb_usage").fetchall()
            return {str(p): int(h) for p, h in rows}
        except Exception as e:  # noqa: BLE001 - 统计失败绝不影响检索
            logger.debug("kb_usage read failed: %s", e)
            return {}
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    def entries(self) -> Dict[str, Tuple[int, float]]:
        """全量 ``{path: (hits, last_used)}``；失败返回空表。

        供治理巡检算「闲置天数」—— ``hits`` 只能回答「被用过没有」，
        回答不了「多久没被用」。
        """
        conn = None
        try:
            conn = self._connect()
            rows = conn.execute(
                "SELECT path, hits, last_used FROM kb_usage").fetchall()
            return {str(p): (int(h), float(lu or 0.0))
                    for p, h, lu in rows}
        except Exception as e:  # noqa: BLE001
            logger.debug("kb_usage read failed: %s", e)
            return {}
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass

    def touch(self, rel_paths: List[str]) -> None:
        """命中计数 +1 并刷新 ``last_used``；失败静默（下次再记）。"""
        if not rel_paths:
            return
        conn = None
        try:
            conn = self._connect()
            now = time.time()
            for rel in rel_paths:
                conn.execute(
                    "INSERT INTO kb_usage(path, hits, last_used) VALUES(?, 1, ?)"
                    " ON CONFLICT(path) DO UPDATE SET"
                    " hits = hits + 1, last_used = excluded.last_used",
                    (rel, now),
                )
            conn.commit()
        except Exception as e:  # noqa: BLE001
            logger.debug("kb_usage touch failed: %s", e)
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass


class KnowledgeBase:
    """知识库门面：vault 检索 + 写入 + 索引投影。"""

    def __init__(self, config: GovernedMemoryConfig):
        self._config = config
        self._vault = _vault.vault_root(config)
        # 独立向量库目录（与 L2 记忆的 l2/ 分开）
        self._index_db = Path(config.l2_db_path).parent / "kb_index"
        self._index: Optional[KBIndex] = None
        # 使用度表（亲和度数据源）。与 kb_index 同级目录，懒加载 ——
        # affinity 关闭时这张表永远不会被创建（读路径零副作用）。
        self._usage_db = Path(config.l2_db_path).parent / "kb_usage.db"
        self._usage: Optional[_UsageStore] = None
        # 检索前的自愈同步互斥：并发检索（recall 线程池）不触发两遍增量同步
        self._sync_lock = threading.Lock()
        # 最近一次探测结论，供 :meth:`last_index_status` 读取
        self._index_status: Dict[str, Any] = {}
        # 最近一次 vault 遍历被拒绝的路径（越界链接等）。必须留痕：
        # 「库里没这篇」和「这篇被拒了」对调用方长得一模一样。
        self._vault_walk_refusals: List[str] = []

    # -- 索引新鲜度探测（廉价，只读元数据） ----------------------------

    def _walk(self) -> List[Path]:
        """单次 vault 遍历，供本类所有需要「库里有哪些笔记」的入口共用。

        走 :func:`iter_vault_notes` —— 全模块唯一的遍历实现，所以索引同步侧与
        检索侧不可能对「库里有什么」给出不同答案。被拒绝的路径（越界链接）
        存进 :attr:`last_walk_refusals`，不静默丢弃。
        """
        refusals: List[str] = []
        paths = list(iter_vault_notes(self._vault, errors=refusals))
        self._vault_walk_refusals = refusals
        return paths

    @property
    def last_walk_refusals(self) -> List[str]:
        """最近一次遍历被拒绝的路径说明（越界链接等）。

        空列表 = 没被拒。这是「库里没这篇」与「这篇被拒了」的唯一区分点 ——
        没有它，一次被拒的遍历看起来跟空库一模一样。
        """
        return list(self._vault_walk_refusals)

    def scan_vault_paths(self) -> Dict[str, Path]:
        """正本侧 ``相对路径 -> Path`` 映射（单次目录遍历，不读文件内容）。"""
        out: Dict[str, Path] = {}
        for path in self._walk():
            out[self._rel(path)] = path
        return out

    def scan_vault_mtimes(self) -> Dict[str, float]:
        """正本侧 ``相对路径 -> mtime`` 映射，**只读 stat、不读文件内容**。

        这是本修复的性能关键：旧代码为了知道「有没有新笔记」必须用
        ``_iter_notes()`` 把每篇笔记 read + 解析 frontmatter（vault 越大越贵），
        而判别陈旧只需要 mtime —— stat 比 read 便宜一到两个数量级（实测
        27 篇：stat 扫描 1.45ms vs 一次真实向量检索 273ms ≈ 1/190）。
        """
        out: Dict[str, float] = {}
        for rel, path in self.scan_vault_paths().items():
            out[rel] = _file_mtime(path)
        return out

    @staticmethod
    def _diff_index_state(
        vault_mtimes: Dict[str, float],
        indexed: Dict[str, float],
        available: bool,
        reason: str = "",
        probe: str = "ok",
        reprobe_count: int = 0,
    ) -> Dict[str, Any]:
        """正本 vs 投影的差异分类（纯计算，无 I/O）。

        返回字段：
            ``available`` / ``reason`` —— **向量后端**是否可用。只由索引
                自身决定，绝不能被「探针能不能跑」污染（见下）。
            ``probe`` —— ``"ok"`` / ``"unavailable"`` / ``"unsupported"``。
                区分「后端没了」和「这个索引实现没有 mtime 表，没法判断」。
            ``reprobe_count`` —— 后端**懒重探**次数（P2）。``available=True``
                且它 ``>0`` 表示「曾经不可用、后来自愈了」；这个恢复事件必须
                可见，否则「一次瞬时故障永久降级」会再犯。
            ``vault_notes`` / ``indexed`` —— 两侧记录数
            ``missing`` / ``missing_paths`` —— 正本有、索引没有（新增/外部写入）
            ``stale`` / ``stale_paths`` —— mtime 不一致（修改过）
            ``orphaned`` —— 索引有、正本没有（已删除）
            ``fresh`` —— 三者皆为 0；``probe != "ok"`` 时为 ``None``（**未知**，
                不是「不可用」）。把「未知」说成「不可用」会让检索路径误砍掉
                整条语义通道 —— 读路径宁可放行，不可砍光。
        """
        missing: List[str] = []
        stale: List[str] = []
        for rel, mt in vault_mtimes.items():
            cur = indexed.get(rel)
            if cur is None:
                missing.append(rel)
            elif abs(cur - mt) > _INDEX_MTIME_TOLERANCE:
                stale.append(rel)
        orphaned = [rel for rel in indexed if rel not in vault_mtimes]
        fresh: Optional[bool]
        fresh = None if probe != "ok" else not (missing or stale or orphaned)
        state = {
            "available": bool(available),
            "reason": reason,
            "probe": probe,
            "reprobe_count": int(reprobe_count),
            "vault_notes": len(vault_mtimes),
            "indexed": len(indexed),
            "missing": len(missing),
            "missing_paths": sorted(missing),
            "stale": len(stale),
            "stale_paths": sorted(stale),
            "orphaned": len(orphaned),
            "orphaned_paths": sorted(orphaned),
            "fresh": fresh,
        }
        return state

    def _probe_state(self, idx: KBIndex) -> Dict[str, Any]:
        """探测当前新鲜度；探针跑不动时如实标 ``probe``，不谎报 ``available``。

        ``path_mtimes`` 是本次修复引入的索引能力，替身 / 第三方 / 旧版索引可能
        没有。那种情况下我们**无法判断**新鲜度（``fresh=None``），但这跟「向量
        后端不可用」是两件完全不同的事：语义通道该照常工作。历史教训之一就是
        把这两件事混成一个布尔值，于是补一个诊断功能顺手把召回砍掉一半。

        读取 ``idx.available`` 本身可能触发一次懒重探（P2）—— 这正是
        「瞬时故障自愈」的入口：可用性恢复后，下一次探测就会走到正常分支。
        """
        available = idx.available
        reprobe_count = int(getattr(idx, "reprobe_count", 0) or 0)
        if not available:
            return self._diff_index_state(
                {}, {}, False, idx.last_error or "index unavailable",
                probe="unavailable", reprobe_count=reprobe_count,
            )
        probe_mtimes = getattr(idx, "path_mtimes", None)
        if not callable(probe_mtimes):
            # available 仍然是 True：后端在，只是探针不可用。
            return self._diff_index_state(
                {}, {}, True, "index has no path_mtimes()", probe="unsupported",
                reprobe_count=reprobe_count,
            )
        return self._diff_index_state(
            self.scan_vault_mtimes(), probe_mtimes(), True, probe="ok",
            reprobe_count=reprobe_count,
        )

    def index_status(self) -> Dict[str, Any]:
        """索引新鲜度诊断（纯元数据，不读文件内容）。

        与 :meth:`sync_index` 共用同一份「陈旧」定义（``_INDEX_MTIME_TOLERANCE``
        容差），所以这里报 ``missing=1`` 就一定意味着下一次同步会补上 1 篇。
        探针不可用时 ``probe="unsupported"`` / ``fresh=None``（未知 ≠ 不可用）。
        """
        return self._probe_state(self._index_get())

    @property
    def last_index_status(self) -> Dict[str, Any]:
        """上一次新鲜度探测的结论（首次检索前为空 dict）。

        供上层把「检索结果可能不全」这件事显式告知用户 —— 数值再准，
        用户看不到就等于没有。
        """
        return dict(self._index_status)

    def _ensure_fresh_index(self) -> Dict[str, Any]:
        """检索门前的廉价自愈：发现陈旧才增量同步，新鲜就立刻返回。

        为什么放在这里而不是只在进程启动时跑一次：``ensure(sync=True)`` 的后台
        同步只覆盖「插件刚起来」那一刻，会话中新写入的笔记（`kb-add` 之外的
        外部直写 vault、同步盘、Obsidian 手工编辑）要到下一次重启才进索引，
        期间 ``kb-search`` 搜不到却什么都不说 —— 静默失败比报错更糟。

        成本控制（实测 27 篇真实 vault）：
        - 一次探针 = stat 扫描 1.45ms + 索引侧 ``path_mtimes`` 7.98ms ≈ 9.4ms
        - 一次真实向量检索（含 embedding API）≈ 273.6ms
        探针比一次检索便宜约 29 倍，因此**每次检索都探测**的开销完全可以
        接受，换来的是「写完立刻能搜到」的确定性；真正贵的内嵌落库只在
        ``fresh=False`` 时发生，且只针对差异部分（:meth:`sync_index` 增量）。

        探针跑不动（索引没有 ``path_mtimes``）时不报「后端不可用」，只把
        ``fresh`` 标成未知然后照常返回 —— 自愈是增强，不能反过来把检索砍掉。

        可用性本身也会自愈（P2）：``idx.available`` 在判定为不可用时按 TTL
        懒重探，因此一次瞬时故障过后，这里会先恢复 ``available``、再照常做
        增量同步 —— 无需重启进程。恢复次数随 ``reprobe_count`` 一起透出。

        返回值的语义（2026-09-17 修正）：增量同步**成功之后重新探测一次**
        （实测 ≈9ms，相对一次检索 273ms 可以接受），所以 ``fresh`` /
        ``missing`` / ``stale`` / ``orphaned`` 一律是**同步后的当前真值**；
        同步前的差异计数不丢，显式留在 ``pre_sync_*`` 里。旧实现直接把同步前
        的快照当结果返回，于是同一个字典里同时出现 ``synced=True`` 和
        ``fresh=False, missing=1`` —— 自愈其实成功了（missing 1→0）却报
        「结果可能不全」。而 ``search()`` 正是在自愈**之后**才构建结果的，
        那个「不全」是误报。同一个字段名两种语义，正是本项目一直在清的坑。
        """
        idx = self._index_get()
        available = idx.available  # 可能触发一次懒重探；必须先读它再读计数
        reprobe_count = int(getattr(idx, "reprobe_count", 0) or 0)
        probe_mtimes = getattr(idx, "path_mtimes", None) if available else None
        if not available or not callable(probe_mtimes):
            # 后端真的不可用，或探针不可用：都没有可同步的依据，直接如实返回。
            self._index_status = self._probe_state(idx)
            return self._index_status

        with self._sync_lock:
            paths = self.scan_vault_paths()
            vault = {rel: _file_mtime(p) for rel, p in paths.items()}
            indexed = probe_mtimes()
            state = self._diff_index_state(vault, indexed, True,
                                           reprobe_count=reprobe_count)
            self._index_status = state
            if state["fresh"] is False:
                # 增量同步，并把已经算好的两侧映射传进去，避免重复 stat / 扫描
                self.sync_index(_vault_mtimes=vault, _note_paths=paths,
                                _index_mtimes=indexed)
                # 同步后重探：fresh / missing / stale / orphaned 必须是**当前真值**。
                # 直接返回同步前的快照会自相矛盾（synced=True 却 fresh=False、
                # missing=1），而 search() 正是在同步之后才构建结果的 —— 那句
                # 「结果可能不全」是误报。同步前的差异计数显式留在 pre_sync_*，
                # 它是「这次自愈补了什么」的唯一记录。
                #
                # 重探只在**本次真的同步过**之后发生：新鲜时多一次 9ms 的 stat
                # 扫描纯属浪费，性能硬约束（新鲜时 5 次检索不得重嵌）不容破坏。
                # 若重探后仍 fresh=False（例如文件持续在变），如实报 —— 那是有
                # 价值的真信号，不能强行折成 True。
                after = self._probe_state(idx)
                self._index_status = dict(
                    after,
                    synced=True,
                    pre_sync_missing=state["missing"],
                    pre_sync_stale=state["stale"],
                    pre_sync_orphaned=state["orphaned"],
                )
            return self._index_status

    # -- 骨架 / 索引 -------------------------------------------------

    def ensure(self, *, sync: bool = True) -> Dict[str, Any]:
        """建 vault 骨架（幂等），并按需在后台补齐向量索引。

        Args:
            sync: 是否触发一次后台增量同步。默认开启 —— 外部写入
                （DeepSeek Harness、Obsidian 手工编辑、同步盘）不经过
                ``_kb.add``，只有这次 mtime 比对能发现它们，否则索引会
                随时间越来越陈旧（实测停滞 15 天，漏掉 1 篇笔记）。
        """
        result = _vault.ensure_skeleton(self._vault)
        if sync:
            try:
                self.sync_index(background=True)
            except Exception as e:  # noqa: BLE001 — 同步是增强，失败不影响主流程
                logger.debug("kb background sync failed to start: %s", e)
        return result

    def _index_get(self) -> KBIndex:
        if self._index is None:
            self._index = KBIndex(self._config, self._index_db)
        return self._index

    @property
    def index_available(self) -> bool:
        return self._index_get().available

    def reindex(self) -> Dict[str, Any]:
        """从 vault 重建向量索引（丢索引后的恢复手段）。"""
        idx = self._index_get()
        if not idx.available:
            return {"ok": False, "reason": idx.last_error or "index unavailable"}
        idx.drop()
        n = 0
        for note in self._iter_notes():
            idx.upsert(self._rel(note.path), note.full_text,
                       mtime=_file_mtime(note.path))
            n += 1
        return {"ok": True, "indexed": n}

    def sync_index(self, *, background: bool = False, force: bool = False,
                   _vault_mtimes: "Optional[Dict[str, float]]" = None,
                   _note_paths: "Optional[Dict[str, Path]]" = None,
                   _index_mtimes: "Optional[Dict[str, float]]" = None
                   ) -> Dict[str, Any]:
        """增量同步：正本（vault）变更 → 索引投影更新。

        与 :meth:`reindex` 的区别：``reindex`` 无条件全量重建（26 篇 = 26 次
        embedding 调用）；``sync_index`` 按 mtime 比对，**只重嵌变化过的笔记**，
        常态下是 0 次调用。

        为什么必须有这个方法（2026-09-16 实测）：索引最后一次更新停在 9-01，
        而 ``notes/共享记忆系统开通.md`` 是 9-04 由 DeepSeek Harness 直接写入
        vault 的 —— 插件的 ``_kb.add`` 会 upsert，但**外部写入**（DSH、
        Obsidian 手工编辑、同步盘）完全不会碰索引。没有兜底同步，索引就只会
        随时间越来越陈旧，`_search_kb` 也永远找不到新笔记。

        Args:
            background: True 时在守护线程里执行，立即返回（供插件初始化调用，
                避免首次全量补齐的 embedding 调用拖慢启动）。
            force: True 时忽略 mtime 比对，全量重嵌。

        Returns:
            统计 dict：``{ok, added, updated, removed, unchanged, errors}``。

        私有参数 ``_vault_mtimes`` / ``_note_paths`` / ``_index_mtimes`` 供
        :meth:`_ensure_fresh_index` 复用已经算好的「正本 mtime 表 / 路径表 /
        索引 mtime 表」，避免一次检索里重复遍历目录、重复扫 LanceDB 两次。

        命名必须带后缀：模块顶层已经 ``from . import _vault``，名字叫 ``_vault``
        的参数会把模块遮住，``_vault.parse_frontmatter`` 直接 AttributeError。
        """
        if background:
            t = threading.Thread(
                target=self.sync_index, kwargs={"force": force},
                name="kb-sync-index", daemon=True,
            )
            t.start()
            return {"ok": True, "scheduled": True}

        idx = self._index_get()
        if not idx.available:
            return {"ok": False, "reason": idx.last_error or "index unavailable"}

        paths = _note_paths if _note_paths is not None else self.scan_vault_paths()
        vault: Dict[str, float] = (
            _vault_mtimes if _vault_mtimes is not None
            else {rel: _file_mtime(p) for rel, p in paths.items()}
        )

        indexed = {} if force else (
            _index_mtimes if _index_mtimes is not None else idx.path_mtimes()
        )

        added: List[str] = []
        updated: List[str] = []
        for rel, mt in vault.items():
            cur = indexed.get(rel)
            if cur is None:
                added.append(rel)
            elif abs(cur - mt) > _INDEX_MTIME_TOLERANCE:  # 避开文件系统时间精度差异
                updated.append(rel)
        removed = [rel for rel in indexed if rel not in vault]

        errors = 0
        # 只读差异部分的内容：批量重建（reindex）才需要读全库，
        # 一次自愈通常只有 1~2 篇是新的，没必要把 27 篇全 parse 一遍。
        for rel in added + updated:
            path = paths.get(rel)
            if path is None:
                errors += 1
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                logger.debug("kb sync read failed for %s: %s", rel, e)
                errors += 1
                continue
            meta, body = _vault.parse_frontmatter(text)
            note = _Note(path, _vault.note_title(path), meta, body)
            try:
                idx.upsert(rel, note.full_text, mtime=mt)
            except Exception as e:  # noqa: BLE001
                logger.debug("kb sync upsert failed for %s: %s", rel, e)
                errors += 1
        for rel in removed:
            try:
                idx.delete(rel)
            except Exception as e:  # noqa: BLE001
                logger.debug("kb sync delete failed for %s: %s", rel, e)
                errors += 1

        result = {
            "ok": True,
            "added": len(added),
            "updated": len(updated),
            "removed": len(removed),
            "unchanged": len(vault) - len(added) - len(updated),
            "errors": errors,
        }
        if added or updated or removed:
            logger.info("kb index sync: %s", result)
        return result

    def index_stats(self) -> Dict[str, Any]:
        """索引新鲜度诊断：正本 vs 投影的差异（供 governed_health 用）。

        历史 key（``available`` / ``vault_notes`` / ``indexed`` / ``missing``）
        保持不变，新增 ``stale`` / ``orphaned`` / ``fresh`` 等细分字段。
        """
        return self.index_status()

    # -- 读取 --------------------------------------------------------

    def _iter_notes(self) -> List[_Note]:
        notes: List[_Note] = []
        for path in self._walk():  # 与索引同步侧共用同一份遍历规则
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta, body = _vault.parse_frontmatter(text)
            notes.append(_Note(path, _vault.note_title(path), meta, body))
        return notes

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._vault)).replace("\\", "/")
        except ValueError:
            return str(path)

    def get(self, title: str) -> Optional[Dict[str, Any]]:
        """按标题读单篇（返回全文）。

        读数本身来自 vault 正本（永远不会陈旧），门前仍走一次
        :meth:`_ensure_fresh_index`：写完立刻被读到 → 下一次 ``search`` 就有它，
        自愈不必等到下一次检索才发生。
        """
        status = self._ensure_fresh_index()
        for note in self._iter_notes():
            if note.title == title or note.path.stem == title:
                return {
                    "title": note.title,
                    "path": self._rel(note.path),
                    "section": note.section,
                    "meta": note.meta,
                    "body": note.body,
                    "index_status": dict(status),
                }
        return None

    def list_notes(self, section: str = "") -> List[Dict[str, Any]]:
        """列举笔记（可按 section 过滤）。"""
        out = []
        for note in self._iter_notes():
            if section and note.section != section:
                continue
            out.append({
                "title": note.title,
                "path": self._rel(note.path),
                "section": note.section,
                "tags": note.tags,
                "concepts": note.concepts,
            })
        return out

    def links(self, title: str) -> List[str]:
        """反链：谁链接了标题为 *title* 的笔记。"""
        back: List[str] = []
        for note in self._iter_notes():
            for m in _LINK_RE.finditer(note.body):
                target = m.group(1).split("/")[-1].strip()
                if target == title:
                    back.append(self._rel(note.path))
                    break
        return back

    # -- 检索 --------------------------------------------------------

    @staticmethod
    def _keyword_score(query: str, note: _Note) -> float:
        """关键词原始打分（中文 bigram 兜底 + 整串命中），量纲 0~12。

        仅供 :meth:`_keyword_score_norm` 归一化使用，以及需要在 UI 上展示
        原始命中强度的场景。**不要**直接把它和语义分（[0,1]）比较或混排
        —— 见 :data:`_KEYWORD_SCORE_MAX` 的说明。
        """
        q = query.strip().lower()
        if not q:
            return 0.0
        short = note.short_text.lower()
        body_l = note.body.lower()
        title = note.title.lower()
        score = 0.0
        if q in title:
            score += 5.0
        if q in short:
            score += 3.0
        if q in body_l:
            score += 2.0
        qg = _bigrams(q)
        if qg:
            sg = _bigrams(short)
            if sg:
                score += len(qg & sg) / len(qg) * 2.0
        return score

    @classmethod
    def _keyword_score_norm(cls, query: str, note: _Note) -> float:
        """关键词分归一化到 [0,1]，可与语义分直接比较/融合。

        归一化分母取 5.0 而非理论上限 12.0：``q in title`` 必然也让
        ``q in short`` 成立（short_text 含 title），所以 title 命中天然拿到
        5+3=8 分。以 5.0 为分母的含义是「标题命中即满格」，body 命中 0.4、
        标签/概念命中 0.6，语义上正好对应「强/弱相关」的直觉分级。
        """
        return min(cls._keyword_score(query, note) / _KEYWORD_SCORE_DENOM, 1.0)

    def search(self, query: str, top_k: int = 10, section: str = "") -> List[Dict[str, Any]]:
        """融合检索：关键词 + 语义（权重融合） + 反链统计。

        两路分数各自归一化到 [0,1] 后按 :data:`_KB_KW_WEIGHT` /
        :data:`_KB_SEM_WEIGHT` 线性融合。``kb.keyword_enabled`` /
        ``kb.semantic_enabled`` 在这里真正生效（此前两个开关只被定义、
        从未被读取）。

        返回项的 ``score`` 是融合分；``kind`` 标明主导来源（``keyword`` /
        ``semantic`` / ``hybrid``），便于诊断。

        门前先做一次 :meth:`_ensure_fresh_index`（廉价 mtime 探测 + 必要时的
        增量同步）：会话中新写入 / 外部写入 vault 的笔记不必等到进程重启才
        能被搜到。索引状态挂在返回项的 ``index_status`` 上，所以「结果可能
        不全」这件事对用户是可见的，而不是静默变少。

        其中 ``fresh`` / ``missing`` / ``stale`` / ``orphaned`` 是**同步后的
        当前真值**（自愈成功就报新鲜），同步前的差异计数在 ``pre_sync_*`` ——
        两者分开是 2026-09-17 的修正：旧实现只挂同步前那一份，于是自愈成功
        了却报「结果可能不全」，是误报。
        """
        if not query.strip():
            return []
        top_k = max(1, min(int(top_k), 50))
        status = self._ensure_fresh_index()
        notes = self._iter_notes()
        if section:
            notes = [n for n in notes if n.section == section]

        kb_cfg = self._config.kb
        keyword_on = bool(getattr(kb_cfg, "keyword_enabled", True))
        semantic_on = bool(getattr(kb_cfg, "semantic_enabled", True))

        # MMR 需要比 top_k 更深的候选池：语义通道按池深取数，否则 sem-only
        # 语料下 池 == top_k，MMR 永远没有可挤的对象（每条查询仍只嵌一次，
        # 代价只是索引多返回几行）。
        lam = float(getattr(kb_cfg, "mmr_lambda", 0.0) or 0.0)
        sem_fetch = top_k
        if 0.0 < lam < 1.0:
            sem_fetch = min(max(top_k * _MMR_POOL_FACTOR, _MMR_POOL_MIN),
                            _MMR_POOL_MAX)

        # 两路各自的归一化分（缺一路时置 0，由权重归一化兜底）
        kw_norm: Dict[str, float] = {}
        if keyword_on:
            for note in notes:
                rel = self._rel(note.path)
                s = self._keyword_score_norm(query, note)
                if s > 0.0:
                    kw_norm[rel] = s

        sem_norm: Dict[str, float] = {}
        idx = self._index_get()
        if semantic_on and idx.available:
            for r in idx.search(query, sem_fetch):
                sem_norm[r["path"]] = float(r.get("score", 0.0) or 0.0)

        # 权重按「实际启用的通道」重新归一化，避免关掉一路后总分整体缩水
        active_kw = _KB_KW_WEIGHT if keyword_on else 0.0
        active_sem = _KB_SEM_WEIGHT if semantic_on else 0.0
        total_w = active_kw + active_sem
        if total_w <= 0.0:
            return []
        active_kw /= total_w
        active_sem /= total_w

        # 融合算法：默认 linear（历史行为）；"rrf" = 按排名融合（见 _rrf_fuse）。
        fusion_mode = str(getattr(kb_cfg, "fusion", "linear") or "").strip().lower()
        if fusion_mode != "rrf":
            if fusion_mode not in ("linear", ""):
                logger.warning("Config kb.fusion = %r unknown — using 'linear'",
                               fusion_mode)
            fusion_mode = "linear"
        fused_meta = (
            _rrf_fuse(kw_norm, sem_norm, active_kw, active_sem)
            if fusion_mode == "rrf" else {}
        )

        # 亲和度：启用时读一次使用度表（失败 = 空表 = 无加成）。
        affinity_on = bool(getattr(kb_cfg, "affinity_enabled", False))
        usage_hits: Dict[str, int] = {}
        if affinity_on:
            usage_hits = self._usage_store().hits()

        note_by_rel: Dict[str, _Note] = {self._rel(n.path): n for n in notes}

        # 每个返回项都带上同一份索引新鲜度快照。字段只放标量计数，不放
        # ``missing_paths`` 这类长列表，免得把每条结果都撑成一屏。
        # ``fresh`` 可能是 None（探针不可用 → 未知），照原样透传，不要强行
        # 折成 True/False —— 把「不知道」说成「新鲜」就是本次要消灭的静默失败。
        snapshot = {
            "fresh": status.get("fresh"),
            "probe": status.get("probe", "unavailable"),
            "available": bool(status.get("available", False)),
            # 懒重探次数：>0 且 available 为真 = 「曾降级、已自愈」，必须可见
            "reprobe_count": status.get("reprobe_count", 0),
            "vault_notes": status.get("vault_notes", 0),
            "indexed": status.get("indexed", 0),
            "missing": status.get("missing", 0),
            "stale": status.get("stale", 0),
            "orphaned": status.get("orphaned", 0),
            "synced": bool(status.get("synced", False)),
            # 同步前的差异计数 = 「这次自愈补了什么」。与上面的 fresh/missing
            # （同步后的当前真值）语义不同，所以分开设字段 —— 一个字段两种
            # 含义正是本项目一直在清的坑。
            "pre_sync_missing": status.get("pre_sync_missing", 0),
            "pre_sync_stale": status.get("pre_sync_stale", 0),
            "pre_sync_orphaned": status.get("pre_sync_orphaned", 0),
        }

        scored: Dict[str, Dict[str, Any]] = {}
        for rel in set(kw_norm) | set(sem_norm):
            kw = kw_norm.get(rel, 0.0)
            sem = sem_norm.get(rel, 0.0)
            note = note_by_rel.get(rel)
            if note is not None:
                title, section_name = note.title, note.section
                tags, concepts = note.tags, note.concepts
                snippet = note.body[:160].strip()
            else:
                # 语义命中但 vault 里已无此文件（索引投影滞后于正本）
                title = Path(rel).stem
                section_name = Path(rel).parent.name
                tags, concepts, snippet = [], [], ""
            if kw > 0.0 and sem > 0.0:
                kind = "hybrid"
            elif sem > 0.0:
                kind = "semantic"
            else:
                kind = "keyword"
            # 基础分：linear = 加权线性融合；rrf = _rrf_fuse 的排名归一分
            # （天花板一致：纯关键词 ≤ 0.4 / 纯语义 ≤ 0.6 / 双路第 1 = 1.0）。
            if fusion_mode == "rrf":
                meta = fused_meta.get(rel, {})
                base = float(meta.get("rrf", 0.0))
                kw_rank, sem_rank = meta.get("kw_rank"), meta.get("sem_rank")
            else:
                base = active_kw * kw + active_sem * sem
                kw_rank = sem_rank = None
            # 亲和度乘数：hits=0 → 1.0（不加成）；hits ≥ 饱和点 → 夹在 _AFFINITY_MAX。
            affinity = 1.0
            if affinity_on:
                hits_count = usage_hits.get(rel, 0)
                if hits_count > 0:
                    affinity = min(
                        _AFFINITY_MAX,
                        1.0 + _AFFINITY_CAP * math.log1p(hits_count)
                        / math.log1p(_AFFINITY_SATURATION))
            # ⚠️ **准入分与排序分必须分开**（P0，2026-09-22 修）：
            # ``score`` 是「这条结果多相关」，KB 提示通道的准入门槛
            # （``kb.recall_min_score``）判的正是它；使用度只该影响**排序**。
            # 此前把乘数直接算进 ``score``，于是纯关键词命中（base=0.4）在
            # hits≥6 时 0.4×1.13 = 0.45 就压过部署门槛 0.45 —— 把
            # 「纯关键词够不到 0.45、必须走关键词走廊」这条不变量打开，
            # 而 ``_config.py`` 的 ``recall_min_kw_score`` 整段推理正靠它成立。
            # 同一模式在 L2 侧早已确立：**门槛量原生分、排序量融合分**
            # （``_recall.py`` 的 ``native_score``）。
            rank_key = min(1.0, base * affinity)
            # 注意缩进：这一句必须在 for 循环体内。曾经因为一次编辑把它
            # 顶格到循环外（只剩最后一轮迭代的值），结果 search() 对每条查询
            # 只返回一条结果 —— 而既有测试是全绿的假象要靠新测试才暴露。
            scored[rel] = {
                "title": title,
                "path": rel,
                "section": section_name,
                "score": round(base, 4),
                "rank_key": round(rank_key, 4),
                "kind": kind,
                "keyword_score": round(kw, 4),
                "semantic_score": round(sem, 4),
                "tags": tags,
                "concepts": concepts,
                "snippet": snippet,
                "index_status": dict(snapshot),
                # 排序诊断：这条为什么排这里 —— 融合模式 / 两路名次 /
                # 亲和度乘数。「没命中」与「没跑」必须可区分（trace 常在）。
                "trace": {
                    "fusion": fusion_mode,
                    "kw_rank": kw_rank,
                    "sem_rank": sem_rank,
                    "base": round(base, 4),
                    "affinity": round(affinity, 4),
                },
            }

        # 排序用 ``rank_key``（= base × 使用度加成）；``score`` 保持**不含使用度**
        # 的原生相关分，专供下面的准入门槛判定 —— 即「排序看加成后的分、
        # 准入看原生分」。两者分开，使用度就只是排序微调，不会抬门槛。
        ranked = sorted(scored.values(), key=lambda x: x["rank_key"], reverse=True)

        # ``kb.min_score`` 过滤。归一化后该阈值是**统一量纲**（[0,1]），
        # 不再像旧版那样受关键词 0~12 分制影响。默认 0.0 时完全不过滤。
        # ⚠️ 判 ``score``（原生），**不是** ``rank_key`` —— 否则使用度会抬门槛。
        min_score = float(getattr(kb_cfg, "min_score", 0.0) or 0.0)
        if min_score > 0.0:
            ranked = [item for item in ranked if item["score"] >= min_score]

        # MMR 去冗余（``mmr_lambda ∈ (0,1)`` 时）：从候选池贪心选 top_k，
        # 近似重复项被后来的多样性项挤掉；关闭时保持历史截断行为。
        if 0.0 < lam < 1.0 and len(ranked) > top_k:
            ranked = self._mmr_select(ranked, top_k, lam)
            for item in ranked:
                item.setdefault("trace", {})["mmr"] = True
        else:
            ranked = ranked[:top_k]

        # 反链统计（只对 top_k 结果补充，避免全库扫描放大）
        backlink_cache: Dict[str, List[str]] = {}
        for item in ranked:
            title = item["title"]
            if title not in backlink_cache:
                backlink_cache[title] = self.links(title)
            item["backlinks"] = backlink_cache[title]
        return ranked

    # -- 使用度（亲和度数据源）与 MMR ----------------------------------

    def _usage_store(self) -> _UsageStore:
        """使用度表句柄（懒建目录/表；失败在 _UsageStore 内部降级）。"""
        if self._usage is None:
            self._usage = _UsageStore(self._usage_db)
        return self._usage

    def touch(self, rel_paths: List[str]) -> None:
        """记录命中使用度（亲和度数据源，也是 Phase 2 晋升/归档的依据）。

        仅在 ``kb.affinity_enabled`` 开启时落表 —— 默认关闭，读路径零副作用。
        调用方是召回提示通道（命中 = 注入了提示），不是搜索本身：搜索不等于
        消费，计数只记真正被用上的条目。统计失败绝不影响召回。
        """
        rel_paths = [p for p in (rel_paths or []) if p]
        if not rel_paths:
            return
        if not bool(getattr(self._config.kb, "affinity_enabled", False)):
            return
        try:
            self._usage_store().touch(rel_paths)
        except Exception as e:  # noqa: BLE001 - 统计失败绝不影响召回
            logger.debug("kb touch failed: %s", e)

    @staticmethod
    def _mmr_select(ranked: List[Dict[str, Any]], top_k: int,
                    lam: float) -> List[Dict[str, Any]]:
        """MMR 贪心选 top_k：``λ·融合分 − (1−λ)·与已选项的最大相似度``。

        相似度取 title+snippet 的 bigram Jaccard（零依赖，中文直接可用）。
        候选池按融合分截取（``top_k*5``，钳 [20, 60]）：池外的结果本来就排
        在池内所有项之后，没有被选中的资格。相似度为 0 时退化为纯融合分
        排序 —— MMR 只挤掉近似重复项，不搅动本来就不相似的结果。
        """
        if len(ranked) <= top_k:
            return ranked
        pool_size = min(len(ranked),
                        max(top_k * _MMR_POOL_FACTOR, _MMR_POOL_MIN),
                        _MMR_POOL_MAX)
        pool = ranked[:max(pool_size, top_k)]

        grams = [_bigrams((it.get("title") or "") + " " + (it.get("snippet") or ""))
                 for it in pool]
        remaining = list(range(len(pool)))
        selected: List[int] = []
        while remaining and len(selected) < top_k:
            best_i, best_val = remaining[0], float("-inf")
            for i in remaining:
                sim = 0.0
                if selected:
                    gi = grams[i]
                    sim = max(
                        (len(gi & grams[j]) / len(gi | grams[j])
                         if (gi | grams[j]) else 0.0)
                        for j in selected
                    )
                # 相关性用 ``rank_key``（= 相关分 × 使用度加成）—— 与排序口径一致。
                # 用 ``score`` 会让使用度在 MMR 内部被忽略，出现「排序认它、
                # 选池不认它」的半套行为。准入仍只看 ``score``（见 search()）。
                _rel = float(pool[i].get("rank_key", pool[i].get("score", 0.0)) or 0.0)
                val = lam * _rel - (1.0 - lam) * sim
                if val > best_val:
                    best_i, best_val = i, val
            selected.append(best_i)
            remaining.remove(best_i)
        return [pool[i] for i in selected]

    # -- 写入 --------------------------------------------------------

    @staticmethod
    def _autolink(body: str, concepts: List[str]) -> str:
        """对 concepts 里的词在正文首次出现处加 [[词]]（已链接的跳过）。"""
        for c in concepts:
            c = c.strip()
            if not c:
                continue
            if re.search(rf"\[\[{re.escape(c)}(\||\])", body):
                continue
            idx = body.find(c)
            if idx != -1:
                body = body[:idx] + f"[[{c}]]" + body[idx + len(c):]
        return body

    @staticmethod
    def _link_if_missing(body: str, target: str) -> Tuple[str, bool]:
        """若 body 尚无 ``[[target]]``（或 ``[[target|alias]]``），追加到「相关笔记」段。

        返回 ``(new_body, added)``；已存在则原样返回且 added=False。
        """
        target = (target or "").strip()
        if not target:
            return body, False
        if re.search(rf"\[\[{re.escape(target)}(\||\])", body):
            return body, False
        return _vault.append_link_section(body, [target]), True

    def _link_related(
        self,
        path: Path,
        title: str,
        meta: Dict[str, Any],
        body: str,
        concepts: List[str],
        tags: List[str],
    ) -> int:
        """双向链接补全：为刚写入的笔记找共享 concepts/tags 的笔记并互加
        ``[[wikilink]]``（本笔记 → 对方，对方 → 本笔记）。返回新增链接数。

        线性扫描即可：个人 vault（数千篇内）毫秒级；未来规模大了再换索引。
        """
        concept_set = {c.strip().lower() for c in concepts if c.strip()}
        tag_set = {t.strip().lower() for t in tags if t.strip()}
        if not concept_set and not tag_set:
            return 0

        related: List[_Note] = []
        for note in self._iter_notes():
            if note.title == title:
                continue
            nc = {c.strip().lower() for c in note.concepts if c.strip()}
            nt = {t.strip().lower() for t in note.tags if t.strip()}
            if (concept_set & nc) or (tag_set & nt):
                related.append(note)
        if not related:
            return 0

        added = 0
        # 反向：把本笔记标题加进每个相关笔记
        for note in related:
            ometa, obody = _vault.read_note(note.path)
            nobody, ok = self._link_if_missing(obody, title)
            if ok:
                _vault.write_note(note.path, ometa, nobody)
                self._index_get().upsert(
                    self._rel(note.path),
                    _Note(note.path, note.title, ometa, nobody).full_text,
                    mtime=_file_mtime(note.path),
                )
                added += 1

        # 正向：把相关笔记标题加进本笔记
        pending = [
            n.title
            for n in related
            if not re.search(rf"\[\[{re.escape(n.title)}(\||\])", body)
        ]
        if pending:
            nbody = _vault.append_link_section(body, pending)
            _vault.write_note(path, meta, nbody)
            self._index_get().upsert(
                self._rel(path), _Note(path, title, meta, nbody).full_text,
                mtime=_file_mtime(path),
            )
            added += len(pending)
        return added

    def add(
        self,
        title: str,
        body: str,
        section: str = "notes",
        tags: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        source: str = "",
        update: bool = False,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """写入/更新一篇笔记到 vault，并更新向量索引。

        ``section`` 必须在标准子目录内，否则回退到 ``inbox``（半自动沉淀的
        安全落点，不会污染目录结构）。

        ``confidence``（0-1）是半自动沉淀的置信信号：传了它，落库分区由
        阈值决定 —— ``>= kb.confidence_threshold`` 进 ``notes``（正库），
        低于阈值进 ``inbox``（frontmatter 标 ``review_required``）；不传则
        按显式 ``section`` 落库（手动场景，行为不变）。

        落库前对 title/body 过密钥闸门（fail-closed）：含密钥一律拒绝，
        绝不写入 vault。
        """
        title = (title or "").strip() or "untitled"
        body = (body or "").strip()
        concepts = _as_list(concepts)
        tags = _as_list(tags)

        # 密钥闸门（fail-closed）：先于任何写盘
        if has_secret_like_text(title) or has_secret_like_text(body):
            return {
                "ok": False,
                "error": "content contains secret-like text; refused (fail-closed)",
            }

        # 置信分流：传了 confidence 就由阈值接管 notes/inbox
        review_required = False
        if confidence is not None:
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                conf = None
            if conf is not None:
                threshold = float(
                    getattr(self._config.kb, "confidence_threshold", 0.7) or 0.7
                )
                if conf < threshold:
                    section = "inbox"
                    review_required = True
                else:
                    section = "notes"

        if section not in _vault.VAULT_SUBDIRS:
            section = "inbox"
        body = self._autolink(body, concepts)

        filename = _vault.slugify_filename(title)
        path = self._vault / section / f"{filename}.md"

        # 包含性查重（跨标题、跨区）：新正文与既有笔记互相包含且都不琐碎 →
        # 返回既有路径而非静默新建第二份。同一篇的更新（同路径）不算重复。
        # 这是 WeKnora 写路径「duplicate resolution」的廉价可解释版：线性扫描
        # 一次（个人 vault 毫秒级），比引入相似度索引简单且无假阴性惊喜。
        norm_new = _normalize_for_dedup(body)
        if len(norm_new) >= _DEDUP_MIN_CHARS:
            for existing in self._iter_notes():
                if existing.path == path:
                    continue
                norm_old = _normalize_for_dedup(existing.body)
                if len(norm_old) < _DEDUP_MIN_CHARS:
                    continue
                if norm_new in norm_old or norm_old in norm_new:
                    return {
                        "ok": False,
                        "duplicate": True,
                        "error": (f"body duplicates existing note "
                                  f"'{existing.title}'"),
                        "path": self._rel(existing.path),
                        "section": existing.section,
                        "hint": ("update the existing note instead (same "
                                 "title), or rewrite to add new information"),
                    }

        now = time_str()
        meta: Dict[str, Any] = {
            "title": title,
            "type": section,
            "status": _status_for_section(section),
            "tags": tags,
            "concepts": concepts,
            "source": source or "",
            "created": now,
            "updated": now,
        }
        # 入库富化（kb.enrich 开启时）：一次 LLM 调用换 summary/questions
        # 进 frontmatter；失败返回 {}，写入照常 —— 富化绝不阻断落库。
        if bool(getattr(self._config.kb, "enrich", False)):
            meta.update(_enrich_note(title, body, self._config))
        if confidence is not None:
            meta["confidence"] = float(confidence)
        if review_required:
            meta["review_required"] = True
        if update:
            existing = self.get(title)
            if existing:
                meta["created"] = existing["meta"].get("created", now)

        _vault.write_note(path, meta, body)

        # 更新向量索引（尽力而为）；索引路径用相对 vault 的路径，与检索去重键一致
        note = _Note(path, title, meta, body)
        self._index_get().upsert(self._rel(path), note.full_text,
                                 mtime=_file_mtime(path))

        # 链接补全：只对正库 notes 做（inbox 待审先不互链，避免噪音）
        links_added = 0
        if section == "notes" and getattr(self._config.kb, "autolink_related", True):
            links_added = self._link_related(path, title, meta, body, concepts, tags)

        return {
            "ok": True,
            "title": title,
            "path": self._rel(path),
            "section": section,
            "tags": tags,
            "concepts": concepts,
            "review_required": review_required,
            "links_added": links_added,
        }

    def list_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出 inbox 待审笔记（``section=inbox`` 或 ``review_required``）。"""
        out: List[Dict[str, Any]] = []
        for note in self._iter_notes():
            review_required = bool(note.meta.get("review_required", False))
            if note.section != "inbox" and not review_required:
                continue
            out.append({
                "title": note.title,
                "path": self._rel(note.path),
                "section": note.section,
                "confidence": note.meta.get("confidence"),
                "review_required": review_required,
                "created": note.meta.get("created", ""),
                "updated": note.meta.get("updated", ""),
                "preview": note.body[:200],
            })
            if len(out) >= limit:
                break
        return out

    def _relocation_blocked(
        self, note: "_Note", new_path: Path, target_section: str, reason: str,
    ) -> Dict[str, Any]:
        """旧笔记删不掉时的失败收尾（见 :meth:`_relocate`）。

        到这里时**新文件已经写好了**，所以第一件事是尽量把现场还原成「只存在于
        inbox」—— 删掉刚写的副本，让一次失败的 ``approve`` 不留副作用。回滚也
        失败时（同一篇笔记 inbox 和 notes 各一份 = 重复数据）必须**明说**：
        调用方要能区分「什么都没发生」和「现在库里有两份」。

        Returns:
            ``ok=False`` 的结果 dict。带 ``rolled_back`` / ``partial`` /
            ``stale_path`` 三个字段，让调用方知道现场到底是什么样子，而不是拿到
            一个孤零零的 False。
        """
        logger.error(
            "[kb] relocate: wrote '%s' but could not remove the original '%s' "
            "(%s) — attempting rollback", new_path, note.path, reason)

        rolled_back = _unlink_with_retry(new_path) is None
        if rolled_back:
            return {
                "ok": False,
                "error": (
                    "could not move '%s' to %s: original still in inbox (%s); "
                    "rolled back, the note is untouched in inbox"
                    % (note.title, target_section, reason)),
                "title": note.title,
                "rolled_back": True,
                "partial": False,
                "stale_path": self._rel(note.path),
            }

        logger.error(
            "[kb] relocate: ROLLBACK FAILED — '%s' now exists in BOTH inbox and "
            "%s; manual cleanup required", note.path.name, target_section)
        return {
            "ok": False,
            "error": (
                "could not move '%s' to %s: original still in inbox (%s); "
                "rollback ALSO failed, so the note now exists in BOTH places "
                "(%s and %s) — remove one copy manually to avoid duplicates"
                % (note.title, target_section, reason,
                   self._rel(note.path), self._rel(new_path))),
            "title": note.title,
            "rolled_back": False,
            "partial": True,
            "stale_path": self._rel(note.path),
            "new_path": self._rel(new_path),
        }

    def _relocate(
        self,
        title: str,
        target_section: str,
        extra_meta: Optional[Dict[str, Any]] = None,
        from_sections: Tuple[str, ...] = ("inbox",),
    ) -> Dict[str, Any]:
        """把待审/自动区笔记移到 target_section（改 frontmatter + 移文件 + 更新索引）。

        默认仅允许从 ``inbox`` 移出（approve/reject 的语义不变）；治理巡检
        归档 ``knowledge/`` 时传 ``from_sections=("inbox", "knowledge")``。
        目标已存在同名文件时拒绝（不覆盖）。

        ``status`` 随目标区更新（notes→curated / archive→archived，见
        :func:`_status_for_section`）—— 移动了位置却不改状态，双库分工契约
        就会在第一次晋升后失效。

        旧笔记删不掉时返回 ``ok=False``（详见 :meth:`_relocation_blocked`）。
        历史上这一步只 ``logger.warning`` 然后照常 ``ok=True`` —— 于是 Windows
        上文件被别的线程/进程打开（WinError 32）时，``approve`` 报告成功、旧文件
        却留在 inbox，笔记**同时存在于 inbox 和 notes**。重复数据是真实产品缺陷，
        而调用方毫不知情：把一次没做完的移动说成成功，正是本项目一直在清的那种
        静默失败。
        """
        for note in self._iter_notes():
            if note.title != title and note.path.stem != title:
                continue
            if note.section not in from_sections:
                return {
                    "ok": False,
                    "error": (f"Note '{title}' is not in "
                              f"{'/'.join(from_sections)} "
                              f"(section={note.section})"),
                }
            new_path = self._vault / target_section / note.path.name
            if new_path != note.path and new_path.exists():
                return {
                    "ok": False,
                    "error": f"Target already exists: {self._rel(new_path)}",
                }

            meta = dict(note.meta)
            meta["type"] = target_section
            meta["status"] = _status_for_section(target_section)
            meta["updated"] = time_str()
            meta.pop("review_required", None)
            meta.pop("promote_suggest", None)
            if extra_meta:
                meta.update(extra_meta)

            body = note.body
            _vault.write_note(new_path, meta, body)
            if new_path != note.path:
                reason = _unlink_with_retry(note.path)
                if reason is not None:
                    return self._relocation_blocked(
                        note, new_path, target_section, reason)

            idx = self._index_get()
            idx.delete(self._rel(note.path))
            idx.upsert(self._rel(new_path), _Note(new_path, title, meta, body).full_text,
                       mtime=_file_mtime(new_path))

            return {
                "ok": True,
                "title": title,
                "path": self._rel(new_path),
                "section": target_section,
                "from": note.section,
            }
        return {"ok": False, "error": f"Note not found: {title}"}

    def approve(self, title: str) -> Dict[str, Any]:
        """批准 inbox 待审笔记 → 移入 notes，并补双向链接。"""
        r = self._relocate(title, "notes")
        if not r.get("ok"):
            return r
        links_added = 0
        note = self.get(title)
        if note and getattr(self._config.kb, "autolink_related", True):
            concepts = _as_list(note.get("meta", {}).get("concepts"))
            tags = _as_list(note.get("meta", {}).get("tags"))
            if concepts or tags:
                abs_path = self._vault / Path(note["path"])
                links_added = self._link_related(
                    abs_path,
                    title,
                    note.get("meta", {}),
                    note.get("body", ""),
                    concepts,
                    tags,
                )
        r["links_added"] = links_added
        return r

    def reject(self, title: str) -> Dict[str, Any]:
        """拒绝 inbox 待审笔记 → 移入 archive（标 review_rejected）。"""
        return self._relocate(title, "archive", {"review_rejected": True})

    def governance(self, apply: bool = False) -> Dict[str, Any]:
        """晋升建议 + 自动归档巡检（双库分工的淘汰闭环，Phase 2）。

        * **晋升建议**：inbox 中命中 ≥ ``kb.promote_hits`` 且未被拒过的笔记
          标 ``promote_suggest: true``。晋升本身仍由人通过
          ``governed_kb_review approve`` 确认 —— 阈值触发的是**建议**，
          不是自动晋升（对应 WeKnora 的 interest_threshold：慢信号过阈值
          才进入长期区，否则 inbox 会变成第二个正库）。
        * **自动归档**：自动区（inbox/knowledge）中**闲置超过
          ``kb.auto_archive_days`` 天**的笔记 → 移入 ``archive/``
          （降级不删除）。闲置 = 有命中记录看 ``last_used``；从未命中则按
          文件 mtime 算（出生即未用的时钟）。

        **fail-closed**：没有任何使用度数据（``kb.affinity_enabled`` 关或表
        空）时**跳过归档**并给出 reason —— 「没数据」≠「没被用过」，否则
        affinity 关闭期间的第一次巡检会按 mtime 把整个自动区误清。

        ``apply=False``（默认）只报告不写盘 —— 批量操作先报 scope。
        Returns:
            ``{ok, applied, promote_suggest, archivable, flagged, archived,
            errors, usage_data, reason}``。
        """
        cfg = self._config.kb
        promote_hits = int(getattr(cfg, "promote_hits", 3) or 3)
        archive_days = int(getattr(cfg, "auto_archive_days", 45) or 45)
        # fail-closed 的**两半**：开关关**或表空**都算「没有使用度数据」。
        # 只判开关会让「affinity 开着、表里还一行没有」时按 mtime 把整个自动区
        # 误清 —— 本方法 docstring 早已承诺「关**或表空**」，此前只实现了前一半。
        # 今日半径碰巧为 0（自动区为空 + mtime 全是新的），但 `add()` 落低置信
        # 笔记进 ``inbox/`` 是常规路径，两个条件都**不是保证**。
        affinity_on = bool(getattr(cfg, "affinity_enabled", False))
        entries: Dict[str, Tuple[int, float]] = (
            self._usage_store().entries() if affinity_on else {})
        usage_available = affinity_on and bool(entries)

        now = time.time()
        promote: List[Dict[str, Any]] = []
        archivable: List[Dict[str, Any]] = []
        flagged: List[str] = []
        moved: List[str] = []
        errors: List[str] = []

        for note in list(self._iter_notes()):
            rel = self._rel(note.path)
            hits, last_used = entries.get(rel, (0, 0.0))

            if (note.section == "inbox"
                    and not note.meta.get("review_rejected")
                    and not note.meta.get("promote_suggest")
                    and hits >= promote_hits > 0):
                promote.append({"title": note.title, "path": rel, "hits": hits})
                if apply:
                    try:
                        meta = dict(note.meta)
                        meta["promote_suggest"] = True
                        _vault.write_note(note.path, meta, note.body)
                        self._index_get().upsert(
                            rel,
                            _Note(note.path, note.title, meta,
                                  note.body).full_text,
                            mtime=_file_mtime(note.path))
                        flagged.append(rel)
                    except OSError as e:
                        errors.append(f"flag failed: {rel}: {e}")

            if note.section not in ("inbox", "knowledge") or not usage_available:
                continue
            if hits > 0:
                idle_days = (now - last_used) / 86400.0 if last_used > 0 else 0.0
            else:
                mtime = _file_mtime(note.path)
                idle_days = ((now - mtime) / 86400.0) if mtime > 0 else 0.0
            if idle_days <= archive_days:
                continue
            archivable.append({"title": note.title, "path": rel,
                               "section": note.section,
                               "idle_days": round(idle_days, 1)})
            if apply:
                r = self._relocate(
                    note.title, "archive",
                    {"archived_reason": "auto-unused"},
                    from_sections=("inbox", "knowledge"))
                if r.get("ok"):
                    moved.append(str(r.get("path", rel)))
                else:
                    errors.append(f"archive failed: {rel}: "
                                  f"{r.get('error', 'unknown')}")

        return {
            "ok": True,
            "applied": bool(apply),
            "promote_suggest": promote,
            "archivable": archivable,
            "flagged": flagged,
            "archived": moved,
            "errors": errors,
            "usage_data": usage_available,
            "reason": ("" if usage_available else
                       "no usage data (kb.affinity_enabled off?) — "
                       "archive evaluation skipped (fail-closed)"),
        }

    def stats(self) -> Dict[str, Any]:
        """知识库统计（供 health 面板）。"""
        notes = self._iter_notes()
        by_section: Dict[str, int] = {}
        for n in notes:
            by_section[n.section] = by_section.get(n.section, 0) + 1
        idx = self._index_get()
        # 索引新鲜度：正本 vs 投影的差异。``index_missing`` / ``index_stale``
        # 长期不为 0 说明后台增量同步没跑起来（外部写入会持续漏索引）。
        indexed = idx.path_mtimes() if idx.available else {}
        missing = 0
        stale = 0
        for n in notes:
            cur = indexed.get(self._rel(n.path))
            if cur is None:
                missing += 1
            elif abs(cur - _file_mtime(n.path)) > 0.5:
                stale += 1
        return {
            "total": len(notes),
            "by_section": by_section,
            "index_available": idx.available,
            "index_error": idx.last_error or "",
            "index_reprobe_count": int(getattr(idx, "reprobe_count", 0) or 0),
            "indexed": len(indexed),
            "index_missing": missing,
            "index_stale": stale,
        }


def time_str() -> str:
    """ISO 日期（frontmatter created/updated 用）。"""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d")
