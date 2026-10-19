# -*- coding: utf-8 -*-
"""API reranker —— 把 ``RerankingConfig`` 从"配置壳"落地成可用的重排器。

设计位来历：``_config.RerankingConfig``（provider / api_key_env / base_url /
model / top_n）早前已挂在主配置上，但全仓库没有任何实现读取它。本模块把它
接通；接入点在 ``_recall.RecallEngine._rerank_l2``（只重排 L2 候选、默认关）。

为什么是 API reranker 而不是本地 CrossEncoder：本机 torch /
sentence-transformers 均不可用，且仓库纪律是不引重依赖。生产环境本就用
siliconflow API 做嵌入（``_embedding``，``BAAI/bge-m3`` 1024 维），重排走
同一条路最自然（``BAAI/bge-reranker-v2-m3``）。请求/响应形状
（``{"model", "query", "documents"}`` → ``results[].index/relevance_score``）
是 siliconflow / cohere / jina 三家共用的形态。

失败哲学（本项目铁律）：未配置 / 密钥缺失 / 调用失败 / 返回格式不符 ——
一律 ``log_degraded`` 后**原样返回**，绝不抛异常、绝不让读路径变空或变慢。
``log_degraded`` 的"永不抛"契约与"只传 ``exc=e``、禁止 ``str(e)`` 急切求值"
（坏 ``__str__`` 会把优雅降级变成崩溃）见 ``_diag``。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from ._config import env_secret
from ._diag import log_degraded

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，避免运行时循环导入
    from ._config import RerankingConfig

logger = logging.getLogger(__name__)


class Reranker:
    """从 ``RerankingConfig`` 构造的 API 重排器。

    对外契约（**分层的**，别当成"整条调用链都不抛"）：

    * :attr:`available` —— provider 与 ``api_key_env`` 均已配置、且密钥可解析
      （``env_secret`` 支持进程环境变量 + ``$HERMES_HOME/.env`` 回退）。
      "配没配"与"开没开"是两个问题：开关（``enabled``）由接入点
      （``_recall._rerank_l2``）先把关，本属性只回答"能不能调"。
    * :meth:`rerank` —— 对**构造良好的 items** 永不抛异常：成功时返回**同一
      批元素**的新顺序，任何失败都记 ``rerank::call_failed`` /
      ``rerank::bad_response`` 后原序返回。集合永不改变（重排不能丢条目）。
    * **入参本身畸形**时（``items=None``、某个元素的 ``content.__str__()``
      会抛）:meth:`rerank` **可能抛** —— 它的承诺只覆盖"构造良好"。**读路径
      永不见异常由接入层** ``_recall._rerank_l2`` 的兜底网保证（那里整包住
      本方法的调用）。⚠️ 不要为了"看起来更稳"而在本方法内把这类入参错误吞成
      空列表返回：那等于把"入参坏了"变成**静默丢条目**，比抛出来更糟。

    只依赖标准库 + 项目既有的 HTTP 调用方式（与
    ``_embedding.EmbeddingService._embed_via_api`` 同款：函数内 ``import
    httpx``、显式 timeout、``raise_for_status``）—— 不引 requests 等新依赖。
    """

    #: API 调用超时（秒）。与 ``EmbeddingService.API_TIMEOUT_SECONDS`` 对齐：
    #: 重排发生在读路径上，超时必须有上限。
    API_TIMEOUT_SECONDS: float = 10.0

    def __init__(self, config: Optional["RerankingConfig"] = None) -> None:
        """从配置构造；``config`` 为 ``None`` 时按"未配置"处理（不可用）。

        密钥在构造期解析一次：``env_secret`` 自身"永不抛"（内部对 ``.env``
        读取失败已兜底），缺失即空串 → :attr:`available` 为 False。
        """
        self._provider: str = str(getattr(config, "provider", "") or "")
        self._base_url: str = str(getattr(config, "base_url", "") or "").rstrip("/")
        self._api_key_env: str = str(getattr(config, "api_key_env", "") or "")
        self._model: str = str(getattr(config, "model", "") or "")
        self._enabled: bool = bool(getattr(config, "enabled", False))
        # ⚠️ 用 `is None` 判缺，**绝不能**写成 `or 10`：``top_n=0`` 是**合法值**
        # （文档语义"<=0 = 全量提交"），而 ``0 or 10`` → 10，会把"想全量重排"的
        # 部署**静默**降级成"只提交前 10 条"且不报任何错（2026-10-19 QA 抓到）。
        # JSON 里显式写 ``"top_n": null`` 走同一条回退（属性存在但为 None）。
        raw_top_n = getattr(config, "top_n", None)
        try:
            self._top_n: int = int(raw_top_n) if raw_top_n is not None else 10
        except (TypeError, ValueError):  # silent-ok: value-fallback — 坏 top_n 配置回退默认 10，非子系统故障
            self._top_n = 10
        self._api_key: str = env_secret(self._api_key_env) if self._api_key_env else ""
        self._available: bool = bool(
            self._provider and self._api_key_env and self._api_key
        )

    # -- 对外契约 --------------------------------------------------------

    @property
    def available(self) -> bool:
        """reranker 是否可用：provider 已配 + api_key_env 有值 + 密钥可解析。

        刻意**不**把 ``enabled`` 算进可用性，也**不**把 ``base_url`` 算进去：
        base_url 缺失会在调用时以 ``rerank::call_failed`` 可见地失败（优雅
        降级），而不是在可用性里被静默吞掉。
        """
        return self._available

    def rerank(self, query: str, items: Sequence[Any],
               top_n: Optional[int] = None) -> List[Any]:
        """对 ``items`` 按 ``query`` 重排，返回**同一批元素**的新顺序列表。

        **不抛的承诺只对构造良好的 ``items`` 成立**：``items=None``、或某个
        元素的 ``content`` / ``__str__()`` 会抛（畸形候选）时，本方法**可能
        抛**。读路径永不见异常由接入层 ``_recall._rerank_l2`` 兜住（那里整包
        住本次调用）；这里**刻意不**把入参错误吞成空列表 —— 那等于静默丢条目。

        Args:
            query: 查询串。空串直通（没有信号可重排）。
            items: 候选序列。元素可以是 ``(content, metadata)`` 元组（接口
                文档形态），也可以是任何带 ``.content`` 属性的对象
                （``_recall.RecallResult``，实际接入形态）—— 重排只**换序**，
                不碰元素本身。
            top_n: 只把前 ``top_n`` 个候选提交给 API（标准 retrieve-then-
                rerank：控制调用成本，候选通常已按分数降序）；``None`` 时取
                配置的 ``reranking.top_n``（配置项为 ``null`` 时回退默认 10）；
                **``0`` 与负值 = 全部提交**（``0`` 是合法值，不是"未配置"）。
                **任何情况下都不丢条目**：未提交的候选按原序留在尾部。

        Returns:
            **一个新列表**。重排成功时元素顺序按重排结果，元素本身是输入那
            同一批；未配置 / 开关关 / 空输入 / 调用失败 / 返回格式不符时：
            元素与顺序都与输入全同 —— 是"同一批元素、同一顺序"，但**不是输入
            那个列表对象**。要断言"同一对象"，请在接入层
            ``_recall._rerank_l2`` 上断言（只有那里会原样透传 ``results``）。
        """
        materialized = list(items)
        # 未配置 / 开关关 / 输入不足：直通，且**不记 degraded**（这不是故障，
        # 是"没什么可做"）。读取路径不能因重排未配置而多一声告警。
        if not self._enabled or not self._available:
            return materialized
        if len(materialized) < 2 or not str(query or "").strip():
            return materialized

        limit = self._resolve_limit(top_n, len(materialized))
        head = materialized[:limit]
        tail = materialized[limit:]
        # 下标对齐契约：documents 与 head 一一对应，API 返回的 index 直接
        # 映射回 head 的下标 —— 不过滤空文档（过滤会造成下标漂移）。
        documents = [self._content_of(item) for item in head]

        try:
            payload = self._post(query, documents)
        except Exception as e:  # noqa: BLE001 — 读路径绝不因重排失败而变空
            log_degraded("rerank", "call_failed", exc=e)
            logger.debug("rerank API call failed: %s", e)
            return materialized

        ranked = self._order_by_scores(payload, head)
        if ranked is None:
            log_degraded(
                "rerank", "bad_response",
                detail="no usable 'results' with index/relevance_score",
            )
            return materialized
        return ranked + tail

    # -- API 调用 --------------------------------------------------------

    def _post(self, query: str, documents: List[str]) -> Any:
        """调用 ``POST {base_url}/rerank`` 并返回解析后的 JSON。

        任何失败（连接错误、超时、非 2xx、JSON 解析失败）都抛给调用方记
        ``rerank::call_failed`` —— 传输层与语义层失败分开计数，排查时能直接
        看出是"没打通"还是"通了但格式不对"。
        """
        import httpx

        payload: dict = {
            "model": self._model,
            "query": query,
            "documents": documents,
        }
        response = httpx.post(
            f"{self._base_url}/rerank",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def _order_by_scores(self, payload: Any,
                         head: List[Any]) -> Optional[List[Any]]:
        """把 API 响应映射成 ``head`` 的重排顺序（同一批对象）。

        Returns:
            重排后的 ``head``；响应格式不符（非 dict / ``results`` 不是
            非空 list / 没有任何可用的 index+relevance_score）时返回
            ``None``（调用方记 ``rerank::bad_response`` 后直通）。

        语义：有分的按 ``relevance_score`` 降序（同分按原下标，保证确定性），
        没拿到分的候选按原序接在后面 —— 部分得分是部分可用，不是整体失败。
        """
        if not isinstance(payload, dict):
            return None
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None

        scored: dict = {}
        for entry in results:
            if not isinstance(entry, dict):
                continue
            index = entry.get("index")
            score = entry.get("relevance_score")
            # bool 是 int 的子类：JSON 里的 true/false 不是合法下标/分数
            if not isinstance(index, int) or isinstance(index, bool):
                continue
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                continue
            if 0 <= index < len(head) and index not in scored:
                scored[index] = float(score)
        if not scored:
            return None

        ranked_idx = sorted(scored, key=lambda i: (-scored[i], i))
        unscored = [i for i in range(len(head)) if i not in scored]
        return [head[i] for i in ranked_idx + unscored]

    def scores(self, query: str, items: Sequence[Any],
               top_n: Optional[int] = None) -> Dict[int, float]:
        """Relevance score per candidate index, or ``{}`` when unavailable.

        ``rerank`` only reorders, so it cannot answer "is this hit any good?".
        The KB channel needs the score itself: it wants to DROP irrelevant
        notes rather than reorder them, and only the reranker's score can tell
        the two apart (the embedding channel's cosine cannot — measured on the
        live vault, relevant and unrelated queries overlap at 0.41-0.49).

        Same posture as :meth:`rerank`: never raises on well-formed input, and
        an empty dict means "could not score" rather than "all zero". The
        caller decides what an unscored candidate means.
        """
        if not self._enabled or not self._available:
            return {}
        materialized = list(items)
        if not materialized or not str(query or "").strip():
            return {}

        limit = self._resolve_limit(top_n, len(materialized))
        documents = [self._content_of(item) for item in materialized[:limit]]

        try:
            payload = self._post(query, documents)
        except Exception as e:  # noqa: BLE001 — a read path never raises
            log_degraded("rerank", "scores_call_failed", exc=e)
            logger.debug("rerank scores call failed: %s", e)
            return {}

        out: Dict[int, float] = {}
        if isinstance(payload, dict):
            for entry in payload.get("results") or []:
                if not isinstance(entry, dict):
                    continue
                index = entry.get("index")
                score = entry.get("relevance_score")
                if isinstance(index, int) and not isinstance(index, bool) \
                        and isinstance(score, (int, float)) \
                        and not isinstance(score, bool) \
                        and 0 <= index < len(documents) and index not in out:
                    out[index] = float(score)
        if not out:
            log_degraded("rerank", "scores_bad_response",
                         detail="no usable 'results' with index/relevance_score")
        return out

    # -- 工具 ------------------------------------------------------------

    def _resolve_limit(self, top_n: Optional[int], total: int) -> int:
        """解析提交给 API 的候选数：``None``/非法 → 配置值；``<= 0`` → 全部。"""
        value: Any = top_n if top_n is not None else self._top_n
        try:
            value = int(value)
        except (TypeError, ValueError):  # silent-ok: value-fallback — 坏 top_n 实参回退配置值，非子系统故障
            value = self._top_n
        if value <= 0:
            return total
        return min(value, total)

    @staticmethod
    def _content_of(item: Any) -> str:
        """取候选的文本内容。

        支持两种形态：``(content, metadata)`` 元组（接口文档形态）与带
        ``.content`` 属性的对象（``_recall.RecallResult``，实际接入形态）。
        """
        if isinstance(item, tuple) and item:
            return str(item[0] or "")
        content = getattr(item, "content", None)
        if content is not None:
            return str(content)
        return str(item)
