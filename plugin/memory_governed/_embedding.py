# -*- coding: utf-8 -*-
"""Embedding backend abstraction for L2 semantic memory.

Backends (selected by config.vector.backend):
- "auto":                 try fastembed → sentence_transformers → none
- "fastembed":            qdrant's fastembed (ONNX, no PyTorch) — lightest, model
                          auto-downloaded on first use
- "sentence_transformers": sentence-transformers (torch) — original backend
- "none":                 no embeddings (L2 falls back to text scan)

All backends implement the Embedder protocol:
    model_name: str
    dim: int
    encode(texts: list[str]) -> list[list[float]]

:class:`EmbeddingService` 是进程内单例门面（T04）：读路径（``_recall``）和
写路径（``_sync``）都通过它取嵌入，避免两端各自内联一份降级阶梯、各自重复
加载模型。探测失败会负缓存，后续调用直接返回不可用 —— 这是"每次查询都重试
一遍 import"这一性能悬崖的根治手段。

Note: switching between backends that use the SAME underlying model
(e.g. fastembed's all-MiniLM-L6-v2 vs sentence-transformers' all-MiniLM-L6-v2)
keeps existing vectors usable. Switching to a DIFFERENT model requires
`python scripts/l2_rebuild.py` to re-embed existing rows.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

from ._config import env_secret
from ._text import sanitize_utf8

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查，避免运行时循环导入
    from ._config import GovernedMemoryConfig

logger = logging.getLogger(__name__)

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


class Embedder(Protocol):
    """Common protocol implemented by every backend."""

    model_name: str
    dim: int

    def encode(self, texts: list[str]) -> Any:
        """Encode texts, returning an object with .tolist() -> list[list[float]]."""
        ...


class _STBackend:
    """sentence-transformers backend (torch-based, original)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> Any:
        return self._model.encode(texts)


class _FastEmbedBackend:
    """fastembed backend (ONNX runtime, no PyTorch — lightest install)."""

    #: 维度探测文本（fastembed 是懒加载，维度只能 probe 出来）
    PROBE_TEXT: str = "dim probe"
    #: fastembed 用完整 repo id 索引模型，裸名需要补这个命名空间前缀
    ST_NAMESPACE: str = "sentence-transformers"

    @classmethod
    def _candidates(cls, model_name: str) -> list[str]:
        """返回按优先级排列的模型名候选。

        fastembed 用完整 repo id 索引模型表：``BAAI/bge-small-zh-v1.5`` 自带
        命名空间可直接用；裸名（旧文档里的 ``all-MiniLM-L6-v2``）必须补
        ``sentence-transformers/`` 前缀，否则会静默 fallback 到 fastembed 的
        默认英文模型（``BAAI/bge-small-en-v1.5``），中文语义质量大幅下降 ——
        而且只有一条 WARNING，用户极难察觉。

        Args:
            model_name: 配置里的模型名（可能带 / 也可能不带）。

        Returns:
            候选列表：原名优先（尊重用户显式指定的 repo），其次补前缀。
        """
        name = (model_name or "").strip()
        if not name:
            return []
        candidates = [name]
        if "/" not in name:
            candidates.append(f"{cls.ST_NAMESPACE}/{name}")
        return candidates

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        last_error: Optional[BaseException] = None

        for candidate in self._candidates(model_name):
            try:
                # 构造 **和** 维度探测必须在同一个 try 里：TextEmbedding 是懒
                # 加载的，模型名非法 / 权重下不动时构造函数可能不报错，真正的
                # 下载与加载发生在 embed() 调用时。把 probe 留在 try 外面会让
                # 加载失败逃逸出候选循环，冒泡到 load_embedder 被记成"整个
                # fastembed 后端失败"，直接降级到 sentence_transformers。
                model = TextEmbedding(model_name=candidate)
                sample = list(model.embed([self.PROBE_TEXT]))
                dim = len(sample[0])
            except Exception as e:  # noqa: BLE001 — 换下一个候选，不放弃整个后端
                last_error = e
                logger.debug("fastembed model %r unusable: %s", candidate, e)
                continue
            if dim <= 0:
                last_error = ValueError(f"empty probe vector for {candidate!r}")
                logger.debug("fastembed model %r returned an empty probe vector", candidate)
                continue
            self._model = model
            self.model_name = candidate
            self.dim = dim
            return

        # 所有候选都失败 —— 退回 fastembed 的默认模型（最后一道兜底）。
        logger.warning(
            "fastembed model %r unavailable (%s) — using default", model_name, last_error
        )
        self._model = TextEmbedding()
        self.model_name = "fastembed-default"
        sample = list(self._model.embed([self.PROBE_TEXT]))
        self.dim = len(sample[0])

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        arr = np.array(list(self._model.embed(texts)), dtype="float32")
        return type("R", (), {"tolist": lambda s: arr.tolist()})()


_BACKENDS = {
    "fastembed": _FastEmbedBackend,
    "sentence_transformers": _STBackend,
}


def load_embedder(backend: str, model_name: str) -> Embedder | None:
    """Load the configured embedding backend.

    backend="auto": fastembed → sentence_transformers → None (text fallback).
    Returns None when no backend is importable — callers fall back to
    non-vector paths (text scan), mirroring the rest of the graceful-degradation
    design.
    """
    order: list[str]
    if backend == "auto":
        order = ["fastembed", "sentence_transformers"]
    elif backend in _BACKENDS:
        order = [backend]
    else:
        logger.warning("unknown embedding backend %r — treating as auto", backend)
        order = ["fastembed", "sentence_transformers"]

    for name in order:
        cls = _BACKENDS.get(name)
        if cls is None:
            continue
        try:
            inst = cls(model_name)
            logger.info("L2 embedding backend: %s (model=%s, dim=%s)", name, inst.model_name, inst.dim)
            return inst
        except ImportError:
            logger.debug("embedding backend %r not installed", name)
        except Exception as e:
            logger.warning("embedding backend %r failed to load: %s", name, e)
    return None


class EmbeddingService:
    """进程内单例的嵌入服务（读路径 / 写路径共用）。

    设计要点：
    - :meth:`get` 双重检查锁单例；首次调用探测后端，失败也返回实例
      （``available=False``）并把原因负缓存，后续调用不再重试。
    - API embedding（``config.embedding.provider`` + ``base_url``）优先级最高，
      其次才是 ``config.vector.backend`` 指定的本地后端。
    - 所有 embed 方法**不抛异常**：失败返回 ``None``，原因写入 :attr:`last_error`。
    - 实际维度与 ``config.vector.dim`` 不一致时，用实际维度覆盖配置并告警
      （配置写错会让 L2 写入直接失败）。
    """

    #: API 调用超时（秒）
    API_TIMEOUT_SECONDS: float = 10.0
    #: 维度探测用的文本
    PROBE_TEXT: str = "dim probe"

    _instance: Optional["EmbeddingService"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, config: "GovernedMemoryConfig") -> None:
        self._config = config
        self._call_lock = threading.RLock()
        self._available: bool = False
        self._backend: Optional[Embedder] = None
        self._backend_name: str = "none"
        self._api_base_url: str = ""
        self._api_key: str = ""
        self._api_model: str = ""
        self._dim: int = self._configured_dim(config)
        self._last_error: str = ""
        self._signature = self._build_signature(config)
        self._probe()

    # -- 单例 ------------------------------------------------------------

    @classmethod
    def get(cls, config: "GovernedMemoryConfig") -> "EmbeddingService":
        """返回进程内单例；配置变化时重建（探针仍然只跑一次）。"""
        signature = cls._build_signature(config)

        instance = cls._instance
        if instance is not None and instance._signature == signature:
            return instance

        with cls._lock:
            instance = cls._instance
            if instance is not None and instance._signature == signature:
                return instance
            instance = cls(config)
            cls._instance = instance
            return instance

    @classmethod
    def reset(cls) -> None:
        """丢弃单例（仅供测试 / 配置热切换使用）。"""
        with cls._lock:
            cls._instance = None

    # -- 对外契约 --------------------------------------------------------

    @property
    def available(self) -> bool:
        """后端是否可用。不可用时所有 embed 方法返回 None。"""
        return self._available

    @property
    def dim(self) -> int:
        """实际向量维度。不可用时返回 config.vector.dim（配置值兜底）。"""
        return self._dim

    @property
    def last_error(self) -> str:
        """最近一次失败原因，供 health 诊断。"""
        return self._last_error

    @property
    def backend_name(self) -> str:
        """后端标识（"api:<provider>" / "local:<backend>" / "none"）。"""
        return self._backend_name

    @staticmethod
    def _sanitize(text: str) -> str:
        """Kept as a thin alias; the implementation lives in :mod:`._text`.

        There used to be three copies of this logic. This one, plus
        ``hgm_mcp._ensure_utf8`` and an inline block in ``memory_cli``, were
        identical — and none of them covered the plugin's own extraction path,
        which is where a fact actually reaches L2. One definition now, and the
        write pipeline sanitizes once at its ingress.
        """
        return sanitize_utf8(text)

    def embed_one(self, text: str) -> Optional[List[float]]:
        """单条嵌入。不可用或失败时返回 None（不抛异常）。"""
        if not text or not str(text).strip():
            return None

        text = self._sanitize(str(text))

        with self._call_lock:
            if not self._available:
                return None
            try:
                if self._api_base_url:
                    vectors = self._embed_via_api([text])
                    return vectors[0] if vectors else None
                if self._backend is not None:
                    vectors = self._encode_local([text])
                    return vectors[0] if vectors else None
            except Exception as e:  # noqa: BLE001 — 嵌入永远不能打断主流程
                self._last_error = f"embed_one failed: {e}"
                logger.debug("EmbeddingService embed_one failed: %s", e)
                return None
        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """批量嵌入，返回与输入等长的列表，失败项为 None。"""
        if not texts:
            return []
        normalized = [self._sanitize("" if t is None else str(t)) for t in texts]

        with self._call_lock:
            if not self._available:
                return [None] * len(normalized)
            try:
                if self._api_base_url:
                    vectors = self._embed_via_api(normalized)
                elif self._backend is not None:
                    vectors = self._encode_local(normalized)
                else:
                    vectors = []
            except Exception as e:  # noqa: BLE001 — 嵌入永远不能打断主流程
                self._last_error = f"embed_batch failed: {e}"
                logger.debug("EmbeddingService embed_batch failed: %s", e)
                vectors = []

        return self._align(vectors, len(normalized))

    # -- 探测 ------------------------------------------------------------

    def _probe(self) -> None:
        """探测后端一次；无论成败都把结果缓存到实例上。

        降级策略：API 优先 → 本地后端 → 不可用。
        当 API 配置存在但探测失败时，自动尝试本地后端作为降级，
        而不是直接标记为不可用。这样可以保证：
        1. API key 有效时使用 API (1024 维)
        2. API key 无效或网络问题时降级到本地 (512 维)
        3. 本地也不可用时才标记为不可用
        """
        config = self._config
        vector = getattr(config, "vector", None)
        embedding = getattr(config, "embedding", None)
        backend = str(getattr(vector, "backend", "auto") or "auto").lower()
        model = getattr(vector, "model", "") or ""

        # 1) API embedding 优先级最高（无需本地模型，跨机器一致性最好）
        provider = str(getattr(embedding, "provider", "") or "")
        base_url = str(getattr(embedding, "base_url", "") or "")
        api_key_env = str(getattr(embedding, "api_key_env", "") or "")
        api_key = env_secret(api_key_env)
        # API 配置存在（即使 key 无效）都标记为 api_configured
        # 这样在降级到本地时可以正确标记 is_fallback=True
        api_configured = bool(provider and base_url and api_key_env)

        if api_configured and api_key:
            self._api_base_url = base_url.rstrip("/")
            self._api_key = api_key
            self._api_model = str(getattr(embedding, "model", "") or "")
            try:
                vectors = self._embed_via_api([self.PROBE_TEXT])
            except Exception as e:  # noqa: BLE001
                # API 失败，记录但继续尝试本地后端
                logger.warning("API embedding probe failed, trying local backend: %s", e)
                log_degraded("embedding", "api_probe_failed",
                            detail=f"{type(e).__name__}: {e}", exc=e)
                # 不 return，继续到本地后端探测
            else:
                if vectors and vectors[0]:
                    self._available = True
                    self._backend_name = f"api:{provider}"
                    self._last_error = ""
                    self._record_actual_dim(len(vectors[0]))
                    logger.info(
                        "EmbeddingService ready: api backend %s (model=%s, dim=%s)",
                        provider, self._api_model, self._dim,
                    )
                    return
                # API 返回空向量，记录但继续尝试本地后端
                logger.warning("API embedding returned empty vector, trying local backend")
                log_degraded("embedding", "api_empty_vector",
                            detail="API probe returned an empty vector")
        elif api_configured and not api_key:
            # API 配置存在但 key 无效/缺失，记录并继续到本地后端
            logger.warning("API embedding key not found (env=%s), trying local backend", api_key_env)
            log_degraded("embedding", "api_key_missing",
                        detail=f"env var {api_key_env!r} not set or empty")

        # 2) 本地后端（fastembed / sentence-transformers）
        #    作为 API 的降级方案，或当 API 未配置时使用
        if backend == "none":
            self._mark_unavailable("vector.backend is 'none'")
            return

        try:
            embedder = load_embedder(backend, model)
        except Exception as e:  # noqa: BLE001
            self._mark_unavailable(f"backend probe raised: {e}", exc=e)
            return

        if embedder is None:
            self._mark_unavailable(
                f"no local embedding backend importable (backend={backend})"
            )
            return

        self._backend = embedder
        self._available = True
        self._backend_name = f"local:{backend}"
        self._last_error = ""
        # 如果是从 API 降级过来的，标记为降级
        is_fallback = api_configured
        self._record_actual_dim(
            int(getattr(embedder, "dim", 0) or 0),
            is_fallback=is_fallback,
        )
        logger.info(
            "EmbeddingService ready: local backend %s (model=%s, dim=%s)%s",
            backend, getattr(embedder, "model_name", model), self._dim,
            " (fallback from API)" if is_fallback else "",
        )

    def _mark_unavailable(self, reason: str, *, exc: BaseException | None = None) -> None:
        """把实例标记为不可用并负缓存原因（后续调用不再重试探测）。"""
        self._available = False
        self._backend = None
        self._backend_name = "none"
        self._last_error = reason
        logger.debug("EmbeddingService unavailable: %s", reason)
        log_degraded("embedding", "unavailable", detail=reason, exc=exc)

    def _record_actual_dim(self, actual_dim: int, *, is_fallback: bool = False) -> None:
        """用实际维度覆盖配置并告警（配置写错会导致 L2 写入直接失败）。

        覆盖目标与场景一致：API 场景写 ``embedding.dimensions``，本地场景写
        ``vector.dim``，避免把 API 的实际维度污染到本地模型配置字段里。

        Args:
            actual_dim: 探测到的实际向量维度。
            is_fallback: 是否是因为降级而切换到此维度（API → 本地）。
                降级时维度变化是正常行为，日志级别降低为 info。
        """
        if actual_dim <= 0 or actual_dim == self._dim:
            return
        configured = self._dim
        self._dim = actual_dim
        try:
            embedding = getattr(self._config, "embedding", None)
            provider = str(getattr(embedding, "provider", "") or "")
            base_url = str(getattr(embedding, "base_url", "") or "")
            if provider and base_url and not is_fallback:
                embedding.dimensions = actual_dim
            else:
                self._config.vector.dim = actual_dim
        except Exception as e:  # pragma: no cover - 配置对象被替换成只读桩时
            logger.debug("Failed to sync dim to config: %s", e)

        if is_fallback:
            # 降级时维度变化是正常行为，记录为 info 而非 warning
            logger.info(
                "Embedding dim changed due to fallback: %d → %d (API → local)",
                configured, actual_dim,
            )
        else:
            log_degraded(
                "embedding",
                "vector_dim_mismatch",
                detail=f"configured={configured}, actual={actual_dim}",
            )

    # -- 后端调用 --------------------------------------------------------

    def _encode_local(self, texts: List[str]) -> List[Optional[List[float]]]:
        """调用本地后端编码（fastembed / sentence-transformers）。"""
        if self._backend is None:
            return []
        encoded = self._backend.encode(list(texts))
        rows = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        return [
            [float(x) for x in row] if row is not None else None
            for row in rows
        ]

    def _embed_via_api(self, texts: List[str]) -> List[Optional[List[float]]]:
        """调用 OpenAI 兼容的 /embeddings 接口。"""
        import httpx

        payload: dict = {"model": self._api_model, "input": texts if len(texts) > 1 else texts[0]}
        response = httpx.post(
            f"{self._api_base_url}/embeddings",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self.API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = (response.json() or {}).get("data")
        if not isinstance(data, list):
            raise ValueError("unexpected embeddings response: 'data' is not a list")

        ordered: List[Optional[List[float]]] = [None] * len(texts)
        for item in data:
            if not isinstance(item, dict):
                continue
            vector = item.get("embedding")
            if not vector:
                continue
            index = item.get("index")
            position = index if isinstance(index, int) and 0 <= index < len(texts) else None
            if position is None and len(texts) == 1:
                position = 0
            if position is not None:
                ordered[position] = [float(x) for x in vector]
        return ordered

    # -- 工具 ------------------------------------------------------------

    @staticmethod
    def _configured_dim(config: "GovernedMemoryConfig") -> int:
        """读取配置维度，异常时回退到 384。

        API 场景（``embedding.provider`` + ``base_url`` 已配置）下维度由
        ``embedding.dimensions`` 决定；本地模型场景由 ``vector.dim`` 决定。
        混用会把 ``vector.dim``（本地默认 512）当成 API 的配置维度，对维度
        正确的 API 后端误报 ``vector_dim_mismatch``。
        """
        try:
            embedding = getattr(config, "embedding", None)
            provider = str(getattr(embedding, "provider", "") or "")
            base_url = str(getattr(embedding, "base_url", "") or "")
            if provider and base_url:
                value = int(getattr(embedding, "dimensions", 0) or 0)
            else:
                value = int(getattr(config.vector, "dim", 0) or 0)
        except (TypeError, ValueError):
            return 384
        return value if value > 0 else 384

    @staticmethod
    def _build_signature(config: "GovernedMemoryConfig") -> tuple:
        """配置指纹：影响后端选择的字段变化时重新探测。

        API 密钥仅以 SHA-256 哈希形式包含在签名中，防止密钥明文通过日志、
        调试器或异常 traceback 泄露。哈希的前 16 字符足以区分不同的密钥，
        同时保持缓存失效的正确性。
        """
        vector = getattr(config, "vector", None)
        embedding = getattr(config, "embedding", None)
        api_key_env = str(getattr(embedding, "api_key_env", "") or "")
        api_key_hash = hashlib.sha256(
            env_secret(api_key_env).encode("utf-8")
        ).hexdigest()[:16] if api_key_env else ""
        return (
            str(getattr(vector, "backend", "auto") or "auto"),
            str(getattr(vector, "model", "") or ""),
            int(getattr(vector, "dim", 0) or 0),
            str(getattr(embedding, "provider", "") or ""),
            str(getattr(embedding, "base_url", "") or ""),
            str(getattr(embedding, "model", "") or ""),
            api_key_env,
            api_key_hash,
        )

    @staticmethod
    def _align(vectors: List[Optional[List[float]]], size: int) -> List[Optional[List[float]]]:
        """保证返回值与输入等长（契约要求）。"""
        if len(vectors) == size:
            return vectors
        if len(vectors) > size:
            return vectors[:size]
        return vectors + [None] * (size - len(vectors))
