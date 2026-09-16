"""Configuration loader for the governed memory provider.

Reads config from:
1. $HERMES_HOME/governed_memory.json (primary)
2. Environment variables (fallback)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RecallConfig:
    """Read-path tuning."""
    l1_budget_tokens: int = 800       # L1 fixed budget, never displaced
    l23_budget_tokens: int = 1200     # L2 + L3 shared recall budget
    prefetch_ttl_seconds: float = 30.0
    l3_time_decay_hours: float = 168.0  # 7-day half-life
    l3_max_results: int = 20
    l2_max_results: int = 15
    parallel_timeout_seconds: float = 2.0
    #: L2（向量语义层）**专用**的最低相关分阈值；``0.0`` = 关闭（沿用
    #: :data:`_recall.MIN_SCORE`，保持历史行为不变）。
    #:
    #: 为什么必须和 L3 分开：``MIN_SCORE`` 同时过滤 L2 与 L3，而两层分数的
    #: 分布完全不同 —— L3 是 FTS5 关键词命中 + 时间衰减，实测有效命中的分数
    #: 集中在 0.3~0.7（例如 SSH 相关对话 0.655）；L2 是 1024 维余弦映射，
    #: 无关查询的地板分就有 0.73。直接把 ``MIN_SCORE`` 提到 0.76 会砍掉
    #: L3 的 82/91 条有效结果，属于典型的"用错尺子量错层"。
    #:
    #: 标定依据（2026-09-16，L2=29 条）：无关查询 top1 0.731~0.741，
    #: 相关查询 top1 0.782~0.912 → 0.76 是两者之间唯一的干净切点。
    l2_min_score: float = 0.0


@dataclass
class SyncConfig:
    """Write-path tuning.

    Note: write-path limits must never be derived from ``RecallConfig`` — the
    read path caps what is *shown* to the model, the write path caps what is
    *stored*.  Conflating them silently truncated fact extraction to the
    recall page size (15) and threw away durable knowledge.
    """
    l3_write: str = "sync"            # "sync" or "async"
    l2_write: str = "async"           # "async" or "sync"
    extract_async: bool = True
    write_queue_maxsize: int = 100
    l2_max_facts_per_turn: int = 15   # write-path cap; independent of recall.l2_max_results
    # 2026-09-16: 50 → 15。`_index_l2` 每次收到的是**整段会话历史**，靠去重兜底，
    # 上限偏大时一次 turn 会灌进几十条（实测 09:58:17 一秒写入 36 条）。


@dataclass
class PersonaConfig:
    """L4 persona refresh."""
    incremental: bool = True
    full_refresh_cron: str = "0 9 * * *"
    incremental_interval_minutes: int = 30


@dataclass
class TextractConfig:
    """Tencent-style auto-extraction (optional)."""
    enabled: bool = False
    confidence_threshold: float = 0.5
    max_candidates_per_turn: int = 5
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = ""


@dataclass
class EmbeddingConfig:
    """Embedding provider config for L2 vector search."""
    provider: str = ""            # "siliconflow", "openai", "local"
    api_key_env: str = ""         # env var name for API key
    base_url: str = ""
    model: str = ""
    dimensions: int = 1024


@dataclass
class RerankingConfig:
    """Reranking provider config for result reordering."""
    provider: str = ""            # "siliconflow", "cohere", "jina"
    api_key_env: str = ""         # env var name for API key
    base_url: str = ""
    model: str = ""
    top_n: int = 10


@dataclass
class MermaidConfig:
    """Mermaid short-term compression (optional)."""
    enabled: bool = False
    canvas_max_tokens: int = 500


@dataclass
class VectorConfig:
    """L2 embedding settings (swap model = update config + rebuild L2)."""
    backend: str = "auto"  # "auto" | "fastembed" | "sentence_transformers" | "none"
    # 默认用纯中文模型：本系统以中文记忆为主，英文 all-MiniLM-L6-v2 对中文
    # 语义区分度差（"数据库"查询会把"周末散步"也打出相近高分）。
    # 模型名必须带 org 前缀（fastembed 对裸名会静默回退到它的默认模型）。
    # 注意：model 与 dim 必须配对 —— _sync._init_l2 建表用 config.vector.dim，
    # 换模型（维度改变）必须同步改 dim 并用 scripts/l2_rebuild.py 重建 L2。
    model: str = "BAAI/bge-small-zh-v1.5"
    dim: int = 512


@dataclass
class KnowledgeConfig:
    """Knowledge base (Obsidian vault) settings."""
    top_k: int = 10
    min_score: float = 0.0
    semantic_enabled: bool = True
    keyword_enabled: bool = True
    #: 半自动沉淀的置信分流阈值：confidence >= 阈值 → notes（正库），
    #: 低于阈值 → inbox（待人工确认）。confidence 未传时按显式 section 落库。
    confidence_threshold: float = 0.7
    #: 落库时是否为共享 concepts/tags 的相关笔记自动补双向 [[wikilink]]。
    autolink_related: bool = True


@dataclass
class AsrConfig:
    """Speech-to-text (ASR) config for audio ingestion.

    OpenAI 兼容的 ``/audio/transcriptions`` 端点（默认硅基流动
    XingChenAGI/XingChenASR-V3.2-Ultra，多语言含中文）。
    """
    provider: str = ""           # "siliconflow"（任意标识）
    api_key_env: str = ""        # 环境变量名（复用 SILICONFLOW_API_KEY）
    base_url: str = ""           # 如 https://api.siliconflow.cn/v1
    model: str = "XingChenAGI/XingChenASR-V3.2-Ultra"
    language: str = ""           # 空 = auto；显式 "zh" 可提升中文转写


@dataclass
class SynthesisConfig:
    """Session 归纳（对话 → 知识卡片）配置。

    归纳用 LLM 与 agent 对话模型**同款**：provider/base_url/model 留空时，
    运行时自动继承 ``config.yaml`` 的 model 段（任何 hermes 环境都自动对齐
    agent 的对话模型，无需手填）。插件侧自己调 LLM，不依赖「正在对话的
    agent」——模型一致即可。
    """
    enabled: bool = False         # 默认关：未显式配置时不自动归纳
    provider: str = ""            # 空 = 自动继承 agent 的 model.provider
    base_url: str = ""            # 空 = 自动继承 agent 的 model.base_url
    api_key_env: str = ""         # 空 = 由 provider 推导 {PROVIDER}_API_KEY
    model: str = ""               # 空 = 自动继承 agent 的 model.default
    max_candidates: int = 5       # 单次归纳最多落库几张卡
    #: 归纳前零成本过滤：session 里无 durable 信号（偏好/决策/配置意图）则
    #: 跳过，省一次 LLM 调用（reasoning 模型归纳一次 ~5-6k token）。
    require_durable_signal: bool = True


@dataclass
class GovernedMemoryConfig:
    """Top-level config."""
    scripts_dir: str = ""
    wiki_dir: str = ""
    l1_memory_path: str = ""    # MEMORY.md
    l1_user_path: str = ""      # USER.md
    l2_db_path: str = ""        # LanceDB directory
    l3_db_path: str = ""        # SQLite file
    l4_persona_path: str = ""   # persona.md
    l4_meta_path: str = ""      # persona_meta.json
    bridge_dir: str = ""        # Scope Recall bridge output
    # Escape hatch for the Bridge fail-closed secret policy. Default OFF:
    # secret-like candidates go to bridge/quarantine.jsonl and NEVER into
    # candidates.jsonl. When turned on they are stored REDACTED and flagged —
    # the L1 hard gate in import_approved() blocks them either way.
    allow_secret_candidates: bool = False
    recall: RecallConfig = field(default_factory=RecallConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    tencent_extract: TextractConfig = field(default_factory=TextractConfig)
    mermaid_compress: MermaidConfig = field(default_factory=MermaidConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    reranking: RerankingConfig = field(default_factory=RerankingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    kb: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)


def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_governed_config(hermes_home: str | Path) -> GovernedMemoryConfig:
    """Load config from governed_memory.json with env var fallbacks."""
    hermes_home = Path(hermes_home)
    config = GovernedMemoryConfig()

    # Set defaults based on HERMES_HOME
    config.l1_memory_path = str(hermes_home / "memory" / "MEMORY.md")
    config.l1_user_path = str(hermes_home / "memory" / "USER.md")
    config.l2_db_path = str(hermes_home / "memory" / "l2")
    config.l3_db_path = str(hermes_home / "memory" / "l3" / "l3.db")
    config.l4_persona_path = str(hermes_home / "memory" / "persona.md")
    config.l4_meta_path = str(hermes_home / "memory" / "persona_meta.json")
    config.bridge_dir = str(hermes_home / "cron" / "output" / "scope_recall_bridge")
    config.scripts_dir = str(hermes_home / "scripts")
    config.wiki_dir = str(hermes_home / "wiki")

    # Load from governed_memory.json if it exists
    config_path = hermes_home / "governed_memory.json"
    if config_path.exists():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            _apply_dict_to_config(config, raw)
        except Exception as e:
            logger.warning("Failed to load config from %s: %s — using defaults", config_path, e)

    _validate_numeric_fields(config)

    # Env var overrides (backward compat)
    _apply_env_overrides(config)

    # synthesis 未显式配置模型时，自动继承 agent 对话模型（config.yaml model 段）
    _inherit_synthesis_model(config, hermes_home)

    return config


def _load_agent_model_config(hermes_home: Path) -> Optional[Dict[str, str]]:
    """从 hermes ``config.yaml`` 读 ``model`` 段（base_url/default/provider）。

    简单逐行解析（model 段是几个标量 ``key: value``），不引入 yaml 依赖。
    只匹配**顶层**的 ``model:``（tts 等子段的 ``model:`` 有缩进，不会误匹配）。
    """
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        return None
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    result: Dict[str, str] = {}
    in_model = False
    for line in text.splitlines():
        if line.startswith("model:"):
            in_model = True
            continue
        if not in_model:
            continue
        m = re.match(r"^(\s+)(base_url|default|provider)\s*:\s*(.+?)\s*$", line)
        if m:
            result[m.group(2)] = m.group(3).strip().strip('"').strip("'")
            continue
        # 遇到非缩进的顶层 key（离开 model 段）
        if line.strip() and not line[0].isspace():
            break

    if not result.get("base_url") and not result.get("default"):
        return None
    return result


def _inherit_synthesis_model(config: GovernedMemoryConfig, hermes_home: Path) -> None:
    """synthesis 未显式配置模型时，自动继承 agent 对话模型。

    归纳永远和 agent 用同一个模型（对齐 config.yaml 的 model 段），用户无需
    手填 provider/base_url/model；api_key_env 由 provider 推导为
    ``{PROVIDER}_API_KEY``。显式配置优先，不会被覆盖。
    """
    syn = config.synthesis
    if syn.base_url and syn.model:
        return  # 已显式配置（json/env），不覆盖

    agent = _load_agent_model_config(hermes_home)
    if not agent:
        return  # 读不到 config.yaml，保持空（synthesize 会因缺配置 skip）

    if not syn.model:
        syn.model = agent.get("default", "")
    if not syn.base_url:
        syn.base_url = agent.get("base_url", "")
    if not syn.provider:
        syn.provider = agent.get("provider", "")
    if not syn.api_key_env:
        provider = agent.get("provider", "")
        if provider:
            syn.api_key_env = provider.upper() + "_API_KEY"


_NUMERIC_FIELDS = {
    "recall": ["l1_budget_tokens", "l23_budget_tokens", "prefetch_ttl_seconds",
               "l3_time_decay_hours", "l3_max_results", "l2_max_results",
               "parallel_timeout_seconds", "l2_min_score"],
    "sync": ["write_queue_maxsize", "l2_max_facts_per_turn"],
    "mermaid_compress": ["canvas_max_tokens"],
    "vector": ["dim"],
    "embedding": ["dimensions"],
    "kb": ["top_k", "min_score"],
}


def _validate_numeric_fields(config: GovernedMemoryConfig) -> None:
    """Warn and coerce wrong-typed numeric fields (e.g. "800" instead of 800)."""
    for section_name, fields in _NUMERIC_FIELDS.items():
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for field_name in fields:
            value = getattr(section, field_name, None)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                try:
                    coerced = type(0.0 if isinstance(value, str) and "." in str(value) else 0)(value)
                    logger.warning(
                        "Config %s.%s should be a number, got %r — coerced to %s",
                        section_name, field_name, value, coerced,
                    )
                    setattr(section, field_name, coerced)
                except (TypeError, ValueError):
                    default_section = GovernedMemoryConfig()
                    default_value = getattr(getattr(default_section, section_name), field_name)
                    logger.warning(
                        "Config %s.%s has invalid value %r — reset to default %s",
                        section_name, field_name, value, default_value,
                    )
                    setattr(section, field_name, default_value)


def _apply_dict_to_config(config: GovernedMemoryConfig, raw: dict) -> None:
    """Apply a raw dict to config dataclass fields."""
    for key, value in raw.items():
        if hasattr(config, key):
            attr = getattr(config, key)
            if isinstance(value, dict) and hasattr(attr, "__dataclass_fields__"):
                for k2, v2 in value.items():
                    if hasattr(attr, k2):
                        setattr(attr, k2, v2)
            else:
                setattr(config, key, value)


def _apply_env_overrides(config: GovernedMemoryConfig) -> None:
    """Apply environment variable overrides."""
    env_map = {
        "GOVERNED_SCRIPTS_DIR": ("scripts_dir", str),
        "GOVERNED_WIKI_DIR": ("wiki_dir", str),
        "GOVERNED_L1_MEMORY_PATH": ("l1_memory_path", str),
        "GOVERNED_L1_USER_PATH": ("l1_user_path", str),
        "GOVERNED_L2_DB_PATH": ("l2_db_path", str),
        "GOVERNED_L3_DB_PATH": ("l3_db_path", str),
        "GOVERNED_L4_PERSONA_PATH": ("l4_persona_path", str),
        "GOVERNED_L4_META_PATH": ("l4_meta_path", str),
        "GOVERNED_BRIDGE_DIR": ("bridge_dir", str),
    }
    for env_var, (attr, _) in env_map.items():
        val = os.environ.get(env_var)
        if val:
            setattr(config, attr, val)

    # Tencent extract env vars
    if os.environ.get("GOVERNED_TENCENT_ENABLED"):
        config.tencent_extract.enabled = True
    if os.environ.get("LLM_API_KEY"):
        config.tencent_extract.llm_api_key = os.environ["LLM_API_KEY"]
    if os.environ.get("LLM_BASE_URL"):
        config.tencent_extract.llm_base_url = os.environ["LLM_BASE_URL"]
    if os.environ.get("LLM_MODEL"):
        config.tencent_extract.llm_model = os.environ["LLM_MODEL"]
