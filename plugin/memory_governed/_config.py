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
from typing import Dict, Optional

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
    #: L2 检索的**项目作用域**，三态：
    #:
    #: * ``None``（不设，默认）—— 不过滤，看得到所有项目的事实（历史行为不变）
    #: * ``""`` —— 只看全局事实（``project IS NULL``）
    #: * ``"名字"`` —— 该项目 **+** 全局事实，看不到别的项目
    #:
    #: 为什么需要它（P0，2026-09-17）：``project`` 维度此前只装在 CLI 的一条读
    #: 路径（``memory_cli.py`` 的 ``search_l2`` / ``recall_l2``）上，而 Hermes
    #: 插件走 ``_recall.py`` —— 同一份存储、两个入口、两套答案：插件会把所有
    #: 项目的事实一锅端注入上下文。
    #:
    #: 行级过滤语义与 CLI 完全一致（``project and pj and pj != project`` →
    #: 保留本项目 + 全局）。唯一的差异：CLI 的 ``project=""`` 表示「不缩小」，
    #: 读路径上无法表达「只看全局」；这里把 ``""`` 定为「只看全局」，因为它
    #: 是三态里唯一能表达该意图的取值。
    #:
    #: **注意仍存在的缺口**：插件的写入路径（本地事实抽取）按设计不给事实
    #: 打 ``project``（见 ``_sync.L2_PROVENANCE_COLUMNS`` 的说明），所以插件
    #: 自己抽出来的事实全是全局的 —— 作用域能挡住别的项目经 CLI 写入的事实，
    #: 但挡不住别的项目会话里抽出的事实。
    project_scope: Optional[str] = None
    #: L2 是否启用**双路 RRF 融合**（向量 + 词法都跑、按排名融合）。
    #:
    #: ``False``（默认）= 历史行为：向量可用则只跑向量，向量不可用才退词法
    #: （二选一链路）。``True`` 时两路都跑：精确子串命中的事实不再因为向量
    #: 通道可用而被整体跳过，双通道确认的事实按 RRF 排名升到前面。
    #:
    #: 分数语义：融合后 ``score`` 是**排名量纲**（RRF 归一化到 (0,1]），
    #: 每条命中另存 ``metadata["native_score"]``（向量=余弦分 / 词法=子串分）；
    #: ``recall.l2_min_score`` 门槛对原生分生效（余弦标尺不量排名）——见
    #: :func:`_recall._fuse_l2_channels` 与 ``format_recall``。
    #:
    #: **默认 False = 现行为**：与本文件其它新行为同一约定，部署侧显式开启。
    l2_fusion: bool = False


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
    #: 每个 agent 的 L2 容量上限。**0 = 关闭（默认 = 现行为：永不淘汰）**。
    #:
    #: 开启（建议 200）后每次 ``remember`` 写入前检查：将超限时把**最少
    #: 使用**（last_used 升序，平手按写入时间）的同 agent 事实降级归档到
    #: ``l2_archive.jsonl``（可恢复，不写墓碑 —— 被挤掉的可以再写回来）。
    #: 容量是唯一的自动遗忘，且遗忘只降级不删除（WeKnora demote-not-delete）。
    #: 开启同时启用召回命中的 ``l2_usage`` 计数 —— 淘汰的受害者选择需要它。
    l2_max_items: int = 0
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

    #: 是否允许知识库参与对话召回（产出一行「有相关笔记」提示）。
    recall_hint_enabled: bool = True
    #: KB 提示通道的融合分门槛。**默认 0.0 = 关闭**，显式配置后才生效
    #: （与 ``recall.l2_min_score`` 同一约定：代码默认不改行为，部署侧开启）。
    #:
    #: 标定参考（vault 26 篇 / 真实 bge-m3）：无关查询 top1 融合分
    #: 0.4054~0.4666，相关 0.4709~0.8934。取 0.45 时相关 5/5 命中、无关
    #: 漏过 1/5 —— 提示行误报成本仅 ~40 token，因此取偏宽松一侧换覆盖率。
    #: 想要更干净可提到 0.50（相关 4/5、无关 0/5）。
    recall_min_score: float = 0.0
    #: 走廊同上：KB 提示通道的**关键词**准入门槛。
    #: 召回判定是两条走廊的**或**：``keyword >= recall_min_kw_score`` OR
    #: ``融合分 >= recall_min_score``。两者同为 0.0 时通道关闭。
    #:
    #: 为什么必须有它（P0，2026-09-17）：融合分是 ``0.4*kw + 0.6*sem``，
    #: 纯关键词命中的理论上限就是 0.4 —— 无论关键词多准，融合分都够不到
    #: 部署值 0.45，关键词这一条路被结构性地关死。两个走廊必须分开设门槛。
    #:
    #: **默认 0.0 = 关闭**：与 ``recall_min_score`` 同一约定 —— 代码默认不改
    #: 行为，部署侧显式开启。建议部署值 **1.0**（= ``kw_norm`` 满格，语义是
    #: 「查询串完整出现在标题/tags/concepts 里」这一最强粒度）：在此粒度下
    #: 实测相关命中 10/10、无关噪音与开启前完全一致（详见 reports）。
    recall_min_kw_score: float = 0.0
    #: 一次提示最多列出几篇笔记。
    recall_max_notes: int = 3

    #: 融合算法：``"linear"``（默认，历史行为）或 ``"rrf"``（加权 RRF）。
    #:
    #: RRF 按**排名**融合关键词/语义两路（``w/(k+rank)``，k=60），而不是按
    #: 原始分加权 —— 两路量纲不同（关键词归一化分 vs 余弦分），线性加权会被
    #: 单路极值主导。归一化后分数天花板与线性模式一致（纯关键词 ≤ 0.4 /
    #: 纯语义 ≤ 0.6 / 双路第一名 = 1.0），所以 ``recall_min_score`` /
    #: ``recall_min_kw_score`` 两条准入走廊在两种模式下语义不变。
    #: 未知值一律按 ``"linear"`` 处理（告警不抛错，读路径绝不因脏配置挂掉）。
    fusion: str = "linear"
    #: MMR 去冗余的 λ（0 = 关闭，保持历史排序；建议 0.7）。
    #:
    #: 选 top_k 时用 ``λ·融合分 − (1−λ)·与已选项的最大相似度``，相似度取
    #: title+snippet 的 bigram Jaccard（零依赖）。相似度为 0 的条目完全按
    #: 融合分排序 —— MMR 只挤掉近似重复项，不搅动本来就不相似的结果。
    mmr_lambda: float = 0.0
    #: 是否启用**使用度亲和度**（含命中计数落表）。
    #:
    #: ``False``（默认）= 零副作用：不读不写使用度表，排序与历史一致。
    #: ``True`` 时排序乘 ``1 + 0.15·log1p(hits)/log1p(8)``（封顶 ×1.15）：
    #: 缓慢复利的微推，不是正反馈回路 —— 被反复召回的笔记略微上浮，
    #: 但永远不会因为「曾经热过」而霸榜。命中计数由召回提示通道写入
    #: （见 ``KnowledgeBase.touch``），也是晋升/归档（Phase 2）的数据源。
    affinity_enabled: bool = False

    #: 晋升门槛：inbox 笔记被召回命中 ≥ N 次后标 ``promote_suggest``，
    #: 由 ``governed_kb_review approve`` 确认晋升进 ``notes/``。
    #: 阈值才晋升（对应 WeKnora interest_threshold=3）—— 命中是慢信号，
    #: 第一次被用上就晋升会把 inbox 变成第二个正库。
    promote_hits: int = 3
    #: 自动归档天数：自动区（inbox/knowledge）笔记命中数为 0 且超过 N 天
    #: 未更新 → 治理巡检建议移入 ``archive/``（降级不删除）。
    #: **无使用度数据时巡检跳过归档**（fail-closed：「没数据」≠「没被用过」）。
    auto_archive_days: int = 45
    #: 入库富化（Phase 4）：``add`` 时用 LLM 生成一行 ``summary`` + 2~3 个
    #: ``questions`` 存进 frontmatter，喂给关键词召回路 —— 纯向量对精确
    #: 术语的漏召回由这些显式词补上（WeKnora 的 summary / generated
    #: questions）。**默认 False = 现行为**；LLM 调用走 ``config.synthesis``
    #: 端点，写入时一次（非查询时），失败不阻断写入。
    enrich: bool = False


@dataclass
class AsrConfig:
    """Speech-to-text (ASR) config for audio ingestion.

    支持两种接口风格（``api_style``）：

    * ``"transcriptions"``（默认）：OpenAI 兼容的 ``/audio/transcriptions``（默认
      硅基流动 XingChenAGI/XingChenASR-V3.2-Ultra，多语言含中文）。
    * ``"chat_audio"``：非 audio 兼容的 chat-completions 端点，音频以 base64 放进
      ``input_audio`` 内容块（如小米 MiMo ``mimo-v2.5-asr``，其
      ``/audio/transcriptions`` 返回 404）。见 ``_ingest._transcribe_chat_audio``。
    """
    provider: str = ""           # "siliconflow"（任意标识）
    api_key_env: str = ""        # 环境变量名（复用 SILICONFLOW_API_KEY）
    base_url: str = ""           # 如 https://api.siliconflow.cn/v1
    model: str = "XingChenAGI/XingChenASR-V3.2-Ultra"
    language: str = ""           # 空 = auto；显式 "zh" 可提升中文转写
    #: 接口风格：``"transcriptions"`` | ``"chat_audio"``。未知/空值会退回
    #: ``"transcriptions"``（告警，绝不抛异常，见 ``_validate_api_style``）。
    api_style: str = "transcriptions"
    #: 单次转写请求的超时上限（秒）。实测该 ASR 端点约 6.7~8.3 秒/音频分钟，
    #: 旧的硬编码 180s 让约 25 分钟以上的录音必然超时；拆段（见
    #: ``_ingest.split_audio``）后每段都远低于此值。
    timeout_seconds: float = 600.0
    #: 超过这个分钟数就自动拆段转写。0 或负数属于脏配置 —— 会退回默认值，
    #: 绝不因此抛异常（见 ``_validate_positive_asr_fields``）。
    chunk_minutes: float = 10.0
    #: ``chat_audio`` 风格下 base64 载荷的字节上限（MiMo 的上限是 10 MB）。
    #: 超限时显式报错并要求调低 ``chunk_minutes``，绝不静默截断。正整数校验，
    #: 脏值退回默认、绝不抛异常。
    max_encoded_bytes: int = 10_000_000


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


# ---------------------------------------------------------------------------
# 密钥读取：进程环境变量优先，回退到 ``$HERMES_HOME/.env``
# ---------------------------------------------------------------------------

def _hermes_home_path() -> Path:
    """Resolve ``$HERMES_HOME`` the way the scripts do (env, then platform default).

    插件拿到的是 ``hermes_home`` **参数**，但密钥读取发生在只持有 config 对象的
    模块里，所以这里按环境变量解析路径 —— 与 ``scripts/hermes_env.py::bootstrap``
    和 ``scripts/hgm_mcp.py`` 用的是同一条规则，保证所有入口读的是同一个 ``.env``。
    """
    home = os.environ.get("HERMES_HOME")
    if home:
        return Path(home)
    return Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "hermes"


def _read_dotenv_values(path: Path) -> Dict[str, str]:
    """Tolerant ``KEY=VALUE`` parse, mirroring ``scripts/hermes_env.load_dotenv``.

    **只读**：与 bootstrap 的加载器不同，它绝不写 ``os.environ``、绝不记录任何值。
    文件缺失或不可读 → 空映射（没有 ``.env`` 的存储不是错误）。空行、``#`` 注释、
    不含 ``=`` 的行都跳过；值两侧的单/双引号会被剥掉。
    """
    values: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = value.strip().strip("\"'")
    return values


def env_secret(name: str) -> str:
    """Look up a secret by env var name: process environment first, ``.env`` fallback.

    优先级是刻意的：进程环境里导出的**非空**值永远胜出，运维可以在单次调用里覆盖
    磁盘上的 ``.env``；只有在环境里**不存在或为空**时才去读文件。名字为空、或两处
    都没有时返回 ``""`` —— 调用方沿用原有的「env not set」文案，值本身绝不进入日志
    或错误。

    **为什么「存在但为空」必须回退**（不要简化回 ``if name in os.environ``）：宿主
    常把密钥当作插值表达式传下来（如 dsh 的 ``cordis.patch.yml`` 里
    ``!!js process.env.SILICONFLOW_API_KEY``）；当父值缺失时它往往求值成**空字符串**
    而不是把变量整个省略。若空值也算「存在」，它就会盖住磁盘上的真值 —— 正是本函数
    要消除的那个故障。

    有人会设想用 ``KEY=""`` 来**故意关闭**某个后端：本系统关闭后端的既定旋钮是
    **按名字**（令 ``api_key_env`` 指向一个不存在的变量），而「空串=关闭」的约定与
    「宿主配错」在现象上无法区分 —— 所以空白即回退是更安全的读法。

    刻意不做进程内缓存：文件很小、这些读取不是热路径，而按粗糙 mtime 缓存在密钥
    轮换后可能返回陈旧值（正确性优先于微优化）。
    """
    name = str(name or "").strip()
    if not name:
        return ""
    value = os.environ.get(name)
    if value is not None and value.strip():
        return value
    return _read_dotenv_values(_hermes_home_path() / ".env").get(name, "")


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
    _validate_positive_fields(config)
    _validate_api_style(config)

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
    "kb": ["top_k", "min_score", "recall_min_score", "recall_min_kw_score",
           "recall_max_notes", "mmr_lambda"],
    "asr": ["timeout_seconds", "chunk_minutes", "max_encoded_bytes"],
}

#: 必须为正的数值字段（超时 / 拆段阈值 / base64 上限）。0 或负数不是「关闭功能」
#: 而是脏配置：0 会让拆段永不触发（回到会超时的老路），或让单次请求没有上限，
#: 或让 base64 上限为 0 从而永远拒绝任何请求。按本文件一贯做法 —— 告警并回落
#: 默认值，绝不抛异常。
_POSITIVE_FIELDS = {
    "asr": ["timeout_seconds", "chunk_minutes", "max_encoded_bytes"],
}

#: ``asr.api_style`` 的合法取值。未知/空值退回 ``"transcriptions"``。
_ASR_API_STYLES = ("transcriptions", "chat_audio")


def _validate_api_style(config: GovernedMemoryConfig) -> None:
    """Reset an unknown/empty ``asr.api_style`` to ``"transcriptions"`` (never raises)."""
    asr = getattr(config, "asr", None)
    if asr is None:
        return
    style = getattr(asr, "api_style", None)
    if style in _ASR_API_STYLES:
        return
    logger.warning(
        "Config asr.api_style = %r is not one of %s — reset to 'transcriptions'",
        style, ", ".join(_ASR_API_STYLES),
    )
    asr.api_style = "transcriptions"


def _validate_positive_fields(config: GovernedMemoryConfig) -> None:
    """Coerce non-positive numeric fields back to their defaults (never raises)."""
    defaults = GovernedMemoryConfig()
    for section_name, fields in _POSITIVE_FIELDS.items():
        section = getattr(config, section_name, None)
        if section is None:
            continue
        for field_name in fields:
            value = getattr(section, field_name, None)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue  # 类型问题交给 _validate_numeric_fields
            if value > 0:
                continue
            default_value = getattr(getattr(defaults, section_name), field_name)
            logger.warning(
                "Config %s.%s must be > 0, got %r — reset to default %s",
                section_name, field_name, value, default_value,
            )
            setattr(section, field_name, default_value)


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
