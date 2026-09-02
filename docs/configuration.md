# Configuration

## Config File

Location: `$HERMES_HOME/governed_memory.json`

```json
{
  "scripts_dir": "~/.hermes/scripts",
  "wiki_dir": "~/wiki",
  "recall": {
    "l1_budget_tokens": 800,
    "l23_budget_tokens": 1200,
    "prefetch_ttl_seconds": 30,
    "l3_time_decay_hours": 168,
    "parallel_timeout_seconds": 2.0,
    "l3_max_results": 20,
    "l2_max_results": 15
  },
  "sync": {
    "l3_write": "sync",
    "l2_write": "async",
    "extract_async": true,
    "write_queue_maxsize": 100
  },
  "persona": {
    "incremental": true,
    "incremental_interval_minutes": 30,
    "full_refresh_cron": "0 9 * * *"
  },
  "tencent_extract": {
    "enabled": false,
    "confidence_threshold": 0.5,
    "max_candidates_per_turn": 5
  },
  "mermaid_compress": {
    "enabled": false,
    "canvas_max_tokens": 500
  },
  "vector": {
    "backend": "auto",
    "model": "BAAI/bge-small-zh-v1.5",
    "dim": 512
  },
  "embedding": {
    "provider": "",
    "base_url": "",
    "api_key_env": "",
    "model": ""
  }
}
```

## Recall Config

| Key | Default | Description |
|-----|---------|-------------|
| l1_budget_tokens | 800 | Fixed token budget for L1 hand-written rules. Never displaced. |
| l23_budget_tokens | 1200 | Shared token budget for L2 + L3 recall results. |
| prefetch_ttl_seconds | 30 | How long prefetch cache stays valid. |
| l3_time_decay_hours | 168 | Half-life for L3 time decay (7 days). |
| parallel_timeout_seconds | 2.0 | Max wait for parallel L2+L3+L4 recall. |
| l3_max_results | 20 | Max L3 search results before ranking. |
| l2_max_results | 15 | Max L2 search results before ranking. |

## Sync Config

| Key | Default | Description |
|-----|---------|-------------|
| l3_write | "sync" | L3 write mode. "sync" for safety, "async" for speed. |
| l2_write | "async" | L2 write mode. Always async recommended. |
| extract_async | true | Run fact extraction in background. |
| write_queue_maxsize | 100 | Max items in write queue before dropping. |

## Embedding Backend (L2 向量检索)

L2 语义检索需要一个 embedding 后端。系统按以下优先级选择，**只用第一个可用者**：

1. **API**（`embedding.provider + base_url + api_key_env` 三者齐全时启用）
2. **本地模型**（`vector.backend` + `vector.model`，默认 `BAAI/bge-small-zh-v1.5`）
3. **文本扫描**（无 embedding，L2 退化为关键词扫描）

### API（OpenAI 兼容 `/embeddings`）

```json
"embedding": {
  "provider": "siliconflow",
  "base_url": "https://api.siliconflow.cn/v1",
  "api_key_env": "SILICONFLOW_API_KEY",
  "model": "BAAI/bge-m3"
}
```

| Key | Description |
|-----|-------------|
| provider | 任意标识，用于日志（如 `siliconflow` / `openai`）|
| base_url | `/embeddings` 接口所在的服务根地址 |
| api_key_env | **环境变量名**（不是 key 本身）。key 请放到该环境变量，例如 `export SILICONFLOW_API_KEY=sk-...` |
| model | API 的 embedding 模型名 |
| dimensions | API 模型输出维度（如 bge-m3 = 1024）。探测会自动对齐；此值用于比对配置与实际维度，不一致时告警 `vector_dim_mismatch` |

> 维度**自动检测**：启动时用一次真实请求探测返回向量的长度（如 bge-m3 是 1024），
> 并自动对齐。`dimensions` 可省略（默认 1024）；显式填写为真实维度可避免误报
> `vector_dim_mismatch`。

### 本地模型

```json
"vector": {
  "backend": "auto",
  "model": "BAAI/bge-small-zh-v1.5",
  "dim": 512
}
```

| Key | Default | Description |
|-----|---------|-------------|
| backend | "auto" | `auto` / `fastembed` / `sentence_transformers` / `none`。`auto` = fastembed → sentence-transformers |
| model | "BAAI/bge-small-zh-v1.5" | 本地模型名（纯中文，512 维）。首次使用需下载约 100MB |
| dim | 512 | 模型输出维度，**必须与 model 匹配** |

### ⚠️ 维度一致性（切换后端必读）

**同一套 L2 表只能有一种向量维度**（LanceDB 的固定 schema）。切换 embedding 后端
（API ↔ 本地、或换模型）时，若维度变化，旧行向量与新向量维度不匹配，L2 检索会失效。

规则：

- 本地模型改 `model` 时，`dim` 必须同步改成该模型的输出维度，否则建表维度与向量
  维度错配、写入静默失败。
- 切换到不同维度（如 API 1024 ↔ 本地 512）后，**必须重建 L2**：

  ```bash
  python scripts/l2_rebuild.py
  ```

- 运行时探测到的实际维度与配置维度不一致时，系统会自动用实际维度覆盖配置并记录
  `vector_dim_mismatch` 告警。覆盖目标与后端一致：API 场景覆盖 `embedding.dimensions`，
  本地场景覆盖 `vector.dim`。建表始终用实际探测维度，因此 API 场景下 `vector.dim`
  （本地默认 512）不会被 API 的实际维度污染。

## Environment Variables

| Variable | Description |
|----------|-------------|
| GOVERNED_SCRIPTS_DIR | Override scripts_dir |
| GOVERNED_WIKI_DIR | Override wiki_dir |
| GOVERNED_L1_MEMORY_PATH | Override MEMORY.md path |
| GOVERNED_L1_USER_PATH | Override USER.md path |
| GOVERNED_L2_DB_PATH | Override LanceDB path |
| GOVERNED_L3_DB_PATH | Override SQLite path |
| GOVERNED_L4_PERSONA_PATH | Override persona.md path |
| GOVERNED_TENCENT_ENABLED | Enable Tencent extraction |
| LLM_API_KEY | LLM API key for extraction |
| LLM_BASE_URL | LLM base URL |
| LLM_MODEL | LLM model name |
