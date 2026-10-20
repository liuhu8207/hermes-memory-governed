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
  },
  "asr": {
    "provider": "siliconflow",
    "api_key_env": "SILICONFLOW_API_KEY",
    "base_url": "https://api.siliconflow.cn/v1",
    "model": "XingChenAGI/XingChenASR-V3.2-Ultra",
    "language": "",
    "api_style": "transcriptions",
    "timeout_seconds": 600,
    "chunk_minutes": 10,
    "max_encoded_bytes": 10000000
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

## ASR Config

Audio ingestion calls a speech-to-text endpoint; everything lives under `asr`. The
two API shapes are selected by `api_style`.

```json
"asr": {
  "provider": "siliconflow",
  "api_key_env": "SILICONFLOW_API_KEY",
  "base_url": "https://api.siliconflow.cn/v1",
  "model": "XingChenAGI/XingChenASR-V3.2-Ultra",
  "language": "",
  "api_style": "transcriptions",
  "timeout_seconds": 600,
  "chunk_minutes": 10,
  "max_encoded_bytes": 10000000
}
```

| Key | Default | Description |
|-----|---------|-------------|
| provider | "" | **Label only** — nothing dispatches on it. Adding a provider is a config change (`base_url` / `model` / `api_key_env` / `api_style`), never a code change. |
| api_key_env | "" | **Name** of the environment variable that holds the key (not the key itself). Lookup is name-first: a non-empty value in the **process environment** wins, and when that variable is missing or blank the same name is read from **`$HERMES_HOME/.env`** — so a key kept only in that file is usable, as long as the process can resolve `HERMES_HOME` (its env var, or the platform default). The value is never logged. |
| base_url | "" | Service root that holds the audio / chat endpoint. |
| model | "XingChenAGI/XingChenASR-V3.2-Ultra" | ASR model name. |
| language | "" | Empty = auto-detect. An explicit code such as `"zh"` can improve Chinese transcription. |
| api_style | "transcriptions" | `transcriptions` or `chat_audio` — see below. An unknown or empty value falls back to `transcriptions` with a warning; it never raises. |
| timeout_seconds | 600 | Per-request ceiling for a single transcription call. Applies to both shapes. |
| chunk_minutes | 10 | A recording longer than this is split into chunks; the chunk transcripts are joined in order. Applies to both shapes. |
| max_encoded_bytes | 10000000 | Cap on the base64 payload, `chat_audio` only. Going over returns an error that names the limit and tells you to lower `chunk_minutes` — it does **not** truncate silently and does **not** shrink the chunks for you. |

### API styles

| `api_style` | Endpoint | Transcript |
|-------------|----------|------------|
| `transcriptions` (default) | `POST {base_url}/audio/transcriptions` (multipart upload) | top-level `text` |
| `chat_audio` | `POST {base_url}/chat/completions`, audio inline as a base64 **data URL** | `choices[0].message.content` (there is no top-level `text`) |

`chat_audio` is for a chat-completions endpoint that is "OpenAI-SDK compatible" but
has **no** audio endpoint. For that shape `language` goes in a top-level
`asr_options` object.

Format and normalisation:

- `chat_audio` **always** transcodes the source to mp3 first, whatever the input
  format, so any ffmpeg-decodable input works there.
- For `transcriptions`, a container outside the accepted list
  (`.flac .m4a .mp3 .mp4 .ogg .wav .webm`) is transcoded to mp3 first when ffmpeg is
  available — **except `.silk`, which this ffmpeg build cannot decode and which is
  therefore unsupported**.
- **A partial result is not a success:** if some chunks fail you get `ok: false`
  plus the text that was obtained. Check `ok`.

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
| GOVERNED_L4_META_PATH | Override persona_meta.json path |
| GOVERNED_BRIDGE_DIR | Override Scope Recall bridge output dir |
| GOVERNED_TENCENT_ENABLED | Enable Tencent extraction |
| LLM_API_KEY | LLM API key for extraction |
| LLM_BASE_URL | LLM base URL |
| LLM_MODEL | LLM model name |

## Knowledge Base Config (`kb`)

The vault side of the system. Every key below is read from `_config.py`; the
defaults shown are the code defaults, and a deployment is free to override any
of them in `$HERMES_HOME/governed_memory.json`.

| Key | Default | Description |
|-----|---------|-------------|
| top_k | 10 | Max notes returned per KB query. |
| min_score | 0.0 | Legacy combined-score floor. Kept for compatibility; the two per-channel gates below are what actually filter. |
| semantic_enabled | true | Semantic (embedding) channel on. |
| keyword_enabled | true | Keyword channel on. |
| semantic_min_score | 0.0 | Cosine floor for the semantic channel. **Off by default.** Calibration on the real vault (bge-m3, 21 notes) put relevant hits at 0.41-0.84 and irrelevant ones at 0.31-0.49 — the bands overlap, so there is no usable cosine threshold. Kept for a deployment that wants a hard cosine cut. |
| rerank_min_score | 0.0 | Floor on the reranker score. **Code default is off (0.0)**; the shipped example config sets `0.05`, which is the calibrated deployment value. This is the real filter: with bge-m3 + bge-reranker-v2-m3 irrelevant queries scored <=0.027 and relevant ones 0.6-0.99. Set to 0.0 to disable. Separate from `semantic_min_score` because the two are on different scales — sharing one threshold once emptied every result below 0.4. |
| rerank_trace | false | Log candidate/kept/dropped counts and the floor for KB rerank. |
| confidence_threshold | 0.7 | Semi-automatic routing: confidence >= threshold goes to `notes/`, below it goes to `inbox/` with `review_required`. |
| autolink_related | true | Add reciprocal `[[wikilink]]`s between a new note and existing notes sharing concepts/tags. |
| recall_hint_enabled | true | Let the KB emit a one-line "related note exists" hint during recall. |
| recall_min_score | 0.0 | Fusion-score gate for the hint channel. **Off by default** (code default does not change behaviour). Calibration on the real vault: irrelevant top-1 fusion 0.4054-0.4666, relevant 0.4709-0.8934. |
| recall_min_kw_score | 0.0 | Keyword gate for the same channel. The hint fires when `keyword >= recall_min_kw_score` **OR** `fusion >= recall_min_score`. The keyword path needs its own gate because the fusion formula caps pure-keyword at 0.4, which can never reach a 0.45 fusion gate. |
| recall_max_notes | 3 | Max notes named in one hint line. |
| fusion | "linear" | `linear` (weighted raw scores) or `rrf` (rank fusion, k=60). RRF combines by rank because keyword and semantic scores are on different scales. Both modes keep the same normalised ceilings (keyword <=0.4, semantic <=0.6, top of both = 1.0), so the two gates above mean the same thing either way. An unknown value falls back to `linear` with a warning. |
| mmr_lambda | 0.0 | MMR redundancy penalty (0 = off, 0.7 suggested). Similarity is bigram Jaccard over title+snippet, so it only drops near-duplicates. |
| affinity_enabled | false | Usage-affinity ranking and hit counting. `false` = zero side effects. When on, ranking is multiplied by `1 + 0.15*log1p(hits)/log1p(8)` (capped at x1.15) — a slow nudge, not a feedback loop. |
| promote_hits | 3 | Inbox notes hit this many times get flagged `promote_suggest`. |
| auto_archive_days | 45 | Auto-zone notes with zero hits and no update for this long are suggested for `archive/`. The sweep **skips archiving when there is no usage data at all** (fail-closed: "no data" is not "unused"). |
| search_archived | false | Whether `status: "archived"` notes take part in retrieval. |
| enrich | false | On `add`, use the LLM to write a one-line `summary` plus 2-3 `questions` into frontmatter, feeding the keyword channel. One call at write time, failure never blocks the write. |

## Reranking Config

| Key | Default | Description |
|-----|---------|-------------|
| enabled | false | Master switch. **Off by default.** |
| provider | "" | Label for logs. |
| api_key_env | "" | **Name** of the environment variable holding the key, not the key. Name-first lookup: process environment wins, then the same name in `$HERMES_HOME/.env`. Never logged. |
| base_url | "" | Service root of the rerank endpoint. |
| model | "" | Reranker model name, e.g. `BAAI/bge-reranker-v2-m3`. |
| top_n | 10 | Candidates kept after rerank. |

## Recall Config Additions

`recall.l2_min_score` and `recall.l2_fusion` are documented in
[`_config.py`](plugin/memory_governed/_config.py) but were missing here:

| Key | Default | Description |
|-----|---------|-------------|
| l2_min_score | 0.0 | Floor on L2 native score. **Off by default** — the code default does not change behaviour; a deployment turns it on explicitly. Measured L2 relevant and irrelevant bands overlap, which is why it ships off. |
| l2_fusion | false | Cross-layer L2/L3 rank fusion. |
| cross_layer_fusion | false | Same family as `l2_fusion`; see the fusion notes in the architecture doc. |

## Synthesis Config

| Key | Default | Description |
|-----|---------|-------------|
| enabled | false | Session synthesis (conversation to knowledge cards). Off by default. |
| provider | "" | Empty = inherit `model.provider` from `config.yaml`. |
| base_url | "" | Empty = inherit from `config.yaml`. |
| api_key_env | "" | Empty = derived from provider as `{PROVIDER}_API_KEY`. |
| model | "" | Empty = inherit the agent's own chat model. |
| max_candidates | 5 | Cards synthesised per run. |
| require_durable_signal | true | Skip the run entirely when the session holds no durable signal (preference / decision / config intent). Saves an LLM call on reasoning models. |

## Sync Config Additions

| Key | Default | Description |
|-----|---------|-------------|
| l2_max_facts_per_turn | 15 | Facts extracted per turn before the rest is dropped. |
| l2_max_items | 0 | Hard cap on L2 rows (0 = unlimited). |

## Persona / Mermaid / Tencent Extraction

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `persona` | incremental | true | Incremental persona refresh. |
| | full_refresh_cron | "0 9 * * *" | Full refresh schedule. |
| | incremental_interval_minutes | 30 | Minimum gap between incremental refreshes. |
| `mermaid_compress` | enabled | false | Compress Mermaid canvas content. |
| | canvas_max_tokens | 500 | Token ceiling for a compressed canvas. |
| `tencent_extract` | enabled | false | Tencent extraction path. |
| | confidence_threshold | 0.5 | Minimum confidence to keep an extracted fact. |
| | max_candidates_per_turn | 5 | Candidates considered per turn. |
| | llm_api_key / llm_base_url / llm_model | "" | Endpoint used by this path. |
