# Hermes Memory Governed

给 [Hermes Agent](https://github.com/hermes-agent) 用的**记忆治理插件**。它把四层记忆栈（L1 手写规则、L2 语义召回、L3 会话归档、L4 画像）、一套以 Obsidian 为载体的知识库（带自动蒸馏 + 审核门），以及配套的运维工具，整合成一个高性能 `MemoryProvider`。

> English: [README.md](README.md)

[![CI](https://github.com/liuhu8207/hermes-memory-governed/actions/workflows/ci.yml/badge.svg)](https://github.com/liuhu8207/hermes-memory-governed/actions)

---

## 目录

- [它能做什么](#它能做什么)
- [设计目标](#设计目标)
- [架构](#架构)
  - [四层记忆栈](#四层记忆栈)
  - [读路径 — 异步预取](#读路径--异步预取)
  - [写路径 — 非阻塞流水线](#写路径--非阻塞流水线)
  - [Token 预算](#token-预算)
- [知识库与蒸馏](#知识库与蒸馏)
  - [三条链路](#三条链路)
  - [审核门](#审核门)
- [工具清单](#工具清单)
- [模块地图](#模块地图)
- [安装](#安装)
- [配置](#配置)
- [运维](#运维)
- [测试与 CI](#测试与-ci)
- [相关文档](#相关文档)

---

## 它能做什么

这个插件给 Hermes 提供一套**受治理（governed）**的记忆系统——所谓「治理」，是指：任何写入都不静默发生，任何内容都不会挤掉人工写下的规则。具体来说：

1. **人工规则永远不丢**——L1（`MEMORY.md` / `USER.md`）始终注入，拥有固定 token 预算，L2/L3 的召回结果永远无法挤掉它。
2. **语义化召回**——L2 对持久事实做向量检索；L3 对完整会话归档做全文检索；L4 把稳定画像放进 system prompt。
3. **绝不卡对话**——读路径异步预取，写路径非阻塞；L3（事实来源）同步写且极快，其余全部入队后台处理。
4. **沉淀知识而不只是聊天记录**——对话会被自动蒸馏成 Obsidian 笔记，经置信门 + 审核门把关，持久知识以你自己拥有的纯 Markdown 形式留存。

## 设计目标

| # | 目标 | 机制 |
|---|------|------|
| 1 | **读得快** | `queue_prefetch()` 后台线程预取；`prefetch()` 命中缓存返回 **<1ms** |
| 2 | **内容全** | L1 固定注入；L2/L3/L4 并行召回；L4 画像进 system prompt |
| 3 | **不卡顿** | `sync_turn()` 非阻塞——L3 同步写 **<10ms**，L2 索引与抽取异步 |
| 4 | **不静默写入** | 事实经置信门、密钥门、审核门流转；失败一定上报，绝不吞掉 |

## 架构

### 四层记忆栈

```
┌────────────────────────────────────────────────────────────┐
│                      Hermes Agent                          │
│               （对话循环 + 工具分发）                         │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │        GovernedMemoryProvider (MemoryProvider ABC)  │   │
│   │                                                    │   │
│   │   读路径（快）                                       │   │
│   │     L1  手写规则      → 内存       <1ms             │   │
│   │     L2  语义召回      → LanceDB    ~50ms            │   │
│   │     L3  全文归档      → SQLite FTS5 ~30ms           │   │
│   │     L4  画像          → 内存       <1ms             │   │
│   │                                                    │   │
│   │   写路径（非阻塞）                                    │   │
│   │     L3  同步写        → SQLite WAL  <10ms           │   │
│   │     L2  异步索引      → 后台线程                     │   │
│   │     抽取             → 后台线程                     │   │
│   │                                                    │   │
│   │   维护（cron 脚本，确定性）                           │   │
│   │     L4 画像生成      → 每日                          │   │
│   │     Bridge 候选导出   → 每日                          │   │
│   │     健康报告         → 每日                          │   │
│   └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

| 层 | 用途 | 存储 | 信任级别 |
|----|------|------|---------|
| **L1** | 手写规则、偏好、约束 | `MEMORY.md` / `USER.md` | 最高（人工） |
| **L2** | 持久语义事实 | LanceDB + embeddings | 中等（自动抽取） |
| **L3** | 完整会话归档 | SQLite FTS5 | 事实来源 |
| **L4** | 画像摘要 | `persona.md` | 稳定摘要 |
| **Bridge** | 审核后的持久候选 | JSONL | 审核后导入 |
| **KB** | 持久知识笔记 | Obsidian vault（Markdown） | 你自己拥有 |

### 读路径 — 异步预取

核心延迟优化：每轮对话结束后，下一轮的召回已在后台预取完成，下次读取直接命中缓存。

```
第 N 轮结束
  └── queue_prefetch(第 N+1 轮的 query)
        ├─ [线程1] L2 向量检索    (~50ms)
        ├─ [线程2] L3 FTS5 检索   (~30ms)
        └─ [线程3] L4 画像读取    (<1ms)
             └── 结果合并 → _prefetch_cache

第 N+1 轮开始
  └── prefetch(query)
        └── _prefetch_cache[query]  →  <1ms
```

缓存未命中时，`prefetch()` 回退到仅 L1 的路径（最快，且永不失败）。

### 写路径 — 非阻塞流水线

```
sync_turn(messages)
  ├── L3 SQLite WAL 写入   (同步，<10ms，RLock 保证线程安全)
  │     └── 镜像进 messages_fts（若无法写入，经 _diag 上报，绝不静默）
  └── _write_queue.put(messages)  → 后台 worker
        ├── L2 LanceDB 索引
        └── 事实抽取（可选）
              └── confidence ≥ 0.8  → 直接写 L2
                  confidence ≥ 0.5  → Bridge 候选（JSONL，可审核）
                  confidence <  0.5 → 丢弃
```

L3 同步写，因为它是事实来源、绝不能丢数据。L2 与抽取是尽力而为的索引，安全入队（worker 退出时按条隔离排空队列，不会静默丢队列里的轮次）。

### Token 预算

```
[system prompt]
  └── L4 persona.md            固定，永不被挤占

[上下文注入 — prefetch]
  ├── L1 手写规则              固定预算（800 tokens）
  └── L2 + L3 共享             动态分配（1200 tokens）
        ├── L2  按相关度分数
        └── L3  按相关度 × 时间衰减
```

L1 永不被 L2/L3 结果挤占——人工写的基线始终优先。

## 知识库与蒸馏

知识库是一个 **Obsidian vault**——Markdown + frontmatter + `[[wikilink]]`。向量索引（`KBIndex`）只是**投影**，可随时通过 `KnowledgeBase.reindex` 重建；任何 agent、任何工具都能直接读写 vault，不依赖本插件。

```
inbox/      低置信条目，待人工确认
notes/      原子笔记（Zettelkasten 卡片，已确认的知识）
projects/   PARA — 有明确产出的项目
areas/      PARA — 长期负责的领域
resources/  PARA — 外部资料/参考
archive/    PARA — 归档
index.md    MOC 导航入口
```

### 三条链路

| 链路 | 载体 | 触发 | 机制 |
|------|------|------|------|
| **记忆** | L1–L4（`MEMORY.md` / LanceDB / SQLite / persona） | `sync_turn` 自动 | 不变，始终开启 |
| **知识** | Obsidian vault | `on_session_end` 自动（可选） | 半自动蒸馏 |
| **摄入** | fetch / read / transcribe → 归纳 → `kb_add` | 手动 | 内容获取 |

### 审核门

蒸馏流程**刻意设计成半自动**——绝不把垃圾直接写进永久笔记：

```
会话结束（on_session_end）
      ↓ 取本会话文本（截断 24k）
LLM 归纳（与 agent 对话模型同款）
      ↓ [{title, body, tags, concepts, confidence}]
密钥闸门（复用 _bridge.SECRET_PATTERNS，fail-closed）
      ├─ 含密钥 → 拒绝写盘（返回明确 error）
      └─ 干净 →
           ├─ confidence ≥ 0.7 → notes（永久）
           └─ confidence <  0.7 → inbox（review_required）
      ↓
链接补全（共享 concepts/tags → 互加 [[wikilink]]，仅 notes、去重）
      ↓
审核门（governed_kb_review：list → approve 进 notes / reject 进 archive）
```

关键架构决策：归纳用**与 agent 对话同款 model id 与端点**（而非「调用 agent」），因为插件跑在 gateway 后台进程，拿不到正在对话的 agent 实例。

## 工具清单

插件注册 **10 个工具**（3 记忆 + 7 KB）：

| 工具 | 类别 | 用途 |
|------|------|------|
| `governed_search` | 记忆 | 检索 L2/L3 的相关记忆 |
| `governed_audit` | 记忆 | 下钻记忆溯源 |
| `governed_health` | 记忆 | 记忆系统健康报告 |
| `governed_kb_search` | KB | 检索笔记（语义 + 关键词 + 反链） |
| `governed_kb_add` | KB | 写入/更新笔记（带治理） |
| `governed_kb_get` | KB | 读取笔记完整正文 |
| `governed_kb_review` | KB | 审核 inbox 候选（approve/reject） |
| `governed_kb_fetch` | KB | 抓取 URL → 文本 |
| `governed_kb_read_file` | KB | 读取本地文件 → 文本 |
| `governed_kb_transcribe` | KB | 音频转写（ASR；长音频自动切分） |

## 模块地图

| 模块 | 职责 |
|------|------|
| `__init__.py` | `GovernedMemoryProvider` 入口、工具注册、`register(ctx)` |
| `_config.py` | 配置加载（`governed_memory.json` + 环境变量兜底） |
| `_recall.py` | 并行读引擎（`queue_prefetch` / `prefetch`）、分数映射 |
| `_sync.py` | 异步写流水线（`sync_turn`、`WriteQueue`、L3 FTS5 镜像） |
| `_embedding.py` | embedding 后端抽象（`auto`/`fastembed`/`sentence_transformers`/`none`） |
| `_kb.py` | 知识库门面（search/add/get/review/reindex）+ `KBIndex` |
| `_vault.py` | Obsidian vault 存储（PARA 骨架、frontmatter、wikilink） |
| `_synthesize.py` | 会话归纳 → 候选卡片（LLM，同款模型） |
| `_ingest.py` | 内容获取（fetch/read/transcribe，含长音频自动切分，全部可选依赖降级） |
| `_bridge.py` | Scope Recall 的持久候选（密钥模式、导出） |
| `_compress.py` | 长任务的 Mermaid 短期压缩 |
| `_migrations.py` | 幂等 schema 迁移（`run_pending`） |
| `_diag.py` | 分叉 / 降级上报（绝不静默） |

### Embedding 后端

`vector.backend` 选择 L2 的 embedding 策略，全部统一在 `Embedder` 协议之后：

| 后端 | 说明 | 备注 |
|------|------|------|
| `auto` | 依次尝试 fastembed → sentence_transformers → none | 默认，优雅降级 |
| `fastembed` | qdrant fastembed（ONNX，无 PyTorch） | 最轻量，模型首次自动下载 |
| `sentence_transformers` | sentence-transformers（torch） | 原始后端 |
| `none` | 无 embedding | L2 回退为文本扫描 |

## 安装

> **这台机器已经装过了？那就不要重装。** 这是**一份共享存储，只需装一次**；
> 之后接入的其它 agent 是「接入」，不是「再装一遍」——照着手工步骤重跑会
> **覆盖手写的 L1 规则**。先跑 `python memory_cli.py health`，若已返回
> `"ok": true`，请看 [docs/install.md](docs/install.md) 第 0 节；
> 接入第二个（非 Hermes）agent 见 [docs/attach-agents.md](docs/attach-agents.md)。

### Agent 安装（推荐）

```bash
# 基础版（L3 + L4 + Bridge + KB）
pip install hermes-memory-governed

# 带 L2 向量检索
pip install hermes-memory-governed[vector]
```

然后在 `~/.hermes/config.yaml` 里启用：

```yaml
memory:
  provider: governed
```

### 脚本安装

```bash
# Linux / macOS
bash install.sh

# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
```

### 验证

```bash
python scripts/memory_pipeline.py health
```

## 配置

配置在 `$HERMES_HOME/governed_memory.json`（完整带注释示例见 [`config/governed_memory.example.json`](config/governed_memory.example.json)）。关键配置段：

| 配置段 | 控制内容 |
|--------|----------|
| `recall` | L1/L23 token 预算、预取 TTL、L3 时间衰减、并行超时 |
| `sync` | L3 同步/异步、L2 异步、抽取开关 |
| `persona` | L4 增量生成 + 间隔 |
| `tencent_extract` | 可选的腾讯式抽取 + 置信阈值 |
| `mermaid_compress` | Mermaid 短期压缩 + canvas token 上限 |
| `vector` | L2 后端、模型、维度 |
| `embedding` | 远程 embedding API（OpenAI 兼容 `/embeddings`） |
| `reranking` | 可选重排器（SiliconFlow bge-reranker） |
| `kb` | KB top-k、阈值、语义/关键词开关、自动链接 |
| `asr` | 音频转写（XingChenASR）——`timeout_seconds`（默认 600：单次请求上限）与 `chunk_minutes`（默认 10：超过此长度的录音自动拆段并按顺序拼接转写结果）。两者均有默认值，无需修改配置。 |
| `synthesis` | 会话蒸馏（provider/model/key、启用开关） |

> ⚠️ **维度一致性**：同一套 L2 表只能有一种向量维度。切换后端（API ↔ 本地）或换模型会导致维度变化，必须重建 L2：`python scripts/l2_rebuild.py`。

### 国内跑 HuggingFace

本地 embedding 需要这两个环境变量（缺了会因 xet CDN 401 失败）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

## 运维

推荐的每日 cron 排程：

| 时间 | 任务 | 脚本 |
|------|------|------|
| 03:40 | Wiki → NotebookLM → L2 | `wiki_nblm_daily.py` |
| 04:25 | Wiki 主题 → 归纳 → L2 | `wiki_nblm_induction.py` |
| 09:10 | L4 画像生成 | `l4_persona_daily.py` |
| 09:15 | Bridge 候选导出 | `scope_recall_bridge.py export` |
| 09:45 | 健康报告 | `memory_health_report.py` |

### 统一流水线

```bash
python scripts/memory_pipeline.py daily           # 每日维护
python scripts/memory_pipeline.py bridge          # 导出 Bridge 候选
python scripts/memory_pipeline.py bridge --dry-run
python scripts/memory_pipeline.py health          # 健康报告
python scripts/memory_pipeline.py persona         # 重新生成 L4 画像
```

### Bridge 审核

```bash
python scripts/memory_pipeline.py bridge --dry-run   # 预览
python scripts/scope_recall_bridge.py status          # 查看状态
python scripts/scope_recall_bridge.py validate        # 校验候选
```

## 测试与 CI

- 本地全量测试：`python -m pytest tests/ -q` → **1327 passed**（2026-09-18）。
- CI（[`.github/workflows/ci.yml`](.github/workflows/ci.yml)）跑 **4 维矩阵**——`py3.10` / `py3.13` × `core` / `vector`：
  - `core` 只装 `.[dev]`（无向量后端 → 验证降级路径）。
  - `vector` 装轻量后端（`lancedb` + `pyarrow` + `fastembed`，**不装** `sentence-transformers`，避免拖 ~2GB 的 torch）。

## 相关文档

- [docs/architecture.md](docs/architecture.md) — 完整架构
- [docs/hybrid-architecture.md](docs/hybrid-architecture.md) — 云端 + 本地混合部署设计
- [docs/phase3-design.md](docs/phase3-design.md) — 知识蒸馏设计
- [docs/configuration.md](docs/configuration.md) — 配置项
- [docs/install.md](docs/install.md) — 安装指南
- [docs/operations.md](docs/operations.md) — cron 顺序和运维
- [docs/acceptance-report.md](docs/acceptance-report.md) — 验收报告

## License

MIT
