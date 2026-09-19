# Phase 3 设计：对话自动沉淀 + 分流 + review 门 + 链接补全

> 状态：已实现并同步 hermes（2026-09-01）
> 验证：项目全量 318 passed；hermes 真实环境端到端通过

## 1. 目标与定位

知识库（Obsidian vault）此前**全靠 agent 手动调 `governed_kb_add` 落库**，
对话里产生的、值得长期保留的知识不会自动沉淀。Phase 3 把「对话 → 知识卡片」
这条链路自动化，同时保持**半自动**的治理约束（高置信进正库、低置信待审）。

**记忆链路（L1–L4）不动** —— 本阶段只改「知识」这条链路。

## 2. 三条链路边界

| 链路 | 载体 | 触发 | Phase 3 前 | Phase 3 后 |
|---|---|---|---|---|
| 记忆 | L1–L4（MEMORY.md / LanceDB / SQLite / persona） | `sync_turn` 自动 | ✅ 已通 | 不变 |
| 知识 | Obsidian vault（inbox/notes/projects/areas/resources/archive） | 手动 `kb_add` | ⚠️ 手动 | ✅ 半自动 |
| 摄入 | fetch/read/transcribe → 归纳 → `kb_add` | 手动 | ✅ 已通 | 不变 |

## 3. 核心架构决策

### 3.1 归纳用「与 agent 对话同款模型」，而非「调用 agent」

插件跑在 gateway 后台进程，`on_session_end` 钩子拿不到「正在对话的 agent 模型实例」。
正确的落地方式是：**插件侧自己调 LLM 做归纳，但模型与 agent 对话用同一个 model id
与端点**，不依赖正在对话的 agent。

- hermes agent 对话模型（`config.yaml` 的 `model` 段）：
  `provider=xiaomi`、`base_url=<provider-base-url>`、
  `model=mimo-v2.5`、`api_key_env=XIAOMI_API_KEY`。
- 归纳配置 `synthesis` 段与之对齐（同模型、同端点、同 key）。

> 澄清记录：早期误把「用 agent 对话模型」理解为「agent 在对话里主动归纳」，
> 据此一度取消了插件侧归纳器；用户澄清后改回插件侧调 LLM，模型选同款。

### 3.2 归纳与治理分离（延续 `_ingest` 的职责边界）

- `_synthesize.py` 只做「对话 → 结构化候选卡片」的机械活（调 LLM + 解析 JSON）。
- 落库的**治理**由 `KnowledgeBase.add(confidence=...)` 承担：密钥闸门、置信分流、链接补全。
- 链路统一为：`归纳（拿候选）→ 治理落库（kb_add）`，任何 agent / 任何入口都走同一套治理。

## 4. 数据流

```
对话 (sync_turn 积累)
      ↓
会话结束 (on_session_end)
      ↓ 取本会话文本（截断 24k）
LLM 归纳（mimo-v2.5，同款模型）
      ↓ [{title, body, tags, concepts, confidence}]
密钥闸门（复用 _bridge.SECRET_PATTERNS，fail-closed）
      ├─ 含密钥 → 拒绝写盘（返回明确 error，不静默丢）
      └─ 干净 →
           ├─ confidence ≥ 0.7 → notes（正库）
           └─ confidence < 0.7 → inbox（review_required，待人工确认）
      ↓
链接补全（共享 concepts/tags 的笔记互加 [[wikilink]]，仅正库、去重）
      ↓
review 门（governed_kb_review: list → approve 进 notes / reject 进 archive）
```

## 5. 阶段拆解与实现

### 3a 链接补全

`_kb.py` 新增 `_link_related` + `_link_if_missing`：

- 新笔记落库后，扫描 vault 中**共享 concepts 或 tags** 的笔记，双向互加 `[[wikilink]]`。
- 仅对 `notes` 正库做（inbox 待审不互链，避免噪音）。
- 去重：`_link_if_missing` 检测 body 已含 `[[标题]]` 则跳过。
- 线性扫描，个人 vault（数千篇内）毫秒级。

### 3b 治理化沉淀

`_kb.add()` 增强：

- 新增 `confidence` 参数（0–1）：
  - 传了 `confidence` → 由 `kb.confidence_threshold` 接管分区（≥0.7 → notes，<0.7 → inbox + `review_required`）。
  - 不传 → 按显式 `section` 落库（手动场景，行为不变，向后兼容）。
- 密钥闸门：落库前对 title/body 过 `_bridge.has_secret_like_text`（fail-closed），
  含密钥返回 `{ok:false, error:...}`，**绝不写盘**。

### 3c review 门

新增 `governed_kb_review` 工具，闭环 inbox 待审：

| action | 行为 |
|---|---|
| `list` | 列 inbox 待审笔记（含 confidence、preview） |
| `approve` | inbox → notes（进正库 + 补双向链接） |
| `reject` | inbox → archive（标 `review_rejected`） |

`_kb.py` 新增 `list_review` / `_relocate` / `approve` / `reject`。
`_relocate` 仅允许从 inbox 移出，目标同名文件存在则拒绝（不覆盖）。

### 3d session 自动归纳

- `_synthesize.py`（新模块）：
  - `synthesize_notes(messages, config)` → 调 OpenAI 兼容 `/chat/completions`，
    把对话归纳为候选卡片；失败/未启用/未配置一律返回 `[]`，绝不抛异常。
  - `_build_transcript`（拼 `role: content` + 截断 24k）、
    `_parse_candidates`（容错解析 JSON 数组，去 markdown 围栏、坏条目丢弃）。
- `__init__.py` 的 `on_session_end` 挂 `_synthesize_session(messages)`：
  归纳 → 逐条 `_kb.add(confidence=...)` 走治理。
- 默认 `enabled=False`（安全），hermes 环境显式开启。

## 6. 配置

`governed_memory.json` 新增/调整：

```json
{
  "kb": {
    "top_k": 10,
    "min_score": 0.0,
    "semantic_enabled": true,
    "keyword_enabled": true,
    "confidence_threshold": 0.7,
    "autolink_related": true
  },
  "synthesis": {
    "enabled": true,
    "provider": "xiaomi",
    "base_url": "<provider-base-url>",
    "api_key_env": "XIAOMI_API_KEY",
    "model": "mimo-v2.5",
    "max_candidates": 5
  }
}
```

`_config.py` 对应新增 `KnowledgeConfig.confidence_threshold` / `autolink_related`
与 `SynthesisConfig`（provider/base_url/api_key_env/model/max_candidates/enabled）。

## 7. 工具清单

Phase 3 后 hermes 共 **10 个工具**（3 记忆 + 7 KB）：

- 记忆：`governed_search` / `governed_audit` / `governed_health`
- KB：`governed_kb_search` / `governed_kb_add` / `governed_kb_get` /
  `governed_kb_review`（新增）/ `governed_kb_fetch` / `governed_kb_read_file` /
  `governed_kb_transcribe`

## 8. 验证

- 项目全量 pytest：**318 passed**（含新增 `tests/test_phase3.py` 22 项、
  `tests/test_synthesis.py` 13 项）。
- hermes 真实环境端到端（临时 vault，不碰真实 wiki）：
  - 密钥闸门拒绝、置信分流（hi→notes / lo→inbox+review）、链接补全、review 闭环全部通过。
  - session 归纳**真实调 `mimo-v2.5`**：2 段对话 → 归纳出 2 张卡（conf=1.0 → notes），落库 ok。
- gateway 已重启（PID 52908，飞书 connected），`synthesis.enabled=true` 生效。

## 9. 遗留与待办

- session 自动归纳的**实际效果**待观察（落库质量、噪音、是否过度沉淀）。
- 归纳 prompt 可继续调优（当前偏通用，未针对领域定制）。
- 转写质量已确认 OK；ASR（XingChenASR-V3.2-Ultra）计费目前为 0，持续留意。
- `confidence_threshold`（0.7）为初始值，可据实际 inbox 命中率回调。
