# Hermes Memory Governed

一个给 Hermes Agent 使用的记忆治理插件。把 L1 手写规则、L2 语义召回、L3 会话归档、L4 画像压缩、可选的腾讯式自动蒸馏和 Mermaid 短期压缩整合成一个高性能 MemoryProvider。

## 设计目标

1. **读得快** — prefetch 异步预计算，缓存命中 <1ms
2. **内容全** — L1 手写规则固定注入，L2/L3 并行召回，L4 画像进 system prompt
3. **不卡顿** — sync_turn 非阻塞（L3 同步 <10ms，L2/蒸馏异步）

## 架构

```
读路径：queue_prefetch() → 并行 L2+L3+L4 → 缓存 → prefetch() <1ms
写路径：sync_turn() → L3 同步写 + L2/蒸馏异步队列
维护：  cron → L4 生成、Bridge 导出、健康报告
```

## 层级

| 层 | 用途 | 存储 | 信任级别 |
|---|------|------|---------|
| L1 | 手写规则、偏好、约束 | MEMORY.md / USER.md | 最高（人工） |
| L2 | 语义事实、自动提取 | LanceDB + embeddings | 中等 |
| L3 | 完整会话归档 | SQLite FTS5 | 事实来源 |
| L4 | 画像摘要 | persona.md | 稳定摘要 |
| Bridge | 审核后的持久候选 | JSONL | 审核后导入 |

## 快速开始

```powershell
# 安装（交互式：会先询问 embedding API，跳过则用本地模型）
powershell -ExecutionPolicy Bypass -File install.ps1

# 或手动：复制 plugin/memory_governed/*.py 到 ~/.hermes/plugins/governed/
# 然后在 config.yaml 设置：
#   memory.provider: governed

# 空跑测试
python scripts/memory_pipeline.py daily --dry-run
python scripts/memory_pipeline.py health
```

## L2 向量检索的 embedding 后端

安装时**先询问是否配置 embedding API**；不输入 API（略过）才使用本地模型：

- **API**（OpenAI 兼容 `/embeddings`）：无需本地模型，维度自动检测（如 1024）。
- **本地模型**（默认 `BAAI/bge-small-zh-v1.5`，512 维，首次下载约 100MB）。

⚠️ **同一套 L2 表只能一种向量维度**。切换后端（API ↔ 本地）或换模型导致维度变化时，
必须重建 L2：`python scripts/l2_rebuild.py`。详见
[docs/configuration.md](docs/configuration.md) 的「维度一致性」一节。

## 相关文档

- [docs/architecture.md](docs/architecture.md) — 完整架构
- [docs/install.md](docs/install.md) — 安装指南
- [docs/configuration.md](docs/configuration.md) — 配置项
- [docs/operations.md](docs/operations.md) — cron 顺序和运维
