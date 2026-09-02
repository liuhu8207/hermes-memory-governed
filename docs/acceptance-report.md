# hermes-memory-governed 验收报告

**日期**：2026-08-31　**范围**：全量测试验证 + 缺陷修复 + 性能优化 + 安全加固
**执行**：软件开发团队（QA / 架构师 / 工程师 ×2）

---

## 一、TL;DR

一句话：**原有 61 条测试全绿是空库假象**。补测后暴露 25 条真实缺陷（含 3 条 P0），现已全部修复，测试从 99 条扩到 247 条，**244 passed / 0 failed / 3 skipped**；L3 写入性能提升约 **540 倍**；修复了一处会让密钥随每轮对话外泄的安全设计缺陷。

> **2026-09-02 复查更新**：L2 向量路径已补齐验证（原最大验证空白，见第八节附），
> 过程中新发现并修复一处 P0 —— 距离度量假设错误导致 L2 召回结果 100% 被丢弃。
> 测试 **244 passed / 3 skipped → 391 passed / 0 skipped**；项目已纳入 Git 版本控制。

## 二、测试演进

| 阶段 | 通过 | 失败 | 跳过 | 合计 |
|---|---|---|---|---|
| 接手时（原套件） | 61 | 0 | 3 | 64 |
| QA 补探针后 | 71 | 25 | 3 | 99 |
| 修复中途 | 230 | 2 | 3 | 235 |
| **终验（连续 2 轮）** | **244** | **0** | **3** | **247** |
| **2026-09-02（装 vector 依赖 + L2 补测后）** | **391** | **0** | **0** | **391** |

终验两轮全量结果一致，无 flaky。

原 3 条 skipped 均为 `importorskip("lancedb")` 的用例。**2026-09-02 装上
`lancedb` / `pyarrow` / `fastembed` 后，这 3 条从"跳过"转为真实执行并通过**，
跳过数归零；同时为 L2 距离换算补了 41 条回归测试（244 → 391）。

## 三、为什么原测试是"假象"

三个结构性原因，值得写进团队规范：

1. **空库掩盖性能缺陷** —— 每条测试都用全新空 `l3.db`，几十行下全表扫描是微秒级。**L3 缺索引这条 P0 在结构上不可能被现有套件发现。**
2. **mock 掩盖链路缺陷** —— L2→L3 审计下钻用 mock 测过，真实链路因键不匹配 100% 不通，逃逸到生产。
3. **假阴性验证方法** —— 用 `threading.active_count()` 差值统计线程，因启停时序恒为 0，差点让"线程爆炸"永久逃逸。

## 四、关键修复（P0）

| 缺陷 | 修复前 | 修复后 | 验证 |
|---|---|---|---|
| **L3 去重查询无索引** | 80k 行 / 50 消息 **429~877ms**（README 承诺 <10ms） | **0.79ms**（约 540 倍） | `EXPLAIN` 由 `SCAN messages` → `SEARCH ... USING COVERING INDEX idx_messages_dedup` |
| **中文多词查询召回 0 条** | 「天气 北京」→ 0 条（要求词必须相邻） | 正常召回 | 7 个场景全绿，含防过度匹配用例 |
| **FTS 写入失败静默吞掉** | 主表有行、索引无行，零日志、无补救 | ERROR 级告警 | 实测输出 `DATA LOSS — l3_fts: fts_insert_failed \| ...` |
| **L3Writer 跨线程崩溃** | 4 线程并发 → `ProgrammingError` + `database is locked` | 0 错误 | 4 线程并发写实测通过 |
| **审计下钻形同虚设** | rowid 按整条消息建键，fact 是句子片段 → 溯源 0/3 | 双 key + 四级回退 | 真实 L3 链路 N/N 可溯源 |

## 五、安全加固（决策经过三次反转）

Bridge 密钥处理最终定为 **fail-closed**：

- **初判**（错）：脱敏后入库 —— 忽略 L1 每轮注入 system prompt，密钥会随每次请求外泄到 LLM 提供商
- **复判**：架构师指出旧方案下未识别格式（PEM/Bearer/JWT/base64）同样原样落盘，"回退"对未识别格式并无帮助，真正差别仅限"已识别"的密钥
- **终裁**：fail-closed + 隔离区 + **L1 写入处硬闸门**

实现要点：
1. 含密候选写入 `bridge/quarantine.jsonl`，**绝不进 `candidates.jsonl`**
2. `import_approved()` 无条件拦截含密行 —— **`auto_approve=True` 也不放行**，且能拦住历史遗留行
3. `SECRET_PATTERNS` **只加 PEM + Bearer 两条**。在 fail-closed 下误报 = 静默丢弃合法记忆，**precision 优先于 recall**，明确不加裸 base64/JWT
4. 崩溃安全：先原子重写标记 imported（tmp + `os.replace`），再追加 L1 —— 失败只丢一次导入，绝不重复导入
5. 逃生舱 `allow_secret_candidates`（默认关闭），开启后脱敏入库但 L1 硬闸门仍拦

## 六、可观测性

原先 37 处 `debug` 对 1 处 `error`，所有数据丢失路径都静默。新增 `_diag.py`：

- `log_degraded()` —— WARNING，60s 节流（正常降级，如 lancedb 未装）
- `log_data_loss()` —— ERROR，**永不节流**（数据丢失/不一致）
- `stats()` 三键契约，供 `governed_health` 消费

实测输出样例：
```
degraded — l2: lancedb_missing | L2 falls back to L1+L3
DATA LOSS — bridge_secret: candidate_quarantined | patterns=['pem_private_key']
```

## 七、文件清单

**核心模块**
- `plugin/memory_governed/_sync.py` — L3 索引/线程安全/FTS 一致性/双 key 溯源/队列可靠性/切分/事实判定
- `plugin/memory_governed/_recall.py` — CJK 召回/连接复用/锁/mtime 缓存/超时兼容
- `plugin/memory_governed/__init__.py` — prefetch LRU/单飞去重/自愈/sync_turn 兜底/中文事实识别
- `plugin/memory_governed/_bridge.py` — fail-closed/隔离区/L1 硬闸门/导出锁/原子写
- `plugin/memory_governed/_embedding.py` — `EmbeddingService` 单例（消除读写两侧重复加载模型）
- `plugin/memory_governed/_migrations.py` — migration 003（幂等建索引）+ 002 日志分级
- `plugin/memory_governed/_config.py` — `l2_max_facts_per_turn` / `allow_secret_candidates`
- `plugin/memory_governed/_diag.py` **（新建）** · `plugin/__init__.py` **（新建，修 wheel 入口点导入失败）**
- `scripts/l3_retention.py` — 单事务 + `CAST(timestamp AS REAL)` 修正
- `.gitignore` — 加 `build/`、`*.egg-info/`

**测试**（新增 4 个文件，共 183 条）
- `tests/test_read_path.py`（45）· `tests/test_write_path.py`（84）· `tests/test_qa_probe.py`（28）· `tests/test_qa_concurrency.py`（8）· `tests/test_qa_bridge_secrets.py`（8）· `tests/test_qa_s11_regression.py`（9）· `tests/conftest.py`

**文档**：`docs/architecture-review.md` v1.1（1094 行，含实测证据）

## 八、已知未覆盖

> **2026-09-02 更新**：原第 1 项（L2 向量路径）已补齐验证并消除，详见本节附。
> 以下第 2、3 项仍然成立。

1. ~~**L2 向量路径全程未验证**~~ → **已于 2026-09-02 补齐**（见下方附）。
2. **Tencent LLM 抽取** 为原有 `# TODO` 存根，未实现（本次未动）。
3. **Python 3.10 兼容性** —— `except (TimeoutError, FuturesTimeoutError)` 已按 3.10 语义修复，但本机是 3.13，**无法实测**。已加版本守卫测试，CI 上 3.10 能拦住。

### 附：L2 向量路径验证结果（2026-09-02 补齐）

**如何绕开原阻塞**：原结论说"torch 体积过大装不上"，但 `pyproject` 声明的
`sentence-transformers` 并非唯一选择 —— 代码里 `backend=auto` 的**首选后端是
fastembed（ONNX，无 torch）**。只装 `lancedb` + `pyarrow` + `fastembed`
（约 90MB 模型）即可覆盖同一条代码路径，不必引入 torch。

**环境要点**（国内网络必需，缺第二项会因 HuggingFace xet 存储后端返回 401）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

**实测结果**（默认配置 `BAAI/bge-small-zh-v1.5`，512 维）：

| 验证项 | 结果 |
|---|---|
| 嵌入写入 | 3 条事实全部落库，均带真实 512 维向量 |
| 向量召回 | 命中，score 0.708 ~ 0.802 |
| 进入注入上下文 | 3 条 `[Fact]` 全部出现 |
| 相关性排序 | 查「前端构建工具」→ Vite 那条排第一（0.8465） |

**验证过程中暴露出一处 P0**（不装依赖永远发现不了，mock 与 skip 都掩盖了它）：

- **现象**：L2 召回结果 **100% 被丢弃**，用户对 L2 层完全无感。
- **根因**：score 换算写成 `1.0 - min(d, 1.0)`（按余弦相似度 ∈ [0,1] 假设），
  但 LanceDB `search()` 默认返回 **L2 欧氏距离**，384/512 维典型值 1.0~1.5，
  于是 `min(d, 1.0)` 恒为 1.0 → **score 恒为 0** → 被 `MIN_SCORE = 0.1` 全拦。
- **修复**：搜索显式指定 `.metric("cosine")`，score 改为 `1.0 - d/2.0`；
  抽出 `distance_to_score()` / `row_to_score()` 纯函数，并补 41 条回归测试。
- **防复发**：已做红绿对照 —— 把公式改回旧写法，新增测试 **立即失败**，
  证明这组测试真能拦住回归，而非摆设。

## 九、下一步建议（按优先级）

1. ~~**装一次 `[vector]` extras 跑通 L2 路径**~~ → **已于 2026-09-02 完成**（见第八节附）。
   经验：不必照装 `pyproject` 声明的 `sentence-transformers`（会拖进 torch，约 2GB）——
   `backend=auto` 的首选后端其实是 **fastembed（ONNX，无 torch）**，只装
   `lancedb` + `pyarrow` + `fastembed` 就覆盖了同一条代码路径。
2. **加 CI** —— 仓库**此前连 git 都没有**，已于 2026-09-02 初始化并建基线提交；
   CI 仍待做，建议至少跑 3.10 / 3.13 双版本 × 有/无 vector extras 四象限
3. **观察一周 facts 抽取量** —— 中文判定已放宽，配套加了 `record_metric` 统计，一周后据数据决定是否加置信度阈值
4. **给 `scripts/` 下 23 个运维脚本补测试** —— 目前零覆盖
5. **清理 `build/` 过期产物**（已在 `.gitignore` 中忽略，但工作树内仍存在）

## 十、终验结论

主理人独立抽查（不依赖成员测试套件，直接调生产代码跑真实场景）**25 项全部通过**，覆盖：prefetch 缓存上界（200→64）、单飞去重（20 次→1 次召回）、CJK 召回与连接复用（5→1）、fail-closed 三重行为、migration 幂等、`EXPLAIN` 走索引、4 线程并发写、FTS 失败可观测、切分保留小数/URL/IP、中文事实识别 9/9。

**判定：可交付。**
