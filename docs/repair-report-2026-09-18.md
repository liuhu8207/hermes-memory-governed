# HGM 共享记忆系统修复报告

> **日期**：2026-09-18  
> **修复者**：DSH (dsh agent)  
> **涉及仓库**：hermes-memory-governed  

---

## 一、问题背景

HGM（Hermes Governed Memory）共享记忆系统在 DSH（DeepSeek Harness）侧暴露两个持续性故障：

1. **L2 语义搜索退化** — 所有查询退化为纯词法匹配（`lexical-dis-max`），语义通道（`semantic-cosine`）完全不可用。
2. **中文写入乱码** — 通过 MCP `hgm_remember` 工具写入的中文事实，在 LanceDB 中存储为乱码字符。

---

## 二、根因分析

### 故障 1：语义搜索退化

| 项目 | 详情 |
|------|------|
| **错误信息** | `embed_one failed: 'utf-8' codec can't encode character '\udcae' in position 59: surrogates not allowed` |
| **根因** | `EmbeddingService.embed_one()` 将含 Unicode surrogate 字符（如 `\udcaf`、`\udcae`）的文本直接传给 SiliconFlow embedding API。Python 的 `httpx` / `json.dumps` 在 UTF-8 编码时拒绝代理对字符，导致 API 调用失败。 |
| **影响** | `embed_one()` 返回 `None`，L2 搜索回退到纯文本扫描（`lexical-dis-max`），语义相似度匹配完全失效。 |
| **来源** | DSH 的 prompt 注入文本中偶尔包含 surrogate 字符（可能来自终端日志、转义序列等）。 |

### 故障 2：中文写入乱码

| 项目 | 详情 |
|------|------|
| **现象** | `hgm_remember` 返回 `ok: true`，但 `content` 字段显示为乱码（如 `璇?涔夋悳绱?`），LanceDB 中存储的也是乱码。 |
| **根因** | `hgm_mcp.py` 启动时只对 `sys.stdout` 和 `sys.stderr` 执行了 `reconfigure(encoding="utf-8")`，**遗漏了 `sys.stdin`**。Windows 上 `sys.stdin` 默认使用控制台编码（GBK/cp936），DSH 发送的 UTF-8 JSON 被 Python 用 GBK 解码，中文字符被错误转码后写入 LanceDB。 |
| **影响** | 所有通过 MCP 写入的中文事实全部乱码，且不可恢复。 |
| **发现过程** | 修复语义搜索后测试写入，发现 WorkBuddy 通过 CLI 写入的数据正常（不经过 MCP stdin），而 DSH 通过 MCP 写入的数据全部乱码。 |

---

## 三、修复方案

### 修复 1：surrogate 字符清洗（`_embedding.py`）

**文件**：`plugin/memory_governed/_embedding.py`

新增 `EmbeddingService._sanitize()` 静态方法，在 `embed_one()` 和 `embed_batch()` 入口统一清洗：

```python
@staticmethod
def _sanitize(text: str) -> str:
    """清洗文本中的 surrogate 字符，防止 UTF-8 编码失败。

    Windows Python 在处理包含代理对（surrogate pair）的文本时，
    json.dumps / httpx 会抛出 surrogates not allowed 错误。
    常见来源：某些日志/终端输出含 \\udxxx 转义序列。
    """
    try:
        text.encode("utf-8")
        return text  # 没问题，无需清洗
    except (UnicodeEncodeError, UnicodeDecodeError):
        # 替换掉无法编码的字符（包括 lone surrogates）
        return text.encode("utf-8", errors="replace").decode("utf-8")
```

- 正常文本零开销（try 成功直接返回）
- 含 surrogate 的文本用 `replace` 策略替换为 `?`，保证 UTF-8 可编码
- 同时在 `embed_batch()` 的 `normalized` 构建中调用 `_sanitize()`

### 修复 2：stdin UTF-8 编码（`hgm_mcp.py`）

**文件**：`scripts/hgm_mcp.py`

```python
# 之前：只重配置 stdout/stderr
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

# 之后：stdin 也必须重配置
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
```

同时在 `tool_remember` 和 `tool_recall` 入口加了 `_ensure_utf8()` 防御性清洗，作为最后防线：

```python
def _ensure_utf8(text: str) -> str:
    """清洗文本中的 surrogate 字符，防止后续 UTF-8 编码失败。"""
    try:
        text.encode("utf-8")
        return text
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text.encode("utf-8", errors="replace").decode("utf-8")
```

### 修复 3：写入路径防御（`memory_cli.py`）

**文件**：`memory_cli.py`

在 `cmd_remember()` 的 `cleaned` 处理后加了 surrogate 清洗，作为最后防线：

```python
cleaned = " ".join(str(text or "").split())
# 清洗 surrogate 字符（Windows Python 的 UTF-8 编码器不接受代理对，
# 会导致 LanceDB Arrow 序列化和 embedding API 调用失败）
try:
    cleaned.encode("utf-8")
except UnicodeEncodeError:
    cleaned = cleaned.encode("utf-8", errors="replace").decode("utf-8")
```

---

## 四、测试结果

### 4.1 语义搜索恢复测试

共 14 个查询，覆盖中文、英文、混合语言、不同主题：

| # | 查询 | 搜索模式 | L2 命中 | KB 命中 | 状态 |
|---|------|----------|---------|---------|------|
| 1 | 密码管理器配置 | `semantic-cosine` | 0 | 0 | ✅ |
| 2 | Hermes gateway configuration | `semantic-cosine` | 2 | 5 | ✅ |
| 3 | DSH 共享记忆接入方式 | `semantic-cosine` | 0 | 1 | ✅ |
| 4 | SSH remote connection setup | `semantic-cosine` | 3 | 2 | ✅ |
| 5 | LanceDB 数据库 schema 问题 | `semantic-cosine` | 2 | 0 | ✅ |
| 6 | network topology router MikroTik | `semantic-cosine` | 0 | 0 | ✅ |
| 7 | embedding 失败 surrogates | `semantic-cosine` | 0 | 1 | ✅ |
| 8 | 飞书 Lark API integration | `semantic-cosine` | 0 | 5 | ✅ |
| 9 | 密码管理器 示例密码管理器 配置 | `semantic-cosine` | 0 | 0 | ✅ |
| 10 | Hermes gateway restart 方法 | `semantic-cosine` | 4 | 5 | ✅ |
| 11 | NAS 示例NAS 配置 | `semantic-cosine` | 0 | 5 | ✅ |
| 12 | git push GitHub proxy | `semantic-cosine` | 2 | 5 | ✅ |
| 13 | Tailscale 组网方案 示例路由器 | `semantic-cosine` | 1 | 0 | ✅ |
| 14 | pyarrow LanceDB add_columns | `semantic-cosine` | 1 | 0 | ✅ |

**语义搜索恢复率：14/14 = 100%**（修复前 0%，全部退化为 `lexical-dis-max`）

### 4.2 写入 + 召回闭环测试

| 测试项 | 结果 | 详情 |
|--------|------|------|
| `hgm_remember` 写入中文 | ✅ `ok: true` | `content` 字段中文正确，无乱码 |
| `hgm_recall` 召回刚写入的内容 | ✅ `semantic-cosine` | score=0.8872，精确匹配 |
| 去重检测 | ✅ 正确 | 重复写入返回 `duplicate: true` |
| `hgm_kb_search` 知识库搜索 | ✅ 正常 | 返回相关笔记 |

### 4.3 多 Agent 署名测试

| Agent | L2 Facts | KB Notes | 状态 |
|-------|----------|----------|------|
| dsh | 4 | 1 | ✅ 正确署名 |
| workbuddy | 13 | 1 | ✅ 正确署名 |
| hermes | 1 | 0 | ✅ 正确署名 |
| unattributed | 41 | — | ✅ 历史数据 |

### 4.4 系统健康检查

| 检查项 | 状态 | 详情 |
|--------|------|------|
| L1 规则层 | ✅ | MEMORY.md + USER.md + persona.md 可读 |
| L2 事实库 | ✅ | LanceDB 存储正常，full_store_visible: true |
| L3 会话归档 | ✅ | SQLite FTS5 数据库正常 |
| L4 画像 | ✅ | persona.md 存在 |
| KB 知识库 | ✅ | 27 篇笔记 |
| MCP 工具接口 | ✅ | 5 个工具全部可用 |
| embedding 后端 | ✅ | api:siliconflow, dim=1024 |
| Python 运行时 | ✅ | Python 3.12, 无缺失模块 |

---

## 五、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|----------|------|
| `plugin/memory_governed/_embedding.py` | 新增方法 + 调用点 | `EmbeddingService._sanitize()` 静态方法；在 `embed_one()` 和 `embed_batch()` 入口调用 |
| `scripts/hgm_mcp.py` | 修改 + 新增 | `sys.stdin` 加入 UTF-8 重配置列表；新增 `_ensure_utf8()` 函数；`tool_remember` 和 `tool_recall` 入口防御性清洗 |
| `memory_cli.py` | 新增防御 | `cmd_remember()` 中 `cleaned` 变量的 surrogate 清洗 |

---

## 六、遗留问题

| 问题 | 严重度 | 说明 |
|------|--------|------|
| 历史乱码数据 | ⚠️ 中 | 修复前 DSH 通过 MCP 写入的 1 条事实（rowid=32）内容为乱码，需要删除并重新写入。WorkBuddy 通过 CLI 写入的数据不受影响（不经过 MCP stdin）。 |
| MCP 进程热更新 | ℹ️ 低 | 修改 `.py` 文件后需要手动杀 MCP 进程才能生效（DSH 不会自动重启 MCP 子进程）。已知限制，非 bug。 |

---

## 七、经验总结

1. **Windows 编码陷阱**：Python 在 Windows 上默认使用 GBK 编码控制台 I/O。即使设置了 `PYTHONIOENCODING=utf-8`，DSH 启动 MCP 子进程时会清洗环境变量，导致该设置不生效。`reconfigure(encoding="utf-8")` 是更可靠的方案——但必须覆盖 **stdin、stdout、stderr 三个流**。

2. **防御性编码**：任何接受外部文本输入的接口（MCP JSON-RPC、CLI 参数、API 调用）都应该在入口处清洗 surrogate 字符。`try/except + errors="replace"` 是最小代价的防御手段。

3. **LanceDB 的编码敏感性**：LanceDB 底层使用 Apache Arrow 存储，Arrow 的 string 类型要求严格的 UTF-8 编码。含 surrogate 字符的文本虽然在 Python 内部可以表示，但序列化为 Arrow 时会失败。

4. **诊断优先**：修复过程中通过 Arrow 原始数据（`table.to_arrow()`）vs pandas 显示（`table.to_pandas()`）的对比，快速定位了"乱码是写入时产生还是显示时产生"的问题，避免了错误归因。

---

## 八、审查修正（2026-09-18 · 独立复核后补记）

> 以下由另一位 agent 独立复现后补记。**根因判断成立、修复有效**（已用 `PYTHONIOENCODING=gbk`
> 直接复现故障、并验证修复后中文正确落库；语义通道独立复跑 8/8 `semantic-cosine`、零词法回退；
> 代理对路径经转义 `\udcae` 端到端验证通过）。以下三点需要修正：

### 修正 1：症状描述不完整 —— 不止"乱码"，还会**硬崩溃**

原文写"所有通过 MCP 写入的中文事实全部乱码"。实际复现到的失败形态是**进程崩溃**：

```
UnicodeDecodeError: 'gbk' codec can't decode byte 0xad in position 113
  at: for line in sys.stdin:        退出码=1，stdout 完全为空
```

GBK 不是任意 UTF-8 的双射解码器，所以**同一个根因有两张脸**：解得出就写成乱码（原文观察到的），
解不出就**直接崩溃、什么都不返回**。这一区别对使用方很重要——崩溃时 agent 看到的不是
"乱码数据"，而是**工具整个失效**。

### 修正 2：修复当时**没有部署**，Hermes 侧仍未生效

复核时部署副本 `$HERMES_HOME/plugins/governed/_embedding.py` 里 `_sanitize` **出现 0 次**。
插件类改动必须同步到部署目录并重启 Gateway 才对 Hermes 生效；报告只在 CLI / MCP 路径上
测试（那两条直接读仓库），所以**结论对它自己的测试成立、对 Hermes 不成立**。
（已于同日完成部署 + 重启，并逐字节校验。）

### 修正 3：署名计数混用了两个量纲

原文 `unattributed 41` 放在 **L2 那一列**，读的人会算成 59 行。
**实际 L2 共 34 行**：`workbuddy 13 / dsh 4 / hermes 1 / 未署名 16`。
那个 41 是 **CLI 的合计值（L2 16 + KB 25）**，不是 L2 的计数。

### 补充：同一段逻辑有三份实现，且未覆盖真正的落库路径

`_embedding._sanitize`、`hgm_mcp._ensure_utf8`、`memory_cli.cmd_remember` 内联块
三份实现完全相同，而**插件自己从 L3 抽取写 L2 的那条路（事实真正落进 L2 的入口）
没有任何清洗**——`_sanitize` 只保住了嵌入调用，保不住 `table.add` 的 Arrow 序列化。

已收敛为单一定义 `plugin/memory_governed/_text.py`（`sanitize_utf8` / `sanitize_messages`），
并在写入管道入口 `sync_turn` **清洗一次**，一处覆盖 L3 归档、FTS 镜像、L2 抽取与嵌入四个消费者。
测试见 `tests/test_text_hygiene.py`（含"只允许一份实现"的源码守卫）与
`tests/test_hgm_mcp_write.py::TestSurrogateInput`。
