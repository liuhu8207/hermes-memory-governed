# Hermes 混合架构设计：云端 + 本地

## 需求

1. 两边记忆共享（config / memory / skills 统一）
2. 本地关机 → 云端独立执行任务
3. 本地开机 → 云端下发任务到本地，或用户直连本地
4. 国内网络 → 飞书/微信/QQ 做控制面板

## 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                    飞书 / 微信 / QQ                           │
│                    （你的遥控面板）                            │
└───────────────────────────┬─────────────────────────────────┘
                            │ 消息
┌───────────────────────────▼─────────────────────────────────┐
│                  云服务器 VPS（常驻）                          │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Hermes Gateway（hermes serve）                       │   │
│  │  ├── 飞书适配器（始终在线，接收消息）                    │   │
│  │  ├── 任务调度器（判断本地是否在线）                      │   │
│  │  ├── 本地执行器（SSH 到本地）                          │   │
│  │  └── 云端执行器（本地离线时自己执行）                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  共享状态层（云端副本）                                  │   │
│  │  ├── config.yaml          ← Syncthing 双向同步        │   │
│  │  ├── memory/MEMORY.md     ← Syncthing 双向同步        │   │
│  │  ├── memory/USER.md       ← Syncthing 双向同步        │   │
│  │  ├── memory/persona.md    ← Syncthing 双向同步        │   │
│  │  ├── skills/              ← Syncthing 双向同步        │   │
│  │  └── state.db             ← 云端独立（不同步）         │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
              本地在线时          本地离线时
                    │               │
            ┌───────▼───────┐  ┌───▼───────────────┐
            │  本地机器      │  │  云端独立执行       │
            │  （家里 PC）   │  │  （VPS 自己跑）     │
            │               │  │                    │
            │  Hermes Agent │  │  终端后端: local    │
            │  终端: local   │  │  可以跑脚本/代码    │
            │  浏览器: 本地  │  │  不能操作本地文件    │
            │  文件: 本地    │  │                    │
            └───────────────┘  └────────────────────┘
```

## 核心机制：任务调度器

```
用户发消息到飞书
  │
  ├── 云端 Gateway 收到
  │
  ├── 检查本地是否在线（心跳机制）
  │     │
  │     ├── 在线 → SSH 下发任务到本地
  │     │         本地 Agent 执行
  │     │         结果通过 SSH 返回
  │     │         云端转发结果到飞书
  │     │
  │     └── 离线 → 云端 Agent 自己执行
  │               结果直接发到飞书
  │               标记 [云端执行]
  │
  └── 用户收到回复
```

## 心跳机制

```
本地机器启动时：
  1. 启动 Hermes Agent
  2. 向云端注册：POST /api/local/register
     { "status": "online", "capabilities": ["terminal", "browser", "files"] }
  3. 每 30 秒发送心跳：POST /api/local/heartbeat
  4. 关机时优雅注销（或云端超时 60s 判定离线）

云端：
  维护 local_status = { online: bool, last_heartbeat: timestamp }
  收到任务时检查 local_status.online
```

## 记忆共享方案

### 方案 A：Syncthing（推荐，零成本）

```
本地 ~/.hermes/          Syncthing          云端 ~/.hermes/
├── config.yaml    ←────────────────→    ├── config.yaml
├── memory/        ←────────────────→    ├── memory/
│   ├── MEMORY.md                           │   ├── MEMORY.md
│   ├── USER.md                             │   ├── USER.md
│   └── persona.md                          │   └── persona.md
├── skills/        ←────────────────→    ├── skills/
└── state.db       ←─── 不同步 ───→      └── state.db
```

同步目录：
- config.yaml — 配置统一
- memory/MEMORY.md — L1 手写记忆
- memory/USER.md — L1 用户画像
- memory/persona.md — L4 画像
- skills/ — 技能同步

不同步：
- state.db — 会话历史各端独立
- logs/ — 日志各端独立
- .env — API key 各端独立（安全考虑）
- memory/l2/ — LanceDB 向量库各端独立（太大，同步慢）
- memory/l3/ — SQLite 会话归档各端独立

### 方案 B：飞书文档做共享记忆（零基础设施）

把 L1 记忆存在飞书文档里，两端都从飞书读取：

```
飞书知识库
├── 《Hermes 记忆》文档
│   ├── 用户偏好（L1 USER.md 内容）
│   ├── 规则约束（L1 MEMORY.md 内容）
│   └── 项目上下文
│
├── 云端 Agent → 飞书 API 读写
└── 本地 Agent → 飞书 API 读写
```

优点：不需要 Syncthing，飞书本身就是同步层
缺点：L2/L3 无法用这个方式同步

### 方案 C：混合（推荐实际使用）

```
L1（手写规则）  → 飞书文档 或 Syncthing
L2（语义记忆）  → 各端独立（通过 sync_turn 各自积累）
L3（会话归档）  → 各端独立
L4（画像）      → Syncthing 同步（persona.md 很小）
Skills          → Syncthing 同步
Config          → Syncthing 同步
```

## 飞书集成详细设计

### 飞书 Bot 作为控制面板

```
你 → 飞书发消息 → 飞书 Bot → 云端 Gateway
  │
  ├── /status     → 显示本地在线状态 + 云端状态
  ├── /local      → 强制本地执行
  ├── /cloud      → 强制云端执行
  ├── /auto       → 自动调度（默认）
  ├── /sync       → 手动触发同步
  ├── /memory     → 查看/编辑共享记忆
  └── 普通消息     → 自动调度执行
```

### 飞书消息格式

```
[自动调度 · 云端执行]
帮我查一下今天的天气

→ 云端 Agent 执行，结果回复到飞书

[自动调度 · 本地执行]
帮我打开 VS Code

→ SSH 到本地，执行 code 命令，结果回复到飞书
```

## 云端 Gateway 配置

```yaml
# 云端 ~/.hermes/config.yaml
memory:
  provider: governed

memory_governed:
  # 指向 Syncthing 同步目录
  l1_memory_path: /mnt/sync/hermes/memory/MEMORY.md
  l1_user_path: /mnt/sync/hermes/memory/USER.md
  l4_persona_path: /mnt/sync/hermes/memory/persona.md
  l2_db_path: /mnt/sync/hermes/memory/l2    # 或各端独立
  l3_db_path: ~/.hermes/memory/l3/l3.db     # 云端独立

terminal:
  backend: ssh        # 默认 SSH 到本地
  ssh:
    host: "Tailscale IP 或 内网穿透地址"
    user: "your-username"
    key: "~/.ssh/id_rsa"

gateway:
  platforms:
    feishu:
      enabled: true
      app_id: "你的飞书应用 ID"
      app_secret: "你的飞书应用密钥"
```

## 本地 Gateway 配置

```yaml
# 本地 ~/.hermes/config.yaml
memory:
  provider: governed

memory_governed:
  l1_memory_path: C:\Users\you\.hermes\memory\MEMORY.md   # 或 Syncthing 路径
  l1_user_path: C:\Users\you\.hermes\memory\USER.md
  l4_persona_path: C:\Users\you\.hermes\memory\persona.md
  l2_db_path: C:\Users\you\.hermes\memory\l2
  l3_db_path: C:\Users\you\.hermes\memory\l3\l3.db

terminal:
  backend: local      # 本地执行
```

## 任务调度器实现思路

云端 Gateway 收到消息后的决策逻辑：

```python
def dispatch_task(message, user):
    # 1. 检查本地在线状态
    local_online = check_local_heartbeat()

    # 2. 检查用户指定
    if user_wants_local(message):
        if local_online:
            return execute_on_local(message)
        else:
            return reply("本地机器离线，是否改在云端执行？")

    if user_wants_cloud(message):
        return execute_on_cloud(message)

    # 3. 自动调度
    if local_online:
        # 优先本地（有完整文件系统和浏览器）
        return execute_on_local(message)
    else:
        # 本地离线，云端执行
        return execute_on_cloud(message)

def execute_on_local(message):
    """SSH 到本地执行"""
    # 通过 SSH 发送任务到本地 Hermes
    # 本地 Agent 执行后返回结果
    result = ssh_execute(
        host=LOCAL_HOST,
        command=f"hermes chat -q '{message}'"
    )
    return format_response(result, source="local")

def execute_on_cloud(message):
    """云端自己执行"""
    # 云端 Agent 执行
    result = local_agent.chat(message)
    return format_response(result, source="cloud")
```

## 网络穿透方案（国内环境）

### 方案 1：Tailscale（推荐）

```
云端 VPS ←── Tailscale ──→ 本地 PC

优点：零配置 P2P，不需要公网 IP
缺点：需要安装 Tailscale 客户端
```

### 方案 2：FRP 内网穿透

```
云端 VPS（frps）←── FRP ──→ 本地 PC（frpc）

优点：完全自控
缺点：需要配置，稳定性依赖隧道
```

### 方案 3：Tailscale + 飞书 Webhook

```
本地 PC → Tailscale → 云端 VPS → 飞书 API
飞书消息 → 云端 VPS → Tailscale → 本地 PC
```

## 完整数据流

### 场景 1：本地在线，你在外面

```
你 → 飞书："帮我看看 D 盘的项目进度"
  │
  ├── 云端 Gateway 收到
  ├── 检查本地在线 → True
  ├── SSH 到本地：hermes chat -q "查看 D 盘项目进度"
  ├── 本地 Agent 执行（读取本地文件）
  ├── 结果通过 SSH 返回云端
  └── 云端转发到飞书 → 你看到回复
```

### 场景 2：本地关机，你在外面

```
你 → 飞书："帮我查一下明天的天气"
  │
  ├── 云端 Gateway 收到
  ├── 检查本地在线 → False
  ├── 云端 Agent 自己执行（web_search）
  └── 结果直接发到飞书 → 你看到回复 [云端执行]
```

### 场景 3：本地开机，自动同步

```
本地 PC 开机
  │
  ├── Syncthing 自动启动
  ├── 开始同步 config / memory / skills
  ├── 本地 Hermes 启动
  ├── 向云端注册：POST /api/local/register
  └── 云端更新 local_status = online

你 → 飞书："把刚才云端做的任务结果同步到本地"
  │
  ├── 云端检查本地在线 → True
  ├── 下发同步指令
  └── 本地 Agent 接收并执行
```

## 部署清单

### 云端 VPS（先买这个）

- [ ] 购买 VPS（推荐国内 2核4G，阿里云/腾讯云）
- [ ] 安装 Python + Hermes
- [ ] 配置飞书 Bot
- [ ] 安装 Syncthing
- [ ] 配置 SSH 密钥（用于连接本地）
- [ ] 配置 Tailscale（可选，用于内网穿透）
- [ ] 启动 hermes serve

### 本地 PC（已有）

- [ ] 安装 Syncthing
- [ ] 配置 Syncthing 对等节点（云端）
- [ ] 安装 Tailscale（可选）
- [ ] 配置 SSH Server（OpenSSH）
- [ ] 配置 memory-governed 插件
- [ ] 设置自动启动（开机自启 Hermes + Syncthing）

### 飞书 Bot

- [ ] 创建飞书企业应用
- [ ] 开启机器人能力
- [ ] 配置事件订阅（接收消息）
- [ ] 获取 app_id + app_secret
- [ ] 配置到 Hermes config.yaml
