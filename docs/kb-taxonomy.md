# KB 分类词表

`category` 是知识库唯一的受控分类字段。取值封闭，写的人从里面选，读的人
知道会读到什么。

## 为什么不是更多文件夹，也不是更多 tag

社区（Obsidian / PKM 圈）对这件事争论了很多年，收敛到的结论是两条路各有
一个结构性缺陷：

- **文件夹**只能表达一个归属，而一篇「飞牛 NAS 硬件配置」同时是环境事实和
  硬件规格。硬要二选一，另一条检索路径就断了。更糟的是文件夹会造出**第二套
  与链接结构竞争的并行分类法**，于是你要同时维护两套分类。
- **tag** 的问题是「未来的你能预测它吗」。实测库内 `tags` 是中英混写、粒度
  不一：`["nas", "硬件", "飞牛", "i3-N305"]` 里既有概念、又有实例、还有语言。
  这种自由文本既不能被人预测，也不能被机器判定归属。

所以这里用**一个受控字段定类目 + 自动 MOC 按类目建索引**：分类可判定，导航
可浏览，一篇笔记可以出现在多个类目下，且整张导航由一个命令重建。

## 词表

分类依据**用途**而非主题（"我在拿这条笔记做什么"）。主题会无限增生，用途不会。

| 值 | 含义 | 库内实例 |
|----|------|----------|
| `environment` | 机器 / 网络 / 系统环境事实 | 飞牛 NAS 硬件配置、本地 PC 硬件、WireGuard 隧道拓扑 |
| `user` | 用户画像与偏好 | 用户所在城市、用户工作流偏好 |
| `tooling` | 工具与服务的配置、选型、排障 | dsh 升级、browser-skill 安装坑、ZCode provider kind |
| `workflow` | 流程与方法 | 怎么做一件事、步骤、约定 |
| `reference` | 外部资料与规格 | MiMo API 端点能力、SR588 CPU 支持规格 |
| `project` | 具体项目的状态与决策 | |
| `synthesis` | 跨笔记归纳卡 | `*_综合归纳` |
| `meta` | 关于这套记忆系统本身 | 共享记忆系统开通、Hermes Gateway 重启方法 |

主题信息仍然写在 `tags` / `concepts` 里 —— 那两个字段管"这条笔记讲什么"，
`category` 管"我拿它做什么"。两者不重叠。

## 怎么用

```bash
# 写入时显式指定（优先于推断）
python memory_cli.py kb-add "标题" "正文" --category tooling

# 不传则按 title/tags/concepts 推断；推断不出就整个键不写
python memory_cli.py kb-add "标题" "正文"

# 给存量笔记补分类（默认 dry-run，先看 scope）
python memory_cli.py kb-migrate-category
python memory_cli.py kb-migrate-category --apply --yes

# 重建 index.md 导航（MOC）
python memory_cli.py kb-moc --yes
```

## 三条不妥协的规则

1. **非法值丢弃，不硬猜。** `--category NAS` 会被归一成空，然后走推断。
   一个错分类比没有分类更糟：它看起来像结论。
2. **人的判断压过机器的推断。** 已有合法 `category` 的笔记，迁移一律不动。
3. **归档区不是分类区。** `archive/` 不参与 MOC 分组 —— 它是淘汰区。

## 新增类目

改 `plugin/memory_governed/_vault.py` 的 `KB_CATEGORIES` 和 `_CATEGORY_HINTS`，
**并同步 `memory_cli.py` 里的同源副本**（CLI 不能依赖插件加载，因此有两份实现）。
`tests/test_kb_category_moc.py` 会钉住两侧一致 —— 一个入口给一种分类，等于同一篇
笔记有两个真相。改完把新类目补进上表。
