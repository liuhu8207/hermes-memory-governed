# -*- coding: utf-8 -*-
"""Obsidian vault store for the governed knowledge base.

vault（``config.wiki_dir``）是知识库的唯一**正本**：Markdown + frontmatter +
``[[wikilink]]``。向量索引（``_kb.KBIndex``）只是投影，可随时用
:meth:`KnowledgeBase.reindex` 从 vault 重建 —— 任何 agent、任何工具都能直接
读写 vault，不依赖本插件。

目录骨架（PARA + Zettelkasten 混合；缺哪个补哪个，绝不删除已有文件）::

    inbox/      待人工确认（半自动沉淀的低置信入口）
    notes/      原子笔记（Zettelkasten 卡片）
    projects/   PARA-Projects（有明确产出的任务）
    areas/      PARA-Areas（长期负责的领域）
    resources/  PARA-Resources（外部资料/参考）
    archive/    PARA-Archives（归档）
    index.md    MOC 导航入口

frontmatter 约定（Obsidian 原生识别）::

    title / type / tags / concepts / source / created / updated
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._diag import log_degraded

logger = logging.getLogger(__name__)

#: vault 的标准子目录（PARA + inbox + notes + knowledge 自动区）
#:
#: ``knowledge`` 是自动沉淀区（knowledge_collector/processor 的落点）。加入
#: 白名单只影响 ``kb-add`` 的 section 归属判定与 backlinks 范围 —— 检索侧
#: （``_kb.iter_vault_notes``）本来就全库扫描，不受此表影响。
VAULT_SUBDIRS = ["inbox", "notes", "projects", "areas", "resources", "archive",
                 "knowledge"]

#: Windows / 通用文件系统非法字符（文件名用）
_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

#: [[wikilink]] 匹配：[[Title]] / [[Title|alias]] / [[dir/Title]]
_WIKILINK = re.compile(r"\[\[([^\]\|#]+)(?:\|[^\]]*)?\]\]")

_INDEX_FILENAME = "index.md"


def vault_root(config) -> Path:
    """vault 根目录（``config.wiki_dir``）。"""
    return Path(config.wiki_dir)


def ensure_skeleton(vault: Path) -> Dict[str, Any]:
    """创建 vault 目录骨架与 index.md。幂等：已存在的不动。返回创建明细。"""
    created_dirs: List[str] = []
    for sub in VAULT_SUBDIRS:
        d = vault / sub
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created_dirs.append(sub)
    index_path = vault / _INDEX_FILENAME
    created_index = False
    if not index_path.exists():
        index_path.write_text(
            "# 知识库导航 (MOC)\n\n"
            "- [收件箱](inbox/) — 待确认的自动沉淀\n"
            "- [笔记](notes/) — 原子知识卡片\n"
            "- [项目](projects/) / [领域](areas/) / [资料](resources/) / [归档](archive/)\n",
            encoding="utf-8",
        )
        created_index = True
    return {"created_dirs": created_dirs, "created_index": created_index}


def slugify_filename(title: str) -> str:
    """标题 → 安全文件名（去扩展名）。保留中文；非法字符替换为空格。"""
    name = _ILLEGAL_FS.sub(" ", str(title or "")).strip()
    name = re.sub(r"\s+", " ", name).strip(" .")
    name = name[:120].strip(" .")
    return name or "untitled"


def note_title(path: Path) -> str:
    """笔记标题 = 文件名（Obsidian 默认 wikilink 解析目标）。"""
    return path.stem


def dump_frontmatter(meta: Dict[str, Any]) -> str:
    """frontmatter 序列化（YAML 子集：标量 + flow list，无第三方依赖）。

    值统一走 JSON 编码保证引号安全（Obsidian 的 YAML 解析器兼容 JSON 风格
    标量与 flow 序列）；日期等直接字符串也加引号，避免 ``2026-08-31`` 被
    解析器当日期对象后格式漂移。
    """
    lines = ["---"]
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            items = [json.dumps(str(v), ensure_ascii=False) for v in value if str(v).strip()]
            rendered = "[" + ", ".join(items) + "]"
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, (int, float)):
            rendered = json.dumps(value)
        else:
            rendered = json.dumps(str(value), ensure_ascii=False)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """解析 frontmatter；无 frontmatter 时返回 ({}, 原文)。

    首行围栏 ``---`` 之后，逐行找闭合围栏（允许首尾空白）——不能用
    ``split("\\n---")``：开头围栏前没有换行，分隔符只会命中闭合围栏，
    永远凑不齐三段。优先 PyYAML（兼容 Obsidian 手写的复杂 frontmatter）；
    不可用或解析失败时回退到内置的行式解析（标量 + flow list）。
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    close_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close_idx = i
            break
    if close_idx is None:
        return {}, text
    raw_fm = "\n".join(lines[1:close_idx])
    body = "\n".join(lines[close_idx + 1:]).lstrip("\n")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw_fm)
        if isinstance(data, dict):
            return data, body
    except Exception as e:  # noqa: BLE001 — 回退到内置解析
        log_degraded("kb_vault", "yaml_frontmatter_fallback", exc=e)
        pass
    data: Dict[str, Any] = {}
    for line in raw_fm.splitlines():
        line = line.strip()
        if not line or ":" not in line or line.startswith("#"):
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value.startswith("[") and raw_value.endswith("]"):
            try:
                data[key] = json.loads(raw_value)
                continue
            except Exception as e:  # noqa: BLE001
                log_degraded("kb_vault", "frontmatter_list_parse_failed", exc=e)
                inner = [v.strip().strip("'\"") for v in raw_value[1:-1].split(",") if v.strip()]
                data[key] = inner
                continue
        if raw_value.lower() in ("true", "false"):
            data[key] = raw_value.lower() == "true"
            continue
        try:
            data[key] = int(raw_value)
            continue
        except ValueError:  # silent-ok: value-fallback — 非整数标量回退为字符串
            pass
        try:
            data[key] = float(raw_value)
            continue
        except ValueError:  # silent-ok: value-fallback — 非浮点标量回退为字符串
            pass
        data[key] = raw_value.strip("'\"")
    return data, body


#: 受控分类词表（``category`` frontmatter 字段的唯一合法取值）。
#:
#: 为什么不在 ``tags`` 上做文章：``tags`` 是自由文本，实测库内中英混写、
#: 粒度不一（``["nas", "硬件", "飞牛", "i3-N305"]`` 里既有概念又有实例还有
#: 语言），既不能被人预测，也不能被机器判定归属。分类要可用，取值必须是
#: **封闭集合**：写的人从里面选，读的人知道会读到什么。
#:
#: 社区实践（Obsidian/PKM 圈的长期争论）收敛到同一条：纯文件夹会造出第二套
#: 与链接结构竞争的并行分类法，纯 tag 又不可预测且无法浏览；可维护的方案是
#: 「一个受控字段定类目 + 自动 MOC 按类目建索引」。这里就是那个受控字段，
#: MOC 侧见 ``KnowledgeBase.rebuild_moc``。
#:
#: 分类依据**用途**而非主题（"我在拿这条笔记做什么"），主题交给 tags/concepts：
#: 主题会无限增生，用途不会。新增类目请同步 ``docs/kb-taxonomy.md``。
KB_CATEGORIES = [
    "environment",   # 机器/网络/系统环境事实（本机、NAS、服务器、隧道）
    "user",          # 用户画像与偏好（城市、硬件倾向、工作流习惯）
    "tooling",       # 工具与服务的配置、排障（选型属于 reference）
    "workflow",      # 流程与方法（怎么做事、步骤、约定）
    "reference",     # 外部资料与规格（API 端点能力、硬件规格表）
    "project",       # 具体项目的状态与决策
    "synthesis",     # 跨笔记归纳卡（*_综合归纳）
    "meta",          # 关于这套记忆系统本身
]

#: ``type`` frontmatter 的旧值与新 ``category`` 的映射（迁移用）。
#: 旧值里 ``note``/``notes`` 只是 section 名重复，没有分类信息，不进映射。
_TYPE_TO_CATEGORY = {
    "category": "meta",
    "moc": "meta",
}

#: 从 tags/concepts/title 推断 category 的关键词表。
#: 顺序即优先级：先匹配到的赢。每条是 (category, [关键词小写])。
#: 受控分类的关键词证据表。
#:
#: 匹配规则分两种（见 :func:`_keyword_pattern`）：
#: * **ASCII 关键词**（``api`` / ``ip`` / ``nas``）走**词边界**匹配。
#:   裸子串匹配会让 ``script`` / ``recipe`` / ``skip`` / ``pipeline`` /
#:   ``shipping`` 全部因为含 "ip" 被判成 environment，``capital`` /
#:   ``apiary`` 因为含 "api" 被判成 tooling —— 英文笔记会大面积误分类。
#: * **CJK 关键词**（``硬件`` / ``配置``）走子串匹配：中文没有词边界，
#:   按词切分会漏掉「内存」这类词内命中。
_CATEGORY_HINTS: List[Tuple[str, List[str]]] = [
    ("synthesis", ["综合归纳"]),
    ("meta", ["hermes", "shared memory", "共享记忆", "hgm", "记忆系统",
              "governed", "plugin"]),
    ("reference", ["规格", "spec", "端点", "endpoint", "capability"]),
    # "配置" 单独出现时不作数：一篇讲 NAS 硬件事实的笔记标题里也有"配置"，
    # 但它描述的是环境，不是工具怎么配。要判 tooling，得是"某工具的配置"。
    ("tooling", ["dsh", "browser-skill", "bsk", "zcode", "mimo", "octop",
                 "api", "config", "troubleshooting", "upgrade",
                 "provider", "model", "gateway", "安装", "坑", "升级"]),
    ("user", ["用户", "偏好", "user", "preference", "location"]),
    ("environment", ["nas", "硬件", "hardware", "pc", "服务器", "server",
                     "network", "网络", "wireguard", "vpn", "routeros",
                     "homelab", "linux", "环境", "cpu", "内存", "存储",
                     "ip", "ssh"]),
    ("workflow", ["workflow", "流程", "方法", "约定", "步骤"]),
    ("project", ["project", "项目"]),
]

#: **标题标志词** —— 中文笔记标题是 ``<主体>_<属性>`` 结构，结尾的属性词说明
#: 这篇笔记**是什么类型**，而不是它讲什么。这比"哪个关键词更强"更接近真
#: 实判据：同一批词摆在不同位置含义完全不同 ——
#: ``hermes_综合归纳`` 结尾的「综合归纳」是类型（synthesis），
#: ``ZCode provider kind 配置`` 结尾的「配置」也是类型（tooling），
#: ``飞牛 NAS 硬件配置`` 结尾的「配置」却是内容（environment），
#: 区别在于它被「硬件」修饰成了「硬件配置」这个整体。
#:
#: 所以匹配规则是**最长匹配优先**：``硬件配置``(4 字) 先于 ``配置``(2 字)
#: 命中，于是前一篇落进 environment、后一篇落进 tooling —— 无需任何权重。
#:
#: 第一组是**无歧义**标志词（标题任意位置出现即可）；
#: 第二组是**歧义**标志词（只有位于标题**结尾**才算），因为它们在句中
#: 常常只是内容描述而非类型声明。
_TITLE_MARKERS: List[Tuple[str, str, bool]] = [
    # (标志词, 类目, 是否必须位于标题结尾)
    ("综合归纳", "synthesis", False),
    ("共享记忆", "meta", False),
    ("记忆系统", "meta", False),
    ("重启验证", "meta", False),
    ("默认模型", "meta", False),
    ("支持规格", "reference", False),
    ("配套规格", "reference", False),
    ("端点能力", "reference", False),
    ("知识库选型", "reference", False),
    ("硬件配置", "environment", False),
    ("系统环境", "environment", False),
    ("隧道拓扑", "environment", False),
    ("升级坑", "tooling", False),
    ("工作流偏好", "user", False),
    # 以下歧义：只在标题结尾才算类型声明
    ("配置", "tooling", True),
    ("安装", "tooling", True),
    ("升级", "tooling", True),
    ("开通", "meta", True),
    ("规格", "reference", True),
    ("选型", "reference", True),
]

#: 标题标志词按长度降序（最长匹配优先）。模块级常量，建一次即可。
_TITLE_MARKERS_SORTED = sorted(
    _TITLE_MARKERS, key=lambda m: -len(m[0]))

#: ASCII 关键词的词边界匹配缓存：``keyword -> compiled_pattern``（None = CJK）。
#: 关键词表是模块级常量，编译一次即可，不必每次推断都重编。
_KEYWORD_PATTERNS: Dict[str, Optional["re.Pattern[str]"]] = {}


def normalize_category(value: Any) -> str:
    """把任意输入归一成受控词表里的合法值；不合法/空 → ""。

    大小写与空白都容忍（``" Environment "`` → ``"environment"``）。
    返回 "" 表示「没有可用分类」，调用方按未分类处理 —— 绝不抛异常，
    也绝不猜一个塞回去（坏分类比没分类更糟：它会让 MOC 分组说谎）。
    """
    if not isinstance(value, str):
        return ""
    v = value.strip().lower()
    return v if v in KB_CATEGORIES else ""


def _keyword_pattern(keyword: str) -> "Optional[re.Pattern[str]]":
    """返回该关键词的匹配正则；CJK 关键词返回 None（走子串）。

    ASCII 关键词加 ``(?<![a-z0-9])`` / ``(?![a-z0-9])`` 前后哨兵：这样
    ``api`` 只在独立词位置命中，不会从 ``apiary`` / ``capital`` 里抠出来。
    连字符与空格（``browser-skill`` / ``provider kind``）由哨兵之外的字符
    放行，词内连字符不影响匹配。
    """
    if keyword in _KEYWORD_PATTERNS:
        return _KEYWORD_PATTERNS[keyword]
    pat: Optional[re.Pattern[str]] = None
    if keyword.isascii():
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(keyword)
                         + r"(?![a-z0-9])", re.IGNORECASE)
    _KEYWORD_PATTERNS[keyword] = pat
    return pat


#: 标题命中的权重倍数。标题是人写的命名约定，tags/concepts 更可能是
#: 自动抽取的噪声 —— 前者应该是更强的证据。
_TITLE_WEIGHT = 2


def _keyword_weight(keyword: str) -> int:
    """证据权重：越具体的关键词权重越高，封顶 12。

    长度是「具体程度」的可用代理 —— ``hermes``（6）比 ``api``（3）更像
    在说"关于这套系统"。只在标题没有标志词时作为**回退**判据使用。
    """
    return min(len(keyword), 12)


def _title_marker_category(title: str) -> str:
    """按标题标志词判定类目；无标志词 → ""。

    最长匹配优先，所以 ``硬件配置`` 永远压过 ``配置``。歧义标志词要求位于
    标题结尾（允许尾随空白），因为只有结尾才构成类型声明。
    """
    t = (title or "").strip()
    if not t:
        return ""
    low = t.lower()
    for marker, category, must_be_tail in _TITLE_MARKERS_SORTED:
        m = marker.lower()
        if must_be_tail:
            if low.endswith(m):
                return category
        elif m in low:
            return category
    return ""

def _category_scores(head: str, meta_text: str) -> List[Tuple[int, int, int, str]]:
    """关键词回退通道：给每个类目算「最强单条证据 + 累加总量」。

    只在**标题标志词**没给出答案时使用（见 :func:`_title_marker_category`）。

    排序键是**单条最强证据**（``max``）而不是累加总量 —— 一篇讲
    「Hermes 记忆系统的 API 端点」的笔记会在 meta / tooling / reference
    三家各命中一个弱词，累加到三家平手，而标题里那个决定性的
    「记忆系统」属于 meta。累加值只用来在最强证据**同分**时做区分。

    返回 ``(最强单条分, 累加总分, -书写顺序, category)``，降序。
    """
    scores: List[Tuple[int, int, int, str]] = []
    for order, (category, keywords) in enumerate(_CATEGORY_HINTS):
        best = 0
        total = 0
        for kw in keywords:
            low = kw.lower()
            pat = _keyword_pattern(low)
            base = _keyword_weight(low)
            hit = 0
            if pat is None:                      # CJK：子串匹配
                if low in head:
                    hit = base * _TITLE_WEIGHT
                elif low in meta_text:
                    hit = base
            else:                                # ASCII：词边界匹配
                if pat.search(head):
                    hit = base * _TITLE_WEIGHT
                elif pat.search(meta_text):
                    hit = base
            if hit:
                best = max(best, hit)
                total += hit
        if best:
            scores.append((best, total, -order, category))
    scores.sort(reverse=True)
    return scores


def infer_category(title: str, tags: Any, concepts: Any = None,
                   body: str = "") -> str:
    """从 title/tags/concepts 推断受控分类。推断不出 → ""（绝不硬猜）。

    **两道通道，标题标志词优先**：

    1. :func:`_title_marker_category` —— 标题结尾的属性词说明这篇笔记
       **是什么类型**（``hermes_综合归纳`` → synthesis、``ZCode ... 配置``
       → tooling）。这是最强的判据，因为它读的是标题的**构词法**。
    2. :func:`_category_scores` —— 关键词加权（标题权重 2 倍，ASCII 走词
       边界）作为回退，用于标题没有标志词的笔记（多为英文标题）。

    两道通道都判不出 → ""，绝不硬猜。

    ``body`` 只在 title/tags/concepts 全空时参与，且只取前 400 字 ——
    分类是导航属性，靠标题和元数据就该能定，正文只是最后的兜底。
    """
    def _as_list(v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        return []

    by_marker = _title_marker_category(title or "")
    if by_marker:
        return by_marker

    head = str(title or "").lower()
    meta_text = " ".join(_as_list(tags) + _as_list(concepts)).lower()
    if not head.strip() and not meta_text.strip():
        head = (body or "")[:400].lower()
    if not (head.strip() or meta_text.strip()):
        return ""

    scores = _category_scores(head, meta_text)
    return scores[0][3] if scores else ""


def write_text_atomic(path: Path, text: str) -> None:
    """原子写纯文本（temp + os.replace）。

    派生视图（如 ``index.md``）也会被多个 agent 同时重建；直接
    ``write_text`` 在写入中途崩溃时会让读者看到半张导航表。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # silent-ok: value-fallback — 临时文件善后不掩盖原异常
            pass
        raise


def write_note(path: Path, meta: Dict[str, Any], body: str) -> None:
    """原子写一篇笔记（temp + os.replace），frontmatter 在前。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = dump_frontmatter(meta) + "\n\n" + body.rstrip() + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:  # silent-ok: value-fallback — 临时文件善后失败不掩盖原异常（best-effort）
            pass
        raise


def read_note(path: Path) -> Tuple[Dict[str, Any], str]:
    """读取一篇笔记 → (frontmatter, body)。读失败返回 ({}, "")。"""
    try:
        return parse_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
    except OSError as e:  # silent-ok: value-fallback — 笔记读不到 → ({}, "")，per-note 遍历里跳过
        logger.debug("read_note failed for %s: %s", path, e)
        return {}, ""


def iter_notes(vault: Path, subdirs: Optional[List[str]] = None) -> List[Path]:
    """枚举 vault 下的 Markdown 笔记（默认 PARA 子目录，不含根 index.md）。"""
    roots = [vault / sub for sub in (subdirs or VAULT_SUBDIRS)]
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(sorted(p for p in root.rglob("*.md") if p.is_file()))
    return out


def append_link_section(body: str, links: List[str]) -> str:
    """把 ``[[wikilink]]`` 列表追加到正文（已有"相关笔记"段则并入）。"""
    clean = [str(l).strip() for l in links if str(l).strip()]
    if not clean:
        return body
    new_lines = [f"- [[{l}]]" for l in clean]
    if "## 相关笔记" in body:
        return body.rstrip() + "\n" + "\n".join(new_lines) + "\n"
    return body.rstrip() + "\n\n## 相关笔记\n" + "\n".join(new_lines) + "\n"


def backlinks_for(vault: Path, title: str, subdirs: Optional[List[str]] = None) -> List[str]:
    """找所有链接到 *title* 的笔记相对路径（``[[title]]`` / ``[[title|alias]]``）。

    线性扫描即可：个人 vault 规模（数千篇内）毫秒级；未来规模大了再换
    knowledge_graph.py 的索引。
    """
    target = title.strip()
    if not target:
        return []
    vault = Path(vault)
    hits: List[str] = []
    for path in iter_notes(vault, subdirs):
        if note_title(path) == target:
            continue  # 自链不算反链
        _, body = read_note(path)
        for m in _WIKILINK.finditer(body):
            link_target = m.group(1).strip().split("/")[-1]
            if link_target == target:
                hits.append(str(path.relative_to(vault)).replace("\\", "/"))
                break
    return hits


def now_iso() -> str:
    """ISO 时间戳（frontmatter created/updated 用，秒级足够）。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
