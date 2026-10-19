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
    "tooling",       # 工具与服务的配置、选型、排障（Hermes、dsh、browser-skill）
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
_CATEGORY_HINTS: List[Tuple[str, List[str]]] = [
    # 顺序即优先级：先匹配到的赢。顺序经过真实库标定 —— 把强信号放前面，
    # 弱信号（"api" 这种到处都有的词）放后面，否则一篇讲某工具怎么装的笔记
    # 会因为正文里出现过 "api" 被判成 reference。
    ("synthesis", ["综合归纳", "synthesis"]),
    ("meta", ["hermes", "shared memory", "共享记忆", "hgm", "记忆系统",
              "governed", "plugin"]),
    ("reference", ["规格", "spec", "端点", "endpoint", "支持规格", "能力",
                   "capability", "选型"]),
    # "配置" 单独出现时不作数：一篇讲 NAS 硬件事实的笔记标题里也有"配置"，
    # 但它描述的是环境，不是工具怎么配。要判 tooling，得是"某工具的配置"。
    ("tooling", ["dsh", "browser-skill", "bsk", "zcode", "mimo", "octop",
                 "api", "config", "troubleshooting", "upgrade",
                 "provider", "model", "gateway", "安装", "坑", "升级",
                 "配置：", "配置:", "provider kind", "改配置"]),
    ("user", ["用户", "偏好", "user", "preference", "工作流偏好",
              "地理位置", "location"]),
    ("environment", ["nas", "硬件", "hardware", "pc", "服务器", "server",
                     "network", "网络", "wireguard", "vpn", "routeros",
                     "homelab", "linux", "环境", "cpu", "内存", "存储",
                     "ip", "ssh"]),
    ("workflow", ["workflow", "工作流", "流程", "方法", "约定", "步骤"]),
    ("project", ["project", "项目"]),
]

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


def infer_category(title: str, tags: Any, concepts: Any = None,
                   body: str = "") -> str:
    """从 title/tags/concepts 推断受控分类。推断不出 → ""（绝不硬猜）。

    关键词全部小写比对，因此中文 tag 原样命中、英文 tag 大小写无关。
    ``body`` 只在 title/tags/concepts 都没命中时参与，且只取前 400 字 ——
    分类是导航属性，靠标题和元数据就该能定，正文只是最后的兜底。
    """
    def _as_list(v: Any) -> List[str]:
        if isinstance(v, str):
            return [v]
        if isinstance(v, (list, tuple, set)):
            return [str(x) for x in v]
        return []

    hay = " ".join(_as_list(tags) + _as_list(concepts) + [title or ""]).lower()
    if not hay.strip():
        hay = (body or "")[:400].lower()

    for category, keywords in _CATEGORY_HINTS:
        for kw in keywords:
            if kw.lower() in hay:
                return category
    return ""


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
