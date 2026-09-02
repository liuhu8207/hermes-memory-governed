# -*- coding: utf-8 -*-
"""Knowledge base over the Obsidian vault (检索 / 写入 / 索引投影).

正本 = vault（``_vault.py`` 的 Markdown + frontmatter + ``[[wikilink]]``）。
本模块提供两层：

- :class:`KBIndex` —— 向量索引**投影**（LanceDB + ``EmbeddingService``）。
  可选、可重建；lancedb / embedding 任一不可用时优雅降级，绝不抛异常。
- :class:`KnowledgeBase` —— 门面：检索（语义 + 关键词 + 反链融合）、写入、
  更新、读取、列举、重建索引。

设计要点：

- **关键词检索始终可用**（纯文件扫描，零第三方依赖）——即使向量后端缺失，
  ``search`` 也能返回结果；语义检索是增强而非必需。
- 中文查询没有空格分词，关键词打分用 bigram 重叠做兜底（title/tags/concepts
  短文本上做），整串命中在 body 上做 ``in`` 匹配。
- 写入时对 ``concepts`` 里的词在正文首次出现处自动加 ``[[词]]`` 双向链接，
  这是「Obsidian 双向链接利用起来」的核心动作。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ._config import GovernedMemoryConfig
from ._embedding import EmbeddingService
from ._bridge import has_secret_like_text
from ._recall import row_to_score
from . import _vault

logger = logging.getLogger(__name__)

#: [[Title]] / [[Title|alias]] / [[dir/Title]] / [[Title#heading]]
_LINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:\|[^\]]*)?\]\]")

#: 索引表名（LanceDB）
_TABLE_NAME = "kb_notes"


def _as_list(value: Any) -> List[str]:
    """归一化 frontmatter 的 tags/concepts 到 List[str]。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _sql_literal(s: str) -> str:
    """SQL 字符串字面量（单引号转义为双单引号）。"""
    return "'" + s.replace("'", "''") + "'"


def _bigrams(s: str) -> set:
    """字符串的 2-gram 集合（中文连续查询的关键词兜底）。"""
    s = s.lower()
    return {s[i:i + 2] for i in range(len(s) - 1)}


class _Note:
    """一篇笔记的内存表示。"""

    __slots__ = ("path", "title", "meta", "body")

    def __init__(self, path: Path, title: str, meta: Dict[str, Any], body: str):
        self.path = path
        self.title = title
        self.meta = meta
        self.body = body

    @property
    def section(self) -> str:
        """相对 vault 的第一段目录名（inbox/notes/projects/...）。"""
        rel = self.path.parent
        return rel.name if rel.name else ""

    @property
    def tags(self) -> List[str]:
        return _as_list(self.meta.get("tags"))

    @property
    def concepts(self) -> List[str]:
        return _as_list(self.meta.get("concepts"))

    @property
    def short_text(self) -> str:
        """用于 bigram 命中的短文本（标题 + 标签 + 概念）。"""
        return " ".join([self.title] + self.tags + self.concepts)

    @property
    def full_text(self) -> str:
        """用于整串命中与向量索引的全文。"""
        return "\n".join([self.title, self.short_text, self.body])


class KBIndex:
    """向量索引投影层（可选）。不可用时 ``available=False`` 并记录原因。"""

    def __init__(self, config: GovernedMemoryConfig, db_path: Path):
        self._config = config
        self._db_path = db_path
        self._service = None
        self._store = None
        self._available = False
        self._last_error = ""
        self._init()

    # -- 初始化 ------------------------------------------------------

    def _init(self) -> None:
        try:
            self._service = EmbeddingService.get(self._config)
        except Exception as e:  # noqa: BLE001 — 索引只是投影，失败不能影响主流程
            self._last_error = f"embedding: {e}"
            return
        if self._service is None or not getattr(self._service, "available", False):
            self._last_error = "embedding unavailable"
            return
        try:
            import lancedb  # noqa: F401
            import pyarrow as pa  # noqa: F401
        except ImportError as e:
            self._last_error = f"lancedb missing: {e.name}"
            return
        try:
            self._db_path.mkdir(parents=True, exist_ok=True)
            db = lancedb.connect(str(self._db_path))
            names = db.table_names()
            if _TABLE_NAME in names:
                self._store = db.open_table(_TABLE_NAME)
            else:
                dim = int(getattr(self._service, "dim", 0) or self._config.vector.dim)
                schema = pa.schema([
                    pa.field("path", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ])
                self._store = db.create_table(_TABLE_NAME, schema=schema)
            self._available = True
        except Exception as e:  # noqa: BLE001
            self._last_error = f"lancedb: {e}"
            self._store = None
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def last_error(self) -> str:
        return self._last_error

    # -- 写 ----------------------------------------------------------

    def upsert(self, path: str, text: str) -> None:
        """写入/更新一条向量。失败仅 debug 记录，不抛。"""
        if not self._available or self._service is None:
            return
        vec = self._service.embed_one(text)
        if vec is None:
            return
        try:
            self.delete(path)
        except Exception:  # noqa: BLE001 — 删不掉就靠搜索侧去重兜底
            pass
        try:
            self._store.add([{"path": path, "text": text, "vector": vec}])
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index upsert failed for %s: %s", path, e)

    def delete(self, path: str) -> None:
        if not self._available or self._store is None:
            return
        self._store.delete(where=f"path = {_sql_literal(path)}")

    def drop(self) -> None:
        """清空并重建索引表（reindex 用）。"""
        if not self._available:
            return
        import lancedb  # noqa: F401
        try:
            db = lancedb.connect(str(self._db_path))
            if _TABLE_NAME in db.table_names():
                db.drop_table(_TABLE_NAME)
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index drop failed: %s", e)
        self._available = False
        self._init()

    # -- 读 ----------------------------------------------------------

    def search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """语义检索；返回 [{path, text, score, kind='semantic'}]。失败返回 []。"""
        if not self._available or self._service is None:
            return []
        vec = self._service.embed_one(query)
        if vec is None:
            return []
        try:
            rows = (
                self._store.search(vec)
                .metric("cosine")
                .limit(max(1, k))
                .to_list()
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("kb index search failed: %s", e)
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            out.append({
                "path": r.get("path", ""),
                "text": r.get("text", ""),
                # 复用 _recall 的换算，避免两份实现漂移（cosine distance ∈ [0,2]）
                "score": row_to_score(r),
                "kind": "semantic",
            })
        return out


class KnowledgeBase:
    """知识库门面：vault 检索 + 写入 + 索引投影。"""

    def __init__(self, config: GovernedMemoryConfig):
        self._config = config
        self._vault = _vault.vault_root(config)
        # 独立向量库目录（与 L2 记忆的 l2/ 分开）
        self._index_db = Path(config.l2_db_path).parent / "kb_index"
        self._index: Optional[KBIndex] = None

    # -- 骨架 / 索引 -------------------------------------------------

    def ensure(self) -> Dict[str, Any]:
        """建 vault 骨架（幂等）。"""
        return _vault.ensure_skeleton(self._vault)

    def _index_get(self) -> KBIndex:
        if self._index is None:
            self._index = KBIndex(self._config, self._index_db)
        return self._index

    @property
    def index_available(self) -> bool:
        return self._index_get().available

    def reindex(self) -> Dict[str, Any]:
        """从 vault 重建向量索引（丢索引后的恢复手段）。"""
        idx = self._index_get()
        if not idx.available:
            return {"ok": False, "reason": idx.last_error or "index unavailable"}
        idx.drop()
        n = 0
        for note in self._iter_notes():
            idx.upsert(self._rel(note.path), note.full_text)
            n += 1
        return {"ok": True, "indexed": n}

    # -- 读取 --------------------------------------------------------

    def _iter_notes(self) -> List[_Note]:
        notes: List[_Note] = []
        if not self._vault.exists():
            return notes
        for path in sorted(self._vault.rglob("*.md")):
            if ".obsidian" in path.parts:
                continue
            if path.name == "index.md":  # MOC 导航不参与检索
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            meta, body = _vault.parse_frontmatter(text)
            notes.append(_Note(path, _vault.note_title(path), meta, body))
        return notes

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._vault)).replace("\\", "/")
        except ValueError:
            return str(path)

    def get(self, title: str) -> Optional[Dict[str, Any]]:
        """按标题读单篇（返回全文）。"""
        for note in self._iter_notes():
            if note.title == title or note.path.stem == title:
                return {
                    "title": note.title,
                    "path": self._rel(note.path),
                    "section": note.section,
                    "meta": note.meta,
                    "body": note.body,
                }
        return None

    def list_notes(self, section: str = "") -> List[Dict[str, Any]]:
        """列举笔记（可按 section 过滤）。"""
        out = []
        for note in self._iter_notes():
            if section and note.section != section:
                continue
            out.append({
                "title": note.title,
                "path": self._rel(note.path),
                "section": note.section,
                "tags": note.tags,
                "concepts": note.concepts,
            })
        return out

    def links(self, title: str) -> List[str]:
        """反链：谁链接了标题为 *title* 的笔记。"""
        back: List[str] = []
        for note in self._iter_notes():
            for m in _LINK_RE.finditer(note.body):
                target = m.group(1).split("/")[-1].strip()
                if target == title:
                    back.append(self._rel(note.path))
                    break
        return back

    # -- 检索 --------------------------------------------------------

    @staticmethod
    def _keyword_score(query: str, note: _Note) -> float:
        """关键词打分（中文 bigram 兜底 + 整串命中）。"""
        q = query.strip().lower()
        if not q:
            return 0.0
        short = note.short_text.lower()
        body_l = note.body.lower()
        title = note.title.lower()
        score = 0.0
        if q in title:
            score += 5.0
        if q in short:
            score += 3.0
        if q in body_l:
            score += 2.0
        qg = _bigrams(q)
        if qg:
            sg = _bigrams(short)
            if sg:
                score += len(qg & sg) / len(qg) * 2.0
        return score

    def search(self, query: str, top_k: int = 10, section: str = "") -> List[Dict[str, Any]]:
        """融合检索：关键词（必有） + 语义（可选） + 反链统计。"""
        if not query.strip():
            return []
        top_k = max(1, min(int(top_k), 50))
        notes = self._iter_notes()
        if section:
            notes = [n for n in notes if n.section == section]

        # 关键词打分（始终可用）
        scored: Dict[str, Dict[str, Any]] = {}
        for note in notes:
            ks = self._keyword_score(query, note)
            if ks <= 0.0:
                continue
            scored[self._rel(note.path)] = {
                "title": note.title,
                "path": self._rel(note.path),
                "section": note.section,
                "score": ks,
                "kind": "keyword",
                "tags": note.tags,
                "concepts": note.concepts,
                "snippet": note.body[:160].strip(),
            }

        # 语义检索（可选增强；按 path 去重合并，取更高分）
        idx = self._index_get()
        if idx.available:
            for r in idx.search(query, top_k):
                p = r["path"]
                if p in scored:
                    if r["score"] > scored[p]["score"]:
                        scored[p]["score"] = r["score"]
                        scored[p]["kind"] = "semantic"
                else:
                    # 语义命中但关键词没中：补一个骨架（正文从索引 text 截取）
                    scored[p] = {
                        "title": Path(p).stem,
                        "path": p,
                        "section": Path(p).parent.name,
                        "score": r["score"],
                        "kind": "semantic",
                        "tags": [],
                        "concepts": [],
                        "snippet": r["text"][:160].strip(),
                    }

        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        # 反链统计（只对 top_k 结果补充，避免全库扫描放大）
        backlink_cache: Dict[str, List[str]] = {}
        for item in ranked:
            title = item["title"]
            if title not in backlink_cache:
                backlink_cache[title] = self.links(title)
            item["backlinks"] = backlink_cache[title]
        return ranked

    # -- 写入 --------------------------------------------------------

    @staticmethod
    def _autolink(body: str, concepts: List[str]) -> str:
        """对 concepts 里的词在正文首次出现处加 [[词]]（已链接的跳过）。"""
        for c in concepts:
            c = c.strip()
            if not c:
                continue
            if re.search(rf"\[\[{re.escape(c)}(\||\])", body):
                continue
            idx = body.find(c)
            if idx != -1:
                body = body[:idx] + f"[[{c}]]" + body[idx + len(c):]
        return body

    @staticmethod
    def _link_if_missing(body: str, target: str) -> Tuple[str, bool]:
        """若 body 尚无 ``[[target]]``（或 ``[[target|alias]]``），追加到「相关笔记」段。

        返回 ``(new_body, added)``；已存在则原样返回且 added=False。
        """
        target = (target or "").strip()
        if not target:
            return body, False
        if re.search(rf"\[\[{re.escape(target)}(\||\])", body):
            return body, False
        return _vault.append_link_section(body, [target]), True

    def _link_related(
        self,
        path: Path,
        title: str,
        meta: Dict[str, Any],
        body: str,
        concepts: List[str],
        tags: List[str],
    ) -> int:
        """双向链接补全：为刚写入的笔记找共享 concepts/tags 的笔记并互加
        ``[[wikilink]]``（本笔记 → 对方，对方 → 本笔记）。返回新增链接数。

        线性扫描即可：个人 vault（数千篇内）毫秒级；未来规模大了再换索引。
        """
        concept_set = {c.strip().lower() for c in concepts if c.strip()}
        tag_set = {t.strip().lower() for t in tags if t.strip()}
        if not concept_set and not tag_set:
            return 0

        related: List[_Note] = []
        for note in self._iter_notes():
            if note.title == title:
                continue
            nc = {c.strip().lower() for c in note.concepts if c.strip()}
            nt = {t.strip().lower() for t in note.tags if t.strip()}
            if (concept_set & nc) or (tag_set & nt):
                related.append(note)
        if not related:
            return 0

        added = 0
        # 反向：把本笔记标题加进每个相关笔记
        for note in related:
            ometa, obody = _vault.read_note(note.path)
            nobody, ok = self._link_if_missing(obody, title)
            if ok:
                _vault.write_note(note.path, ometa, nobody)
                self._index_get().upsert(
                    self._rel(note.path),
                    _Note(note.path, note.title, ometa, nobody).full_text,
                )
                added += 1

        # 正向：把相关笔记标题加进本笔记
        pending = [
            n.title
            for n in related
            if not re.search(rf"\[\[{re.escape(n.title)}(\||\])", body)
        ]
        if pending:
            nbody = _vault.append_link_section(body, pending)
            _vault.write_note(path, meta, nbody)
            self._index_get().upsert(
                self._rel(path), _Note(path, title, meta, nbody).full_text
            )
            added += len(pending)
        return added

    def add(
        self,
        title: str,
        body: str,
        section: str = "notes",
        tags: Optional[List[str]] = None,
        concepts: Optional[List[str]] = None,
        source: str = "",
        update: bool = False,
        confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        """写入/更新一篇笔记到 vault，并更新向量索引。

        ``section`` 必须在标准子目录内，否则回退到 ``inbox``（半自动沉淀的
        安全落点，不会污染目录结构）。

        ``confidence``（0-1）是半自动沉淀的置信信号：传了它，落库分区由
        阈值决定 —— ``>= kb.confidence_threshold`` 进 ``notes``（正库），
        低于阈值进 ``inbox``（frontmatter 标 ``review_required``）；不传则
        按显式 ``section`` 落库（手动场景，行为不变）。

        落库前对 title/body 过密钥闸门（fail-closed）：含密钥一律拒绝，
        绝不写入 vault。
        """
        title = (title or "").strip() or "untitled"
        body = (body or "").strip()
        concepts = _as_list(concepts)
        tags = _as_list(tags)

        # 密钥闸门（fail-closed）：先于任何写盘
        if has_secret_like_text(title) or has_secret_like_text(body):
            return {
                "ok": False,
                "error": "content contains secret-like text; refused (fail-closed)",
            }

        # 置信分流：传了 confidence 就由阈值接管 notes/inbox
        review_required = False
        if confidence is not None:
            try:
                conf = float(confidence)
            except (TypeError, ValueError):
                conf = None
            if conf is not None:
                threshold = float(
                    getattr(self._config.kb, "confidence_threshold", 0.7) or 0.7
                )
                if conf < threshold:
                    section = "inbox"
                    review_required = True
                else:
                    section = "notes"

        if section not in _vault.VAULT_SUBDIRS:
            section = "inbox"
        body = self._autolink(body, concepts)

        filename = _vault.slugify_filename(title)
        path = self._vault / section / f"{filename}.md"

        now = time_str()
        meta: Dict[str, Any] = {
            "title": title,
            "type": section,
            "tags": tags,
            "concepts": concepts,
            "source": source or "",
            "created": now,
            "updated": now,
        }
        if confidence is not None:
            meta["confidence"] = float(confidence)
        if review_required:
            meta["review_required"] = True
        if update:
            existing = self.get(title)
            if existing:
                meta["created"] = existing["meta"].get("created", now)

        _vault.write_note(path, meta, body)

        # 更新向量索引（尽力而为）；索引路径用相对 vault 的路径，与检索去重键一致
        note = _Note(path, title, meta, body)
        self._index_get().upsert(self._rel(path), note.full_text)

        # 链接补全：只对正库 notes 做（inbox 待审先不互链，避免噪音）
        links_added = 0
        if section == "notes" and getattr(self._config.kb, "autolink_related", True):
            links_added = self._link_related(path, title, meta, body, concepts, tags)

        return {
            "ok": True,
            "title": title,
            "path": self._rel(path),
            "section": section,
            "tags": tags,
            "concepts": concepts,
            "review_required": review_required,
            "links_added": links_added,
        }

    def list_review(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出 inbox 待审笔记（``section=inbox`` 或 ``review_required``）。"""
        out: List[Dict[str, Any]] = []
        for note in self._iter_notes():
            review_required = bool(note.meta.get("review_required", False))
            if note.section != "inbox" and not review_required:
                continue
            out.append({
                "title": note.title,
                "path": self._rel(note.path),
                "section": note.section,
                "confidence": note.meta.get("confidence"),
                "review_required": review_required,
                "created": note.meta.get("created", ""),
                "updated": note.meta.get("updated", ""),
                "preview": note.body[:200],
            })
            if len(out) >= limit:
                break
        return out

    def _relocate(
        self,
        title: str,
        target_section: str,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """把 inbox 笔记移到 target_section（改 frontmatter + 移文件 + 更新索引）。

        仅允许从 ``inbox`` 移出；目标已存在同名文件时拒绝（不覆盖）。
        """
        for note in self._iter_notes():
            if note.title != title and note.path.stem != title:
                continue
            if note.section != "inbox":
                return {
                    "ok": False,
                    "error": f"Note '{title}' is not in inbox (section={note.section})",
                }
            new_path = self._vault / target_section / note.path.name
            if new_path != note.path and new_path.exists():
                return {
                    "ok": False,
                    "error": f"Target already exists: {self._rel(new_path)}",
                }

            meta = dict(note.meta)
            meta["type"] = target_section
            meta["updated"] = time_str()
            meta.pop("review_required", None)
            if extra_meta:
                meta.update(extra_meta)

            body = note.body
            _vault.write_note(new_path, meta, body)
            if new_path != note.path:
                try:
                    note.path.unlink()
                except OSError as e:  # noqa: BLE001
                    logger.warning("unlink old note failed for %s: %s", note.path, e)

            idx = self._index_get()
            idx.delete(self._rel(note.path))
            idx.upsert(self._rel(new_path), _Note(new_path, title, meta, body).full_text)

            return {
                "ok": True,
                "title": title,
                "path": self._rel(new_path),
                "section": target_section,
                "from": note.section,
            }
        return {"ok": False, "error": f"Note not found: {title}"}

    def approve(self, title: str) -> Dict[str, Any]:
        """批准 inbox 待审笔记 → 移入 notes，并补双向链接。"""
        r = self._relocate(title, "notes")
        if not r.get("ok"):
            return r
        links_added = 0
        note = self.get(title)
        if note and getattr(self._config.kb, "autolink_related", True):
            concepts = _as_list(note.get("meta", {}).get("concepts"))
            tags = _as_list(note.get("meta", {}).get("tags"))
            if concepts or tags:
                abs_path = self._vault / Path(note["path"])
                links_added = self._link_related(
                    abs_path,
                    title,
                    note.get("meta", {}),
                    note.get("body", ""),
                    concepts,
                    tags,
                )
        r["links_added"] = links_added
        return r

    def reject(self, title: str) -> Dict[str, Any]:
        """拒绝 inbox 待审笔记 → 移入 archive（标 review_rejected）。"""
        return self._relocate(title, "archive", {"review_rejected": True})

    def stats(self) -> Dict[str, Any]:
        """知识库统计（供 health 面板）。"""
        notes = self._iter_notes()
        by_section: Dict[str, int] = {}
        for n in notes:
            by_section[n.section] = by_section.get(n.section, 0) + 1
        idx = self._index_get()
        return {
            "total": len(notes),
            "by_section": by_section,
            "index_available": idx.available,
            "index_error": idx.last_error or "",
        }


def time_str() -> str:
    """ISO 日期（frontmatter created/updated 用）。"""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d")
