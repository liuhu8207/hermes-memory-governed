#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Governed Memory CLI — access the SHARED hermes-memory-governed store.

Every command reads or writes the SAME on-disk store that Hermes uses, so any
agent (DeepSeek Harness, Hermes, CLI) sees the same L1 rules, L4 persona, L2
facts, L3 conversation archive, and Obsidian knowledge base.

This CLI is intentionally dependency-light: it reads L1/L4 as plain Markdown,
searches L3 via SQLite FTS5 (read-only, WAL-safe), keyword-scans L2 (LanceDB,
no embeddings needed), and reads/writes the Obsidian vault as Markdown. No
embedding model download, no API keys, no network.

Usage (run with the repo venv, i.e. `.venv\\Scripts\\python.exe`):

    python memory_cli.py recall <query> [--top-k N]
    python memory_cli.py kb-search <query> [--top-k N] [--section S]
    python memory_cli.py kb-get <title>
    python memory_cli.py kb-add <title> <body> [--section S] [--tags a b] [--concepts x y] [--confidence 0.8]
    python memory_cli.py l1                          # MEMORY.md + USER.md + persona.md
    python memory_cli.py health

Paths/config come from HERMES_HOME/governed_memory.json (default
D:\\Sync\\Drive\\项目\\github\\hermes-home).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_HERMES_HOME = r"D:\repos\Drive\项目\github\hermes-home"
VAULT_SUBDIRS = ["inbox", "notes", "projects", "areas", "resources", "archive"]

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{15,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9_]{15,}"),
    re.compile(r"(?i)(api[_-]?key|app[_-]?secret|token|password|passwd|pwd|secret)"
               r"\s*[:=]\s*([^\s,;，；]+)"),
]
_ILLEGAL_FS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# -- config -----------------------------------------------------------------
def hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or DEFAULT_HERMES_HOME


def load_config() -> dict:
    cfg_path = Path(hermes_home()) / "governed_memory.json"
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def wiki_dir(config: dict) -> Path:
    return Path(config.get("wiki_dir") or r"D:\repos\Drive\项目\github\wiki")


def read_text(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace").strip()


# -- L1 / L4 ----------------------------------------------------------------
def cmd_l1():
    h = hermes_home()
    return {
        "memory_rules_md": read_text(f"{h}/memory/MEMORY.md"),
        "user_profile_md": read_text(f"{h}/memory/USER.md"),
        "persona_md": read_text(f"{h}/memory/persona.md"),
        "note": "Standing human-authored rules/profile (L1/L4). Always honour them.",
    }


# -- L3: SQLite FTS5 (read-only, WAL-safe) ----------------------------------
def open_l3_ro():
    db = Path(hermes_home()) / "memory" / "l3" / "l3.db"
    if not db.exists():
        return None
    uri = "file:" + str(db).replace("\\", "/") + "?mode=ro&immutable=1"
    import sqlite3
    return sqlite3.connect(uri, uri=True)


def search_l3(query: str, top_k: int) -> list:
    conn = open_l3_ro()
    if conn is None:
        return []
    try:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
        if not tokens:
            return []
        results = []
        # 1) ranked FTS5 (helps English term/prefix matching)
        try:
            fts = " ".join('"%s"' % t for t in tokens)
            rows = conn.execute(
                "select rowid from messages_fts where messages_fts match ? limit ?",
                (fts, max(top_k * 2, 10)),
            ).fetchall()
            seen = set()
            for (rid,) in rows:
                if rid in seen:
                    continue
                seen.add(rid)
                row = conn.execute(
                    "select role, content, timestamp from messages where id=?",
                    (rid,),
                ).fetchone()
                if row:
                    results.append({"layer": "l3", "role": row[0],
                                    "content": (row[1] or "")[:600],
                                    "timestamp": row[2], "score": 1.0})
                if len(results) >= top_k:
                    return results
        except Exception:
            pass
        # 2) LIKE fallback/complement (reliable, esp. CJK)
        if len(results) < top_k:
            where = " OR ".join("content LIKE ?" for t in tokens)
            like = [f"%{t}%" for t in tokens]
            rows = conn.execute(
                f"select role, content, timestamp from messages where {where} "
                "order by id desc limit ?",
                [*like, top_k],
            ).fetchall()
            for role, content, ts in rows:
                results.append({"layer": "l3", "role": role, "content": (content or "")[:600],
                                "timestamp": ts, "score": 0.8})
        return results
    finally:
        try:
            conn.close()
        except Exception:
            pass


# -- L2: LanceDB keyword scan (no embeddings) -------------------------------
def search_l2(query: str, top_k: int) -> list:
    try:
        import lancedb
        l2 = Path(hermes_home()) / "memory" / "l2"
        if not l2.exists():
            return []
        db = lancedb.connect(str(l2))
        names = getattr(db, "list_tables", None)
        nlist = names() if callable(names) else db.table_names()
        if not isinstance(nlist, (list, tuple)):
            nlist = getattr(nlist, "tables", []) or [nlist]
        tables = [t.name if hasattr(t, "name") else str(t) for t in nlist]
        if "memories" not in tables:
            return []
        table = db.open_table("memories")
        arr = table.to_arrow()
        if "content" not in arr.column_names:
            return []
        contents = arr["content"].to_pylist()
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
        hits = []
        for c in contents:
            c = str(c or "")
            if not c:
                continue
            score = sum(1 for t in tokens if t.lower() in c.lower())
            if score > 0:
                hits.append({"layer": "l2", "content": c[:600], "score": 0.9})
                if len(hits) >= top_k:
                    break
        return hits
    except Exception:
        return []


# -- KB vault ---------------------------------------------------------------
def iter_notes(config: dict, subdirs: list = None) -> list:
    vault = wiki_dir(config)
    roots = [vault / s for s in (subdirs or VAULT_SUBDIRS)]
    out = []
    for root in roots:
        if root.exists():
            out.extend(sorted(p for p in root.rglob("*.md") if p.is_file()))
    return out


def parse_frontmatter(text: str):
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text
    close = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            close = i
            break
    if close is None:
        return {}, text
    raw = "\n".join(lines[1:close])
    body = "\n".join(lines[close + 1:]).lstrip("\n")
    data = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, _, v = line.partition(":")
        k, v = k.strip(), v.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                data[k] = json.loads(v)
            except Exception:
                data[k] = [x.strip().strip("'\"") for x in v[1:-1].split(",") if x.strip()]
        else:
            data[k] = v.strip("'\"")
    return data, body


def slugify(title: str) -> str:
    name = _ILLEGAL_FS.sub(" ", str(title or "")).strip()
    name = re.sub(r"\s+", " ", name).strip(" .")
    return (name[:120].strip(" .") or "untitled")


def dump_frontmatter(meta: dict) -> str:
    lines = ["---"]
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            items = [json.dumps(str(x), ensure_ascii=False) for x in v if str(x).strip()]
            lines.append(f"{k}: [{', '.join(items)}]")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif isinstance(v, (int, float)):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {json.dumps(str(v), ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines)


def cmd_kb_search(config: dict, query: str, top_k: int, section: str) -> list:
    subdirs = [section] if section else VAULT_SUBDIRS
    tokens = re.findall(r"[\w\u4e00-\u9fff]+", query)
    results = []
    for p in iter_notes(config, subdirs):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        title = meta.get("title") or p.stem
        hay = f"{title} {body}".lower()
        score = sum(1 for t in tokens if t.lower() in hay)
        if score > 0:
            snippet = re.sub(r"\s+", " ", body[:200])
            results.append({"title": title, "path": str(p.relative_to(wiki_dir(config))),
                            "section": p.parent.name, "score": score, "snippet": snippet})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def cmd_kb_get(config: dict, title: str):
    target = slugify(title).lower()
    for p in iter_notes(config):
        meta, body = parse_frontmatter(p.read_text(encoding="utf-8", errors="replace"))
        if (meta.get("title") or p.stem).lower() == target or p.stem.lower() == target:
            return {"title": meta.get("title") or p.stem, "path": str(p),
                    "meta": meta, "body": body}
    return None


def cmd_kb_add(config: dict, title: str, body: str, section: str,
               tags: list, concepts: list, confidence: float):
    for pat in SECRET_PATTERNS:
        if pat.search(body):
            return {"ok": False, "error": "refused: body looks like it contains a secret"}
    threshold = float(config.get("kb", {}).get("confidence_threshold", 0.7))
    if confidence is not None:
        section = "notes" if confidence >= threshold else "inbox"
    else:
        section = section or "notes"

    vault = wiki_dir(config)
    subdir = vault / section
    subdir.mkdir(parents=True, exist_ok=True)
    filename = slugify(title) + ".md"
    path = subdir / filename
    now = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    meta = {"title": title.strip(), "type": "note", "tags": tags, "concepts": concepts,
            "source": "dsh", "created": now, "updated": now}
    for c in concepts:
        if c and f"[[{c}]]" not in body:
            body = body.rstrip() + f"\n\nRelated: [[{c}]]" if "Related:" not in body else body
    text = dump_frontmatter(meta) + "\n\n" + body.rstrip() + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(subdir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"ok": True, "path": str(path), "section": section, "title": title.strip()}


# -- health ----------------------------------------------------------------
def cmd_health(config: dict):
    h = hermes_home()
    l2 = Path(h) / "memory" / "l2"
    l3 = Path(h) / "memory" / "l3" / "l3.db"
    l1 = cmd_l1()
    count_notes = len(iter_notes(config))
    return {
        "hermes_home": h,
        "l1_memory_md": bool(l1["memory_rules_md"]),
        "l1_user_md": bool(l1["user_profile_md"]),
        "l4_persona_md": bool(l1["persona_md"]),
        "l2_dir_exists": l2.exists(),
        "l3_db_exists": l3.exists(),
        "wiki_dir": str(wiki_dir(config)),
        "vault_note_count": count_notes,
        "ok": bool(l1["memory_rules_md"] or l1["user_profile_md"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Governed memory CLI (shared store)")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("recall", help="Search L1+L2+L3+L4 + KB")
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)

    ks = sub.add_parser("kb-search", help="Search the Obsidian knowledge base")
    ks.add_argument("query")
    ks.add_argument("--top-k", type=int, default=10)
    ks.add_argument("--section", default="")

    kg = sub.add_parser("kb-get", help="Read a full note by title")
    kg.add_argument("title")

    ka = sub.add_parser("kb-add", help="Write a durable note into the vault (governed)")
    ka.add_argument("title")
    ka.add_argument("body")
    ka.add_argument("--section", default="notes")
    ka.add_argument("--tags", nargs="*", default=[])
    ka.add_argument("--concepts", nargs="*", default=[])
    ka.add_argument("--confidence", type=float, default=None)

    sub.add_parser("l1", help="Print the standing L1 rules + L4 persona")
    sub.add_parser("health", help="Memory health report")

    args = parser.parse_args()
    config = load_config()

    if args.cmd == "l1":
        print(json.dumps(cmd_l1(), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "health":
        print(json.dumps(cmd_health(config), ensure_ascii=False, indent=2))
        return 0
    if args.cmd == "recall":
        out = {
            "query": args.query,
            "l1": cmd_l1(),
            "l2": search_l2(args.query, args.top_k),
            "l3": search_l3(args.query, args.top_k),
            "kb": cmd_kb_search(config, args.query, args.top_k, ""),
        }
    elif args.cmd == "kb-search":
        out = {"results": cmd_kb_search(config, args.query, args.top_k, args.section)}
    elif args.cmd == "kb-get":
        out = cmd_kb_get(config, args.title) or {"error": f"note not found: {args.title}"}
    elif args.cmd == "kb-add":
        out = cmd_kb_add(config, args.title, args.body, args.section,
                         args.tags, args.concepts, args.confidence)
    else:
        parser.print_help()
        return 2
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
