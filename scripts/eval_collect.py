#!/usr/bin/env python3
"""Collect candidate evaluation queries from the L3 conversation archive.

The point is to evaluate retrieval against **how the user actually asks**, not
against note titles. This is read-only: it never writes to L3 and never sends
anything off the machine.

除了收集，本脚本还负责**入库前的那两道闸**（守则见 ``evals/README.md``）：

1. **去重**：查询串归一化（大小写、全半角、标点空白、实体、时间词）后同键
   只留一条；跨集也要去，种子集优先；同一目标标题最多贡献 N 条查询 ——
   否则一条笔记就能独占整个指标。
2. **脱敏**：把主机名 / IP / 路径 / 用户名 / 项目代号换成占位符后再入库。
   真实集之所以不提交，原因是**含个人上下文与凭据片段**，不是体积；脱敏
   版与原版必须一一对应且仅差占位符（脱敏不能改难度）。

Usage:
    python scripts/eval_collect.py --days 14
    python scripts/eval_collect.py --out D:/path/candidates.jsonl
    python scripts/eval_collect.py --input gold.json --dedupe-against seed.json \
        --desensitize-map entity_map.json --out gold.desensitized.json
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hermes_env import bootstrap, env_path

HERMES_HOME = bootstrap(__file__)
MEMORY_DIR = env_path("HERMES_MEMORY_DIR", HERMES_HOME / "memory")
L3_DB_PATH = env_path("L3_DB_PATH", MEMORY_DIR / "l3" / "l3.db")

_SKIP = re.compile(r"^(/|!|python |pip |git |cd |ls |dir )", re.IGNORECASE)

# -- 归一化（去重键） -------------------------------------------------------
# 实体先去掉：主机名 / IP / 路径 / 邮箱 / 网址 / 版本号 / 日期时间。
# 顺序有意义：先长后短（邮箱与网址先于裸 IP）。
_ENTITY_PATTERNS = (
    re.compile(r"https?://\S+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{1,4}\b"),
    re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    re.compile(r"[A-Za-z]:[\\/][^\s，。；、]*"),
    re.compile(r"/(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"),
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)+(?:[-_][\w.]+)?"),
    re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}日?"),
    re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"),
)
#: 时间词：归一化时抹掉（"今天"与"明天"是同一个问题）。
_TIME_WORDS = ("今天", "明天", "昨天", "前天", "后天", "今日", "昨日", "明日",
               "现在", "目前", "最近", "刚才", "今晚", "今早", "上午", "下午",
               "晚上", "早上", "本周", "上周", "下周", "今年", "去年", "明年")


def normalize_query(text: str) -> str:
    """查询串归一化（去重键）。

    规则：全半角统一（NFKC）→ 小写 → 去实体（IP/路径/网址/邮箱/版本号/
    日期时间）→ 去时间词 → 去标点与空白。归一化后同键的查询视为同一个
    问题，只留一条。
    """
    out = unicodedata.normalize("NFKC", str(text or "")).lower()
    for pattern in _ENTITY_PATTERNS:
        out = pattern.sub(" ", out)
    for word in _TIME_WORDS:
        out = out.replace(word, " ")
    return re.sub(r"[^\w]+", "", out, flags=re.UNICODE)


def desensitize_text(text: str, mapping: dict) -> str:
    """按映射把实体替换成占位符（长键优先，避免部分替换）。

    与 ``scripts/kb_eval.py`` 里的同名函数保持一致 —— 两份实现是**故意**
    的重复：kb_eval 必须能被单独按路径加载（测试与外部调用都这么干），
    不能依赖本模块。
    """
    out = str(text or "")
    for key in sorted(mapping, key=len, reverse=True):
        if key:
            out = out.replace(key, mapping[key])
    return out


def desensitize_entries(entries: list, mapping: dict) -> list:
    """整份 gold 脱敏（查询串与目标标题一起脱敏，否则匹配不成立）。"""
    out = []
    for entry in entries:
        out.append({
            "q": desensitize_text(entry.get("q", ""), mapping),
            "expect": [desensitize_text(t, mapping)
                       for t in (entry.get("expect") or [])],
        })
    return out


def dedupe_entries(entries: list, *, seen_keys: set | None = None,
                   title_counts: dict | None = None,
                   per_title_cap: int = 2) -> tuple:
    """去重：同键只留一条；同一目标标题最多贡献 ``per_title_cap`` 条查询。

    两条规则的**作用域不同**，别混：

    * **键去重是跨集的**（``seen_keys``）：同一个问题在两个集里各写一遍，
      两边的数字就没有可比性。跨集时先来先得 ⇒ 调用方按**优先级**处理各
      集（种子集先来）。
    * **标题额度是每集之内的**（``title_counts``）：它防的是「一条笔记在
      **一张**记分板上独占指标」。算成全局额度会把后一个集挤空（实测：
      验收集 18 条被前面的集挤到只剩 5 条）。

    Args:
        entries: ``[{"q": ..., "expect": [...]}, ...]``。
        seen_keys: 已占用的归一化键（跨集去重时传入）。
        title_counts: 本集内各目标标题已贡献的查询数。
        per_title_cap: 同一标题的查询数上限。

    Returns:
        ``(kept, dropped)``；``dropped`` 每项带 ``reason``：
        ``duplicate_key`` / ``title_cap``。
    """
    keys = set() if seen_keys is None else seen_keys
    counts = {} if title_counts is None else title_counts
    kept, dropped = [], []
    for entry in entries:
        query = str(entry.get("q") or "")
        key = normalize_query(query)
        if not key:
            dropped.append({"q": query, "reason": "empty_after_normalize"})
            continue
        if key in keys:
            dropped.append({"q": query, "reason": "duplicate_key",
                            "key": key})
            continue
        expect = [str(t) for t in (entry.get("expect") or [])]
        title = expect[0] if expect else ""
        if title and counts.get(title, 0) >= per_title_cap:
            dropped.append({"q": query, "reason": "title_cap",
                            "title": title,
                            "cap": per_title_cap})
            continue
        keys.add(key)
        if title:
            counts[title] = counts.get(title, 0) + 1
        kept.append(entry)
    return kept, dropped


def _clean(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _load_entries(path: str) -> list:
    """读一份 gold/候选文件（JSON 数组或 JSONL 都吃）。"""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.lstrip().startswith("["):
        data = json.loads(text)
        return [x for x in data if isinstance(x, dict)]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            out.append(item)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect eval query candidates")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--out", default=str(HERMES_HOME / "evals" / "candidates.jsonl"))
    parser.add_argument("--min-chars", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=200)
    parser.add_argument("--input", default="",
                        help="处理已有文件（JSON 数组或 JSONL），不查 L3")
    parser.add_argument("--dedupe-against", action="append", default=[],
                        help="已被占用的集（可多次；按传入顺序 = 优先级，"
                             "先来的占住键）")
    parser.add_argument("--per-title-cap", type=int, default=2,
                        help="同一目标标题最多贡献几条查询（默认 2）")
    parser.add_argument("--desensitize-map", default="",
                        help="实体→占位符的 JSON 映射；映射文件**不提交**"
                             "（键就是敏感实体本身）")
    args = parser.parse_args()

    seen_keys: set = set()
    for path in args.dedupe_against:
        # 「已被占用的集」整份占住**键**（跨集去重）：先来先得。
        # 注意：这里不占标题额度 —— 「同一目标标题最多 N 条」是**每集之内**
        # 的规则（防的是一条笔记在**一张**记分板上独占指标）；若把它算成
        # 全局额度，后一个集会被前面的集挤空（实测：验收集 18 条只剩 5 条）。
        for entry in _load_entries(path):
            key = normalize_query(str(entry.get("q") or ""))
            if key:
                seen_keys.add(key)
    # 标题额度按**本集**计
    title_counts: dict = {}

    if args.input:
        entries = _load_entries(args.input)
    else:
        if not L3_DB_PATH.exists():
            print(json.dumps({"ok": False, "error": "L3 database not found",
                              "path": str(L3_DB_PATH)}, indent=2))
            return 1
        since = (datetime.now() - timedelta(days=max(1, args.days))).timestamp()
        conn = sqlite3.connect("file:%s?mode=ro" % L3_DB_PATH.as_posix(), uri=True)
        try:
            rows = conn.execute(
                "SELECT role, content, timestamp, session_id FROM messages "
                "WHERE timestamp >= ? ORDER BY id", (since,)).fetchall()
        finally:
            conn.close()
        seen = set()
        entries = []
        for role, content, ts, session in rows:
            if str(role).lower() != "user":
                continue
            text = _clean(content)
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            if _SKIP.match(text) or text.startswith(("http://", "https://")):
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            entries.append({"q": text,
                            "date": datetime.fromtimestamp(float(ts)).strftime(
                                "%Y-%m-%d") if ts else "",
                            "session": session or ""})

    before = len(entries)
    kept, dropped = dedupe_entries(
        entries, seen_keys=seen_keys, title_counts=title_counts,
        per_title_cap=args.per_title_cap)
    if args.desensitize_map:
        mapping = json.loads(Path(args.desensitize_map).read_text(
            encoding="utf-8"))
        kept = desensitize_entries(kept, mapping)

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(kept, ensure_ascii=False, indent=1) + "\n",
                    encoding="utf-8")
    print(json.dumps({
        "ok": True,
        "input": args.input or ("L3:%s" % L3_DB_PATH),
        "before": before,
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_reasons": {
            reason: len([d for d in dropped if d.get("reason") == reason])
            for reason in sorted({d.get("reason") for d in dropped})
        },
        "per_title_cap": args.per_title_cap,
        "desensitized": bool(args.desensitize_map),
        "out": str(dest),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
