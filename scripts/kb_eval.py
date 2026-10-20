#!/usr/bin/env python3
"""Evaluate KB retrieval against a curated gold set.

Gold format (JSON array):
    [{"q": "家里 NAS 用的什么 CPU",
      "expect": ["飞牛 NAS 硬件配置"]},
     {"q": "今天上海天气", "expect": []}]

``expect: []`` is a **negative** query: a good system should return nothing
above the configured floor. Reporting it separately keeps "always return
something" from looking like high recall.

Usage:
    python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json
    python scripts/kb_eval.py --gold ... --save-baseline baseline.json
    python scripts/kb_eval.py --gold ... --baseline baseline.json --max-drop 0.02
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap

HERMES_HOME = bootstrap(__file__)


def _load_gold(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("gold file must be a JSON array")
    return data


def evaluate(kb, gold: list, top_k: int) -> dict:
    positives = [g for g in gold if g.get("expect")]
    negatives = [g for g in gold if not g.get("expect")]
    p1 = r5 = mrr = 0.0
    misses = []
    for g in positives:
        q = str(g["q"])
        expect = [str(x) for x in g["expect"]]
        hits = kb.search(q, top_k=top_k)
        titles = [str(h.get("title", "")) for h in hits]
        if titles and titles[0] in expect:
            p1 += 1
        if any(t in expect for t in titles[:5]):
            r5 += 1
        rank = next((i + 1 for i, t in enumerate(titles) if t in expect), None)
        if rank:
            mrr += 1.0 / rank
        else:
            misses.append({"q": q, "expect": expect, "got": titles[:5]})
    n = max(1, len(positives))
    negative_fp = 0
    negative_details = []
    for g in negatives:
        hits = kb.search(str(g["q"]), top_k=top_k)
        if hits:
            negative_fp += 1
            negative_details.append({"q": g["q"],
                                     "got": [h.get("title") for h in hits[:3]]})
    return {
        "queries": len(gold),
        "positives": len(positives),
        "negatives": len(negatives),
        "p_at_1": round(p1 / n, 4),
        "recall_at_5": round(r5 / n, 4),
        "mrr": round(mrr / n, 4),
        "negative_false_positive_rate": (
            round(negative_fp / max(1, len(negatives)), 4)),
        "misses": misses,
        "negative_false_positives": negative_details,
        "top_k": top_k,
        "affinity_enabled": bool(getattr(
            getattr(getattr(kb, "_config", None), "kb", None),
            "affinity_enabled", False)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-baseline", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--max-drop", type=float, default=0.02,
                        help="Allowed P@1 / recall@5 drop vs baseline")
    parser.add_argument("--with-affinity", action="store_true",
                        help="Measure the production ranking including usage "
                             "affinity. Off by default: affinity updates as "
                             "queries run, so the same set scores differently "
                             "on each pass (the evaluator would contaminate "
                             "itself).")
    args = parser.parse_args()

    gold = _load_gold(Path(args.gold))

    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed._kb import KnowledgeBase

    cfg = load_governed_config(HERMES_HOME)
    if not args.with_affinity:
        # 评测必须可复现：affinity 的命中计数会在查询过程中增长，同一份
        # gold 集连跑两次会得到不同排序。默认关掉，只测核心相关性；
        # 要看生产完整排序时显式传 --with-affinity（结果不保证可复现）。
        cfg.kb.affinity_enabled = False
    kb = KnowledgeBase(cfg)
    result = evaluate(kb, gold, args.top_k)
    result["ok"] = True

    if args.save_baseline:
        dest = Path(args.save_baseline)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        result["baseline_saved"] = str(dest)

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        deltas = {}
        regressions = []
        for key in ("p_at_1", "recall_at_5", "mrr"):
            delta = round(result[key] - float(base.get(key, 0.0)), 4)
            deltas[key] = delta
            if delta < -abs(args.max_drop):
                regressions.append({"metric": key, "delta": delta})
        fp_delta = round(
            result["negative_false_positive_rate"]
            - float(base.get("negative_false_positive_rate", 0.0)), 4)
        deltas["negative_false_positive_rate"] = fp_delta
        if fp_delta > abs(args.max_drop):
            regressions.append({"metric": "negative_false_positive_rate",
                                "delta": fp_delta})
        result["baseline_deltas"] = deltas
        result["regressions"] = regressions
        result["ok"] = not regressions

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
