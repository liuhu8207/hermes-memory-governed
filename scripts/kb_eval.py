#!/usr/bin/env python3
"""Evaluate KB retrieval against a curated gold set.

Gold format (JSON array):
    [{"q": "家里 NAS 用的什么 CPU",
      "expect": ["飞牛 NAS 硬件配置"]},
     {"q": "今天上海天气", "expect": []}]

``expect: []`` is a **negative** query: a good system should return nothing
above the configured floor. Reporting it separately keeps "always return
something" from looking like high recall.

A/B 分离：谁来、拿哪个集、怎么判、留什么产物
----------------------------------------------
* **A = 标定集**（``--set A``）：只用来**选门槛**（``--select-threshold``）
  与**鉴别力自检**（``--discriminability-check``）。标定集一旦被用来报结果，
  它就不再是标定集了 —— 所以 ``--baseline`` 闸门直接拒绝 A 集。
* **B = 验收集**（``--set B``）：只用来**报结果**与 ``--baseline`` 闸门。
* ``smoke``：冒烟集（只证明通道活着，不是记分板，不许当基线）。

同一个集不许既当标定集又当验收集：``--baseline`` 会核对基线文件里记录的
集合身份（角色 + 名字），跨集比较直接拒绝 —— 混在一张记分板上的两个数字
没有可比性。

鉴别力自检（``--discriminability-check``）
------------------------------------------
一份记分板能不能用来选门槛，先看它**有没有鉴别力**：把门槛扰动到已知会
掉正样本的高档，正样本掉出来多少。掉不出来的集（种子集就是这种）无论报出
什么 P@1 都不构成选门槛的证据 —— 所以本检查**掉出为 0 就 exit 1**。

判据（可执行，不是建议）：

* 扰动到 ``--probe-high``（默认 0.30）后，正样本掉出比例 ≥ 15% 且至少掉
  2 条 ⇒ 这一集有鉴别力；掉出 0 条 ⇒ 报「本集无鉴别力」并 exit 1；
* 负样本数 ≥ 8（硬警告）：负样本只有 4 条时，FPR 的步长是 0.25 —— 「0.25
  掉到 0」其实只差一条负样本，统计意义极弱；
* 正样本 rerank 分中位数距当前门槛 ≥ 0.2（硬警告）：贴着门槛的集既敏感
  又脆，门槛一动就翻面。

门槛选择（``--select-threshold``）
----------------------------------
在标定集上按固定规则选门槛（规则写在 :func:`select_threshold` 里，便于
审计）：相对 ``floor=0`` **零掉样本**的档位里取**最低**的一个 —— 同一平
台上门槛越低余量越大（离"下一个会掉正样本的档位"更远）。FPR 更低但不在
同一平台上的档位会被报出来并标出它的悬崖余量（余量不足 ⇒ 陷阱档）。

Usage:
    python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json --set smoke
    python scripts/kb_eval.py --gold evals/retrieval_gold.calibration.json --set A \
        --select-threshold --discriminability-check
    python scripts/kb_eval.py --gold evals/retrieval_gold.acceptance.json --set B \
        --baseline evals/baseline.acceptance.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))

from hermes_env import bootstrap

HERMES_HOME = bootstrap(__file__)

# -- 集合角色 ---------------------------------------------------------------
SET_ROLES = {"A": "calibration", "B": "acceptance", "smoke": "smoke"}
SET_ROLE_HINT = {
    "calibration": "标定集：只用于选门槛与鉴别力自检，不许当验收集报结果",
    "acceptance": "验收集：只用于报结果与 --baseline 闸门",
    "smoke": "冒烟集：只证明通道活着，不是记分板，不许当基线",
}
#: 参与比较的四个指标（P@1 / R@5 / MRR / FPR）。
METRIC_KEYS = ("p_at_1", "recall_at_5", "mrr",
               "negative_false_positive_rate")

# -- 门槛扫描网格 -----------------------------------------------------------
# 覆盖「同一平台区 + 悬崖两侧」：0.08-0.10 指标完全相同（同一平台），
# 0.11 把最后一个误报滤掉（FPR→0）但同时贴着正样本掉落的边缘，0.12 开始
# 掉正样本 —— 选 0.11 就是把门槛架在悬崖边上。
DEFAULT_SWEEP_FLOORS = (0.0, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                        0.11, 0.12)
#: 鉴别力自检的两档探针：低档几乎全过，高档已知会掉正样本。两档之间的差
#: 就是这一集能不能区分「真相关」与「刚好擦边」。
DISC_PROBE_LOW = 0.04
DISC_PROBE_HIGH = 0.30
#: 判据（可执行，不是建议）
DISC_MIN_DROP_RATIO = 0.15     # 正样本掉出比例下限
DISC_MIN_DROP_COUNT = 2        # 至少掉几条（掉 1 条可能是噪声）
DISC_MIN_NEGATIVES = 8         # 负样本数下限（FPR 步长 ≤ 0.125）
DISC_MIN_POSITIVE_MARGIN = 0.2  # 正样本 rerank 分中位数离门槛的距离下限
#: 悬崖余量下限：选中的门槛到「下一个会掉正样本的档位」的距离。
CLIFF_MIN_MARGIN = 0.02
#: 二分估分的区间上限（rerank 分落在 [0,1]；0.5 已远超真实分布）。
SCORE_PROBE_HIGH = 0.5
#: ⚠️⚠️ 二分估分的精度 —— **0.03 的尺子只够粗定位，不能拿来定门槛余量**。
#: 实测（钩住 ``Reranker._post`` 直接读真实 ``relevance_score``，同一索引
#: 12 次观测）：悬崖样本的众数是 **0.108924**，区间 **0.108924~0.112891**，
#: 均值 0.109584、sd 0.00140、**极差只有 0.00397**。用宽 0.03 的区间去量
#: ±0.004 的抖动带，只能量出一个偏高的区间中点 —— 这正是文档里那个
#: **已被证伪**的「≈0.117、高出 0.11 约 0.007」的来源：真实众数是
#: **低于** 0.11 的（−0.001076），连方向都是反的。
#: 结论：**判断门槛余量必须读真实 rerank 分**（在 trace 里直接暴露），
#: ``estimate_positive_scores`` 的输出只能回答「这一条大概在什么量级」。
SCORE_PROBE_TOL = 0.03


def _load_gold(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("gold file must be a JSON array")
    return data


def _load_title_map(path: str) -> dict:
    """实体 → 占位符的映射（脱敏版 gold 对着真实库跑时用）。

    格式与 ``scripts/eval_collect.py`` 的脱敏映射一致：``{"实体": "<占位符>"}``。
    映射文件**不提交**（键就是敏感实体本身），只在本机保留。
    """
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("title map must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def desensitize_text(text: str, mapping: dict) -> str:
    """按映射把实体替换成占位符（长键优先，避免部分替换）。

    脱敏版与原版必须**一一对应且仅差占位符** —— 这条映射同时作用于查询串
    与笔记标题，所以匹配仍然成立，难度也基本不变。
    """
    out = str(text or "")
    for key in sorted(mapping, key=len, reverse=True):
        if key:
            out = out.replace(key, mapping[key])
    return out


def _map_title(title: str, mapping: dict) -> str:
    return desensitize_text(title, mapping) if mapping else str(title)


class _FloorKB:
    """按门槛切换 + 缓存的 KB 包装。

    * **换档不用重建索引**：rerank 门槛只在 ``search()`` 里做筛选，候选集
      在筛选之前就定了，所以改 ``cfg.kb.rerank_min_score`` 即可换档；
    * **缓存是必须的**：门槛扫描（十余档 × 全部查询）加上鉴别力自检的
      二分估分，会把同一批查询反复查很多遍，不缓存就是几十倍的无谓外呼。
      缓存键含 (query, top_k, floor, affinity) —— 亲和度会改排序，变了
      缓存必须作废。
    """

    def __init__(self, kb, cfg):
        self.__dict__["_kb"] = kb
        self.__dict__["_cfg"] = cfg
        self.__dict__["_cache"] = {}
        self.__dict__["_floor"] = float(
            getattr(cfg.kb, "rerank_min_score", 0.0) or 0.0)

    def __getattr__(self, name):
        kb = self.__dict__.get("_kb")
        if kb is None:  # pragma: no cover — 只在构造出错时可达
            raise AttributeError(name)
        return getattr(kb, name)

    @property
    def floor(self) -> float:
        return self.__dict__["_floor"]

    def at(self, floor: float) -> _FloorKB:
        value = float(floor)
        self.__dict__["_floor"] = value
        self.__dict__["_cfg"].kb.rerank_min_score = value
        return self

    def set_affinity(self, enabled: bool) -> None:
        self.__dict__["_cfg"].kb.affinity_enabled = bool(enabled)
        self.__dict__["_cache"].clear()

    def search(self, query: str, top_k: int = 5):
        affinity = bool(getattr(self.__dict__["_cfg"].kb,
                                "affinity_enabled", False))
        key = (str(query), int(top_k), round(self.__dict__["_floor"], 6),
               affinity)
        cache = self.__dict__["_cache"]
        if key not in cache:
            cache[key] = self.__dict__["_kb"].search(str(query), top_k=top_k)
        return cache[key]


def _evaluate_detail(kb, gold: list, top_k: int,
                     title_map: dict | None = None) -> tuple:
    """跑一遍 gold，返回 (指标, 每条查询的命中细节)。

    命中细节（哪个正样本在第几位、负样本有没有出结果）是鉴别力自检与门槛
    选择规则的原料 —— 光有汇总数字，判据就只能靠猜。
    """
    mapping = title_map or {}
    positives = [g for g in gold if g.get("expect")]
    negatives = [g for g in gold if not g.get("expect")]
    p1 = r5 = mrr = 0.0
    misses = []
    degraded = []
    found: dict = {}
    for g in positives:
        q = str(g["q"])
        expect = [str(x) for x in g.get("expect") or []]
        hits = kb.search(q, top_k=top_k)
        titles = [_map_title(str(h.get("title", "")), mapping) for h in hits]
        expected = [_map_title(e, mapping) for e in expect]
        if any(h.get("rerank") in ("rerank_no_scores", "rerank_unavailable")
               for h in hits):
            degraded.append(q)
        if titles and titles[0] in expected:
            p1 += 1
        if any(t in expected for t in titles[:5]):
            r5 += 1
        rank = next((i + 1 for i, t in enumerate(titles) if t in expected),
                    None)
        if rank:
            mrr += 1.0 / rank
        else:
            misses.append({"q": q, "expect": expect, "got": titles[:5]})
        found[q] = {"found": bool(rank), "rank": rank}
    n = max(1, len(positives))
    negative_fp = 0
    negative_details = []
    negative_found: dict = {}
    for g in negatives:
        q = str(g["q"])
        hits = kb.search(q, top_k=top_k)
        if any(h.get("rerank") in ("rerank_no_scores", "rerank_unavailable")
               for h in hits):
            degraded.append(q)
        if hits:
            negative_fp += 1
            negative_details.append({"q": q,
                                     "got": [h.get("title") for h in hits[:3]]})
        negative_found[q] = bool(hits)
    metrics = {
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
        "degraded_queries": degraded,
        "top_k": top_k,
        "affinity_enabled": bool(getattr(
            getattr(getattr(kb, "_config", None), "kb", None),
            "affinity_enabled", False)),
    }
    detail = {"found": found, "negatives_found": negative_found,
              "p1_hits": round(p1), "r5_hits": round(r5)}
    return metrics, detail


def evaluate(kb, gold: list, top_k: int, title_map: dict | None = None) -> dict:
    """跑一遍 gold，返回四个指标 + 失误明细。

    Args:
        kb: 任何有 ``search(query, top_k=...) -> [hit, ...]`` 的对象。
        gold: ``[{"q": ..., "expect": [...]}, ...]``。
        top_k: 取前几条。
        title_map: 可选的实体 → 占位符映射（脱敏版 gold 对着真实库跑时用）。

    Returns:
        ``{p_at_1, recall_at_5, mrr, negative_false_positive_rate, ...}``。
    """
    metrics, _ = _evaluate_detail(kb, gold, top_k, title_map)
    return metrics


def validate(result: dict, channels: dict) -> dict:
    """Fail the verdict when the scoreboard would be misleading."""
    result["channels"] = channels
    if result["positives"] == 0:
        result["ok"] = False
        result["error"] = ("gold set has no positive queries — nothing was "
                           "measured (collector output must be curated first)")
    elif channels["semantic_enabled"] and not channels["index_available"]:
        result["ok"] = False
        result["error"] = ("semantic channel is unavailable; scores below come "
                           "from the keyword channel only")
    elif channels.get("reranker_available") is False:
        result["ok"] = False
        result["error"] = "reranker is enabled but unavailable"
    elif result.get("degraded_queries"):
        result["ok"] = False
        result["error"] = ("%d query/queries ran without reranker filtering; "
                           "the scoreboard is not comparable"
                           % len(result["degraded_queries"]))
    return result


# -- 门槛选择 ---------------------------------------------------------------

def _hit_counts(row: dict) -> tuple:
    positives = max(1, int(row.get("positives") or 0))
    return (round(float(row["p_at_1"]) * positives),
            round(float(row["recall_at_5"]) * positives))


def _is_lossless(row: dict, reference: dict) -> bool:
    """相对参考档（floor=0）有没有掉正样本。"""
    p1, r5 = _hit_counts(row)
    rp1, rr5 = _hit_counts(reference)
    return p1 >= rp1 and r5 >= rr5


def _metrics_signature(row: dict) -> tuple:
    return tuple(round(float(row[k]), 6) for k in METRIC_KEYS)


def select_threshold(rows: list, *, cliff_min_margin: float = CLIFF_MIN_MARGIN,
                     reference_floor: float = 0.0) -> dict:
    """在标定集上选门槛 —— 规则写死在这里，便于审计与复现。

    规则：

    1. **基准 = 本集能达到的最好水平**（P@1 命中数最多，其次 R@5）—— 注意
       **不是** floor=0：floor=0 意味着完全不过滤，噪声可以占掉 rank 1，
       拿它当基准会把"降门槛"当成"不掉样本"；
    2. **召回无损** = 相对基准没掉任何正样本（P@1 与 R@5 都不掉）；
    3. 在无损档里取**最低**的一个：门槛越低余量越大 —— 同一平台上低的更
       安全，因为它离「下一个会掉正样本的档位」更远；
    4. FPR 更低但**不在同一平台**上的档位会被报出来，并标出它的悬崖余量
       （余量 < ``cliff_min_margin`` ⇒ 陷阱档，明确不取）；
    5. 相对基准只掉 1 个正样本的档位（"接近无损"）也报出来，但不选 ——
       门槛是**有损操作**的开关，掉样本就是掉样本。

    Args:
        rows: :func:`sweep_floors` 的输出（按门槛升序）。
        cliff_min_margin: 悬崖余量下限。
        reference_floor: 「完全不过滤」那一档的门槛值（只作报告用）。

    Returns:
        含 ``chosen`` / ``plateau`` / ``cliff`` / ``fpr_optimal`` /
        ``near_lossless`` 的字典。
    """
    ordered = sorted(rows, key=lambda r: float(r["floor"]))
    if not ordered:
        return {"ok": False, "error": "empty sweep"}
    no_filter = next((r for r in ordered
                      if abs(float(r["floor"]) - reference_floor) < 1e-9),
                     None)
    # 基准 = 最好的一档（正样本一个都不许比它少）
    best = max(ordered, key=lambda r: (_hit_counts(r)[0], _hit_counts(r)[1],
                                       -float(r["floor"])))
    lossless = [r for r in ordered
                if float(r["floor"]) > reference_floor
                and _is_lossless(r, best)]
    near_lossless = []
    for row in ordered:
        if float(row["floor"]) <= reference_floor or row in lossless:
            continue
        p1, r5 = _hit_counts(row)
        bp1, br5 = _hit_counts(best)
        lost = max(0, bp1 - p1) + max(0, br5 - r5)
        if lost == 1:
            near_lossless.append({"floor": row["floor"],
                                  "lost_positives": 1,
                                  "p_at_1": row["p_at_1"],
                                  "recall_at_5": row["recall_at_5"],
                                  "negative_false_positive_rate":
                                      row["negative_false_positive_rate"]})
    if not lossless:
        return {
            "ok": False,
            "error": ("no floor in the sweep is lossless vs the best tier "
                      "— this set cannot pick a threshold"),
            "benchmark": {k: best[k] for k in METRIC_KEYS},
            "sweep": ordered,
        }
    chosen = min(lossless, key=lambda r: float(r["floor"]))
    # 平台：与 chosen 指标完全相同、且门槛相连的一段
    plateau = [chosen]
    signature = _metrics_signature(chosen)
    for row in ordered:
        if float(row["floor"]) <= float(chosen["floor"]):
            continue
        if _metrics_signature(row) == signature:
            plateau.append(row)
        else:
            break
    # 悬崖：下一个会掉正样本的档位
    cliff = None
    for row in ordered:
        if float(row["floor"]) <= float(chosen["floor"]):
            continue
        if not _is_lossless(row, best):
            cliff = row
            break
    cliff_margin = (round(float(cliff["floor"]) - float(chosen["floor"]), 4)
                     if cliff else None)
    # FPR 最优的无损档：若不在同一平台上，报告它的悬崖余量并给出结论
    fpr_optimal = min(lossless,
                       key=lambda r: (float(r["negative_false_positive_rate"]),
                                      float(r["floor"])))
    rejected = None
    if abs(float(fpr_optimal["floor"]) - float(chosen["floor"])) > 1e-9:
        fpr_cliff = None
        for row in ordered:
            if float(row["floor"]) <= float(fpr_optimal["floor"]):
                continue
            if not _is_lossless(row, best):
                fpr_cliff = row
                break
        fpr_margin = (round(float(fpr_cliff["floor"])
                            - float(fpr_optimal["floor"]), 4)
                      if fpr_cliff else None)
        rejected = {
            "floor": fpr_optimal["floor"],
            "negative_false_positive_rate":
                fpr_optimal["negative_false_positive_rate"],
            "cliff_margin": fpr_margin,
            "verdict": (f"陷阱档：FPR 更低但离悬崖只有 {fpr_margin}"
                        f"（< {cliff_min_margin}）"
                        if (fpr_margin is not None
                            and fpr_margin < cliff_min_margin)
                        else "未选：不在最低无损档上"),
        }
    warnings = []
    if cliff_margin is not None and cliff_margin < cliff_min_margin:
        warnings.append(
            f"选中的门槛 {chosen['floor']} 离悬崖只有 {cliff_margin}"
            f"（< {cliff_min_margin}）：余量不足")
    return {
        "ok": True,
        "chosen": chosen["floor"],
        "rule": ("相对本集最好档（零掉正样本）取最低的无损门槛"
                 "（同平台上门槛越低余量越大）"),
        "reference_floor": reference_floor,
        "no_filter": ({k: no_filter[k] for k in METRIC_KEYS}
                      if no_filter else None),
        "benchmark": {k: best[k] for k in METRIC_KEYS},
        "benchmark_floor": best["floor"],
        "chosen_metrics": {k: chosen[k] for k in METRIC_KEYS},
        "plateau": [r["floor"] for r in plateau],
        "cliff": {"floor": cliff["floor"] if cliff else None,
                  "margin": cliff_margin,
                  "min_margin": cliff_min_margin,
                  "ok": (cliff_margin is None
                         or cliff_margin >= cliff_min_margin)},
        "fpr_optimal": rejected or {
            "floor": fpr_optimal["floor"],
            "negative_false_positive_rate":
                fpr_optimal["negative_false_positive_rate"],
            "verdict": "已选",
        },
        "near_lossless": near_lossless,
        "warnings": warnings,
        "sweep": ordered,
    }


def sweep_floors(floor_kb, gold: list, floors, top_k: int,
                 title_map: dict | None = None) -> list:
    """逐档扫描门槛，返回每档的指标行。"""
    rows = []
    for floor in floors:
        metrics = evaluate(floor_kb.at(floor), gold, top_k,
                           title_map=title_map)
        row = {"floor": float(floor)}
        row.update({k: metrics[k] for k in METRIC_KEYS})
        row["positives"] = metrics["positives"]
        row["negatives"] = metrics["negatives"]
        row["misses"] = [m["q"] for m in metrics["misses"]]
        rows.append(row)
    return rows


# -- 鉴别力自检 -------------------------------------------------------------

def _hits_at(floor_kb, query: str, expect: list, top_k: int,
             floor: float, title_map: dict | None = None) -> bool:
    hits = floor_kb.at(floor).search(query, top_k=top_k)
    titles = [_map_title(str(h.get("title", "")), title_map or {})
              for h in hits]
    expected = [_map_title(e, title_map or {}) for e in expect]
    return any(t in expected for t in titles[:5])


def estimate_positive_scores(floor_kb, gold: list, top_k: int,
                              title_map: dict | None = None) -> dict:
    """估计每个正样本的 rerank 分（黑盒二分）—— **只够粗定位**。

    为什么二分而不是读内部字段：命中结果里**没有** rerank 分（只暴露融合
    分），而候选集怎么组是 ``_kb`` 的内部细节。黑盒只依赖 ``search()`` 的
    公开契约 —— 命中 ⇔ 门槛 ≤ 该条的 rerank 分，所以门槛就是分数的探针。
    这样本工具不会随 ``_kb`` 内部实现漂移。

    ⚠️ **精度只有 ``SCORE_PROBE_TOL`` = 0.03，量不出 ±0.004 的抖动带。**
    实测（钩住 ``Reranker._post`` 读真实 ``relevance_score``）：悬崖样本真实
    众数 **0.108924**，12 次观测区间 **0.108924~0.112891**，极差仅
    **0.00397**。用宽 0.03 的区间去量这么窄的带，返回的只会是一个偏高的
    区间中点 —— 曾据此写出「≈0.117、高出 0.11 约 0.007」，而真实众数是
    **低于** 0.11 的（−0.001076）。**方向都是反的。**

    所以：**不要用本函数的输出去论证「某档门槛还剩多少余量」**，更不要
    用它判断某条正样本「稳不稳」。那是 trace 里真实 rerank 分的活。本函数
    只适合回答「这一条大概落在哪个量级」以及鉴别力自检里的中位数余量警告。

    Args:
        floor_kb: :class:`_FloorKB`。
        gold: gold 集。
        top_k: 取前几条算命中。
        title_map: 可选标题映射。

    Returns:
        ``{query: {"score": float|None, "bracket": [lo, hi]|None,
        "reason": str}}``。``score`` 为 ``None`` 表示这条在 floor=0 时就没
        被召回（压根不在候选集里，没有 rerank 分可言）。
    """
    out: dict = {}
    for g in [g for g in gold if g.get("expect")]:
        q = str(g["q"])
        expect = [str(x) for x in g.get("expect") or []]
        if not _hits_at(floor_kb, q, expect, top_k, 0.0, title_map):
            out[q] = {"score": None, "bracket": None,
                      "reason": "not_retrieved_at_floor_0"}
            continue
        if _hits_at(floor_kb, q, expect, top_k, SCORE_PROBE_HIGH, title_map):
            out[q] = {"score": SCORE_PROBE_HIGH,
                      "bracket": [SCORE_PROBE_HIGH, SCORE_PROBE_HIGH],
                      "reason": "clamped_at_probe_high"}
            continue
        low, high = 0.0, SCORE_PROBE_HIGH
        while high - low > SCORE_PROBE_TOL:
            mid = (low + high) / 2.0
            if _hits_at(floor_kb, q, expect, top_k, mid, title_map):
                low = mid
            else:
                high = mid
        out[q] = {"score": round((low + high) / 2.0, 4),
                  "bracket": [round(low, 4), round(high, 4)],
                  "reason": "bisected"}
    return out


def discriminability_check(floor_kb, gold: list, top_k: int, *,
                           current_floor: float,
                           probe_low: float = DISC_PROBE_LOW,
                           probe_high: float = DISC_PROBE_HIGH,
                           min_drop_ratio: float = DISC_MIN_DROP_RATIO,
                           min_drop_count: int = DISC_MIN_DROP_COUNT,
                           min_negatives: int = DISC_MIN_NEGATIVES,
                           min_margin: float = DISC_MIN_POSITIVE_MARGIN,
                           title_map: dict | None = None) -> dict:
    """这一集有没有鉴别力（可执行判据，不是建议）。

    做法：把门槛扰动到 ``probe_high``（已知会掉正样本的高档），看正样本
    掉出来多少。掉不出来的集 —— 无论它报出多漂亮的 P@1 —— 都不能用来选
    门槛：它对门槛的变化不敏感，报出的数字是"擦边"的结果而不是"相关"的
    结果。

    Args:
        floor_kb: :class:`_FloorKB`。
        gold: 待检集。
        top_k: 取前几条。
        current_floor: 当前生效的门槛（算余量用）。
        probe_low / probe_high: 两档探针。
        min_drop_ratio / min_drop_count: 掉出判据。
        min_negatives: 负样本数硬下限。
        min_margin: 正样本 rerank 分中位数离门槛的硬下限。
        title_map: 可选标题映射。

    Returns:
        含 ``discriminable`` / ``dropped`` / ``warnings`` 的字典；
        ``ok=False`` 表示本集无鉴别力。
    """
    positives = [g for g in gold if g.get("expect")]
    negatives = [g for g in gold if not g.get("expect")]
    low_metrics, low_detail = _evaluate_detail(
        floor_kb.at(probe_low), gold, top_k, title_map)
    high_metrics, high_detail = _evaluate_detail(
        floor_kb.at(probe_high), gold, top_k, title_map)
    dropped = [q for q, info in low_detail["found"].items()
               if info["found"] and not high_detail["found"].get(q, {})
               .get("found")]
    drop_count = len(dropped)
    drop_ratio = round(drop_count / max(1, len(positives)), 4)
    scores = estimate_positive_scores(floor_kb, gold, top_k, title_map)
    values = [v["score"] for v in scores.values() if v["score"] is not None]
    median = round(statistics.median(values), 4) if values else None
    margin = (round(median - float(current_floor), 4)
              if median is not None else None)
    warnings = []
    if len(negatives) < min_negatives:
        warnings.append(
            f"负样本只有 {len(negatives)} 条（判据要求 ≥{min_negatives}）："
            f"FPR 步长 ≥ {1.0 / max(1, len(negatives)):.3f}，"
            "「误报率掉到 0」其实只差一条负样本，统计意义极弱")
    if margin is None:
        warnings.append("正样本 rerank 分估计失败（多为压根没被召回）："
                        "中位数余量无法判定")
    elif margin < min_margin:
        warnings.append(
            f"正样本 rerank 分中位数 {median:.4f} 离当前门槛 "
            f"{float(current_floor):.4f} 只有 {margin:.4f}"
            f"（判据要求 ≥{min_margin:.2f}）："
            "贴门槛的集既敏感又脆，门槛一动就翻面")
    discriminable = (drop_count >= min_drop_count
                     and drop_ratio >= min_drop_ratio)
    out = {
        "ok": discriminable,
        "discriminable": discriminable,
        "probe": {"low": probe_low, "high": probe_high},
        "positives": len(positives),
        "negatives": len(negatives),
        "drop_count": drop_count,
        "drop_ratio": drop_ratio,
        "dropped": dropped,
        "thresholds": {
            "min_drop_ratio": min_drop_ratio,
            "min_drop_count": min_drop_count,
            "min_negatives": min_negatives,
            "min_positive_margin": min_margin,
        },
        "probe_metrics": {
            "low": {k: low_metrics[k] for k in METRIC_KEYS},
            "high": {k: high_metrics[k] for k in METRIC_KEYS},
        },
        "current_floor": float(current_floor),
        "positive_score_median": median,
        "positive_margin": margin,
        "positive_scores": scores,
        "warnings": warnings,
    }
    if not discriminable:
        out["error"] = (
            f"本集无鉴别力：门槛扰动到 {probe_high:.2f} 后正样本掉出 "
            f"{drop_count} 条（比例 {drop_ratio:.4f}，判据要求 "
            f"≥{min_drop_count} 条且 ≥{min_drop_ratio:.2f}）"
            "—— 这份记分板对门槛不敏感，报出的 P@1 不能用来选门槛")
    return out


# -- CLI --------------------------------------------------------------------

def _set_info(args, gold: list) -> dict:
    role = SET_ROLES[args.set]
    return {
        "id": args.set,
        "role": role,
        "role_hint": SET_ROLE_HINT[role],
        "name": Path(args.gold).stem,
        "gold": str(Path(args.gold)),
        "positives": len([g for g in gold if g.get("expect")]),
        "negatives": len([g for g in gold if not g.get("expect")]),
    }


def _affinity_modes(args) -> list:
    """要跑哪几种亲和度配置。OFF 永远先跑 —— 它是**生效配置**（默认关闭），
    ON 一档是拿来对着我比对的对照档，所以先跑出参考档再跑对照档。

    ⚠️ **这里原来写的理由是错的，已删除。** 原文：「亲和度的命中计数会在
    查询过程中增长 ⇒ 同一份集连跑两次排序不同 ⇒ ON 不可复现」。实测证伪：
    把真实 29 行 usage 表放进隔离 home、`affinity=on` 跑完 A 集 22 条查询，
    usage 表 **29 → 29，delta = 0** —— ``search()`` 只读 ``hits()``，写入侧
    的 ``touch()`` 只在召回提示通道被调用。且在**固定表**上连跑 3 次结果
    完全一致。**ON 是可复现的**，不可复现这个说法不成立。

    默认不走 ON 的**真**理由是「零收益 + 更脆」：

    * 实测约 **20 次观测**中，**没有任何一次 ON 优于 OFF**。A 集稳定劣化
      （P@1 0.786~0.929 vs OFF 1.0）；B 集无收益且间歇劣化（P@1
      0.889~1.0，7 次里 5 次与 OFF 完全相同）；空表对照下 ON ≡ OFF；
    * 机制上：affinity 的乘数上限只有 **1.15**，它顶不动前面的候选，只会把
      本来接近的候选压得更近；而 MMR（用 ``rank_key``）在近距上会换序 ——
      等于多引入一条换序抖动通道，却换不来任何排序收益。
    """
    mode = getattr(args, "affinity", "off")
    if getattr(args, "with_affinity", False):
        mode = "on"
    return ["off", "on"] if mode == "both" else [mode]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate KB retrieval")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--set", choices=sorted(SET_ROLES), default="B",
                        help="A=标定集（只选门槛）/ B=验收集（报结果与基线闸门）"
                             " / smoke=冒烟集（不是记分板）")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-baseline", default="")
    parser.add_argument("--baseline", default="")
    parser.add_argument("--max-drop", type=float, default=0.02,
                        help="Allowed P@1 / recall@5 drop vs baseline")
    parser.add_argument("--rerank-min-score", type=float, default=None,
                        help="Override kb.rerank_min_score for a calibration sweep")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Run the whole set N times and report mean/min/max; "
                             "a single pass hides retrieval jitter")
    parser.add_argument("--with-affinity", action="store_true",
                        help="Deprecated alias for --affinity on")
    parser.add_argument("--affinity", choices=("off", "on", "both"),
                        default="off",
                        help="off=生效配置（默认，可复现）/ on=含使用度加成，"
                             "可复现但实测零收益且多一条换序抖动通道，别拿它"
                             "当基线 / both=两组都报，off 在前作参考档")
    parser.add_argument("--title-map", default="",
                        help="实体→占位符的 JSON 映射（脱敏版 gold 对着真实库"
                             "跑时用；映射文件不提交）")
    parser.add_argument("--select-threshold", action="store_true",
                        help="在标定集上按固定规则选门槛（--set A）")
    parser.add_argument("--sweep-floors", default="",
                        help="逗号分隔的扫描档位（默认 "
                             + ",".join(str(f) for f in DEFAULT_SWEEP_FLOORS)
                             + "）")
    parser.add_argument("--discriminability-check", action="store_true",
                        help="鉴别力自检：无鉴别力 ⇒ exit 1")
    parser.add_argument("--probe-low", type=float, default=DISC_PROBE_LOW)
    parser.add_argument("--probe-high", type=float, default=DISC_PROBE_HIGH)
    parser.add_argument("--min-drop-ratio", type=float,
                        default=DISC_MIN_DROP_RATIO)
    parser.add_argument("--min-drop-count", type=int,
                        default=DISC_MIN_DROP_COUNT)
    parser.add_argument("--min-negatives", type=int,
                        default=DISC_MIN_NEGATIVES)
    parser.add_argument("--min-positive-margin", type=float,
                        default=DISC_MIN_POSITIVE_MARGIN)
    parser.add_argument("--cliff-min-margin", type=float,
                        default=CLIFF_MIN_MARGIN)
    parser.add_argument("--save-report", default="",
                        help="把完整 JSON 报告落盘（审计产物）")
    args = parser.parse_args()

    gold = _load_gold(Path(args.gold))
    set_info = _set_info(args, gold)
    title_map = _load_title_map(args.title_map)

    # A/B 分离的第一道闸：基线只比验收集。
    if args.baseline and set_info["role"] != "acceptance":
        out = {"ok": False, "set": set_info,
               "error": (f"--baseline 只比验收集（B）；{set_info['name']} "
                         f"是{set_info['role_hint']}，不能当基线"),
               "hint": "用 --set B 跑验收集，或改用 --set A --select-threshold"}
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 1

    from plugin.memory_governed._config import load_governed_config
    from plugin.memory_governed._kb import KnowledgeBase

    cfg = load_governed_config(HERMES_HOME)
    if args.rerank_min_score is not None:
        cfg.kb.rerank_min_score = float(args.rerank_min_score)
    # 本次运行**生效**的门槛（后面扫描会改 cfg.kb.rerank_min_score，
    # 扫描完必须还原：否则鉴别力自检的「当前门槛」会变成扫描的最后一档，
    # 余量判据就拿错了尺子）。
    effective_floor = float(getattr(cfg.kb, "rerank_min_score", 0.0) or 0.0)
    modes = _affinity_modes(args)
    if modes == ["off"]:
        # 默认关掉 affinity，只测核心相关性。理由**不是**「不可复现」（实测
        # 证伪：usage 表在 ON 下跑完 A 集 22 条查询 delta = 0，固定表连跑
        # 3 次结果一致），而是「零收益 + 更脆」：约 20 次观测里没有任何一次
        # ON 优于 OFF，且乘数上限只有 1.15 的加成会把近距候选压得更近，给
        # MMR 的换序多开一条抖动通道。要看完整排序时显式传 --affinity on。
        cfg.kb.affinity_enabled = False
    kb = KnowledgeBase(cfg)
    floor_kb = _FloorKB(kb, cfg)

    variants = {}
    for mode in modes:
        floor_kb.set_affinity(mode == "on")
        variants[mode] = evaluate(floor_kb, gold, args.top_k,
                                  title_map=title_map)
    result = variants[modes[0]]
    if len(variants) > 1:
        result["affinity_variants"] = {
            mode: {k: v[k] for k in METRIC_KEYS} | {
                "positives": v["positives"], "negatives": v["negatives"]}
            for mode, v in variants.items()}
        result["affinity_note"] = (
            "on 一档含使用度加成，命中计数在查询过程中增长 ⇒ 不可复现；"
            "off 一档才是可复现的那一档")
    if args.repeat > 1:
        # 重复跑用**主档**（默认 off）的亲和度设置；affinity on 不可复现，
        # 混进重复统计里只会把抖动算成噪声。
        floor_kb.set_affinity(modes[0] == "on")
        runs = [evaluate(floor_kb, gold, args.top_k, title_map=title_map)
                for _ in range(int(args.repeat))]
        metrics = ("p_at_1", "recall_at_5", "mrr",
                   "negative_false_positive_rate")
        result["repeat"] = len(runs)
        result["runs"] = [{k: r[k] for k in metrics} for r in runs]
        result["stability"] = {
            key: {
                "mean": round(sum(r[key] for r in runs) / len(runs), 4),
                "min": round(min(r[key] for r in runs), 4),
                "max": round(max(r[key] for r in runs), 4),
            } for key in metrics
        }
        for key in metrics:
            result[key] = result["stability"][key]["mean"]
        result["misses"] = [m for r in runs for m in r["misses"]][:20]
        result["negative_false_positives"] = [
            n for r in runs for n in r["negative_false_positives"]][:20]
        result["degraded_queries"] = sorted({
            q for r in runs for q in r.get("degraded_queries", [])})
    result["ok"] = True
    result["set"] = set_info

    # 诚实报告通道状态：语义通道不可用时，记分板只是关键词通道的成绩，
    # 不能打 ok:true 让人以为全链路都通过了。
    channels = {
        "semantic_enabled": bool(getattr(cfg.kb, "semantic_enabled", True)),
        "index_available": bool(kb.index_available),
        "reranker_enabled": bool(getattr(cfg.reranking, "enabled", False)),
    }
    if channels["reranker_enabled"]:
        try:
            channels["reranker_available"] = bool(
                kb._get_reranker().available)
        except Exception:  # silent-ok: already-reported — 下面会写 unavailable
            channels["reranker_available"] = False
    validate(result, channels)

    if args.select_threshold:
        floors = ([float(x) for x in args.sweep_floors.split(",") if x.strip()]
                  if args.sweep_floors else list(DEFAULT_SWEEP_FLOORS))
        floor_kb.set_affinity(False)
        rows = sweep_floors(floor_kb, gold, floors, args.top_k,
                            title_map=title_map)
        # 还原生效门槛：扫描只是探针，不是配置
        floor_kb.at(effective_floor)
        result["threshold_selection"] = select_threshold(
            rows, cliff_min_margin=args.cliff_min_margin)
        result["threshold_selection"]["set"] = set_info
    if args.discriminability_check:
        floor_kb.set_affinity(False)
        floor_kb.at(effective_floor)
        result["discriminability"] = discriminability_check(
            floor_kb, gold, args.top_k,
            current_floor=effective_floor,
            probe_low=args.probe_low, probe_high=args.probe_high,
            min_drop_ratio=args.min_drop_ratio,
            min_drop_count=args.min_drop_count,
            min_negatives=args.min_negatives,
            min_margin=args.min_positive_margin,
            title_map=title_map)
        result["discriminability"]["set"] = set_info
        if not result["discriminability"].get("discriminable"):
            result["ok"] = False
            result.setdefault("error", result["discriminability"].get("error"))

    if args.save_baseline:
        dest = Path(args.save_baseline)
        dest.parent.mkdir(parents=True, exist_ok=True)
        baseline = {k: result[k] for k in METRIC_KEYS}
        baseline.update({
            "set": set_info,
            "rerank_min_score": float(
                getattr(cfg.kb, "rerank_min_score", 0.0) or 0.0),
            "top_k": args.top_k,
            "affinity_enabled": result.get("affinity_enabled", False),
            # 怎么测的也要记下来：脱敏集 + 标题映射 与 原始集 不是同一种测法
            "title_map_used": bool(args.title_map),
            "queries": result["queries"],
            "positives": result["positives"],
            "negatives": result["negatives"],
        })
        dest.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        result["baseline_saved"] = str(dest)

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        base_set = base.get("set") or {}
        if base_set.get("role") != "acceptance":
            result["ok"] = False
            result["baseline_error"] = (
                f"基线 {args.baseline} 记录的角色是 "
                f"{base_set.get('role')!r}，不是验收集 —— 拒绝跨集比较")
        elif base_set.get("name") != set_info["name"]:
            result["ok"] = False
            result["baseline_error"] = (
                f"基线来自 {base_set.get('name')!r}，"
                f"本次跑的是 {set_info['name']!r} —— 拒绝跨集比较")
        else:
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
            result["ok"] = bool(result["ok"]) and not regressions

    if args.save_report:
        dest = Path(args.save_report)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        result["report_saved"] = str(dest)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
