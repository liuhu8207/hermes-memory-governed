"""Hard-negative recall bench — 检索层能不能被相似但错误的内容骗到。

为什么需要它
============
``cmd_eval`` 的默认自查询集（问句 = 事实本身，期望 = 找回自己）测的是**通道
健康**：分数换算、门槛、通道退化，任何 P0 都会让它掉下去。这块已经有价值。

它测不出的是**分辨力**。一条硬性区分如下：

    自查询   「L2 从 2058 剪到 32 行」→ 期望召回自己
    hard neg 「L2 现在有多少行」      → 期望 32，而库里还有 2058 那条旧记录

后者是 agent 真正会问的问题，也是记忆系统真正会出错的方式：库里同时躺着
旧值和新值，谁被排在前面决定了答案是对的还是过时的。hit@k 在这种集上恒等于
1.0，因为正确答案一定在库里——问题在于它排在第几位、以及错误的那些有没有
挤掉它。

三类 hard negative
------------------
``outdated``   同一 slot 的旧值仍在库里（配置改过，旧记录没撤）
``denial``     一句否定被当成事实写入（本次新增门禁 ``non_assertive`` 防的正
               是这一类，所以这一项直接量化该门禁的效果）
``near_miss``  高词汇重叠但主题不同（另一个项目/另一台机器的同名配置）

判定口径
========
对每题记录 ``rank_of_answer``。一个系统可以 hit@k=1.0 而 rank_of_answer=6——
后者才是危险的，因为 top_k 截断会把正确答案切掉。

这个模块已经改过三次判定，每一次都是因为上一次**量错了东西**，记下来以免
有人把它当成一把稳定的尺子：

1. **子串匹配**（首版）：heldout 报 0.0，而当时每道计分题其实都答对了 ——
   它量的是我写给期望答案的措辞，不是检索能力。改成按特征词覆盖判定。
2. **分母 = 有命中的条数**：一个什么都不返回的检索能算到 1.0。改成
   ``decided``（所有可判定题），详见 ``run_bench`` 里的注释。
3. **不校验样本本身**（2026-10-19）：两道 FAIL 与检索无关 ——
   一道的答案住在 KB 不在 L2，一道的答案从来没被写进过任何层。bench 年年替
   我自己的样本错误扣检索的分。现在这两类退出分母并被点名，见
   ``grounded`` / ``answer_found_in`` / ``ungrounded_items``。
   同一次还发现 12 题里有 6 题的**负例根本不在被测层**，
   ``negatives_ranked_above`` 对它们恒为 0 —— 于是 ``clean_pass_rate`` 会在
   一批没人打扰的题上记满分。现在改看 ``live_negatives`` /
   ``contested_clean_rate``：没有竞争者时它是 ``None`` 而不是 1.0。

一句话：**先看答案在不在被测的那一层，再看有没有人真的站在场上竞争，
最后才谈排序**。

留出集
======
``--split dev``（默认）与 ``--split heldout``。两套样本刻意不同源：
dev 的负例从现网事实改写，heldout 的负例是合成但语义更远。这样 heldout 不是
dev 的重排，避免"在 dev 上调好、heldout 上不成立"。

用法::

    python memory_cli.py bench --top-k 5
    python memory_cli.py bench --split heldout --top-k 3
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# 样本定义
# ---------------------------------------------------------------------------

#: ``(query, answer, [negative...], kind, note)``
#:
#: 负例写成**完整句子**而不是短语，因为 bench 要检验的是排序面对一整条真实
#: 竞争内容时的表现，而不是词法打分器对 token 的偏好。
BENCH_DEV: list[dict[str, Any]] = [
    {
        "kind": "outdated",
        "query": "L2 事实表现在有多少行",
        "answer": "L2 从 2058 剪到 32 行",
        "negatives": [
            "HGM 维护 2026-09-22：L2 从 2058 剪到 32 行（删 category=rebuild×2023 "
            "+ testing×3，备份 memories.bak_planB_20260922_220719）",
        ],
        "note": "旧值与新值同库并存，检验 recency 是否被使用",
    },
    {
        "kind": "outdated",
        "query": "Hermes 默认模型当前是哪个版本",
        "answer": "model.default 从 mimo-v2.5 改为 mimo-v2.6-flash",
        "negatives": [
            "Hermes 默认模型已切到 mimo-v2.6-flash：config.yaml 的 model.default "
            "已从 mimo-v2.5 改为 mimo-v2.6-flash，改动需重启才生效",
        ],
        "note": "答案比负例更具体，旧版本信息混在负例里",
    },
    {
        "kind": "denial",
        "query": "值班表改动后分页组怎么变",
        "answer": "分页组已改成 oncall-rotation",
        "negatives": [
            "如果值班表调整，#ops-pager 分页组就会改成 oncall-rotation",
            "假如值班表调整，分页组会改成 oncall-rotation",
        ],
        "note": "条件句与陈述句词汇高度重叠，只有语义角色不同",
    },
    {
        "kind": "near_miss",
        "query": "网关重启要用什么命令",
        "answer": "schtasks /run /tn Hermes_Gateway",
        "negatives": [
            "memories 表 memories 新增 project 列，recall 加 --project 时保留本项目",
        ],
        "note": "负例与问句共享「网关/重启/命令」的表层词汇",
    },
    {
        "kind": "near_miss",
        "query": "家庭网络代理节点挂在哪个地址",
        "answer": "判断的是代理节点在不在，不是端口",
        "negatives": [
            "开源项目 byJoey/cfnew 本质是在 Cloudflare Workers/Pages/Snippets "
            "上自建 VLESS/Trojan 代理节点，CF 免费版硬约束 100k 请求/天",
        ],
        "note": "跨项目：问家庭，负例是别人的「代理节点」（同一批词，不同所指）",
    },
    # -- 下面三题的答案都取自现网 L2（会由 ``default_locate_cli`` 定位到 "l2"），
    #    用来证明 dev 不只是"两道可计分题"。上一版只有 5 题里有 2 题计分，
    #    其中还有 1 题的答案根本不在被测层 —— 那种样本量读不出任何趋势。
    {
        "kind": "outdated",
        "query": "企业微信机器人的 AI 能力现在能用吗",
        "answer": "wecom-cli 授权已恢复：todo list 成功返回，message aibot send "
                  "回包 success 且消息实际送达",
        "negatives": [
            "企业微信机器人「刘虎的机器人」的 AI 能力权限自 2026-09-25 起因授权"
            "到期而失效：wecom-cli 的 todo list 返回 errcode 850003，message "
            "aibot send 返回 850003，而 auth show --status 仍显示 authorized",
        ],
        "note": "库里同时躺着『失效』(09-25)与『已恢复』(10-17)两条，"
                "检验被替换掉的旧值会不会顶在前面",
    },
    {
        "kind": "denial",
        "query": "L3 会话归档里有 WorkBuddy 的会话吗",
        "answer": "L3 只由 Hermes 自己的插件写入，实测不含 WorkBuddy 与 DSH 的会话",
        "negatives": [
            "据说会话归档对所有接入的 agent 都开了，WorkBuddy 与 DSH 的会话也进了 L3",
            "每天 03:00 的 cron 任务 memory-daily-maintenance 维护 L3",
        ],
        "note": "真相是否定式（只有 Hermes 会写），负例一个转述句、一个同cron子句",
    },
    {
        "kind": "near_miss",
        "query": "SecretStore 要在两台机器上跑要注意什么",
        "answer": "必须把主实例的 rsa.key 同步到另一个实例，否则两个实例加密的"
                  "数据互不兼容，且同一时间只能在一个实例写入",
        "negatives": [
            "跨机不能共享 SQLite 文件",
        ],
        "note": "跨系统：问 SecretStore 的多实例约束，负例是 Octop 的跨机要求",
    },
]


BENCH_HELDOUT: list[dict[str, Any]] = [
    {
        "kind": "denial",
        "query": "同步盘导致 push 失败的真实原因是什么",
        "answer": "不是远端没了，是同步盘的目录清理行为让 refs 建不起来",
        "negatives": [
            "如果同步盘清理了 refs 目录，push 就会失败",
            "要是远端被删了，重新 clone 就能恢复",
        ],
        "note": "真相是「不是 A 是 B」，两个负例一个假设一个错误诊断",
    },
    {
        "kind": "outdated",
        "query": "dsh 的记忆插件现在用哪个",
        "answer": "dsh 走内置 MCP 客户端，自研插件 dsh-memory-governed 已停用",
        "negatives": [
            "dsh-memory-governed 是 DSH 的独立记忆插件，通过 execFile 调用 memory_cli.py",
        ],
        "note": "旧插件说明仍在库里，答案是「已停用」",
    },
    {
        "kind": "near_miss",
        "query": "SSH 免密现在配了几把密钥",
        "answer": "两把无口令密钥 id_ed25519 与 id_rsa",
        "negatives": [
            "MiMo Desktop 的 MCP 配置在 mimocode.jsonc 顶层 mcp 段",
        ],
        "note": "词汇不重叠，检验排序是否被长度而非相关性主导",
    },
    {
        "kind": "denial",
        "query": "MCP 工具写中文事实的已知问题是什么",
        "answer": "hgm_remember 传含中文的事实会报 embedding_failed，需改走 CLI",
        "negatives": [
            "据说 hgm_remember 的 embedding 问题已经修好了",
            "他表示改走 memory_cli.py 就能绕过",
        ],
        "note": "两个转述型负例，正确答案是第一人称的实测结论",
    },
]

SPLITS: dict[str, list[dict[str, Any]]] = {
    "dev": BENCH_DEV,
    "heldout": BENCH_HELDOUT,
}


# ---------------------------------------------------------------------------
# 判定
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _mentions_correction_marker(text: str) -> bool:
    """答案是否自带"已改/不再/已停用"这类标记。

    硬性区分：outdated 类样本里，正确答案往往**就是**那条更新过的记录，
    而负例是旧记录。两者的内容高度重叠，靠 token 匹配分不开——必须承认
    「这一类题在这个规模的数据上不可判定」，报 ``undecidable`` 而不是硬算
    命中。否则 bench 会给出一个自己都不信的数字。
    """
    return bool(re.search(
        r"(已改|已切|已从|不再|已停用|已撤|现已|改成|剪到|降到)", _norm(text)))


@dataclass
class ItemResult:
    kind: str
    query: str
    rank_of_answer: int | None
    negatives_ranked_above: int
    undecidable: bool = False
    note: str = ""
    negatives_above: list[str] = field(default_factory=list)
    #: 答案是否真的存在于被测层。False = 这道题无论怎么调排序都不可能过。
    grounded: bool = True
    #: 未 grounded 时，答案实际待在哪一层（尽力值，允许为空字符串）。
    answer_found_in: str = ""
    #: 这道题的**负例里有几个真的存在于被测层**。
    #:
    #: 0 表示没人站在场上：``negatives_ranked_above`` 恒为 0，`clean_pass_rate`
    #: 因此会在一道根本没被干扰的题上记一分（2026-10-18 的 dev/heldout 里
    #: 12 题中就有 6 题的负例全不在库里）。它不是失败，但它量的东西不同 ——
    #: 只测"能不能找回答案"，不测"会不会被相似内容骗"。
    live_negatives: int = 0

    @property
    def passed(self) -> bool:
        if not self.grounded or self.undecidable:
            return True          # 不计入失败，但也不计入命中
        return (self.rank_of_answer is not None
                and self.rank_of_answer <= 3
                and self.negatives_ranked_above == 0)


@dataclass
class BenchReport:
    split: str
    top_k: int
    evaluated: int
    decided: int
    undecidable: int
    answer_at_1: float
    answer_at_3: float
    median_rank: float | None
    clean_pass_rate: float
    engine: str
    ungrounded: int = 0
    contested: int = 0
    contested_clean_rate: float | None = None
    items: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rank_of(query_norm: str, hit_norm: str) -> bool:
    """答案命中判定。

    The first version compared the expected string against the hit by
    SUBSTRING, and that made the bench measure my own phrasing rather than the
    retrieval: 「两把无口令密钥 id_ed25519 与 id_rsa」 is not a substring of the
    stored row (which reads 「两把无口令密钥 —— id_ed25519(注释 herm…)」), so a
    search that put the RIGHT answer first scored 0. Held-out answer@1 came out
    0.0 while the retrieval was in fact correct on every counted item.

    So the expected answer is matched by COVERAGE of its distinctive terms
    instead. Each term is a token that must be individually distinctive — a
    bare 「的」 or 「密钥」 would match half the store — and the hit has to carry
    a majority of them. That decouples the ruler from phrasing while staying
    strict enough that an unrelated row cannot pass by accident.
    """
    if not query_norm or not hit_norm:
        return False
    if query_norm in hit_norm:          # exact phrasing still counts
        return True
    terms = _distinctive_terms(query_norm)
    if not terms:
        return False
    present = sum(1 for t in terms if t in hit_norm)
    return present * 2 >= len(terms)


#: 答案里用于覆盖判定的词：中文按 2-gram、英文/标识符按单词。
#: 「至少 2 字」与「多数命中」合起来，挡掉「的」「是」这类虚词凑数。
_DISTINCTIVE_TERM_RE = re.compile(
    r"[a-z0-9][a-z0-9_.-]{1,}|[一-鿿]{2,}")


def _distinctive_terms(expected: str) -> list[str]:
    """把期望答案切成用于覆盖判定的词。"""
    norm = _norm(expected)
    return [t for t in _DISTINCTIVE_TERM_RE.findall(norm) if len(t) >= 2]


def _resolve_grounding(answer: str, locate, measured_layer: str) -> tuple:
    """``(grounded, found_in)`` — 答案有没有真的住在被测层。

    为什么这道前置校验不能省（2026-10-18 踩到的）
    --------------------------------------------
    dev 里 ``Hermes 默认模型当前是哪个版本`` 长期 FAIL，而检索并没有做错什么：
    期望的答案写在 **KB 笔记** ``Hermes 默认模型已切到 mimo-v2.6-flash`` 里，
    L2 一条都没有。bench 只跑 L2，于是它年复一年地扣掉一分 —— 扣的是"我编了
    一个在被测层里不存在的答案"，不是"排序不好"。

    没有这道校验，FAIL 与"答案确实在库里"这个必要前提就永久缺失：读数的人看
    到 0.333，无从判断该去改检索还是去改样本。所以这类题一律标 ungrounded、
    退出分母，并**把它实际所在的层报出来**，而不是默默丢掉。

    定位失败时保守地判 grounded：让一道不可判定的题继续按旧行为计分，好过让
    一个探针的故障把样本悄悄删掉。
    """
    if locate is None:
        return True, ""
    try:
        where = locate(answer)
    except Exception as exc:  # noqa: BLE001
        # Reported, not swallowed: a dead locator must leave a trace. KEEPING the
        # item scored is then a choice, not a silent one — dropping it here would
        # turn 0.333 into 1.0 without touching a line of retrieval code.
        from ._diag import log_degraded

        log_degraded("bench_grounding", "locate_failed",
                     detail="keeping the item scored", exc=exc)
        return True, f"locate-failed:{type(exc).__name__}"
    if where == measured_layer:
        return True, ""
    return False, str(where or "")


def run_bench(
    search: Callable[[str, int], list[str]],
    split: str = "dev",
    top_k: int = 5,
    engine_name: str = "cli-recall_l2",
    locate: Callable[[str], str | None] = None,
    measured_layer: str = "l2",
) -> BenchReport:
    """对每个样本跑一次检索，记录答案名次与被压过的负例数。

    ``search`` 是一个 ``(query, top_k) -> [hit_text, ...]`` 的可注入函数，
    所以同一份样本可以喂给 CLI 通道与 plugin 引擎做对照，而不必让本模块
    知道任何一方的内部结构。

    ``locate`` 是可选的**答案定位器** ``(answer) -> layer | None``：答案实际
    存在于哪一层。返回值不等于 ``measured_layer`` 的题目记 ``ungrounded`` 并
    退出计分分母 —— 见 :func:`_resolve_grounding` 里为什么这条前置校验不能省。

    ``locate`` 为 ``None``（默认）时**一切照旧**：所有题目都当 grounded。
    这是刻意的兼容：定位需要一次额外的全量查找，单元测试与纯排序实验不必付
    这个代价。
    """
    items = SPLITS.get(split)
    if not items:
        raise ValueError(f"unknown split {split!r}; "
                         f"known: {sorted(SPLITS)}")

    results: list[ItemResult] = []
    for spec in items:
        q = str(spec["query"])
        answer = str(spec["answer"])
        negatives = [str(n) for n in spec.get("negatives") or []]
        try:
            hits = list(search(q, top_k))
        except Exception as exc:  # noqa: BLE001
            # silent-ok: already-reported — a probe that raised is a RESULT,
            # not a swallowed error: it is recorded on the item (undecidable +
            # the reason) and counted in the report, so the next reader sees
            # which questions the retrieval could not answer. Raising here would
            # abandon the whole split, and the remaining items are still valid
            # measurements.
            #
            # ``_safe_str``, not an f-string: an exception object is untrusted
            # input (a custom ``__str__`` may itself raise), and this runs
            # INSIDE an except block — so an eager ``{exc}`` would turn the
            # bench's graceful degradation into a crash, which is exactly what
            # ``_diag._safe_str`` exists to prevent.
            from ._diag import _safe_str
            results.append(ItemResult(
                kind=spec["kind"], query=q, rank_of_answer=None,
                negatives_ranked_above=0, undecidable=True,
                note=f"search raised {type(exc).__name__}: {_safe_str(exc)}"))
            continue

        hit_norms = [_norm(h) for h in hits]
        ans_norm = _norm(answer)
        neg_norms = [_norm(n) for n in negatives]

        rank = None
        for i, h in enumerate(hit_norms, start=1):
            if _rank_of(ans_norm, h):
                rank = i
                break

        above: list[str] = []
        for i, h in enumerate(hit_norms, start=1):
            if any(_rank_of(n, h) for n in neg_norms):
                if rank is None or i < rank:
                    above.append(h[:90])

        undecidable = _mentions_correction_marker(answer)
        grounded, found_in = _resolve_grounding(answer, locate, measured_layer)
        live = 0
        if locate is not None:
            for neg in negatives:
                ok, _where = _resolve_grounding(neg, locate, measured_layer)
                live += 1 if ok else 0
        results.append(ItemResult(
            kind=spec["kind"], query=q, rank_of_answer=rank,
            negatives_ranked_above=len(above), undecidable=undecidable,
            note=str(spec.get("note") or ""),
            negatives_above=above,
            grounded=grounded, answer_found_in=found_in,
            live_negatives=live))

    scorable = [r for r in results if r.grounded]
    decided = [r for r in scorable if not r.undecidable]
    ungrounded = len(results) - len(scorable)
    # 有竞争的那些题单独结算：``clean_pass_rate`` 会把"没人干扰"的题也算成干净
    # 通过，于是 1.0 可以被读成"抗骗通过"，而实际上一半题库里根本没有骗手。
    contested = [r for r in decided if r.live_negatives > 0]
    ranks = [r.rank_of_answer for r in decided if r.rank_of_answer]
    at3 = (sum(1 for r in ranks if r <= 3) / len(decided)) if decided else 0.0
    # Denominator is every DECIDED item, not every ranked one: an answer the
    # retrieval never returned is a miss, and counting it out would let a
    # search that returns nothing score 1.0.
    at1 = (len(ranks) / len(decided)) if decided else 0.0
    ordered = sorted(ranks)
    median = (ordered[len(ordered) // 2] if ordered else None)
    clean = (sum(1 for r in decided if r.passed) / len(decided)
             if decided else 0.0)
    # ``None`` 而不是 1.0 —— 没有竞争者时谈"抗骗"是假的，宁可不给数字。
    contested_clean = (sum(1 for r in contested if r.passed) / len(contested)
                       if contested else None)

    notes = [
        f"split={split}: {len(decided)} decided, "
        f"{len(scorable) - len(decided)} undecidable, "
        f"{ungrounded} ungrounded, {len(contested)} contested",
        "undecidable = 答案自带『已改/不再/已停用』标记，"
        "与负例内容重叠到无法按内容判定；这类题不计分。",
        "ungrounded = 答案不在被测层（见该项的 answer_found_in），"
        "再好的排序也不可能答对；这类题同样不计分，"
        "但会被列出来——不能让一次样本错误冒充检索缺陷。",
        "contested = 负例里至少有一个真的存在于被测层（见该项的 live_negatives）；"
        "只有这类题在测『会不会被相似内容骗』，"
        "contested_clean_rate 因此只看它们，且没有竞争者时是 None 而不是 1.0。",
    ]

    return BenchReport(
        split=split, top_k=top_k, evaluated=len(results),
        decided=len(decided), undecidable=len(scorable) - len(decided),
        ungrounded=ungrounded,
        answer_at_1=round(at1, 4), answer_at_3=round(at3, 4),
        median_rank=median, clean_pass_rate=round(clean, 4),
        contested=len(contested),
        contested_clean_rate=(round(contested_clean, 4)
                              if contested_clean is not None else None),
        engine=engine_name,
        items=[asdict(r) for r in results], notes=notes)


def default_search_cli(top_k_default: int = 5) -> Callable[[str, int], list[str]]:
    """默认检索器：走 memory_cli.recall_l2（与 agent 实际同一条路）。"""
    import memory_cli  # 延迟导入：允许本模块被单独测试

    def _search(query: str, k: int) -> list[str]:
        res = memory_cli.recall_l2(query, k, [], project="")
        return [str(h.get("content") or "") for h in res.get("hits", [])]

    return _search


def default_locate_cli(config: dict) -> Callable[[str], str | None]:
    """答案住在哪一层（CLI 视角：先看 L2，再看 KB）。

    返回一个 ``(answer) -> layer | None``，喂给 :func:`run_bench` 的
    ``locate``。``"l2"`` 表示被测层里有它；``"kb:<笔记名>"`` 表示它住在知识库
    —— 那是**另一种结果**，不是失败，且不进排序的分母。

    两次搜索各自的失败都必须被**说出来**（:func:`_diag.log_degraded`）而不是
    ``pass``：一次不可读的 KB 会把答案误判成"不存在"，从而悄悄把一道题从评测
    里删掉 —— 那和把 0.333 改成 1.0 是一回事，只不过没人看得见。
    """
    import memory_cli  # 延迟导入：允许本模块被单独测试

    def _in_l2(answer: str) -> bool:
        lex = memory_cli._l2_lexical_hits(  # noqa: SLF001 — 同一个 CLI 的同层函数
            answer, 0, "", 0.0, keep_all=True)
        if not lex.get("ok"):
            return False
        ans = _norm(answer)
        return any(_rank_of(ans, _norm(c))
                   for _score, c, _ag, _pj in lex.get("candidates") or [])

    def _in_kb(answer: str) -> str:
        from ._diag import log_degraded

        try:
            notes = memory_cli.cmd_kb_search(config, answer, 3, "", [])
        except Exception as exc:  # noqa: BLE001
            log_degraded("bench_locate", "kb_search_failed", exc=exc)
            return ""
        ans = _norm(answer)
        terms = sorted(_distinctive_terms(answer), key=len, reverse=True)[:2]
        for n in notes or []:
            title = str(n.get("title") or "")
            hay = _norm(f"{title} {n.get('snippet') or ''} "
                        f"{n.get('path') or ''}")
            # 先按覆盖判定认：认不出再看"答案里两个最长的专名是否同篇出现"——
            # snippet 只有 200 字，逐字复述的期望常常装不下。
            if _rank_of(ans, hay) or (
                    len(terms) == 2 and all(t in hay for t in terms)):
                return f"kb:{title}"
        return ""

    def _locate(answer: str) -> str | None:
        from ._diag import log_degraded

        try:
            if _in_l2(answer):
                return "l2"
        except Exception as exc:  # noqa: BLE001
            log_degraded("bench_locate", "l2_scan_failed", exc=exc)
            return None
        kb = _in_kb(answer)
        return kb or None

    return _locate


def load_bench_file(path: str) -> list[dict[str, Any]]:
    """从 JSON 文件加载自定义样本（供项目自己维护的题集）。"""
    raw = json.loads(open(path, encoding="utf-8").read())
    if not isinstance(raw, list):
        raise ValueError("bench file must be a JSON list of items")
    return [r for r in raw if isinstance(r, dict)]
