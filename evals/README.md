# Retrieval evaluation

Three gold sets, three jobs. They are **not** interchangeable, and the tool
refuses to mix them.

`decision_gold.seed.json` is separate from retrieval scoring. It is the
replayable contract for the optional KB-novelty decision layer: duplicate,
extends, conflicts, new, and uncertain. It contains synthetic note summaries
only, so it can exercise Jev or a future local compatible backend without
leaking production content.

For local evaluation, build a private set directly from the live KB, active L2,
L2 archive, and current maintenance state:

```bash
python scripts/decision_gold_build.py --dry-run
python scripts/decision_gold_build.py --count 100
```

The default output is `$HERMES_HOME/evals/decision_gold.real.json`; the builder
refuses to write it inside the public repository. Every item carries source IDs,
source kinds, a deterministic label method, confidence, and an explicit
`rule-labeled-not-human-reviewed` status. Temporal old/new status changes are
kept as `uncertain` rather than being forced into `conflicts`.

| Set | File | Role | What it may decide |
|-----|------|------|--------------------|
| seed | `retrieval_gold.seed.json` | smoke | nothing — it only proves the channel answers. It has **no discriminability**, so it can never pick a floor. |
| A | `retrieval_gold.calibration.json` | calibration | the rerank floor (`--select-threshold`) and the discriminability self-check. |
| B | `retrieval_gold.acceptance.json` | acceptance | the reported numbers and the `--baseline` gate. |

Why two gold sets and not one: the floor is **chosen** on A and **reported** on
B. Choosing on the same set you report from means the report describes the
choice, not the system. `scripts/kb_eval.py` enforces the split — `--set A|B`
declares the role, `--baseline` refuses anything that is not the acceptance
set, and it also refuses a baseline file produced from a *different* set (two
numbers from two sets are not comparable).

## Running

```bash
# smoke — is the channel alive? (never a baseline)
python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json --set smoke

# A — pick the floor, and prove this set is allowed to pick one
python scripts/kb_eval.py --gold evals/retrieval_gold.calibration.json --set A \
    --select-threshold --discriminability-check

# B — report results and compare against the acceptance baseline
python scripts/kb_eval.py --gold evals/retrieval_gold.acceptance.json --set B \
    --baseline evals/baseline.acceptance.json

# affinity on/off on either set (off first: affinity is not reproducible)
python scripts/kb_eval.py --gold evals/retrieval_gold.acceptance.json --set B \
    --affinity both
```

`--baseline` exits non-zero when a metric drops more than `--max-drop`
(default 0.02) or when the negative false-positive rate rises.

## The floor rule (calibration set A)

> Take the **lowest** floor that loses **no** positive sample; the benchmark
> is the best tier the set can reach — *not* `floor=0`, because "no filtering
> at all" lets noise take rank 1 and would make lowering the floor look like
> "losing nothing". Among the floors on the chosen plateau, prefer the lowest
> FPR.

Measured on A (14 positives + 8 negatives):

| floor | P@1 | recall@5 | MRR | negative FPR |
|-------|-----|----------|-----|--------------|
| 0.00 | 0.9286 | 0.9286 | 0.9286 | 1.0 |
| 0.04 | 0.9286 | 1.0 | 0.9524 | 0.25 |
| 0.05 | 0.9286 | 1.0 | 0.9524 | 0.125 |
| 0.06 / 0.07 | 0.9286 | 1.0 | 0.9643 | 0.125 |
| **0.08 / 0.09 / 0.10** | **1.0** | **1.0** | **1.0** | **0.125** |
| 0.11 | 0.9286 | 0.9286 | 0.9286 | 0.0 |
| 0.12 | 0.9286 | 0.9286 | 0.9286 | 0.0 |

`0.08` is the lowest lossless tier and therefore the pick — it keeps 0.02 more
headroom than `0.10`, and the whole `0.08`–`0.10` plateau has identical
metrics. **`0.11` is a trap, but the reason is instability, not a better FPR.**
The positive *"now default model?"* scores **0.108924** (mode of 12 runs,
range 0.108924–0.112891, sd 0.0014) — about **0.0011 below** 0.11, not above it.
At 0.11 that query flips between identical runs (8 of 16 queries flipped in one
sweep; three whole-set runs gave P@1 0.9286 / 1.0 / 1.0). The earlier "≈0.117,
0.007 above 0.11" figure was an artifact: ``SCORE_PROBE_TOL = 0.03`` (bisection)
is too coarse to measure a ±0.004 jitter band. The tool reports the FPR-optimal
tier together with its cliff margin and does not pick it.

The shipped example config therefore sets `kb.rerank_min_score: 0.08`
(`config/governed_memory.example.json`); the **code default stays 0.0** (off).

## Discriminability self-check

A scoreboard that does not react to the floor cannot pick one.
`--discriminability-check` perturbs the floor to `0.30` (known to drop
positives) and requires:

* at least **15%** of the positives — and at least **2** of them — to drop out.
  **Zero dropped ⇒ "this set has no discriminability" ⇒ exit 1.** The seed set
  fails exactly here, which is why it is only a smoke set.
* **at least 8 negatives** (warning, written into the JSON): with 4 negatives
  the FPR step is 0.25, so "FPR fell to 0" would mean "one negative fewer" —
  statistically near-meaningless.
* the **median** positive rerank score at least **0.2** away from the current
  floor (warning): a set hugging the floor is both sensitive and brittle.

The positive scores are estimated by **bisection on the floor** (the search
result does not expose the rerank score, and the candidate set is `_kb`
internals — a black-box probe only depends on `search()`'s public contract and
cannot drift with the implementation).

## What is committed, and what is not

The **real** set — built from actual user questions — is not committed,
because it contains personal context and credential fragments. The reason is
the **content**, not the size. What *is* committed is the **desensitised
copy**: the private note titles are replaced by placeholders
(`<PROJECT>`, `<MODEL>`, `<HOST>`, `<PRODUCT>`, `<HW>`), with the sentence
structure and the semantic structure kept intact.

Two rules keep the copy trustworthy:

1. the desensitised set corresponds **one-to-one** with the original and
   differs from it **only** by placeholders;
2. the desensitised set's P@1 is within **0.05** of the original's — if
   anonymising changed the difficulty, the copy is not a copy.

Measured (same corpus, same floor 0.08, original vs desensitised back to back
in one process): **Δ P@1 = 0.000 and Δ FPR = 0.000 on both sets** — the
anonymisation costs nothing measurable.

### Where the placeholders go, and why not in the queries

**The note titles are anonymised in full** — they are the user's private file
names ("联想 ThinkServer SR588 服务器 CPU 支持规格"), and they are the thing
that must not enter the repository.

**The queries are anonymised only where the entity is not already in this
repository.** That is a measured decision, not a preference:

* Replacing the entity *inside a query* removes that query's anchor. "dsh 升到
  哪个版本了" → "<PROJECT> 升到哪个版本了" no longer points at anything: the
  note body still says "dsh". Measured on the full-anonymisation version:
  **P@1 1.0 → 0.643 (calibration set) and 1.0 → 0.778 (acceptance set)** —
  far past the 0.05 criterion, because five of the fourteen queries have no
  other anchor. A copy that measures something else is not a copy.
* The entities that *are* already in this repository (the project codes
  `Hermes`, `dsh`, `WorkBuddy`, `MiMo`, `browser-skill`, `ZCode`, `Octop`,
  `HGM` and the company↔home VPN note title — all in
  `retrieval_gold.seed.json`) would gain **no** privacy from being replaced
  again, and would cost the measurement. They stay.
* What is genuinely new does get replaced: the one project code that appears
  nowhere else in the repository becomes `<PROJECT>`, and the home router
  model becomes `<HOST>`.

Public protocol and product names (WireGuard, RouterOS, Windows, Python, ZFS,
TrueNAS, Unraid, iPhone) are kept — they identify no one, and removing them
would strip the semantic structure the set exists to preserve.

The mapping itself (entity → placeholder) is **not** committed: the keys *are*
the sensitive part. Keep it local and pass it when you run a committed set
against your own store — the returned titles are mapped to placeholders before
matching, so the comparison still works:

```bash
python scripts/kb_eval.py --gold evals/retrieval_gold.acceptance.json --set B \
    --title-map "$HERMES_HOME/evals/entity_map.json"
```

`evals/entity_map.example.json` shows the format with dummy keys.

⚠️ An earlier caveat claimed the negative *"analyse project X"* sat on the
0.08 boundary and made acceptance FPR alternate between 0.0 and 0.25. That was
**not reproduced**: at floors 0.08 / 0.11 / 0.12 it was 0/8 false positives, and
the acceptance set was 0.0 across ~20 observations. The earlier 0.0/0.25 reading
came from a different/noisier setup, not from a floor-boundary property.

⚠️ The seed set predates this policy and still carries the raw note titles
(it was committed before the desensitisation rule existed). It is the smoke
set; the two sets below are the anonymised ones.

## Dedup rules

Applied by `scripts/eval_collect.py` before a set is committed, across sets:

* **query-key dedup is cross-set**: the query is normalised (case, full/half
  width, punctuation and whitespace, entities — hostname / IP / path / URL /
  e-mail / version / date-time — and time words) and the same key is kept only
  once, **seed set first**.
* **the per-title cap is per set**: one target note may contribute at most
  **2** queries to a scoreboard, so a single note cannot monopolise the
  metric. Making this cap global instead would starve the later set (measured:
  the acceptance set dropped from 18 to 5 entries).

Measured on the current sets: seed 21 → 21, A 22 → 22, B 18 → 13 (4 queries
were exact duplicates of seed queries; 1 was the third query for the same
target note).

## Artefacts

| File | What it is |
|------|------------|
| `retrieval_gold.seed.json` | the smoke set (raw titles, committed before the desensitisation policy). |
| `retrieval_gold.calibration.json` | **A** — the desensitised independent set (14 positives + 8 negatives). The only set allowed to pick a floor. |
| `retrieval_gold.acceptance.json` | **B** — the desensitised real set (9 positives + 4 negatives, after dedup). The only set the gate reports on. |
| `baseline.acceptance.json` | the acceptance baseline (set B at the chosen floor), with the set identity inside — `--baseline` refuses it for any other set. |
| `baseline.json` | the historical smoke-set baseline (seed set). Kept for reference; it is **not** a gate any more. |
| `entity_map.example.json` | the shape of the entity → placeholder map (dummy keys). |

## Affinity on/off (both sets, floor 0.08, real usage table)

| Set | affinity | P@1 | recall@5 | MRR | FPR |
|-----|----------|-----|----------|-----|-----|
| A (calibration) | off | 1.0 | 1.0 | 1.0 | 0.125 |
| A (calibration) | **on** | **0.857** | 1.0 | 0.929 | 0.125 |
| B (acceptance) | off | 1.0 | 1.0 | 1.0 | 0.0 |
| B (acceptance) | **on** | **0.889–1.0** | 1.0 | 0.917–1.0 | 0.0 |

A drops **stably**; B drops **intermittently** (5 of 7 runs were identical to
off). Across ~20 observations affinity was **never better** than off. The
mechanism is not "unreproducible counting": ``search()`` only reads usage and
does not touch it, and a populated 29-row table gives stable results. The real
reason to keep it off is **zero benefit + more fragility** — the multiplier is
capped at 1.15, it pulls near-ties closer, and MMR then reorders them. The
earlier "ON and OFF are identical" reading came from an **empty** usage table;
populate it before believing any affinity number.
