# Retrieval evaluation

`retrieval_gold.seed.json` is a small, curated set of queries written the way a
person actually asks, not as note titles. It exists because "feed a note its own
title" only proves the channel is alive; it does not prove retrieval works.

Two kinds of rows:

- `expect: ["笔记标题", ...]` — a positive query. Metrics: P@1, recall@5, MRR.
- `expect: []` — a negative query. It should return nothing above the configured
  floors; this is what stops "always return something" from looking like recall.

Run:

```bash
python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json
python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json --save-baseline evals/baseline.json
python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json --baseline evals/baseline.json
```

The last form exits non-zero if P@1 / recall@5 / MRR drops more than
`--max-drop` (default 0.02), or if the negative false-positive rate rises.

A curated set built from real questions belongs under
`$HERMES_HOME/evals/retrieval_gold.real.json`, **not in this repository** — the
raw candidates contain personal context and occasional credential fragments.

`scripts/eval_collect.py` pulls recent **real user questions** from L3 into a
candidate file. It is read-only and writes nothing back to the archive. Curate a
few of them into the gold set every few weeks:

```bash
python scripts/eval_collect.py --days 14
```
