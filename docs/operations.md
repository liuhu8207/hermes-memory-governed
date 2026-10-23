# Operations

## Cron Jobs

Recommended daily schedule:

| Time | Task | Script | Agent |
|------|------|--------|-------|
| 03:40 | Wiki → NotebookLM → L2 | wiki_nblm_daily.py | no_agent |
| 04:25 | Wiki topics → induction → L2 | wiki_nblm_induction.py | no_agent |
| 09:10 | L4 persona generation | l4_persona_daily.py | no_agent |
| 09:15 | Bridge candidate export | scope_recall_bridge.py export | no_agent |
| 09:45 | Health report | memory_health_report.py | no_agent |

Setup:

```bash
hermes cron add --name l4-persona \
  --schedule "0 9 * * *" \
  --script scripts/l4_persona_daily.py \
  --no-agent

hermes cron add --name bridge-export \
  --schedule "0 9 * * *" \
  --script scripts/scope_recall_bridge.py export \
  --no-agent

hermes cron add --name health-report \
  --schedule "0 10 * * *" \
  --script scripts/memory_health_report.py \
  --no-agent
```

### Deployed cron scripts

A cron job's `script` must be a filename under `$HERMES_HOME/scripts`, so the
deployed copy cannot infer the checkout from its own path. Set `source_repo` in
`governed_memory.json` (or `HGM_REPO`):

```json
"source_repo": "D:/path/to/hermes-memory-governed"
```

`cron_memory_daily.py` resolves it once, exports `HGM_REPO` to every child step,
and clears the gateway's inherited `PYTHONPATH`/`PYTHONHOME` so the repo venv
does not load incompatible compiled wheels.

## Destructive Maintenance — Safety Gates

These scripts delete or rewrite data. Each one refuses rather than guessing, so
a non-zero exit here means "the guard fired", not "the script is broken".

| Script | Refuses when | Override |
|--------|--------------|----------|
| `kb_reindex.py` | the vault scans as empty (usually a mount or `wiki_dir` problem) — syncing would clear every index row | `--allow-empty` |
| `l2_capacity.py` | the pre-delete table backup fails | none — fix the backup target |
| `l2_capacity.py` | there is no usage data at all; without it eviction degrades to "delete the oldest", which has removed a fact that had been recalled | none — let recall populate `l2_usage.db` |
| `redact_l3_archive.py` | never refuses, but keeps only the newest `--keep-backups` (default 3) pre-redaction copies | `--keep-backups N` (minimum 1) |
| `l2_rebuild.py` | **never** lets `--no-gate` bypass the tombstone gate. `--no-gate` turns off only the quality and secret gates; a sentence that was explicitly retracted stays retracted. | none — tombstoned facts cannot be rebuilt |

A pre-redaction backup is a **full plaintext copy of L3**. It is a real restore
point and must exist before an update, but it must not accumulate — that is why
retention is on by default.

```bash
python scripts/kb_reindex.py --dry-run          # show what would sync
python scripts/l2_capacity.py --dry-run         # show rows and overage
python scripts/redact_l3_archive.py --dry-run   # show how many rows would change
```

## Backup Restore Drill

A backup that has never been restored is not a backup. Run this monthly:

```bash
python scripts/restore_drill.py --dry-run   # show which backups would be used
python scripts/restore_drill.py             # restore into a temp dir and verify
```

It copies the newest L2 / L3 / vault backups into a tree under
`$HERMES_HOME/restore_drills/` (not the system temp dir), opens them the way the
runtime does, checks L2 columns, SQLite integrity and vault frontmatter, and
prints row counts. The live store is only read. The tree is deleted unless
`--keep`; kept trees are pruned after `--keep-days` (default 7).

A drill that would need to rebuild the L3 FTS index exits **non-zero**: a backup
that is not directly usable is a bad backup, and cron only reads the exit code.
Pass `--allow-repair` to downgrade that to a warning while investigating.

> The restored tree contains a **full plaintext copy of L3**. Keep it only as
> long as necessary.

If an older L3 backup reports `malformed inverted index for FTS5`, the drill
rebuilds the index **in the restored copy** and reports it as a warning. Repair
the live archive with:

```bash
python scripts/l3_fts_repair.py --dry-run
python scripts/l3_fts_repair.py --yes       # makes its own backup first
```

## Silent-Exception Audit

`# silent-ok:` is the guard's escape hatch; a human still has to check that the
reason is true. List every exemption with its enclosing function:

```bash
python scripts/silent_ok_audit.py
python scripts/silent_ok_audit.py --check   # fails on short/empty reasons
```

Review this list quarterly and tighten the per-file ratchet baseline as sites
are cleaned up.

## Vector Index Maintenance

Every upsert/delete leaves a LanceDB version behind. The real `kb_notes` table
had **10,911 versions for 52 rows** before cleanup. Compact periodically:

```bash
python scripts/kb_index_optimize.py --dry-run
python scripts/kb_index_optimize.py --days 7
```

`--days 7` keeps a week of time-travel history; `--days 0` keeps only the
current version. It also prunes `kb_notes.bak_*` table backups to
`--keep-index-backups` (default 2).

## Retrieval Evaluation

`evals/retrieval_gold.seed.json` is a curated set of realistic queries
(positives with expected notes, plus negative queries that should return
nothing). It measures P@1, recall@5, MRR and the negative false-positive rate.

There are three sets and they have **different jobs** (see
[`evals/README.md`](../evals/README.md) and
[`docs/configuration.md`](configuration.md#choosing-the-rerank-floor-ab-separated-auditable)):

```bash
# smoke: does the channel answer at all? (not a scoreboard, never a baseline)
python scripts/kb_eval.py --gold evals/retrieval_gold.seed.json --set smoke

# A (calibration): pick the rerank floor + prove the set has discriminability
python scripts/kb_eval.py --gold evals/retrieval_gold.calibration.json --set A \
    --select-threshold --discriminability-check

# B (acceptance): report results and compare against the acceptance baseline
python scripts/kb_eval.py --gold evals/retrieval_gold.acceptance.json --set B \
    --baseline evals/baseline.acceptance.json
```

`memory_cli eval` has the same split: it defaults to L2 facts and **auto-selects
KB when the QA file's `expect` is a list**. To be explicit:

```bash
python memory_cli.py eval --qa evals/retrieval_gold.acceptance.json \
    --path kb --title-map "$HERMES_HOME/evals/entity_map.json" --top-k 5
```

Before this split, feeding a KB gold set to `eval --qa` measured L2 facts and
reported `hit@k = 0.0` — a false red, not a retrieval failure.

The baseline form exits non-zero when a metric drops more than `--max-drop`.
It only accepts the **acceptance** set: `--baseline` refuses a calibration or
smoke run, and refuses a baseline file produced from a *different* set — two
numbers from two different sets are not comparable.

Affinity is disabled by default because it updates during each query and makes
the same set score differently every run; pass `--affinity on` (or the legacy
`--with-affinity`) only when you explicitly want the production full path.
`--affinity both` reports both variants in one run, off first.

## Backup Retention

Every ``retract`` creates a full L2 table backup, so one-by-one cleanup can
grow ``l2_backups`` quickly (a 19-fact cleanup grew it from 25 to 46 copies,
~690 MB). The standard is enforced by ``scripts/backup_retention.py`` and runs
as a daily cron step (``backup-retention``):

| Backup | Keep |
|--------|------|
| L2 table directories | newest 7 + newest of each of the last 3 months + named milestones (`cleanup`, `planB`, `pre_restore`) |
| L3 databases | newest 5 |
| KB index backups | 2 |
| Vault `.kb-backups-*` | never kept in the vault; all moved to `$HERMES_HOME/memory/vault_backups/`, where newest 3 are kept |

Timestamps are parsed from the **name**, not mtime (copied directories often
inherit an old mtime). The script is report-only by default; cron passes
``--apply --yes``. Preview manually with:

```bash
python scripts/backup_retention.py --dry-run
```

The first real run reclaimed 544.4 MB (774 MB → 229.6 MB) and left the
milestone backups intact.

## L2 Category Migration

New facts use the same controlled vocabulary as KB navigation. Existing rows
can be normalized with a report-only pass first:

```bash
python scripts/l2_categorize.py --dry-run
python scripts/l2_categorize.py --apply --yes
```

The apply path backs up the full Lance table before changing categories and
reports per-row failures rather than partially hiding them.

## KB Graph Audit

`kb_graph_audit.py` separates three cases that a raw broken-link count conflates:
resolved notes, intentional concept placeholders, and genuinely unknown
targets. It also reports notes with neither incoming nor outgoing graph links.

```bash
python scripts/kb_graph_audit.py --json
python scripts/kb_graph_audit.py --strict  # fail on orphans or unknown targets
```

Concept placeholders are not failures: they preserve a future note target and
remain visible in frontmatter. Only `unknown_links` and `orphans` are strict
failures.

## Test Isolation vs Production Scripts

These are two different situations and must not be conflated:

* **pytest** — the repository-root `conftest.py` points `HERMES_HOME` at a
  temporary directory before any test module imports a script. Scripts also
  refuse a production home inside a pytest session (second line of defence).
* **normal CLI / cron / manual ops** — `bootstrap()` resolves the real
  `HERMES_HOME` and prints a WARNING that it is about to touch production.
  It does **not** refuse: refusing here would stop `kb_reindex`,
  `restore_drill`, `l2_capacity` and every scheduled backup/cleanup.
* **force refusal anyway** — set `HGM_STRICT_ISOLATION=1` (CI or a one-off
  investigation). The gateway launcher also sets `HGM_ALLOW_PRODUCTION=1`
  for its own process tree only.

If a script is invoked without `HERMES_HOME`, it falls back to the platform
default and warns. Set the variable explicitly in cron jobs so the target is
never ambiguous.

## Health Check

```bash
python scripts/memory_pipeline.py health
```

Output includes:
- L1 file existence and freshness
- L2 LanceDB table status
- L3 SQLite table counts and FTS5 status
- L4 persona freshness
- Bridge candidate counts

## Bridge Review

```bash
# Preview candidates
python scripts/memory_pipeline.py bridge --dry-run

# Export candidates
python scripts/memory_pipeline.py bridge

# Check status
python scripts/scope_recall_bridge.py status

# Validate
python scripts/scope_recall_bridge.py validate
```

## Manual Persona Regeneration

```bash
python scripts/memory_pipeline.py persona
```

## Troubleshooting

### L2 not indexing

Check if lancedb is installed:
```bash
python -c "import lancedb; print(lancedb.__version__)"
```

### L3 FTS5 not working

SQLite must be compiled with FTS5 support. Check:
```bash
python -c "import sqlite3; conn = sqlite3.connect(':memory:'); conn.execute('CREATE VIRTUAL TABLE t USING fts5(c)'); print('FTS5 OK')"
```

### Prefetch cache stale

Increase `prefetch_ttl_seconds` in config, or check that `queue_prefetch` is being called.
