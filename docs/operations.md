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
