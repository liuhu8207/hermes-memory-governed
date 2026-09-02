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
