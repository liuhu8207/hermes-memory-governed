# Architecture

## Layered Memory Architecture

```
┌─────────────────────────────────────────────────────┐
│  Hermes Agent (conversation loop + tools)            │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  GovernedMemoryProvider (MemoryProvider ABC)  │   │
│  │                                               │   │
│  │  Read Path (fast)                             │   │
│  │  ├── L1 cache (in-memory, <1ms)              │   │
│  │  ├── L2 vector search (LanceDB, ~50ms)       │   │
│  │  ├── L3 FTS5 search (SQLite, ~30ms)          │   │
│  │  └── L4 persona cache (in-memory, <1ms)      │   │
│  │                                               │   │
│  │  Write Path (non-blocking)                    │   │
│  │  ├── L3 sync write (SQLite WAL, <10ms)       │   │
│  │  ├── L2 async index (background thread)       │   │
│  │  └── Fact extraction (optional, async)        │   │
│  │                                               │   │
│  │  Maintenance (cron, deterministic scripts)    │   │
│  │  ├── L4 persona generation (daily)            │   │
│  │  ├── Bridge candidate export (daily)          │   │
│  │  └── Health report (daily)                    │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Read Path: Async Prefetch

The key optimization: `queue_prefetch()` runs after each turn, `prefetch()` returns cached results before the next API call.

```
Turn N ends
  └── queue_prefetch(query for turn N+1)
        ├── [Thread 1] L2 vector search (~50ms)
        ├── [Thread 2] L3 FTS5 search (~30ms)
        └── [Thread 3] L4 persona read (<1ms)
              └── Results merged → _prefetch_cache

Turn N+1 starts
  └── prefetch(query)
        └── _prefetch_cache[query] → <1ms return
```

If cache misses, prefetch falls back to L1 only (fastest path).

## Write Path: Non-blocking Pipeline

```
sync_turn(messages)
  ├── L3 SQLite WAL write (synchronous, <10ms)
  └── _write_queue.put(messages) → background thread
        ├── L2 LanceDB indexing
        └── Fact extraction (optional)
```

L3 is synchronous because it's the source of truth and must not lose data.
L2 and extraction are async because they're best-effort indexes.

## Token Budget Allocation

```
[system prompt]
  └── L4 persona.md           fixed, never displaced

[context injection - prefetch]
  ├── L1 hand-written rules   fixed budget (800 tokens)
  └── L2 + L3 shared          dynamic allocation (1200 tokens)
        ├── L2 by relevance score
        └── L3 by relevance × time decay
```

L1 is never displaced by L2/L3 results.

## Bridge Review Gate

Auto-extracted facts go through a review gate:

```
Extracted fact
  ├── confidence >= 0.8 → direct L2 write
  ├── confidence >= 0.5 → Bridge candidate (JSONL, reviewable)
  └── confidence < 0.5 → discarded
```

Bridge candidates can be reviewed before importing into Scope Recall.
