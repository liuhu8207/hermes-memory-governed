# Hermes Memory Governed

A **governed memory provider plugin** for [Hermes Agent](https://github.com/hermes-agent). It fuses a four-layer memory stack (L1 handwritten rules, L2 semantic recall, L3 conversation archive, L4 persona), an Obsidian-backed knowledge base with auto-distillation and a review gate, plus maintenance tooling into a single high-performance `MemoryProvider`.

> 中文说明见 [README_CN.md](README_CN.md)。

[![CI](https://github.com/liuhu8207/hermes-memory-governed/actions/workflows/ci.yml/badge.svg)](https://github.com/liuhu8207/hermes-memory-governed/actions)

---

## Table of Contents

- [What it does](#what-it-does)
- [Design goals](#design-goals)
- [Architecture](#architecture)
  - [Layered memory stack](#layered-memory-stack)
  - [Read path — async prefetch](#read-path--async-prefetch)
  - [Write path — non-blocking pipeline](#write-path--non-blocking-pipeline)
  - [Token budget](#token-budget)
- [Knowledge base & distillation](#knowledge-base--distillation)
  - [Three pipelines](#three-pipelines)
  - [Review gate](#review-gate)
- [Tools](#tools)
- [Module map](#module-map)
- [Installation](#installation)
- [Configuration](#configuration)
- [Operations](#operations)
- [Testing & CI](#testing--ci)
- [See also](#see-also)

---

## What it does

The plugin gives Hermes a **governed** memory system — meaning nothing is written silently and nothing displaces human-authored facts. Concretely:

1. **Never forgets the human rules** — L1 (`MEMORY.md` / `USER.md`) is always injected, with a fixed token budget that L2/L3 results can never displace.
2. **Recalls semantically** — L2 does vector search over durable facts; L3 does full-text search over the full conversation archive; L4 keeps a stable persona in the system prompt.
3. **Never blocks the conversation** — the read path is async-prefetched and the write path is non-blocking; L3 (source of truth) is synchronous and fast, everything else is queued.
4. **Accumulates knowledge, not just chat** — conversations are auto-distilled into Obsidian notes through a confidence gate and a review gate, so durable knowledge survives as plain Markdown you own.

## Design goals

| # | Goal | Mechanism |
|---|------|-----------|
| 1 | **Fast reads** | `queue_prefetch()` prefetches in background threads; `prefetch()` returns cached results in **<1ms** |
| 2 | **Complete content** | L1 always injected; L2/L3/L4 recalled in parallel; L4 persona lives in the system prompt |
| 3 | **No latency impact** | `sync_turn()` is non-blocking — L3 sync write **<10ms**, L2 indexing & extraction are async |
| 4 | **No silent writes** | Facts flow through confidence gates, secret gates, and a review gate; failures are reported, never swallowed |

## Architecture

### Layered memory stack

```
┌────────────────────────────────────────────────────────────┐
│                      Hermes Agent                          │
│            (conversation loop + tool dispatch)             │
│                                                            │
│   ┌────────────────────────────────────────────────────┐   │
│   │        GovernedMemoryProvider (MemoryProvider ABC)  │   │
│   │                                                    │   │
│   │   READ PATH (fast)                                 │   │
│   │     L1  hand-written rules   → in-memory  <1ms     │   │
│   │     L2  semantic recall      → LanceDB    ~50ms    │   │
│   │     L3  full-text archive    → SQLite FTS5 ~30ms   │   │
│   │     L4  persona              → in-memory  <1ms     │   │
│   │                                                    │   │
│   │   WRITE PATH (non-blocking)                        │   │
│   │     L3  sync write           → SQLite WAL  <10ms   │   │
│   │     L2  async index          → background thread   │   │
│   │     extraction               → background thread   │   │
│   │                                                    │   │
│   │   MAINTENANCE (cron scripts, deterministic)        │   │
│   │     L4 persona generation    → daily               │   │
│   │     Bridge candidate export  → daily               │   │
│   │     Health report            → daily               │   │
│   └────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

| Layer | Purpose | Storage | Trust |
|-------|---------|---------|-------|
| **L1** | Hand-written rules, preferences, constraints | `MEMORY.md` / `USER.md` | Highest (human-authored) |
| **L2** | Durable semantic facts | LanceDB + embeddings | Medium (auto-extracted) |
| **L3** | Complete conversation archive | SQLite FTS5 | Source of truth |
| **L4** | Persona summary | `persona.md` | Stable summary |
| **Bridge** | Reviewed durable candidates | JSONL | Review-before-import |
| **KB** | Durable knowledge notes | Obsidian vault (Markdown) | You own it |

### Read path — async prefetch

The core latency optimization: after every turn, the next turn's recall is prefetched in the background, so the subsequent read is a cache hit.

```
Turn N ends
  └── queue_prefetch(query for turn N+1)
        ├─ [Thread 1] L2 vector search   (~50ms)
        ├─ [Thread 2] L3 FTS5 search     (~30ms)
        └─ [Thread 3] L4 persona read    (<1ms)
             └── results merged → _prefetch_cache

Turn N+1 starts
  └── prefetch(query)
        └── _prefetch_cache[query]  →  <1ms
```

On a cache miss, `prefetch()` falls back to the L1-only path (fastest, never fails).

### Write path — non-blocking pipeline

```
sync_turn(messages)
  ├── L3 SQLite WAL write   (synchronous, <10ms, thread-safe via RLock)
  │     └── mirrored into messages_fts (divergence reported via _diag)
  └── _write_queue.put(messages)  → background worker
        ├── L2 LanceDB indexing
        └── fact extraction (optional)
              └── confidence ≥ 0.8  → direct L2 write
                  confidence ≥ 0.5  → Bridge candidate (JSONL, reviewable)
                  confidence <  0.5 → discarded
```

L3 is synchronous because it is the source of truth and must never lose data. L2 and extraction are best-effort indexes and are safely queued (the worker drains with per-item error isolation at exit).

### Token budget

```
[system prompt]
  └── L4 persona.md            fixed, never displaced

[context injection — prefetch]
  ├── L1 hand-written rules    fixed budget (800 tokens)
  └── L2 + L3 shared           dynamic (1200 tokens)
        ├── L2  by relevance score
        └── L3  by relevance × time decay
```

L1 is never displaced by L2/L3 results — the human-authored baseline always wins.

## Knowledge base & distillation

The KB is an **Obsidian vault** — Markdown + frontmatter + `[[wikilink]]`. The vector index (`KBIndex`) is only a *projection*; it can be rebuilt at any time via `KnowledgeBase.reindex`, and any agent or tool can read/write the vault directly, independent of this plugin.

```
inbox/      low-confidence entries awaiting human confirmation
notes/      atomic Zettelkasten cards (approved knowledge)
projects/   PARA — projects with deliverables
areas/      PARA — long-term responsibilities
resources/  PARA — external references
archive/    PARA — archived items
index.md    MOC navigation entry
```

### Three pipelines

| Pipeline | Carrier | Trigger | Mechanism |
|----------|---------|---------|-----------|
| **Memory** | L1–L4 (`MEMORY.md` / LanceDB / SQLite / persona) | `sync_turn` auto | unchanged, always on |
| **Knowledge** | Obsidian vault | `on_session_end` auto (optional) | semi-auto distillation |
| **Ingest** | fetch / read / transcribe → distill → `kb_add` | manual | content acquisition |

### Review gate

The distillation flow is deliberately **semi-automatic** — it never writes junk straight into the permanent notes:

```
Conversation ends (on_session_end)
      ↓ transcript (truncated 24k)
LLM distillation (same model as the agent's chat model)
      ↓ [{title, body, tags, concepts, confidence}]
Secret gate (reuse _bridge.SECRET_PATTERNS, fail-closed)
      ├─ contains a secret → refuse to write (explicit error)
      └─ clean →
           ├─ confidence ≥ 0.7 → notes  (permanent)
           └─ confidence <  0.7 → inbox  (review_required)
      ↓
Link completion (shared concepts/tags → mutual [[wikilink]], notes only, deduped)
      ↓
review gate (governed_kb_review: list → approve → notes / reject → archive)
```

Key architectural decision: distillation uses the **same model id and endpoint** as the agent's chat model (not "the agent"), because the plugin runs in the gateway background process and has no handle to the live agent instance.

## Tools

The plugin registers **10 tools** (3 memory + 7 KB):

| Tool | Category | Purpose |
|------|----------|---------|
| `governed_search` | Memory | Search L2/L3 for relevant memory |
| `governed_audit` | Memory | Drill down into memory provenance |
| `governed_health` | Memory | Memory system health report |
| `governed_kb_search` | KB | Search notes (semantic + keyword + backlinks) |
| `governed_kb_add` | KB | Write/update a note (with governance) |
| `governed_kb_get` | KB | Fetch a note's full body |
| `governed_kb_review` | KB | Approve/reject inbox candidates |
| `governed_kb_fetch` | KB | Fetch a URL → text |
| `governed_kb_read_file` | KB | Read a local file → text |
| `governed_kb_transcribe` | KB | Transcribe audio → text (ASR; long audio auto-split) |

## Module map

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | `GovernedMemoryProvider` entry, tool registration, `register(ctx)` |
| `_config.py` | Config loader (`governed_memory.json` + env fallback) |
| `_recall.py` | Parallel read engine (`queue_prefetch` / `prefetch`), score mapping |
| `_sync.py` | Async write pipeline (`sync_turn`, `WriteQueue`, L3 FTS5 mirror) |
| `_embedding.py` | Embedding backend abstraction (`auto`/`fastembed`/`sentence_transformers`/`none`) |
| `_kb.py` | Knowledge base facade (search/add/get/review/reindex) + `KBIndex` |
| `_vault.py` | Obsidian vault store (PARA skeleton, frontmatter, wikilinks) |
| `_synthesize.py` | Session distillation → candidate cards (LLM, same model) |
| `_ingest.py` | Content acquisition (fetch/read/transcribe incl. long-audio auto-split, all optional-dep degradable) |
| `_bridge.py` | Durable-memory candidates for Scope Recall (secret patterns, export) |
| `_compress.py` | Mermaid short-term compression for long tasks |
| `_migrations.py` | Idempotent schema migrations (`run_pending`) |
| `_diag.py` | Divergence / degradation reporting (never silent) |

### Embedding backends

`vector.backend` selects the L2 embedding strategy, all behind one `Embedder` protocol:

| Backend | Description | Notes |
|---------|-------------|-------|
| `auto` | try fastembed → sentence_transformers → none | default, graceful degradation |
| `fastembed` | qdrant fastembed (ONNX, no PyTorch) | lightest, model auto-downloaded |
| `sentence_transformers` | sentence-transformers (torch) | original backend |
| `none` | no embeddings | L2 falls back to text scan |

## Installation

> **Already installed on this machine?** Then **do not reinstall.** This is one
> shared store, installed once. Another agent attaches to it instead — and
> re-running the manual steps would overwrite the hand-written L1 rules.
> Check with `python memory_cli.py health` and read
> [docs/install.md](docs/install.md) §0 where the answer is `"ok": true`.
> For wiring up a second (non-Hermes) agent, see
> [docs/attach-agents.md](docs/attach-agents.md).

### Agent install (recommended)

```bash
# Basic (L3 + L4 + Bridge + KB)
pip install hermes-memory-governed

# With L2 vector search
pip install hermes-memory-governed[vector]
```

Then enable it in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: governed
```

### Script install

```bash
# Linux / macOS
bash install.sh

# Windows
powershell -ExecutionPolicy Bypass -File install.ps1
```

### Verify

```bash
python scripts/memory_pipeline.py health
```

## Configuration

Configuration lives in `$HERMES_HOME/governed_memory.json` (see [`config/governed_memory.example.json`](config/governed_memory.example.json) for the full annotated example). Key sections:

| Section | Controls |
|---------|----------|
| `recall` | L1/L23 token budgets, prefetch TTL, L3 time decay, parallel timeout |
| `sync` | L3 sync vs async, L2 async, extraction toggle |
| `persona` | L4 incremental generation + interval |
| `tencent_extract` | optional Tencent-style extraction + confidence threshold |
| `mermaid_compress` | Mermaid short-term compression + canvas token cap |
| `vector` | L2 backend, model, dimension |
| `embedding` | remote embedding API (OpenAI-compatible `/embeddings`) |
| `reranking` | optional reranker (SiliconFlow bge-reranker) |
| `kb` | KB top-k, thresholds, semantic/keyword toggles, autolink |
| `asr` | audio transcription (XingChenASR) |
| `synthesis` | session distillation (provider/model/key, enabled flag) |

> ⚠️ **Dimension consistency**: a given L2 table can hold only one vector dimension. Switching backends (API ↔ local) or models changes the dimension, so you must rebuild L2: `python scripts/l2_rebuild.py`.

### HuggingFace in China

Local embeddings need these two environment variables (xet CDN returns 401 without them):

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

## Operations

Recommended daily cron schedule:

| Time | Task | Script |
|------|------|--------|
| 03:40 | Wiki → NotebookLM → L2 | `wiki_nblm_daily.py` |
| 04:25 | Wiki topics → induction → L2 | `wiki_nblm_induction.py` |
| 09:10 | L4 persona generation | `l4_persona_daily.py` |
| 09:15 | Bridge candidate export | `scope_recall_bridge.py export` |
| 09:45 | Health report | `memory_health_report.py` |

### Unified pipeline

```bash
python scripts/memory_pipeline.py daily          # daily maintenance
python scripts/memory_pipeline.py bridge         # export Bridge candidates
python scripts/memory_pipeline.py bridge --dry-run
python scripts/memory_pipeline.py health         # health report
python scripts/memory_pipeline.py persona        # regenerate L4 persona
```

### Bridge review

```bash
python scripts/memory_pipeline.py bridge --dry-run   # preview
python scripts/scope_recall_bridge.py status          # check status
python scripts/scope_recall_bridge.py validate        # validate candidates
```

## Testing & CI

- Local full suite: `python -m pytest tests/ -q` → **1327 passed** (2026-09-18).
- CI (`[.github/workflows/ci.yml](.github/workflows/ci.yml)`) runs a **4-dimension matrix** — `py3.10` / `py3.13` × `core` / `vector`:
  - `core` installs `.[dev]` only (no vector backend → exercises the degradation path).
  - `vector` installs the lightweight backend (`lancedb` + `pyarrow` + `fastembed`, **not** `sentence-transformers`, which would pull ~2GB of torch).

## See also

- [docs/architecture.md](docs/architecture.md) — full architecture
- [docs/hybrid-architecture.md](docs/hybrid-architecture.md) — cloud + local hybrid deployment design
- [docs/phase3-design.md](docs/phase3-design.md) — knowledge distillation design
- [docs/configuration.md](docs/configuration.md) — all config options
- [docs/install.md](docs/install.md) — installation guide (start at §0: already installed?)
- [docs/attach-agents.md](docs/attach-agents.md) — attaching a second agent: CLI, MCP tools, host hooks
- [docs/operations.md](docs/operations.md) — cron order and operations
- [docs/acceptance-report.md](docs/acceptance-report.md) — acceptance verification

## License

MIT
