# Hermes Memory Governed

A governed memory provider plugin for Hermes Agent. Combines handwritten L1 rules, semantic L2 recall, conversation L3 archives, L4 persona compression, optional Tencent-style auto-extraction, and Mermaid short-term compression into a single high-performance MemoryProvider.

## Design Goals

1. **Fast reads** — prefetch is async (background threads), returns cached results in <1ms
2. **Complete content** — L1 hand-written rules always injected, L2/L3 parallel recall, L4 persona in system prompt
3. **No latency impact** — sync_turn is non-blocking (L3 sync <10ms, L2/extraction async)

## Architecture

```
Read path:   queue_prefetch() → parallel L2+L3+L4 → cache → prefetch() <1ms
Write path:  sync_turn() → L3 sync + L2/extraction async queue
Maintenance: cron → L4 generation, Bridge export, Health report
```

## Layers

| Layer | Purpose | Storage | Trust |
|-------|---------|---------|-------|
| L1 | Handwritten rules, preferences, constraints | MEMORY.md / USER.md | Highest (human) |
| L2 | Semantic facts, auto-extracted | LanceDB + embeddings | Medium |
| L3 | Full conversation archive | SQLite FTS5 | Source of truth |
| L4 | Persona summary | persona.md | Stable summary |
| Bridge | Reviewed durable candidates | JSONL | Review before import |

## Quick Start

### Agent Install (recommended)

```bash
# Basic (L3 + L4 + Bridge)
pip install hermes-memory-governed

# With L2 vector search
pip install hermes-memory-governed[vector]
```

Then in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: governed
```

### Script Install

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

## See Also

- [docs/architecture.md](docs/architecture.md) — full architecture
- [docs/install.md](docs/install.md) — installation guide
- [docs/configuration.md](docs/configuration.md) — all config options
- [docs/operations.md](docs/operations.md) — cron order and operations
