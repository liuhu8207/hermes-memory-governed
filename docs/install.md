# Installation

## 0. First: is it already installed? (read this before anything else)

This system is **one authoritative store that several agents share**. A machine
needs it installed **once**. After that, another agent **attaches** — it does not
install again. Follow the numbered sections below only when the answer is "not
installed yet".

Works from anywhere, before you have even cloned this repo. `HERMES_HOME` may be
unset, so all three locations the installer can use are checked:

```bash
for d in "${HERMES_HOME:-}" "$HOME/.hermes" "${LOCALAPPDATA:-}/hermes"; do
  [ -n "$d" ] && [ -f "$d/memory/MEMORY.md" ] && echo "already installed: $d"
done
```

```powershell
# Windows, same three locations
@($env:HERMES_HOME, "$HOME\.hermes", "$env:LOCALAPPDATA\hermes") |
  Where-Object { $_ -and (Test-Path "$_\memory\MEMORY.md") } |
  ForEach-Object { "already installed: $_" }
```

**Any line printed means it is already installed** → stop, and read *Attaching
another agent* below. No output means a fresh machine → §1.

Once you have the repo, this gives the same answer with more detail:

```bash
python memory_cli.py health
```

| Result | What it means |
|---|---|
| Prints JSON with `"ok": true` | **Already installed.** Stop here — go to *Attaching another agent* below. Do **not** run §1–§4. |
| Command not found, or `"ok": false` | Not usable yet — continue with §1. |

### ⛔ On a machine that already has it, never do these

Each one overwrites something that cannot be regenerated:

| Don't | Why |
|---|---|
| `cat > …/memory/MEMORY.md` | **Overwrites L1 — the hand-written rules.** Those are authored by a human, not derived; there is no way to recover them from the store. |
| `cat > …/memory/USER.md` | Same. |
| `cp config/governed_memory.example.json …/governed_memory.json` | **Overwrites the live config** (embedding backend, credentials, `wiki_dir`). |
| `cp plugin/memory_governed/*.py …/plugins/governed/` | Overwrites the plugin a running process is using — and may **downgrade** it. |
| Running `pip install` a second time "to be safe" | Reinstalling does not migrate data, and is not how you upgrade — see §4. |

`install.sh` / `install.ps1` guard every one of those writes with `if [ ! -f … ]`,
so running the script on an installed machine is safe. **The risk is copying the
commands in §3 by hand** — they are written without the guards, for brevity.

### Attaching another agent

A second agent does not install anything. It talks to the store that is already
there. Entry points, identity rules and a verification checklist are in
[attach-agents.md](attach-agents.md).

---

## 1. Agent Install (Recommended)

### Via pip (with Hermes Agent plugin discovery)

```bash
# Basic install (L1/L3/L4/Bridge — no vector search)
pip install hermes-memory-governed

# With L2 vector search (recommended for full functionality)
pip install hermes-memory-governed[vector]
```

The plugin is automatically discovered by Hermes Agent via the `hermes.memory_provider` entry point. After install, set the active provider in `~/.hermes/config.yaml`:

```yaml
memory:
  provider: governed
```

### Via pipx (isolated environment)

```bash
pipx install hermes-memory-governed
# or with vector support:
pipx install "hermes-memory-governed[vector]"
```

## 2. One-Command Install Scripts

### Linux / macOS

```bash
bash install.sh
```

Options:

```bash
# Custom HERMES_HOME
bash install.sh --hermes-home ~/.hermes

# With L2 vector search
bash install.sh --with-vector

# Auto-setup cron jobs
bash install.sh --configure-cron

# All options
bash install.sh --hermes-home ~/.hermes --wiki-dir ~/wiki --with-vector --configure-cron
```

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

Options:

```powershell
# Custom HERMES_HOME
powershell -ExecutionPolicy Bypass -File install.ps1 -HermesHome "$env:LOCALAPPDATA\hermes"

# Include cron job setup
powershell -ExecutionPolicy Bypass -File install.ps1 -ConfigureCron
```

## 3. Manual Install

> ⚠️ **Only on a machine that does not have it yet.** These steps write to the
> same paths as §1–§2 and are shown without the guards the scripts use. If
> `python memory_cli.py health` already prints `"ok": true`, running step 4 or 5
> will **destroy the hand-written L1 rules and the live config** — see §0.

### 1. Clone or copy the project

```bash
git clone https://github.com/your-org/hermes-memory-governed.git
cd hermes-memory-governed
```

### 2. Copy plugin files

Back up first if the target directory already exists — this overwrites it:

```bash
mkdir -p ~/.hermes/plugins/governed
[ -d ~/.hermes/plugins/governed ] && mv ~/.hermes/plugins/governed \
    ~/.hermes/plugins/governed.bak-$(date +%Y%m%d%H%M%S)
mkdir -p ~/.hermes/plugins/governed
cp plugin/memory_governed/*.py ~/.hermes/plugins/governed/
```

### 3. Copy scripts

```bash
mkdir -p ~/.hermes/scripts/
cp scripts/*.py ~/.hermes/scripts/
```

### 4. Create L1 files

Guarded, because these are the files a human authors:

```bash
mkdir -p ~/.hermes/memory
if [ ! -f ~/.hermes/memory/MEMORY.md ]; then
cat > ~/.hermes/memory/MEMORY.md << 'EOF'
# Memory Rules
## Project Rules
<!-- Add rules here -->
## Behavioral Rules
<!-- Add constraints here -->
EOF
fi

if [ ! -f ~/.hermes/memory/USER.md ]; then
cat > ~/.hermes/memory/USER.md << 'EOF'
# User Profile
## Identity
<!-- Name, role, timezone -->
## Preferences
<!-- Communication style, tools -->
EOF
fi
```

### 5. Create config

```bash
if [ ! -f ~/.hermes/governed_memory.json ]; then
    cp config/governed_memory.example.json ~/.hermes/governed_memory.json
fi
# Edit paths as needed
```

### 6. Set as active provider

In `~/.hermes/config.yaml`:

```yaml
memory:
  provider: governed
```

### 7. Install Python dependencies

```bash
# Basic (L3 + L4 + Bridge)
pip install httpx

# Full (with L2 vector search)
pip install lancedb sentence-transformers
```

### 8. Verify

```bash
python ~/.hermes/scripts/memory_pipeline.py health
```

## 4. Upgrade

```bash
pip install --upgrade hermes-memory-governed
# or with vector:
pip install --upgrade "hermes-memory-governed[vector]"
```

## Dependencies

| Package | Version | Required | Purpose |
|---------|---------|----------|---------|
| Python | >= 3.10 | Yes | Runtime |
| httpx | >= 0.28 | Yes | LLM calls in persona generation |
| lancedb | >= 0.17 | Optional | L2 semantic memory (vector search) |
| sentence-transformers | >= 2.0 | Optional | L2 embedding generation |
| SQLite3 | built-in | Yes | L3 conversation archive (FTS5) |
