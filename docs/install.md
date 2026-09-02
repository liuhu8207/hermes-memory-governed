# Installation

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

### 1. Clone or copy the project

```bash
git clone https://github.com/your-org/hermes-memory-governed.git
cd hermes-memory-governed
```

### 2. Copy plugin files

```bash
mkdir -p ~/.hermes/plugins/governed
cp plugin/memory_governed/*.py ~/.hermes/plugins/governed/
```

### 3. Copy scripts

```bash
cp scripts/*.py ~/.hermes/scripts/
```

### 4. Create L1 files

```bash
mkdir -p ~/.hermes/memory
cat > ~/.hermes/memory/MEMORY.md << 'EOF'
# Memory Rules
## Project Rules
<!-- Add rules here -->
## Behavioral Rules
<!-- Add constraints here -->
EOF

cat > ~/.hermes/memory/USER.md << 'EOF'
# User Profile
## Identity
<!-- Name, role, timezone -->
## Preferences
<!-- Communication style, tools -->
EOF
```

### 5. Create config

```bash
cp config/governed_memory.example.json ~/.hermes/governed_memory.json
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
