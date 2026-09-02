#!/usr/bin/env bash
# Hermes Memory Governed — Installer for Linux / macOS
#
# Usage:
#   bash install.sh                    # interactive install
#   bash install.sh --yes              # skip prompts (default: no vector)
#   bash install.sh --with-vector      # non-interactive with vector
#   bash install.sh --hermes-home ~/.hermes
#   bash install.sh --configure-cron

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
WIKI_DIR="${WIKI_DIR:-$HOME/wiki}"
EMBEDDING_BACKEND=""          # api | local | none
EMBEDDING_PROVIDER=""
EMBEDDING_BASE_URL=""
EMBEDDING_API_KEY_ENV=""
EMBEDDING_MODEL=""
CONFIGURE_CRON=false
SKIP_PROMPTS=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --hermes-home)    HERMES_HOME="$2";     shift 2 ;;
        --wiki-dir)       WIKI_DIR="$2";        shift 2 ;;
        --with-vector)    EMBEDDING_BACKEND="local"; shift ;;
        --without-vector) EMBEDDING_BACKEND="none";  shift ;;
        --embedding-api)  EMBEDDING_BACKEND="api";   shift ;;
        --configure-cron) CONFIGURE_CRON=true;  shift ;;
        --yes|-y)         SKIP_PROMPTS=true;    shift ;;
        -h|--help)
            echo "Usage: bash install.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --hermes-home DIR    Hermes home directory (default: ~/.hermes)"
            echo "  --wiki-dir DIR       Wiki directory (default: ~/wiki)"
            echo "  --with-vector        Use local embedding model (bge-small-zh)"
            echo "  --without-vector     Skip vector search (keyword only)"
            echo "  --embedding-api      Use API embedding backend (no local model)"
            echo "  --configure-cron     Show cron setup instructions"
            echo "  --yes, -y            Skip interactive prompts"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Interactive prompts ───────────────────────────────────────────────────────
if [ "$SKIP_PROMPTS" = false ] && [ -z "$EMBEDDING_BACKEND" ]; then
    echo "=== Hermes Memory Governed Installer ==="
    echo ""
    echo "HERMES_HOME = $HERMES_HOME"
    echo "WIKI_DIR    = $WIKI_DIR"
    echo ""

    echo "Embedding backend for L2 semantic search:"
    echo "  1) API         — OpenAI-compatible /embeddings, no local model (recommended)"
    echo "  2) Local model — bge-small-zh-v1.5 (512-dim, ~100MB download, offline)"
    echo "  3) Skip        — L3 FTS5 keyword search only"
    echo ""
    read -r -p "  Choose [2]: " emb_answer
    case "$emb_answer" in
        1|api)       EMBEDDING_BACKEND="api"  ;;
        3|none|skip) EMBEDDING_BACKEND="none" ;;
        *)           EMBEDDING_BACKEND="local" ;;
    esac
    echo ""

    if [ "$EMBEDDING_BACKEND" = "api" ]; then
        read -r -p "  provider (e.g. siliconflow/openai): " EMBEDDING_PROVIDER
        read -r -p "  base_url (e.g. https://api.siliconflow.cn/v1): " EMBEDDING_BASE_URL
        read -r -p "  model (e.g. BAAI/bge-m3): " EMBEDDING_MODEL
        read -r -p "  API key env var name (key stays in env, e.g. SILICONFLOW_API_KEY): " EMBEDDING_API_KEY_ENV
        echo ""
    fi

    read -r -p "  Show cron job setup instructions? [y/N]: " cron_answer
    case "$cron_answer" in
        [yY]|[yY][eE][sS]) CONFIGURE_CRON=true ;;
        *)                  CONFIGURE_CRON=false ;;
    esac
    echo ""
fi

if [ -z "$EMBEDDING_BACKEND" ]; then
    EMBEDDING_BACKEND="none"
fi

echo "=== Hermes Memory Governed Installer ==="
echo ""
echo "HERMES_HOME       = $HERMES_HOME"
echo "WIKI_DIR          = $WIKI_DIR"
echo "EMBEDDING_BACKEND = $EMBEDDING_BACKEND"
echo ""

# ── 1. Create directories ────────────────────────────────────────────────────
echo "[1/6] Creating directories..."
for d in \
    "$HERMES_HOME/memory" \
    "$HERMES_HOME/memory/l2" \
    "$HERMES_HOME/memory/l3" \
    "$HERMES_HOME/scripts" \
    "$HERMES_HOME/plugins/governed" \
    "$HERMES_HOME/cron/output/scope_recall_bridge" \
    "$WIKI_DIR"; do
    mkdir -p "$d"
done
echo "  Done."

# ── 2. Copy plugin ───────────────────────────────────────────────────────────
echo "[2/6] Installing plugin..."
PLUGIN_SRC="$SCRIPT_DIR/plugin/memory_governed"
PLUGIN_DST="$HERMES_HOME/plugins/governed"
mkdir -p "$PLUGIN_DST"
cp "$PLUGIN_SRC/"*.py "$PLUGIN_DST/"
echo "  Plugin installed to: $PLUGIN_DST"

# ── 3. Copy scripts ──────────────────────────────────────────────────────────
echo "[3/6] Installing scripts..."
cp -r "$SCRIPT_DIR/scripts/"*.py "$HERMES_HOME/scripts/"
echo "  Scripts installed to: $HERMES_HOME/scripts"

# ── 4. Create default L1 files ───────────────────────────────────────────────
echo "[4/6] Creating default L1 files..."

MEMORY_MD="$HERMES_HOME/memory/MEMORY.md"
if [ ! -f "$MEMORY_MD" ]; then
    cat > "$MEMORY_MD" << 'EOF'
# Memory Rules

> Hand-written rules for the AI agent. Edit this file directly.

## Project Rules
<!-- Add your project-specific rules here -->

## Behavioral Rules
<!-- Add things the AI should/shouldn't do here -->
EOF
    echo "  Created: $MEMORY_MD"
fi

USER_MD="$HERMES_HOME/memory/USER.md"
if [ ! -f "$USER_MD" ]; then
    cat > "$USER_MD" << 'EOF'
# User Profile

> Hand-written user information. Edit this file directly.

## Identity
<!-- Your name, role, timezone, etc. -->

## Preferences
<!-- Communication style, tools used, etc. -->

## Current Projects
<!-- Active projects and context -->
EOF
    echo "  Created: $USER_MD"
fi

# ── 5. Create config ─────────────────────────────────────────────────────────
echo "[5/6] Creating config..."
CONFIG_PATH="$HERMES_HOME/governed_memory.json"
if [ ! -f "$CONFIG_PATH" ]; then
    if [ "$EMBEDDING_BACKEND" = "api" ]; then
        cat > "$CONFIG_PATH" << EOF
{
  "scripts_dir": "$HERMES_HOME/scripts",
  "wiki_dir": "$WIKI_DIR",
  "embedding": {
    "provider": "$EMBEDDING_PROVIDER",
    "base_url": "$EMBEDDING_BASE_URL",
    "api_key_env": "$EMBEDDING_API_KEY_ENV",
    "model": "$EMBEDDING_MODEL"
  }
}
EOF
    elif [ "$EMBEDDING_BACKEND" = "local" ]; then
        cat > "$CONFIG_PATH" << EOF
{
  "scripts_dir": "$HERMES_HOME/scripts",
  "wiki_dir": "$WIKI_DIR",
  "vector": {
    "backend": "auto",
    "model": "BAAI/bge-small-zh-v1.5",
    "dim": 512
  }
}
EOF
    else
        cat > "$CONFIG_PATH" << EOF
{
  "scripts_dir": "$HERMES_HOME/scripts",
  "wiki_dir": "$WIKI_DIR"
}
EOF
    fi
    echo "  Created: $CONFIG_PATH"
fi

# ── 6. Install Python dependencies ───────────────────────────────────────────
echo "[6/6] Installing Python dependencies..."
if command -v pip3 &>/dev/null; then
    PIP=pip3
elif command -v pip &>/dev/null; then
    PIP=pip
else
    echo "  WARNING: pip not found, skipping dependency install"
    PIP=""
fi

if [ -n "$PIP" ]; then
    $PIP install "httpx>=0.28.1"
    if [ "$EMBEDDING_BACKEND" = "local" ]; then
        echo "  Installing local embedding deps (lancedb + fastembed)..."
        $PIP install "lancedb>=0.37,<1" "fastembed>=0.8,<1"
    elif [ "$EMBEDDING_BACKEND" = "api" ]; then
        echo "  API backend: no local model needed (httpx already installed)."
    else
        echo "  Skipped vector dependencies."
        echo "  To install later: pip install lancedb fastembed"
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=== Installation Complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit $MEMORY_MD with your rules"
echo "  2. Edit $USER_MD with your profile"
echo "  3. Set memory.provider: governed in config.yaml"
echo "  4. Run: python scripts/memory_pipeline.py health"
echo ""
echo "Session synthesis LLM: auto-inherits the agent chat model from"
echo "  config.yaml 'model:' section at runtime - no manual synthesis.model"
echo "  config needed (explicit values in governed_memory.json take precedence)."
if [ "$EMBEDDING_BACKEND" = "api" ]; then
    echo ""
    echo "  Remember to export your API key before starting Hermes:"
    echo "    export $EMBEDDING_API_KEY_ENV=sk-..."
fi
echo ""

if [ "$CONFIGURE_CRON" = true ]; then
    echo "Cron setup (add to your crontab):"
    echo ""
    echo "  0  9 * * *  cd $HERMES_HOME && python scripts/l4_persona_daily.py"
    echo "  5  9 * * *  cd $HERMES_HOME && python scripts/scope_recall_bridge.py export"
    echo "  10 9 * * *  cd $HERMES_HOME && python scripts/memory_health_report.py"
    echo ""
fi
