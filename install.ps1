<#
    Hermes Memory Governed — One-command installer for Windows

    Usage:
        powershell -ExecutionPolicy Bypass -File install.ps1
        powershell -ExecutionPolicy Bypass -File install.ps1 -WithVector
        powershell -ExecutionPolicy Bypass -File install.ps1 -WithoutVector
        powershell -ExecutionPolicy Bypass -File install.ps1 -EmbeddingApi
        powershell -ExecutionPolicy Bypass -File install.ps1 -Yes
        powershell -ExecutionPolicy Bypass -File install.ps1 -ConfigureCron
#>

param(
    [string]$HermesHome = "$env:LOCALAPPDATA\hermes",
    [string]$WikiDir = "$HOME\wiki",
    [switch]$WithVector,
    [switch]$WithoutVector,
    [switch]$EmbeddingApi,
    [switch]$ConfigureCron,
    [switch]$AutoDetectLlm,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# ── Interactive prompts ───────────────────────────────────────────────────────
$EmbeddingBackend = $null   # "api" | "local" | "none"
$EmbeddingProvider = ""
$EmbeddingBaseUrl = ""
$EmbeddingApiKeyEnv = ""
$EmbeddingModel = ""

if (-not $Yes -and -not $WithVector -and -not $WithoutVector -and -not $EmbeddingApi) {
    Write-Host "=== Hermes Memory Governed Installer ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "HERMES_HOME = $HermesHome" -ForegroundColor Gray
    Write-Host "WIKI_DIR    = $WikiDir" -ForegroundColor Gray
    Write-Host ""

    Write-Host "Embedding backend for L2 semantic search:" -ForegroundColor Yellow
    Write-Host "  1) API         - OpenAI-compatible /embeddings, no local model (recommended)"
    Write-Host "  2) Local model - bge-small-zh-v1.5 (512-dim, ~100MB download, offline)"
    Write-Host "  3) Skip        - L3 FTS5 keyword search only"
    Write-Host ""

    $embAnswer = Read-Host "  Choose [2]"
    if ($embAnswer -match "^(1|api)$") {
        $EmbeddingBackend = "api"
    } elseif ($embAnswer -match "^(3|none|skip)$") {
        $EmbeddingBackend = "none"
    } else {
        $EmbeddingBackend = "local"
    }
    Write-Host ""

    if ($EmbeddingBackend -eq "api") {
        $EmbeddingProvider  = Read-Host "  provider (e.g. siliconflow/openai)"
        $EmbeddingBaseUrl   = Read-Host "  base_url (e.g. https://api.siliconflow.cn/v1)"
        $EmbeddingModel     = Read-Host "  model (e.g. BAAI/bge-m3)"
        $EmbeddingApiKeyEnv = Read-Host "  API key env var name (key stays in env, e.g. SILICONFLOW_API_KEY)"
        Write-Host ""
    }

    $cronAnswer = Read-Host "  Show cron job setup instructions? [y/N]"
    if ($cronAnswer -match "^[yY]") {
        $ConfigureCron = $true
    }
    Write-Host ""
}

if ($WithVector)    { $EmbeddingBackend = "local" }
if ($WithoutVector) { $EmbeddingBackend = "none" }
if ($EmbeddingApi)  { $EmbeddingBackend = "api" }
if (-not $EmbeddingBackend) { $EmbeddingBackend = "none" }

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host "=== Hermes Memory Governed Installer ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "HERMES_HOME       = $HermesHome" -ForegroundColor Gray
Write-Host "WIKI_DIR          = $WikiDir" -ForegroundColor Gray
Write-Host "EMBEDDING_BACKEND = $EmbeddingBackend" -ForegroundColor Gray
Write-Host ""

# ── 1. Create directories ────────────────────────────────────────────────────
Write-Host "[1/6] Creating directories..." -ForegroundColor Yellow
$dirs = @(
    "$HermesHome\memory",
    "$HermesHome\memory\l2",
    "$HermesHome\memory\l3",
    "$HermesHome\scripts",
    "$HermesHome\plugins\governed",
    "$HermesHome\cron\output\scope_recall_bridge",
    "$WikiDir"
)
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    }
}

# ── 2. Copy plugin ───────────────────────────────────────────────────────────
Write-Host "[2/6] Installing plugin..." -ForegroundColor Yellow
$pluginSrc = "$ScriptDir\plugin\memory_governed"
$pluginDst = "$HermesHome\plugins\governed"
if (-not (Test-Path $pluginDst)) {
    New-Item -ItemType Directory -Path $pluginDst -Force | Out-Null
}
Copy-Item "$pluginSrc\*.py" -Destination $pluginDst -Force
Write-Host "  Plugin installed to: $pluginDst" -ForegroundColor Gray

# ── 3. Copy scripts ──────────────────────────────────────────────────────────
Write-Host "[3/6] Installing scripts..." -ForegroundColor Yellow
Copy-Item "$ScriptDir\scripts\*" -Destination "$HermesHome\scripts" -Recurse -Force
Write-Host "  Scripts installed to: $HermesHome\scripts" -ForegroundColor Gray

# ── 4. Create default L1 files ───────────────────────────────────────────────
Write-Host "[4/6] Creating default L1 files..." -ForegroundColor Yellow
$memoryMd = "$HermesHome\memory\MEMORY.md"
$userMd = "$HermesHome\memory\USER.md"

if (-not (Test-Path $memoryMd)) {
    @"
# Memory Rules

> Hand-written rules for the AI agent. Edit this file directly.

## Project Rules
<!-- Add your project-specific rules here -->

## Behavioral Rules
<!-- Add things the AI should/shouldn't do here -->
"@ | Out-File -FilePath $memoryMd -Encoding utf8
    Write-Host "  Created: $memoryMd" -ForegroundColor Gray
}

if (-not (Test-Path $userMd)) {
    @"
# User Profile

> Hand-written user information. Edit this file directly.

## Identity
<!-- Your name, role, timezone, etc. -->

## Preferences
<!-- Communication style, tools used, etc. -->

## Current Projects
<!-- Active projects and context -->
"@ | Out-File -FilePath $userMd -Encoding utf8
    Write-Host "  Created: $userMd" -ForegroundColor Gray
}

# ── 5. Create config ─────────────────────────────────────────────────────────
Write-Host "[5/6] Creating config..." -ForegroundColor Yellow
$configPath = "$HermesHome\governed_memory.json"
if (-not (Test-Path $configPath)) {
    if ($EmbeddingBackend -eq "api") {
        @{
            scripts_dir = "$HermesHome\scripts"
            wiki_dir = $WikiDir
            embedding = @{
                provider = $EmbeddingProvider
                base_url = $EmbeddingBaseUrl
                api_key_env = $EmbeddingApiKeyEnv
                model = $EmbeddingModel
            }
        } | ConvertTo-Json -Depth 6 | Out-File -FilePath $configPath -Encoding utf8
    } elseif ($EmbeddingBackend -eq "local") {
        @{
            scripts_dir = "$HermesHome\scripts"
            wiki_dir = $WikiDir
            vector = @{
                backend = "auto"
                model = "BAAI/bge-small-zh-v1.5"
                dim = 512
            }
        } | ConvertTo-Json -Depth 6 | Out-File -FilePath $configPath -Encoding utf8
    } else {
        @{
            scripts_dir = "$HermesHome\scripts"
            wiki_dir = $WikiDir
        } | ConvertTo-Json | Out-File -FilePath $configPath -Encoding utf8
    }
    Write-Host "  Created: $configPath" -ForegroundColor Gray
}

# ── 6. Install Python dependencies ───────────────────────────────────────────
Write-Host "[6/6] Installing Python dependencies..." -ForegroundColor Yellow
$pip = $null
if (Get-Command pip3 -ErrorAction SilentlyContinue) {
    $pip = "pip3"
} elseif (Get-Command pip -ErrorAction SilentlyContinue) {
    $pip = "pip"
} else {
    Write-Host "  WARNING: pip not found, skipping dependency install" -ForegroundColor Red
}

if ($pip) {
    & $pip install "httpx>=0.28.1"
    if ($EmbeddingBackend -eq "local") {
        Write-Host "  Installing local embedding deps (lancedb + fastembed)..." -ForegroundColor Gray
        & $pip install "lancedb>=0.37,<1" "fastembed>=0.8,<1"
    } elseif ($EmbeddingBackend -eq "api") {
        Write-Host "  API backend: no local model needed (httpx already installed)." -ForegroundColor Gray
    } else {
        Write-Host "  Skipped vector dependencies." -ForegroundColor Gray
        Write-Host "  To install later: pip install lancedb fastembed" -ForegroundColor Gray
    }
}

# ── Configure cron (optional) ────────────────────────────────────────────────
if ($ConfigureCron) {
    Write-Host ""
    Write-Host "Cron configuration requires manual setup." -ForegroundColor Yellow
    Write-Host "Add these jobs to Hermes cron:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  hermes cron add --name l4-persona --schedule '0 9 * * *' --script scripts/l4_persona_daily.py --no-agent"
    Write-Host "  hermes cron add --name bridge-export --schedule '0 9 * * *' --script scripts/scope_recall_bridge.py --no-agent"
    Write-Host "  hermes cron add --name health-report --schedule '0 10 * * *' --script scripts/memory_health_report.py --no-agent"
}

# ── Auto-detect LLM (optional) ──────────────────────────────────────────────
if ($AutoDetectLlm) {
    Write-Host ""
    Write-Host "Session synthesis LLM: auto-inherits the agent chat model" -ForegroundColor Yellow
    $hermesConfig = "$HermesHome\config.yaml"
    if (Test-Path $hermesConfig) {
        Write-Host "  Found config: $hermesConfig" -ForegroundColor Gray
        Write-Host "  The governed plugin reads the 'model:' section at runtime and uses" -ForegroundColor Gray
        Write-Host "  the same provider/base_url/model for session synthesis. No manual" -ForegroundColor Gray
        Write-Host "  synthesis.model config needed; explicit values in governed_memory.json" -ForegroundColor Gray
        Write-Host "  take precedence if you ever set them." -ForegroundColor Gray
    } else {
        Write-Host "  No config.yaml found; synthesis stays disabled until the agent" -ForegroundColor Yellow
        Write-Host "  model is configured." -ForegroundColor Yellow
    }
}

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== Installation Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Edit $memoryMd with your rules"
Write-Host "  2. Edit $userMd with your profile"
Write-Host "  3. Set memory.provider: governed in config.yaml"
Write-Host "  4. Run: python scripts/memory_pipeline.py health"
if ($EmbeddingBackend -eq "api") {
    Write-Host ""
    Write-Host "  Remember to set your API key environment variable before starting Hermes:" -ForegroundColor Cyan
    Write-Host "    `$env:$EmbeddingApiKeyEnv = 'sk-...'"
}
Write-Host ""
