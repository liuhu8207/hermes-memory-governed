"""Shared environment helpers for Hermes memory governance scripts."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def setup_logging(name: str | None = None, level: int | None = None) -> logging.Logger:
    if level is None:
        env_level = os.getenv("HERMES_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    fmt = os.getenv("HERMES_LOG_FORMAT", "%(asctime)s %(name)s %(levelname)s %(message)s")
    datefmt = os.getenv("HERMES_LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger("hermes_governed")
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)

    logger = logging.getLogger(f"hermes_governed.{name}") if name else root
    return logger


def load_dotenv(path: Path, overwrite: bool = True) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if overwrite or key not in os.environ:
            os.environ[key] = value


def bootstrap(script_file: str) -> Path:
    """Detect HERMES_HOME and load .env."""
    script_dir = Path(script_file).resolve().parent
    repo_or_home = script_dir.parent if script_dir.name.lower() == "scripts" else script_dir

    # Try repo-local .env first
    load_dotenv(repo_or_home / ".env")

    # HERMES_HOME from env or default
    hermes_home = Path(os.environ.get(
        "HERMES_HOME",
        os.environ.get("LOCALAPPDATA", str(Path.home())) + "/hermes"
    ))

    # Try HERMES_HOME/.env
    load_dotenv(hermes_home / ".env")

    # Ensure dirs
    for subdir in ["memory", "memory/l2", "memory/l3", "cron/output"]:
        (hermes_home / subdir).mkdir(parents=True, exist_ok=True)

    return hermes_home


def env_path(key: str, default: Path) -> Path:
    """Read an env var as a Path, falling back to default."""
    val = os.environ.get(key)
    return Path(val) if val else default


def env_int(key: str, default: int = 0) -> int:
    """Read an env var as an int, falling back to default on error."""
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def load_config(hermes_home: Path) -> dict:
    """Load config.yaml (minimal loader, no PyYAML dependency)."""
    config_path = hermes_home / "config.yaml"
    if not config_path.exists():
        return {}

    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: basic key: value parsing
        config = {}
        for line in config_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                config[key.strip()] = value.strip().strip('"').strip("'")
        return config
