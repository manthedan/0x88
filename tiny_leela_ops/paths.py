from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("TINY_LEELA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def ops_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "artifacts" / "ops"


def registry_path(root: Path | None = None) -> Path:
    override = os.environ.get("TINY_LEELA_RUN_REGISTRY")
    if override:
        return Path(override).expanduser().resolve()
    return ops_dir(root) / "runs.jsonl"


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: str | Path, root: Path | None = None) -> str:
    root = root or repo_root()
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)
