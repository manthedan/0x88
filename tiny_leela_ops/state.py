from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .paths import now_iso


class PipelineState:
    """Simple durable phase marker directory for idempotent pipelines."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dir = self.root / ".tlops_state"
        self.dir.mkdir(parents=True, exist_ok=True)

    def marker(self, phase: str, state: str) -> Path:
        return self.dir / f"{phase}.{state}.json"

    def mark(self, phase: str, state: str = "done", **attrs: Any) -> Path:
        payload = {"phase": phase, "state": state, "ts": now_iso(), **attrs}
        path = self.marker(phase, state)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def is_done(self, phase: str) -> bool:
        return self.marker(phase, "done").exists()

    def failed(self, phase: str, **attrs: Any) -> Path:
        return self.mark(phase, "failed", **attrs)

    def status(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(self.dir.glob("*.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                out.append({"phase_file": str(path), "state": "unreadable"})
        return out

    def require_done(self, phase: str) -> None:
        if not self.is_done(phase):
            raise SystemExit(f"phase is not done: {phase}")
