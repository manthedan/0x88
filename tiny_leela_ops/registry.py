from __future__ import annotations

import fcntl
import json
import os
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .paths import now_iso, registry_path, repo_root

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "done"}


@dataclass
class RunEvent:
    run_id: str
    event: str = "upsert"
    ts: str = field(default_factory=now_iso)
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        payload = {
            "ts": self.ts,
            "host": socket.gethostname(),
            "event": self.event,
            "run_id": self.run_id,
            **self.attrs,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class RunRegistry:
    """Append-only JSONL run registry.

    Each line is an event keyed by run_id. `runs()` folds events into the latest
    run view while preserving the raw audit trail on disk.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else registry_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: RunEvent | dict[str, Any]) -> None:
        if isinstance(event, dict):
            run_id = str(event.pop("run_id"))
            ev = RunEvent(run_id=run_id, event=str(event.pop("event", "upsert")), attrs=event)
        else:
            ev = event
        line = ev.to_json() + "\n"
        with self.path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"bad JSON in {self.path}:{line_no}: {exc}") from exc
        return out

    def runs(self) -> dict[str, dict[str, Any]]:
        folded: dict[str, dict[str, Any]] = {}
        for ev in self.events():
            run_id = str(ev.get("run_id", ""))
            if not run_id:
                continue
            current = folded.setdefault(run_id, {"run_id": run_id, "events": 0})
            current["events"] = int(current.get("events", 0)) + 1
            current["updated_at"] = ev.get("ts")
            event_name = ev.get("event")
            if event_name:
                current["last_event"] = event_name
            for key, value in ev.items():
                if key in {"event", "host"}:
                    continue
                if key == "ts":
                    current.setdefault("created_at", value)
                    current["updated_at"] = value
                    continue
                current[key] = value
        return folded

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self.runs().get(run_id)

    def list(self, kind: str | None = None, status: str | None = None, active: bool = False) -> list[dict[str, Any]]:
        runs = list(self.runs().values())
        if kind:
            runs = [r for r in runs if r.get("kind") == kind]
        if status:
            runs = [r for r in runs if r.get("status") == status]
        if active:
            runs = [r for r in runs if str(r.get("status", "")).lower() not in TERMINAL_STATUSES]
        return sorted(runs, key=lambda r: str(r.get("updated_at", "")), reverse=True)


def default_run_id(kind: str, name: str | None = None) -> str:
    safe_kind = kind.replace("/", "-")
    safe_name = (name or "run").replace("/", "-").replace(" ", "_")
    stamp = now_iso().replace(":", "").replace("+0000", "Z").replace("+00:00", "Z")
    return f"{safe_kind}-{safe_name}-{stamp}"


def git_sha() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(), text=True).strip()
    except Exception:
        return None
