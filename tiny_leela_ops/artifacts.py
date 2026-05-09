from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .paths import now_iso, rel, repo_root
from .registry import RunRegistry

DEFAULT_EXCLUDES = [
    "artifacts/cloud_h8_dataset_100m/raw_zst",
    "artifacts/cloud_h8_dataset_100m/submit_latest.log",
    "data/datasets/supervised_10m_elite_tcec_h8_v1",
    "artifacts/lc0_lite_squareformer/h7_h8_10m",
]


@dataclass
class ArtifactItem:
    path: Path
    bytes: int
    mtime: float
    active: bool = False
    reason: str = ""

    def as_dict(self, root: Path) -> dict:
        return {
            "path": rel(self.path, root),
            "bytes": self.bytes,
            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.mtime)),
            "active": self.active,
            "reason": self.reason,
        }


def path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def active_paths(root: Path) -> dict[Path, str]:
    active: dict[Path, str] = {}
    for ex in DEFAULT_EXCLUDES:
        active[(root / ex).resolve()] = "hard exclusion"

    for pid_file in list((root / "artifacts").glob("**/*.pid")) if (root / "artifacts").exists() else []:
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip().split()[0])
        except Exception:
            continue
        if is_pid_alive(pid):
            active[pid_file.parent.resolve()] = f"live pid {pid} from {rel(pid_file, root)}"

    reg = RunRegistry()
    for run in reg.list(active=True):
        for key in ("local_path", "out_dir", "log", "status_file"):
            val = run.get(key)
            if val:
                active[(root / str(val)).resolve()] = f"active run {run.get('run_id')}"
        local_paths = run.get("local_paths", []) or []
        if isinstance(local_paths, str):
            local_paths = [x.strip() for x in local_paths.split(",") if x.strip()]
        for val in local_paths:
            active[(root / str(val)).resolve()] = f"active run {run.get('run_id')}"
    return active


def under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def inventory(paths: Iterable[str | Path], *, root: Path | None = None, depth: int = 1) -> list[ArtifactItem]:
    root = root or repo_root()
    active = active_paths(root)
    items: list[ArtifactItem] = []
    for base in paths:
        base_path = (root / base).resolve() if not Path(base).is_absolute() else Path(base).resolve()
        if not base_path.exists():
            continue
        candidates = [base_path]
        if base_path.is_dir() and depth > 0:
            candidates = [p for p in base_path.iterdir() if not p.name.startswith(".")]
        for p in sorted(candidates):
            reason = ""
            is_active = False
            for ap, why in active.items():
                if p.resolve() == ap or under(ap, p) or under(p, ap):
                    is_active = True
                    reason = why
                    break
            try:
                st = p.stat()
            except OSError:
                continue
            items.append(ArtifactItem(path=p, bytes=path_size(p), mtime=st.st_mtime, active=is_active, reason=reason))
    return sorted(items, key=lambda x: x.bytes, reverse=True)


def cold_store(
    candidates: list[ArtifactItem],
    destination: str | Path,
    *,
    root: Path | None = None,
    dry_run: bool = True,
    older_than_days: float = 0,
    min_bytes: int = 0,
) -> dict:
    root = root or repo_root()
    dest = Path(destination).expanduser().resolve()
    cutoff = time.time() - older_than_days * 86400
    moves = []
    skipped = []
    for item in candidates:
        if item.active:
            skipped.append({**item.as_dict(root), "skip": item.reason or "active"})
            continue
        if item.bytes < min_bytes:
            skipped.append({**item.as_dict(root), "skip": "below min_bytes"})
            continue
        if item.mtime > cutoff:
            skipped.append({**item.as_dict(root), "skip": "newer than threshold"})
            continue
        rel_path = Path(rel(item.path, root))
        target = dest / rel_path
        moves.append({"src": str(item.path), "dest": str(target), "bytes": item.bytes})
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.path), str(target))
    manifest = {"ts": now_iso(), "dry_run": dry_run, "destination": str(dest), "moves": moves, "skipped": skipped}
    if not dry_run:
        dest.mkdir(parents=True, exist_ok=True)
        mpath = dest / f"cold_store_manifest_{now_iso().replace(':', '').replace('+00:00', 'Z')}.json"
        mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["manifest_path"] = str(mpath)
    return manifest
