#!/usr/bin/env python3
"""Summarize a downloaded Gumbel self-play cloud run directory.

Expected layout:
  run_dir/shard_000000/{cloud_manifest.json,validate.log,adapted_validate.log,adapter_manifest.json,report.log}

This is intentionally dependency-free so it can run in the repo venv or system Python.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def metric_from_log(path: Path, name: str) -> float | None:
    if not path.exists():
        return None
    prefix = f"METRIC {name}="
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return None
    return None


def rows_from_validate(path: Path) -> int | None:
    if not path.exists():
        return None
    m = re.search(r"rows=(\d+)", path.read_text(encoding="utf-8", errors="replace"))
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="downloaded run directory containing shard_* dirs")
    ap.add_argument("--expect-shards", type=int, default=None)
    ap.add_argument("--expect-model-sha256", default="")
    ap.add_argument("--expect-meta-sha256", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.run_dir)
    shard_dirs = sorted([p for p in root.glob("shard_*") if p.is_dir()])
    if not shard_dirs and (root / "cloud_manifest.json").exists():
        shard_dirs = [root]

    errors: list[str] = []
    seen_indexes: set[int] = set()
    seen_seeds: set[int] = set()
    shards = []
    total_raw = 0
    total_adapted = 0

    if args.expect_shards is not None and len(shard_dirs) != args.expect_shards:
        errors.append(f"expected {args.expect_shards} shards, found {len(shard_dirs)}")

    for shard in shard_dirs:
        manifest_path = shard / "cloud_manifest.json"
        if not manifest_path.exists():
            errors.append(f"{shard}: missing cloud_manifest.json")
            continue
        manifest = read_json(manifest_path)
        idx = int(manifest.get("shard_index", -1))
        seed = int(manifest.get("seed", -1))
        if idx in seen_indexes:
            errors.append(f"duplicate shard_index={idx}")
        seen_indexes.add(idx)
        if seed in seen_seeds:
            errors.append(f"duplicate seed={seed}")
        seen_seeds.add(seed)
        if args.expect_model_sha256 and manifest.get("model_sha256") != args.expect_model_sha256:
            errors.append(f"{shard}: model sha mismatch")
        if args.expect_meta_sha256 and manifest.get("meta_sha256") != args.expect_meta_sha256:
            errors.append(f"{shard}: meta sha mismatch")
        raw_ok = metric_from_log(shard / "validate.log", "selfplay_validate_ok") == 1
        adapted_ok = metric_from_log(shard / "adapted_validate.log", "selfplay_validate_ok") == 1
        if not raw_ok:
            errors.append(f"{shard}: raw validation did not report ok")
        if not adapted_ok:
            errors.append(f"{shard}: adapted validation did not report ok")
        raw_rows = rows_from_validate(shard / "validate.log") or 0
        adapted_rows = rows_from_validate(shard / "adapted_validate.log") or 0
        total_raw += raw_rows
        total_adapted += adapted_rows
        adapter = read_json(shard / "adapter_manifest.json") if (shard / "adapter_manifest.json").exists() else {}
        if adapter.get("skipped_unknown_moves", 0):
            errors.append(f"{shard}: skipped_unknown_moves={adapter.get('skipped_unknown_moves')}")
        shards.append({
            "path": str(shard),
            "shard_index": idx,
            "seed": seed,
            "raw_rows": raw_rows,
            "adapted_rows": adapted_rows,
            "raw_ok": raw_ok,
            "adapted_ok": adapted_ok,
            "lane": manifest.get("lane"),
            "source_model": manifest.get("source_model"),
            "model_sha256": manifest.get("model_sha256", ""),
            "meta_sha256": manifest.get("meta_sha256", ""),
        })

    summary = {
        "schema": "tiny_leela_gumbel_selfplay_run_summary_v1",
        "run_dir": str(root),
        "ok": not errors,
        "errors": errors,
        "shards": len(shards),
        "total_raw_rows": total_raw,
        "total_adapted_rows": total_adapted,
        "shard_indexes": sorted(seen_indexes),
        "seeds": sorted(seen_seeds),
        "items": shards,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
