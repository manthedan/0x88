#!/usr/bin/env python3
"""Validate and merge cloud SquareFormer cache shard metadata from S3.

This is intentionally AWS-CLI based so it works in the current tiny-leela
operator environment without boto3.  It verifies that every expected train/dev
shard has the small metadata files uploaded by worker_squareformer_cache.sh and
then writes a compact cache manifest suitable for handoff/download workflows.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def run(cmd: list[str], *, text: bool = True) -> str:
    return subprocess.check_output(cmd, text=text)


def s3_cp(src: str, dst: str | Path, region: str) -> None:
    subprocess.check_call(["aws", "s3", "cp", src, str(dst), "--region", region], stdout=subprocess.DEVNULL)


def read_json_s3(uri: str, region: str, tmp: Path) -> dict[str, Any]:
    local = tmp / uri.replace("s3://", "").replace("/", "__")
    s3_cp(uri, local, region)
    return json.loads(local.read_text())


def s3_exists(uri: str, region: str) -> bool:
    try:
        subprocess.check_call(["aws", "s3", "ls", uri, "--region", region], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-prefix", required=True, help="s3://.../cache_squareformer_hN")
    ap.add_argument("--dataset-manifest-s3", required=True)
    ap.add_argument("--history-plies", type=int, required=True)
    ap.add_argument("--train-shards", type=int, required=True)
    ap.add_argument("--expect-train-rows", type=int, default=None)
    ap.add_argument("--expect-dev-rows", type=int, default=None)
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--out", required=True, help="local manifest/report path")
    ap.add_argument("--upload", action="store_true", help="also upload manifest to CACHE_PREFIX/cache_manifest.json")
    ap.add_argument("--strict", action="store_true", help="fail on row-count/history mismatches")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cache = args.cache_prefix.rstrip("/")
    errors: list[str] = []
    warnings: list[str] = []
    train_entries: list[dict[str, Any]] = []
    total_train_rows = 0
    first_meta: dict[str, Any] | None = None

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for i in range(args.train_shards):
            shard = f"shard_{i:04d}"
            prefix = f"{cache}/train/{shard}"
            required = ["meta.json", "worker_manifest.json", "cache.tar.zst", "cache.tar.zst.sha256"]
            missing = [name for name in required if not s3_exists(f"{prefix}/{name}", args.region)]
            if missing:
                errors.append(f"train/{shard} missing {','.join(missing)}")
                continue
            meta = read_json_s3(f"{prefix}/meta.json", args.region, tmp)
            worker = read_json_s3(f"{prefix}/worker_manifest.json", args.region, tmp)
            first_meta = first_meta or meta
            rows = int(meta.get("rows", 0))
            total_train_rows += rows
            if int(meta.get("history_plies", -1)) != args.history_plies:
                errors.append(f"train/{shard} history_plies={meta.get('history_plies')} expected {args.history_plies}")
            train_entries.append({
                "shard": shard,
                "prefix": prefix,
                "rows": rows,
                "archive_bytes": int(worker.get("archive_bytes", 0)),
                "archive_sha256": worker.get("archive_sha256"),
                "input_s3": worker.get("input_s3"),
            })

        dev_prefix = f"{cache}/dev/shard_0000"
        dev_required = ["meta.json", "worker_manifest.json", "cache.tar.zst", "cache.tar.zst.sha256"]
        dev_missing = [name for name in dev_required if not s3_exists(f"{dev_prefix}/{name}", args.region)]
        dev_rows = 0
        dev_entry: dict[str, Any] | None = None
        if dev_missing:
            errors.append(f"dev/shard_0000 missing {','.join(dev_missing)}")
        else:
            dev_meta = read_json_s3(f"{dev_prefix}/meta.json", args.region, tmp)
            dev_worker = read_json_s3(f"{dev_prefix}/worker_manifest.json", args.region, tmp)
            dev_rows = int(dev_meta.get("rows", 0))
            if int(dev_meta.get("history_plies", -1)) != args.history_plies:
                errors.append(f"dev/shard_0000 history_plies={dev_meta.get('history_plies')} expected {args.history_plies}")
            dev_entry = {
                "shard": "shard_0000",
                "prefix": dev_prefix,
                "rows": dev_rows,
                "archive_bytes": int(dev_worker.get("archive_bytes", 0)),
                "archive_sha256": dev_worker.get("archive_sha256"),
                "input_s3": dev_worker.get("input_s3"),
            }

    if args.expect_train_rows is not None and total_train_rows != args.expect_train_rows:
        msg = f"train rows={total_train_rows} expected {args.expect_train_rows}"
        (errors if args.strict else warnings).append(msg)
    if args.expect_dev_rows is not None and dev_rows != args.expect_dev_rows:
        msg = f"dev rows={dev_rows} expected {args.expect_dev_rows}"
        (errors if args.strict else warnings).append(msg)

    validation = {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": {"train": total_train_rows, "dev": dev_rows},
        "train_shards_found": len(train_entries),
        "train_shards_expected": args.train_shards,
        "token_features": int((first_meta or {}).get("token_features", 0)),
        "policy_size": int((first_meta or {}).get("policy_size", 0)),
        "history_plies": args.history_plies,
    }
    manifest = {
        "schema": "tiny_leela.squareformer_cache_cloud_manifest.v2",
        "dataset_manifest": args.dataset_manifest_s3,
        "cache_prefix": cache,
        "history_plies": args.history_plies,
        "archive_name": "cache.tar.zst",
        "train": train_entries,
        "dev": dev_entry,
        "shards": [e["prefix"] for e in train_entries],
        "dev_cache": dev_prefix if dev_entry else None,
        "validation": validation,
        "notes": "Cloud cache manifest for S3 cache archives. Download/extract shard archives before local PyTorch training.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.upload:
        s3_cp(str(out), f"{cache}/cache_manifest.json", args.region)
    print(json.dumps(validation, indent=2))
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
