#!/usr/bin/env python3
"""Validate/finalize the cloud h8 supervised dataset from S3.

This is intentionally cheap by default: it downloads manifest.json, lists S3 output,
and optionally samples only the first train/dev shard locally.  It does not upload
or mutate S3 state.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import pyzstd  # type: ignore
except Exception:  # pragma: no cover
    pyzstd = None


def run(cmd: list[str], *, check: bool = True, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=text, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def aws(*args: str, region: str) -> subprocess.CompletedProcess:
    cmd = ["aws", *args]
    if region:
        cmd += ["--region", region]
    return run(cmd)


def s3_join(prefix: str, rel: str) -> str:
    return prefix.rstrip("/") + "/" + rel.lstrip("/")


@contextlib.contextmanager
def open_text(path: Path):
    if str(path).endswith(".zst"):
        if pyzstd is not None:
            with pyzstd.open(path, "rt") as f:
                yield f
            return
        p = subprocess.Popen(["zstdcat", str(path)], stdout=subprocess.PIPE, text=True)
        assert p.stdout is not None
        try:
            yield p.stdout
        finally:
            p.stdout.close()
            rc = p.wait()
            # Breaking after --sample-rows can close stdout early; zstdcat then exits
            # with SIGPIPE/141, which is harmless for prefix sampling.
            if rc and rc not in (-13, 141):
                raise subprocess.CalledProcessError(rc, ["zstdcat", str(path)])
        return
    with path.open("rt", encoding="utf-8") as f:
        yield f


def sample_shard(uri: str, tmp: Path, region: str, n: int, expect_history: int) -> dict[str, Any]:
    local = tmp / Path(uri).name
    aws("s3", "cp", uri, str(local), region=region)
    rows = 0
    bad_json = 0
    bad_history_len = 0
    missing_history = 0
    bad_history_type = 0
    first_id = None
    last_id = None
    examples: list[dict[str, Any]] = []
    with open_text(local) as f:
        for line in f:
            if rows >= n:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            rows += 1
            rid = row.get("id")
            first_id = first_id or rid
            last_id = rid
            hf = row.get("history_fens")
            if hf is None:
                missing_history += 1
            elif not isinstance(hf, list):
                bad_history_type += 1
            elif len(hf) > expect_history:
                bad_history_len += 1
            if len(examples) < 3:
                examples.append({
                    "id": rid,
                    "history_len": None if not isinstance(hf, list) else len(hf),
                    "has_fen": bool(row.get("fen")),
                    "policy_type": type(row.get("policy")).__name__,
                })
    return {
        "uri": uri,
        "local_bytes": local.stat().st_size,
        "sampled_rows": rows,
        "bad_json": bad_json,
        "missing_history_fens": missing_history,
        "bad_history_type": bad_history_type,
        "bad_history_len_gt_expected": bad_history_len,
        "first_id": first_id,
        "last_id": last_id,
        "examples": examples,
        "ok": rows > 0 and bad_json == 0 and missing_history == 0 and bad_history_type == 0 and bad_history_len == 0,
    }


def s3_summarize(prefix: str, region: str) -> dict[str, Any]:
    cp = aws("s3", "ls", prefix.rstrip("/") + "/", "--recursive", "--summarize", region=region)
    total_objects = 0
    total_size = 0
    lines = cp.stdout.splitlines()
    for line in lines:
        s = line.strip()
        if s.startswith("Total Objects:"):
            total_objects = int(s.split(":", 1)[1].strip())
        elif s.startswith("Total Size:"):
            total_size = int(s.split(":", 1)[1].strip())
    return {"total_objects": total_objects, "total_size": total_size, "tail": lines[-20:]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-s3-prefix", default="s3://tiny-leela-distributed-ddbb/h8_dataset_10m/datasets/supervised_10m_elite_tcec_h8_v1")
    ap.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2")
    ap.add_argument("--expect-history", type=int, default=8)
    ap.add_argument("--expect-train", type=int, default=10_000_000)
    ap.add_argument("--expect-dev", type=int, default=500_000)
    ap.add_argument("--expect-train-shards", type=int, default=40)
    ap.add_argument("--sample-rows", type=int, default=25)
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--out", default="artifacts/cloud_h8_dataset_10m/h8_dataset_s3_validation.json")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "schema": "tiny_leela.h8_dataset_s3_validation.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_s3_prefix": args.dataset_s3_prefix,
        "region": args.region,
    }
    with tempfile.TemporaryDirectory(prefix="tiny_leela_h8_validate_") as td:
        tmp = Path(td)
        manifest_path = tmp / "manifest.json"
        manifest_uri = s3_join(args.dataset_s3_prefix, "manifest.json")
        try:
            aws("s3", "cp", manifest_uri, str(manifest_path), region=args.region)
        except subprocess.CalledProcessError as e:
            errors.append(f"manifest not available: {manifest_uri}: {e.stderr.strip()}")
            report["s3_summary"] = s3_summarize(args.dataset_s3_prefix, args.region)
            report["errors"] = errors
            report["warnings"] = warnings
            report["ok"] = False
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2))
            return 1

        manifest = json.loads(manifest_path.read_text())
        report["manifest"] = {
            "name": manifest.get("name"),
            "history_plies": manifest.get("history_plies"),
            "total_train_rows": manifest.get("total_train_rows"),
            "total_dev_rows": manifest.get("total_dev_rows"),
            "train_shards": len(manifest.get("train_shards") or []),
            "dev": manifest.get("dev"),
            "zst": any(str(p).endswith(".zst") for p in manifest.get("train_shards") or []),
        }
        if int(manifest.get("history_plies") or 0) < args.expect_history:
            errors.append(f"history_plies={manifest.get('history_plies')} < expected {args.expect_history}")
        if int(manifest.get("total_train_rows") or 0) != args.expect_train:
            errors.append(f"total_train_rows={manifest.get('total_train_rows')} != expected {args.expect_train}")
        if int(manifest.get("total_dev_rows") or 0) != args.expect_dev:
            errors.append(f"total_dev_rows={manifest.get('total_dev_rows')} != expected {args.expect_dev}")
        if len(manifest.get("train_shards") or []) != args.expect_train_shards:
            errors.append(f"train_shards={len(manifest.get('train_shards') or [])} != expected {args.expect_train_shards}")
        if not manifest.get("dev"):
            errors.append("manifest missing dev shard")

        report["s3_summary"] = s3_summarize(args.dataset_s3_prefix, args.region)
        if not args.no_sample and not errors:
            train_rel = (manifest.get("train_shards") or [None])[0]
            dev_rel = manifest.get("dev")
            samples = []
            for rel in [train_rel, dev_rel]:
                if rel:
                    samples.append(sample_shard(s3_join(args.dataset_s3_prefix, rel), tmp, args.region, args.sample_rows, args.expect_history))
            report["samples"] = samples
            for s in samples:
                if not s["ok"]:
                    errors.append(f"sample failed: {s['uri']}")

    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = not errors
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
