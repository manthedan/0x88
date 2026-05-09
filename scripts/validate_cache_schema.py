#!/usr/bin/env python3
"""Validate supervised dataset/cache schema compatibility without reading shard payloads.

This is a cheap preflight for avoiding 38-vs-46 plane, policy-size, state-plane,
and checkpoint/cache mismatches before launching expensive training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(errors="replace"))


def fail(errors: list[str], msg: str) -> None:
    errors.append(msg)


def warn(warnings: list[str], msg: str) -> None:
    warnings.append(msg)


def infer_meta_from_checkpoint(path: Path) -> Path | None:
    candidates = [
        path.with_suffix(".meta.json"),
        Path(str(path).replace(".pt", ".meta.json")),
        path.parent / "model.meta.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def compact_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_planes": meta.get("input_planes"),
        "policy_size": meta.get("policy_size") or len(meta.get("moves") or []),
        "history_plies": meta.get("history_plies"),
        "state_planes": meta.get("state_planes"),
        "policy_map": meta.get("policy_map"),
        "architecture": meta.get("architecture"),
        "channels": meta.get("channels"),
        "blocks": meta.get("blocks"),
    }


def load_cache_metas(cache_dir: Path, cache_manifest: Path | None, warnings: list[str]) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any] | None]:
    manifest = None
    meta_paths: list[Path] = []
    if cache_manifest and cache_manifest.exists():
        manifest = read_json(cache_manifest)
        for shard in manifest.get("shards") or []:
            meta_paths.append(Path(shard) / "meta.json")
        if manifest.get("dev_cache"):
            meta_paths.append(Path(manifest["dev_cache"]) / "meta.json")
    else:
        if cache_manifest:
            warn(warnings, f"cache manifest not found; falling back to partial meta scan: {cache_manifest}")
        meta_paths = sorted(cache_dir.glob("train/*/meta.json"))
        dev_meta = cache_dir / "dev" / "meta.json"
        if dev_meta.exists():
            meta_paths.append(dev_meta)
    metas = []
    existing_paths = []
    for p in meta_paths:
        if p.exists():
            metas.append(read_json(p))
            existing_paths.append(p)
        else:
            warn(warnings, f"listed cache meta missing: {p}")
    return existing_paths, metas, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-manifest", default="data/datasets/supervised_100m_elite_tcec_v1/manifest.json")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--cache-manifest", default=None)
    ap.add_argument("--expect-input-planes", type=int, default=None)
    ap.add_argument("--expect-policy-size", type=int, default=None)
    ap.add_argument("--expect-history-plies", type=int, default=None)
    ap.add_argument("--expect-state-planes", choices=["true", "false"], default=None)
    ap.add_argument("--allow-partial", action="store_true", help="allow in-progress caches with fewer than all train shards")
    ap.add_argument("--model-meta", action="append", default=[], help="model .meta.json files that must match cache input/policy")
    ap.add_argument("--resume", action="append", default=[], help="checkpoint/model paths; neighboring .meta.json is inferred when possible")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []
    ds_path = Path(args.dataset_manifest)
    cache_dir = Path(args.cache_dir)
    cache_manifest = Path(args.cache_manifest) if args.cache_manifest else cache_dir / "cache_manifest.json"

    ds = read_json(ds_path) if ds_path.exists() else None
    if ds is None:
        fail(errors, f"dataset manifest missing: {ds_path}")
        ds = {}

    meta_paths, metas, cache_man = load_cache_metas(cache_dir, cache_manifest, warnings)
    if not metas:
        fail(errors, f"no cache meta files found under {cache_dir}")

    ref = compact_meta(metas[0]) if metas else {}
    for i, m in enumerate(metas):
        cm = compact_meta(m)
        for key in ("input_planes", "policy_size", "history_plies", "state_planes"):
            if cm.get(key) != ref.get(key):
                fail(errors, f"cache meta mismatch {meta_paths[i]}: {key}={cm.get(key)!r}, expected {ref.get(key)!r}")
        if (m.get("moves") or []) != (metas[0].get("moves") or []):
            fail(errors, f"cache policy move list mismatch: {meta_paths[i]}")

    expected = {
        "input_planes": args.expect_input_planes,
        "policy_size": args.expect_policy_size,
        "history_plies": args.expect_history_plies,
        "state_planes": None if args.expect_state_planes is None else (args.expect_state_planes == "true"),
    }
    for key, value in expected.items():
        if value is not None and ref.get(key) != value:
            fail(errors, f"cache {key}={ref.get(key)!r}, expected {value!r}")

    train_meta_count = sum(1 for p in meta_paths if "/train/" in str(p))
    expected_train_shards = len(ds.get("train_shards") or [])
    dev_present = any("/dev/" in str(p) for p in meta_paths)
    if expected_train_shards and train_meta_count != expected_train_shards:
        msg = f"cache train shard metas={train_meta_count}, dataset train shards={expected_train_shards}"
        if args.allow_partial:
            warn(warnings, msg)
        else:
            fail(errors, msg)
    if ds.get("dev") and not dev_present:
        msg = "cache dev meta is missing"
        if args.allow_partial:
            warn(warnings, msg)
        else:
            fail(errors, msg)

    train_rows = sum(int(m.get("rows", 0)) for p, m in zip(meta_paths, metas) if "/train/" in str(p))
    dev_rows = sum(int(m.get("rows", 0)) for p, m in zip(meta_paths, metas) if "/dev/" in str(p))
    if ds.get("total_train_rows") and train_meta_count == expected_train_shards and train_rows != int(ds.get("total_train_rows", 0)):
        fail(errors, f"cache train rows={train_rows}, dataset total_train_rows={ds.get('total_train_rows')}")
    if ds.get("total_dev_rows") and dev_present and dev_rows != int(ds.get("total_dev_rows", 0)):
        fail(errors, f"cache dev rows={dev_rows}, dataset total_dev_rows={ds.get('total_dev_rows')}")

    model_meta_paths = [Path(p) for p in args.model_meta]
    for r in args.resume:
        rp = Path(r)
        inferred = infer_meta_from_checkpoint(rp)
        if inferred is None:
            warn(warnings, f"could not infer meta for resume checkpoint: {rp}")
        else:
            model_meta_paths.append(inferred)
    checked_models = []
    for mp in model_meta_paths:
        if not mp.exists():
            fail(errors, f"model meta missing: {mp}")
            continue
        mm = read_json(mp)
        checked_models.append(str(mp))
        mplanes = mm.get("input_planes")
        mpolicy = len(mm.get("moves") or []) or mm.get("policy_size")
        if mplanes is not None and ref.get("input_planes") is not None and int(mplanes) != int(ref["input_planes"]):
            fail(errors, f"model/cache input-plane mismatch: {mp} input_planes={mplanes}, cache input_planes={ref.get('input_planes')}")
        if mpolicy and ref.get("policy_size") and int(mpolicy) != int(ref["policy_size"]):
            fail(errors, f"model/cache policy-size mismatch: {mp} policy={mpolicy}, cache policy_size={ref.get('policy_size')}")

    result = {
        "dataset_manifest": str(ds_path),
        "cache_dir": str(cache_dir),
        "cache_manifest": str(cache_manifest),
        "cache_manifest_exists": cache_manifest.exists(),
        "cache_ref": ref,
        "cache_train_meta_count": train_meta_count,
        "dataset_train_shards": expected_train_shards,
        "cache_train_rows_seen": train_rows,
        "dataset_total_train_rows": ds.get("total_train_rows"),
        "cache_dev_rows_seen": dev_rows,
        "dataset_total_dev_rows": ds.get("total_dev_rows"),
        "model_metas_checked": checked_models,
        "cache_manifest_summary": {
            "state_planes": cache_man.get("state_planes") if isinstance(cache_man, dict) else None,
            "input_mode": cache_man.get("input_mode") if isinstance(cache_man, dict) else None,
            "workers": cache_man.get("workers") if isinstance(cache_man, dict) else None,
        },
        "warnings": warnings,
        "errors": errors,
        "ok": not errors,
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
