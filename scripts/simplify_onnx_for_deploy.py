#!/usr/bin/env python3
"""Simplify an ONNX model for deployment with onnxsim and optional meta copy.

This is intended for release/deploy candidates, not training artifacts. It keeps
source artifacts untouched and writes a sibling deployment ONNX.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import onnx


def bundle_bytes(path: Path) -> int:
    total = path.stat().st_size if path.exists() else 0
    sidecar = Path(str(path) + ".data")
    if sidecar.exists():
        total += sidecar.stat().st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="input ONNX model")
    ap.add_argument("--out", required=True, help="simplified output ONNX model")
    ap.add_argument("--meta", default="", help="optional source runtime meta JSON")
    ap.add_argument("--meta-out", default="", help="optional output runtime meta JSON with ONNX path updated")
    ap.add_argument("--check-n", type=int, default=0, help="onnxsim random-input check count; use 0 for dynamic/candidate models")
    ap.add_argument("--no-large-tensor", action=argparse.BooleanOptionalAction, default=True, help="pass --no-large-tensor to onnxsim; default true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    model = Path(args.model)
    out = Path(args.out)
    if not model.exists():
        raise SystemExit(f"missing model: {model}")
    if args.skip_existing and out.exists() and out.stat().st_size > 0:
        print(f"[onnxsim-deploy] skip existing {out}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "onnxsim", str(model), str(out)]
    if args.no_large_tensor:
        cmd.append("--no-large-tensor")
    if args.check_n > 0:
        cmd.append(str(args.check_n))
    print("[onnxsim-deploy]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    # Basic structural validation; semantic parity is handled by eval/onnx_parity_check.mjs.
    onnx.checker.check_model(onnx.load(str(out)))
    before = bundle_bytes(model)
    after = bundle_bytes(out)
    print(f"METRIC onnxsim_input_bundle_bytes={before}")
    print(f"METRIC onnxsim_output_bundle_bytes={after}")
    if before:
        print(f"METRIC onnxsim_bundle_ratio={after / before:.8f}")

    if args.meta_out:
        meta_out = Path(args.meta_out)
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        if args.meta:
            meta = json.loads(Path(args.meta).read_text())
        else:
            meta = {}
        meta["onnx"] = str(out)
        meta["deploy_optimized"] = True
        meta["deploy_optimizer"] = "onnxsim"
        try:
            import onnxsim  # type: ignore
            meta["onnxsim_version"] = getattr(onnxsim, "__version__", "unknown")
        except Exception:
            meta["onnxsim_version"] = "unknown"
        meta["onnx_simplified_from"] = str(model)
        meta["onnx_simplified_utc"] = datetime.now(timezone.utc).isoformat()
        meta["onnx_original_bundle_bytes"] = before
        meta["onnx_simplified_bundle_bytes"] = after
        meta_out.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[onnxsim-deploy] wrote meta {meta_out}")
    elif args.meta:
        # Convenience: if no meta-out is requested, leave metadata untouched.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
