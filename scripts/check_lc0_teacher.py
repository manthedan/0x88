#!/usr/bin/env python3
"""Check whether an lc0 teacher is configured for distillation."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", default=os.environ.get("LC0_BIN", "lc0"))
    parser.add_argument("--weights", default=os.environ.get("LC0_WEIGHTS", ""))
    args = parser.parse_args()

    lc0_path = shutil.which(args.lc0) if not Path(args.lc0).exists() else args.lc0
    lc0_ready = int(bool(lc0_path))
    weights_ready = int(bool(args.weights and Path(args.weights).exists()))

    version_ok = 0
    if lc0_path:
        try:
            proc = subprocess.run([lc0_path, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
            version_ok = int(proc.returncode == 0 and bool(proc.stdout.strip()))
        except Exception:
            version_ok = 0

    ready = int(lc0_ready and weights_ready and version_ok)
    print(f"METRIC lc0_binary_ready={lc0_ready}")
    print(f"METRIC lc0_weights_ready={weights_ready}")
    print(f"METRIC lc0_version_ready={version_ok}")
    print(f"METRIC distillation_ready={ready}")
    if not ready:
        print("Missing real lc0 teacher setup. Set LC0_BIN=/path/to/lc0 and LC0_WEIGHTS=/path/to/weights.pb.gz.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
