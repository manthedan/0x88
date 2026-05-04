#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_BIN", "stockfish"))
    args = parser.parse_args()
    stockfish_path = shutil.which(args.stockfish) if not Path(args.stockfish).exists() else args.stockfish
    ready = int(bool(stockfish_path))
    version_ready = 0
    if stockfish_path:
        try:
            proc = subprocess.run([stockfish_path], input="uci\nquit\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=5)
            version_ready = int(proc.returncode == 0 and "uciok" in proc.stdout)
        except Exception:
            version_ready = 0
    print(f"METRIC stockfish_binary_ready={ready}")
    print(f"METRIC stockfish_uci_ready={version_ready}")
    print(f"METRIC stockfish_distillation_ready={int(ready and version_ready)}")
    if not (ready and version_ready):
        print("Missing Stockfish setup. Install stockfish or set STOCKFISH_BIN=/path/to/stockfish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
