#!/usr/bin/env python3
"""Export Phase B fixed FENs as seed positions for teacher labeling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", default="eval/benchmark_spec.json")
    parser.add_argument("--out", default="data/seed_positions.fen")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for item in spec["fixed_policy_suite"]:
            f.write(f"{item['id']}\t{item['fen']}\n")
    print(f"METRIC seed_positions_exported={len(spec['fixed_policy_suite'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
