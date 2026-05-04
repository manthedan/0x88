#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/teacher_labels.jsonl")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        ps = sum(row["policy"].values())
        ws = sum(row["wdl"])
        if not math.isclose(ps, 1.0, abs_tol=1e-6):
            raise SystemExit(f"{row.get('id')} policy sums to {ps}")
        if not math.isclose(ws, 1.0, abs_tol=1e-6):
            raise SystemExit(f"{row.get('id')} wdl sums to {ws}")
        if not -1.0 <= row["q"] <= 1.0:
            raise SystemExit(f"{row.get('id')} q out of range")
    print(f"METRIC teacher_labels_valid=1")
    print(f"METRIC teacher_label_positions={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
