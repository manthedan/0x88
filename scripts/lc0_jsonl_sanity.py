#!/usr/bin/env python3
"""Sanity-check LC0 normalized JSONL produced by training/lc0_adapter.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import chess


def check_file(path: Path, *, limit: int = 0) -> dict[str, Any]:
    rows = 0
    bad = 0
    issues: dict[str, int] = {}
    samples: dict[str, list[dict[str, Any]]] = {}

    def issue(name: str, sample: dict[str, Any]) -> None:
        nonlocal bad
        issues[name] = issues.get(name, 0) + 1
        bucket = samples.setdefault(name, [])
        if len(bucket) < 5:
            bucket.append(sample)
        bad += 1

    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            if limit and rows >= limit:
                break
            if not line.strip():
                continue
            rows += 1
            try:
                rec = json.loads(line)
            except Exception as exc:
                issue("json_parse_error", {"line": line_no, "error": str(exc)})
                continue
            fen = rec.get("board", {}).get("fen")
            try:
                board = chess.Board(fen)
            except Exception as exc:
                issue("bad_fen", {"line": line_no, "fen": fen, "error": str(exc)})
                continue
            legal = {m.uci() for m in board.legal_moves}
            listed = set(rec.get("legal_moves_uci", []))
            if listed != legal:
                issue("legal_move_list_mismatch", {"line": line_no, "fen": fen, "missing": sorted(legal - listed)[:8], "extra": sorted(listed - legal)[:8]})
            policy = rec.get("policy_target_uci", {})
            if not policy:
                issue("empty_policy", {"line": line_no, "fen": fen})
            mass = 0.0
            for move, prob in policy.items():
                try:
                    p = float(prob)
                except Exception:
                    issue("non_numeric_policy", {"line": line_no, "move": move, "prob": prob})
                    continue
                mass += p
                if move not in legal:
                    issue("illegal_policy_move", {"line": line_no, "fen": fen, "move": move, "prob": p})
                if not math.isfinite(p) or p <= 0.0 or p > 1.0:
                    issue("bad_policy_probability", {"line": line_no, "move": move, "prob": p})
            if abs(mass - 1.0) > 1e-5:
                issue("policy_mass_not_one", {"line": line_no, "mass": mass})
            for target_name in ("wdl_root", "wdl_result", "wdl_best", "wdl_played"):
                wdl = rec.get("value_targets", {}).get(target_name)
                if not wdl:
                    continue
                vals = [float(wdl.get(k, float("nan"))) for k in ("win", "draw", "loss")]
                if any(not math.isfinite(v) or v < -1e-5 or v > 1.0 + 1e-5 for v in vals):
                    issue("bad_wdl_range", {"line": line_no, "target": target_name, "wdl": wdl})
                if abs(sum(vals) - 1.0) > 1e-5:
                    issue("wdl_sum_not_one", {"line": line_no, "target": target_name, "sum": sum(vals), "wdl": wdl})
    return {
        "schema": "tiny_leela.lc0_jsonl_sanity.v1",
        "path": str(path),
        "rows_checked": rows,
        "bad_checks": bad,
        "ok": bad == 0 and rows > 0,
        "issues": dict(sorted(issues.items())),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = check_file(Path(args.input), limit=args.limit)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
