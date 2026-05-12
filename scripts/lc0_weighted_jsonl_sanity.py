#!/usr/bin/env python3
"""Sanity-check LC0 weighted hard-label JSONL exported by training/lc0_adapter.py."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess


def _position_key(rec: dict[str, Any], line_no: int) -> str:
    src = rec.get("source_ref") or {}
    if isinstance(src, dict):
        chunk = src.get("chunk") or src.get("input_path") or "unknown"
        record_idx = src.get("record_idx")
        if record_idx is not None:
            return f"{chunk}:{record_idx}"
    row_id = str(rec.get("id") or "")
    if row_id.startswith("lc0_public:"):
        # lc0_public:<chunk>:<record_idx>:<move>; chunk can contain colons rarely,
        # so use source_ref when available and otherwise keep all but the last field.
        return row_id.rsplit(":", 1)[0]
    return f"line:{line_no}"


def check_file(path: Path, *, limit: int = 0, weight_eps: float = 1e-5) -> dict[str, Any]:
    rows = 0
    bad = 0
    issues: dict[str, int] = {}
    samples: dict[str, list[dict[str, Any]]] = {}
    pos_weights: dict[str, float] = defaultdict(float)
    pos_rows: dict[str, int] = defaultdict(int)
    teacher_counts: dict[str, int] = defaultdict(int)
    source_norm_counts: dict[str, int] = defaultdict(int)

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
            teacher_counts[str(rec.get("teacher"))] += 1
            source_norm_counts[str(rec.get("source_board_normalization"))] += 1
            fen = rec.get("fen")
            try:
                board = chess.Board(fen)
            except Exception as exc:
                issue("bad_fen", {"line": line_no, "fen": fen, "error": str(exc)})
                continue
            policy = rec.get("policy")
            if not isinstance(policy, dict) or len(policy) != 1:
                issue("policy_not_one_hot", {"line": line_no, "policy": policy})
                continue
            move, prob = next(iter(policy.items()))
            try:
                p = float(prob)
            except Exception:
                issue("non_numeric_policy", {"line": line_no, "move": move, "prob": prob})
                continue
            if abs(p - 1.0) > 1e-8:
                issue("policy_prob_not_one", {"line": line_no, "move": move, "prob": p})
            if move not in {m.uci() for m in board.legal_moves}:
                issue("illegal_policy_move", {"line": line_no, "fen": fen, "move": move})
            try:
                weight = float(rec.get("weight"))
            except Exception:
                issue("bad_weight", {"line": line_no, "weight": rec.get("weight")})
                weight = float("nan")
            if not math.isfinite(weight) or weight <= 0.0 or weight > 1.0 + weight_eps:
                issue("bad_weight", {"line": line_no, "weight": rec.get("weight")})
            if "_weight" in rec:
                try:
                    if abs(float(rec["_weight"]) - weight) > weight_eps:
                        issue("weight_alias_mismatch", {"line": line_no, "weight": weight, "_weight": rec["_weight"]})
                except Exception:
                    issue("bad_weight_alias", {"line": line_no, "_weight": rec.get("_weight")})
            wdl = rec.get("wdl")
            if not isinstance(wdl, list) or len(wdl) != 3:
                issue("bad_wdl_shape", {"line": line_no, "wdl": wdl})
            else:
                vals = []
                for v in wdl:
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(float("nan"))
                if any(not math.isfinite(v) or v < -1e-5 or v > 1.0 + 1e-5 for v in vals):
                    issue("bad_wdl_range", {"line": line_no, "wdl": wdl})
                if abs(sum(vals) - 1.0) > 1e-5:
                    issue("wdl_sum_not_one", {"line": line_no, "sum": sum(vals), "wdl": wdl})
            key = _position_key(rec, line_no)
            if math.isfinite(weight):
                pos_weights[key] += weight
            pos_rows[key] += 1

    bad_position_weight = 0
    for key, weight in pos_weights.items():
        if abs(weight - 1.0) > weight_eps:
            bad_position_weight += 1
            issue("position_weight_sum_not_one", {"position_key": key, "weight_sum": weight, "rows": pos_rows[key]})

    positions = len(pos_rows)
    weight_sum = sum(pos_weights.values())
    return {
        "schema": "tiny_leela.lc0_weighted_jsonl_sanity.v1",
        "path": str(path),
        "rows_checked": rows,
        "positions_checked": positions,
        "bad_checks": bad,
        "ok": bad == 0 and rows > 0 and positions > 0,
        "weight_sum": weight_sum,
        "mean_rows_per_position": rows / positions if positions else 0.0,
        "mean_weight_per_position": weight_sum / positions if positions else 0.0,
        "bad_position_weight_sums": bad_position_weight,
        "teacher_counts": dict(sorted(teacher_counts.items())),
        "source_board_normalization_counts": dict(sorted(source_norm_counts.items())),
        "issues": dict(sorted(issues.items())),
        "samples": samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--weight-eps", type=float, default=1e-5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    report = check_file(Path(args.input), limit=args.limit, weight_eps=args.weight_eps)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
