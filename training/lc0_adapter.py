#!/usr/bin/env python3
"""LC0 public chunk -> Tiny Leela normalized-example adapter scaffolding.

Record-level LC0 V6 parsing is intentionally not guessed here. This module owns
stable target conversions, audit/drop accounting, and the normalized JSONL shape
that future chunk decoders should emit.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

BOARD_NORMALIZATION = "stm_white_rankflip_v1"
TEACHER = "lc0_public"


@dataclasses.dataclass(frozen=True)
class Wdl:
    win: float
    draw: float
    loss: float

    def as_dict(self) -> dict[str, float]:
        return {"win": self.win, "draw": self.draw, "loss": self.loss}


def qd_to_wdl(q: float, d: float, *, clamp: bool = False) -> Wdl:
    """Convert LC0 Q/D targets to WDL from side-to-move perspective.

    Formula:
      W = (1 - D + Q) / 2
      D = D
      L = (1 - D - Q) / 2
    """
    if not math.isfinite(q) or not math.isfinite(d):
        raise ValueError(f"non-finite q/d: q={q!r} d={d!r}")
    w = (1.0 - d + q) / 2.0
    l = (1.0 - d - q) / 2.0
    out = Wdl(w, d, l)
    if clamp:
        vals = [max(0.0, min(1.0, x)) for x in (out.win, out.draw, out.loss)]
        s = sum(vals)
        if s <= 0:
            raise ValueError(f"cannot clamp/renormalize q/d: q={q!r} d={d!r}")
        out = Wdl(vals[0] / s, vals[1] / s, vals[2] / s)
    return out


def validate_wdl(wdl: Wdl, *, eps: float = 1e-5) -> list[str]:
    issues: list[str] = []
    vals = [wdl.win, wdl.draw, wdl.loss]
    if any(not math.isfinite(v) for v in vals):
        issues.append("non_finite_wdl")
    if any(v < -eps or v > 1.0 + eps for v in vals):
        issues.append("wdl_out_of_range")
    if abs(sum(vals) - 1.0) > eps:
        issues.append("wdl_sum_not_one")
    return issues


@dataclasses.dataclass
class DropAudit:
    total_records: int = 0
    emitted_records: int = 0
    drop_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    samples: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    max_samples_per_reason: int = 5

    def drop(self, reason: str, sample: dict[str, Any] | None = None) -> None:
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1
        if sample is not None:
            bucket = self.samples.setdefault(reason, [])
            if len(bucket) < self.max_samples_per_reason:
                bucket.append(sample)

    def as_dict(self) -> dict[str, Any]:
        dropped = sum(self.drop_counts.values())
        return {
            "schema": "tiny_leela.lc0_adapter_audit.v1",
            "total_records": self.total_records,
            "emitted_records": self.emitted_records,
            "dropped_records": dropped,
            "drop_rate": (dropped / self.total_records) if self.total_records else 0.0,
            "drop_counts": dict(sorted(self.drop_counts.items())),
            "samples": self.samples,
        }


def normalize_policy(policy: dict[str, float], legal_moves: set[str], audit: DropAudit | None = None) -> dict[str, float] | None:
    """Normalize sparse UCI policy over legal moves.

    This is a temporary adapter boundary. The final LC0 decoder should map from
    LC0 1858 indices to Tiny Leela MoveIds directly, with UCI only as an audit
    view.
    """
    illegal_positive = {m: p for m, p in policy.items() if p > 0.0 and m not in legal_moves}
    if illegal_positive:
        if audit:
            audit.drop("illegal_positive_policy_mass", {"illegal": illegal_positive})
        return None
    kept = {m: float(p) for m, p in policy.items() if m in legal_moves and p > 0.0 and math.isfinite(float(p))}
    mass = sum(kept.values())
    if mass <= 0.0:
        if audit:
            audit.drop("zero_legal_policy_mass", {"legal_moves": sorted(legal_moves)[:16]})
        return None
    return {m: p / mass for m, p in sorted(kept.items())}


def build_normalized_example(
    *,
    source_ref: dict[str, Any],
    board: dict[str, Any],
    legal_moves: Iterable[str],
    policy_uci: dict[str, float],
    root_q: float,
    root_d: float,
    extra: dict[str, Any] | None = None,
    audit: DropAudit | None = None,
) -> dict[str, Any] | None:
    legal = set(legal_moves)
    sparse_policy = normalize_policy(policy_uci, legal, audit=audit)
    if sparse_policy is None:
        return None
    wdl = qd_to_wdl(root_q, root_d)
    issues = validate_wdl(wdl)
    if issues:
        if audit:
            audit.drop("invalid_root_wdl", {"root_q": root_q, "root_d": root_d, "issues": issues})
        return None
    return {
        "schema": "tiny_leela.lc0_normalized_example.v1",
        "teacher": TEACHER,
        "source_ref": source_ref,
        "board_normalization": BOARD_NORMALIZATION,
        "board": board,
        "legal_moves_uci": sorted(legal),
        "policy_target_uci": sparse_policy,
        "value_targets": {
            "root_q": root_q,
            "root_d": root_d,
            "wdl_root": wdl.as_dict(),
        },
        "metadata": extra or {},
    }


def convert_jsonl_smoke(input_path: Path, output_path: Path, audit_path: Path) -> None:
    """Convert already-normalized smoke JSONL records into canonical shape.

    This is useful for tests and for validating downstream trainer plumbing while
    real LC0 binary parsing is under construction. Expected input fields:
    source_ref, board, legal_moves_uci, policy_target_uci, root_q, root_d.
    """
    audit = DropAudit()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as src, output_path.open("w") as out:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                continue
            audit.total_records += 1
            try:
                rec = json.loads(line)
                ex = build_normalized_example(
                    source_ref=rec.get("source_ref", {"line": line_no}),
                    board=rec.get("board", {}),
                    legal_moves=rec["legal_moves_uci"],
                    policy_uci=rec["policy_target_uci"],
                    root_q=float(rec["root_q"]),
                    root_d=float(rec["root_d"]),
                    extra=rec.get("metadata", {}),
                    audit=audit,
                )
            except Exception as exc:
                audit.drop("parse_or_convert_error", {"line": line_no, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if ex is None:
                continue
            out.write(json.dumps(ex, sort_keys=True) + "\n")
            audit.emitted_records += 1
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit.as_dict(), indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    qd = sub.add_parser("qd-to-wdl", help="Convert one Q/D pair to WDL")
    qd.add_argument("--q", type=float, required=True)
    qd.add_argument("--d", type=float, required=True)
    qd.add_argument("--clamp", action="store_true")

    smoke = sub.add_parser("jsonl-smoke", help="Canonicalize smoke JSONL records; not a binary LC0 parser")
    smoke.add_argument("--input", required=True)
    smoke.add_argument("--output", required=True)
    smoke.add_argument("--audit", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "qd-to-wdl":
        out = qd_to_wdl(args.q, args.d, clamp=args.clamp)
        print(json.dumps(out.as_dict(), indent=2, sort_keys=True))
        return 0
    if args.cmd == "jsonl-smoke":
        convert_jsonl_smoke(Path(args.input), Path(args.output), Path(args.audit))
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
