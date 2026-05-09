#!/usr/bin/env python3
"""Estimate supervised dataset row capacity with the cloud shard-builder rules.

This is a submit-time guardrail: it streams the same raw JSONL/.zst inputs and
uses the same selection caps/dev split as scripts/build_supervised_dataset_shards.py,
but does not write shards. It can stop once a requested safety-margin target is
reached.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_supervised_dataset_shards import (  # noqa: E402
    ReadStats,
    SelectionStats,
    iter_games,
    select_game,
    stable_unit,
    summarize_stream,
)


class NullWriter:
    def __init__(self) -> None:
        self.total = 0

    def write(self, row: dict) -> None:
        self.total += 1


class NullSeen:
    """Set-like object for non-dedupe runs without retaining every FEN."""

    def __contains__(self, key: object) -> bool:
        return False

    def add(self, key: object) -> None:
        return None


def parse_source_caps(specs: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for spec in specs:
        if not spec:
            continue
        if "=" not in spec:
            raise SystemExit(f"--source-cap must be SOURCE=N, got {spec!r}")
        k, v = spec.rsplit("=", 1)
        out[k] = int(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight source row capacity for supervised dataset cloud builds.")
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--max-rows", type=int, required=True, help="target train rows")
    ap.add_argument("--dev-rows", type=int, required=True, help="target dev rows")
    ap.add_argument("--margin", type=float, default=1.15, help="safety margin target; default 1.15")
    ap.add_argument("--max-rows-per-game", type=int, default=64)
    ap.add_argument("--max-rows-per-opening", type=int, default=100000)
    ap.add_argument("--max-rows-per-source", type=int, default=0, help="0 disables the global source cap")
    ap.add_argument("--source-cap", action="append", default=[], metavar="SOURCE=N")
    ap.add_argument("--skip-plies", type=int, default=10)
    ap.add_argument("--history-plies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dedupe-fen", action="store_true")
    ap.add_argument("--progress-games", type=int, default=25000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    started = time.time()
    source_caps = parse_source_caps(args.source_cap)
    max_rows_per_source = args.max_rows_per_source or None
    train_target = math.ceil(args.max_rows * args.margin)
    dev_target = math.ceil(args.dev_rows * args.margin)
    rng = random.Random(args.seed)
    read_stats = ReadStats()
    train_stats = SelectionStats()
    dev_stats = SelectionStats()
    train_writer = NullWriter()
    dev_writer = NullWriter()
    train_seen = set() if args.dedupe_fen else NullSeen()
    dev_seen = set() if args.dedupe_fen else NullSeen()
    dev_rate = min(0.2, max(0.01, args.dev_rows / max(1, args.max_rows + args.dev_rows)))

    for g, src, rows in iter_games(args.input, args.skip_plies, read_stats):
        prefer_dev = stable_unit(args.seed, str(g)) < dev_rate
        if prefer_dev and dev_stats.rows < dev_target:
            select_game(rows, src, rng, dev_target, args.max_rows_per_game, args.max_rows_per_opening, max_rows_per_source, source_caps, args.dedupe_fen, args.history_plies, dev_seen, dev_stats, dev_writer)
        elif train_stats.rows < train_target:
            select_game(rows, src, rng, train_target, args.max_rows_per_game, args.max_rows_per_opening, max_rows_per_source, source_caps, args.dedupe_fen, args.history_plies, train_seen, train_stats, train_writer)
        elif dev_stats.rows < dev_target:
            select_game(rows, src, rng, dev_target, args.max_rows_per_game, args.max_rows_per_opening, max_rows_per_source, source_caps, args.dedupe_fen, args.history_plies, dev_seen, dev_stats, dev_writer)
        if args.progress_games and read_stats.input_games % args.progress_games == 0:
            print(
                f"progress input_games={read_stats.input_games} train_rows={train_stats.rows} dev_rows={dev_stats.rows} source={src}",
                file=sys.stderr,
                flush=True,
            )
        if train_stats.rows >= train_target and dev_stats.rows >= dev_target:
            break

    report = {
        "ok": train_stats.rows >= train_target and dev_stats.rows >= dev_target,
        "margin": args.margin,
        "targets": {
            "train_rows": args.max_rows,
            "dev_rows": args.dev_rows,
            "train_rows_with_margin": train_target,
            "dev_rows_with_margin": dev_target,
        },
        "capacity_observed_until_stop": {
            "train_rows": train_stats.rows,
            "dev_rows": dev_stats.rows,
            "reached_train_margin": train_stats.rows >= train_target,
            "reached_dev_margin": dev_stats.rows >= dev_target,
        },
        "read": {
            "input_games": read_stats.input_games,
            "input_rows_after_basic_filters": read_stats.input_rows,
            "bad_json_lines": read_stats.bad_json_lines,
            "skipped_non_single_policy": read_stats.skipped_non_single_policy,
        },
        "selection": {
            "skip_plies": args.skip_plies,
            "history_plies": args.history_plies,
            "max_rows_per_game": args.max_rows_per_game,
            "max_rows_per_opening": args.max_rows_per_opening,
            "max_rows_per_source": max_rows_per_source,
            "source_caps": source_caps,
            "dev_rate": dev_rate,
            "dedupe_fen": args.dedupe_fen,
        },
        "train": summarize_stream(train_stats),
        "dev": summarize_stream(dev_stats),
        "wall_time_seconds": time.time() - started,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"ok={report['ok']}")
        print(f"train_capacity_observed={train_stats.rows} target_with_margin={train_target}")
        print(f"dev_capacity_observed={dev_stats.rows} target_with_margin={dev_target}")
        print(f"input_games={read_stats.input_games} input_rows_after_basic_filters={read_stats.input_rows}")
        print(f"wall_time_seconds={report['wall_time_seconds']:.1f}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
