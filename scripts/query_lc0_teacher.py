#!/usr/bin/env python3
"""Query lc0 via UCI and emit teacher labels JSONL.

This intentionally uses only the UCI surface so the first real teacher pipeline is easy to audit.
Policy labels are approximated from MultiPV scores when available; richer lc0 search-visit extraction
can replace this under a new dataset version.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

INFO_RE = re.compile(r"multipv\s+(\d+).*?(?:score\s+(cp|mate)\s+(-?\d+)).*?\spv\s+(\S+)")
VERBOSE_RE = re.compile(r"info string\s+(\S+)\s+\([^)]*\)\s+N:\s*(\d+).*?\(P:\s*([0-9.]+)%\).*?\(WL:\s*([-0-9.]+)\).*?\(D:\s*([0-9.]+)\)")


def require(path_or_bin: str, label: str) -> str:
    found = shutil.which(path_or_bin) if not Path(path_or_bin).exists() else path_or_bin
    if not found:
        raise SystemExit(f"Missing {label}: {path_or_bin}. Set LC0_BIN/LC0_WEIGHTS or pass flags.")
    return found


def send(proc: subprocess.Popen[str], line: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def read_until(proc: subprocess.Popen[str], token: str, timeout_lines: int = 10000) -> list[str]:
    assert proc.stdout is not None
    lines: list[str] = []
    for _ in range(timeout_lines):
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        lines.append(line)
        if line.startswith(token):
            return lines
    raise RuntimeError(f"lc0 did not emit {token}")


def parse_search(lines: list[str]) -> tuple[dict[str, float], list[float], float]:
    # Prefer Lc0 verbose root stats: they expose visit counts, policy prior, WL and draw estimates.
    verbose: dict[str, tuple[int, float, float, float]] = {}
    for line in lines:
        m = VERBOSE_RE.search(line)
        if not m:
            continue
        move, visits, prior_pct, wl, draw = m.groups()
        if move == "node":
            continue
        verbose[move] = (int(visits), float(prior_pct) / 100.0, float(wl), float(draw))
    visited = {move: vals for move, vals in verbose.items() if vals[0] > 0}
    if visited:
        total_visits = sum(vals[0] for vals in visited.values())
        policy = {move: vals[0] / total_visits for move, vals in visited.items()}
        best_move, best_vals = max(visited.items(), key=lambda item: item[1][0])
        _visits, _prior, q, draw = best_vals
        win = max(0.0, (1.0 - draw + q) / 2.0)
        loss = max(0.0, (1.0 - draw - q) / 2.0)
        total = win + draw + loss
        return policy, [win / total, draw / total, loss / total], q

    # Fallback to the final MultiPV lines. Overwrite by multipv so early shallow lines do not freeze labels.
    by_mpv: dict[int, tuple[str, float]] = {}
    for line in lines:
        m = INFO_RE.search(line)
        if not m:
            continue
        multipv, kind, raw, move = m.groups()
        score = float(raw)
        if kind == "mate":
            score = 100000.0 if score > 0 else -100000.0
        by_mpv[int(multipv)] = (move, score / 100.0)
    scores = {move: score for move, score in by_mpv.values()}
    if not scores:
        best = next((line.split()[1] for line in lines if line.startswith("bestmove ") and len(line.split()) > 1), None)
        if best and best != "(none)":
            scores[best] = 0.0
    if not scores:
        raise RuntimeError("No teacher moves found in lc0 output")
    max_score = max(scores.values())
    exps = {m: math.exp((s - max_score) / 0.75) for m, s in scores.items()}
    total = sum(exps.values())
    policy = {m: v / total for m, v in exps.items()}
    q = math.tanh(max_score / 4.0)
    win = max(0.0, q)
    loss = max(0.0, -q)
    draw = max(0.0, 1.0 - win - loss)
    return policy, [win, draw, loss], q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc0", default=os.environ.get("LC0_BIN", "lc0"))
    parser.add_argument("--weights", default=os.environ.get("LC0_WEIGHTS", ""))
    parser.add_argument("--positions", default="data/seed_positions.fen")
    parser.add_argument("--out", default="data/teacher_labels.jsonl")
    parser.add_argument("--nodes", type=int, default=64)
    parser.add_argument("--multipv", type=int, default=4)
    args = parser.parse_args()

    lc0 = require(args.lc0, "lc0 binary")
    weights = require(args.weights, "lc0 weights")
    positions = [line.rstrip("\n").split("\t", 1) for line in Path(args.positions).read_text().splitlines() if line.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen([lc0], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        send(proc, "uci")
        read_until(proc, "uciok")
        send(proc, f"setoption name WeightsFile value {weights}")
        send(proc, f"setoption name MultiPV value {args.multipv}")
        send(proc, "setoption name VerboseMoveStats value true")
        send(proc, "isready")
        read_until(proc, "readyok")
        count = 0
        with Path(args.out).open("w") as out:
            for pos_id, fen in positions:
                send(proc, f"position fen {fen}")
                send(proc, f"go nodes {args.nodes}")
                lines = read_until(proc, "bestmove")
                policy, wdl, q = parse_search(lines)
                out.write(json.dumps({"id": pos_id, "fen": fen, "policy": policy, "wdl": wdl, "q": q, "teacher": "lc0", "nodes": args.nodes}, separators=(",", ":")) + "\n")
                count += 1
        print(f"METRIC teacher_labels_written={count}")
        return 0
    finally:
        try:
            send(proc, "quit")
        except Exception:
            pass
        proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
