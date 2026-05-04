#!/usr/bin/env python3
"""Query Stockfish through UCI and emit teacher-label JSONL.

This is a tactical/value fallback teacher, not a Leela-style policy teacher. Keep outputs in a
separate dataset unless a future benchmark explicitly combines teacher sources.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

INFO_RE = re.compile(r"multipv\s+(\d+).*?score\s+(cp|mate)\s+(-?\d+).*?\spv\s+(\S+)")


def find_engine(path_or_bin: str) -> str:
    found = shutil.which(path_or_bin) if not Path(path_or_bin).exists() else path_or_bin
    if not found:
        raise SystemExit("Missing stockfish. Install it or set STOCKFISH_BIN=/path/to/stockfish.")
    return found


def send(proc: subprocess.Popen[str], line: str) -> None:
    assert proc.stdin is not None
    proc.stdin.write(line + "\n")
    proc.stdin.flush()


def read_until(proc: subprocess.Popen[str], token: str, limit: int = 20000) -> list[str]:
    assert proc.stdout is not None
    lines: list[str] = []
    for _ in range(limit):
        line = proc.stdout.readline()
        if not line:
            break
        line = line.strip()
        lines.append(line)
        if line.startswith(token):
            return lines
    raise RuntimeError(f"Stockfish did not emit {token}")


def parse(lines: list[str]) -> tuple[dict[str, float], list[float], float]:
    scores: dict[str, float] = {}
    bestmove = None
    for line in lines:
        if line.startswith("bestmove "):
            parts = line.split()
            if len(parts) > 1:
                bestmove = parts[1]
        m = INFO_RE.search(line)
        if not m:
            continue
        _mpv, kind, raw, move = m.groups()
        score = float(raw)
        if kind == "mate":
            score = 100000.0 if score > 0 else -100000.0
        scores.setdefault(move, score / 100.0)
    if not scores and bestmove and bestmove != "(none)":
        scores[bestmove] = 0.0
    if not scores:
        raise RuntimeError("No Stockfish move scores parsed")
    max_score = max(scores.values())
    exps = {move: math.exp((score - max_score) / 0.75) for move, score in scores.items()}
    total = sum(exps.values())
    policy = {move: value / total for move, value in exps.items()}
    q = math.tanh(max_score / 4.0)
    win = max(0.0, q)
    loss = max(0.0, -q)
    draw = max(0.0, 1.0 - win - loss)
    return policy, [win, draw, loss], q


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stockfish", default=os.environ.get("STOCKFISH_BIN", "stockfish"))
    parser.add_argument("--positions", default="data/seed_positions.fen")
    parser.add_argument("--out", default="data/stockfish_teacher_labels.jsonl")
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--multipv", type=int, default=4)
    args = parser.parse_args()

    engine = find_engine(args.stockfish)
    positions = [line.rstrip("\n").split("\t", 1) for line in Path(args.positions).read_text().splitlines() if line.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([engine], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        send(proc, "uci")
        read_until(proc, "uciok")
        send(proc, f"setoption name MultiPV value {args.multipv}")
        send(proc, "isready")
        read_until(proc, "readyok")
        count = 0
        with Path(args.out).open("w") as out:
            for pos_id, fen in positions:
                send(proc, f"position fen {fen}")
                send(proc, f"go depth {args.depth}")
                lines = read_until(proc, "bestmove")
                policy, wdl, q = parse(lines)
                out.write(json.dumps({"id": pos_id, "fen": fen, "policy": policy, "wdl": wdl, "q": q, "teacher": "stockfish", "depth": args.depth}, separators=(",", ":")) + "\n")
                count += 1
        print(f"METRIC stockfish_teacher_labels_written={count}")
        return 0
    finally:
        try:
            send(proc, "quit")
        except Exception:
            pass
        proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
