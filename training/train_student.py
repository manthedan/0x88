#!/usr/bin/env python3
"""Train a tiny pure-Python distillation student on teacher-label JSONL.

This is intentionally dependency-free. It is a real optimization loop over generated teacher labels,
but it is not yet a chess-strength claim: with the bootstrap seed set it only proves the pipeline and
provides policy/WDL losses for comparing future student designs.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

PIECES = "PNBRQKpnbrqk"


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [x / total for x in exps]


def fen_features(fen: str) -> list[float]:
    board, side, *_ = fen.split()
    counts = {p: 0 for p in PIECES}
    for ch in board:
        if ch in counts:
            counts[ch] += 1
    vals = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    white_mat = sum(counts[p] * vals[p] for p in "PNBRQK")
    black_mat = sum(counts[p] * vals[p.upper()] for p in "pnbrqk")
    feats = [1.0, 1.0 if side == "w" else -1.0]
    feats += [(counts[p] - 2.0) / 8.0 for p in PIECES]
    feats += [(white_mat - black_mat) / 39.0]
    return feats


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_fen_rows(rows: list[dict]) -> list[dict]:
    """Average multiple teacher labels for the same position into one consensus target."""
    merged: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        fen = row["fen"]
        if fen not in merged:
            order.append(fen)
            merged[fen] = {"fen": fen, "policy": {}, "wdl": [0.0, 0.0, 0.0], "count": 0}
        acc = merged[fen]
        acc["count"] += 1
        for move, prob in row["policy"].items():
            acc["policy"][move] = acc["policy"].get(move, 0.0) + prob
        for i, prob in enumerate(row["wdl"]):
            acc["wdl"][i] += prob
    out: list[dict] = []
    for fen in order:
        acc = merged[fen]
        count = acc["count"]
        policy = {move: prob / count for move, prob in acc["policy"].items()}
        mass = sum(policy.values())
        if mass > 0:
            policy = {move: prob / mass for move, prob in policy.items()}
        out.append({"fen": fen, "policy": policy, "wdl": [prob / count for prob in acc["wdl"]]})
    return out


def cross_entropy(target: list[float], pred: list[float]) -> float:
    return -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, pred))


def evaluate(rows: list[dict], moves: list[str], wp: list[list[float]], ww: list[list[float]]) -> tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0
    p_ce = 0.0
    w_ce = 0.0
    top1 = 0.0
    for row in rows:
        x = fen_features(row["fen"])
        logits = [sum(w * v for w, v in zip(weights, x)) for weights in wp]
        probs = softmax(logits)
        target = [row["policy"].get(move, 0.0) for move in moves]
        mass = sum(target)
        if mass <= 0:
            continue
        target = [v / mass for v in target]
        p_ce += cross_entropy(target, probs)
        wdl_logits = [sum(w * v for w, v in zip(weights, x)) for weights in ww]
        wdl = softmax(wdl_logits)
        w_ce += cross_entropy(row["wdl"], wdl)
        if moves[max(range(len(moves)), key=lambda i: probs[i])] in row["policy"]:
            top1 += 1.0
    n = len(rows)
    return p_ce / n, w_ce / n, top1 / n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=["data/teacher_labels.jsonl", "data/stockfish_teacher_labels.jsonl"])
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--holdout-mod", type=int, default=2, help="hold out rows whose index modulo this value is 0")
    parser.add_argument("--out", default="artifacts/student_linear.json")
    parser.add_argument("--merge-fen", action="store_true", help="average labels that share the same FEN before splitting")
    args = parser.parse_args()

    raw_rows = load_rows(args.train)
    rows = merge_fen_rows(raw_rows) if args.merge_fen else raw_rows
    if len(rows) < 2:
        raise SystemExit("Need at least two teacher rows")
    moves = sorted({move for row in rows for move in row["policy"]})
    train_rows = [row for i, row in enumerate(rows) if i % args.holdout_mod != 0]
    dev_rows = [row for i, row in enumerate(rows) if i % args.holdout_mod == 0]
    feat_dim = len(fen_features(rows[0]["fen"]))
    rng = random.Random(7)
    wp = [[rng.uniform(-0.01, 0.01) for _ in range(feat_dim)] for _ in moves]
    ww = [[rng.uniform(-0.01, 0.01) for _ in range(feat_dim)] for _ in range(3)]

    for _epoch in range(args.epochs):
        rng.shuffle(train_rows)
        for row in train_rows:
            x = fen_features(row["fen"])
            policy_logits = [sum(w * v for w, v in zip(weights, x)) for weights in wp]
            probs = softmax(policy_logits)
            target = [row["policy"].get(move, 0.0) for move in moves]
            mass = sum(target)
            target = [v / mass for v in target]
            for i in range(len(moves)):
                grad = probs[i] - target[i]
                for j, val in enumerate(x):
                    wp[i][j] -= args.lr * grad * val
            wdl_logits = [sum(w * v for w, v in zip(weights, x)) for weights in ww]
            wdl = softmax(wdl_logits)
            for i in range(3):
                grad = wdl[i] - row["wdl"][i]
                for j, val in enumerate(x):
                    ww[i][j] -= args.lr * grad * val

    train_policy_ce, train_wdl_ce, train_top1 = evaluate(train_rows, moves, wp, ww)
    dev_policy_ce, dev_wdl_ce, dev_top1 = evaluate(dev_rows, moves, wp, ww)
    quality = 100.0 / (1.0 + dev_policy_ce + dev_wdl_ce)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kind": "linear_fen_student", "moves": moves, "policy_weights": wp, "wdl_weights": ww}, separators=(",", ":")))
    print(f"METRIC distill_student_score={quality:.6f}")
    print(f"METRIC train_policy_ce={train_policy_ce:.6f}")
    print(f"METRIC train_wdl_ce={train_wdl_ce:.6f}")
    print(f"METRIC dev_policy_ce={dev_policy_ce:.6f}")
    print(f"METRIC dev_wdl_ce={dev_wdl_ce:.6f}")
    print(f"METRIC dev_policy_top1={dev_top1:.6f}")
    print(f"METRIC teacher_rows={len(rows)}")
    print(f"METRIC raw_teacher_rows={len(raw_rows)}")
    print(f"METRIC move_vocab={len(moves)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
