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
PIECE_INDEX = {piece: i for i, piece in enumerate(PIECES)}


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


def wdl_features(fen: str) -> list[float]:
    """Value head features: base aggregate features plus compact tempo interactions."""
    base = fen_features(fen)
    side_sign = base[1]
    return base + [side_sign * v for v in base[2:]]


def stable_weight(*values: int) -> float:
    seed = 0x9E3779B97F4A7C15
    for value in values:
        seed ^= (value + 0x9E3779B9 + (seed << 6) + (seed >> 2)) & 0xFFFFFFFFFFFFFFFF
        seed &= 0xFFFFFFFFFFFFFFFF
    return (((seed % 2001) / 1000.0) - 1.0) / math.sqrt(len(values) + 1.0)


def conv_student_features(fen: str, channels: int, layers: int) -> list[float]:
    """Frozen tiny conv-tower features for comparing browser-sized student shapes."""
    board, side, *_ = fen.split()
    maps = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(13)]
    rank = 0
    file = 0
    for ch in board:
        if ch == "/":
            rank += 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        elif ch in PIECE_INDEX:
            maps[PIECE_INDEX[ch]][rank][file] = 1.0
            file += 1
    side_value = 1.0 if side == "w" else -1.0
    maps[12] = [[side_value for _ in range(8)] for _ in range(8)]
    prev = maps
    for layer in range(layers):
        out = [[[0.0 for _ in range(8)] for _ in range(8)] for _ in range(channels)]
        prev_channels = len(prev)
        for c in range(channels):
            for r in range(8):
                for f in range(8):
                    acc = stable_weight(layer, c, 99)
                    for pc in range(prev_channels):
                        for dr in (-1, 0, 1):
                            rr = r + dr
                            if rr < 0 or rr >= 8:
                                continue
                            for df in (-1, 0, 1):
                                ff = f + df
                                if 0 <= ff < 8:
                                    acc += prev[pc][rr][ff] * stable_weight(layer, c, pc, dr + 1, df + 1)
                    out[c][r][f] = math.tanh(acc / math.sqrt(prev_channels * 4.0))
        prev = out
    feats = [1.0, side_value]
    for channel in prev:
        flat = [v for row in channel for v in row]
        feats.append(sum(flat) / 64.0)
        feats.append(max(flat))
        feats.append(min(flat))
    return feats


def cached_feature_fn(fn):
    cache: dict[str, list[float]] = {}
    def inner(fen: str) -> list[float]:
        if fen not in cache:
            cache[fen] = fn(fen)
        return cache[fen]
    return inner


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


def average_weights(weights: list[list[float]], totals: list[list[float]], count: int) -> list[list[float]]:
    if count <= 0:
        return weights
    return [[total / count for total in row] for row in totals]


def evaluate(rows: list[dict], moves: list[str], wp: list[list[float]], ww: list[list[float]], policy_feature_fn=fen_features, value_feature_fn=wdl_features) -> tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0
    p_ce = 0.0
    w_ce = 0.0
    top1 = 0.0
    for row in rows:
        x = policy_feature_fn(row["fen"])
        xv = value_feature_fn(row["fen"])
        logits = [sum(w * v for w, v in zip(weights, x)) for weights in wp]
        probs = softmax(logits)
        target = [row["policy"].get(move, 0.0) for move in moves]
        mass = sum(target)
        if mass <= 0:
            continue
        target = [v / mass for v in target]
        p_ce += cross_entropy(target, probs)
        wdl_logits = [sum(w * v for w, v in zip(weights, xv)) for weights in ww]
        wdl = softmax(wdl_logits)
        w_ce += cross_entropy(row["wdl"], wdl)
        if moves[max(range(len(moves)), key=lambda i: probs[i])] in row["policy"]:
            top1 += 1.0
    n = len(rows)
    return p_ce / n, w_ce / n, top1 / n


def train_once(rows: list[dict], moves: list[str], epochs: int, lr: float, holdout_mod: int, holdout_offset: int, average: bool, average_policy_only: bool, policy_feature_fn=fen_features, value_feature_fn=wdl_features) -> dict:
    train_rows = [row for i, row in enumerate(rows) if i % holdout_mod != holdout_offset]
    dev_rows = [row for i, row in enumerate(rows) if i % holdout_mod == holdout_offset]
    feat_dim = len(policy_feature_fn(rows[0]["fen"]))
    value_feat_dim = len(value_feature_fn(rows[0]["fen"]))
    rng = random.Random(7 + holdout_offset)
    wp = [[rng.uniform(-0.01, 0.01) for _ in range(feat_dim)] for _ in moves]
    ww = [[rng.uniform(-0.01, 0.01) for _ in range(value_feat_dim)] for _ in range(3)]
    avg_wp = [[0.0 for _ in range(feat_dim)] for _ in moves]
    avg_ww = [[0.0 for _ in range(value_feat_dim)] for _ in range(3)]
    avg_count = 0

    for epoch in range(epochs):
        rng.shuffle(train_rows)
        for row in train_rows:
            x = policy_feature_fn(row["fen"])
            xv = value_feature_fn(row["fen"])
            policy_logits = [sum(w * v for w, v in zip(weights, x)) for weights in wp]
            probs = softmax(policy_logits)
            target = [row["policy"].get(move, 0.0) for move in moves]
            mass = sum(target)
            target = [v / mass for v in target]
            for i in range(len(moves)):
                grad = probs[i] - target[i]
                for j, val in enumerate(x):
                    wp[i][j] -= lr * grad * val
            wdl_logits = [sum(w * v for w, v in zip(weights, xv)) for weights in ww]
            wdl = softmax(wdl_logits)
            for i in range(3):
                grad = wdl[i] - row["wdl"][i]
                for j, val in enumerate(xv):
                    ww[i][j] -= lr * grad * val
        if average and epoch >= epochs // 2:
            avg_count += 1
            for i, row in enumerate(wp):
                for j, val in enumerate(row):
                    avg_wp[i][j] += val
            for i, row in enumerate(ww):
                for j, val in enumerate(row):
                    avg_ww[i][j] += val

    final_wp = average_weights(wp, avg_wp, avg_count) if average else wp
    final_ww = average_weights(ww, avg_ww, avg_count) if average and not average_policy_only else ww
    train_policy_ce, train_wdl_ce, _train_top1 = evaluate(train_rows, moves, final_wp, final_ww, policy_feature_fn, value_feature_fn)
    dev_policy_ce, dev_wdl_ce, dev_top1 = evaluate(dev_rows, moves, final_wp, final_ww, policy_feature_fn, value_feature_fn)
    quality = 100.0 / (1.0 + dev_policy_ce + dev_wdl_ce)
    return {
        "score": quality,
        "train_policy_ce": train_policy_ce,
        "train_wdl_ce": train_wdl_ce,
        "dev_policy_ce": dev_policy_ce,
        "dev_wdl_ce": dev_wdl_ce,
        "dev_top1": dev_top1,
        "policy_weights": final_wp,
        "wdl_weights": final_ww,
        "feature_dim": feat_dim,
        "wdl_feature_dim": value_feat_dim,
        "weight_average_count": avg_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=["data/teacher_labels.jsonl", "data/stockfish_teacher_labels.jsonl"])
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--holdout-mod", type=int, default=2, help="hold out rows whose index modulo this value is 0")
    parser.add_argument("--out", default="artifacts/student_linear.json")
    parser.add_argument("--merge-fen", action="store_true", help="average labels that share the same FEN before splitting")
    parser.add_argument("--average-weights", action="store_true", help="use Polyak/SWA-style averaged weights from the second half of training")
    parser.add_argument("--average-policy-only", action="store_true", help="when averaging, average only policy weights and keep final WDL weights")
    parser.add_argument("--report-folds", action="store_true", help="also train alternate holdout folds for robustness reporting without changing the primary metric")
    parser.add_argument("--compare-conv-archs", action="store_true", help="train frozen-conv feature students for 16x2/24x3/48x5/64x6 architecture diagnostics")
    args = parser.parse_args()

    raw_rows = load_rows(args.train)
    rows = merge_fen_rows(raw_rows) if args.merge_fen else raw_rows
    if len(rows) < 2:
        raise SystemExit("Need at least two teacher rows")
    moves = sorted({move for row in rows for move in row["policy"]})
    result = train_once(rows, moves, args.epochs, args.lr, args.holdout_mod, 0, args.average_weights, args.average_policy_only)
    fold_scores = [result["score"]]
    if args.report_folds:
        for offset in range(1, args.holdout_mod):
            fold_scores.append(train_once(rows, moves, args.epochs, args.lr, args.holdout_mod, offset, args.average_weights, args.average_policy_only)["score"])
    fold_mean = sum(fold_scores) / len(fold_scores)
    fold_std = math.sqrt(sum((score - fold_mean) ** 2 for score in fold_scores) / len(fold_scores))

    conv_best = None
    conv_results = []
    if args.compare_conv_archs:
        for channels, layers in ((16, 2), (24, 3), (48, 5), (64, 6)):
            feature_fn = cached_feature_fn(lambda fen, c=channels, l=layers: conv_student_features(fen, c, l))
            conv = train_once(rows, moves, args.epochs, args.lr, args.holdout_mod, 0, args.average_weights, args.average_policy_only, feature_fn, feature_fn)
            conv_params = (13 * channels * 9) + max(0, layers - 1) * (channels * channels * 9)
            head_params = (len(moves) + 3) * conv["feature_dim"]
            bytes_est = 4 * (conv_params + head_params)
            score_per_kb = conv["score"] / max(bytes_est / 1024.0, 1e-9)
            item = {"channels": channels, "layers": layers, "score": conv["score"], "bytes": bytes_est, "score_per_kb": score_per_kb}
            conv_results.append(item)
            if conv_best is None or item["score_per_kb"] > conv_best["score_per_kb"]:
                conv_best = item

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kind": "linear_fen_student", "moves": moves, "policy_weights": result["policy_weights"], "wdl_weights": result["wdl_weights"], "policy_feature_dim": result["feature_dim"], "wdl_feature_dim": result["wdl_feature_dim"], "weight_average_count": result["weight_average_count"]}, separators=(",", ":")))
    print(f"METRIC distill_student_score={result['score']:.6f}")
    print(f"METRIC train_policy_ce={result['train_policy_ce']:.6f}")
    print(f"METRIC train_wdl_ce={result['train_wdl_ce']:.6f}")
    print(f"METRIC dev_policy_ce={result['dev_policy_ce']:.6f}")
    print(f"METRIC dev_wdl_ce={result['dev_wdl_ce']:.6f}")
    print(f"METRIC dev_policy_top1={result['dev_top1']:.6f}")
    print(f"METRIC teacher_rows={len(rows)}")
    print(f"METRIC raw_teacher_rows={len(raw_rows)}")
    print(f"METRIC move_vocab={len(moves)}")
    print(f"METRIC feature_dim={result['feature_dim']}")
    print(f"METRIC wdl_feature_dim={result['wdl_feature_dim']}")
    print(f"METRIC weight_average_count={result['weight_average_count']}")
    print(f"METRIC average_policy_only={1 if args.average_policy_only else 0}")
    print(f"METRIC fold_score_mean={fold_mean:.6f}")
    print(f"METRIC fold_score_std={fold_std:.6f}")
    if conv_best is not None:
        for item in conv_results:
            print(f"METRIC conv_{item['channels']}x{item['layers']}_score={item['score']:.6f}")
            print(f"METRIC conv_{item['channels']}x{item['layers']}_score_per_kb={item['score_per_kb']:.6f}")
        print(f"METRIC best_conv_channels={conv_best['channels']}")
        print(f"METRIC best_conv_layers={conv_best['layers']}")
        print(f"METRIC best_conv_score={conv_best['score']:.6f}")
        print(f"METRIC best_conv_bytes={conv_best['bytes']}")
        print(f"METRIC best_conv_score_per_kb={conv_best['score_per_kb']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
