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


def cached_feature_fn(fn, disk_path: str | None = None):
    cache: dict[str, list[float]] = {}
    path = Path(disk_path) if disk_path else None
    if path and path.exists():
        try:
            cache.update(json.loads(path.read_text()))
        except Exception:
            cache.clear()
    dirty = False
    def inner(fen: str) -> list[float]:
        nonlocal dirty
        if fen not in cache:
            cache[fen] = fn(fen)
            dirty = True
        return cache[fen]
    def flush() -> None:
        nonlocal dirty
        if path and dirty:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(cache, separators=(",", ":")))
            tmp.replace(path)
            dirty = False
    inner.flush = flush  # type: ignore[attr-defined]
    inner.cache_size = lambda: len(cache)  # type: ignore[attr-defined]
    return inner


def load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_selfplay_rows(paths: list[str], weight: float) -> list[dict]:
    rows: list[dict] = []
    for row in load_rows(paths):
        if not row.get("policy") or not row.get("result"):
            continue
        policy = {move: float(prob) for move, prob in row["policy"].items() if float(prob) > 0}
        mass = sum(policy.values())
        if mass <= 0:
            continue
        rows.append({
            "fen": row["fen"],
            "policy": {move: prob / mass for move, prob in policy.items()},
            "wdl": [float(v) for v in row["result"]],
            "q": float(row["result"][0]) - float(row["result"][2]),
            "_source": "selfplay",
            "_weight": weight,
        })
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
        q = (acc["wdl"][0] - acc["wdl"][2]) / count
        out.append({"fen": fen, "policy": policy, "wdl": [prob / count for prob in acc["wdl"]], "q": q})
    return out


def cross_entropy(target: list[float], pred: list[float]) -> float:
    return -sum(t * math.log(max(p, 1e-12)) for t, p in zip(target, pred))


def average_weights(weights: list[list[float]], totals: list[list[float]], count: int) -> list[list[float]]:
    if count <= 0:
        return weights
    return [[total / count for total in row] for row in totals]


def q_to_wdl(q: float) -> list[float]:
    q = max(-1.0, min(1.0, q))
    return [max(q, 0.0), 1.0 - abs(q), max(-q, 0.0)]


def evaluate_q_wdl(rows: list[dict], ww: list[list[float]], value_feature_fn=wdl_features) -> float:
    if not rows:
        return 0.0
    total = 0.0
    for row in rows:
        xv = value_feature_fn(row["fen"])
        wdl_logits = [sum(w * v for w, v in zip(weights, xv)) for weights in ww]
        total += cross_entropy(q_to_wdl(row.get("q", row["wdl"][0] - row["wdl"][2])), softmax(wdl_logits))
    return total / len(rows)


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
            row_weight = float(row.get("_weight", 1.0))
            for i in range(len(moves)):
                grad = row_weight * (probs[i] - target[i])
                for j, val in enumerate(x):
                    wp[i][j] -= lr * grad * val
            wdl_logits = [sum(w * v for w, v in zip(weights, xv)) for weights in ww]
            wdl = softmax(wdl_logits)
            for i in range(3):
                grad = row_weight * (wdl[i] - row["wdl"][i])
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
    dev_q_wdl_ce = evaluate_q_wdl(dev_rows, final_ww, value_feature_fn)
    quality = 100.0 / (1.0 + dev_policy_ce + dev_wdl_ce)
    return {
        "score": quality,
        "train_policy_ce": train_policy_ce,
        "train_wdl_ce": train_wdl_ce,
        "dev_policy_ce": dev_policy_ce,
        "dev_wdl_ce": dev_wdl_ce,
        "dev_q_wdl_ce": dev_q_wdl_ce,
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
    parser.add_argument("--selfplay-train", nargs="*", default=[], help="optional self-play JSONL rows with visit policy and result targets")
    parser.add_argument("--selfplay-weight", type=float, default=0.0, help="per-row training weight for self-play rows")
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--holdout-mod", type=int, default=2, help="hold out rows whose index modulo this value is 0")
    parser.add_argument("--out", default="artifacts/student_linear.json")
    parser.add_argument("--merge-fen", action="store_true", help="average labels that share the same FEN before splitting")
    parser.add_argument("--average-weights", action="store_true", help="use Polyak/SWA-style averaged weights from the second half of training")
    parser.add_argument("--average-policy-only", action="store_true", help="when averaging, average only policy weights and keep final WDL weights")
    parser.add_argument("--report-folds", action="store_true", help="also train alternate holdout folds for robustness reporting without changing the primary metric")
    parser.add_argument("--compare-conv-archs", action="store_true", help="train frozen-conv feature students for 16x2/24x3/48x5/64x6 architecture diagnostics")
    parser.add_argument("--primary-conv-arch", choices=["16x2", "24x3", "48x5", "64x6"], help="use the selected frozen-conv feature student as the primary artifact and metric")
    parser.add_argument("--feature-cache", help="optional JSON cache for expensive frozen-conv features keyed by FEN")
    args = parser.parse_args()

    raw_rows = load_rows(args.train)
    teacher_rows = merge_fen_rows(raw_rows) if args.merge_fen else raw_rows
    for row in teacher_rows:
        row.setdefault("_source", "teacher")
        row.setdefault("_weight", 1.0)
    selfplay_rows = load_selfplay_rows(args.selfplay_train, args.selfplay_weight) if args.selfplay_train and args.selfplay_weight > 0 else []
    rows = teacher_rows + selfplay_rows
    if len(rows) < 2:
        raise SystemExit("Need at least two teacher rows")
    moves = sorted({move for row in rows for move in row["policy"]})
    primary_kind = "linear_fen_student"
    primary_feature_fn = fen_features
    primary_value_feature_fn = wdl_features
    primary_conv_channels = 0
    primary_conv_layers = 0
    if args.primary_conv_arch:
        primary_conv_channels, primary_conv_layers = [int(v) for v in args.primary_conv_arch.split("x")]
        cache_path = args.feature_cache or f"artifacts/cache/conv_features_{args.primary_conv_arch}.json"
        primary_feature_fn = cached_feature_fn(lambda fen, c=primary_conv_channels, l=primary_conv_layers: conv_student_features(fen, c, l), cache_path)
        primary_value_feature_fn = primary_feature_fn
        primary_kind = "frozen_conv_fen_student"
    result = train_once(rows, moves, args.epochs, args.lr, args.holdout_mod, 0, args.average_weights, args.average_policy_only, primary_feature_fn, primary_value_feature_fn)
    fold_scores = [result["score"]]
    if args.report_folds:
        for offset in range(1, args.holdout_mod):
            fold_scores.append(train_once(rows, moves, args.epochs, args.lr, args.holdout_mod, offset, args.average_weights, args.average_policy_only, primary_feature_fn, primary_value_feature_fn)["score"])
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
    if hasattr(primary_feature_fn, "flush"):
        primary_feature_fn.flush()  # type: ignore[attr-defined]
    artifact = {"kind": primary_kind, "moves": moves, "policy_weights": result["policy_weights"], "wdl_weights": result["wdl_weights"], "policy_feature_dim": result["feature_dim"], "wdl_feature_dim": result["wdl_feature_dim"], "weight_average_count": result["weight_average_count"], "conv_channels": primary_conv_channels, "conv_layers": primary_conv_layers}
    artifact_text = json.dumps(artifact, separators=(",", ":"))
    out.write_text(artifact_text)
    model_weight_count = (len(moves) * result["feature_dim"]) + (3 * result["wdl_feature_dim"])
    estimated_eval_ops = model_weight_count + (primary_conv_channels * primary_conv_layers * 64 if primary_conv_channels else 0)
    print(f"METRIC distill_student_score={result['score']:.6f}")
    print(f"METRIC train_policy_ce={result['train_policy_ce']:.6f}")
    print(f"METRIC train_wdl_ce={result['train_wdl_ce']:.6f}")
    print(f"METRIC dev_policy_ce={result['dev_policy_ce']:.6f}")
    print(f"METRIC dev_wdl_ce={result['dev_wdl_ce']:.6f}")
    print(f"METRIC dev_q_wdl_ce={result['dev_q_wdl_ce']:.6f}")
    print(f"METRIC dev_wdl_q_ce_gap={result['dev_q_wdl_ce'] - result['dev_wdl_ce']:.6f}")
    print(f"METRIC dev_policy_only_score={100.0 / (1.0 + result['dev_policy_ce']):.6f}")
    print(f"METRIC dev_wdl_only_score={100.0 / (1.0 + result['dev_wdl_ce']):.6f}")
    print(f"METRIC dev_loss_balance_gap={result['dev_policy_ce'] - result['dev_wdl_ce']:.6f}")
    print(f"METRIC dev_policy_top1={result['dev_top1']:.6f}")
    print(f"METRIC teacher_rows={len(teacher_rows)}")
    print(f"METRIC raw_teacher_rows={len(raw_rows)}")
    print(f"METRIC selfplay_train_rows={len(selfplay_rows)}")
    print(f"METRIC selfplay_weight={args.selfplay_weight:.6f}")
    print(f"METRIC move_vocab={len(moves)}")
    print(f"METRIC feature_dim={result['feature_dim']}")
    print(f"METRIC wdl_feature_dim={result['wdl_feature_dim']}")
    print(f"METRIC weight_average_count={result['weight_average_count']}")
    print(f"METRIC average_policy_only={1 if args.average_policy_only else 0}")
    print(f"METRIC primary_conv_channels={primary_conv_channels}")
    print(f"METRIC primary_conv_layers={primary_conv_layers}")
    print(f"METRIC model_json_bytes={len(artifact_text.encode('utf-8'))}")
    print(f"METRIC model_weight_count={model_weight_count}")
    print(f"METRIC estimated_eval_ops={estimated_eval_ops}")
    print(f"METRIC feature_cache_entries={primary_feature_fn.cache_size() if hasattr(primary_feature_fn, 'cache_size') else 0}")
    print(f"METRIC browser_json_compatible=1")
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
