#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, statistics, sys, time
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

PIECES = "PNBRQKpnbrqk"
PIECES_WITH_EMPTY = ".PNBRQKpnbrqk"
ROLE_INDEX = {"p": 1, "n": 2, "b": 3, "r": 4, "q": 5, "k": 6}
PIECE_VALUE = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}
PROMO_INDEX = {"n": 1, "b": 2, "r": 3, "q": 4}
PROMO_CLASS = {"n": 0, "b": 1, "r": 2, "q": 3}


def uci_to_move_parts(uci: str):
    move = chess.Move.from_uci(uci)
    promo = None
    if move.promotion:
        promo = {chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}[move.promotion]
    return move.from_square, move.to_square, promo


def action_id(move: chess.Move) -> int:
    promo = 0
    if move.promotion:
        promo = {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
    return ((move.from_square * 64 + move.to_square) * 5) + promo


def chessbench_class(move: chess.Move) -> int:
    ft = move.from_square * 64 + move.to_square
    if not move.promotion:
        return ft
    promo = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3}[move.promotion]
    return 4096 + ft * 4 + promo


def piece_char(board: chess.Board, sq: int) -> str | None:
    p = board.piece_at(sq)
    if p is None:
        return None
    ch = p.symbol()
    return ch


def add_piece_planes(data: np.ndarray, board: chess.Board, offset: int):
    placement = board.board_fen()
    rank = 0
    file = 0
    for ch in placement:
        if ch == "/":
            rank += 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            pi = PIECES.find(ch)
            if pi >= 0 and offset + pi < data.shape[0]:
                data[offset + pi, rank, file] = 1.0
            file += 1


def onnx_input_planes(board: chess.Board, meta: dict, history_fens: list[str] | None = None) -> np.ndarray:
    history_fens = history_fens or []
    input_planes = int(meta.get("input_planes", 14))
    history = int(meta.get("history_plies", 0))
    data = np.zeros((input_planes, 8, 8), dtype=np.float32)
    add_piece_planes(data, board, 0)
    for h, fen in enumerate(history_fens[:history]):
        add_piece_planes(data, chess.Board(fen), 12 * (h + 1))
    state0 = 12 * (history + 1)
    if state0 < input_planes:
        data[state0, :, :] = 1.0 if board.turn == chess.WHITE else -1.0
    if input_planes - state0 >= 10:
        if board.has_kingside_castling_rights(chess.WHITE): data[state0 + 1, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): data[state0 + 2, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): data[state0 + 3, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): data[state0 + 4, :, :] = 1.0
        if board.ep_square is not None:
            f = chess.square_file(board.ep_square)
            r = 7 - chess.square_rank(board.ep_square)
            data[state0 + 5, r, f] = 1.0
        data[state0 + 6, :, :] = 1.0
        data[state0 + 7, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    elif state0 + 1 < input_planes:
        data[state0 + 1, :, :] = 1.0
    return data


def chebyshev(a: int, b: int) -> int:
    return max(abs(chess.square_file(a) - chess.square_file(b)), abs(chess.square_rank(a) - chess.square_rank(b)))


def moveformer_legal_inputs(boards: list[chess.Board], width: int, feature_count: int):
    action_ids = np.full((len(boards), width), 20480, dtype=np.int64)
    features = np.zeros((len(boards), width, feature_count), dtype=np.float32)
    mask = np.zeros((len(boards), width), dtype=np.float32)
    legal_lists = []
    for bi, board in enumerate(boards):
        moves = list(board.legal_moves)
        legal_lists.append(moves)
        own_king = board.king(board.turn) or (4 if board.turn == chess.WHITE else 60)
        enemy_king = board.king(not board.turn) or (60 if board.turn == chess.WHITE else 4)
        for j, move in enumerate(moves[:width]):
            action_ids[bi, j] = action_id(move)
            mask[bi, j] = 1.0
            moving = piece_char(board, move.from_square)
            captured = piece_char(board, move.to_square)
            moving_role = moving.lower() if moving else ""
            captured_role = captured.lower() if captured else ""
            promo_ch = {chess.KNIGHT: "n", chess.BISHOP: "b", chess.ROOK: "r", chess.QUEEN: "q"}.get(move.promotion, "")
            vals = [
                ROLE_INDEX.get(moving_role, 0),
                ROLE_INDEX.get(captured_role, 0),
                PROMO_INDEX.get(promo_ch, 0),
                1 if captured else 0,
                0,
                0,
                1 if promo_ch else 0,
                0, 0, 0, 0, 0, 0, 0,
                PIECE_VALUE.get(moving_role, 0),
                PIECE_VALUE.get(captured_role, 0),
                PIECE_VALUE.get(captured_role, 0) + ((PIECE_VALUE.get(promo_ch, 0) - 1) if promo_ch else 0),
                0,
                chebyshev(move.to_square, enemy_king),
                chebyshev(move.to_square, own_king),
            ]
            features[bi, j, : min(feature_count, len(vals))] = vals[:feature_count]
    return legal_lists, action_ids, features, mask


def piece_id(board: chess.Board, sq: int) -> int:
    p = board.piece_at(sq)
    if p is None:
        return 0
    return max(0, PIECES_WITH_EMPTY.find(p.symbol()))


def castle_mask(board: chess.Board) -> int:
    return int(board.has_kingside_castling_rights(chess.WHITE)) | (int(board.has_queenside_castling_rights(chess.WHITE)) << 1) | (int(board.has_kingside_castling_rights(chess.BLACK)) << 2) | (int(board.has_queenside_castling_rights(chess.BLACK)) << 3)


def squareformer_tokens(boards: list[chess.Board], meta: dict):
    compact = meta.get("input_mode") == "embedding" or "compact" in str(meta.get("input_format", ""))
    history = int(meta.get("history_plies", 0))
    stride = int(meta.get("token_features", history + 9 if compact else meta.get("input_dim", 0)))
    if compact:
        data = np.zeros((len(boards), 64, stride), dtype=np.int64)
        for bi, board in enumerate(boards):
            for sq in chess.SQUARES:
                data[bi, sq, 0] = piece_id(board, sq)
                base = history + 1
                if base + 0 < stride: data[bi, sq, base + 0] = 1 if board.turn == chess.WHITE else 2
                if base + 1 < stride: data[bi, sq, base + 1] = castle_mask(board)
                if base + 2 < stride: data[bi, sq, base + 2] = 1 if board.ep_square == sq else 0
                if base + 3 < stride: data[bi, sq, base + 3] = max(0, min(255, board.halfmove_clock))
                if base + 4 < stride: data[bi, sq, base + 4] = chess.square_rank(sq)
                if base + 5 < stride: data[bi, sq, base + 5] = chess.square_file(sq)
                if base + 6 < stride: data[bi, sq, base + 6] = (chess.square_rank(sq) + chess.square_file(sq)) & 1
                if base + 7 < stride: data[bi, sq, base + 7] = sq
        return data
    data = np.zeros((len(boards), 64, stride), dtype=np.float32)
    planes_per_board = 13
    for bi, board in enumerate(boards):
        for sq in chess.SQUARES:
            data[bi, sq, piece_id(board, sq)] = 1.0
            base = (history + 1) * planes_per_board
            if base + 0 < stride: data[bi, sq, base + 0] = 1.0 if board.turn == chess.WHITE else 0.0
            if base + 1 < stride: data[bi, sq, base + 1] = 1.0 if board.turn == chess.BLACK else 0.0
            if base + 2 < stride: data[bi, sq, base + 2] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
            if base + 3 < stride: data[bi, sq, base + 3] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
            if base + 4 < stride: data[bi, sq, base + 4] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
            if base + 5 < stride: data[bi, sq, base + 5] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
            if base + 6 < stride: data[bi, sq, base + 6] = 1.0 if board.ep_square == sq else 0.0
            if base + 7 < stride: data[bi, sq, base + 7] = min(255, board.halfmove_clock) / 100.0
    return data


def sample_boards(n: int, fen: str) -> list[chess.Board]:
    out = []
    board = chess.Board(fen)
    for i in range(n):
        out.append(board.copy(stack=False))
        moves = list(board.legal_moves)
        if moves:
            board.push(moves[(i * 7 + 3) % len(moves)])
        else:
            board = chess.Board(fen)
    return out


def feeds_for(boards: list[chess.Board], meta: dict) -> dict[str, np.ndarray]:
    kind = str(meta.get("kind", ""))
    arch = str(meta.get("architecture", ""))
    if kind in ("squareformer", "squareformer_v2") or "square" in arch:
        feeds = {"tokens": squareformer_tokens(boards, meta)}
        av_width = int(meta.get("onnx_fixed_legal_moves", meta.get("max_legal_moves", 0)) or 0)
        if meta.get("av_head_exported") and av_width > 0:
            ids = np.zeros((len(boards), av_width), dtype=np.int64)
            for bi, board in enumerate(boards):
                for j, move in enumerate(list(board.legal_moves)[:av_width]):
                    ids[bi, j] = chessbench_class(move)
            feeds["legal_action_ids"] = ids
        return feeds
    planes = np.stack([onnx_input_planes(b, meta) for b in boards], axis=0)
    feeds = {"planes": planes}
    if "move_token" in arch or "move_transformer" in arch:
        width = int(meta.get("onnx_fixed_legal_moves", meta.get("max_legal_moves", 128)) or 128)
        feature_count = int(meta.get("num_move_features", 20) or 20)
        _, action_ids, features, mask = moveformer_legal_inputs(boards, width, feature_count)
        feeds.update({"legal_action_ids": action_ids, "legal_features": features, "legal_mask": mask})
    return feeds


def median(xs):
    return statistics.median(xs) if xs else 0.0


def p90(xs):
    if not xs: return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, math.floor(len(s) * 0.9))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--provider", action="append", default=[])
    ap.add_argument("--require-provider", action="append", default=[], help="Fail before benchmarking unless this ORT provider is available and active")
    ap.add_argument("--label", default="model")
    ap.add_argument("--positions", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--batches", default="1,2,4,8,16,32")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    args = ap.parse_args()
    meta = json.loads(Path(args.meta).read_text())
    providers = args.provider or ["CPUExecutionProvider"]
    available_providers = ort.get_available_providers()
    missing = [p for p in args.require_provider if p not in available_providers]
    if missing:
        print(f"ERROR missing_required_providers={missing} available_providers={available_providers}", file=sys.stderr)
        sys.exit(2)
    sess = ort.InferenceSession(args.model, providers=providers)
    inactive = [p for p in args.require_provider if p not in sess.get_providers()]
    if inactive:
        print(f"ERROR inactive_required_providers={inactive} requested_providers={providers} active_providers={sess.get_providers()} available_providers={available_providers}", file=sys.stderr)
        sys.exit(3)
    available_inputs = {i.name for i in sess.get_inputs()}
    available_outputs = [o.name for o in sess.get_outputs()]
    batch_sizes = [int(x) for x in args.batches.split(",") if x]
    boards = sample_boards(max(args.positions, args.warmup, max(batch_sizes)), args.fen)
    print(f"INFO label={args.label} providers={sess.get_providers()} inputs={sorted(available_inputs)} outputs={available_outputs}")
    for b in boards[: args.warmup]:
        feeds = {k: v for k, v in feeds_for([b], meta).items() if k in available_inputs}
        sess.run(None, feeds)
    for batch in batch_sizes:
        times = []
        per_pos = []
        for _ in range(args.repeats):
            count = 0
            t0 = time.perf_counter()
            for i in range(0, args.positions, batch):
                chunk = boards[i: min(args.positions, i + batch)]
                feeds = {k: v for k, v in feeds_for(chunk, meta).items() if k in available_inputs}
                sess.run(None, feeds)
                count += len(chunk)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)
            per_pos.append(elapsed_ms / max(1, count))
        med = median(per_pos)
        q90 = p90(per_pos)
        print(f"RESULT label={args.label} batch={batch} positions={args.positions} repeats={args.repeats} median_ms_per_pos={med:.4f} p90_ms_per_pos={q90:.4f} median_positions_per_s={1000.0 / max(1e-9, med):.1f} total_median_ms={median(times):.3f}")
        metric_label = ''.join(c if c.isalnum() or c == '_' else '_' for c in args.label)
        print(f"METRIC {metric_label}_b{batch}_native_median_ms_per_pos={med:.4f}")


if __name__ == "__main__":
    main()
