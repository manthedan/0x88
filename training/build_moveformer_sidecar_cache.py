#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from train_residual_torch import fixed_policy_moves, planes, input_plane_count

PROMO_TO_ID = {None: 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4}
PIECE_VALUE = {0: 0.0, 1: 1.0, 2: 3.0, 3: 3.0, 4: 5.0, 5: 9.0, 6: 0.0}
FEATURE_NAMES = [
    'moving_piece_type', 'captured_piece_type', 'promotion_type',
    'is_capture', 'is_check', 'is_castle', 'is_promotion', 'is_en_passant',
    'from_attacked_by_enemy_pre', 'from_defended_by_own_pre',
    'to_attacked_by_enemy_after', 'to_defended_by_own_after',
    'to_enemy_attackers_after_capped8', 'to_own_defenders_after_capped8',
    'moving_piece_value', 'captured_piece_value', 'material_delta',
    'from_piece_pinned_pre', 'king_distance_to_enemy_after', 'king_distance_to_own_after',
]

@contextmanager
def opener(path):
    path = str(path)
    if path.endswith('.zst'):
        p = subprocess.Popen(['zstd', '-dc', path], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout is not None
            yield p.stdout
        finally:
            if p.stdout is not None:
                p.stdout.close()
            p.wait()
    else:
        with open(path) as f:
            yield f

def square_dist(a, b):
    af, ar = a % 8, a // 8
    bf, br = b % 8, b // 8
    return max(abs(af-bf), abs(ar-br))

def move_to_action_id(m):
    return (m.from_square * 64 + m.to_square) * 5 + PROMO_TO_ID[m.promotion and {1:'n',2:'b',3:'r',4:'q',5:'q'}.get(m.promotion, None)]

def move_to_policy_uci(m):
    return m.uci()

def move_feature(chess, board, move):
    piece = board.piece_at(move.from_square)
    moved_type = piece.piece_type if piece else 0
    moved_color = piece.color if piece else board.turn
    enemy = not moved_color
    cap = board.piece_at(move.to_square)
    is_ep = bool(piece and piece.piece_type == chess.PAWN and board.ep_square == move.to_square and cap is None and chess.square_file(move.from_square) != chess.square_file(move.to_square))
    if is_ep:
        cap = board.piece_at(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    cap_type = cap.piece_type if cap else 0
    promo_id = PROMO_TO_ID[move.promotion and {chess.KNIGHT:'n', chess.BISHOP:'b', chess.ROOK:'r', chess.QUEEN:'q'}.get(move.promotion)]
    from_enemy = len(board.attackers(enemy, move.from_square))
    from_own = len(board.attackers(moved_color, move.from_square))
    pinned = bool(piece and board.is_pinned(moved_color, move.from_square))
    gives_check = board.gives_check(move)
    is_castle = board.is_castling(move)
    after = board.copy(stack=False)
    after.push(move)
    enemy_after = after.turn
    own_after = not after.turn
    to_enemy = len(after.attackers(enemy_after, move.to_square))
    to_own = len(after.attackers(own_after, move.to_square))
    own_king = after.king(own_after)
    enemy_king = after.king(enemy_after)
    promo_gain = 0.0
    if move.promotion:
        promo_gain = PIECE_VALUE.get(move.promotion, 0.0) - PIECE_VALUE[chess.PAWN]
    material_delta = PIECE_VALUE.get(cap_type, 0.0) + promo_gain
    return [
        float(moved_type), float(cap_type), float(promo_id),
        float(cap_type != 0 or is_ep), float(gives_check), float(is_castle), float(move.promotion is not None), float(is_ep),
        float(from_enemy > 0), float(from_own > 0),
        float(to_enemy > 0), float(to_own > 0),
        float(min(8, to_enemy)), float(min(8, to_own)),
        PIECE_VALUE.get(moved_type, 0.0), PIECE_VALUE.get(cap_type, 0.0), float(material_delta),
        float(pinned), float(square_dist(move.to_square, enemy_king) if enemy_king is not None else 8), float(square_dist(move.to_square, own_king) if own_king is not None else 8),
    ]

def count_eligible(paths, max_rows, policy_index):
    n = skipped_multi = skipped_unknown = malformed = 0
    for path in paths:
        with opener(path) as f:
            for line in f:
                if n >= max_rows:
                    return n, skipped_multi, skipped_unknown, malformed
                try:
                    r = json.loads(line)
                    pol = r.get('policy') or {}
                    if len(pol) != 1:
                        skipped_multi += 1; continue
                    mv = next(iter(pol))
                    if mv not in policy_index:
                        skipped_unknown += 1; continue
                    n += 1
                except Exception:
                    malformed += 1
    return n, skipped_multi, skipped_unknown, malformed

def main():
    ap = argparse.ArgumentParser(description='Build MoveFormer legal-move sidecar cache from supervised JSONL/ZST shards.')
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-rows', type=int, default=10000)
    ap.add_argument('--max-legal-moves', type=int, default=128)
    ap.add_argument('--history-plies', type=int, default=2)
    ap.add_argument('--state-planes', action='store_true')
    ap.add_argument('--no-board-cache', action='store_true')
    ap.add_argument('--no-legal-uci', action='store_true', help='Do not materialize legal_uci.uint8; trainer/runtime do not need it.')
    ap.add_argument('--assume-rows', type=int, default=0, help='Trust this eligible row count and skip the JSONL counting pass.')
    ap.add_argument('--progress-every', type=int, default=10000)
    args = ap.parse_args()

    import chess
    moves = fixed_policy_moves(); policy_index = {m:i for i,m in enumerate(moves)}
    if args.assume_rows > 0:
        n = min(int(args.assume_rows), int(args.max_rows))
        skipped_multi = skipped_unknown = malformed = 0
        print(f'METRIC assumed_rows={n}', flush=True)
    else:
        n, skipped_multi, skipped_unknown, malformed = count_eligible(args.input, args.max_rows, policy_index)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    K = args.max_legal_moves; F = len(FEATURE_NAMES); C = input_plane_count(args.history_plies, args.state_planes)
    # Important for walltime: np.memmap('w+') creates sparse/truncated files.  Do not
    # whole-array-fill hundreds of GiB just to initialize padded legal slots.  Padded
    # slots are ignored by legal_mask=0, and unwritten sparse extents read back as 0.
    x = None if args.no_board_cache else np.memmap(out/'x.int8', np.int8, 'w+', shape=(n,C,8,8))
    y = np.memmap(out/'policy_index.int64', np.int64, 'w+', shape=(n,))
    yslot = np.memmap(out/'policy_legal_slot.int16', np.int16, 'w+', shape=(n,))
    wdl = np.memmap(out/'wdl.float32', np.float32, 'w+', shape=(n,3)); q = np.memmap(out/'q.float32', np.float32, 'w+', shape=(n,))
    lpi = np.memmap(out/'legal_policy_indices.int64', np.int64, 'w+', shape=(n,K))
    laid = np.memmap(out/'legal_action_ids.int64', np.int64, 'w+', shape=(n,K))
    luci = None if args.no_legal_uci else np.memmap(out/'legal_uci.uint8', np.uint8, 'w+', shape=(n,K,5))
    lf = np.memmap(out/'legal_features.float32', np.float32, 'w+', shape=(n,K,F))
    mask = np.memmap(out/'legal_mask.float32', np.float32, 'w+', shape=(n,K))

    written = legal_found = idx_found = trunc = legal_total = bad_fen = 0
    for path in args.input:
        with opener(path) as f:
            for line in f:
                if written >= n: break
                try:
                    r = json.loads(line); pol = r.get('policy') or {}
                    if len(pol) != 1: continue
                    target_uci = next(iter(pol))
                    if target_uci not in policy_index: continue
                    board = chess.Board(r['fen'])
                    legals = list(board.legal_moves)
                except Exception:
                    bad_fen += 1; continue
                y[written] = policy_index[target_uci]
                yslot[written] = -1
                wdl[written] = np.asarray(r.get('wdl', [.25,.5,.25]), dtype=np.float32)
                q[written] = float(r.get('q', wdl[written,0] - wdl[written,2]))
                if x is not None:
                    x[written] = np.asarray(planes(r['fen'], r.get('history_fens', [])[:args.history_plies], args.history_plies, args.state_planes), dtype=np.int8)
                if len(legals) > K:
                    trunc += 1
                legal_total += len(legals)
                for j,m in enumerate(legals[:K]):
                    uci = move_to_policy_uci(m)
                    pi = policy_index.get(uci, -1)
                    if pi >= 0: idx_found += 1
                    lpi[written,j] = pi
                    laid[written,j] = move_to_action_id(m)
                    if luci is not None:
                        b = uci.encode('ascii')[:5]
                        luci[written,j,:len(b)] = np.frombuffer(b, dtype=np.uint8)
                    lf[written,j] = np.asarray(move_feature(chess, board, m), dtype=np.float32)
                    mask[written,j] = 1.0
                    if uci == target_uci:
                        yslot[written] = j
                if yslot[written] >= 0:
                    legal_found += 1
                written += 1
                if args.progress_every and written % args.progress_every == 0:
                    print(f'METRIC rows_written={written}', flush=True)
        if written >= n: break

    arrays = [a for a in [x,y,yslot,wdl,q,lpi,laid,luci,lf,mask] if a is not None]
    for a in arrays: a.flush()
    meta = {
        'format': 'moveformer_sidecar_cache_v1', 'rows': int(written), 'allocated_rows': int(n), 'max_legal_moves': int(K),
        'move_feature_names': FEATURE_NAMES, 'num_move_features': int(F),
        'has_board_cache': x is not None, 'has_legal_uci': luci is not None, 'input_planes': int(C) if x is not None else None,
        'history_plies': int(args.history_plies), 'state_planes': bool(args.state_planes),
        'policy_map': 'uci_queen_knight_promo_v1', 'policy_size': len(moves), 'moves': moves,
        'action_id_mapping': '(from * 64 + to) * 5 + promo, promo n=1,b=2,r=3,q=4',
        'source_inputs': [str(p) for p in args.input],
        'skipped_multi_policy': int(skipped_multi), 'skipped_unknown_policy': int(skipped_unknown), 'malformed_count_estimate': int(malformed), 'bad_fen_rows': int(bad_fen),
        'policy_target_legal_rate': float(legal_found / max(1, written)),
        'legal_truncation_rate': float(trunc / max(1, written)),
        'avg_legal_moves': float(legal_total / max(1, written)),
    }
    (out/'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'METRIC rows_written={written}')
    print(f'METRIC policy_target_legal_rate={meta["policy_target_legal_rate"]:.6f}')
    print(f'METRIC policy_index_found_rate={1.0 if legal_total == 0 else idx_found / max(1, min(legal_total, written*K)):.6f}')
    print(f'METRIC legal_truncation_rate={meta["legal_truncation_rate"]:.6f}')
    print(f'METRIC avg_legal_moves={meta["avg_legal_moves"]:.6f}')
    print(f'METRIC bad_fen_rows={bad_fen}')

if __name__ == '__main__':
    main()
