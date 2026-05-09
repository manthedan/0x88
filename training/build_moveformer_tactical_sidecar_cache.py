#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from train_residual_torch import fixed_policy_moves, input_plane_count
from build_moveformer_sidecar_cache import PROMO_TO_ID, PIECE_VALUE, FEATURE_NAMES as BASE_FEATURE_NAMES, opener, count_eligible, move_to_action_id, move_to_policy_uci, move_feature

ATTACK_FEATURE_NAMES = [
    'from_enemy_attackers_count_pre_capped8','from_own_defenders_count_pre_capped8',
    'to_enemy_attackers_after_pawn','to_enemy_attackers_after_knight','to_enemy_attackers_after_bishop','to_enemy_attackers_after_rook','to_enemy_attackers_after_queen','to_enemy_attackers_after_king',
    'to_own_defenders_after_pawn','to_own_defenders_after_knight','to_own_defenders_after_bishop','to_own_defenders_after_rook','to_own_defenders_after_queen','to_own_defenders_after_king',
    'to_lva_enemy_after','to_lvd_own_after','to_see_lite_after','moved_piece_hanging_after','moved_piece_defended_after',
]
DELTA_FEATURE_NAMES = [
    'queen_lost_immediately_after','own_queen_en_prise_after','enemy_can_capture_own_queen_after','enemy_can_capture_moved_piece_after',
    'capture_value_minus_moved_value','promotion_gain','material_delta_after_move_signed','own_legal_moves_pre_capped64','enemy_replies_after_capped64',
    'is_quiet_hanging_move','is_capture_to_undefended_square','is_sacrifice_like',
]
RAY_FEATURE_NAMES = [
    'own_slider_attackers_enemy_king_pre','own_slider_attackers_enemy_king_after','own_slider_attackers_enemy_king_delta',
    'own_slider_attackers_enemy_queen_pre','own_slider_attackers_enemy_queen_after','own_slider_attackers_enemy_queen_delta',
    'enemy_slider_attackers_own_king_after','enemy_slider_attackers_own_queen_after',
    'moved_piece_slider_pressure_enemy_king_after','moved_piece_slider_pressure_enemy_queen_after',
    'move_opens_line_from_own_rook_bishop_queen','queen_exposed_to_slider_after',
]
KINGZONE_FEATURE_NAMES = [
    'move_to_enemy_king_zone','move_from_own_king_zone','capture_in_enemy_king_zone','check_or_adjacent_enemy_king',
    'own_attacks_enemy_king_zone_pre_capped16','own_attacks_enemy_king_zone_after_capped16','own_attacks_enemy_king_zone_delta',
    'enemy_attacks_own_king_zone_pre_capped16','enemy_attacks_own_king_zone_after_capped16','enemy_attacks_own_king_zone_delta',
    'enemy_king_escape_squares_after_capped8','own_king_escape_squares_after_capped8',
]
GROUPS = {
    'base': BASE_FEATURE_NAMES,
    'attack': ATTACK_FEATURE_NAMES,
    'attackmap': ATTACK_FEATURE_NAMES,
    'delta': DELTA_FEATURE_NAMES,
    'afterstate': DELTA_FEATURE_NAMES,
    'ray': RAY_FEATURE_NAMES,
    'ray_summary': RAY_FEATURE_NAMES,
    'king': KINGZONE_FEATURE_NAMES,
    'kingzone': KINGZONE_FEATURE_NAMES,
}

def group_names(spec: str):
    raw=[x.strip().lower() for x in spec.replace('+',',').split(',') if x.strip()]
    if not raw: raw=['base']
    if 'all' in raw or 'all_tactical' in raw:
        raw=['base','attack','delta','ray','kingzone']
    if 'base' not in raw: raw=['base']+raw
    seen=[]
    for g in raw:
        if g not in GROUPS: raise SystemExit(f'unknown feature group {g}; choose base,attack,delta,ray,kingzone,all')
        if g not in seen: seen.append(g)
    return seen

def feature_names(groups):
    out=[]
    for g in groups: out += GROUPS[g]
    return out

def attackers_by_type(chess, board, color, sq):
    out=[0]*6
    for a in board.attackers(color, sq):
        p=board.piece_at(a)
        if p: out[p.piece_type-1]+=1
    return out

def lva(chess, board, color, sq):
    vals=[]
    for a in board.attackers(color, sq):
        p=board.piece_at(a)
        if p: vals.append(PIECE_VALUE.get(p.piece_type, 0.0))
    return min(vals) if vals else 0.0

def square_dist(chess, a, b):
    return max(abs(chess.square_file(a)-chess.square_file(b)), abs(chess.square_rank(a)-chess.square_rank(b)))

def king_zone(chess, king_sq):
    if king_sq is None: return []
    f0=chess.square_file(king_sq); r0=chess.square_rank(king_sq); out=[]
    for df in (-1,0,1):
        for dr in (-1,0,1):
            f=f0+df; r=r0+dr
            if 0 <= f < 8 and 0 <= r < 8: out.append(chess.square(f,r))
    return out

def zone_attack_count(chess, board, color, zone):
    return sum(len(board.attackers(color, sq)) for sq in zone)

def escape_squares(chess, board, color):
    k=board.king(color)
    if k is None: return 0
    enemy=not color; n=0
    for sq in king_zone(chess,k):
        if sq == k: continue
        p=board.piece_at(sq)
        if p and p.color == color: continue
        if not board.is_attacked_by(enemy, sq): n += 1
    return n

def slider_attackers_to(chess, board, color, target):
    if target is None: return 0
    n=0
    for a in board.attackers(color, target):
        p=board.piece_at(a)
        if p and p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN): n += 1
    return n

def moved_slider_pressure(chess, board_after, move, color, target):
    if target is None: return 0
    p=board_after.piece_at(move.to_square)
    if not p or p.color != color or p.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN): return 0
    return 1 if move.to_square in board_after.attackers(color, target) else 0

def append_attack(chess, board, after, move, moved_color, feats):
    enemy=not moved_color
    moved=board.piece_at(move.from_square)
    moved_val=PIECE_VALUE.get(moved.piece_type if moved else 0,0.0)
    enemy_counts=attackers_by_type(chess, after, enemy, move.to_square)
    own_counts=attackers_by_type(chess, after, moved_color, move.to_square)
    lva_enemy=lva(chess, after, enemy, move.to_square); lvd_own=lva(chess, after, moved_color, move.to_square)
    feats += [
        float(min(8,len(board.attackers(enemy, move.from_square)))), float(min(8,len(board.attackers(moved_color, move.from_square)))),
        *[float(min(8,x)) for x in enemy_counts], *[float(min(8,x)) for x in own_counts],
        float(lva_enemy), float(lvd_own), float(moved_val - lva_enemy if lva_enemy else moved_val),
        float(lva_enemy > 0 and (lvd_own == 0 or lva_enemy < moved_val)), float(lvd_own > 0),
    ]

def append_delta(chess, board, after, move, moved_color, cap_type, promo_id, feats):
    enemy=not moved_color; moved=board.piece_at(move.from_square); moved_val=PIECE_VALUE.get(moved.piece_type if moved else 0,0.0)
    own_q_before=[sq for sq,p in board.piece_map().items() if p.color==moved_color and p.piece_type==chess.QUEEN]
    own_q_after=[sq for sq,p in after.piece_map().items() if p.color==moved_color and p.piece_type==chess.QUEEN]
    qlost=len(own_q_after) < len(own_q_before)
    q_en=any(after.is_attacked_by(enemy, q) for q in own_q_after)
    q_cap=False; moved_cap=False
    for r in after.legal_moves:
        target=after.piece_at(r.to_square)
        if target and target.color==moved_color and target.piece_type==chess.QUEEN: q_cap=True
        if r.to_square == move.to_square and target and target.color==moved_color: moved_cap=True
        if q_cap and moved_cap: break
    promo_piece={1:chess.KNIGHT,2:chess.BISHOP,3:chess.ROOK,4:chess.QUEEN}.get(promo_id,0)
    promo_gain=(PIECE_VALUE.get(promo_piece,0.0)-PIECE_VALUE.get(chess.PAWN,0.0)) if promo_id else 0.0
    cap_val=PIECE_VALUE.get(cap_type,0.0)
    mat_signed=cap_val + promo_gain - (moved_val if moved_cap and cap_val < moved_val else 0.0)
    own_legal=sum(1 for _ in board.legal_moves); replies=sum(1 for _ in after.legal_moves)
    hanging=moved_cap and not after.is_attacked_by(moved_color, move.to_square)
    feats += [float(qlost), float(q_en), float(q_cap), float(moved_cap), float(cap_val-moved_val), float(promo_gain), float(mat_signed), float(min(64,own_legal)), float(min(64,replies)), float(hanging and cap_val==0), float(cap_val>0 and not after.is_attacked_by(moved_color, move.to_square)), float(moved_cap and cap_val+promo_gain < moved_val)]

def append_ray(chess, board, after, move, moved_color, feats):
    enemy=not moved_color
    ek_pre=board.king(enemy); ok_after=after.king(moved_color); ek_after=after.king(enemy)
    oq_after=next((sq for sq,p in after.piece_map().items() if p.color==moved_color and p.piece_type==chess.QUEEN), None)
    eq_pre=next((sq for sq,p in board.piece_map().items() if p.color==enemy and p.piece_type==chess.QUEEN), None)
    eq_after=next((sq for sq,p in after.piece_map().items() if p.color==enemy and p.piece_type==chess.QUEEN), None)
    kpre=slider_attackers_to(chess, board, moved_color, ek_pre); kafter=slider_attackers_to(chess, after, moved_color, ek_after)
    qpre=slider_attackers_to(chess, board, moved_color, eq_pre); qafter=slider_attackers_to(chess, after, moved_color, eq_after)
    enemy_ok=slider_attackers_to(chess, after, enemy, ok_after); enemy_oq=slider_attackers_to(chess, after, enemy, oq_after)
    moved_k=moved_slider_pressure(chess, after, move, moved_color, ek_after); moved_q=moved_slider_pressure(chess, after, move, moved_color, eq_after)
    opens=float((kafter>kpre) or (qafter>qpre))
    feats += [float(kpre),float(kafter),float(kafter-kpre),float(qpre),float(qafter),float(qafter-qpre),float(enemy_ok),float(enemy_oq),float(moved_k),float(moved_q),opens,float(enemy_oq>0)]

def append_kingzone(chess, board, after, move, moved_color, cap_type, feats):
    enemy=not moved_color
    ok_pre=board.king(moved_color); ek_pre=board.king(enemy); ok_after=after.king(moved_color); ek_after=after.king(enemy)
    ez_pre=king_zone(chess, ek_pre); oz_pre=king_zone(chess, ok_pre); ez_after=king_zone(chess, ek_after); oz_after=king_zone(chess, ok_after)
    own_pre=zone_attack_count(chess, board, moved_color, ez_pre); own_after=zone_attack_count(chess, after, moved_color, ez_after)
    enemy_pre=zone_attack_count(chess, board, enemy, oz_pre); enemy_after=zone_attack_count(chess, after, enemy, oz_after)
    near_enemy=ek_after is not None and square_dist(chess, move.to_square, ek_after) <= 1
    from_own=ok_pre is not None and square_dist(chess, move.from_square, ok_pre) <= 1
    capture_enemy_zone=cap_type and near_enemy
    feats += [float(near_enemy), float(from_own), float(capture_enemy_zone), float(board.gives_check(move) or near_enemy), float(min(16,own_pre)), float(min(16,own_after)), float(max(-16,min(16,own_after-own_pre))), float(min(16,enemy_pre)), float(min(16,enemy_after)), float(max(-16,min(16,enemy_after-enemy_pre))), float(min(8,escape_squares(chess, after, enemy))), float(min(8,escape_squares(chess, after, moved_color)))]

def tactical_feature(chess, board, move, groups):
    feats = move_feature(chess, board, move)
    piece=board.piece_at(move.from_square); moved_color=piece.color if piece else board.turn
    cap=board.piece_at(move.to_square)
    is_ep=bool(piece and piece.piece_type == chess.PAWN and board.ep_square == move.to_square and cap is None and chess.square_file(move.from_square) != chess.square_file(move.to_square))
    if is_ep: cap=board.piece_at(chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square)))
    cap_type=cap.piece_type if cap else 0
    promo_id=PROMO_TO_ID[move.promotion and {chess.KNIGHT:'n', chess.BISHOP:'b', chess.ROOK:'r', chess.QUEEN:'q'}.get(move.promotion)]
    after=board.copy(stack=False); after.push(move)
    if any(g in groups for g in ('attack','attackmap')): append_attack(chess, board, after, move, moved_color, feats)
    if any(g in groups for g in ('delta','afterstate')): append_delta(chess, board, after, move, moved_color, cap_type, promo_id, feats)
    if any(g in groups for g in ('ray','ray_summary')): append_ray(chess, board, after, move, moved_color, feats)
    if any(g in groups for g in ('king','kingzone')): append_kingzone(chess, board, after, move, moved_color, cap_type, feats)
    return feats

def main():
    ap=argparse.ArgumentParser(description='Build MoveFormer tactical legal-move sidecar cache with optional attack/delta/ray/kingzone feature groups.')
    ap.add_argument('--input', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--feature-groups', default='base,attack', help='base,attack,delta,ray,kingzone,all')
    ap.add_argument('--max-rows', type=int, default=1000000); ap.add_argument('--max-legal-moves', type=int, default=128)
    ap.add_argument('--history-plies', type=int, default=2); ap.add_argument('--state-planes', action='store_true')
    ap.add_argument('--assume-rows', type=int, default=0); ap.add_argument('--progress-every', type=int, default=10000)
    args=ap.parse_args()
    import chess
    groups=group_names(args.feature_groups); names=feature_names(groups)
    moves=fixed_policy_moves(); policy_index={m:i for i,m in enumerate(moves)}
    if args.assume_rows>0:
        n=min(int(args.assume_rows), int(args.max_rows)); skipped_multi=skipped_unknown=malformed=0; print(f'METRIC assumed_rows={n}', flush=True)
    else:
        n, skipped_multi, skipped_unknown, malformed=count_eligible(args.input,args.max_rows,policy_index)
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    K=args.max_legal_moves; F=len(names); C=input_plane_count(args.history_plies,args.state_planes)
    y=np.memmap(out/'policy_index.int64',np.int64,'w+',shape=(n,)); yslot=np.memmap(out/'policy_legal_slot.int16',np.int16,'w+',shape=(n,))
    wdl=np.memmap(out/'wdl.float32',np.float32,'w+',shape=(n,3)); q=np.memmap(out/'q.float32',np.float32,'w+',shape=(n,))
    lpi=np.memmap(out/'legal_policy_indices.int64',np.int64,'w+',shape=(n,K)); laid=np.memmap(out/'legal_action_ids.int64',np.int64,'w+',shape=(n,K))
    lf=np.memmap(out/'legal_features.float32',np.float32,'w+',shape=(n,K,F)); mask=np.memmap(out/'legal_mask.float32',np.float32,'w+',shape=(n,K))
    written=legal_found=idx_found=trunc=legal_total=bad_fen=0
    for path in args.input:
        with opener(path) as f:
            for line in f:
                if written>=n: break
                try:
                    r=json.loads(line); pol=r.get('policy') or {}
                    if len(pol)!=1: continue
                    target_uci=next(iter(pol))
                    if target_uci not in policy_index: continue
                    board=chess.Board(r['fen']); legals=list(board.legal_moves)
                except Exception:
                    bad_fen += 1; continue
                y[written]=policy_index[target_uci]; yslot[written]=-1
                wdl[written]=np.asarray(r.get('wdl',[.25,.5,.25]), dtype=np.float32); q[written]=float(r.get('q', wdl[written,0]-wdl[written,2]))
                if len(legals)>K: trunc += 1
                legal_total += len(legals)
                for j,m in enumerate(legals[:K]):
                    uci=move_to_policy_uci(m); pi=policy_index.get(uci,-1)
                    if pi>=0: idx_found += 1
                    lpi[written,j]=pi; laid[written,j]=move_to_action_id(m)
                    lf[written,j]=np.asarray(tactical_feature(chess, board, m, groups), dtype=np.float32)
                    mask[written,j]=1.0
                    if uci==target_uci: yslot[written]=j
                if yslot[written]>=0: legal_found += 1
                written += 1
                if args.progress_every and written % args.progress_every == 0: print(f'METRIC rows_written={written}', flush=True)
        if written>=n: break
    for a in [y,yslot,wdl,q,lpi,laid,lf,mask]: a.flush()
    meta={'format':'moveformer_tactical_sidecar_cache_v1','rows':int(written),'allocated_rows':int(n),'max_legal_moves':int(K),'feature_groups':groups,'move_feature_names':names,'num_move_features':int(F),'has_board_cache':False,'has_legal_uci':False,'input_planes':None,'history_plies':int(args.history_plies),'state_planes':bool(args.state_planes),'policy_map':'uci_queen_knight_promo_v1','policy_size':len(moves),'moves':moves,'action_id_mapping':'(from * 64 + to) * 5 + promo, promo n=1,b=2,r=3,q=4','source_inputs':[str(p) for p in args.input],'skipped_multi_policy':int(skipped_multi),'skipped_unknown_policy':int(skipped_unknown),'malformed_count_estimate':int(malformed),'bad_fen_rows':int(bad_fen),'policy_target_legal_rate':float(legal_found/max(1,written)),'legal_truncation_rate':float(trunc/max(1,written)),'avg_legal_moves':float(legal_total/max(1,written))}
    (out/'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'METRIC rows_written={written}'); print(f'METRIC num_move_features={F}')
    print(f'METRIC policy_target_legal_rate={meta["policy_target_legal_rate"]:.6f}'); print(f'METRIC legal_truncation_rate={meta["legal_truncation_rate"]:.6f}'); print(f'METRIC avg_legal_moves={meta["avg_legal_moves"]:.6f}'); print(f'METRIC bad_fen_rows={bad_fen}')

if __name__=='__main__': main()
