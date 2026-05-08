#!/usr/bin/env python3
"""Parallel fast MoveFormer sidecar builder for clean supervised shards.

This is intentionally optimized for walltime on the 10M supervised dataset:
- no counting pass; rows are assigned per input shard
- sparse memmap creation; no whole-array preinitialization
- optional omission of legal_uci.uint8, which the trainer does not read
- multiprocessing across independent JSONL/ZST shards

Use the original build_moveformer_sidecar_cache.py for fully generic compaction of
messy inputs with unknown skip rates.
"""
from __future__ import annotations
import argparse, json, math, multiprocessing as mp, os, subprocess, time
from pathlib import Path
from contextlib import contextmanager
import numpy as np
from train_residual_torch import fixed_policy_moves
from build_moveformer_sidecar_cache import FEATURE_NAMES, move_feature, move_to_action_id, move_to_policy_uci

@contextmanager
def opener(path):
    path = str(path)
    if path.endswith('.zst'):
        p = subprocess.Popen(['zstd', '-dc', path], stdout=subprocess.PIPE, text=True)
        try:
            assert p.stdout is not None
            yield p.stdout
        finally:
            if p.stdout is not None: p.stdout.close()
            p.wait()
    else:
        with open(path) as f: yield f

def _touch_arrays(out: Path, rows: int, K: int, F: int, write_uci: bool):
    # Create sparse files of the final shape without filling padded slots.
    arrays = [
        np.memmap(out/'policy_index.int64', np.int64, 'w+', shape=(rows,)),
        np.memmap(out/'policy_legal_slot.int16', np.int16, 'w+', shape=(rows,)),
        np.memmap(out/'wdl.float32', np.float32, 'w+', shape=(rows,3)),
        np.memmap(out/'q.float32', np.float32, 'w+', shape=(rows,)),
        np.memmap(out/'legal_policy_indices.int64', np.int64, 'w+', shape=(rows,K)),
        np.memmap(out/'legal_action_ids.int64', np.int64, 'w+', shape=(rows,K)),
        np.memmap(out/'legal_features.float32', np.float32, 'w+', shape=(rows,K,F)),
        np.memmap(out/'legal_mask.float32', np.float32, 'w+', shape=(rows,K)),
    ]
    if write_uci:
        arrays.append(np.memmap(out/'legal_uci.uint8', np.uint8, 'w+', shape=(rows,K,5)))
    for a in arrays: a.flush()
    del arrays

def _worker(job):
    import chess
    path, out_s, offset, limit, rows, K, F, write_uci, progress_every = job
    out = Path(out_s)
    moves = fixed_policy_moves(); policy_index = {m:i for i,m in enumerate(moves)}
    y = np.memmap(out/'policy_index.int64', np.int64, 'r+', shape=(rows,))
    yslot = np.memmap(out/'policy_legal_slot.int16', np.int16, 'r+', shape=(rows,))
    wdl = np.memmap(out/'wdl.float32', np.float32, 'r+', shape=(rows,3))
    q = np.memmap(out/'q.float32', np.float32, 'r+', shape=(rows,))
    lpi = np.memmap(out/'legal_policy_indices.int64', np.int64, 'r+', shape=(rows,K))
    laid = np.memmap(out/'legal_action_ids.int64', np.int64, 'r+', shape=(rows,K))
    lf = np.memmap(out/'legal_features.float32', np.float32, 'r+', shape=(rows,K,F))
    mask = np.memmap(out/'legal_mask.float32', np.float32, 'r+', shape=(rows,K))
    luci = np.memmap(out/'legal_uci.uint8', np.uint8, 'r+', shape=(rows,K,5)) if write_uci else None
    written = skipped_multi = skipped_unknown = bad_fen = trunc = legal_total = legal_found = idx_found = 0
    t0 = time.time()
    with opener(path) as f:
        for line in f:
            if written >= limit: break
            try:
                r = json.loads(line); pol = r.get('policy') or {}
                if len(pol) != 1:
                    skipped_multi += 1; continue
                target_uci = next(iter(pol))
                if target_uci not in policy_index:
                    skipped_unknown += 1; continue
                board = chess.Board(r['fen'])
                legals = list(board.legal_moves)
            except Exception:
                bad_fen += 1; continue
            row = offset + written
            y[row] = policy_index[target_uci]; yslot[row] = -1
            w = np.asarray(r.get('wdl', [.25,.5,.25]), dtype=np.float32)
            wdl[row] = w; q[row] = float(r.get('q', w[0] - w[2]))
            if len(legals) > K: trunc += 1
            legal_total += len(legals)
            for j, m in enumerate(legals[:K]):
                uci = move_to_policy_uci(m); pi = policy_index.get(uci, -1)
                if pi >= 0: idx_found += 1
                lpi[row,j] = pi; laid[row,j] = move_to_action_id(m)
                if luci is not None:
                    b = uci.encode('ascii')[:5]; luci[row,j,:len(b)] = np.frombuffer(b, dtype=np.uint8)
                lf[row,j] = np.asarray(move_feature(chess, board, m), dtype=np.float32)
                mask[row,j] = 1.0
                if uci == target_uci: yslot[row] = j
            if yslot[row] >= 0: legal_found += 1
            written += 1
            if progress_every and written % progress_every == 0:
                print(f'METRIC shard={Path(path).stem} rows_written={written} seconds={time.time()-t0:.1f}', flush=True)
    for a in [y,yslot,wdl,q,lpi,laid,lf,mask,luci]:
        if a is not None: a.flush()
    return {
        'path': str(path), 'offset': offset, 'limit': limit, 'written': written,
        'skipped_multi_policy': skipped_multi, 'skipped_unknown_policy': skipped_unknown, 'bad_fen_rows': bad_fen,
        'legal_truncation_count': trunc, 'legal_total': legal_total, 'legal_found': legal_found, 'idx_found': idx_found,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-rows', type=int, required=True)
    ap.add_argument('--rows-per-input', type=int, default=0, help='Rows assigned to each input before the final clipped shard.')
    ap.add_argument('--max-legal-moves', type=int, default=64)
    ap.add_argument('--history-plies', type=int, default=2)
    ap.add_argument('--state-planes', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--no-legal-uci', action='store_true')
    ap.add_argument('--progress-every', type=int, default=50000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = int(args.max_rows); K = int(args.max_legal_moves); F = len(FEATURE_NAMES); write_uci = not args.no_legal_uci
    _touch_arrays(out, rows, K, F, write_uci)
    rpi = int(args.rows_per_input or math.ceil(rows / len(args.input)))
    jobs = []
    off = 0
    for p in args.input:
        if off >= rows: break
        lim = min(rpi, rows - off)
        jobs.append((p, str(out), off, lim, rows, K, F, write_uci, args.progress_every))
        off += lim
    print(f'METRIC parallel_jobs={len(jobs)}')
    print(f'METRIC rows_allocated={rows}')
    print(f'METRIC max_legal_moves={K}')
    with mp.Pool(processes=args.workers) as pool:
        results = list(pool.imap_unordered(_worker, jobs))
        for r in results:
            print(f'METRIC shard_done={Path(r["path"]).stem} rows_written={r["written"]}', flush=True)
    results_sorted = sorted(results, key=lambda r: r['offset'])
    written = sum(r['written'] for r in results_sorted)
    legal_found = sum(r['legal_found'] for r in results_sorted); legal_total = sum(r['legal_total'] for r in results_sorted)
    trunc = sum(r['legal_truncation_count'] for r in results_sorted); idx_found = sum(r['idx_found'] for r in results_sorted)
    meta = {
        'format': 'moveformer_sidecar_cache_v1', 'builder': 'parallel_fast_v1', 'rows': int(written), 'allocated_rows': int(rows),
        'max_legal_moves': K, 'move_feature_names': FEATURE_NAMES, 'num_move_features': F,
        'has_board_cache': False, 'has_legal_uci': write_uci, 'input_planes': None,
        'history_plies': int(args.history_plies), 'state_planes': bool(args.state_planes),
        'policy_map': 'uci_queen_knight_promo_v1', 'policy_size': len(fixed_policy_moves()), 'moves': fixed_policy_moves(),
        'action_id_mapping': '(from * 64 + to) * 5 + promo, promo n=1,b=2,r=3,q=4',
        'source_inputs': [str(p) for p in args.input], 'shard_results': results_sorted,
        'skipped_multi_policy': int(sum(r['skipped_multi_policy'] for r in results_sorted)),
        'skipped_unknown_policy': int(sum(r['skipped_unknown_policy'] for r in results_sorted)),
        'bad_fen_rows': int(sum(r['bad_fen_rows'] for r in results_sorted)),
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

if __name__ == '__main__': main()
