#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from collections import Counter
from pathlib import Path
import msgpack
from teacher_overlay_lib import SCHEMA_ACTION_VALUE, position_key, write_jsonl_zst, write_manifest

def q_wdl_from_win_prob(p: float):
    q = max(-1.0, min(1.0, 2.0 * float(p) - 1.0))
    w = max(0.0, q); l = max(0.0, -q); d = max(0.0, 1.0 - w - l)
    return q, [w, d, l]

def mate_to_q(mate):
    if mate is None: return None
    if mate == '#': return 1.0
    try:
        m = int(mate)
        return 1.0 if m > 0 else -1.0
    except Exception:
        return None

def iter_msgpack_zst(path: str | Path):
    p = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
    try:
        assert p.stdout is not None
        unpacker = msgpack.Unpacker(p.stdout, raw=False)
        for rec in unpacker:
            yield rec
    finally:
        try:
            if p.stdout: p.stdout.close()
        finally:
            p.kill()
            p.wait()

def iter_input_records(paths):
    for path in paths:
        if str(path).endswith('.zst'):
            yield from iter_msgpack_zst(path)
        else:
            with open(path, 'rb') as f:
                unpacker = msgpack.Unpacker(f, raw=False)
                yield from unpacker

def main() -> int:
    ap = argparse.ArgumentParser(description='Convert ChessBench-full-policy-value msgpack(.zst) shards to canonical ActionValue overlay.')
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataset-id', default='chessbench_full_policy_value_sample_v1')
    ap.add_argument('--source-name', default='chessbench_full_policy_value')
    ap.add_argument('--max-positions', type=int, default=0)
    ap.add_argument('--max-moves-per-position', type=int, default=0, help='0 keeps all legal moves')
    ap.add_argument('--top-k', type=int, default=0, help='If set, keep top-k by win_prob before max-moves cap')
    ap.add_argument('--keep-raw', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out_dir); shard = out_dir / 'shards' / 'part_000000.jsonl.zst'
    c = Counter(); t0 = time.time()
    def rows():
        for rec in iter_input_records(args.input):
            if args.max_positions and c['positions_scanned'] >= args.max_positions: break
            c['positions_scanned'] += 1
            fen = rec.get('fen') if isinstance(rec, dict) else None
            moves = rec.get('moves') if isinstance(rec, dict) else None
            if not fen or not isinstance(moves, dict):
                c['bad_record'] += 1; continue
            items = list(moves.items())
            def wp(item):
                payload = item[1] if isinstance(item[1], dict) else {'win_prob': item[1]}
                try: return float(payload.get('win_prob', 0.5))
                except Exception: return 0.5
            items.sort(key=wp, reverse=True)
            if args.top_k: items = items[:args.top_k]
            if args.max_moves_per_position: items = items[:args.max_moves_per_position]
            best_wp = wp(items[0]) if items else 0.5
            best_cp_proxy = int(round((2.0 * best_wp - 1.0) * 400.0))
            pkey = position_key(fen)
            for rank, (move, payload0) in enumerate(items, start=1):
                payload = payload0 if isinstance(payload0, dict) else {'win_prob': payload0}
                try: win_prob = float(payload.get('win_prob'))
                except Exception:
                    c['bad_win_prob'] += 1; continue
                mate = payload.get('mate')
                q_mate = mate_to_q(mate)
                if q_mate is not None:
                    q = q_mate
                    wdl = [1.0,0.0,0.0] if q > 0 else [0.0,0.0,1.0]
                else:
                    q, wdl = q_wdl_from_win_prob(win_prob)
                cp_proxy = int(round(q * 400.0))
                out = {
                    'schema': SCHEMA_ACTION_VALUE,
                    'source': args.source_name,
                    'position_key': pkey,
                    'fen': fen,
                    'move': str(move),
                    'teacher': 'stockfish16_chessbench',
                    'value': q,
                    'wdl': wdl,
                    'win_prob': win_prob,
                    'eval_cp': None,
                    'mate': mate,
                    'regret_cp': best_cp_proxy - cp_proxy,
                    'rank': rank,
                    'reasons': ['chessbench_legal_move'],
                    'quality_weight': 1.0,
                }
                if args.keep_raw: out['raw'] = payload
                c['action_values_emitted'] += 1
                yield out
    n = write_jsonl_zst(rows(), shard)
    report = {'dataset_id': args.dataset_id, 'input': args.input, 'out': str(shard), 'seconds': time.time()-t0, 'counters': dict(c), 'top_k': args.top_k, 'max_moves_per_position': args.max_moves_per_position}
    (out_dir/'reports').mkdir(parents=True, exist_ok=True)
    (out_dir/'reports'/'conversion_report.json').write_text(json.dumps(report, indent=2))
    write_manifest(out_dir, {'dataset_id': args.dataset_id, 'schema': SCHEMA_ACTION_VALUE, 'source': args.source_name, 'shards': [str(shard.relative_to(out_dir))], 'created_at_unix': time.time(), 'report': 'reports/conversion_report.json'})
    print(f'METRIC chessbench_msgpack_positions={c["positions_scanned"]}')
    print(f'METRIC chessbench_msgpack_action_values={n}')
    print(f'METRIC chessbench_msgpack_seconds={report["seconds"]:.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
