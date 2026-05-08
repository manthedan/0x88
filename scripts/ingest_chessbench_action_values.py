#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
from teacher_overlay_lib import SCHEMA_ACTION_VALUE, eval_to_wdl_q, iter_jsonl, position_key, write_jsonl_zst, write_manifest

def as_float(x):
    if x is None or x == '': return None
    try: return float(x)
    except Exception: return None

def as_int(x):
    v = as_float(x)
    return None if v is None else int(v)

def value_to_wdl_q(v):
    if v is None: return None, None
    v = float(v)
    # Accept either [-1,1] value or [0,1] win probability.
    q = 2.0 * v - 1.0 if 0.0 <= v <= 1.0 else max(-1.0, min(1.0, v))
    win = max(0.0, q); loss = max(0.0, -q); draw = max(0.0, 1.0 - win - loss)
    return [win, draw, loss], q

def iter_move_items(row: dict):
    # Flexible JSONL adapter for processed ChessBench-like exports.
    # Supported shapes:
    #   {fen, moves:{uci:{win_prob|value|cp|mate|rank|...}}}
    #   {fen, moves:{uci:0.62}}
    #   {fen, action_values:{uci:...}}
    #   {fen, move:"e2e4", value:...}
    moves = row.get('moves') or row.get('action_values') or row.get('policy_value')
    if isinstance(moves, dict):
        for move, payload in moves.items():
            if isinstance(payload, dict):
                yield str(move), payload
            else:
                yield str(move), {'value': payload}
    elif row.get('move'):
        yield str(row['move']), row

def convert_one(row: dict, move: str, payload: dict, source_name: str, keep_raw: bool = False) -> dict | None:
    fen = row.get('fen') or row.get('FEN')
    if not fen or not move: return None
    cp = as_int(payload.get('cp', payload.get('eval_cp', payload.get('centipawn'))))
    mate = as_int(payload.get('mate'))
    value = as_float(payload.get('value', payload.get('q')))
    win_prob = as_float(payload.get('win_prob', payload.get('winrate', payload.get('p_win'))))
    if win_prob is not None and value is None:
        value = win_prob
    if cp is not None or mate is not None:
        wdl, q = eval_to_wdl_q(cp, mate)
    else:
        wdl, q = value_to_wdl_q(value)
    if wdl is None:
        return None
    out = {
        'schema': SCHEMA_ACTION_VALUE,
        'source': source_name,
        'position_key': position_key(fen),
        'fen': fen,
        'move': move,
        'teacher': str(payload.get('teacher', row.get('teacher', 'stockfish'))),
        'value': q,
        'wdl': wdl,
        'win_prob': win_prob,
        'eval_cp': cp,
        'mate': mate,
        'regret_cp': as_int(payload.get('regret_cp')),
        'rank': as_int(payload.get('rank')),
        'reasons': payload.get('reasons', ['public_action_value']),
        'quality_weight': as_float(payload.get('quality_weight')) or 1.0,
    }
    if keep_raw:
        out['raw'] = payload
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description='Convert ChessBench-like JSONL action-value records into canonical ActionValue overlay.')
    ap.add_argument('--input', required=True, help='JSONL/JSONL.zst export with fen and moves/action_values dict')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataset-id', default='chessbench_action_value_sample_v1')
    ap.add_argument('--source-name', default='chessbench')
    ap.add_argument('--max-rows', type=int, default=0, help='Max input positions to scan')
    ap.add_argument('--max-moves-per-position', type=int, default=0)
    ap.add_argument('--keep-raw', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    shard = out_dir / 'shards' / 'part_000000.jsonl.zst'
    counters = Counter(); t0 = time.time()
    def rows():
        scanned = 0
        for r in iter_jsonl(args.input):
            scanned += 1; counters['positions_scanned'] += 1
            if args.max_rows and scanned > args.max_rows: break
            emitted_here = 0
            for move, payload in iter_move_items(r):
                if args.max_moves_per_position and emitted_here >= args.max_moves_per_position: break
                c = convert_one(r, move, payload, args.source_name, args.keep_raw)
                if c is None:
                    counters['move_skipped'] += 1
                    continue
                counters['move_emitted'] += 1; emitted_here += 1
                yield c
            if emitted_here == 0: counters['position_without_moves'] += 1
    n = write_jsonl_zst(rows(), shard)
    report = {'dataset_id': args.dataset_id, 'input': args.input, 'out': str(shard), 'seconds': time.time()-t0, 'counters': dict(counters)}
    (out_dir / 'reports').mkdir(parents=True, exist_ok=True)
    (out_dir / 'reports' / 'conversion_report.json').write_text(json.dumps(report, indent=2))
    write_manifest(out_dir, {
        'dataset_id': args.dataset_id,
        'schema': SCHEMA_ACTION_VALUE,
        'source': args.source_name,
        'shards': [str(shard.relative_to(out_dir))],
        'created_at_unix': time.time(),
        'report': 'reports/conversion_report.json',
    })
    print(f'METRIC chessbench_positions_scanned={counters["positions_scanned"]}')
    print(f'METRIC chessbench_action_values={n}')
    print(f'METRIC chessbench_seconds={report["seconds"]:.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
