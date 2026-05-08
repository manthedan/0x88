#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
from teacher_overlay_lib import SCHEMA_ACTION_VALUE, eval_to_wdl_q, iter_jsonl, write_jsonl_zst, write_manifest

def main() -> int:
    ap = argparse.ArgumentParser(description='Convert Stockfish root MultiPV labels into candidate ActionValue overlay rows.')
    ap.add_argument('--input', required=True, help='root_stockfish_label.py JSONL/JSONL.zst output')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataset-id', default='stockfish_root_multipv_action_value_v1')
    ap.add_argument('--source-name', default='stockfish_root_multipv')
    ap.add_argument('--max-rows', type=int, default=0, help='max root positions to scan')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    shard = out_dir / 'shards' / 'part_000000.jsonl.zst'
    c = Counter(); t0 = time.time()
    def rows():
        for i, r in enumerate(iter_jsonl(args.input), start=1):
            if args.max_rows and i > args.max_rows: break
            c['positions_scanned'] += 1
            scores = r.get('scores_cp') or {}
            if not scores:
                c['positions_without_scores'] += 1
                continue
            best_cp = max(int(v) for v in scores.values())
            # Rank by cp descending. Duplicate moves should already be collapsed.
            for rank, (move, cp_raw) in enumerate(sorted(scores.items(), key=lambda kv: -int(kv[1])), start=1):
                cp = int(cp_raw)
                wdl, q = eval_to_wdl_q(cp, None)
                out = {
                    'schema': SCHEMA_ACTION_VALUE,
                    'source': args.source_name,
                    'position_key': r.get('position_key'),
                    'history_key': r.get('history_key'),
                    'fen': r.get('fen'),
                    'move': move,
                    'teacher': r.get('teacher', 'stockfish'),
                    'teacher_bin': r.get('teacher_bin'),
                    'depth': r.get('depth'),
                    'multipv': r.get('multipv'),
                    'value': q,
                    'wdl': wdl,
                    'win_prob': None,
                    'eval_cp': cp,
                    'mate': None,
                    'regret_cp': best_cp - cp,
                    'rank': rank,
                    'reasons': ['root_multipv'],
                    'quality_weight': 1.0,
                    'pv': (r.get('pv') or {}).get(move),
                }
                c['action_values_emitted'] += 1
                yield out
    n = write_jsonl_zst(rows(), shard)
    report = {'dataset_id': args.dataset_id, 'input': args.input, 'out': str(shard), 'seconds': time.time()-t0, 'counters': dict(c)}
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
    print(f'METRIC sf_root_av_positions={c["positions_scanned"]}')
    print(f'METRIC sf_root_av_rows={n}')
    print(f'METRIC sf_root_av_seconds={report["seconds"]:.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
