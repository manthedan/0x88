#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from collections import Counter
from pathlib import Path
from teacher_overlay_lib import (
    SCHEMA_POSITION_EVAL, eval_to_wdl_q, iter_jsonl, position_key, pv_first_move,
    policy_from_best, quality_from_depth_nodes, write_jsonl_zst, write_manifest,
)

def as_int(x):
    if x is None or x == '': return None
    try: return int(float(x))
    except Exception: return None

def pick_eval(row: dict) -> dict | None:
    # Lichess native format commonly has {fen, evals:[{depth, knodes, pvs:[{cp|mate,line}]}]}.
    if isinstance(row.get('evals'), list) and row['evals']:
        def score(e): return (as_int(e.get('depth')) or 0, as_int(e.get('knodes')) or 0)
        return max(row['evals'], key=score)
    # HF/parquet-export-like JSONL can be flat.
    if any(k in row for k in ('cp','mate','centipawn','depth','pv','line')):
        return row
    return None

def first_pv(e: dict) -> dict:
    pvs = e.get('pvs')
    if isinstance(pvs, list) and pvs:
        # Prefer the first PV; Lichess usually orders by engine preference.
        p = pvs[0]
        if isinstance(p, dict): return p
        if isinstance(p, str): return {'line': p}
    return e

def convert(row: dict, source_name: str, keep_raw: bool = False) -> dict | None:
    fen = row.get('fen') or row.get('FEN')
    if not fen: return None
    e = pick_eval(row)
    if not e: return None
    p = first_pv(e)
    cp = as_int(p.get('cp', p.get('centipawn', p.get('score'))))
    mate = as_int(p.get('mate'))
    depth = as_int(e.get('depth', row.get('depth')))
    knodes = as_int(e.get('knodes', row.get('knodes')))
    nodes = as_int(e.get('nodes', row.get('nodes')))
    pv = p.get('line', p.get('pv'))
    if isinstance(pv, str):
        pv_list = pv.strip().split()
    elif isinstance(pv, list):
        pv_list = [str(x) for x in pv]
    else:
        pv_list = []
    best = pv_first_move(pv_list)
    wdl, q = eval_to_wdl_q(cp, mate)
    if wdl is None:
        return None
    out = {
        'schema': SCHEMA_POSITION_EVAL,
        'source': source_name,
        'position_key': position_key(fen),
        'fen': fen,
        'teacher': 'stockfish',
        'teacher_version': str(row.get('engine', row.get('teacher_version', 'lichess-public-eval'))),
        'depth': depth,
        'nodes': nodes,
        'knodes': knodes,
        'best': best,
        'q': q,
        'wdl': wdl,
        'eval_cp': cp,
        'mate': mate,
        'pv': pv_list,
        'policy': policy_from_best(best),
        'quality_weight': quality_from_depth_nodes(depth, nodes, knodes),
    }
    if keep_raw:
        out['raw'] = row
    return out

def main() -> int:
    ap = argparse.ArgumentParser(description='Convert Lichess public position eval JSONL(.zst) into teacher PositionEval overlay.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataset-id', default='lichess_position_eval_sample_v1')
    ap.add_argument('--source-name', default='lichess_position_evaluations')
    ap.add_argument('--max-rows', type=int, default=0)
    ap.add_argument('--keep-raw', action='store_true')
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    shard = out_dir / 'shards' / 'part_000000.jsonl.zst'
    counters = Counter(); t0 = time.time()
    def rows():
        scanned = 0
        for r in iter_jsonl(args.input):
            scanned += 1; counters['scanned'] += 1
            if args.max_rows and scanned > args.max_rows: break
            c = convert(r, args.source_name, args.keep_raw)
            if c is None:
                counters['skipped'] += 1
                continue
            counters['emitted'] += 1
            if c.get('mate') is not None: counters['mate'] += 1
            if c.get('best'): counters['with_best'] += 1
            yield c
    n = write_jsonl_zst(rows(), shard)
    report = {'dataset_id': args.dataset_id, 'input': args.input, 'out': str(shard), 'seconds': time.time()-t0, 'counters': dict(counters)}
    (out_dir / 'reports').mkdir(parents=True, exist_ok=True)
    (out_dir / 'reports' / 'conversion_report.json').write_text(json.dumps(report, indent=2))
    write_manifest(out_dir, {
        'dataset_id': args.dataset_id,
        'schema': SCHEMA_POSITION_EVAL,
        'source': args.source_name,
        'shards': [str(shard.relative_to(out_dir))],
        'created_at_unix': time.time(),
        'report': 'reports/conversion_report.json',
    })
    print(f'METRIC lichess_eval_scanned={counters["scanned"]}')
    print(f'METRIC lichess_eval_emitted={n}')
    print(f'METRIC lichess_eval_seconds={report["seconds"]:.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
