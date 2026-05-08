#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from collections import Counter
from teacher_overlay_lib import iter_jsonl, position_key

def valid_wdl(w):
    return isinstance(w, list) and len(w) == 3 and all(isinstance(x, (int,float)) and 0 <= x <= 1 for x in w) and abs(sum(w)-1.0) < 1e-3

def main() -> int:
    ap = argparse.ArgumentParser(description='Validate canonical teacher overlay JSONL(.zst).')
    ap.add_argument('input')
    ap.add_argument('--max-rows', type=int, default=0)
    args = ap.parse_args()
    c = Counter(); examples = []
    for i, r in enumerate(iter_jsonl(args.input), start=1):
        if args.max_rows and i > args.max_rows: break
        c['rows'] += 1
        schema = r.get('schema'); c['schema:'+str(schema)] += 1
        fen = r.get('fen')
        if not fen: c['missing_fen'] += 1
        elif r.get('position_key') != position_key(fen): c['bad_position_key'] += 1
        if schema == 'teacher.position_eval.v1':
            if not valid_wdl(r.get('wdl')): c['bad_wdl'] += 1
            if r.get('q') is None or not -1.0001 <= float(r.get('q')) <= 1.0001: c['bad_q'] += 1
            pol = r.get('policy')
            if pol is not None and abs(sum(float(x) for x in pol.values()) - 1.0) > 1e-3: c['bad_policy_sum'] += 1
        elif schema == 'teacher.action_value.v1':
            if not r.get('move'): c['missing_move'] += 1
            if not valid_wdl(r.get('wdl')): c['bad_wdl'] += 1
            if r.get('value') is None or not -1.0001 <= float(r.get('value')) <= 1.0001: c['bad_value'] += 1
        else:
            c['unknown_schema'] += 1
        if len(examples) < 3: examples.append(r)
    print(json.dumps({'counters': dict(c), 'examples': examples}, indent=2)[:8000])
    fatal = sum(c[k] for k in c if k.startswith('bad_') or k.startswith('missing_') or k == 'unknown_schema')
    print(f'METRIC teacher_overlay_rows={c["rows"]}')
    print(f'METRIC teacher_overlay_errors={fatal}')
    return 1 if fatal else 0

if __name__ == '__main__':
    raise SystemExit(main())
