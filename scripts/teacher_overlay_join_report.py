#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time
from collections import Counter
from teacher_overlay_lib import iter_jsonl

def main() -> int:
    ap = argparse.ArgumentParser(description='Report exact position_key overlap between a registry shard and teacher overlay shard.')
    ap.add_argument('--registry', required=True)
    ap.add_argument('--overlay', required=True)
    ap.add_argument('--max-registry-rows', type=int, default=0)
    ap.add_argument('--max-overlay-rows', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    t0 = time.time(); keys = set(); c = Counter()
    for i, r in enumerate(iter_jsonl(args.registry), start=1):
        if args.max_registry_rows and i > args.max_registry_rows: break
        k = r.get('position_key')
        if k: keys.add(k); c['registry_rows'] += 1
    seen_overlay = set()
    for i, r in enumerate(iter_jsonl(args.overlay), start=1):
        if args.max_overlay_rows and i > args.max_overlay_rows: break
        c['overlay_rows'] += 1
        k = r.get('position_key')
        if k in keys:
            c['overlay_rows_joined'] += 1
            seen_overlay.add(k)
    report = {
        'registry': args.registry,
        'overlay': args.overlay,
        'registry_rows': c['registry_rows'],
        'registry_unique_keys': len(keys),
        'overlay_rows': c['overlay_rows'],
        'overlay_rows_joined': c['overlay_rows_joined'],
        'overlay_row_join_rate': c['overlay_rows_joined'] / max(1, c['overlay_rows']),
        'registry_keys_covered': len(seen_overlay),
        'registry_key_coverage': len(seen_overlay) / max(1, len(keys)),
        'seconds': time.time() - t0,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, 'w').write(text)
    print(text)
    print(f'METRIC join_overlay_rows={c["overlay_rows"]}')
    print(f'METRIC join_overlay_rows_joined={c["overlay_rows_joined"]}')
    print(f'METRIC join_registry_key_coverage={report["registry_key_coverage"]:.8f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
