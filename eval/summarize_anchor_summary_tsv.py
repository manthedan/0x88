#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def bucket(anchor: str) -> str:
    if anchor.startswith('stockfish_lite_') and anchor.split('_')[-1].isdigit():
        return 'stockfish_lite_elo'
    if anchor.startswith('stockfish_depth'):
        return 'stockfish_depth'
    if anchor.startswith('stockfish_lite_depth'):
        return 'stockfish_lite_depth'
    return 'other'


def short_name(file: str) -> str:
    n = Path(file).name
    return n.replace('_stockfish_lite_shallow_v32_pairs8.json', '')


def main() -> int:
    ap = argparse.ArgumentParser(description='Summarize uci_anchor_arena summary.tsv by candidate and anchor bucket.')
    ap.add_argument('summary_tsv')
    ap.add_argument('--md-out', default='')
    ap.add_argument('--json-out', default='')
    args = ap.parse_args()

    rows = []
    with open(args.summary_tsv, newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            r['games'] = int(r['games']); r['wins'] = int(r['wins']); r['draws'] = int(r['draws']); r['losses'] = int(r['losses']); r['illegal'] = int(r['illegal'])
            r['scoreRate'] = float(r['scoreRate'])
            r['points'] = r['wins'] + 0.5*r['draws']
            r['label'] = short_name(r['file'])
            r['bucket'] = bucket(r['anchor'])
            rows.append(r)

    agg = defaultdict(lambda: {'games':0,'wins':0,'draws':0,'losses':0,'points':0.0,'illegal':0})
    for r in rows:
        for key in [(r['label'], 'all'), (r['label'], r['bucket'])]:
            a = agg[key]
            for k in ('games','wins','draws','losses','illegal'):
                a[k] += r[k]
            a['points'] += r['points']
    out_rows = []
    for (label, b), a in sorted(agg.items()):
        a = dict(a)
        a['label'] = label; a['bucket'] = b; a['scoreRate'] = a['points']/a['games'] if a['games'] else 0.0
        out_rows.append(a)
    rank_all = sorted([r for r in out_rows if r['bucket']=='all'], key=lambda r: (r['scoreRate'], r['points']), reverse=True)
    rank_lite_elo = sorted([r for r in out_rows if r['bucket']=='stockfish_lite_elo'], key=lambda r: (r['scoreRate'], r['points']), reverse=True)

    result = {'rows': rows, 'aggregates': out_rows, 'rank_all': rank_all, 'rank_stockfish_lite_elo': rank_lite_elo}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2)+'\n')

    lines = []
    lines.append('# Lite/shallow Stockfish anchor summary')
    lines.append('')
    lines.append(f'Source: `{args.summary_tsv}`')
    lines.append('')
    lines.append('## Overall ranking')
    lines.append('')
    lines.append('| rank | label | score | W-D-L | illegal |')
    lines.append('|---:|---|---:|---:|---:|')
    for i,r in enumerate(rank_all,1):
        lines.append(f"| {i} | `{r['label']}` | {r['points']:.1f}/{r['games']} ({r['scoreRate']:.1%}) | {r['wins']}-{r['draws']}-{r['losses']} | {r['illegal']} |")
    lines.append('')
    lines.append('## Stockfish-lite Elo anchors only')
    lines.append('')
    lines.append('| rank | label | score | W-D-L | illegal |')
    lines.append('|---:|---|---:|---:|---:|')
    for i,r in enumerate(rank_lite_elo,1):
        lines.append(f"| {i} | `{r['label']}` | {r['points']:.1f}/{r['games']} ({r['scoreRate']:.1%}) | {r['wins']}-{r['draws']}-{r['losses']} | {r['illegal']} |")
    lines.append('')
    lines.append('## Bucket view')
    lines.append('')
    lines.append('| label | bucket | score | W-D-L | illegal |')
    lines.append('|---|---|---:|---:|---:|')
    for r in sorted(out_rows, key=lambda x: (x['label'], x['bucket'])):
        lines.append(f"| `{r['label']}` | {r['bucket']} | {r['points']:.1f}/{r['games']} ({r['scoreRate']:.1%}) | {r['wins']}-{r['draws']}-{r['losses']} | {r['illegal']} |")
    md='\n'.join(lines)+'\n'
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md_out).write_text(md)
    print(md)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
