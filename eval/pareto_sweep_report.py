#!/usr/bin/env python3
"""CARBS/Protein-inspired Pareto report for Tiny Leela sweep ledgers.

Input: JSONL rows with at least {trial_id, params, score, cost}. Cost may be a
number or an object containing wall_seconds/gpu_hours/positions/etc.
Higher score is better; lower cost is better.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def cost_scalar(cost) -> float:
    if isinstance(cost, (int, float)):
        return float(cost)
    if not isinstance(cost, dict):
        return math.inf
    if 'objective' in cost:
        return float(cost['objective'])
    gpu = float(cost.get('gpu_hours', 0.0))
    wall = float(cost.get('wall_seconds', 0.0)) / 3600.0
    cpu = float(cost.get('cpu_hours', 0.0)) * 0.05
    positions = float(cost.get('positions', 0.0)) / 10_000_000.0 * 0.01
    return gpu + wall + cpu + positions


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        row = json.loads(line)
        row['_line'] = line_no
        row['_score'] = float(row.get('score', row.get('objective', float('-inf'))))
        row['_cost'] = cost_scalar(row.get('cost', row.get('cost_objective', math.inf)))
        row['_status'] = row.get('status', 'succeeded')
        rows.append(row)
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    good = [r for r in rows if r['_status'] == 'succeeded' and math.isfinite(r['_score']) and math.isfinite(r['_cost'])]
    front = []
    for r in good:
        dominated = False
        for q in good:
            if q is r:
                continue
            if q['_score'] >= r['_score'] and q['_cost'] <= r['_cost'] and (q['_score'] > r['_score'] or q['_cost'] < r['_cost']):
                dominated = True
                break
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: (r['_cost'], -r['_score']))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('ledger')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()
    rows = load_rows(Path(args.ledger))
    front = pareto_front(rows)
    best = sorted([r for r in rows if r['_status'] == 'succeeded'], key=lambda r: (-r['_score'], r['_cost']))[:args.top]
    if args.json:
        print(json.dumps({'trials': len(rows), 'pareto': front, 'best': best}, indent=2, default=str))
    else:
        print(f'METRIC sweep_trials={len(rows)}')
        print(f'METRIC sweep_pareto={len(front)}')
        print('pareto_rank\ttrial_id\tscore\tcost\tparams')
        for i, r in enumerate(front[:args.top], 1):
            print(f"{i}\t{r.get('trial_id', r['_line'])}\t{r['_score']:.6f}\t{r['_cost']:.6f}\t{json.dumps(r.get('params', {}), sort_keys=True)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
