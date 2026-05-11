#!/usr/bin/env python3
"""Summarize Gumbel-Zero self-play chunks without trusting them for training.

Accepts JSONL/JSONL.ZST produced by scripts/gumbel_zero_selfplay.mjs and prints
schema/metric diagnostics for review before any long self-play launch.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


@contextmanager
def _zstd_cli_reader(path: Path):
    if not shutil.which('zstd'):
        raise SystemExit(f'zstandard python module or zstd CLI is required to read {path}')
    proc = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    with proc.stdout, open(proc.stdout.fileno(), mode='r', encoding='utf-8', closefd=False) as fh:
        yield fh
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f'zstd failed while reading {path} rc={rc}')


def open_text(path: Path):
    if str(path).endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8')
    if str(path).endswith('.zst'):
        try:
            import zstandard as zstd  # type: ignore
            fh = path.open('rb')
            return zstd.open(fh, mode='rt', encoding='utf-8')  # type: ignore[return-value]
        except ImportError:
            return _zstd_cli_reader(path)
        except Exception as exc:
            raise SystemExit(f'zstandard failed to read {path}: {exc}')
    return path.open('r', encoding='utf-8')


def mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def pct(n: int, d: int) -> float:
    return n / max(1, d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('chunks', nargs='+')
    ap.add_argument('--json', action='store_true')
    args = ap.parse_args()

    rows = 0
    games: set[str] = set()
    schemas = Counter()
    terminal_reasons = Counter()
    visits = Counter()
    candidate_counts = Counter()
    selected_outside_top1 = 0
    entropies: list[float] = []
    q_spreads: list[float] = []
    regrets: list[float] = []
    legal_counts: list[int] = []
    policy_mass_bad = 0
    candidate_visit_hist = Counter()
    candidates_per_row: list[int] = []
    rows_by_file: dict[str, int] = {}

    for chunk in map(Path, args.chunks):
        before = rows
        with open_text(chunk) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows += 1
                if isinstance(row.get('game_id'), str):
                    games.add(row['game_id'])
                schemas[str(row.get('schema', '<missing>'))] += 1
                terminal_reasons[str(row.get('terminal_reason', '<missing>'))] += 1
                visits[int(row.get('requested_visits', row.get('visits', 0)) or 0)] += 1
                candidate_counts[int(row.get('candidate_count', 0) or 0)] += 1
                if row.get('search_overturned_policy_top1'):
                    selected_outside_top1 += 1
                if 'policy_entropy' in row:
                    entropies.append(float(row['policy_entropy']))
                if 'candidate_q_spread' in row:
                    q_spreads.append(float(row['candidate_q_spread']))
                if 'selected_regret' in row:
                    regrets.append(float(row['selected_regret']))
                if 'legal_count' in row:
                    legal_counts.append(int(row['legal_count']))
                policy = row.get('policy') or {}
                mass = sum(float(v) for v in policy.values()) if isinstance(policy, dict) else 0.0
                if not math.isfinite(mass) or abs(mass - 1.0) > 1e-4:
                    policy_mass_bad += 1
                cands = row.get('candidates') or []
                candidates_per_row.append(len(cands))
                for c in cands:
                    candidate_visit_hist[int(c.get('visits', 0) or 0)] += 1
        rows_by_file[str(chunk)] = rows - before

    report = {
        'ok': rows > 0 and policy_mass_bad == 0,
        'rows': rows,
        'games': len(games),
        'rows_by_file': rows_by_file,
        'schemas': dict(schemas),
        'terminal_reasons': dict(terminal_reasons),
        'requested_visits': dict(visits),
        'candidate_counts': dict(candidate_counts),
        'search_overturned_policy_top1_rate': pct(selected_outside_top1, rows),
        'policy_entropy_mean': mean(entropies),
        'candidate_q_spread_mean': mean(q_spreads),
        'selected_regret_mean': mean(regrets),
        'legal_count_mean': mean([float(x) for x in legal_counts]),
        'candidates_per_row_mean': mean([float(x) for x in candidates_per_row]),
        'candidate_visit_hist': dict(candidate_visit_hist),
        'policy_mass_bad_rows': policy_mass_bad,
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"rows={rows} games={len(games)} ok={1 if report['ok'] else 0}")
        print(f"schemas={dict(schemas)}")
        print(f"terminal_reasons={dict(terminal_reasons)}")
        print(f"requested_visits={dict(visits)} candidate_counts={dict(candidate_counts)}")
        print(f"search_overturned_policy_top1_rate={report['search_overturned_policy_top1_rate']:.6f}")
        print(f"policy_entropy_mean={report['policy_entropy_mean']:.6f}")
        print(f"candidate_q_spread_mean={report['candidate_q_spread_mean']:.6f}")
        print(f"selected_regret_mean={report['selected_regret_mean']:.6f}")
        print(f"legal_count_mean={report['legal_count_mean']:.6f} candidates_per_row_mean={report['candidates_per_row_mean']:.6f}")
        print(f"candidate_visit_hist={dict(candidate_visit_hist)}")
        print(f"policy_mass_bad_rows={policy_mass_bad}")
        print(f"METRIC gumbel_zero_report_ok={1 if report['ok'] else 0}")
        print(f"METRIC gumbel_zero_report_rows={rows}")
    return 0 if report['ok'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
