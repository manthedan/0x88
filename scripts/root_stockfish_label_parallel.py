#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser(description='Run parallel Stockfish root-label workers over a registry shard.')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-rows', type=int, default=0, help='Total output rows target; 0 means all input rows')
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--multipv', type=int, default=4)
    ap.add_argument('--temperature-cp', type=float, default=100.0)
    ap.add_argument('--stockfish', default='')
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + '.parts')
    tmp.mkdir(parents=True, exist_ok=True)
    per = 0 if args.max_rows == 0 else (args.max_rows + args.workers - 1) // args.workers
    procs = []
    t0 = time.time()
    for w in range(args.workers):
        part = tmp / f'part_{w:03d}.jsonl'
        log = tmp / f'part_{w:03d}.log'
        cmd = [sys.executable, 'scripts/root_stockfish_label.py', '--input', args.input, '--out', str(part), '--depth', str(args.depth), '--multipv', str(args.multipv), '--temperature-cp', str(args.temperature_cp), '--stride', str(args.workers), '--offset', str(w)]
        if per: cmd += ['--max-rows', str(per)]
        if args.stockfish: cmd += ['--stockfish', args.stockfish]
        lf = open(log, 'wt')
        procs.append((part, log, lf, subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)))
    failed = 0
    for part, log, lf, p in procs:
        rc = p.wait(); lf.close()
        if rc != 0:
            failed += 1
            print(f'[root-label-parallel] worker failed rc={rc} log={log}', file=sys.stderr)
    if failed:
        return 1
    merged = out.with_suffix('.jsonl.tmp') if str(out).endswith('.zst') else out.with_suffix(out.suffix + '.tmp')
    rows = 0
    with merged.open('wt', encoding='utf-8') as o:
        for part, _, _, _ in procs:
            with part.open('rt', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    o.write(line); rows += 1
    if str(out).endswith('.zst'):
        subprocess.check_call(['zstd', '-q', '-f', '-T0', str(merged), '-o', str(out)])
        merged.unlink()
    else:
        merged.replace(out)
    report = {'input': args.input, 'out': args.out, 'workers': args.workers, 'rows': rows, 'depth': args.depth, 'multipv': args.multipv, 'temperature_cp': args.temperature_cp, 'seconds': time.time() - t0}
    (tmp / 'merge_report.json').write_text(json.dumps(report, indent=2))
    print(f'METRIC stockfish_parallel_rows={rows}')
    print(f'METRIC stockfish_parallel_seconds={report["seconds"]:.3f}')
    print(f'METRIC stockfish_parallel_rows_per_sec={rows/max(report["seconds"],1e-9):.3f}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
