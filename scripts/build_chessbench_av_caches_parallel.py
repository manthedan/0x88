#!/usr/bin/env python3
from __future__ import annotations
import argparse, concurrent.futures, glob, json, os, subprocess, sys, time
from pathlib import Path

def shard_name(path: str) -> str:
    return Path(path).name.replace('.msgpack.zst', '').replace('.msgpack', '')

def run_one(py: str, inp: str, out: Path, top_k: int, max_candidates: int, history: int, max_positions: int, force: bool) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'build.log'
    meta = out / 'meta.json'
    if meta.exists() and not force:
        return {'input': inp, 'out': str(out), 'status': 'exists'}
    cmd = [py, 'training/build_chessbench_av_cache.py', '--input', inp, '--out', str(out), '--top-k', str(top_k), '--max-candidates', str(max_candidates), '--history-plies', str(history)]
    if max_positions:
        cmd += ['--max-positions', str(max_positions)]
    t0 = time.time()
    with log.open('wt', encoding='utf-8') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    status = 'ok' if p.returncode == 0 else 'failed'
    rows = cand = 0
    if meta.exists():
        try:
            m = json.loads(meta.read_text()); rows = int(m.get('rows', 0)); cand = int(m.get('candidate_rows', 0))
        except Exception:
            pass
    return {'input': inp, 'out': str(out), 'status': status, 'returncode': p.returncode, 'seconds': time.time()-t0, 'rows': rows, 'candidate_rows': cand, 'log': str(log)}

def main() -> int:
    ap = argparse.ArgumentParser(description='Build compact ChessBench AV caches independently/parallel per raw shard.')
    ap.add_argument('--input-glob', default='data/public_teacher_raw/chessbench_full_policy_value/train-*.msgpack.zst')
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 4)//2)))
    ap.add_argument('--max-shards', type=int, default=0)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--max-candidates', type=int, default=8)
    ap.add_argument('--history-plies', type=int, default=2)
    ap.add_argument('--max-positions-per-shard', type=int, default=0)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    paths = sorted(glob.glob(args.input_glob))
    if args.max_shards:
        paths = paths[:args.max_shards]
    if not paths:
        raise SystemExit(f'no inputs matched {args.input_glob}')
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = []
        for p in paths:
            out = out_root / f'{shard_name(p)}_top{args.top_k}_compact_v1'
            futs.append(ex.submit(run_one, sys.executable, p, out, args.top_k, args.max_candidates, args.history_plies, args.max_positions_per_shard, args.force))
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result(); results.append(r)
            print(json.dumps(r, sort_keys=True), flush=True)
    results.sort(key=lambda r: r['out'])
    ok = [r for r in results if r['status'] in ('ok', 'exists')]
    manifest = {
        'format': 'chessbench_av_cache_collection_v1',
        'created_at_unix': time.time(),
        'input_glob': args.input_glob,
        'workers': args.workers,
        'top_k': args.top_k,
        'max_candidates': args.max_candidates,
        'history_plies': args.history_plies,
        'seconds': time.time() - t0,
        'caches': [r['out'] for r in ok],
        'results': results,
    }
    (out_root / 'collection_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'METRIC chessbench_cache_shards={len(ok)}')
    print(f'METRIC chessbench_cache_positions={sum(int(r.get("rows",0)) for r in ok)}')
    print(f'METRIC chessbench_cache_candidate_rows={sum(int(r.get("candidate_rows",0)) for r in ok)}')
    print(f'METRIC chessbench_cache_seconds={manifest["seconds"]:.3f}')
    failed = [r for r in results if r['status'] == 'failed']
    return 1 if failed else 0

if __name__ == '__main__':
    raise SystemExit(main())
