#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, hashlib, json, re, subprocess, sys, time
from collections import Counter, defaultdict, deque
from pathlib import Path


def gid(row: dict) -> str:
    rid = str(row.get('id', ''))
    m = re.match(r'(.+)_([0-9]+)$', rid)
    return m.group(1) if m else rid


def ply(row: dict) -> int:
    if 'ply' in row:
        try: return int(row['ply'])
        except Exception: pass
    rid = str(row.get('id', ''))
    m = re.search(r'_([0-9]+)$', rid)
    return int(m.group(1)) if m else 0


@contextlib.contextmanager
def open_text(path: Path, mode: str):
    if str(path).endswith('.zst'):
        if 'r' in mode:
            p = subprocess.Popen(['zstd', '-dc', str(path)], stdout=subprocess.PIPE, text=True)
            try:
                yield p.stdout
            finally:
                if p.stdout: p.stdout.close()
                rc = p.wait()
                if rc: raise subprocess.CalledProcessError(rc, ['zstd', '-dc', str(path)])
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = open(path, 'wb')
            p = subprocess.Popen(['zstd', '-q', '-T0', '-c'], stdin=subprocess.PIPE, stdout=raw, text=True)
            try:
                yield p.stdin
            finally:
                if p.stdin: p.stdin.close()
                rc = p.wait(); raw.close()
                if rc: raise subprocess.CalledProcessError(rc, ['zstd', '-q', '-T0', '-c'])
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode, encoding='utf-8') as f:
            yield f


def source_name(path: Path) -> str:
    n = path.name
    for s in ('.jsonl.zst', '.jsonl'):
        if n.endswith(s): return n[:-len(s)]
    return path.stem


def stable_dev(g: str, seed: int, pct: int) -> bool:
    h = hashlib.blake2b(f'{seed}:{g}'.encode(), digest_size=8).digest()
    return int.from_bytes(h, 'little') % 100 < pct


def main():
    ap = argparse.ArgumentParser(description='Streaming sharded supervised dataset builder for very large JSONL inputs.')
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--name', default='supervised_streaming')
    ap.add_argument('--max-rows', type=int, default=100_000_000)
    ap.add_argument('--dev-rows', type=int, default=1_000_000)
    ap.add_argument('--rows-per-shard', type=int, default=1_000_000)
    ap.add_argument('--history-plies', type=int, default=2)
    ap.add_argument('--skip-plies', type=int, default=10)
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--dev-percent', type=int, default=2)
    ap.add_argument('--zst', action='store_true')
    args = ap.parse_args()
    out = Path(args.out_dir); ext = '.jsonl.zst' if args.zst else '.jsonl'
    train_dir = out / 'train'; dev_dir = out / 'dev'; rep_dir = out / 'reports'
    train_dir.mkdir(parents=True, exist_ok=True); dev_dir.mkdir(parents=True, exist_ok=True); rep_dir.mkdir(parents=True, exist_ok=True)

    train_rows = dev_rows = bad = skipped_policy = rows_with_history = 0
    source_counts = Counter(); train_shards = []
    last_gid = None; hist = deque(maxlen=args.history_plies)
    shard_idx = -1; shard_rows = 0; shard_f = None
    dev_rel = f'dev/dev_{args.dev_rows}{ext}'
    start = time.time()

    def open_next_shard():
        nonlocal shard_idx, shard_rows, shard_f
        if shard_f is not None:
            shard_f.__exit__(None, None, None)
        shard_idx += 1; shard_rows = 0
        rel = f'train/shard_{shard_idx:04d}{ext}'
        train_shards.append(rel)
        shard_f = open_text(out / rel, 'wt')
        return shard_f.__enter__()

    tw = open_next_shard()
    with open_text(out / dev_rel, 'wt') as dw:
        for ip in map(Path, args.input):
            src = source_name(ip)
            with open_text(ip, 'rt') as f:
                for line in f:
                    if train_rows >= args.max_rows and dev_rows >= args.dev_rows: break
                    if not line.strip(): continue
                    try: r = json.loads(line)
                    except Exception: bad += 1; continue
                    if len(r.get('policy', {})) != 1:
                        skipped_policy += 1; continue
                    g = gid(r)
                    if g != last_gid:
                        hist.clear(); last_gid = g
                    nr = dict(r)
                    if args.history_plies and hist:
                        nr['history_fens'] = list(reversed(hist))
                        rows_with_history += 1
                    hist.append(r.get('fen', ''))
                    line_out = json.dumps(nr, separators=(',', ':')) + '\n'
                    if dev_rows < args.dev_rows and stable_dev(g, args.seed, args.dev_percent):
                        dw.write(line_out); dev_rows += 1
                    elif train_rows < args.max_rows:
                        if shard_rows >= args.rows_per_shard:
                            tw = open_next_shard()
                        tw.write(line_out); train_rows += 1; shard_rows += 1; source_counts[src] += 1
                    if (train_rows + dev_rows) % 1_000_000 == 0:
                        print(f'METRIC streaming_dataset_rows_seen={train_rows + dev_rows}', flush=True)
            if train_rows >= args.max_rows and dev_rows >= args.dev_rows: break
    if shard_f is not None:
        shard_f.__exit__(None, None, None)
    if shard_rows == 0 and train_shards:
        # leave empty final shard only if created exactly at boundary? remove it.
        empty = out / train_shards[-1]
        try: empty.unlink()
        except FileNotFoundError: pass
        train_shards.pop()

    manifest = {'name': args.name, 'format': 'jsonl.zst' if args.zst else 'jsonl', 'train_shards': train_shards, 'dev': dev_rel, 'rows_per_shard': args.rows_per_shard, 'total_train_rows': train_rows, 'total_dev_rows': dev_rows, 'history_plies': args.history_plies, 'skip_plies': args.skip_plies, 'caps': {'streaming': True, 'dev_percent': args.dev_percent}, 'reproducibility': {'inputs': args.input, 'seed': args.seed, 'script': 'scripts/build_supervised_dataset_streaming.py', 'argv': sys.argv}, 'report': 'reports/dataset_report.json'}
    report = {'name': args.name, 'train': {'rows': train_rows, 'source_counts': dict(source_counts)}, 'dev': {'rows': dev_rows}, 'bad_json_lines': bad, 'skipped_non_single_policy': skipped_policy, 'rows_with_history': rows_with_history, 'wall_time_seconds': time.time() - start}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (rep_dir / 'dataset_report.json').write_text(json.dumps(report, indent=2))
    print(f'METRIC dataset_train_rows={train_rows}')
    print(f'METRIC dataset_dev_rows={dev_rows}')
    print(f'METRIC dataset_train_shards={len(train_shards)}')

if __name__ == '__main__': main()
