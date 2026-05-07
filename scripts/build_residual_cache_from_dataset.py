#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def run(cmd):
    print('+ ' + ' '.join(map(str, cmd)), flush=True)
    subprocess.check_call(list(map(str, cmd)))


def shard_name(rel: str) -> str:
    name = Path(rel).name
    for suf in ('.jsonl.zst', '.jsonl'):
        if name.endswith(suf):
            return name[:-len(suf)]
    return Path(rel).stem


def load_cache_meta(path: str | Path) -> dict:
    return json.loads((Path(path) / 'meta.json').read_text())


def validate_cache_metas(cache_paths: list[str], dev_dir: Path, require_side_info: bool) -> dict:
    metas = [load_cache_meta(p) for p in cache_paths]
    dev_meta = load_cache_meta(dev_dir)
    all_metas = metas + [dev_meta]
    if not all_metas:
        raise SystemExit('no cache shards built')
    keys = ['input_planes', 'policy_size', 'history_plies', 'state_planes']
    ref = {k: all_metas[0].get(k) for k in keys}
    for i, m in enumerate(all_metas):
        for k, v in ref.items():
            if m.get(k) != v:
                raise SystemExit(f'cache meta mismatch at index {i}: {k}={m.get(k)!r} expected {v!r}')
        if bool(m.get('has_side_info', False)) != bool(require_side_info):
            raise SystemExit(f'cache side-info mismatch at index {i}: has_side_info={m.get("has_side_info")!r} expected {require_side_info!r}')
        if m.get('moves') != all_metas[0].get('moves'):
            raise SystemExit(f'cache policy moves mismatch at index {i}')
    return {'input_planes': ref['input_planes'], 'policy_size': ref['policy_size'], 'history_plies': ref['history_plies'], 'state_planes': ref['state_planes'], 'moves': all_metas[0].get('moves'), 'rows': {'train': sum(int(m.get('rows', 0)) for m in metas), 'dev': int(dev_meta.get('rows', 0))}}


def main():
    ap = argparse.ArgumentParser(description='Build residual feature-cache shards from a supervised dataset manifest.')
    ap.add_argument('--dataset-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--history-plies', type=int, default=None)
    ap.add_argument('--state-planes', action='store_true')
    ap.add_argument('--current-board-18', action='store_true', help='Build Maia-like 18-plane current-board cache')
    ap.add_argument('--side-info', action='store_true', help='Write Maia-style side-info aux labels into caches')
    ap.add_argument('--max-rows-per-shard', type=int, default=10**12)
    ap.add_argument('--workers', type=int, default=1, help='Parallel train-shard cache builders')
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    out = Path(args.out_dir)
    man = json.loads((root / 'manifest.json').read_text())
    history = man.get('history_plies', 2) if args.history_plies is None else args.history_plies

    def build_train(rel: str) -> str:
        inp = root / rel
        cdir = out / 'train' / shard_name(rel)
        cmd = [args.python, 'training/build_residual_feature_cache.py', '--input', inp, '--out', cdir, '--max-rows', args.max_rows_per_shard, '--history-plies', history]
        if args.state_planes: cmd.append('--state-planes')
        if args.current_board_18: cmd.append('--current-board-18')
        if args.side_info: cmd.append('--side-info')
        run(cmd)
        return str(cdir)

    if args.workers <= 1:
        shard_cache_paths = [build_train(rel) for rel in man['train_shards']]
    else:
        shard_cache_paths = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_train, rel): rel for rel in man['train_shards']}
            for fut in as_completed(futs):
                shard_cache_paths.append(fut.result())
        shard_cache_paths.sort()

    dev_dir = out / 'dev'
    cmd = [args.python, 'training/build_residual_feature_cache.py', '--input', root / man['dev'], '--out', dev_dir, '--max-rows', man.get('total_dev_rows', 10**12), '--history-plies', history]
    if args.state_planes: cmd.append('--state-planes')
    if args.current_board_18: cmd.append('--current-board-18')
    if args.side_info: cmd.append('--side-info')
    run(cmd)
    validation = validate_cache_metas(shard_cache_paths, dev_dir, args.side_info)
    out.mkdir(parents=True, exist_ok=True)
    cache_manifest = {'dataset_manifest': str(root / 'manifest.json'), 'shards': shard_cache_paths, 'dev_cache': str(dev_dir), 'history_plies': 0 if args.current_board_18 else history, 'state_planes': False if args.current_board_18 else args.state_planes, 'input_mode': 'current_board_18' if args.current_board_18 else 'history', 'has_side_info': args.side_info, 'workers': args.workers, 'validation': validation}
    (out / 'cache_manifest.json').write_text(json.dumps(cache_manifest, indent=2))
    print(f'METRIC cache_manifest_input_planes={validation["input_planes"]}')
    print(f'METRIC cache_manifest_policy_size={validation["policy_size"]}')
    print(f'METRIC cache_manifest_shards={len(shard_cache_paths)}')
    print(f'METRIC cache_manifest_dev=1')

if __name__ == '__main__':
    main()
