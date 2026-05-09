#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, tempfile
from pathlib import Path


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def s3_cp(src: str, dst: str, region: str) -> None:
    subprocess.check_call(['aws', 's3', 'cp', src, dst, '--region', region], stdout=subprocess.DEVNULL)


def read_json_from_s3(uri: str, region: str) -> dict:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'x.json'
        s3_cp(uri, str(p), region)
        return json.loads(p.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description='Merge S3 SquareFormer cache worker metadata into a cloud cache manifest.')
    ap.add_argument('--cache-prefix', required=True, help='s3://.../caches/DATASET/cache_squareformer_hN')
    ap.add_argument('--dataset-manifest-s3', required=True)
    ap.add_argument('--history-plies', type=int, required=True)
    ap.add_argument('--train-shards', type=int, required=True)
    ap.add_argument('--out', required=True, help='local JSON output path')
    ap.add_argument('--region', default='us-west-2')
    args = ap.parse_args()

    cache = args.cache_prefix.rstrip('/')
    train_metas = []
    rows_train = 0
    first_meta = None
    for i in range(args.train_shards):
        uri = f'{cache}/train/shard_{i:04d}/meta.json'
        meta = read_json_from_s3(uri, args.region)
        train_metas.append(uri.rsplit('/', 1)[0])
        rows_train += int(meta['rows'])
        first_meta = first_meta or meta
    dev_meta_uri = f'{cache}/dev/shard_0000/meta.json'
    dev_meta = read_json_from_s3(dev_meta_uri, args.region)
    manifest = {
        'schema': 'tiny_leela.squareformer_cache_cloud_manifest.v1',
        'dataset_manifest': args.dataset_manifest_s3,
        'cache_prefix': cache,
        'shards': train_metas,
        'dev_cache': f'{cache}/dev/shard_0000',
        'history_plies': args.history_plies,
        'workers': args.train_shards + 1,
        'archive_name': 'cache.tar.zst',
        'validation': {
            'rows': {'train': rows_train, 'dev': int(dev_meta['rows'])},
            'token_features': int(first_meta['token_features']),
            'policy_size': int(first_meta['policy_size']),
        },
        'notes': 'Cloud manifest for S3 cache archives. Download/extract to local cache dirs before PyTorch training.'
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['validation'], indent=2))

if __name__ == '__main__':
    main()
