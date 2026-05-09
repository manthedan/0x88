#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def s3_join(prefix: str, rel: str) -> str:
    return prefix.rstrip('/') + '/' + rel.lstrip('/')


def main() -> None:
    ap = argparse.ArgumentParser(description='Write S3 shard-list files for AWS Batch SquareFormer cache jobs.')
    ap.add_argument('--dataset-manifest', required=True)
    ap.add_argument('--dataset-s3-prefix', required=True, help='S3 prefix containing manifest/train/dev layout')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    manifest_path = Path(args.dataset_manifest)
    manifest = json.loads(manifest_path.read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train = [s3_join(args.dataset_s3_prefix, p) for p in manifest['train_shards']]
    dev = [s3_join(args.dataset_s3_prefix, manifest['dev'])]
    (out / 'train_shards.s3.txt').write_text('\n'.join(train) + '\n')
    (out / 'dev_shards.s3.txt').write_text('\n'.join(dev) + '\n')
    summary = {
        'dataset_manifest': str(manifest_path),
        'dataset_s3_prefix': args.dataset_s3_prefix,
        'train_shards': len(train),
        'dev_shards': len(dev),
        'history_plies_in_dataset': manifest.get('history_plies'),
        'total_train_rows': manifest.get('total_train_rows'),
        'total_dev_rows': manifest.get('total_dev_rows'),
    }
    (out / 'shard_lists.summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
