#!/usr/bin/env bash
set -euo pipefail
usage(){ cat <<'USAGE'
Download/extract cloud-built SquareFormer cache archives into local training layout.

Required:
  --cache-prefix s3://bucket/prefix/caches/DATASET/cache_squareformer_hN
  --out-dir data/datasets/.../cache_squareformer_hN
  --train-shards N
  --history N
  --dataset-manifest data/datasets/.../manifest.json

This writes local cache_manifest.json compatible with training streams.
USAGE
}
CACHE_PREFIX=""; OUT_DIR=""; TRAIN_SHARDS=""; HISTORY=""; DATASET_MANIFEST=""; REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cache-prefix) CACHE_PREFIX="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --train-shards) TRAIN_SHARDS="$2"; shift 2;;
    --history) HISTORY="$2"; shift 2;;
    --dataset-manifest) DATASET_MANIFEST="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage; exit 2;;
  esac
done
[[ -n "$CACHE_PREFIX" && -n "$OUT_DIR" && -n "$TRAIN_SHARDS" && -n "$HISTORY" && -n "$DATASET_MANIFEST" ]] || { usage >&2; exit 2; }
mkdir -p "$OUT_DIR/train" "$OUT_DIR/dev"
for i in $(seq 0 $((TRAIN_SHARDS-1))); do
  shard=$(printf 'shard_%04d' "$i")
  dest="$OUT_DIR/train/$shard"
  mkdir -p "$dest"
  echo "DOWNLOAD $shard"
  aws s3 cp "$CACHE_PREFIX/train/$shard/cache.tar.zst" "/tmp/$shard.cache.tar.zst" --region "$REGION"
  tar -C "$dest" -I zstd -xf "/tmp/$shard.cache.tar.zst"
  rm -f "/tmp/$shard.cache.tar.zst"
done
mkdir -p "$OUT_DIR/dev"
aws s3 cp "$CACHE_PREFIX/dev/shard_0000/cache.tar.zst" "/tmp/dev.cache.tar.zst" --region "$REGION"
tar -C "$OUT_DIR/dev" -I zstd -xf "/tmp/dev.cache.tar.zst"
rm -f "/tmp/dev.cache.tar.zst"
python3 - "$OUT_DIR" "$DATASET_MANIFEST" "$HISTORY" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); dataset_manifest=sys.argv[2]; hist=int(sys.argv[3])
shards=sorted(out.glob('train/shard_*'))
rows=sum(int(json.loads((p/'meta.json').read_text())['rows']) for p in shards)
dev=json.loads((out/'dev/meta.json').read_text())
first=json.loads((shards[0]/'meta.json').read_text())
man={'dataset_manifest':dataset_manifest,'shards':[str(p) for p in shards],'dev_cache':str(out/'dev'),'history_plies':hist,'workers':len(shards),'validation':{'rows':{'train':rows,'dev':int(dev['rows'])},'token_features':int(first['token_features']),'policy_size':int(first['policy_size'])},'notes':'Downloaded/extracted from AWS Batch cloud cache archives.'}
(out/'cache_manifest.json').write_text(json.dumps(man,indent=2)+'\n')
print(json.dumps(man['validation'],indent=2))
PY
