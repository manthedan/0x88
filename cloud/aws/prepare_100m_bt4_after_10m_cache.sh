#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Gate the 100M BT4/h7-h8 cloud pipeline on the 10M h7/h8 cache pilot.

Default is read-only: verify whether the 10M cache manifests are finalized and
print the next 100M commands. Use --submit-dataset to launch only the 100M h8
dataset job after the 10M pilot is OK. Cache fanout is intentionally separate
and should be run after the 100M dataset manifest validates.

Usage:
  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/prepare_100m_bt4_after_10m_cache.sh --dry-run

  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/prepare_100m_bt4_after_10m_cache.sh --submit-dataset --compress-inputs --upload-inputs
USAGE
}

MODE="dry-run"
COMPRESS=0
UPLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift;;
    --submit-dataset) MODE="submit-dataset"; shift;;
    --compress-inputs) COMPRESS=1; shift;;
    --upload-inputs) UPLOAD=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
TEN_BUCKET="${TEN_BUCKET:-s3://tiny-leela-distributed-ddbb/h8_dataset_10m}"
TEN_DATASET="${TEN_DATASET:-supervised_10m_elite_tcec_h8_v1}"
TEN_CACHE_BASE="$TEN_BUCKET/caches/$TEN_DATASET"

s3_has(){ aws s3 ls "$1" --region "$REGION" >/dev/null 2>&1; }

h7="$TEN_CACHE_BASE/cache_squareformer_h7/cache_manifest.json"
h8="$TEN_CACHE_BASE/cache_squareformer_h8/cache_manifest.json"
if s3_has "$h7" && s3_has "$h8"; then
  pilot_ok=1
else
  pilot_ok=0
fi

cat <<EOF
100M BT4 GATE
  10M h7 manifest: $h7
  10M h8 manifest: $h8
  pilot_ok: $pilot_ok
  mode: $MODE
EOF

if [[ "$pilot_ok" != "1" ]]; then
  echo "10M cache pilot is not finalized yet; not launching 100M dataset."
  echo "Monitor pilot: AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=$REGION cloud/aws/watch_squareformer_cache_jobs.sh --once"
  exit 0
fi

echo "Before first 100M submit, rebuild/push generalized dataset worker image:"
echo "  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=$REGION cloud/aws/build_push_dataset_worker.sh"
echo
cmd=(cloud/aws/submit_100m_h8_dataset_job.sh --submit)
if [[ "$COMPRESS" == "1" ]]; then cmd+=(--compress-inputs); fi
if [[ "$UPLOAD" == "1" ]]; then cmd+=(--upload-inputs); fi

printf 'DATASET_COMMAND:'
printf ' %q' "${cmd[@]}"
printf '\n'

echo "After 100M dataset succeeds:"
echo "  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=$REGION cloud/aws/submit_100m_h7_h8_cache_after_h8_dataset.sh --submit"
echo "  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=$REGION cloud/aws/watch_100m_squareformer_cache_jobs.sh --wait --finalize-on-success"

if [[ "$MODE" == "dry-run" ]]; then
  echo "dry-run: not submitting 100M dataset."
  exit 0
fi

"${cmd[@]}"
