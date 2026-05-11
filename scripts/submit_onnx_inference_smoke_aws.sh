#!/usr/bin/env bash
set -euo pipefail

REGION="${AWS_DEFAULT_REGION:-${AWS_REGION:-us-west-2}}"
JOB_QUEUE="${JOB_QUEUE:-tiny-leela-cache-queue}"
JOB_DEFINITION="${JOB_DEFINITION:-tiny-leela-onnx-native-smoke}"
BUCKET_URI="${BUCKET_URI:-s3://tiny-leela-distributed-ddbb}"
IMAGE_URI="${IMAGE_URI:-public.ecr.aws/docker/library/python:3.12-slim}"
VCPUS="${VCPUS:-2}"
MEMORY="${MEMORY:-4096}"
MODEL=""
META=""
LABEL="onnx_native_smoke"
POSITIONS="4"
REPEATS="1"
BATCHES="1,2"
SUBMIT=0
WAIT=0
REGISTER=1
JOB_ROLE_ARN="${JOB_ROLE_ARN:-}"

usage(){ cat <<'USAGE'
Usage: cloud/aws/submit_onnx_inference_smoke.sh --model MODEL.onnx --meta META.json [options]

Options:
  --label LABEL             default onnx_native_smoke
  --positions N             default 4
  --repeats N               default 1
  --batches CSV             default 1,2
  --bucket-uri S3_URI       default s3://tiny-leela-distributed-ddbb
  --job-queue NAME          default tiny-leela-cache-queue
  --job-definition NAME     default tiny-leela-onnx-native-smoke
  --image-uri URI           default public.ecr.aws/docker/library/python:3.12-slim
  --job-role-arn ARN        default inferred from tiny-leela-cache-squareformer-cache
  --no-register             reuse existing job definition
  --submit                  actually submit; otherwise dry-run after upload/register
  --wait                    poll until terminal and print CloudWatch tail/result fetch hints
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --meta) META="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --positions) POSITIONS="$2"; shift 2;;
    --repeats) REPEATS="$2"; shift 2;;
    --batches) BATCHES="$2"; shift 2;;
    --bucket-uri) BUCKET_URI="$2"; shift 2;;
    --job-queue) JOB_QUEUE="$2"; shift 2;;
    --job-definition) JOB_DEFINITION="$2"; shift 2;;
    --image-uri) IMAGE_URI="$2"; shift 2;;
    --job-role-arn) JOB_ROLE_ARN="$2"; shift 2;;
    --no-register) REGISTER=0; shift;;
    --submit) SUBMIT=1; shift;;
    --wait) WAIT=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ -s "$MODEL" ]] || { echo "missing --model: $MODEL" >&2; exit 2; }
[[ -s "$META" ]] || { echo "missing --meta: $META" >&2; exit 2; }
command -v aws >/dev/null || { echo "aws CLI required" >&2; exit 2; }

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SAFE_LABEL="$(printf '%s' "$LABEL" | tr -c 'A-Za-z0-9_.-' '_')"
RUN_ID="onnx-inference-smoke-${SAFE_LABEL}-${STAMP}"
S3_PREFIX="${BUCKET_URI%/}/onnx_inference_smoke/${SAFE_LABEL}/${STAMP}"
ART_DIR="artifacts/aws_onnx_inference_smoke/${SAFE_LABEL}_${STAMP}"
mkdir -p "$ART_DIR"

aws s3 cp --region "$REGION" "$MODEL" "$S3_PREFIX/input/model.onnx" >/dev/null
aws s3 cp --region "$REGION" "$META" "$S3_PREFIX/input/model.meta.json" >/dev/null
aws s3 cp --region "$REGION" eval/onnx_native_inference_benchmark.py "$S3_PREFIX/input/onnx_native_inference_benchmark.py" >/dev/null

if [[ -z "$JOB_ROLE_ARN" ]]; then
  JOB_ROLE_ARN="$(aws batch describe-job-definitions --region "$REGION" --job-definition-name tiny-leela-cache-squareformer-cache --status ACTIVE --query 'jobDefinitions[0].containerProperties.jobRoleArn' --output text)"
fi
[[ -n "$JOB_ROLE_ARN" && "$JOB_ROLE_ARN" != "None" ]] || { echo "could not infer --job-role-arn" >&2; exit 2; }

if [[ "$REGISTER" == 1 ]]; then
  cat > "$ART_DIR/job_definition.json" <<JSON
{
  "image": "$IMAGE_URI",
  "vcpus": $VCPUS,
  "memory": $MEMORY,
  "jobRoleArn": "$JOB_ROLE_ARN",
  "command": ["bash", "-lc", "echo override command required"],
  "environment": [
    {"name": "AWS_DEFAULT_REGION", "value": "$REGION"}
  ]
}
JSON
  aws batch register-job-definition \
    --region "$REGION" \
    --job-definition-name "$JOB_DEFINITION" \
    --type container \
    --container-properties "file://$ART_DIR/job_definition.json" > "$ART_DIR/register_job_definition.json"
fi

REMOTE_CMD="set -euo pipefail; python -m pip install --quiet --no-cache-dir numpy chess onnxruntime awscli; mkdir -p /work/input /work/output; aws s3 cp '$S3_PREFIX/input/model.onnx' /work/input/model.onnx >/dev/null; aws s3 cp '$S3_PREFIX/input/model.meta.json' /work/input/model.meta.json >/dev/null; aws s3 cp '$S3_PREFIX/input/onnx_native_inference_benchmark.py' /work/input/onnx_native_inference_benchmark.py >/dev/null; python /work/input/onnx_native_inference_benchmark.py --model /work/input/model.onnx --meta /work/input/model.meta.json --provider CPUExecutionProvider --require-provider CPUExecutionProvider --label '$LABEL' --positions '$POSITIONS' --repeats '$REPEATS' --batches '$BATCHES' | tee /work/output/run.log; aws s3 cp /work/output/run.log '$S3_PREFIX/output/run.log' >/dev/null"
python3 - "$REMOTE_CMD" > "$ART_DIR/container_overrides.json" <<'PY'
import json, sys
cmd = sys.argv[1]
print(json.dumps({"command": ["bash", "-lc", cmd]}))
PY

echo "RUN_ID=$RUN_ID" | tee "$ART_DIR/summary.env"
echo "S3_PREFIX=$S3_PREFIX" | tee -a "$ART_DIR/summary.env"
echo "JOB_QUEUE=$JOB_QUEUE" | tee -a "$ART_DIR/summary.env"
echo "JOB_DEFINITION=$JOB_DEFINITION" | tee -a "$ART_DIR/summary.env"

if [[ "$SUBMIT" != 1 ]]; then
  echo "DRY_RUN=1" | tee -a "$ART_DIR/summary.env"
  echo "prepared $ART_DIR"
  exit 0
fi

aws batch submit-job \
  --region "$REGION" \
  --job-name "$RUN_ID" \
  --job-queue "$JOB_QUEUE" \
  --job-definition "$JOB_DEFINITION" \
  --container-overrides "file://$ART_DIR/container_overrides.json" > "$ART_DIR/submit_job.json"
JOB_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["jobId"])' "$ART_DIR/submit_job.json")"
echo "JOB_ID=$JOB_ID" | tee -a "$ART_DIR/summary.env"

if [[ "$WAIT" == 1 ]]; then
  while :; do
    STATUS="$(aws batch describe-jobs --region "$REGION" --jobs "$JOB_ID" --query 'jobs[0].status' --output text)"
    echo "$(date -Is) status=$STATUS" | tee -a "$ART_DIR/watch.log"
    case "$STATUS" in
      SUCCEEDED|FAILED) break;;
    esac
    sleep 20
  done
  aws batch describe-jobs --region "$REGION" --jobs "$JOB_ID" > "$ART_DIR/describe_job.final.json"
  aws s3 cp --region "$REGION" "$S3_PREFIX/output/run.log" "$ART_DIR/run.log" || true
  [[ "$STATUS" == SUCCEEDED ]]
fi
