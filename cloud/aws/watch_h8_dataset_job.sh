#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-fba43ce3-3cdc-4a15-a852-d5e8713c98fd}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
OUT="${OUT:-artifacts/cloud_h8_dataset_10m/live_status.json}"
LOG_TAIL="${LOG_TAIL:-artifacts/cloud_h8_dataset_10m/live_cloudwatch_tail.log}"
DATASET_PREFIX="${DATASET_PREFIX:-s3://tiny-leela-distributed-ddbb/h8_dataset_10m/datasets/supervised_10m_elite_tcec_h8_v1}"
LOG_GROUP="${LOG_GROUP:-/aws/batch/job}"
SPOT_LOW="${SPOT_LOW:-0.006}"
SPOT_HIGH="${SPOT_HIGH:-0.025}"
ONDEMAND="${ONDEMAND:-0.05}"

mkdir -p "$(dirname "$OUT")"
DESC_FILE=$(mktemp)
trap 'rm -f "$DESC_FILE"' EXIT
AWS_DEFAULT_REGION="$REGION" aws batch describe-jobs --jobs "$JOB_ID" --region "$REGION" --output json > "$DESC_FILE"
python3 - "$DESC_FILE" "$OUT" "$LOG_TAIL" "$DATASET_PREFIX" "$LOG_GROUP" "$REGION" "$SPOT_LOW" "$SPOT_HIGH" "$ONDEMAND" <<'PY'
import json, os, re, subprocess, sys, time
from datetime import datetime, timezone

desc_file, out, log_tail, dataset_prefix, log_group, region, spot_low, spot_high, ondemand = sys.argv[1:10]
desc = json.load(open(desc_file))
job = (desc.get('jobs') or [{}])[0]
container = job.get('container') or {}
stream = container.get('logStreamName')
now_ms = int(time.time()*1000)
started = job.get('startedAt') or None
stopped = job.get('stoppedAt') or None
end_ms = stopped or now_ms
runtime_h = max(0, (end_ms - started)/3600000) if started else 0
vcpus = int(container.get('vcpus') or 0)

def run(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

messages=[]
progress=[]
if stream:
    cp = run(['aws','logs','get-log-events','--log-group-name',log_group,'--log-stream-name',stream,'--limit','200','--start-from-head','--region',region,'--output','json'])
    if cp.returncode == 0:
        ev = json.loads(cp.stdout).get('events') or []
        messages = [e.get('message','') for e in ev]
        open(log_tail,'w').write('\n'.join(messages[-80:])+'\n')
        rx = re.compile(r'progress input_games=(\d+) train_rows=(\d+) dev_rows=(\d+) source=([^\s]+)')
        for m in messages:
            mo = rx.search(m)
            if mo:
                progress.append({'input_games':int(mo.group(1)), 'train_rows':int(mo.group(2)), 'dev_rows':int(mo.group(3)), 'source':mo.group(4)})
    else:
        messages.append('LOG_FETCH_FAILED: '+cp.stderr.strip())

s3_cp = run(['aws','s3','ls',dataset_prefix.rstrip('/')+'/', '--recursive','--summarize','--region',region])
objects=size=0
if s3_cp.returncode == 0:
    for line in s3_cp.stdout.splitlines():
        s=line.strip()
        if s.startswith('Total Objects:'):
            objects=int(s.split(':',1)[1])
        elif s.startswith('Total Size:'):
            size=int(s.split(':',1)[1])
latest = progress[-1] if progress else None
result={
  'created_at': datetime.now(timezone.utc).isoformat(),
  'job_id': job.get('jobId'),
  'job_name': job.get('jobName'),
  'status': job.get('status'),
  'status_reason': job.get('statusReason'),
  'started_at_ms': started,
  'runtime_hours_so_far': runtime_h,
  'vcpus': vcpus,
  'vcpu_hours_so_far': runtime_h*vcpus,
  'estimated_compute_usd_so_far': {
    'spot_low': runtime_h*vcpus*float(spot_low),
    'spot_high': runtime_h*vcpus*float(spot_high),
    'ondemand_upper': runtime_h*vcpus*float(ondemand),
  },
  'log_stream': stream,
  'progress_events_seen': len(progress),
  'latest_progress': latest,
  'estimated_completion_from_rows': None,
  's3_output': {'objects': objects, 'bytes': size, 'prefix': dataset_prefix},
}
if latest and latest.get('train_rows'):
    done = min(1.0, latest['train_rows']/10000000)
    elapsed = runtime_h
    result['estimated_completion_from_rows'] = {
      'train_fraction': done,
      'eta_hours_at_current_rate': (elapsed/done-elapsed) if done > 0 else None,
      'train_rows_per_hour': latest['train_rows']/elapsed if elapsed > 0 else None,
    }
open(out,'w').write(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY

echo "WROTE $OUT"
echo "WROTE $LOG_TAIL"
