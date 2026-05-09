#!/usr/bin/env bash
set -euo pipefail

usage(){ cat <<'USAGE'
Watch AWS Batch SquareFormer cache jobs and write a durable status/cost report.

Defaults match the h7/h8 cloud dataset cache plan. The script is read-only unless
--finalize-on-success is passed, in which case it calls finalize_squareformer_cache_s3.py
for histories whose train+dev jobs have succeeded.

Usage:
  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/watch_squareformer_cache_jobs.sh --once

  AWS_PROFILE=tiny-leela AWS_DEFAULT_REGION=us-west-2 \
  cloud/aws/watch_squareformer_cache_jobs.sh --wait --interval 120 --finalize-on-success

Options:
  --histories CSV          default 7,8
  --dataset-name NAME      default supervised_10m_elite_tcec_h8_v1
  --bucket-uri S3          default s3://tiny-leela-distributed-ddbb/h8_dataset_10m
  --job-queue NAME         default tiny-leela-cache-queue
  --region REGION          default env AWS_REGION/AWS_DEFAULT_REGION/us-west-2
  --train-shards N         default 40
  --expect-train N         default 10000000
  --expect-dev N           default 500000
  --out-dir DIR            default artifacts/cloud_h8_dataset_10m/cache_jobs
  --interval SEC           default 120
  --once                   single poll, default
  --wait                   poll until all found jobs are terminal
  --finalize-on-success    merge/validate S3 cache manifests after success
USAGE
}

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
HISTORIES="7,8"
DATASET_NAME="supervised_10m_elite_tcec_h8_v1"
BUCKET_URI="s3://tiny-leela-distributed-ddbb/h8_dataset_10m"
JOB_QUEUE="tiny-leela-cache-queue"
TRAIN_SHARDS="40"
EXPECT_TRAIN="10000000"
EXPECT_DEV="500000"
OUT_DIR="artifacts/cloud_h8_dataset_10m/cache_jobs"
INTERVAL="120"
WAIT=0
FINALIZE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --histories) HISTORIES="$2"; shift 2;;
    --dataset-name) DATASET_NAME="$2"; shift 2;;
    --bucket-uri) BUCKET_URI="$2"; shift 2;;
    --job-queue) JOB_QUEUE="$2"; shift 2;;
    --region) REGION="$2"; shift 2;;
    --train-shards) TRAIN_SHARDS="$2"; shift 2;;
    --expect-train) EXPECT_TRAIN="$2"; shift 2;;
    --expect-dev) EXPECT_DEV="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --interval) INTERVAL="$2"; shift 2;;
    --once) WAIT=0; shift;;
    --wait) WAIT=1; shift;;
    --finalize-on-success) FINALIZE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2;;
  esac
done

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 2; }
mkdir -p "$OUT_DIR"
STATUSES=(SUBMITTED PENDING RUNNABLE STARTING RUNNING SUCCEEDED FAILED)

poll_once(){
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN
  local names=()
  IFS=',' read -r -a HLIST <<< "$HISTORIES"
  for h in "${HLIST[@]}"; do
    h="$(echo "$h" | tr -d '[:space:]')"
    names+=("sqf-h${h}-${DATASET_NAME}-train" "sqf-h${h}-${DATASET_NAME}-dev")
  done

  local list_files=()
  for name in "${names[@]}"; do
    # AWS Batch JOB_NAME filters cannot be combined with job-status filters.
    # The filtered call returns recent jobs across statuses; pick newest below.
    f="$tmp/list_${name}.json"
    aws batch list-jobs \
      --region "$REGION" \
      --job-queue "$JOB_QUEUE" \
      --filters "name=JOB_NAME,values=$name" > "$f" || true
    list_files+=("$f")
  done

  python3 - "$tmp/job_names.json" "${names[@]}" <<'PY'
import json, sys
out = sys.argv[1]
json.dump(sys.argv[2:], open(out, 'w'))
PY

  python3 - "$tmp/latest_ids.txt" "$tmp/job_names.json" "${list_files[@]}" <<'PY'
import json, sys
out, names_path, *files = sys.argv[1:]
names = json.load(open(names_path))
by_name = {n: [] for n in names}
for fp in files:
    try:
        data = json.load(open(fp))
    except Exception:
        continue
    for j in data.get('jobSummaryList') or []:
        name = j.get('jobName')
        if name in by_name:
            by_name[name].append(j)
rows=[]
for name, jobs in by_name.items():
    if not jobs:
        rows.append((name, '-', 'MISSING'))
        continue
    jobs.sort(key=lambda j: int(j.get('createdAt') or 0), reverse=True)
    j=jobs[0]
    rows.append((name, j.get('jobId',''), j.get('status','UNKNOWN')))
with open(out,'w') as f:
    for row in rows:
        f.write('\t'.join(row)+'\n')
PY

  local ids=()
  while IFS=$'\t' read -r _name jid _status; do
    if [[ -n "${jid:-}" && "$jid" != "-" ]]; then
      seen=0
      for old in "${ids[@]:-}"; do [[ "$old" == "$jid" ]] && seen=1; done
      (( seen == 1 )) || ids+=("$jid")
    fi
  done < "$tmp/latest_ids.txt"

  if (( ${#ids[@]} > 0 )); then
    aws batch describe-jobs --region "$REGION" --jobs "${ids[@]}" > "$tmp/parents.json"
  else
    echo '{"jobs":[]}' > "$tmp/parents.json"
  fi

  # Child summaries/descriptions for train array jobs.
  local child_list_files=()
  while IFS=$'\t' read -r name jid _status; do
    [[ -n "${jid:-}" && "$jid" != "-" ]] || continue
    [[ "$name" == *-train ]] || continue
    for st in "${STATUSES[@]}"; do
      f="$tmp/children_${jid}_${st}.json"
      aws batch list-jobs --region "$REGION" --array-job-id "$jid" --job-status "$st" > "$f" || true
      child_list_files+=("$f")
    done
  done < "$tmp/latest_ids.txt"

  python3 - "$tmp/child_ids.txt" "${child_list_files[@]}" <<'PY'
import json, sys
out, *files = sys.argv[1:]
ids=[]
for fp in files:
    try: data=json.load(open(fp))
    except Exception: continue
    for j in data.get('jobSummaryList') or []:
        jid=j.get('jobId')
        if jid and jid not in ids: ids.append(jid)
open(out,'w').write('\n'.join(ids)+'\n')
PY

  child_ids=()
  while IFS= read -r cid; do
    [[ -n "$cid" ]] && child_ids+=("$cid")
  done < "$tmp/child_ids.txt"
  if (( ${#child_ids[@]} > 0 )); then
    # Current arrays are small (40), but chunk defensively at 100.
    : > "$tmp/children_desc_parts.jsonl"
    local start=0
    while (( start < ${#child_ids[@]} )); do
      local chunk=("${child_ids[@]:start:100}")
      aws batch describe-jobs --region "$REGION" --jobs "${chunk[@]}" >> "$tmp/children_desc_parts.jsonl"
      echo >> "$tmp/children_desc_parts.jsonl"
      start=$((start+100))
    done
  else
    : > "$tmp/children_desc_parts.jsonl"
  fi

  python3 - \
    "$tmp/latest_ids.txt" "$tmp/parents.json" "$tmp/children_desc_parts.jsonl" \
    "$OUT_DIR/status.json" "$OUT_DIR/status.md" "$BUCKET_URI" "$DATASET_NAME" "$HISTORIES" "$EXPECT_TRAIN" "$EXPECT_DEV" <<'PY'
import datetime as dt, json, sys, time
ids_path, parents_path, children_path, out_json, out_md, bucket, dataset, histories_csv, expect_train, expect_dev = sys.argv[1:]
parents = json.load(open(parents_path)).get('jobs') or []
parent_by_id = {j.get('jobId'): j for j in parents}
latest=[]
for line in open(ids_path):
    name, jid, status = line.rstrip('\n').split('\t')
    latest.append({'name': name, 'job_id': jid or None, 'list_status': status})
children=[]
for line in open(children_path):
    line=line.strip()
    if not line: continue
    try: children.extend(json.loads(line).get('jobs') or [])
    except Exception: pass
now_ms=int(time.time()*1000)
def vcpus(job):
    c=job.get('container') or {}
    for rr in c.get('resourceRequirements') or []:
        if rr.get('type') == 'VCPU':
            try: return int(float(rr.get('value')))
            except Exception: pass
    return int(c.get('vcpus') or 1)
def hours(job):
    s=job.get('startedAt') or 0
    if not s: return 0.0
    e=job.get('stoppedAt') or now_ms
    return max(0, e-s)/3600000.0
def cost_jobs(jobs):
    vh=sum(vcpus(j)*hours(j) for j in jobs)
    return {'vcpu_hours': vh, 'spot_low': vh*0.006, 'spot_high': vh*0.025, 'ondemand_upper': vh*0.05}
entries=[]
all_terminal=True
any_failed=False
for row in latest:
    jid=row['job_id']
    parent=parent_by_id.get(jid, {}) if jid else {}
    status=parent.get('status') or row['list_status']
    if status not in ('SUCCEEDED','FAILED'):
        all_terminal=False
    if status == 'FAILED':
        any_failed=True
    e={'name': row['name'], 'job_id': (jid if jid != '-' else None), 'status': status}
    if parent.get('arrayProperties'):
        e['array_status_summary'] = parent['arrayProperties'].get('statusSummary') or {}
        e['array_size'] = parent['arrayProperties'].get('size')
    if parent.get('statusReason'):
        e['status_reason']=parent.get('statusReason')
    entries.append(e)
report={
  'schema':'tiny_leela.squareformer_cache_job_watch.v1',
  'created_at': dt.datetime.now(dt.timezone.utc).isoformat(),
  'job_queue': parent_by_id[next(iter(parent_by_id))].get('jobQueue') if parent_by_id else None,
  'bucket_uri': bucket,
  'dataset_name': dataset,
  'histories': [int(x.strip()) for x in histories_csv.split(',') if x.strip()],
  'expected_rows': {'train': int(expect_train), 'dev': int(expect_dev)},
  'jobs': entries,
  'children_described': len(children),
  'cost_estimate': cost_jobs(children + [j for j in parents if not (j.get('arrayProperties') or {}).get('size')]),
  'terminal': all_terminal and bool(entries),
  'ok': (all_terminal and not any_failed and bool(entries)),
}
open(out_json,'w').write(json.dumps(report, indent=2)+'\n')
md=['# SquareFormer cache AWS Batch status', '', f"created: `{report['created_at']}`", '', '| job | status | job id | array |', '|---|---:|---|---:|']
for e in entries:
    arr=e.get('array_status_summary') or {}
    arrs=', '.join(f'{k}:{v}' for k,v in sorted(arr.items())) if arr else ''
    md.append(f"| `{e['name']}` | `{e['status']}` | `{e.get('job_id') or ''}` | {arrs} |")
ce=report['cost_estimate']
md += ['', f"vCPU-hours described: `{ce['vcpu_hours']:.4f}`", f"spot estimate: `${ce['spot_low']:.4f}`–`${ce['spot_high']:.4f}`", f"on-demand upper: `${ce['ondemand_upper']:.4f}`", '', f"terminal: `{report['terminal']}`", f"ok: `{report['ok']}`", '']
open(out_md,'w').write('\n'.join(md))
print(json.dumps({'out': out_json, 'terminal': report['terminal'], 'ok': report['ok'], 'jobs': [(e['name'], e['status']) for e in entries]}, indent=2))
PY

  if (( FINALIZE == 1 )); then
    python3 - "$OUT_DIR/status.json" "$BUCKET_URI" "$DATASET_NAME" "$TRAIN_SHARDS" "$EXPECT_TRAIN" "$EXPECT_DEV" "$REGION" <<'PY'
import json, subprocess, sys
status_path, bucket, dataset, train_shards, expect_train, expect_dev, region = sys.argv[1:]
r=json.load(open(status_path))
if not r.get('ok'):
    print('finalize skipped: jobs not all succeeded')
    raise SystemExit(0)
for h in r['histories']:
    cache=f'{bucket}/caches/{dataset}/cache_squareformer_h{h}'
    out=f'artifacts/cloud_h8_dataset_10m/cache_jobs/cache_squareformer_h{h}_manifest.json'
    cmd=['python3','cloud/aws/finalize_squareformer_cache_s3.py','--cache-prefix',cache,'--dataset-manifest-s3',f'{bucket}/datasets/{dataset}/manifest.json','--history-plies',str(h),'--train-shards',train_shards,'--expect-train-rows',expect_train,'--expect-dev-rows',expect_dev,'--region',region,'--out',out,'--upload','--strict']
    print('+', ' '.join(cmd))
    subprocess.check_call(cmd)
PY
  fi
}

while true; do
  poll_once
  if (( WAIT == 0 )); then break; fi
  if python3 - "$OUT_DIR/status.json" <<'PY'
import json, sys
r=json.load(open(sys.argv[1]))
raise SystemExit(0 if r.get('terminal') else 1)
PY
  then
    break
  fi
  sleep "$INTERVAL"
done
