#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ddbb/projects/tiny_leela}
cd "$ROOT"
ACTION=${1:-status}
JOB_ID=${JOB_ID:-7TedUMyjk6}
STORE=${STORE:-.pi/schedule-prompts.json}
case "$ACTION" in
  enable|disable|status) ;;
  *) echo "usage: $0 {status|enable|disable}" >&2; exit 2 ;;
esac
python3 - "$ACTION" "$JOB_ID" "$STORE" <<'PY'
import json, pathlib, sys, datetime
act, job_id, store = sys.argv[1:]
p = pathlib.Path(store)
if not p.exists():
    raise SystemExit(f'missing schedule store: {p}')
data = json.loads(p.read_text())
for job in data.get('jobs', []):
    if job.get('id') == job_id or job.get('name') == job_id:
        if act == 'status':
            print(f"{job.get('name')} ({job.get('id')}) enabled={job.get('enabled')} schedule={job.get('schedule')} type={job.get('type')} lastRun={job.get('lastRun')}")
        else:
            job['enabled'] = (act == 'enable')
            job['updatedAt'] = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z')
            p.write_text(json.dumps(data, indent=2) + '\n')
            print(f"{act}d {job.get('name')} ({job.get('id')}) in {p}")
            print('Note: disabling prevents future fires; it does not abort an already-running scheduled subagent.')
        break
else:
    raise SystemExit(f'job not found: {job_id}')
PY
