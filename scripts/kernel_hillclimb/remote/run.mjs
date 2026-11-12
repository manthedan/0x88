#!/usr/bin/env node
// Remote kernel-bench runner. See README.md in this directory for the page contract
// and the per-host scheduling rules (Yukon: gpu-queue; Mac mini: lock + wait-for-idle;
// local: benchlock.sh).
//
//   node scripts/kernel_hillclimb/remote/run.mjs --host yukon|mac-mini|local \
//        --dist <local dir> --page <path?query> [--timeout 120] [options]
//
// Options:
//   --include a,b/**,c     only sync these paths/globs (rsync include patterns, relative to --dist)
//   --with-models          do not apply the default model-blob excludes
//   --exclude p,q          extra rsync exclude patterns
//   --estimate-min N       honest minutes for gpu-queue -e (yukon, default 5)
//   --name NAME            job name (default: page basename)
//   --no-wait              mac-mini: skip the wait-for-idle gate (adapter/feature probes only;
//                          the result is marked timingValid:false)
//   --load-max 3 --gpu-max 10 --hold 60 --poll 30   mac-mini idle gate thresholds
//   --max-wait S           give up waiting locally after S seconds (default 21600). The remote
//                          job keeps going; collect it later with --fetch.
//   --fetch RUN_ID         don't start anything: poll/fetch the result of an earlier run
//   --chrome PATH          override the host's Chrome binary
//   --flags-json JSON      override the host's Chrome flags (array)
//   --add-flags a,b        append Chrome flags to the host defaults
//   --gpu-info             include CDP SystemInfo.getInfo (chrome://gpu equivalent) in meta
//   --out FILE             also write the final JSON to FILE
//   --dry-run              show what would be synced/submitted, then stop
// Exit: 0 ok, 2 page error/timeout, 3 runner/agent failure, 4 refused (0x88 job already
// waiting), 75 confirmed never started (host busy), 76 still pending remotely (--fetch later).
import { spawnSync, spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, statSync, mkdtempSync } from 'node:fs';
import { dirname, join, resolve, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import { makeRunId, validateEstimate, q, slug, lockCommand, parseQueueList } from './lib.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const BENCHLOCK = '/Users/macthedan/projects/lc0_browser/notes/kernel-hillclimb/benchlock.sh';

// ssh/rsync options: bounded connection setup and dead-link detection (see sh()).
const SSH_OPTS = ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=4'];
const RSYNC_SSH = ['-e', `ssh ${SSH_OPTS.join(' ')}`, '--timeout=120'];
const COMMON_FLAGS = ['--headless=new', '--enable-unsafe-webgpu', '--enable-webgpu-developer-features'];
export const HOSTS = {
  yukon: {
    ssh: 'yukon', base: 'kernel-hillclimb', scheduler: 'gpu-queue', platform: 'linux',
    pathPrefix: 'export PATH=$HOME/.local/bin:$PATH;',
    chrome: '/home/ddbb/.cache/puppeteer/chrome/linux-151.0.7922.71/chrome-linux64/chrome',
    // --no-sandbox is required: without it Chrome never comes up on Yukon (no usable
    // setuid/userns sandbox for the CfT build). Every no-sandbox-less config failed in the
    // 2026-11-11 bring-up matrix (gpu-queue job 585).
    // vulkan_enable_f16_on_nvidia: Dawn hides shader-f16 on NVIDIA Vulkan by default.
    flags: [...COMMON_FLAGS, '--no-sandbox', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--use-angle=vulkan',
      '--ignore-gpu-blocklist', '--disable-vulkan-surface', '--enable-gpu', '--enable-dawn-features=vulkan_enable_f16_on_nvidia'],
  },
  'mac-mini': {
    ssh: 'mac-mini', base: 'kernel-hillclimb', scheduler: 'idle-wait', platform: 'darwin',
    pathPrefix: 'export PATH=/opt/homebrew/bin:$PATH;',
    chrome: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    flags: COMMON_FLAGS,
  },
  local: {
    scheduler: 'benchlock',
    chrome: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    flags: COMMON_FLAGS,
  },
};
// Model blobs and other large payloads nobody's kernel page should need by default.
const DEFAULT_EXCLUDES = ['*.onnx', '*.onnx.data', '*.pb', '*.pb.gz', '*.pt', '*.safetensors', '*.nnue', '*.lc0web', '*.lc0web/',
  'models/', 'tensor-cache/', 'params_shard_*.bin', '*.map', '.DS_Store'];

const args = parseArgs(process.argv.slice(2));
const hostName = args.host;
const H = HOSTS[hostName];
if (!H) die(`--host must be one of ${Object.keys(HOSTS).join(', ')}`);
const timeoutS = Number(args.timeout ?? 120);
const maxWaitS = Number(args['max-wait'] ?? 21600);
const chrome = args.chrome ?? H.chrome;
const flags = args['flags-json'] ? JSON.parse(args['flags-json']) : [...H.flags, ...(args['add-flags'] ? String(args['add-flags']).split(',') : [])];
let estimate;
try { estimate = validateEstimate(args['estimate-min'] ?? 5); } catch (e) { die(e.message); }

main().catch((e) => die(String(e?.stack ?? e)));

async function main() {
  if (hostName === 'local') return runLocal();
  const home = sh(H, 'printf %s "$HOME"').stdout;
  if (args.fetch && !/^[A-Za-z0-9_.-]+$/.test(args.fetch)) die('--fetch takes a run id');
  const runId = args.fetch ?? makeRunId(args.name ?? basename((args.page ?? 'page').split('?')[0]));
  const remoteDir = `${home}/${H.base}/${runId}`;
  if (!args.fetch) await launch(home, runId, remoteDir);
  const result = await waitForResult(runId, remoteDir);
  finish(result);
}

async function launch(home, runId, remoteDir) {
  if (!args.dist || !args.page) die('--dist and --page are required');
  const dist = resolve(args.dist);
  if (!existsSync(dist) || !statSync(dist).isDirectory()) die(`--dist ${dist} is not a directory`);
  const filters = rsyncFilters();
  if (H.scheduler === 'gpu-queue' && !args['dry-run']) {
    const rows = queueRows();
    if (!rows) die('could not read gpu-queue list');
    const waiting = rows.filter((r) => r.state === 'queued' && r.label.startsWith('[0x88:'));
    if (waiting.length) die(`a 0x88 job is already waiting in gpu-queue (${waiting.map((r) => r.id).join(', ')}); policy: at most one. Use --fetch on it or wait.`, 4);
  }
  // Hard-link unchanged files against the most recent earlier dist on the host.
  const prev = sh(H, `ls -td ${home}/${H.base}/*/dist 2>/dev/null | head -1`, { allowFail: true }).stdout.trim();
  const rs = ['-a', '--prune-empty-dirs', '--stats', ...filters, ...(prev ? [`--link-dest=${prev}`] : []), ...(args['dry-run'] ? ['--dry-run'] : [])];
  // Exclusive allocation: plain mkdir fails if the run dir already exists.
  sh(H, `mkdir -p ${q(`${home}/${H.base}`)} && mkdir ${q(remoteDir)} && mkdir ${q(remoteDir)}/dist`, { dry: args['dry-run'] });
  const r1 = spawnSync('rsync', [...RSYNC_SSH, ...rs, `${dist}/`, `${H.ssh}:${remoteDir}/dist/`], { encoding: 'utf8' });
  if (r1.status !== 0) die(`rsync failed: ${r1.stderr}`);
  const sent = r1.stdout.match(/Total transferred file size: ([\d,.]+\w*)/)?.[1];
  const total = r1.stdout.match(/Total file size: ([\d,.]+\w*)/)?.[1];
  log(`synced dist -> ${hostName}:${remoteDir}/dist (total ${total} bytes, transferred ${sent} bytes${prev ? `, link-dest ${prev}` : ''})`);
  if (args['dry-run']) { log(r1.stdout); return process.exit(0); }

  const nodeBin = sh(H, 'command -v node').stdout.trim();
  const agentArgs = ['--root', 'dist', '--page', args.page, '--chrome', chrome, '--flags-json', JSON.stringify(flags),
    '--timeout', String(timeoutS), '--out', 'result.json', ...(args['gpu-info'] ? ['--gpu-info'] : [])];
  let job;
  if (H.scheduler === 'gpu-queue') {
    job = jobScript(remoteDir, nodeBin, agentArgs, true);
  } else {
    const gate = [];
    if (!args['no-wait']) gate.push('--wait-idle', '--load-max', String(Number(args['load-max'] ?? 3)), '--gpu-max', String(Number(args['gpu-max'] ?? 10)),
      '--hold', String(Number(args.hold ?? 60)), '--poll', String(Number(args.poll ?? 30)), '--max-wait', String(maxWaitS));
    // Our own jobs serialize on a kernel advisory lock held for the job's whole life.
    const lock = lockCommand(H.platform, `${home}/${H.base}/.host.lockf`, maxWaitS);
    job = jobScript(remoteDir, nodeBin, [...agentArgs, ...gate], false, lock);
  }
  const tmp = mkdtempSync(join(os.tmpdir(), 'khc-run-'));
  writeFileSync(join(tmp, 'job.sh'), job, { mode: 0o755 });
  writeFileSync(join(tmp, 'run.json'), JSON.stringify({ runId, host: hostName, page: args.page, dist, chrome, flags, scheduler: H.scheduler, noWait: !!args['no-wait'], submittedAt: new Date().toISOString() }, null, 1));
  const r2 = spawnSync('rsync', [...RSYNC_SSH, '-a', join(HERE, 'agent.mjs'), join(HERE, 'lib.mjs'), join(tmp, 'job.sh'), join(tmp, 'run.json'), `${H.ssh}:${remoteDir}/`], { encoding: 'utf8' });
  if (r2.status !== 0) die(`rsync tools failed: ${r2.stderr}`);

  if (H.scheduler === 'gpu-queue') {
    const name = slug(args.name ?? basename(args.page.split('?')[0]));
    // Admission check and submit are one critical section on the host (flock), so two
    // controllers cannot both see "no 0x88 job waiting" and both submit.
    const admit = [
      `exec 9>>${q(`${home}/${H.base}/.submit.flock`)}`,
      'flock -w 300 9 || { echo ADMISSION_LOCK_TIMEOUT; exit 5; }',
      `if gpu-queue list | awk '$2=="queued"' | grep -qF '[0x88:'; then echo ALREADY_WAITING; exit 4; fi`,
      // The job id is recorded on the host in the same command, so --fetch works even if
      // this ssh call dies after the submit.
      `cd ${q(remoteDir)} || exit 3`,
      `id=$(gpu-queue submit -p 0x88 -e ${q(estimate)} -n ${q(name)} -o ${q(remoteDir + '/job.log')} -- ${q(remoteDir + '/job.sh')}) || { echo "$id"; exit 6; }`,
      `id=$(printf %s "$id" | tail -1 | tr -d '[:space:]'); printf %s "$id" > queue-job-id; echo "$id"`,
    ].join('\n');
    const r = sh(H, admit, { allowFail: true, timeoutMs: 420_000 });
    if (r.timedOut) pendingUnknown(runId, remoteDir, 'gpu-queue admission/submit');
    const out = r.stdout.trim();
    if (r.status === 4) die(`a 0x88 job is already waiting in gpu-queue; policy: at most one. Nothing submitted (run dir ${remoteDir} left unused).`, 4);
    if (r.status !== 0) die(`gpu-queue submit failed (${r.status}): ${out} ${r.stderr}`);
    const id = out.split('\n').pop().trim();
    if (!/^\d+$/.test(id)) die(`unexpected gpu-queue submit output: ${out}`);
    log(`submitted gpu-queue job ${id} (-p 0x88 -e ${estimate} -n ${name}); run id ${runId}`);
  } else {
    // NB: `cd X && nohup … &` would background the whole list and keep ssh's stdout open.
    const r = sh(H, `cd ${q(remoteDir)} || exit 1; nohup ./job.sh > job.log 2>&1 < /dev/null & echo $! > ${q(remoteDir + '/job.pid')}; cat ${q(remoteDir + '/job.pid')}`, { allowFail: true });
    if (r.timedOut) pendingUnknown(runId, remoteDir, 'detached job start');
    if (r.status !== 0) die(`starting the job failed (${r.status}): ${r.stderr}`);
    log(`started detached job on ${hostName} (pid ${r.stdout.trim()}); run id ${runId}`);
  }
}

function jobScript(remoteDir, nodeBin, agentArgs, nvidia, lock = null) {
  const gpuSnap = nvidia
    ? 'nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
    : 'uptime; ioreg -r -d 1 -w 0 -c IOAccelerator | grep -o \'"Device Utilization %"=[0-9]*\' | head -1';
  return `#!/usr/bin/env bash
# Generated by run.mjs. The whole lifecycle (static server, headless Chrome, page, teardown)
# happens inside agent.mjs; nothing it starts outlives it.
set -u
cd ${q(remoteDir)} || exit 3
echo "job start $(date) pid $$"; ${gpuSnap}
${lock ? `echo "[job $(date)] waiting for host lock ${lock[lock.length - 1]}" >> agent.err
# --wait-until bounds the idle wait so lock wait + idle wait never exceed max-wait in total.
WAIT_UNTIL=$(( ($(date +%s) + ${Number(maxWaitS)}) * 1000 ))
${lock.map(q).join(' ')} ${q(nodeBin)} agent.mjs ${agentArgs.map(q).join(' ')} --wait-until "$WAIT_UNTIL" > agent.out 2>> agent.err
rc=$?
if [ "$rc" -eq 75 ] && [ ! -s result.json ]; then
  printf '%s\n' '{"ok":false,"result":null,"error":"busy: host lock not acquired within max-wait","busy":true,"meta":{}}' > result.json
fi` : `${q(nodeBin)} agent.mjs ${agentArgs.map(q).join(' ')} > agent.out 2> agent.err
rc=$?`}
echo "job end $(date) rc=$rc"; ${gpuSnap}
echo "leftover khc-chrome processes (expected none):"; ps -u "$(id -u)" -o pid=,command= | grep -F 'khc-chrome-' | grep -v grep || true
exit $rc
`;
}

async function waitForResult(runId, remoteDir) {
  const t0 = Date.now(), budgetEnd = t0 + maxWaitS * 1000;
  // Each status call gets at most 120 s and never more than the remaining budget (min 5 s),
  // so a hung ssh cannot hold the controller past --max-wait.
  const rsh = (cmd) => sh(H, cmd, { allowFail: true, timeoutMs: Math.min(120_000, Math.max(5_000, budgetEnd - Date.now())) });
  let lastNote = '', lastLogAt = 0, jobId = null;
  const pendingOut = () => {
    log(`local max-wait reached; the remote job keeps going. Collect later: node scripts/kernel_hillclimb/remote/run.mjs --host ${hostName} --fetch ${runId}`);
    // Not 75: the job may still start (or be running). 75 is only for confirmed never-started.
    return { j: null, jobId, runId, remoteDir, exit: 76, pending: true, failure: 'local max-wait reached (or remote status unconfirmable); job still pending remotely' };
  };
  while (true) {
    let note;
    const r = rsh(`cat ${q(remoteDir + '/result.json')} 2>/dev/null`);
    if (r.status === 0 && r.stdout.trim()) {
      const j = JSON.parse(r.stdout.trim().split('\n').pop());
      if (!jobId && H.scheduler === 'gpu-queue') jobId = rsh(`cat ${q(remoteDir + '/queue-job-id')} 2>/dev/null`).stdout.trim() || null;
      return { j, jobId, runId, remoteDir };
    }
    // Checked here (not only at the bottom) so no `continue` path can outlive the budget.
    if (Date.now() >= budgetEnd) return pendingOut();
    if (r.status !== 0 && r.status !== 1) note = 'status unknown (ssh failed or timed out); retrying';
    else if (H.scheduler === 'gpu-queue') {
      if (!jobId) {
        const idr = rsh(`cat ${q(remoteDir + '/queue-job-id')} 2>/dev/null`);
        if (idr.status === 0 && /^\d+$/.test(idr.stdout.trim())) jobId = idr.stdout.trim();
        else if (idr.status === 1) return { j: null, jobId, runId, remoteDir, failure: `no gpu-queue job id recorded in ${remoteDir}` };
      }
      const rows = jobId ? queueRows(rsh) : null;
      if (!rows) note = 'gpu-queue status unknown (ssh failed or timed out); retrying';
      else {
        const row = rows.find((x) => x.id === jobId);
        if (!row) return { j: null, jobId, runId, remoteDir, failure: `gpu-queue job ${jobId} not found` };
        if (row.state === 'finished' || row.state === 'skipped') {
          await sleep(2000);
          const again = rsh(`test -s ${q(remoteDir + '/result.json')}`);
          if (again.status !== 1) continue; // result appeared, or status unknown: re-check
          const tail = rsh(`tail -20 ${q(remoteDir + '/job.log')} 2>/dev/null; tail -5 ${q(remoteDir + '/agent.err')} 2>/dev/null`).stdout;
          return { j: null, jobId, runId, remoteDir, exit: row.elevel === '75' ? 75 : 3, failure: `gpu-queue job ${jobId} ${row.state} exit=${row.elevel} without result.json${row.elevel === '75' ? ' (never started: GPU busy for 30 min; investigate, do not resubmit in a loop)' : ''}\n${tail}` };
        }
        const ahead = rows.filter((x) => (x.state === 'running' || x.state === 'queued') && Number(x.id) < Number(jobId));
        note = `gpu-queue job ${jobId}: ${row.state}; ahead: ${ahead.map((x) => `${x.id}${x.label}`).join(' ') || 'none'}`;
      }
    } else {
      const a = rsh(`if kill -0 "$(cat ${q(remoteDir + '/job.pid')})" 2>/dev/null; then echo alive; else echo dead; fi; tail -1 ${q(remoteDir + '/agent.err')} 2>/dev/null`);
      const [state = '', errTail = ''] = a.stdout.split('\n');
      if (a.status !== 0 || (state !== 'alive' && state !== 'dead')) note = 'job status unknown (ssh failed or timed out); retrying';
      else if (state === 'dead') {
        await sleep(1000);
        const again = rsh(`test -s ${q(remoteDir + '/result.json')}`);
        if (again.status !== 1) continue;
        return { j: null, jobId, runId, remoteDir, failure: `remote job exited without result.json\n${rsh(`tail -20 ${q(remoteDir + '/job.log')}; tail -20 ${q(remoteDir + '/agent.err')}`).stdout}` };
      } else note = errTail.trim() || 'running';
    }
    // Log changes, but rate-limit repeated "waiting" lines that only differ in counters.
    const key = note.replace(/\[agent [^\]]*\]/, '').replace(/[\d.]+/g, '#');
    if (key !== lastNote || Date.now() - lastLogAt > 300_000) { log(note); lastNote = key; lastLogAt = Date.now(); }
    if (Date.now() >= budgetEnd) return pendingOut();
    await sleep(Math.max(0, Math.min(H.scheduler === 'gpu-queue' ? 20_000 : 15_000, budgetEnd - Date.now())));
  }
}

// The remote side may or may not have acted (the ssh call timed out): report pending (76)
// with the run id, so the caller collects or inspects it with --fetch instead of resubmitting.
function pendingUnknown(runId, remoteDir, what) {
  log(`${what} timed out; outcome unknown. Inspect/collect with: node scripts/kernel_hillclimb/remote/run.mjs --host ${hostName} --fetch ${runId}`);
  finish({ j: null, jobId: null, runId, remoteDir, exit: 76, pending: true, failure: `${what}: ssh timed out, remote outcome unknown` });
  return new Promise(() => {});
}

function finish({ j, jobId, runId, remoteDir, failure, exit, pending }) {
  const runner = { host: hostName, runId, remoteDir, scheduler: H?.scheduler, queueJobId: jobId ?? null, fetchedAt: new Date().toISOString() };
  if (H?.ssh) {
    if (remoteDir) { try { runner.run = JSON.parse(sh(H, `cat ${q(remoteDir + '/run.json')}`, { allowFail: true, timeoutMs: 30_000 }).stdout); } catch {} }
  }
  // Valid only when the agent positively verified the run was uncontended (meta.verified ===
  // true: every during-run CPU/GPU attribution sample succeeded). Results from older agents
  // without that field are not valid.
  const verifiedClean = j?.meta?.verified === true && j?.meta?.contended === false;
  if (H?.scheduler === 'idle-wait') runner.timingValid = !(runner.run?.noWait) && verifiedClean;
  if (H?.scheduler === 'gpu-queue' || H?.scheduler === 'benchlock') runner.timingValid = verifiedClean;
  const out = j ? { ...j, runner } : { ok: false, result: null, error: failure, ...(pending ? { pending: true } : {}), runner };
  const text = JSON.stringify(out, null, 1);
  if (args.out) writeFileSync(args.out, text + '\n');
  const code = j ? (j.ok ? 0 : j.busy ? 75 : 2) : (exit ?? 3);
  // Exit only after stdout drained: process.exit() can truncate a piped write.
  process.stdout.write(text + '\n', () => process.exit(code));
}

function runLocal() {
  if (!args.dist || !args.page) die('--dist and --page are required');
  const agentArgs = [join(HERE, 'agent.mjs'), '--root', resolve(args.dist), '--page', args.page, '--chrome', chrome,
    '--flags-json', JSON.stringify(flags), '--timeout', String(timeoutS), ...(args['gpu-info'] ? ['--gpu-info'] : [])];
  // Local timing runs are serialized with the other agents through benchlock.sh.
  const r = spawnSync(BENCHLOCK, [`remote-run-local-${slug(args.name ?? 'page')}`, '--', process.execPath, ...agentArgs], { encoding: 'utf8', maxBuffer: 256 << 20, stdio: ['ignore', 'pipe', 'inherit'] });
  let j = null; try { j = JSON.parse(r.stdout.trim().split('\n').pop()); } catch {}
  finish({ j, jobId: null, runId: 'local', remoteDir: null, failure: j ? null : `agent produced no JSON (exit ${r.status})` });
}

// Returns null when the list could not be read (failure or timeout): status unknown.
function queueRows(run = (cmd) => sh(H, cmd, { allowFail: true })) {
  const r = run('gpu-queue list');
  return r.status === 0 ? parseQueueList(r.stdout) : null;
}

function rsyncFilters() {
  const f = [];
  for (const p of (args.exclude ? String(args.exclude).split(',') : [])) f.push(`--exclude=${p}`);
  if (!args['with-models']) for (const p of DEFAULT_EXCLUDES) f.push(`--exclude=${p}`);
  if (args.include) {
    for (const p of String(args.include).split(',')) {
      // include every parent directory of the pattern, then the pattern itself (and its contents if a dir)
      const parts = p.replace(/^\/+/, '').split('/');
      for (let i = 1; i < parts.length; i++) f.push(`--include=/${parts.slice(0, i).join('/')}/`);
      f.push(`--include=/${parts.join('/')}`, `--include=/${parts.join('/')}/**`);
    }
    f.push('--exclude=*');
  }
  return f;
}

// Every ssh call is bounded: ConnectTimeout covers connection setup, ServerAlive* a dead
// established link, and the spawnSync timeout (default 120 s, or the caller's remaining
// budget) a remote command that hangs. A timed-out call has r.timedOut = true and is a
// failure (die unless allowFail). Callers that cannot tell whether the remote side acted
// must treat a timeout as "unknown", never as "did not happen".
function sh(h, cmd, { allowFail = false, dry = false, timeoutMs = 120_000 } = {}) {
  if (dry) { log(`[dry] ${h.ssh}: ${cmd}`); return { status: 0, stdout: '', stderr: '' }; }
  const r = spawnSync('ssh', [...SSH_OPTS, h.ssh, `bash -lc ${q(`${h.pathPrefix ?? ''} ${cmd}`)}`],
    { encoding: 'utf8', maxBuffer: 256 << 20, timeout: Math.max(1000, Math.round(timeoutMs)), killSignal: 'SIGKILL' });
  r.timedOut = r.error?.code === 'ETIMEDOUT';
  r.stdout ??= ''; r.stderr ??= '';
  if (r.timedOut) r.status = null;
  if (r.status !== 0 && !allowFail) die(`ssh ${h.ssh} ${r.timedOut ? `timed out after ${Math.round(timeoutMs / 1000)}s` : `failed (${r.status})`}: ${cmd}\n${r.stderr}`);
  return r;
}

function log(s) { process.stderr.write(`[run ${hostName} ${new Date().toISOString().slice(11, 19)}] ${s}\n`); }
function die(msg, code = 3) { process.stderr.write(`run.mjs: ${msg}\n`); process.exit(code); }
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
function parseArgs(argv) {
  const o = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]; if (!a.startsWith('--')) continue;
    const eq = a.indexOf('=');
    if (eq > 0) { o[a.slice(2, eq)] = a.slice(eq + 1); continue; }
    const k = a.slice(2); const n = argv[i + 1];
    if (n === undefined || n.startsWith('--')) o[k] = true; else { o[k] = n; i++; }
  }
  return o;
}
