// Remote kernel-bench runner: run ids, estimate validation, static-root containment,
// host lock (kernel advisory lock, released when the holder dies), queue parsing,
// idle gate, and the page error contract.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, realpathSync, rmSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir, platform } from 'node:os';
import { spawn, spawnSync } from 'node:child_process';
import {
  makeRunId, validateEstimate, q, resolveStatic, lockCommand, parseQueueList, idleVerdict,
  waitForIdle, parseAgxClients, summarizeSamples, assessRun,
} from '../scripts/kernel_hillclimb/remote/lib.mjs';
import { settleProbe, publish } from '../lab/kernel-hillclimb/probe-contract.mjs';

test('run ids are unique within the same second and shell-safe', () => {
  const now = new Date('2026-11-11T22:07:10.123Z');
  const ids = new Set(Array.from({ length: 2000 }, () => makeRunId('device-probe.html', now)));
  assert.equal(ids.size, 2000);
  for (const id of ids) assert.match(id, /^20261111T220710Z-device-probe-[0-9a-f]{12}$/);
  assert.match(makeRunId('a b/../$(x);', now), /^20261111T220710Z-[A-Za-z0-9_.-]+-[0-9a-f]{12}$/);
});

test('estimate must be a positive number of minutes', () => {
  assert.equal(validateEstimate('5'), '5');
  assert.equal(validateEstimate(2.5), '2.5');
  assert.equal(validateEstimate(undefined), '5');
  for (const bad of ['0', '-1', '', 'abc', '5;rm -rf ~', '$(id)', '1e3', '`id`', '5 6', '100000', true]) {
    assert.throws(() => validateEstimate(bad), /positive number/, `accepted ${JSON.stringify(bad)}`);
  }
  assert.equal(q("it's"), `'it'\\''s'`);
});

test('static server resolves inside the root and refuses symlink escapes', () => {
  const base = realpathSync(mkdtempSync(join(tmpdir(), 'khc-static-')));
  try {
    const root = join(base, 'dist');
    const outside = join(base, 'secret');
    mkdirSync(join(root, 'sub'), { recursive: true });
    mkdirSync(outside);
    writeFileSync(join(root, 'a.mjs'), 'export {}');
    writeFileSync(join(root, 'page.html'), '<p>');
    writeFileSync(join(root, 'sub', 'index.html'), '<i>');
    writeFileSync(join(outside, 'id_rsa'), 'KEY');
    symlinkSync(join(outside, 'id_rsa'), join(root, 'leak'));
    symlinkSync(outside, join(root, 'leakdir'));
    symlinkSync(join(root, 'a.mjs'), join(root, 'alias.mjs'));
    symlinkSync(join(root, 'nope'), join(root, 'dangling'));
    const r = (u) => resolveStatic(root, u);
    assert.equal(r('/a.mjs').file, join(root, 'a.mjs'));
    assert.equal(r('/a.mjs?x=1').size, 9);
    assert.equal(r('/page').file, join(root, 'page.html'));
    assert.equal(r('/sub/').file, join(root, 'sub', 'index.html'));
    assert.equal(r('/alias.mjs').file, join(root, 'a.mjs')); // inside-root symlink is fine
    assert.equal(r('/leak').status, 403);
    assert.equal(r('/leakdir/id_rsa').status, 403);
    // URL normalisation resolves (encoded) dot segments inside the root: never served.
    assert.equal(r('/../secret/id_rsa').status, 404);
    assert.equal(r('/%2e%2e/secret/id_rsa').status, 404);
    assert.equal(r('/sub/%2e%2e/%2e%2e/secret/id_rsa').status, 404);
    assert.equal(r('/dangling').status, 404);
    assert.equal(r('/missing.wasm').status, 404);
    assert.equal(r('/%E0%A4%A').status, 400);
  } finally { rmSync(base, { recursive: true, force: true }); }
});

test('lockCommand uses kernel advisory locks that exit 75 on timeout', () => {
  assert.deepEqual(lockCommand('darwin', '/h/.host.lockf', 21600), ['lockf', '-k', '-t', '21600', '/h/.host.lockf']);
  assert.deepEqual(lockCommand('linux', '/h/.host.lockf', 30.9), ['flock', '-E', '75', '-w', '30', '/h/.host.lockf']);
});

const lockTool = platform() === 'darwin' ? 'lockf' : 'flock';
const haveLockTool = spawnSync('sh', ['-c', `command -v ${lockTool}`]).status === 0;

test('host lock serializes, times out with 75, and is reclaimed when the holder dies', { skip: !haveLockTool && `${lockTool} not installed` }, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'khc-lock-'));
  const file = join(dir, '.host.lockf');
  const marker = join(dir, 'holder-running');
  try {
    // Holder takes the lock, then dies by SIGKILL (no cleanup code runs).
    const holder = spawn(...splitCmd([...lockCommand(platform(), file, 0), 'sh', '-c', `touch '${marker}'; sleep 1; kill -9 $$`]), { stdio: 'ignore' });
    const holderExit = new Promise((r) => holder.on('exit', r));
    while (!existsSync(marker)) await new Promise((r) => setTimeout(r, 20));
    const busy = spawnSync(...splitCmd([...lockCommand(platform(), file, 0), 'true']));
    assert.equal(busy.status, 75, 'contender must get 75 while the lock is held');
    const waited = spawnSync(...splitCmd([...lockCommand(platform(), file, 10), 'true']));
    assert.equal(waited.status, 0, 'lock must be free once the holder was killed');
    await holderExit;
  } finally { rmSync(dir, { recursive: true, force: true }); }
});
function splitCmd(argv) { return [argv[0], argv.slice(1)]; }

test('gpu-queue list parsing finds queued 0x88 jobs and exit levels', () => {
  const txt = `ID   State      Output               E-Level  Times(r/u/s)   Command [run=1/1]
588  queued     (file)                                       [0x88:device-probe:3m]/home/ddbb/.local/bin/gpu-queue _exec /x/job.log -- /x/job.sh
584  running    /tmp/ts-out.vYlv3N                           [tiny-leela:tl2-s4fix-m:35m]/home/ddbb/.local/bin/gpu-queue _exec a -- b
10   finished   /tmp/ts-out.liAdZ3   1        361.95/350.17/37.72 [tiny-leela:tl2-preflight2:30m]/home/ddbb/.local/bin/gpu-queue _exec c -- d
99   finished   /tmp/ts-out.x        75       0.01/0.00/0.00 [0x88:p:2.5m]/home/ddbb/.local/bin/gpu-queue _exec e -- f`;
  const rows = parseQueueList(txt);
  assert.deepEqual(rows.map((r) => [r.id, r.state, r.elevel, r.label]), [
    ['588', 'queued', null, '[0x88:device-probe:3m]'],
    ['584', 'running', null, '[tiny-leela:tl2-s4fix-m:35m]'],
    ['10', 'finished', '1', '[tiny-leela:tl2-preflight2:30m]'],
    ['99', 'finished', '75', '[0x88:p:2.5m]'],
  ]);
});

test('idle gate never treats an unknown GPU reading as idle', () => {
  const th = { loadMax: 3, gpuMax: 10 };
  assert.equal(idleVerdict({ load1: 1, gpuUtilPct: 0 }, th).ok, true);
  assert.equal(idleVerdict({ load1: 1, gpuUtilPct: null }, th).ok, false);
  assert.equal(idleVerdict({ load1: 1, gpuUtilPct: undefined }, th).ok, false);
  assert.equal(idleVerdict({ load1: 1, gpuUtilPct: NaN }, th).ok, false);
  assert.equal(idleVerdict({ load1: 3, gpuUtilPct: 0 }, th).ok, false);
  assert.equal(idleVerdict({ load1: 1, gpuUtilPct: 10 }, th).ok, false);
});

test('page error contract: failures never publish a partial ok result', () => {
  const good = { webgpu: { available: true }, bench: { fmaF32: {} } };
  assert.deepEqual(settleProbe({ result: good, benchRequested: true }), { ok: true, result: good });
  const cases = [
    [{ benchError: new Error('createBuffer OOM') }, /benchmark failed/],
    [{ gpuErrors: ['GPUValidationError: bad bind group'] }, /WebGPU errors/],
    [{ deviceLost: 'unknown: TDR' }, /device lost/],
    [{ result: { webgpu: { available: false, reason: 'requestAdapter() returned null' } } }, /no WebGPU adapter/],
    [{ result: { webgpu: { available: true, deviceError: 'OperationError' } } }, /requestDevice failed/],
    [{ result: { webgpu: { available: true } } }, /no result/],
  ];
  for (const [over, re] of cases) {
    const o = settleProbe({ result: good, benchRequested: true, ...over });
    assert.equal(o.ok, false); assert.match(o.error, re);
    const win = {}; publish(win, o);
    assert.equal(win.__kernelBenchResult, undefined); assert.match(win.__kernelBenchError, re);
  }
  // Adapter-only probe (bench=0): "no adapter" is a finding, not an error.
  const none = { webgpu: { available: false, reason: 'navigator.gpu missing' } };
  assert.equal(settleProbe({ result: none, benchRequested: false }).ok, true);
});

// ---- idle wait deadline (review r2 #3) ----
function fakeClock(start = 1_000_000) {
  let t = start; const sleeps = [];
  return { now: () => t, sleep: async (ms) => { sleeps.push(ms); t += ms; }, sleeps, get t() { return t; } };
}
const IDLE_TH = { loadMax: 3, gpuMax: 10 };
const idleRead = () => ({ load1: 0.5, gpuUtilPct: 0 });

test('idle hold that would complete after the deadline gives busy, not a start', async () => {
  const c = fakeClock();
  const r = await waitForIdle({ read: idleRead, th: IDLE_TH, deadline: c.t + 59_000, holdMs: 60_000, pollMs: 30_000, now: c.now, sleep: c.sleep });
  assert.equal(r.ok, false);
  assert.ok(c.t <= 1_000_000 + 59_001, `polled past the deadline: ${c.t}`);
});

test('idle hold completing at or before the deadline is accepted', async () => {
  const c = fakeClock();
  const r = await waitForIdle({ read: idleRead, th: IDLE_TH, deadline: c.t + 61_000, holdMs: 60_000, pollMs: 30_000, now: c.now, sleep: c.sleep });
  assert.equal(r.ok, true);
  assert.equal(r.waitedMs, 60_000);
});

test('sleeps are bounded by the remaining wait budget', async () => {
  const c = fakeClock(); const deadline = c.t + 45_000;
  const r = await waitForIdle({ read: () => ({ load1: 9, gpuUtilPct: 0 }), th: IDLE_TH, deadline, holdMs: 60_000, pollMs: 30_000, now: c.now, sleep: c.sleep });
  assert.equal(r.ok, false); assert.match(r.reason, /load1=9/);
  assert.deepEqual(c.sleeps, [30_000, 15_001]);
  assert.equal(c.t, deadline + 1);
});

test('a busy reading resets the hold', async () => {
  const c = fakeClock(); let n = 0;
  const read = () => (n++ === 3 ? { load1: 5, gpuUtilPct: 0 } : idleRead());
  const r = await waitForIdle({ read, th: IDLE_TH, deadline: c.t + 600_000, holdMs: 60_000, pollMs: 30_000, now: c.now, sleep: c.sleep });
  assert.equal(r.ok, true);
  assert.ok(r.waitedMs >= 30_000 + 30_000 + 60_000, `hold not reset: ${r.waitedMs}`);
});

// ---- sampling failures and GPU attribution (review r2 #1, #2) ----
const AGX = `+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100000b6a, !registered, !matched, active, busy 0, retain 5>
      "AppUsage" = ({"API"="Metal","lastSubmittedTime"=69507635623541,"accumulatedGPUTime"=3085876989458})
      "IOUserClientCreator" = "pid 170, WindowServer"
+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100000cff, !registered, !matched, active, busy 0, retain 5>
      "AppUsage" = ()
      "IOUserClientCreator" = "pid 176, runningboardd"
+-o AGXDeviceUserClient  <class AGXDeviceUserClient, id 0x100000d0e, !registered, !matched, active, busy 0, retain 5>
      "AppUsage" = ({"API"="Metal","lastSubmittedTime"=0,"accumulatedGPUTime"=5},{"API"="Metal","lastSubmittedTime"=69506154736541,"accumulatedGPUTime"=10})
      "IOUserClientCreator" = "pid 174, loginwindow"`;

test('ioreg AGX client parsing sums AppUsage per client', () => {
  assert.deepEqual(parseAgxClients(AGX), [
    { id: '0x100000b6a', pid: 170, gpuNs: 3085876989458 },
    { id: '0x100000cff', pid: 176, gpuNs: 0 },
    { id: '0x100000d0e', pid: 174, gpuNs: 15 },
  ]);
  assert.deepEqual(parseAgxClients('garbage'), []);
});

const okCpu = (t, cores = 0.2) => ({ t, load1: 1, cpuOk: true, foreignCpuCores: cores });
const mac = (t, clients, extra = {}) => ({ ...okCpu(t), gpuOk: true, foreignGpuClients: clients, ...extra });
const quiet = { loadavg: [1], gpu: { deviceUtilPct: 0 } };
const macRun = (samples) => assessRun({ platform: 'darwin', samples, loadStart: quiet, loadEnd: quiet, gated: true, th: IDLE_TH });

test('macOS: a foreign Metal job during the run is contended, not clean', () => {
  // 0 -> 10 s; a client that appears mid-run and accrues 5 s of GPU time = 50%.
  const s = [mac(0, { 'a/1': 100 }), mac(5_000, { 'a/1': 100, 'b/2': 2e9 }), mac(10_000, { 'a/1': 100, 'b/2': 5e9 })];
  const r = macRun(s);
  assert.equal(r.during.foreignGpuPct, 50);
  assert.match(r.contendedReasons.join('|'), /foreign GPU 50% >= 10%/);
  assert.deepEqual(r.unverifiedReasons, []);
  const clean = macRun([mac(0, { 'a/1': 100 }), mac(10_000, { 'a/1': 100 + 1e6 })]);
  assert.deepEqual([clean.contendedReasons, clean.unverifiedReasons], [[], []]);
});

test('any failed during-run sample makes the run unverified', () => {
  const good = [mac(0, { 'a/1': 0 }), mac(1_000, { 'a/1': 0 }), mac(2_000, { 'a/1': 0 })];
  assert.deepEqual(macRun(good).unverifiedReasons, []);
  // One CPU failure among successes (previously dropped silently).
  const cpuGap = [good[0], { ...good[1], cpuOk: false, foreignCpuCores: null }, good[2]];
  assert.match(macRun(cpuGap).unverifiedReasons.join('|'), /1\/3 during-run CPU samples failed/);
  // One GPU attribution failure.
  const gpuGap = [good[0], { ...good[1], gpuOk: false, foreignGpuClients: undefined }, good[2]];
  assert.match(macRun(gpuGap).unverifiedReasons.join('|'), /1\/3 during-run GPU attribution samples failed/);
  // No usable GPU attribution at all on macOS.
  const noGpu = good.map((x) => ({ ...x, gpuOk: false, foreignGpuClients: undefined }));
  assert.match(macRun(noGpu).unverifiedReasons.join('|'), /no attributable GPU reading/);
});

test('linux: every nvidia-smi failure is unverified, foreign compute apps are contended', () => {
  const lx = (t, apps, ok = true) => ({ ...okCpu(t), gpuOk: ok, ...(ok ? { foreignGpuApps: apps } : {}) });
  const run = (samples, start = { loadavg: [9], gpu: { utilGpuPct: 0, computeApps: [] } }) =>
    assessRun({ platform: 'linux', samples, loadStart: start, loadEnd: start, gated: false, th: IDLE_TH });
  const clean = run([lx(0, []), lx(1000, [])]);
  assert.deepEqual([clean.contendedReasons, clean.unverifiedReasons], [[], []]); // loadavg ignored when not gated
  assert.match(run([lx(0, [], false), lx(1000, [], false)]).unverifiedReasons.join('|'), /2\/2 during-run GPU attribution/);
  assert.match(run([lx(0, []), lx(1000, ['4242 python 900'])]).contendedReasons.join('|'), /4242 python/);
  assert.match(run([lx(0, []), lx(1000, [])], { loadavg: [0], gpu: { error: 'nvidia-smi: not found' } }).unverifiedReasons.join('|'), /gpu util unknown at start: nvidia-smi/);
  assert.match(run([lx(0, [])]).unverifiedReasons.join('|'), /only 1 during-run samples/);
});

test('summarizeSamples ignores failed samples for maxima but counts them', () => {
  const d = summarizeSamples([okCpu(0, 1.5), { t: 1, load1: 2, cpuOk: false, foreignCpuCores: null, gpuOk: false }]);
  assert.equal(d.foreignCpuCoresMax, 1.5); assert.equal(d.cpuFailures, 1); assert.equal(d.gpuFailures, 2);
});
