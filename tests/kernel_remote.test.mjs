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
