#!/usr/bin/env node
// Node/worker baseline of the real browser adapter and shipped WASI worker.
// Run timing only under the project's shared benchlock. No engine builds/fetches.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { Worker as NodeWorker } from 'node:worker_threads';

const args = process.argv.slice(2);
function option(name, fallback) {
  const i = args.indexOf(`--${name}`);
  return i < 0 ? fallback : args[i + 1];
}
const repeats = Number(option('repeat', '5'));
const depth = Number(option('depth', '12'));
const hashMb = Number(option('hash', '128'));
const cancelDepth = Number(option('cancel-depth', String(depth)));
assert(Number.isSafeInteger(repeats) && repeats > 0);
assert(Number.isSafeInteger(depth) && depth > 0);
assert(Number.isSafeInteger(hashMb) && hashMb > 0);
assert(Number.isSafeInteger(cancelDepth) && cancelDepth >= depth);
const wasmPath = path.resolve(option('wasm', 'public/viridithas/viridithas-relaxed-simd128.wasm'));
const outputPath = path.resolve(option('out', '.local_cpu_cancel_probe.json'));
const comparePath = option('compare-wasm', null);
const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'cpu-cancel-probe-'));
const workerPath = path.join(dir, 'worker.ts');
const now = () => performance.timeOrigin + performance.now();
let currentSearch = null;
let workerSeq = 0;
let currentRun = null;
const workers = new Set();
class BrowserWorker {
  constructor() {
    this.id = ++workerSeq;
    currentRun.workerIds.push(this.id);
    this.worker = new NodeWorker(workerPath);
    workers.add(this);
    this.worker.on('message', (data) => {
      if (data.type === 'probe-instantiate') {
        currentRun.instantiates.push({ workerId: this.id, at: data.at });
        return;
      }
      if (data.type === 'probe-info' || (data.type === 'persistent-line' && data.line?.startsWith('info depth '))) {
        if (currentSearch) {
          if (currentSearch.firstInfoMs === null) currentSearch.firstInfoMs = now() - currentSearch.startedAt;
          currentSearch.finalInfoLine = data.line;
        }
      }
      if (currentSearch && data.type === 'persistent-line' && data.line?.startsWith('bestmove ')) currentSearch.bestmove = data.line;
      if (currentSearch && data.type === 'result') currentSearch.bestmove = data.stdout.find((line) => line.startsWith('bestmove '));
      if (data.type === 'probe-info') return;
      this.onmessage?.({ data });
    });
    this.worker.on('error', (error) => this.onerror?.({ message: error.message }));
  }
  postMessage(data) {
    this.worker.postMessage(data);
  }
  terminate() {
    currentRun.terminatedWorkerIds.push(this.id);
    workers.delete(this);
    // Awaited during cleanup, including synchronous adapter aborts.
    terminations.push(this.worker.terminate());
  }
}
const terminations = [];
globalThis.Worker = BrowserWorker;
globalThis.crossOriginIsolated = true;

const startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const positions = [[], ['e2e4'], ['e2e4', 'e7e5']];
const runs = [];
try {
  let workerSource = await fs.readFile('src/lc0/recklessWasiWorker.ts', 'utf8');
  const old = 'lineCollector(stdout, undefined, isUsefulUciStdoutLine)';
  assert(workerSource.includes(old), 'one-shot instrumentation anchor changed');
  workerSource = workerSource.replace(
    old,
    "lineCollector(stdout, (line) => { if (line.startsWith('info depth ')) self.postMessage({ type: 'probe-info', line }); }, isUsefulUciStdoutLine)",
  );
  const bootstrap = `import { parentPort } from 'node:worker_threads';
import fs from 'node:fs/promises';
globalThis.self = { postMessage: (data) => parentPort.postMessage(data), addEventListener: (_type, fn) => parentPort.on('message', (data) => fn({ data })) };
globalThis.fetch = async (url) => new Response(await fs.readFile(url), { headers: { 'Content-Type': 'application/wasm' } });
const instantiate = WebAssembly.instantiate;
WebAssembly.instantiate = async (...args) => { const result = await instantiate(...args); parentPort.postMessage({ type: 'probe-instantiate', at: performance.timeOrigin + performance.now() }); return result; };
`;
  workerSource = workerSource.replace("'@bjorn3/browser_wasi_shim'", JSON.stringify(import.meta.resolve('@bjorn3/browser_wasi_shim')));
  await fs.writeFile(workerPath, `${bootstrap}\n${workerSource}`);
  const { ViridithasEngine } = await import('../src/lc0/viridithasEngine.ts');
  // Untimed warmup; its data is retained separately, not mixed into samples.
  const orders = comparePath
    ? [
        ['baseline', false],
        ['candidate', false],
      ]
    : [
        ['oneshot', false],
        ['persistent', false],
        ['oneshot', true],
        ['persistent', true],
      ];
  for (let repeat = -1; repeat < repeats; repeat++) {
    // ABAB paired runs; reverse order every other repetition to reduce drift bias.
    const order = repeat % 2 === 1 ? [...orders].reverse() : orders;
    for (const [mode, cancel] of order) {
      currentRun = { repeat, mode, cancel, workerIds: [], terminatedWorkerIds: [], instantiates: [], searches: [] };
      const selectedWasm = comparePath && mode === 'baseline' ? path.resolve(comparePath) : wasmPath;
      const engine = new ViridithasEngine({ depth, hashMb }, selectedWasm, { forceOneShot: mode === 'oneshot', disablePersistentFallback: true });
      const startedAt = now();
      try {
        for (let index = 0; index < positions.length; index++) {
          const controller = new AbortController();
          const row = { index, moves: positions[index], startedAt: now(), firstInfoMs: null, aborted: false };
          currentSearch = row;
          const timer =
            cancel && index === 1
              ? setTimeout(() => {
                  row.abortAt = now();
                  controller.abort();
                }, 200)
              : null;
          try {
            const searchDepth = timer ? cancelDepth : depth;
            row.info = await engine.analyze(startFen, { depth: searchDepth, moves: positions[index], signal: controller.signal });
            row.cancelMissed = Boolean(timer);
            assert.equal(row.info[0]?.depth, depth);
          } catch (error) {
            if (!timer || error.name !== 'AbortError') throw error;
            row.aborted = true;
            row.abortRejectMs = now() - row.abortAt;
          } finally {
            clearTimeout(timer);
            row.wallMs = now() - row.startedAt;
            currentSearch = null;
          }
          currentRun.searches.push(row);
        }
        currentRun.wallMs = now() - startedAt;
        currentRun.instantiateCount = currentRun.instantiates.length;
        currentRun.runtime = engine.runtimeStatus();
      } finally {
        engine.dispose();
        await Promise.all(terminations.splice(0));
      }
      runs.push(currentRun);
      console.error(
        JSON.stringify({
          repeat,
          mode,
          cancel,
          wallMs: currentRun.wallMs,
          instantiateCount: currentRun.instantiateCount,
          firstInfoMs: currentRun.searches.map((row) => row.firstInfoMs),
        }),
      );
    }
  }
  const bytes = await fs.readFile(wasmPath);
  const report = {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: os.cpus()[0]?.model,
    depth,
    cancelDepth,
    hashMb,
    cancelMs: 200,
    wasmPath,
    comparePath,
    wasmBytes: bytes.length,
    wasmSha256: createHash('sha256').update(bytes).digest('hex'),
    note: 'Actual adapters; local file fetch instead of network. One-shot first-info is instrumentation only (the product adapter returns output at completion). Cancel is on the middle request, after a completed search, to test a warm runtime. Each sequence begins cold; TT survives as in the Analysis continuity path. repeat=-1 is untimed warmup excluded from statistics.',
    runs,
  };
  await fs.writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  console.log(outputPath);
} finally {
  for (const worker of workers) worker.terminate();
  await Promise.all(terminations);
  await fs.rm(dir, { recursive: true, force: true });
}
