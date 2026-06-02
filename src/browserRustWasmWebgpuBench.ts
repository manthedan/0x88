import * as ort from './nn/ortRuntime.ts';

type Mode = 'js-direct-zero' | 'rust-wasm-memory-view' | 'rust-wasm-copy-each-run';
type Row = { batch: number; mode: Mode; iters: number; totalMs: number; msPerRun: number; overheadPct?: number; checksum: number };
type Result = { startedAt: string; finishedAt?: string; model: string; meta: string; wasm: string; ortEp: string; diagnostics?: ort.OrtRuntimeDiagnostics; rows: Row[]; errors: string[] };
type Meta = { token_features?: number; history_plies?: number; onnx_fixed_legal_moves?: number; max_legal_moves?: number };
type BridgeExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  token_ptr: () => number;
  legal_ptr: () => number;
  max_batch: () => number;
  fill_batch: (batch: number, tokenStride: number, legalWidth: number) => void;
};

declare global {
  interface Window {
    tinyLeelaRustWasmWebgpuBench?: { run: () => Promise<Result>; state: Result };
  }
}

const params = new URLSearchParams(location.search);
const modelPath = params.get('onnx') ?? '/models/bt4_sampled1b_best.onnx';
const metaPath = params.get('meta') ?? '/models/bt4_sampled1b_best.meta.json';
const wasmPath = params.get('wasm') ?? '/rust_bridge/tl_rust_bridge.wasm';
const statusEl = document.getElementById('status')!;
const diagnosticsEl = document.getElementById('diagnostics')!;
const rowsEl = document.getElementById('rows')!;
const jsonEl = document.getElementById('json')!;

function intParam(name: string, fallback: number): number {
  const raw = params.get(name);
  const n = raw === null ? fallback : Number(raw);
  return Number.isFinite(n) ? Math.max(1, Math.floor(n)) : fallback;
}

function batchList(): number[] {
  return (params.get('batches') ?? '1,8,16,32,64')
    .split(',')
    .map((x) => Math.max(1, Math.floor(Number(x))))
    .filter((x) => Number.isFinite(x));
}

async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}

async function loadBridge(url: string): Promise<BridgeExports> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  const bytes = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exports = instance.exports as BridgeExports;
  if (!exports.memory || !exports.token_ptr || !exports.legal_ptr || !exports.fill_batch) {
    throw new Error(`WASM bridge missing required exports: ${Object.keys(exports).join(',')}`);
  }
  return exports;
}

function makeJsFeeds(batch: number, tokenStride: number, legalWidth: number): { tokens: BigInt64Array; legal: BigInt64Array; tokenShape: [number, number, number]; legalShape: [number, number] } {
  const tokens = new BigInt64Array(batch * 64 * tokenStride);
  const legal = new BigInt64Array(batch * legalWidth);
  // Deterministic non-zero-ish toy encoding. Values do not need to be legal chess positions for inference throughput.
  for (let b = 0; b < batch; b++) {
    for (let sq = 0; sq < 64; sq++) {
      const base = (b * 64 + sq) * tokenStride;
      for (let f = 0; f < tokenStride; f++) tokens[base + f] = BigInt((sq + f + b) & 15);
    }
    for (let j = 0; j < legalWidth; j++) legal[b * legalWidth + j] = BigInt((j * 37) % 20480);
  }
  return { tokens, legal, tokenShape: [batch, 64, tokenStride], legalShape: [batch, legalWidth] };
}

function makeWasmViews(wasm: BridgeExports, batch: number, tokenStride: number, legalWidth: number): { tokens: BigInt64Array; legal: BigInt64Array; tokenShape: [number, number, number]; legalShape: [number, number] } {
  wasm.fill_batch(batch, tokenStride, legalWidth);
  const tokens = new BigInt64Array(wasm.memory.buffer, wasm.token_ptr(), batch * 64 * tokenStride);
  const legal = new BigInt64Array(wasm.memory.buffer, wasm.legal_ptr(), batch * legalWidth);
  return { tokens, legal, tokenShape: [batch, 64, tokenStride], legalShape: [batch, legalWidth] };
}

function tensorFeeds(src: { tokens: BigInt64Array; legal: BigInt64Array; tokenShape: [number, number, number]; legalShape: [number, number] }): Record<string, ort.Tensor> {
  return {
    tokens: new ort.Tensor('int64', src.tokens, src.tokenShape),
    legal_action_ids: new ort.Tensor('int64', src.legal, src.legalShape),
  };
}

function checksum(outputs: Record<string, ort.Tensor>): number {
  let sum = 0;
  for (const name of ['q', 'wdl', 'action_values']) {
    const data = outputs[name]?.data;
    if (data instanceof Float32Array) for (let i = 0; i < Math.min(data.length, 16); i++) sum += data[i] * (i + 1);
  }
  return sum;
}

async function benchMode(session: ort.InferenceSession, wasm: BridgeExports, mode: Mode, batch: number, tokenStride: number, legalWidth: number, iters: number, warmup: number): Promise<Row> {
  const js = makeJsFeeds(batch, tokenStride, legalWidth);
  const wasmViews = makeWasmViews(wasm, batch, tokenStride, legalWidth);
  let acc = 0;
  for (let i = 0; i < warmup; i++) {
    const src = mode === 'js-direct-zero' ? js : wasmViews;
    acc += checksum(await session.run(tensorFeeds(src)));
  }
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    let src = mode === 'js-direct-zero' ? js : wasmViews;
    if (mode === 'rust-wasm-copy-each-run') {
      src = { tokens: new BigInt64Array(wasmViews.tokens), legal: new BigInt64Array(wasmViews.legal), tokenShape: wasmViews.tokenShape, legalShape: wasmViews.legalShape };
    }
    acc += checksum(await session.run(tensorFeeds(src)));
  }
  const totalMs = performance.now() - t0;
  return { batch, mode, iters, totalMs, msPerRun: totalMs / iters, checksum: acc };
}

function diagnosticSummary(diag?: ort.OrtRuntimeDiagnostics): string {
  if (!diag) return 'diagnostics pending';
  const adapter = diag.adapter ? (diag.adapter.ok ? diag.adapter.summary ?? 'adapter ok' : `adapter unavailable: ${diag.adapter.error}`) : 'adapter not probed';
  const attempts = diag.sessionAttempts.map((a) => `${a.providers.join('+')}:${a.ok ? 'ok' : 'fail'}:${a.ms.toFixed(0)}ms`).join(' | ') || 'none';
  return `requested=${diag.requestedEp} resolved=${diag.resolvedExecutionProviders.join(',')} webgpu=${diag.webgpuAvailable ? 1 : 0} wasmThreads=${diag.wasm.numThreads ?? '?'} adapter=${adapter} sessions=${attempts}`;
}

function render(state: Result) {
  diagnosticsEl.textContent = diagnosticSummary(state.diagnostics);
  const directByBatch = new Map(state.rows.filter((r) => r.mode === 'js-direct-zero').map((r) => [r.batch, r.msPerRun]));
  rowsEl.innerHTML = state.rows.map((r) => {
    const base = directByBatch.get(r.batch);
    const overhead = r.mode === 'js-direct-zero' || !base ? '—' : `${(((r.msPerRun / base) - 1) * 100).toFixed(1)}%`;
    return `<tr><td>${r.batch}</td><td>${r.mode}</td><td>${r.iters}</td><td>${r.msPerRun.toFixed(3)}</td><td>${overhead}</td><td>${r.checksum.toFixed(4)}</td></tr>`;
  }).join('');
  jsonEl.textContent = JSON.stringify(state, null, 2);
}

const state: Result = { startedAt: new Date().toISOString(), model: modelPath, meta: metaPath, wasm: wasmPath, ortEp: params.get('ortEp') ?? params.get('ep') ?? 'webgpu', rows: [], errors: [] };

async function run(): Promise<Result> {
  state.startedAt = new Date().toISOString();
  state.finishedAt = undefined;
  state.rows = [];
  state.errors = [];
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  const [meta, wasm] = await Promise.all([loadJson<Meta>(metaPath), loadBridge(wasmPath)]);
  const tokenStride = Number(meta.token_features ?? ((meta.history_plies ?? 8) + 9));
  const legalWidth = Number(meta.onnx_fixed_legal_moves ?? meta.max_legal_moves ?? 128);
  const maxBatch = wasm.max_batch ? wasm.max_batch() : 64;
  statusEl.textContent = `creating ORT session; wasm max batch=${maxBatch}`;
  const session = await ort.createOrtSession(modelPath);
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  const iters = intParam('iters', 20);
  const warmup = intParam('warmup', 3);
  for (const batch of batchList()) {
    if (batch > maxBatch) {
      state.errors.push(`batch=${batch}: exceeds wasm max_batch=${maxBatch}`);
      continue;
    }
    const rowStart = state.rows.length;
    for (const mode of ['js-direct-zero', 'rust-wasm-memory-view', 'rust-wasm-copy-each-run'] as Mode[]) {
      statusEl.textContent = `running batch=${batch} mode=${mode}`;
      try {
        state.rows.push(await benchMode(session, wasm, mode, batch, tokenStride, legalWidth, iters, warmup));
      } catch (err) {
        state.errors.push(`batch=${batch} mode=${mode}: ${err instanceof Error ? err.stack ?? err.message : String(err)}`);
      }
      render(state);
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    const base = state.rows[rowStart]?.mode === 'js-direct-zero' ? state.rows[rowStart].msPerRun : undefined;
    if (base) for (const row of state.rows.slice(rowStart)) if (row.mode !== 'js-direct-zero') row.overheadPct = ((row.msPerRun / base) - 1) * 100;
  }
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  state.finishedAt = new Date().toISOString();
  statusEl.textContent = `done: ${state.rows.length} rows, ${state.errors.length} errors`;
  render(state);
  return state;
}

window.tinyLeelaRustWasmWebgpuBench = { run, state };
(document.getElementById('run') as HTMLButtonElement).onclick = () => { void run(); };
(document.getElementById('copyJson') as HTMLButtonElement).onclick = () => navigator.clipboard?.writeText(JSON.stringify(state, null, 2));
if (params.get('autorun') === '1') void run();
