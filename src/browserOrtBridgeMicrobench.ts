import * as ort from './nn/ortRuntime.ts';
import { START_FEN, parseFen, type BoardState } from './chess/board.ts';
import { legalMoves } from './chess/movegen.ts';
import { squareformerCompactInput, squareformerFloatInput, squareformerLegalCandidateInputs, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';

type Mode = 'direct' | 'linear-memory-view' | 'linear-memory-copy-each-run';
type Row = { batch: number; mode: Mode; iters: number; totalMs: number; msPerRun: number; overheadPct?: number; checksum: number };
type Result = { startedAt: string; finishedAt?: string; model: string; meta: string; ortEp: string; diagnostics?: ort.OrtRuntimeDiagnostics; rows: Row[]; errors: string[] };

declare global {
  interface Window {
    tinyLeelaOrtBridgeMicrobench?: { run: () => Promise<Result>; state: Result };
  }
}

const params = new URLSearchParams(location.search);
const modelPath = params.get('onnx') ?? '/models/bt4_sampled1b_best.onnx';
const metaPath = params.get('meta') ?? '/models/bt4_sampled1b_best.meta.json';
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
  return (params.get('batches') ?? '1,8,16,32')
    .split(',')
    .map((x) => Math.max(1, Math.floor(Number(x))))
    .filter((x) => Number.isFinite(x));
}

async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}

function isCompact(meta: SquareFormerMeta): boolean {
  return meta.input_mode === 'embedding' || meta.input_format === 'compact_uint8_embeddings' || meta.input_format === 'compact_uint8_tokens';
}

function makeBoards(batch: number): BoardState[] {
  return Array.from({ length: batch }, () => parseFen(START_FEN));
}

function makeFeeds(meta: SquareFormerMeta, batch: number): { tokens: BigInt64Array | Float32Array; legal: BigInt64Array; tokenShape: [number, number, number]; legalShape: [number, number] } {
  const boards = makeBoards(batch);
  const compact = isCompact(meta);
  const stride = compact ? meta.token_features ?? meta.history_plies + 9 : meta.input_dim;
  const one = 64 * stride;
  const tokens = compact ? new BigInt64Array(batch * one) : new Float32Array(batch * one);
  for (let i = 0; i < batch; i++) {
    const row = compact ? squareformerCompactInput(boards[i], meta, [], 'int64') as BigInt64Array : squareformerFloatInput(boards[i], meta, []);
    (tokens as BigInt64Array | Float32Array).set(row as never, i * one);
  }
  const width = Math.max(1, Number(meta.onnx_fixed_legal_moves ?? meta.max_legal_moves ?? 128));
  const contexts = boards.map((board) => ({ legalMoves: legalMoves(board), historyFens: [] }));
  const legal = squareformerLegalCandidateInputs(boards, width, contexts, 'int64').classes as BigInt64Array;
  return { tokens, legal, tokenShape: [batch, 64, stride], legalShape: [batch, width] };
}

function copyToLinearMemory<T extends BigInt64Array | Float32Array>(tokens: T, legal: BigInt64Array): { memory: WebAssembly.Memory; tokensView: T; legalView: BigInt64Array } {
  const tokenBytes = tokens.byteLength;
  const legalOffset = Math.ceil(tokenBytes / 8) * 8;
  const totalBytes = legalOffset + legal.byteLength;
  const pages = Math.max(1, Math.ceil(totalBytes / 65536));
  const memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
  const tokensView = tokens instanceof BigInt64Array
    ? new BigInt64Array(memory.buffer, 0, tokens.length) as T
    : new Float32Array(memory.buffer, 0, tokens.length) as T;
  const legalView = new BigInt64Array(memory.buffer, legalOffset, legal.length);
  tokensView.set(tokens as never);
  legalView.set(legal);
  return { memory, tokensView, legalView };
}

function tensorType(data: BigInt64Array | Float32Array): 'int64' | 'float32' {
  return data instanceof BigInt64Array ? 'int64' : 'float32';
}

function makeTensorFeeds(tokens: BigInt64Array | Float32Array, legal: BigInt64Array, tokenShape: [number, number, number], legalShape: [number, number]): Record<string, ort.Tensor> {
  return {
    tokens: new ort.Tensor(tensorType(tokens), tokens as never, tokenShape),
    legal_action_ids: new ort.Tensor('int64', legal, legalShape),
  };
}

function checksum(outputs: Record<string, ort.Tensor>): number {
  let sum = 0;
  for (const name of ['q', 'wdl', 'action_values']) {
    const data = outputs[name]?.data;
    if (data instanceof Float32Array) {
      for (let i = 0; i < Math.min(data.length, 16); i++) sum += data[i] * (i + 1);
    }
  }
  return sum;
}

async function benchMode(session: ort.InferenceSession, mode: Mode, meta: SquareFormerMeta, batch: number, iters: number, warmup: number): Promise<Row> {
  const src = makeFeeds(meta, batch);
  const linear = mode === 'direct' ? null : copyToLinearMemory(src.tokens, src.legal);
  let acc = 0;
  for (let i = 0; i < warmup; i++) {
    const feeds = linear ? makeTensorFeeds(linear.tokensView, linear.legalView, src.tokenShape, src.legalShape) : makeTensorFeeds(src.tokens, src.legal, src.tokenShape, src.legalShape);
    acc += checksum(await session.run(feeds));
  }
  const t0 = performance.now();
  for (let i = 0; i < iters; i++) {
    if (mode === 'linear-memory-copy-each-run' && linear) {
      linear.tokensView.set(src.tokens as never);
      linear.legalView.set(src.legal);
    }
    const feeds = linear ? makeTensorFeeds(linear.tokensView, linear.legalView, src.tokenShape, src.legalShape) : makeTensorFeeds(src.tokens, src.legal, src.tokenShape, src.legalShape);
    acc += checksum(await session.run(feeds));
  }
  const totalMs = performance.now() - t0;
  return { batch, mode, iters, totalMs, msPerRun: totalMs / iters, checksum: acc };
}

function diagnosticSummary(diag?: ort.OrtRuntimeDiagnostics): string {
  if (!diag) return 'diagnostics pending';
  const attempts = diag.sessionAttempts.map((a) => `${a.providers.join('+')}:${a.ok ? 'ok' : 'fail'}:${a.ms.toFixed(0)}ms`).join(' | ') || 'none';
  const adapter = diag.adapter ? (diag.adapter.ok ? diag.adapter.summary ?? 'adapter ok' : `adapter unavailable: ${diag.adapter.error}`) : 'adapter not probed';
  return `requested=${diag.requestedEp} resolved=${diag.resolvedExecutionProviders.join(',')} webgpu=${diag.webgpuAvailable ? 1 : 0} wasmThreads=${diag.wasm.numThreads ?? '?'} adapter=${adapter} sessions=${attempts}`;
}

function render(state: Result) {
  diagnosticsEl.textContent = diagnosticSummary(state.diagnostics);
  const directByBatch = new Map(state.rows.filter((r) => r.mode === 'direct').map((r) => [r.batch, r.msPerRun]));
  rowsEl.innerHTML = state.rows.map((r) => {
    const base = directByBatch.get(r.batch);
    const overhead = r.mode === 'direct' || !base ? '—' : `${(((r.msPerRun / base) - 1) * 100).toFixed(1)}%`;
    return `<tr><td>${r.batch}</td><td>${r.mode}</td><td>${r.iters}</td><td>${r.msPerRun.toFixed(3)}</td><td>${overhead}</td><td>${r.checksum.toFixed(4)}</td></tr>`;
  }).join('');
  jsonEl.textContent = JSON.stringify(state, null, 2);
}

const state: Result = { startedAt: new Date().toISOString(), model: modelPath, meta: metaPath, ortEp: params.get('ortEp') ?? params.get('ep') ?? 'wasm', rows: [], errors: [] };

async function run(): Promise<Result> {
  state.startedAt = new Date().toISOString();
  state.finishedAt = undefined;
  state.rows = [];
  state.errors = [];
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  const meta = await loadJson<SquareFormerMeta>(metaPath);
  statusEl.textContent = 'creating ORT session';
  const session = await ort.createOrtSession(modelPath);
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  const iters = intParam('iters', 20);
  const warmup = intParam('warmup', 3);
  for (const batch of batchList()) {
    const rowStart = state.rows.length;
    for (const mode of ['direct', 'linear-memory-view', 'linear-memory-copy-each-run'] as Mode[]) {
      statusEl.textContent = `running batch=${batch} mode=${mode}`;
      try {
        state.rows.push(await benchMode(session, mode, meta, batch, iters, warmup));
      } catch (err) {
        state.errors.push(`batch=${batch} mode=${mode}: ${err instanceof Error ? err.message : String(err)}`);
      }
      render(state);
      await new Promise((resolve) => setTimeout(resolve, 25));
    }
    const base = state.rows[rowStart]?.mode === 'direct' ? state.rows[rowStart].msPerRun : undefined;
    if (base) for (const row of state.rows.slice(rowStart)) if (row.mode !== 'direct') row.overheadPct = ((row.msPerRun / base) - 1) * 100;
  }
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  state.finishedAt = new Date().toISOString();
  statusEl.textContent = `done: ${state.rows.length} rows, ${state.errors.length} errors`;
  render(state);
  return state;
}

window.tinyLeelaOrtBridgeMicrobench = { run, state };
(document.getElementById('run') as HTMLButtonElement).onclick = () => { void run(); };
(document.getElementById('copyJson') as HTMLButtonElement).onclick = () => navigator.clipboard?.writeText(JSON.stringify(state, null, 2));
if (params.get('autorun') === '1') void run();
