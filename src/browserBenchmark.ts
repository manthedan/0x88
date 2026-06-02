import { parseFen, boardToFen, START_FEN, type BoardState } from './chess/board.ts';
import { legalMoves, makeMove } from './chess/movegen.ts';
import { moveToUci, type Move } from './chess/moveCodec.ts';
import { chooseMove, advanceSearchRoot, type Node as PuctNode, type SearchStats } from './search/puct.ts';
import { CachedEvaluator, type CachedEvaluatorMetrics, type Evaluator } from './nn/evaluator.ts';
import { OnnxEvaluator, type OnnxStudentMeta } from './nn/onnxEvaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';
import { collectOrtRuntimeDiagnostics, describeOrtBackendConfig, type OrtRuntimeDiagnostics } from './nn/ortRuntime.ts';

type ModelConfig = { key: string; onnx: string; meta: string; label: string };
type BenchRowConfig = { visits: number; evalCache: boolean; rootReuse: boolean; transpositions: boolean; plies: number; batchSize: number; repeat: number };
type BenchRowResult = BenchRowConfig & {
  index: number;
  model: string;
  label: string;
  requestedEp: string;
  resolvedEp: string;
  evalInputCacheEntries: string;
  totalMs: number;
  searches: number;
  avgMsPerSearch: number;
  moves: string[];
  fens: string[];
  stats: SearchStats[];
  evalCacheMetrics?: CachedEvaluatorMetrics & { entries: number; hitRate: number };
  error?: string;
};
type BenchResult = { startedAt: string; finishedAt?: string; userAgent: string; diagnostics?: OrtRuntimeDiagnostics; rows: BenchRowResult[]; errors: string[] };

declare global {
  interface Window {
    tinyLeelaBrowserBenchmark?: {
      run: (matrix?: string) => Promise<BenchResult>;
      state: BenchResult;
    };
  }
}

const MODELS: Record<string, ModelConfig> = {
  'bt4-sampled1b-best': {
    key: 'bt4-sampled1b-best',
    onnx: '/models/bt4_sampled1b_best.onnx',
    meta: '/models/bt4_sampled1b_best.meta.json',
    label: 'BT4 sampled-1B best',
  },
  'bt4-h8-100m-e5': {
    key: 'bt4-h8-100m-e5',
    onnx: '/models/bt4_h8_100m_e5.onnx',
    meta: '/models/bt4_h8_100m_e5.meta.json',
    label: 'BT4 h8 100M e5',
  },
};

const params = new URLSearchParams(location.search);
const requestedEp = params.get('ortEp') ?? params.get('ep') ?? params.get('executionProviders') ?? 'wasm';
const model = MODELS[params.get('model') ?? 'bt4-sampled1b-best'] ?? MODELS['bt4-sampled1b-best'];
const statusEl = document.getElementById('status')!;
const diagnosticsEl = document.getElementById('diagnostics');
const rowsEl = document.getElementById('rows')!;
const jsonEl = document.getElementById('json')!;

function numParam(name: string, fallback: number): number {
  const raw = params.get(name);
  if (raw === null || raw.trim() === '') return fallback;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function boolParam(name: string, fallback: boolean): boolean {
  const raw = params.get(name);
  if (raw === null) return fallback;
  return raw === '1' || raw === 'true' || raw === 'on' || raw === 'yes';
}

async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}

function isSquareFormerMeta(meta: OnnxStudentMeta | SquareFormerMeta): meta is SquareFormerMeta {
  return meta.kind === 'squareformer' || meta.kind === 'squareformer_v2';
}

async function createBaseEvaluator(): Promise<Evaluator> {
  const meta = await loadJson<OnnxStudentMeta | SquareFormerMeta>(model.meta);
  return isSquareFormerMeta(meta)
    ? await SquareFormerEvaluator.create(model.onnx, meta)
    : await OnnxEvaluator.create(model.onnx, meta);
}

async function createEvaluator(evalCache: boolean): Promise<{ evaluator: Evaluator; cache?: CachedEvaluator }> {
  const base = await createBaseEvaluator();
  if (!evalCache) return { evaluator: base };
  const maxEntries = Math.max(1, Math.floor(numParam('evalCacheEntries', 8192)));
  const cache = new CachedEvaluator(base, { maxEntries, includeHistory: true, includeLegalMoves: true, label: 'browser-bench' });
  return { evaluator: cache, cache };
}

function fixedRows(matrix: string): BenchRowConfig[] {
  const plies = Math.max(1, Math.floor(numParam('plies', matrix === 'full' ? 6 : 4)));
  const batchSize = Math.max(1, Math.floor(numParam('batch', 16)));
  const repeat = Math.max(1, Math.floor(numParam('repeat', 1)));
  if (params.get('row')) {
    return [{
      visits: Math.max(1, Math.floor(numParam('visits', 64))),
      evalCache: boolParam('evalCache', true),
      rootReuse: boolParam('puctRootReuse', true),
      transpositions: boolParam('puctTranspositions', false),
      plies,
      batchSize,
      repeat,
    }];
  }
  const visits = (params.get('visitsList') ?? (matrix === 'full' ? '1,16,64,128' : '1,16,64'))
    .split(',')
    .map((v) => Math.max(1, Math.floor(Number(v))))
    .filter((v) => Number.isFinite(v));
  const rows: BenchRowConfig[] = [];
  for (const v of visits) {
    for (const evalCache of [false, true]) {
      for (const rootReuse of [false, true]) {
        rows.push({ visits: v, evalCache, rootReuse, transpositions: false, plies, batchSize, repeat });
      }
    }
  }
  if (matrix === 'full') rows.push({ visits: 64, evalCache: true, rootReuse: true, transpositions: true, plies, batchSize, repeat });
  return rows;
}

function rowHtml(row: BenchRowResult): string {
  const hit = row.evalCacheMetrics ? `${(row.evalCacheMetrics.hitRate * 100).toFixed(1)}%` : '—';
  const moves = row.moves.join(' ');
  return `<tr><td>${row.index}</td><td>${row.resolvedEp}</td><td>${row.visits}</td><td>${row.evalCache ? 1 : 0}</td><td>${row.rootReuse ? 1 : 0}</td><td>${row.transpositions ? 1 : 0}</td><td>${row.plies}</td><td>${row.totalMs.toFixed(1)}</td><td>${row.avgMsPerSearch.toFixed(1)}</td><td>${row.stats.reduce((s, x) => s + x.evalCalls, 0)}</td><td>${hit}</td><td><code>${moves}</code></td></tr>`;
}

function diagnosticSummary(diag?: OrtRuntimeDiagnostics): string {
  if (!diag) return 'diagnostics pending';
  const attempts = diag.sessionAttempts.map((a) => `${a.providers.join('+')}:${a.ok ? 'ok' : 'fail'}:${a.ms.toFixed(0)}ms`).join(' | ') || 'none';
  const adapter = diag.adapter ? (diag.adapter.ok ? diag.adapter.summary ?? 'adapter ok' : `adapter unavailable: ${diag.adapter.error}`) : 'adapter not probed';
  return `requested=${diag.requestedEp} resolved=${diag.resolvedExecutionProviders.join(',')} webgpu=${diag.webgpuAvailable ? 1 : 0} wasmThreads=${diag.wasm.numThreads ?? '?'} coi=${diag.crossOriginIsolated ? 1 : 0} adapter=${adapter} sessions=${attempts}`;
}

function render(state: BenchResult) {
  if (diagnosticsEl) diagnosticsEl.textContent = diagnosticSummary(state.diagnostics);
  rowsEl.innerHTML = state.rows.map(rowHtml).join('');
  jsonEl.textContent = JSON.stringify(state, null, 2);
}

function appendHistory(historyFens: string[], board: BoardState): string[] {
  return [boardToFen(board), ...historyFens].slice(0, 16);
}

async function runRow(config: BenchRowConfig, index: number): Promise<BenchRowResult> {
  const { evaluator, cache } = await createEvaluator(config.evalCache);
  // Warm session and input preparation outside measured search time.
  let board = parseFen(START_FEN);
  let historyFens: string[] = [];
  await evaluator.evaluate(board, { historyFens, legalMoves: legalMoves(board) });
  let root: PuctNode | null = null;
  const tt = config.transpositions ? new Map<string, PuctNode>() : undefined;
  const stats: SearchStats[] = [];
  const moves: string[] = [];
  const fens = [boardToFen(board)];
  const t0 = performance.now();
  for (let r = 0; r < config.repeat; r++) {
    board = parseFen(START_FEN);
    historyFens = [];
    root = null;
    for (let ply = 0; ply < config.plies; ply++) {
      const result = await chooseMove(board, evaluator, {
        visits: config.visits,
        batchSize: config.batchSize,
        historyFens,
        includePv: false,
        root: config.rootReuse ? root : null,
        transpositionTable: tt,
        yieldEveryMs: 12,
      });
      if (!result.move) break;
      stats.push(result.stats!);
      const move: Move = result.move;
      moves.push(moveToUci(move));
      const nextBoard = makeMove(board, move);
      const nextHistory = appendHistory(historyFens, board);
      root = config.rootReuse ? advanceSearchRoot(result.root ?? root, move, nextBoard, nextHistory) : null;
      board = nextBoard;
      historyFens = nextHistory;
      fens.push(boardToFen(board));
    }
  }
  const totalMs = performance.now() - t0;
  return {
    ...config,
    index,
    model: model.key,
    label: model.label,
    requestedEp,
    resolvedEp: describeOrtBackendConfig(),
    evalInputCacheEntries: params.get('evalInputCacheEntries') ?? '4096',
    totalMs,
    searches: stats.length,
    avgMsPerSearch: stats.length ? totalMs / stats.length : totalMs,
    moves,
    fens,
    stats,
    ...(cache ? { evalCacheMetrics: cache.metrics() } : {}),
  };
}

const state: BenchResult = { startedAt: new Date().toISOString(), userAgent: navigator.userAgent, rows: [], errors: [] };

async function run(matrix = params.get('matrix') ?? 'smoke'): Promise<BenchResult> {
  state.startedAt = new Date().toISOString();
  state.finishedAt = undefined;
  state.diagnostics = await collectOrtRuntimeDiagnostics({ probeAdapter: requestedEp !== 'wasm' });
  state.rows = [];
  state.errors = [];
  render(state);
  const rows = fixedRows(matrix);
  for (let i = 0; i < rows.length; i++) {
    statusEl.textContent = `running ${i + 1}/${rows.length}: visits=${rows[i].visits} cache=${rows[i].evalCache ? 1 : 0} root=${rows[i].rootReuse ? 1 : 0} tt=${rows[i].transpositions ? 1 : 0}`;
    try {
      state.rows.push(await runRow(rows[i], i + 1));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      state.errors.push(message);
      state.rows.push({ ...rows[i], index: i + 1, model: model.key, label: model.label, requestedEp, resolvedEp: describeOrtBackendConfig(), evalInputCacheEntries: params.get('evalInputCacheEntries') ?? '4096', totalMs: 0, searches: 0, avgMsPerSearch: 0, moves: [], fens: [], stats: [], error: message });
    }
    state.diagnostics = await collectOrtRuntimeDiagnostics({ probeAdapter: requestedEp !== 'wasm' });
    render(state);
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  state.diagnostics = await collectOrtRuntimeDiagnostics({ probeAdapter: requestedEp !== 'wasm' });
  state.finishedAt = new Date().toISOString();
  statusEl.textContent = `done: ${state.rows.length} rows, ${state.errors.length} errors`;
  render(state);
  return state;
}

window.tinyLeelaBrowserBenchmark = { run, state };
(document.getElementById('runSmoke') as HTMLButtonElement).onclick = () => { void run('smoke'); };
(document.getElementById('runFull') as HTMLButtonElement).onclick = () => { void run('full'); };
(document.getElementById('copyJson') as HTMLButtonElement).onclick = () => navigator.clipboard?.writeText(JSON.stringify(state, null, 2));
if (params.get('autorun') === '1') void run(params.get('matrix') ?? 'smoke');
