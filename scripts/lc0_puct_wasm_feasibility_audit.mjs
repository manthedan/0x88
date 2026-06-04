import { writeFileSync } from 'node:fs';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';

function parseArgs(argv) {
  const args = {
    fen: START_FEN,
    visits: [64, 256, 1024],
    batches: [1, 4, 8],
    iters: 3,
    warmup: 1,
    out: undefined,
  };
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    const readValue = () => {
      if (arg.includes('=')) return arg.slice(arg.indexOf('=') + 1);
      i += 1;
      if (i >= argv.length) throw new Error(`Missing value for ${arg}`);
      return argv[i];
    };
    if (arg === '--fen' || arg.startsWith('--fen=')) args.fen = readValue();
    else if (arg === '--visits' || arg.startsWith('--visits=')) args.visits = readValue().split(',').map((v) => Number(v.trim())).filter(Number.isFinite);
    else if (arg === '--batches' || arg.startsWith('--batches=')) args.batches = readValue().split(',').map((v) => Number(v.trim())).filter(Number.isFinite);
    else if (arg === '--iters' || arg.startsWith('--iters=')) args.iters = Number(readValue());
    else if (arg === '--warmup' || arg.startsWith('--warmup=')) args.warmup = Number(readValue());
    else if (arg === '--out' || arg.startsWith('--out=')) args.out = readValue();
    else if (arg === '--help') {
      console.log('Usage: node --experimental-strip-types scripts/lc0_puct_wasm_feasibility_audit.mjs [--visits 64,256,1024] [--batches 1,4,8] [--iters 3] [--warmup 1] [--out /tmp/audit.json]');
      process.exit(0);
    } else throw new Error(`Unknown argument: ${arg}`);
  }
  if (!args.visits.length) throw new Error('--visits must include at least one finite number');
  if (!args.batches.length) throw new Error('--batches must include at least one finite number');
  if (!Number.isFinite(args.iters)) throw new Error('--iters must be a finite number');
  if (!Number.isFinite(args.warmup)) throw new Error('--warmup must be a finite number');
  args.visits = args.visits.map((v) => Math.max(1, Math.floor(v)));
  args.batches = args.batches.map((v) => Math.max(1, Math.floor(v)));
  args.iters = Math.max(1, Math.floor(args.iters));
  args.warmup = Math.max(0, Math.floor(args.warmup));
  return args;
}

function percentile(values, p) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((sorted.length - 1) * p)));
  return sorted[idx];
}

function summarize(values) {
  const mean = values.reduce((a, b) => a + b, 0) / (values.length || 1);
  return { mean, min: Math.min(...values), p50: percentile(values, 0.5), max: Math.max(...values) };
}

class SyntheticLc0SearchEvaluator {
  constructor() {
    this.evaluateCalls = 0;
    this.evaluateBatchCalls = 0;
  }

  build(board, context) {
    const moves = context?.legalMoves ?? legalMoves(board);
    const policy = new Map();
    const fallback = moves.length ? 1 / moves.length : 0;
    for (const move of moves) policy.set(moveToActionId(move), fallback);
    return { policy, wdl: [0.45, 0.10, 0.45] };
  }

  async evaluate(board, context) {
    this.evaluateCalls += 1;
    return this.build(board, context);
  }

  async evaluateBatch(boards, contexts = []) {
    this.evaluateBatchCalls += 1;
    this.evaluateCalls += boards.length;
    return boards.map((board, i) => this.build(board, contexts[i]));
  }
}

async function runRow({ fen, visits, batchSize, iters, warmup }) {
  const times = [];
  const visitsPerSecond = [];
  let lastResult;
  let lastEvaluator;
  const totalRuns = warmup + iters;
  for (let run = 0; run < totalRuns; run++) {
    const evaluator = new SyntheticLc0SearchEvaluator();
    const t0 = performance.now();
    const result = await searchRoot(parseFen(fen), evaluator, {
      visits,
      batchSize,
      includePv: false,
      temperature: 0,
      cpuctSchedule: 'constant',
      fpuStrategy: 'constant',
      batchCollisionMode: 'retry',
      batchCollisionRetryLimit: batchSize * 4,
      yieldEveryMs: 0,
    });
    const elapsedMs = performance.now() - t0;
    if (run >= warmup) {
      times.push(elapsedMs);
      visitsPerSecond.push(result.stats.completedVisits / Math.max(1e-9, elapsedMs / 1000));
      lastResult = result;
      lastEvaluator = evaluator;
    }
  }
  return {
    visits,
    batchSize,
    elapsedMs: summarize(times),
    visitsPerSecond: summarize(visitsPerSecond),
    completedVisits: lastResult?.stats.completedVisits ?? 0,
    expansions: lastResult?.stats.expansions ?? 0,
    evalCalls: lastResult?.stats.evalCalls ?? 0,
    batchEvalCalls: lastResult?.stats.batchEvalCalls ?? 0,
    maxEvalBatch: lastResult?.stats.maxEvalBatch ?? 0,
    evalBatchSizeHistogram: lastResult?.stats.evalBatchSizeHistogram ?? {},
    syntheticEvaluateCalls: lastEvaluator?.evaluateCalls ?? 0,
    syntheticEvaluateBatchCalls: lastEvaluator?.evaluateBatchCalls ?? 0,
  };
}

const args = parseArgs(process.argv.slice(2));
const rows = [];
for (const visits of args.visits) {
  for (const batchSize of args.batches) {
    rows.push(await runRow({ fen: args.fen, visits, batchSize, iters: args.iters, warmup: args.warmup }));
  }
}

const fastest = rows.reduce((best, row) => !best || row.visitsPerSecond.mean > best.visitsPerSecond.mean ? row : best, undefined);
const output = {
  generatedAt: new Date().toISOString(),
  mode: 'synthetic-js-puct-zero-neural-latency',
  fen: args.fen,
  iters: args.iters,
  warmup: args.warmup,
  rows,
  audit: {
    fastestMeanVisitsPerSecond: fastest?.visitsPerSecond.mean ?? 0,
    fastestRow: fastest ? { visits: fastest.visits, batchSize: fastest.batchSize } : undefined,
    interpretation: [
      'This isolates the current JavaScript PUCT/tree/movegen path with a zero-latency synthetic evaluator.',
      'Any SIMD-WASM PUCT rewrite can only improve this CPU-side envelope, while real browser LC0 search remains dominated by neural/WebGPU evaluation and host/GPU synchronization until those costs drop.',
      'PUCT is branchy tree/graph bookkeeping; SIMD is not an obvious fit unless the tree is first moved to packed typed-array/WASM-owned memory.',
    ],
  },
};

if (args.out) writeFileSync(args.out, `${JSON.stringify(output, null, 2)}\n`);
console.log(JSON.stringify(output, null, 2));
