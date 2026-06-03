#!/usr/bin/env node
// Drift/perf sweep across the LC0 fixture suites under Node (ORT WASM).
//
// Baseline: f32/WASM. Candidate: f16/WASM. Reports best-move parity, WDL/Q/MLH
// drift, top-prior drift, and eval/s + visits/s per backend. The f16/WebGPU and
// worker dimensions run only in a browser; this CLI covers the WASM baselines.
//
// Usage: npm run lc0:drift-sweep [f32Model] [f16Model] [visits]
import { existsSync, readFileSync } from 'node:fs';
import { Lc0OnnxEvaluator } from '../src/lc0/onnxEvaluator.ts';
import { Lc0PuctSearcher } from '../src/lc0/search.ts';
import { compareEvalSweeps, runEvalSweep, runSearchSweep } from '../src/lc0/driftSweep.ts';

const F32 = process.argv[2] ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f32.onnx';
const F16 = process.argv[3] ?? '../models/lc0-bestnets/onnx/t1-256x10-distilled-swa-2432500.batch1.f16.onnx';
const VISITS = Number(process.argv[4] ?? 16);

function loadFixtures() {
  const fenOnly = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'));
  const history = JSON.parse(readFileSync('fixtures/lc0/history.json', 'utf8'));
  return [...fenOnly, ...history];
}

function fmt(n, digits = 5) {
  return Number.isFinite(n) ? n.toFixed(digits) : String(n);
}

for (const path of [F32, F16]) {
  if (!existsSync(path)) {
    console.error(`Missing model: ${path}`);
    process.exit(1);
  }
}

const fixtures = loadFixtures();
const f32 = await Lc0OnnxEvaluator.create(readFileSync(F32));
const f16 = await Lc0OnnxEvaluator.create(readFileSync(F16));

const baseline = await runEvalSweep('f32/wasm', f32, fixtures);
const candidate = await runEvalSweep('f16/wasm', f16, fixtures);
const comparison = compareEvalSweeps(baseline, candidate);

const f32Search = await runSearchSweep('f32/wasm', new Lc0PuctSearcher(f32), fixtures, VISITS);
const f16Search = await runSearchSweep('f16/wasm', new Lc0PuctSearcher(f16), fixtures, VISITS);
const searchMismatches = f32Search.records.filter((r) => {
  const c = f16Search.records.find((x) => x.id === r.id);
  return c && c.bestMove !== r.bestMove;
});

console.log('# LC0 drift/perf sweep (f32/wasm baseline vs f16/wasm candidate)');
console.log(`fixtures: ${fixtures.length}`);
console.log(`eval throughput: f32 ${fmt(baseline.evalsPerSecond, 1)} eval/s · f16 ${fmt(candidate.evalsPerSecond, 1)} eval/s`);
console.log(`search throughput (${VISITS} visits): f32 ${fmt(f32Search.visitsPerSecond, 1)} visits/s · f16 ${fmt(f16Search.visitsPerSecond, 1)} visits/s`);
console.log('');
console.log('| fixture | best-move | wdlDrift | qDrift | mlhDrift | topPriorDrift |');
console.log('|---|---|---|---|---|---|');
for (const m of comparison.perFixture) {
  const move = m.bestMoveMatch ? m.candidateBestMove : `${m.baselineBestMove}→${m.candidateBestMove}`;
  console.log(`| ${m.id} | ${move} | ${fmt(m.wdlDrift)} | ${fmt(m.qDrift)} | ${fmt(m.mlhDrift, 3)} | ${fmt(m.topPriorDrift)} |`);
}
console.log('');
console.log('## Aggregate');
console.log(JSON.stringify({
  evalBestMoveMismatches: comparison.bestMoveMismatches,
  searchBestMoveMismatches: searchMismatches.length,
  maxWdlDrift: comparison.maxWdlDrift,
  maxQDrift: comparison.maxQDrift,
  maxMlhDrift: comparison.maxMlhDrift,
  maxTopPriorDrift: comparison.maxTopPriorDrift,
}, null, 2));
