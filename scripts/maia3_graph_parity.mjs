#!/usr/bin/env node
// Output parity of a rewritten Maia3 ONNX (scripts/maia3_graph_rewrite.py)
// against the shipped model, on ORT WASM in Node.
//
// Both models go through the product's load path: the DequantizeLinear fold
// (src/nn/onnxDequantFold.ts) and ORT with graphOptimizationLevel 'all'. Every
// row runs at the given physical batch (rows padded per chunk like the
// product's evaluators). Rows: the shared fixture corpus plus seeded random
// playouts and promotion-heavy positions, each under the four (self, opponent)
// Elo conditions of the kernel-bench gate (lab/kernel-hillclimb/gates.mjs
// MAIA3_PARITY_CONDITIONS); at batch > 1 also each position under an Elo grid
// (one position, `batch` ratings, like evaluateConditions).
//
// Reported: raw max-abs on logits_move (all and legal entries) and
// logits_value; the legal-move policy gate (softmax over legal moves, best
// move, top-3 order; legalPolicyParity); the value head as probabilities
// (model order is L, D, W, not W, D, L) and as expected score W + D/2;
// best-move and top-3 match rates. Exit 1 if any limit fails.
//
//   node --experimental-strip-types scripts/maia3_graph_parity.mjs \
//     --candidate artifacts/maia3-graph/maia3.einsum-promo.onnx --batch 1 [--batch 11] [--out report.json]
import { readFileSync, writeFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { legalPolicyParity, MAIA3_PARITY_CONDITIONS, maxAbsDiff, runChunked } from '../lab/kernel-hillclimb/gates.mjs';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { boardToMaia3Tokens, maia3InternalsForTests } from '../src/lc0/maia3.ts';
import { foldOnnxDequantizeLinear } from '../src/nn/onnxDequantFold.ts';
import * as ort from '../src/nn/ortRuntime.ts';

const { values: args } = parseArgs({
  options: {
    reference: { type: 'string', default: 'public/models/maia3/maia3_simplified.qdq8.onnx' },
    candidate: { type: 'string' },
    batch: { type: 'string', multiple: true },
    playouts: { type: 'string', default: '4' },
    plies: { type: 'string', default: '40' },
    'max-prob-abs': { type: 'string', default: '0.01' },
    'max-value-prob-abs': { type: 'string', default: '0.01' },
    'tie-epsilon': { type: 'string', default: '0.02' },
    out: { type: 'string' },
  },
});
if (!args.candidate) throw new Error('--candidate is required');
const batches = (args.batch ?? ['1']).map(Number);
const limits = { maxProbAbs: Number(args['max-prob-abs']), maxValueProbAbs: Number(args['max-value-prob-abs']), tieEpsilon: Number(args['tie-epsilon']) };

// ---- corpus
const fixtureFens = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'))
  .map((f) => f.fen)
  .filter(Boolean);
const promotionFens = [
  '8/1P5k/8/8/8/8/8/K7 w - - 0 1',
  '7k/8/8/8/8/8/1p6/4K3 b - - 0 1',
  'r3k3/1P6/8/8/8/8/6p1/4K2R w Kq - 0 1',
  'r3k3/1P6/8/8/8/8/6p1/4K2R b Kq - 0 1',
  '1n2k3/PPP5/8/8/8/8/5ppp/4K1N1 w - - 0 1',
  '1n2k3/PPP5/8/8/8/8/5ppp/4K1N1 b - - 0 1',
  '8/P6P/8/2k5/8/8/p6p/2K5 w - - 0 1',
];
let rng = 0x2545f491;
const rand = () => {
  rng ^= rng << 13;
  rng ^= rng >>> 17;
  rng ^= rng << 5;
  return (rng >>> 0) / 0x1_0000_0000;
};
const playoutFens = [];
for (let game = 0; game < Number(args.playouts); game++) {
  let board = parseFen(START_FEN);
  for (let ply = 0; ply < Number(args.plies); ply++) {
    const moves = legalMoves(board);
    if (!moves.length) break;
    board = makeMove(board, moves[Math.floor(rand() * moves.length)]);
    if (ply % 4 === 3) playoutFens.push(boardToFen(board));
  }
}
const fens = [...fixtureFens, ...promotionFens, ...playoutFens].filter((fen) => maia3InternalsForTests.legalMaia3Moves(parseFen(fen)).length > 0);
const conditionRows = fens.flatMap((fen) => MAIA3_PARITY_CONDITIONS.map(([eloSelf, eloOppo]) => ({ fen, eloSelf, eloOppo })));
const gridRows = (batch) =>
  fens.flatMap((fen) => Array.from({ length: batch }, (_, i) => ({ fen, eloSelf: 600 + Math.round((2000 / Math.max(1, batch - 1)) * i), eloOppo: 1500 })));

// ---- sessions (product load path: dequant fold, then ORT 'all')
function prepare(path) {
  const bytes = new Uint8Array(readFileSync(path));
  const fold = foldOnnxDequantizeLinear(bytes);
  return {
    path,
    bytes: fold.bytes,
    fold: { foldedNodes: fold.foldedNodes, skippedNodes: fold.skippedNodes, bytesBefore: fold.bytesBefore, bytesAfter: fold.bytesAfter },
  };
}
const reference = prepare(args.reference);
const candidate = prepare(args.candidate);
const refSession = await ort.InferenceSession.create(reference.bytes, ort.sessionOptions(['wasm']));
const candSession = await ort.InferenceSession.create(candidate.bytes, ort.sessionOptions(['wasm']));

function feeds(rows) {
  const tokens = new Float32Array(rows.length * 64 * 12);
  for (const [i, r] of rows.entries()) tokens.set(boardToMaia3Tokens(parseFen(r.fen)), i * 64 * 12);
  return {
    tokens: new ort.Tensor('float32', tokens, [rows.length, 64, 12]),
    elo_self: new ort.Tensor(
      'float32',
      Float32Array.from(rows, (r) => r.eloSelf),
      [rows.length],
    ),
    elo_oppo: new ort.Tensor(
      'float32',
      Float32Array.from(rows, (r) => r.eloOppo),
      [rows.length],
    ),
  };
}
const run = (session) => async (chunk) => {
  const out = await session.run(feeds(chunk));
  return { logits_move: Float32Array.from(out.logits_move.data), logits_value: Float32Array.from(out.logits_value.data) };
};

function softmax(values) {
  const max = Math.max(...values);
  const e = values.map((x) => Math.exp(x - max));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

function compare(rows, cand, ref) {
  const raw = maxAbsDiff(cand, ref);
  const legalRows = rows.map(({ fen }) => {
    const legal = maia3InternalsForTests.legalMaia3Moves(parseFen(fen));
    return { fen, indices: legal.map((m) => m.index), ucis: legal.map((m) => m.uci) };
  });
  let legalLogitMax = 0;
  let top3Same = 0;
  let bestSame = 0;
  legalRows.forEach(({ indices }, r) => {
    const c = indices.map((i) => cand.logits_move[r * 4352 + i]);
    const f = indices.map((i) => ref.logits_move[r * 4352 + i]);
    for (let k = 0; k < c.length; k++) legalLogitMax = Math.max(legalLogitMax, Math.abs(c[k] - f[k]));
    const order = (v) => v.map((x, k) => [x, k]).sort((a, b) => b[0] - a[0]);
    const oc = order(c);
    const of = order(f);
    if (oc[0][1] === of[0][1]) bestSame++;
    if (oc.slice(0, 3).every(([, k], i) => k === of[i]?.[1])) top3Same++;
  });
  const legalPolicy = legalPolicyParity(cand.logits_move, ref.logits_move, legalRows, {
    maxProbAbs: limits.maxProbAbs,
    tieEpsilon: limits.tieEpsilon,
    topK: 3,
  });
  // Value head: model order L, D, W.
  let valueProbMax = 0;
  let expectedScoreMax = 0;
  for (let r = 0; r < rows.length; r++) {
    const [lc, dc, wc] = softmax(Array.from(cand.logits_value.subarray(r * 3, r * 3 + 3)));
    const [lr, dr, wr] = softmax(Array.from(ref.logits_value.subarray(r * 3, r * 3 + 3)));
    valueProbMax = Math.max(valueProbMax, Math.abs(lc - lr), Math.abs(dc - dr), Math.abs(wc - wr));
    expectedScoreMax = Math.max(expectedScoreMax, Math.abs(wc + dc / 2 - (wr + dr / 2)));
  }
  const problems = [...legalPolicy.problems];
  for (const [name, o] of Object.entries(raw))
    if (o.maxAbs === null || o.nonFinite || o.length !== o.referenceLength) problems.push(`${name}: missing, non-finite or length mismatch`);
  if (valueProbMax > limits.maxValueProbAbs) problems.push(`value prob diff ${valueProbMax.toFixed(5)} > ${limits.maxValueProbAbs}`);
  return {
    rows: rows.length,
    maxAbs: { logitsMove: raw.logits_move.maxAbs, logitsMoveLegal: legalLogitMax, logitsValue: raw.logits_value.maxAbs },
    legalProbMaxAbs: legalPolicy.maxLegalProbAbs,
    legalKlMax: legalPolicy.maxKl,
    valueProbMaxAbs: valueProbMax,
    expectedScoreMaxAbs: expectedScoreMax,
    bestMoveMatch: `${bestSame}/${rows.length}`,
    top3OrderMatch: `${top3Same}/${rows.length}`,
    tieTolerated: legalPolicy.tieTolerated,
    problems,
    pass: problems.length === 0,
  };
}

const results = [];
for (const batch of batches) {
  const sets = { conditions: conditionRows, ...(batch > 1 ? { grid: gridRows(batch) } : {}) };
  for (const [set, rows] of Object.entries(sets)) {
    const ref = await runChunked(rows, batch, run(refSession));
    const cand = await runChunked(rows, batch, run(candSession));
    const r = { batch, set, ...compare(rows, cand, ref) };
    results.push(r);
    console.log(
      `b${batch} ${set}: rows ${r.rows}  logits_move ${r.maxAbs.logitsMove.toExponential(2)} (legal ${r.maxAbs.logitsMoveLegal.toExponential(2)})  logits_value ${r.maxAbs.logitsValue.toExponential(2)}  legal prob ${r.legalProbMaxAbs.toExponential(2)}  value prob ${r.valueProbMaxAbs.toExponential(2)}  best ${r.bestMoveMatch}  top3 ${r.top3OrderMatch}  ${r.pass ? 'PASS' : `FAIL ${r.problems.slice(0, 3).join('; ')}`}`,
    );
  }
}
await refSession.release();
await candSession.release();
const report = {
  schema: 'maia3-graph-parity.v1',
  generatedAt: new Date().toISOString(),
  runtime: 'onnxruntime-web WASM (Node), dequant fold on, graphOptimizationLevel all',
  reference: { path: reference.path, fold: reference.fold },
  candidate: { path: candidate.path, fold: candidate.fold },
  corpus: {
    fixtures: fixtureFens.length,
    promotion: promotionFens.length,
    playouts: playoutFens.length,
    positions: fens.length,
    eloConditions: MAIA3_PARITY_CONDITIONS,
  },
  limits,
  results,
  pass: results.every((r) => r.pass) && candidate.fold.foldedNodes === reference.fold.foldedNodes,
};
if (candidate.fold.foldedNodes !== reference.fold.foldedNodes)
  console.log(`FAIL dequant fold: candidate folded ${candidate.fold.foldedNodes}, reference ${reference.fold.foldedNodes}`);
if (args.out) writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`);
console.log(report.pass ? 'maia3 graph parity: PASS' : 'maia3 graph parity: FAIL');
process.exit(report.pass ? 0 : 1);
