import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = '') {
  const i = process.argv.indexOf(name);
  return i >= 0 ? (process.argv[i + 1] ?? fallback) : fallback;
}
const model = arg('--model');
const metaPath = arg('--meta');
const generated = Number(arg('--generated', '8'));
if (!model || !metaPath) throw new Error('usage: --model model.onnx --meta model.meta.json');
const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
const rustBin = 'rust/tiny_leela_core/target/release/tiny-leela-rust-eval';
execFileSync('cargo', ['build', '--release', '--quiet', '--features', 'native-ort', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-eval'], { stdio: 'inherit' });

const seedFens = [
  START_FEN,
  'rnbqkbnr/pppp1ppp/4p3/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
  '8/8/8/3pP3/8/8/8/4K2k w - d6 0 1',
  '4k3/P6P/8/8/8/8/p6p/4K3 w - - 0 1',
  '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1',
];
// Keep deterministic handpicked cases for now; generated games are covered by board-encoding parity.
const fens = [...new Set(seedFens.slice(0, Math.max(1, Math.min(seedFens.length, generated))))];

const tsEvaluator = (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2')
  ? await SquareFormerEvaluator.create(model, meta)
  : await OnnxEvaluator.create(model, meta);

function rustEval(fen) {
  const out = execFileSync(rustBin, ['--onnx-eval-json', '--onnx', model, '--meta', metaPath, '--fen', fen], { encoding: 'utf8', maxBuffer: 1024 * 1024 * 32 });
  return JSON.parse(out);
}
function policyStats(tsPolicy, rustPolicy) {
  const keys = new Set([...tsPolicy.keys(), ...rustPolicy.keys()]);
  let l1 = 0, maxAbs = 0;
  for (const k of keys) {
    const d = Math.abs((tsPolicy.get(k) ?? 0) - (rustPolicy.get(k) ?? 0));
    l1 += d;
    maxAbs = Math.max(maxAbs, d);
  }
  return { l1, maxAbs };
}
let cases = 0, wdlMaxAbs = 0, policyMaxL1 = 0, policyMaxAbs = 0;
let worst = null;
for (const fen of fens) {
  const board = parseFen(fen);
  const ts = await tsEvaluator.evaluate(board, { legalMoves: legalMoves(board) });
  const rust = rustEval(fen);
  const rustPolicy = new Map(rust.policy.map((p) => [Number(p.actionId), Number(p.probability)]));
  const ps = policyStats(ts.policy, rustPolicy);
  const wdlErr = Math.max(...ts.wdl.map((v, i) => Math.abs(v - Number(rust.wdl[i]))));
  cases++;
  if (wdlErr > wdlMaxAbs || ps.l1 > policyMaxL1 || ps.maxAbs > policyMaxAbs) worst = { fen, wdlErr, policyL1: ps.l1, policyMaxAbs: ps.maxAbs };
  wdlMaxAbs = Math.max(wdlMaxAbs, wdlErr);
  policyMaxL1 = Math.max(policyMaxL1, ps.l1);
  policyMaxAbs = Math.max(policyMaxAbs, ps.maxAbs);
}
console.log(`METRIC rust_ts_onnx_eval_cases=${cases}`);
console.log(`METRIC rust_ts_onnx_eval_wdl_max_abs=${wdlMaxAbs.toExponential(6)}`);
console.log(`METRIC rust_ts_onnx_eval_policy_max_l1=${policyMaxL1.toExponential(6)}`);
console.log(`METRIC rust_ts_onnx_eval_policy_max_abs=${policyMaxAbs.toExponential(6)}`);
if (worst) console.log(`WORST ${JSON.stringify(worst)}`);
if (wdlMaxAbs > 1e-5 || policyMaxL1 > 1e-4 || policyMaxAbs > 1e-5) process.exit(1);
