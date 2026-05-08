#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function num(name, fallback) { return Number(arg(name, String(fallback))); }
async function load(model, meta) {
  return meta.kind === 'squareformer' ? SquareFormerEvaluator.create(model, meta) : OnnxEvaluator.create(model, meta);
}
function sampleBoards(n) {
  const boards = [];
  let board = parseFen(arg('--fen', START_FEN));
  for (let i = 0; i < n; i++) {
    boards.push(board);
    const moves = legalMoves(board);
    if (!moves.length) board = parseFen(START_FEN);
    else board = makeMove(board, moves[(i * 7 + 3) % moves.length]);
  }
  return boards;
}
function maxMapDiff(a, b) {
  const keys = new Set([...(a?.keys?.() ?? []), ...(b?.keys?.() ?? [])]);
  let max = 0;
  let count = 0;
  for (const key of keys) {
    count++;
    max = Math.max(max, Math.abs((a?.get(key) ?? 0) - (b?.get(key) ?? 0)));
  }
  return { max, count };
}
function updateMax(maxes, eva, evb) {
  maxes.policy = Math.max(maxes.policy, maxMapDiff(eva.policy, evb.policy).max);
  for (let i = 0; i < 3; i++) maxes.wdl = Math.max(maxes.wdl, Math.abs((eva.wdl[i] ?? 0) - (evb.wdl[i] ?? 0)));
  maxes.actionValues = Math.max(maxes.actionValues, maxMapDiff(eva.actionValues, evb.actionValues).max);
  maxes.rankScores = Math.max(maxes.rankScores, maxMapDiff(eva.rankScores, evb.rankScores).max);
  maxes.regrets = Math.max(maxes.regrets, maxMapDiff(eva.regrets, evb.regrets).max);
  maxes.risks = Math.max(maxes.risks, maxMapDiff(eva.risks, evb.risks).max);
  maxes.uncertainties = Math.max(maxes.uncertainties, maxMapDiff(eva.uncertainties, evb.uncertainties).max);
}

const modelA = arg('--model-a');
const modelB = arg('--model-b');
const metaPath = arg('--meta');
const positions = num('--positions', 8);
const tolerance = num('--tolerance', 1e-5);
if (!modelA || !modelB || !metaPath) throw new Error('usage: node --experimental-strip-types eval/onnx_parity_check.mjs --model-a a.onnx --model-b b.onnx --meta model.meta.json');
const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
const [evalA, evalB] = await Promise.all([load(modelA, meta), load(modelB, meta)]);
const boards = sampleBoards(positions);
const maxes = { policy: 0, wdl: 0, actionValues: 0, rankScores: 0, regrets: 0, risks: 0, uncertainties: 0 };
for (const board of boards) updateMax(maxes, await evalA.evaluate(board), await evalB.evaluate(board));
if (evalA.evaluateBatch && evalB.evaluateBatch) {
  const [batchA, batchB] = await Promise.all([evalA.evaluateBatch(boards), evalB.evaluateBatch(boards)]);
  for (let i = 0; i < boards.length; i++) updateMax(maxes, batchA[i], batchB[i]);
}
const worst = Math.max(...Object.values(maxes));
for (const [k, v] of Object.entries(maxes)) console.log(`METRIC onnx_parity_${k}_max_diff=${v}`);
console.log(`METRIC onnx_parity_positions=${positions}`);
console.log(`METRIC onnx_parity_worst_max_diff=${worst}`);
if (worst > tolerance) {
  console.error(`ONNX parity failed: worst=${worst} tolerance=${tolerance}`);
  process.exit(1);
}
console.log(`ONNX parity ok: worst=${worst} tolerance=${tolerance}`);
