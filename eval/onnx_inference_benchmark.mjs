#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves, makeMove } from '../src/chess/movegen.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = '') {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function nums(name, fallback) { return arg(name, fallback).split(',').filter(Boolean).map(Number); }
function median(xs) { const a = [...xs].sort((x, y) => x - y); return a[Math.floor(a.length / 2)] ?? 0; }
function p90(xs) { const a = [...xs].sort((x, y) => x - y); return a[Math.min(a.length - 1, Math.floor(a.length * 0.9))] ?? 0; }
function sampleBoards(n, fen = START_FEN) {
  const out = [];
  let board = parseFen(fen);
  for (let i = 0; i < n; i++) {
    out.push(board);
    const moves = legalMoves(board);
    board = moves.length ? makeMove(board, moves[(i * 7 + 3) % moves.length]) : parseFen(fen);
  }
  return out;
}
async function load(model, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  return (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2')
    ? SquareFormerEvaluator.create(model, meta)
    : OnnxEvaluator.create(model, meta);
}

const model = arg('--model');
const meta = arg('--meta');
const positions = Number(arg('--positions', '64'));
const repeats = Number(arg('--repeats', '5'));
const warmup = Number(arg('--warmup', '8'));
const batchSizes = nums('--batches', '1,2,4,8,16,32');
const label = arg('--label', model);
const fen = arg('--fen', START_FEN);
if (!model || !meta) throw new Error('usage: node --experimental-strip-types eval/onnx_inference_benchmark.mjs --model m.onnx --meta m.meta.json');

const evaluator = await load(model, meta);
const boards = sampleBoards(Math.max(positions, Math.max(...batchSizes), warmup), fen);
for (let i = 0; i < Math.min(warmup, boards.length); i++) await evaluator.evaluate(boards[i]);

for (const batchSize of batchSizes) {
  const times = [];
  const perPosition = [];
  for (let r = 0; r < repeats; r++) {
    const t0 = performance.now();
    let count = 0;
    for (let i = 0; i < positions; i += batchSize) {
      const chunk = boards.slice(i, Math.min(positions, i + batchSize));
      if (batchSize === 1 || !evaluator.evaluateBatch) {
        for (const board of chunk) await evaluator.evaluate(board);
      } else {
        await evaluator.evaluateBatch(chunk);
      }
      count += chunk.length;
    }
    const elapsed = performance.now() - t0;
    times.push(elapsed);
    perPosition.push(elapsed / Math.max(1, count));
  }
  const med = median(perPosition);
  const q90 = p90(perPosition);
  const totalMed = median(times);
  console.log(`RESULT label=${label} batch=${batchSize} positions=${positions} repeats=${repeats} median_ms_per_pos=${med.toFixed(4)} p90_ms_per_pos=${q90.toFixed(4)} median_positions_per_s=${(1000 / Math.max(1e-9, med)).toFixed(1)} total_median_ms=${totalMed.toFixed(3)}`);
  console.log(`METRIC ${String(label).replace(/[^A-Za-z0-9_]+/g, '_')}_b${batchSize}_median_ms_per_pos=${med.toFixed(4)}`);
}
