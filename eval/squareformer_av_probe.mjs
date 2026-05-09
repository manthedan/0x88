#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function top(entries, key, n = 5) { return [...entries].sort((a, b) => b[key] - a[key]).slice(0, n); }

const specs = arg('--models', '').split(',').filter(Boolean).map((s) => {
  const [name, onnx, meta] = s.split(':');
  if (!name || !onnx || !meta) throw new Error(`bad --models spec: ${s}`);
  return { name, onnx, meta };
});
if (!specs.length) throw new Error('need --models=name:model.onnx:model.meta.json,...');
const out = arg('--out', 'artifacts/lc0_lite_squareformer/e2_av_plumbing/av_probe.json');
const fenLines = (arg('--fens', '') ? readFileSync(arg('--fens'), 'utf8').split(/\r?\n/) : [START_FEN])
  .map((s) => s.trim()).filter((s) => s && !s.startsWith('#')).slice(0, Number(arg('--max-fens', '8')));

const results = [];
for (const spec of specs) {
  const meta = JSON.parse(readFileSync(spec.meta, 'utf8'));
  const evaluator = await SquareFormerEvaluator.create(spec.onnx, meta);
  const positions = [];
  for (const fen of fenLines) {
    const board = parseFen(fen);
    const legal = legalMoves(board);
    const ev = await evaluator.evaluate(board, { legalMoves: legal });
    const rows = legal.map((move) => {
      const actionId = moveToActionId(move);
      return { move: moveToUci(move), policy: ev.policy.get(actionId) ?? 0, actionValue: ev.actionValues?.get(actionId) ?? null };
    });
    positions.push({
      fen,
      legalMoves: legal.length,
      policyMass: rows.reduce((s, r) => s + r.policy, 0),
      actionValueCount: rows.filter((r) => r.actionValue !== null).length,
      actionValueMin: Math.min(...rows.map((r) => r.actionValue ?? 0)),
      actionValueMax: Math.max(...rows.map((r) => r.actionValue ?? 0)),
      topPolicy: top(rows, 'policy'),
      topActionValue: top(rows.filter((r) => r.actionValue !== null), 'actionValue'),
      wdl: ev.wdl,
    });
  }
  results.push({ name: spec.name, onnx: spec.onnx, meta: spec.meta, positions });
}
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ createdUtc: new Date().toISOString(), results }, null, 2));
console.log(`WROTE ${out}`);
