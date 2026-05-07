#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { searchRoot } from '../src/search/puct.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find(v => v.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
async function loadEvaluator(model, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  return meta.kind === 'squareformer' ? SquareFormerEvaluator.create(model, meta) : OnnxEvaluator.create(model, meta);
}
const model = arg('--model'), meta = arg('--meta'), pos = arg('--positions-json');
const limit = Number(arg('--limit', '100'));
const topk = Number(arg('--topk', '8'));
if (!model || !meta || !pos) throw new Error('usage --model --meta --positions-json [--limit 100] [--topk 8]');
const data = JSON.parse(readFileSync(pos, 'utf8'));
const fens = (data.positions ?? data).map(x => x.fen ?? x.fen_before).filter(Boolean).slice(0, limit);
const evaluator = await loadEvaluator(model, meta);
let checked = 0, exactTop1 = 0, exactTopK = 0, maxPriorDiff = 0;
for (const fen of fens) {
  const board = parseFen(fen);
  const ev = await evaluator.evaluate(board);
  const legal = legalMoves(board);
  const raw = legal.map(move => ({ uci: moveToUci(move), raw: Math.max(0, ev.policy.get(moveToActionId(move)) ?? 0) }));
  const total = raw.reduce((s, x) => s + x.raw, 0);
  const expected = raw.map(x => ({ uci: x.uci, prior: total > 0 ? x.raw / total : 1 / raw.length })).sort((a,b)=>b.prior-a.prior);
  const root = await searchRoot(board, evaluator, { visits: 1, temperature: 0 });
  const actual = root.policy.map(x => ({ uci: moveToUci(x.move), prior: x.prior })).sort((a,b)=>b.prior-a.prior);
  checked++;
  if (expected[0]?.uci === actual[0]?.uci) exactTop1++;
  if (expected.slice(0, topk).map(x=>x.uci).join(',') === actual.slice(0, topk).map(x=>x.uci).join(',')) exactTopK++;
  const amap = new Map(actual.map(x => [x.uci, x.prior]));
  for (const e of expected) maxPriorDiff = Math.max(maxPriorDiff, Math.abs(e.prior - (amap.get(e.uci) ?? -1)));
}
console.log(`METRIC root_prior_positions=${checked}`);
console.log(`METRIC root_prior_top1_match=${exactTop1}`);
console.log(`METRIC root_prior_top${topk}_match=${exactTopK}`);
console.log(`METRIC root_prior_max_abs_diff=${maxPriorDiff}`);
if (exactTop1 !== checked || exactTopK !== checked || maxPriorDiff > 1e-9) process.exitCode = 1;
