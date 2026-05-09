#!/usr/bin/env node
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { readFileSync } from 'node:fs';
import { parseFen } from '../src/chess/board.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { GumbelRootPolicy, searchRoot } from '../src/search/puct.ts';

function arg(name, fallback='') {
  const p = `${name}=`;
  const x = process.argv.find(v => v.startsWith(p));
  if (x) return x.slice(p.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function listArg(name, fallback) { return String(arg(name, fallback)).split(',').map(x => x.trim()).filter(Boolean); }
function nums(name, fallback) { return listArg(name, fallback).map(Number).filter(Number.isFinite); }
async function load(model, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  return (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') ? SquareFormerEvaluator.create(model, meta) : OnnxEvaluator.create(model, meta);
}
function valueFromWdl(wdl) { return wdl[0] - wdl[2]; }
function topPolicy(board, ev, n) {
  return legalMoves(board)
    .map(move => ({ move: moveToUci(move), prior: ev.policy.get(moveToActionId(move)) ?? 0 }))
    .sort((a, b) => b.prior - a.prior)
    .slice(0, n);
}
function topSearch(r, n) {
  return r.policy.slice()
    .sort((a, b) => b.visits - a.visits || b.prior - a.prior)
    .slice(0, n)
    .map(e => ({ move: moveToUci(e.move), visits: e.visits, prior: e.prior, q: e.q, probability: e.probability }));
}

const DEFAULT_FENS = [
  { name: 'start', fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' },
  { name: 'kings-pawn', fen: 'rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2' },
  { name: 'queen-gambit', fen: 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2' },
  { name: 'castling-choice', fen: 'r1bq1rk1/ppp2ppp/2np1n2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 8' },
  { name: 'sharp-center', fen: 'r2qkbnr/ppp2ppp/2np4/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R w KQkq - 2 6' },
  { name: 'tactical-skirmish', fen: 'r1bq1rk1/pp2bppp/2nppn2/8/2BPP3/2N2N2/PPP2PPP/R1BQ1RK1 w - - 4 8' },
];

const model = arg('--model', 'artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.onnx');
const meta = arg('--meta', 'artifacts/top3_100m_overnight_20260509/cnn96x8_100m/e08/model.meta.json');
const out = arg('--out', 'artifacts/gumbel_demos/cnn96_e08_gumbel_root_demo.json');
const mdOut = arg('--md-out', out.replace(/\.jsonl?$/i, '.md'));
const visitsList = nums('--visits', '16,32,64');
const seeds = nums('--seeds', '1,2,3,4');
const candidateCount = Number(arg('--candidate-count', '8'));
const cpuct = Number(arg('--cpuct', '1.5'));
const batchSize = Number(arg('--batch-size', '8'));
const topN = Number(arg('--top', '8'));
const gumbelScale = Number(arg('--gumbel-scale', '1'));
const qWeight = Number(arg('--q-weight', '1'));
const priorWeight = Number(arg('--prior-weight', '1'));
const visitPenalty = Number(arg('--visit-penalty', '0.15'));

mkdirSync(dirname(out), { recursive: true });
const evaluator = await load(model, meta);
const rows = [];
const summary = [];
console.log(`GUMBEL_DEMO model=${model} visits=${visitsList.join(',')} seeds=${seeds.join(',')} candidate_count=${candidateCount}`);
for (const pos of DEFAULT_FENS) {
  const board = parseFen(pos.fen);
  const ev = await evaluator.evaluate(board);
  const priorTop = topPolicy(board, ev, topN);
  console.log(`POSITION ${pos.name} value=${valueFromWdl(ev.wdl).toFixed(4)} top_prior=${priorTop.map(x => `${x.move}:${x.prior.toFixed(3)}`).join(' ')}`);
  for (const visits of visitsList) {
    const classic = await searchRoot(board, evaluator, { visits, cpuct, batchSize, temperature: 0 });
    const classicMove = classic.move ? moveToUci(classic.move) : 'null';
    rows.push({ mode: 'classic', position: pos.name, fen: pos.fen, visits, selected: classicMove, value: classic.value, totalVisits: classic.visits, stats: classic.stats, priorTop, searchTop: topSearch(classic, topN) });
    const gMoves = [];
    for (const seed of seeds) {
      const policy = new GumbelRootPolicy({ candidateCount, seed, gumbelScale, qWeight, priorWeight, visitPenalty });
      const g = await searchRoot(board, evaluator, { visits, cpuct, batchSize, temperature: 0, searchPolicy: policy });
      const selected = g.move ? moveToUci(g.move) : 'null';
      gMoves.push(selected);
      rows.push({ mode: 'gumbel_root', position: pos.name, fen: pos.fen, visits, seed, candidateCount, selected, value: g.value, totalVisits: g.visits, stats: g.stats, priorTop, searchTop: topSearch(g, topN) });
    }
    const unique = [...new Set(gMoves)];
    const changed = unique.filter(m => m !== classicMove);
    summary.push({ position: pos.name, visits, classic: classicMove, gumbelUnique: unique, changedCount: changed.length, seeds: seeds.length });
    console.log(`  visits=${visits} classic=${classicMove} gumbel_unique=${unique.join(',')} changed=${changed.length}/${unique.length}`);
  }
}
writeFileSync(out, JSON.stringify({ model, meta, visitsList, seeds, candidateCount, cpuct, batchSize, gumbel: { gumbelScale, qWeight, priorWeight, visitPenalty }, summary, rows }, null, 2));
const md = ['# Gumbel-root demo', '', `model: \`${model}\``, '', '| position | visits | classic | gumbel unique moves | changed unique |', '|---|---:|---|---|---:|', ...summary.map(s => `| ${s.position} | ${s.visits} | ${s.classic} | ${s.gumbelUnique.join(', ')} | ${s.changedCount} |`), '', `JSON: \`${out}\``].join('\n');
writeFileSync(mdOut, md + '\n');
console.log(`WROTE ${out}`);
console.log(`WROTE ${mdOut}`);
console.log(`METRIC gumbel_demo_positions=${DEFAULT_FENS.length}`);
console.log(`METRIC gumbel_demo_rows=${rows.length}`);
