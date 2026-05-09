#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { parseFen, START_FEN, boardToFen } from '../src/chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove } from '../src/search/puct.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function terminalWhiteScore(board) {
  const moves = legalMoves(board);
  if (moves.length) return null;
  if (!inCheck(board)) return 0.5;
  return board.turn === 'w' ? 0 : 1;
}
function elo(scoreRate) {
  const s = Math.min(0.999, Math.max(0.001, scoreRate));
  return 400 * Math.log10(s / (1 - s));
}
function modelSpecs() {
  return arg('--models', '').split(',').filter(Boolean).map((entry) => {
    const [name, onnx, meta] = entry.split(':');
    if (!name || !onnx || !meta) throw new Error(`Bad --models entry: ${entry}`);
    return { name, onnx, meta };
  });
}
const specs = modelSpecs();
if (specs.length < 2) throw new Error('Need --models=name:onnx:meta,...');
const gamesPerPair = Number(arg('--games-per-pair', '4'));
const visits = Number(arg('--visits', '32'));
const cpuct = Number(arg('--cpuct', '1.5'));
const maxPlies = Number(arg('--max-plies', '80'));
const out = arg('--out', 'artifacts/arena_10m/onnx_round_robin.json');
const openingsFile = arg('--openings-file', '');
const defaultOpenings = [
  START_FEN,
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1',
  'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1',
];
const openings = openingsFile ? readFileSync(openingsFile, 'utf8').split(/\r?\n/).map(s => s.trim()).filter(s => s && !s.startsWith('#')).map(s => s.split(/\s+#/)[0].trim()) : defaultOpenings;
function cacheKey(board, context = {}) {
  const history = (context.historyFens ?? []).slice(0, 2).join('|');
  return `${boardToFen(board)}#${history}`;
}

class CachedEvaluator {
  constructor(inner, maxEntries = 50000) {
    this.inner = inner;
    this.maxEntries = maxEntries;
    this.cache = new Map();
    this.hits = 0;
    this.misses = 0;
  }
  async evaluate(board, context = {}) {
    const key = cacheKey(board, context);
    const cached = this.cache.get(key);
    if (cached) { this.hits++; return cached; }
    this.misses++;
    const value = await this.inner.evaluate(board, context);
    this.cache.set(key, value);
    if (this.cache.size > this.maxEntries) this.cache.delete(this.cache.keys().next().value);
    return value;
  }
}

const evalCacheEntries = Number(arg('--eval-cache-entries', process.env.EVAL_CACHE_ENTRIES ?? '50000'));
async function loadEvaluator(onnx, metaPath) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const inner = (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') ? await SquareFormerEvaluator.create(onnx, meta) : await OnnxEvaluator.create(onnx, meta);
  return evalCacheEntries > 0 ? new CachedEvaluator(inner, evalCacheEntries) : inner;
}
const models = new Map();
for (const s of specs) {
  models.set(s.name, await loadEvaluator(s.onnx, s.meta));
}
const table = Object.fromEntries(specs.map(s => [s.name, { wins: 0, draws: 0, losses: 0, score: 0, games: 0, illegal: 0 }]));
const games = [];
for (let i = 0; i < specs.length; i++) for (let j = i + 1; j < specs.length; j++) {
  const a = specs[i].name, b = specs[j].name;
  for (let g = 0; g < gamesPerPair; g++) {
    const aColor = g % 2 === 0 ? 'w' : 'b';
    let board = parseFen(openings[(i * 17 + j * 7 + g) % openings.length]);
    const history = [];
    let whiteScore = terminalWhiteScore(board), illegal = null, plies = 0;
    for (; whiteScore === null && plies < maxPlies; plies++) {
      const sideName = board.turn === aColor ? a : b;
      const evaluator = models.get(sideName);
      const legalUci = new Set(legalMoves(board).map(moveToUci));
      const result = await chooseMove(board, evaluator, { visits, cpuct, historyFens: history.slice(-2).reverse() });
      if (!result.move) { whiteScore = inCheck(board) ? (board.turn === 'w' ? 0 : 1) : 0.5; break; }
      const uci = moveToUci(result.move);
      if (!legalUci.has(uci)) {
        illegal = sideName;
        whiteScore = board.turn === 'w' ? 0 : 1;
        table[sideName].illegal++;
        break;
      }
      history.push(boardToFen(board));
      board = makeMove(board, result.move);
      whiteScore = terminalWhiteScore(board);
    }
    if (whiteScore === null) {
      const ev = await models.get(a).evaluate(board, { historyFens: history.slice(-2).reverse() });
      const q = ev.wdl[0] - ev.wdl[2];
      if (Math.abs(q) < 0.05) whiteScore = 0.5;
      else whiteScore = ((board.turn === 'w') === (q > 0)) ? 1 : 0;
    }
    const aScore = whiteScore === 0.5 ? 0.5 : ((whiteScore === 1) === (aColor === 'w') ? 1 : 0);
    const bScore = 1 - aScore;
    for (const [name, score] of [[a, aScore], [b, bScore]]) {
      table[name].games++; table[name].score += score;
      if (score === 1) table[name].wins++; else if (score === 0) table[name].losses++; else table[name].draws++;
    }
    games.push({ white: aColor === 'w' ? a : b, black: aColor === 'w' ? b : a, whiteScore, a, b, aScore, plies, illegal });
    process.stderr.write(`[onnx-arena] ${a} vs ${b} game ${g + 1}/${gamesPerPair} aScore=${aScore} plies=${plies}\n`);
  }
}
const standings = Object.entries(table).map(([name, r]) => ({ name, ...r, scoreRate: r.score / Math.max(1, r.games), eloVsPool: elo(r.score / Math.max(1, r.games)) })).sort((a,b)=>b.scoreRate-a.scoreRate);
mkdirSync(dirname(out), { recursive: true });
const cacheStats = Object.fromEntries([...models.entries()].map(([name, evaluator]) => [name, { hits: evaluator.hits ?? 0, misses: evaluator.misses ?? 0, entries: evaluator.cache?.size ?? 0 }]));
const protocol = { kind:'onnx_round_robin_arena', models:specs, visits, cpuct, maxPlies, gamesPerPair, openingsFile, openings: openings.length, evalCacheEntries, cacheStats, ortThreads: process.env.ORT_INTRA_OP_NUM_THREADS ?? process.env.ORT_NUM_THREADS ?? null, createdUtc:new Date().toISOString() };
writeFileSync(out, JSON.stringify({ visits, cpuct, maxPlies, gamesPerPair, openingsFile, openings: openings.length, protocol, standings, games }, null, 2));
writeFileSync(`${out}.protocol.json`, JSON.stringify(protocol, null, 2));
standings.forEach((r, idx) => {
  console.log(`METRIC arena_rank_${idx + 1}_${r.name}_score_rate=${r.scoreRate.toFixed(6)}`);
  console.log(`METRIC arena_${r.name}_games=${r.games}`);
  console.log(`METRIC arena_${r.name}_wdl=${r.wins}_${r.draws}_${r.losses}`);
});
console.log(`METRIC arena_models=${specs.length}`);
console.log(`METRIC arena_games=${games.length}`);
