#!/usr/bin/env node
import { readFileSync, writeFileSync, mkdirSync, existsSync, statSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { dirname } from 'node:path';
import { parseFen, START_FEN, boardToFen } from '../src/chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { chooseMove, classicPuctPolicy, actionValuePuctPolicy, auxPuctPolicy } from '../src/search/puct.ts';
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
function bundleBytes(path) {
  if (!path || !existsSync(path)) return 0;
  const sidecar = `${path}.data`;
  return statSync(path).size + (existsSync(sidecar) ? statSync(sidecar).size : 0);
}
function cacheKey(board, context = {}) {
  const history = (context.historyFens ?? []).slice(0, 2).join('|');
  return `${boardToFen(board)}#${history}`;
}
function parsePlayers() {
  return arg('--players', '').split(',').filter(Boolean).map((entry) => {
    const parts = entry.split(':');
    const [name, onnx, meta, mode = 'puct', avWeight = '0.25', rankWeight = '0', regretWeight = '0', riskWeight = '0', uncertaintyWeight = '0', cpuctOverride = '', fpuOverride = ''] = parts;
    if (!name || !onnx || !meta) throw new Error(`Bad --players entry: ${entry}`);
    if (!['policy', 'puct', 'av', 'aux'].includes(mode)) throw new Error(`Bad player mode for ${name}: ${mode}`);
    return { name, onnx, meta, mode, avWeight: Number(avWeight), rankWeight: Number(rankWeight), regretWeight: Number(regretWeight), riskWeight: Number(riskWeight), uncertaintyWeight: Number(uncertaintyWeight), cpuct: cpuctOverride === '' ? null : Number(cpuctOverride), fpu: fpuOverride === '' ? null : Number(fpuOverride) }; 
  });
}
function valueFromWdl(wdl) { return wdl[0] - wdl[2]; }

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
  async evaluateBatch(boards, contexts = []) {
    const out = new Array(boards.length);
    const missBoards = [];
    const missContexts = [];
    const missSlots = [];
    for (let i = 0; i < boards.length; i++) {
      const context = contexts[i] ?? {};
      const key = cacheKey(boards[i], context);
      const cached = this.cache.get(key);
      if (cached) { this.hits++; out[i] = cached; }
      else { this.misses++; missBoards.push(boards[i]); missContexts.push(context); missSlots.push({ i, key }); }
    }
    if (missBoards.length) {
      const evals = this.inner.evaluateBatch ? await this.inner.evaluateBatch(missBoards, missContexts) : await Promise.all(missBoards.map((board, i) => this.inner.evaluate(board, missContexts[i])));
      for (let j = 0; j < evals.length; j++) {
        const { i, key } = missSlots[j];
        out[i] = evals[j];
        this.cache.set(key, evals[j]);
        if (this.cache.size > this.maxEntries) this.cache.delete(this.cache.keys().next().value);
      }
    }
    return out;
  }
}

async function loadEvaluator(onnx, metaPath, evalCacheEntries) {
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  const inner = (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') ? await SquareFormerEvaluator.create(onnx, meta) : await OnnxEvaluator.create(onnx, meta);
  return evalCacheEntries > 0 ? new CachedEvaluator(inner, evalCacheEntries) : inner;
}
async function loadSharedEvaluator(onnx, metaPath, evalCacheEntries, shared) {
  const key = `${onnx}\0${metaPath}\0${evalCacheEntries}`;
  if (!shared.has(key)) shared.set(key, await loadEvaluator(onnx, metaPath, evalCacheEntries));
  return shared.get(key);
}
async function choosePolicyMove(board, evaluator, historyFens) {
  const moves = legalMoves(board);
  if (!moves.length) return { move: null, visits: 0, value: terminalWhiteScore(board) ?? 0, policy: [] };
  const ev = await evaluator.evaluate(board, { historyFens });
  let best = moves[0];
  let bestP = -Infinity;
  const entries = [];
  for (const move of moves) {
    const p = ev.policy.get(moveToActionId(move)) ?? 0;
    entries.push({ move, visits: 0, prior: p, q: 0, probability: 0 });
    if (p > bestP) { bestP = p; best = move; }
  }
  for (const entry of entries) if (entry.move === best) entry.probability = 1;
  return { move: best, visits: 0, value: valueFromWdl(ev.wdl), policy: entries };
}
async function choosePlayerMove(player, board, evaluator, opts, history) {
  const historyFens = history.slice(-2).reverse();
  if (player.mode === 'policy') return choosePolicyMove(board, evaluator, historyFens);
  return chooseMove(board, evaluator, {
    visits: opts.visits,
    cpuct: player.cpuct ?? opts.cpuct,
    fpu: player.fpu ?? opts.fpu,
    batchSize: opts.batchSize,
    historyFens,
    searchPolicy: player.mode === 'aux' ? auxPuctPolicy : (player.mode === 'av' ? actionValuePuctPolicy : classicPuctPolicy),
    avWeight: player.avWeight,
    rankWeight: player.rankWeight,
    regretWeight: player.regretWeight,
    riskWeight: player.riskWeight,
    uncertaintyWeight: player.uncertaintyWeight,
  });
}
async function adjudicateWhiteScore(board, history, judge, threshold) {
  const terminal = terminalWhiteScore(board);
  if (terminal !== null) return terminal;
  if (!judge) return 0.5;
  const ev = await judge.evaluate(board, { historyFens: history.slice(-2).reverse() });
  const stmValue = valueFromWdl(ev.wdl);
  if (Math.abs(stmValue) < threshold) return 0.5;
  const sideToMoveWins = stmValue > 0;
  return ((board.turn === 'w') === sideToMoveWins) ? 1 : 0;
}

const players = parsePlayers();
if (players.length < 2) throw new Error('Need --players=name:onnx:meta:mode[:avWeight],...');
const backend = arg('--backend', 'ts');
const gamesPerPair = Number(arg('--games-per-pair', '2'));
const visits = Number(arg('--visits', '32'));
const cpuct = Number(arg('--cpuct', '1.5'));
const fpu = Number(arg('--fpu', '0'));
const batchSize = Number(arg('--batch-size', '16'));
const maxPlies = Number(arg('--max-plies', '80'));
const out = arg('--out', 'artifacts/search_mode_arena/arena.json');
const openingsFile = arg('--openings-file', '');
const maxOpenings = Number(arg('--max-openings', '0'));
const evalCacheEntries = Number(arg('--eval-cache-entries', process.env.EVAL_CACHE_ENTRIES ?? '50000'));
const recordMoves = ['1','true','yes','on'].includes(String(arg('--record-moves', '0')).toLowerCase());
const judgeModel = arg('--judge-model', '');
const judgeMeta = arg('--judge-meta', '');
const adjudicateThreshold = Number(arg('--adjudicate-threshold', '0.05'));
const shardCount = Math.max(1, Number(arg('--shard-count', '1')));
const shardIndex = Math.max(0, Number(arg('--shard-index', '0')));
const openingOffset = Number(arg('--opening-offset', '0'));
const anchorPlayer = arg('--anchor-player', '');
if (shardIndex >= shardCount) throw new Error(`Bad shard index ${shardIndex} for shard count ${shardCount}`);
if (anchorPlayer && !players.some((p) => p.name === anchorPlayer)) throw new Error(`Unknown --anchor-player: ${anchorPlayer}`);
const defaultOpenings = [
  START_FEN,
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1',
  'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1BQKBNR b KQkq - 1 1',
];
let openings = openingsFile ? readFileSync(openingsFile, 'utf8').split(/\r?\n/).map(s => s.trim()).filter(s => s && !s.startsWith('#')).map(s => s.split(/\s+#/)[0].trim()) : defaultOpenings;
if (maxOpenings > 0) openings = openings.slice(0, maxOpenings);
if (backend === 'rust') {
  if (players.length !== 2) throw new Error('--backend rust currently supports exactly two players');
  if (players.some((p) => p.mode !== 'puct')) throw new Error('--backend rust currently supports puct mode only');
  if (judgeModel || judgeMeta) throw new Error('--backend rust currently adjudicates with terminal/value/stockfish only, not a separate judge model');
  const [candidate, baseline] = players;
  const cCpuct = candidate.cpuct ?? cpuct;
  const bCpuct = baseline.cpuct ?? cpuct;
  const cFpu = candidate.fpu ?? fpu;
  const bFpu = baseline.fpu ?? fpu;
  if (cCpuct !== bCpuct || cFpu !== bFpu) throw new Error('--backend rust currently requires identical cpuct/fpu for both players');
  const rustArgs = [
    'run', '--release', '--features', 'native-ort', '--manifest-path', 'rust/tiny_leela_core/Cargo.toml', '--bin', 'tiny-leela-rust-arena', '--',
    '--candidate-onnx', candidate.onnx, '--candidate-meta', candidate.meta, '--candidate-name', candidate.name,
    '--baseline-onnx', baseline.onnx, '--baseline-meta', baseline.meta, '--baseline-name', baseline.name,
    '--games', String(gamesPerPair), '--visits', String(visits), '--cpuct', String(cCpuct), '--fpu', String(cFpu),
    '--max-plies', String(maxPlies), '--adjudicate', 'value', '--adjudicate-threshold', String(adjudicateThreshold),
    '--openings', openings.join('|'), '--out', out,
  ];
  const child = spawnSync('cargo', rustArgs, { stdio: 'inherit', env: process.env });
  process.exit(child.status ?? 1);
}
if (backend !== 'ts') throw new Error(`Unknown --backend ${backend}`);
const sharedEvaluators = new Map();
const evaluators = new Map();
for (const p of players) evaluators.set(p.name, await loadSharedEvaluator(p.onnx, p.meta, evalCacheEntries, sharedEvaluators));
const judge = judgeModel && judgeMeta ? await loadSharedEvaluator(judgeModel, judgeMeta, evalCacheEntries, sharedEvaluators) : null;
const table = Object.fromEntries(players.map(p => [p.name, { wins: 0, draws: 0, losses: 0, score: 0, games: 0, illegal: 0 }]));
const pairTable = {};
const games = [];
const jobs = [];
for (let i = 0; i < players.length; i++) for (let j = i + 1; j < players.length; j++) {
  const a = players[i], b = players[j];
  const pairKey = `${a.name}__${b.name}`;
  pairTable[pairKey] = { a: a.name, b: b.name, aScore: 0, games: 0, aWdl: [0, 0, 0] };
  if (!anchorPlayer || a.name === anchorPlayer || b.name === anchorPlayer) {
    for (let g = 0; g < gamesPerPair; g++) jobs.push({ i, j, g, a, b, pairKey });
  }
}
const selectedJobs = jobs.filter((_, idx) => idx % shardCount === shardIndex);
const startedAt = Date.now();
process.stderr.write(`[search-mode-arena] shard ${shardIndex + 1}/${shardCount} running ${selectedJobs.length}/${jobs.length} games\n`);
for (const job of selectedJobs) {
  const { i, j, g, a, b, pairKey } = job;
  const aColor = g % 2 === 0 ? 'w' : 'b';
  const opening = openings[(openingOffset + i * 17 + j * 7 + g) % openings.length];
  let board = parseFen(opening);
  const history = [];
  const moves = [];
  let whiteScore = terminalWhiteScore(board), illegal = null, plies = 0;
  for (; whiteScore === null && plies < maxPlies; plies++) {
    const side = board.turn === aColor ? a : b;
    const evaluator = evaluators.get(side.name);
    const legalUci = new Set(legalMoves(board).map(moveToUci));
    const result = await choosePlayerMove(side, board, evaluator, { visits, cpuct, fpu, batchSize }, history);
    if (!result.move) { whiteScore = inCheck(board) ? (board.turn === 'w' ? 0 : 1) : 0.5; break; }
    const fenBefore = recordMoves ? boardToFen(board) : null;
    const uci = moveToUci(result.move);
    if (!legalUci.has(uci)) {
      illegal = side.name;
      whiteScore = board.turn === 'w' ? 0 : 1;
      table[side.name].illegal++;
      break;
    }
    if (recordMoves) moves.push({ ply: plies + 1, side: board.turn, engine: side.name, uci, fenBefore });
    history.push(boardToFen(board));
    board = makeMove(board, result.move);
    whiteScore = terminalWhiteScore(board);
  }
  if (whiteScore === null) whiteScore = await adjudicateWhiteScore(board, history, judge, adjudicateThreshold);
  const aScore = whiteScore === 0.5 ? 0.5 : ((whiteScore === 1) === (aColor === 'w') ? 1 : 0);
  const bScore = 1 - aScore;
  for (const [name, score] of [[a.name, aScore], [b.name, bScore]]) {
    table[name].games++; table[name].score += score;
    if (score === 1) table[name].wins++; else if (score === 0) table[name].losses++; else table[name].draws++;
  }
  pairTable[pairKey].aScore += aScore;
  pairTable[pairKey].games++;
  pairTable[pairKey].aWdl[aScore === 1 ? 0 : aScore === 0.5 ? 1 : 2]++;
  games.push({ white: aColor === 'w' ? a.name : b.name, black: aColor === 'w' ? b.name : a.name, opening, whiteScore, a: a.name, b: b.name, aScore, plies, finalFen: boardToFen(board), illegal, ...(recordMoves ? { moves } : {}) });
  process.stderr.write(`[search-mode-arena] ${a.name} vs ${b.name} game ${g + 1}/${gamesPerPair} aScore=${aScore} plies=${plies} elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}\n`);
}
const standings = Object.entries(table).map(([name, r]) => ({ name, ...r, scoreRate: r.score / Math.max(1, r.games), eloVsPool: elo(r.score / Math.max(1, r.games)) })).sort((a,b)=>b.scoreRate-a.scoreRate);
const pairs = Object.values(pairTable).map((r) => ({ ...r, aScoreRate: r.aScore / Math.max(1, r.games) }));
const cacheStats = Object.fromEntries([...evaluators.entries()].map(([name, evaluator]) => [name, { hits: evaluator.hits ?? 0, misses: evaluator.misses ?? 0, entries: evaluator.cache?.size ?? 0 }]));
const modelResources = Object.fromEntries(players.map((p) => [p.name, { onnx: p.onnx, meta: p.meta, bundleBytes: bundleBytes(p.onnx) }]));
const protocol = { kind:'search_mode_arena', backend, players, visits, cpuct, fpu, batchSize, maxPlies, gamesPerPair, openingsFile, openings: openings.length, openingOffset, evalCacheEntries, recordMoves, cacheStats, modelResources, judgeModel, judgeMeta, adjudicateThreshold, anchorPlayer, shardCount, shardIndex, shardGames: selectedJobs.length, totalGames: jobs.length, ortThreads: process.env.ORT_INTRA_OP_NUM_THREADS ?? process.env.ORT_NUM_THREADS ?? null, elapsedMs: Date.now() - startedAt, createdUtc:new Date().toISOString() };
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ protocol, standings, pairs, games }, null, 2));
writeFileSync(`${out}.protocol.json`, JSON.stringify(protocol, null, 2));
standings.forEach((r, idx) => {
  console.log(`METRIC arena_rank_${idx + 1}_${r.name}_score_rate=${r.scoreRate.toFixed(6)}`);
  console.log(`METRIC arena_${r.name}_games=${r.games}`);
  console.log(`METRIC arena_${r.name}_wdl=${r.wins}_${r.draws}_${r.losses}`);
});
console.log(`METRIC arena_models=${players.length}`);
console.log(`METRIC arena_games=${games.length}`);
console.log(`METRIC arena_elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(3)}`);
