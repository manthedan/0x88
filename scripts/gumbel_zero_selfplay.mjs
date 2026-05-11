#!/usr/bin/env node
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../src/chess/movegen.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { OnnxEvaluator } from '../src/nn/onnxEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import { UniformEvaluator } from '../src/nn/evaluator.ts';
import { ClassicPUCTPolicy, edgeQForParent, edgeSelectVisits, searchRoot } from '../src/search/puct.ts';

function arg(name, fallback = undefined) {
  const prefix = `${name}=`;
  const inline = process.argv.find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const i = process.argv.indexOf(name);
  return i >= 0 ? process.argv[i + 1] : fallback;
}
function boolArg(name, fallback = false) {
  const v = arg(name, fallback ? 'true' : 'false');
  return ['1', 'true', 'yes', 'on'].includes(String(v).toLowerCase());
}
function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 2 ** 32;
  };
}
function splitmix32(seed) {
  let x = seed >>> 0;
  return () => {
    x = (x + 0x9e3779b9) >>> 0;
    let z = x;
    z = Math.imul(z ^ (z >>> 16), 0x21f0aaad);
    z = Math.imul(z ^ (z >>> 15), 0x735a2d97);
    z = (z ^ (z >>> 15)) >>> 0;
    return (z + 0.5) / 0x100000000;
  };
}
function gumbel01(rng) {
  const u = Math.min(1 - 1e-12, Math.max(1e-12, rng()));
  return -Math.log(-Math.log(u));
}
function softmaxScores(items, temp) {
  if (!items.length) return [];
  if (temp <= 0) {
    let best = 0;
    for (let i = 1; i < items.length; i++) if (items[i].score > items[best].score) best = i;
    return items.map((_, i) => i === best ? 1 : 0);
  }
  const m = Math.max(...items.map((x) => x.score));
  const ws = items.map((x) => Math.exp((x.score - m) / temp));
  const total = ws.reduce((a, b) => a + b, 0) || 1;
  return ws.map((w) => w / total);
}
function chooseWeighted(entries, rng) {
  let r = rng();
  for (const entry of entries) {
    r -= entry.probability;
    if (r <= 0) return entry;
  }
  return entries.at(-1) ?? null;
}
function entropy(entries) {
  return -entries.reduce((sum, entry) => {
    const p = Math.max(0, Number(entry.probability ?? 0));
    return p > 0 ? sum + p * Math.log(p) : sum;
  }, 0);
}
function fenKey(fen) {
  return fen.split(/\s+/).slice(0, 4).join(' ');
}
function terminalWhiteScore(board, repetitions = new Map()) {
  if (board.halfmove >= 100) return { score: 0.5, reason: 'fifty_move' };
  if ((repetitions.get(fenKey(boardToFen(board))) ?? 0) >= 3) return { score: 0.5, reason: 'threefold' };
  const legal = legalMoves(board);
  if (legal.length) return null;
  if (!inCheck(board)) return { score: 0.5, reason: 'stalemate' };
  return { score: board.turn === 'w' ? 0 : 1, reason: 'checkmate' };
}
function resultForTurn(whiteScore, turn) {
  if (whiteScore === 0.5) return [0, 1, 0];
  const sideWon = (whiteScore === 1 && turn === 'w') || (whiteScore === 0 && turn === 'b');
  return sideWon ? [1, 0, 0] : [0, 0, 1];
}
async function loadEvaluator(kind, model, metaPath) {
  if (kind === 'uniform') return new UniformEvaluator();
  const meta = JSON.parse(readFileSync(metaPath, 'utf8'));
  if (meta.kind === 'squareformer' || meta.kind === 'squareformer_v2') return SquareFormerEvaluator.create(model, meta);
  return OnnxEvaluator.create(model, meta);
}

class GumbelZeroRootPolicy extends ClassicPUCTPolicy {
  constructor(options = {}) {
    super();
    this.options = {
      candidateCount: Math.max(1, Math.floor(options.candidateCount ?? 16)),
      seed: Math.floor(options.seed ?? 1),
      gumbelScale: Number(options.gumbelScale ?? 1),
      qWeight: Number(options.qWeight ?? 1),
      priorWeight: Number(options.priorWeight ?? 1),
      visitPenalty: Number(options.visitPenalty ?? 0.05),
      targetTemperature: Number(options.targetTemperature ?? 1),
      minCandidateVisits: Math.max(0, Math.floor(options.minCandidateVisits ?? 1)),
      estimatedQ: String(options.estimatedQ ?? 'fpu'),
    };
    this.rng = splitmix32(this.options.seed);
    this.states = new WeakMap();
  }
  rootState(node) {
    let state = this.states.get(node);
    if (state) return state;
    const scored = node.edges.map((edge) => {
      const noise = gumbel01(this.rng) * this.options.gumbelScale;
      return { edge, noise, score: Math.log(Math.max(edge.prior, 1e-12)) + noise };
    }).sort((a, b) => b.score - a.score);
    const keep = new Set(scored.slice(0, Math.min(this.options.candidateCount, scored.length)).map((x) => x.edge));
    const rank = new Map(scored.map((x, i) => [x.edge, i]));
    const noise = new Map(scored.map((x) => [x.edge, x.noise]));
    state = { candidates: keep, rank, noise };
    this.states.set(node, state);
    return state;
  }
  estimatedQFor(edge, node, context) {
    if (edge.visits) return edgeQForParent(edge, context.fpu);
    if (this.options.estimatedQ === 'visited-mean' || this.options.estimatedQ === 'pessimistic') {
      const qs = node.edges.filter((e) => e.visits > 0).map((e) => edgeQForParent(e, context.fpu));
      if (qs.length) {
        const mean = qs.reduce((a, b) => a + b, 0) / qs.length;
        return this.options.estimatedQ === 'pessimistic' ? mean - 1 / (qs.length + 1) : mean;
      }
    }
    return context.fpu;
  }
  scoreEdge(node, edge, context) {
    if (!node.isRoot) return super.scoreEdge(node, edge, context);
    const state = this.rootState(node);
    if (!state.candidates.has(edge)) return -Infinity;
    const rank = state.rank.get(edge) ?? 999999;
    if (edge.visits < this.options.minCandidateVisits) return 1_000_000 - rank - edge.visits;
    const parentVisits = node.edges.reduce((sum, e) => sum + edgeSelectVisits(e), 0);
    const sv = edgeSelectVisits(edge);
    const q = this.estimatedQFor(edge, node, context);
    const puct = context.cpuct * edge.prior * Math.sqrt(parentVisits + 1) / (1 + sv);
    const g = state.noise.get(edge) ?? 0;
    const logPrior = Math.log(Math.max(edge.prior, 1e-12));
    return this.options.qWeight * q
      + this.options.priorWeight * (logPrior + g) / (1 + sv)
      + puct
      - this.options.visitPenalty * sv;
  }
  rootPolicy(edges, context, node = undefined) {
    if (!edges.length) return [];
    const state = node ? this.rootState(node) : { candidates: new Set(edges), noise: new Map(), rank: new Map() };
    const candidateEdges = edges.filter((edge) => state.candidates.has(edge));
    const scored = candidateEdges.map((edge) => {
      const q = this.estimatedQFor(edge, node ?? { edges }, context);
      const logPrior = Math.log(Math.max(edge.prior, 1e-12));
      const g = state.noise.get(edge) ?? 0;
      return { edge, q, g, score: this.options.priorWeight * logPrior + g + this.options.qWeight * q };
    });
    const probs = softmaxScores(scored, this.options.targetTemperature);
    const probByEdge = new Map(scored.map((x, i) => [x.edge, probs[i] ?? 0]));
    const scoreByEdge = new Map(scored.map((x) => [x.edge, x.score]));
    const qByEdge = new Map(scored.map((x) => [x.edge, x.q]));
    return edges.map((edge) => ({
      move: edge.move,
      visits: edge.visits,
      prior: edge.prior,
      q: qByEdge.get(edge) ?? edgeQForParent(edge, context.fpu),
      probability: probByEdge.get(edge) ?? 0,
      candidate: state.candidates.has(edge),
      gumbel: state.noise.get(edge) ?? null,
      rootScore: scoreByEdge.get(edge) ?? null,
      candidateRank: state.rank.get(edge) ?? null,
    }));
  }
  chooseFinalMove(entries) {
    if (!entries.length) return null;
    return entries.reduce((a, b) => b.probability > a.probability ? b : a);
  }
}

const outPath = arg('--out', 'data/selfplay/gumbel_zero/bootstrap.jsonl');
const games = Number(arg('--games', '2'));
const visits = Number(arg('--visits', '16'));
const candidateCount = Number(arg('--candidate-count', '16'));
const maxPlies = Number(arg('--max-plies', '120'));
const seed = Number(arg('--seed', '1'));
const cpuct = Number(arg('--cpuct', '1.5'));
const fpu = Number(arg('--fpu', '0'));
const batchSize = Number(arg('--batch-size', '1'));
const targetTemperature = Number(arg('--target-temperature', '1'));
const moveSelection = arg('--move-selection', 'argmax');
const evaluatorKind = arg('--evaluator', 'uniform');
const model = arg('--model', '');
const meta = arg('--meta', '');
const openingFensPath = arg('--opening-fens', '');
const progressEvery = Math.max(1, Number(arg('--progress-every', '1')));
const emitAllLegal = boolArg('--emit-all-legal', false);
const estimatedQ = arg('--estimated-q', 'pessimistic');
const minCandidateVisits = Number(arg('--min-candidate-visits', '1'));
const qWeight = Number(arg('--q-weight', '1'));
const priorWeight = Number(arg('--prior-weight', '1'));
const gumbelScale = Number(arg('--gumbel-scale', '1'));
const visitPenalty = Number(arg('--visit-penalty', '0.05'));

if (evaluatorKind !== 'uniform' && (!model || !meta)) throw new Error('--model and --meta are required for non-uniform evaluators');
const openingFens = openingFensPath ? readFileSync(openingFensPath, 'utf8').split(/\r?\n/).map((line) => line.trim()).filter((line) => line && !line.startsWith('#')).map((line) => line.includes('\t') ? line.split('\t').at(-1) : line) : [];
const rng = makeRng(seed);
const evaluator = await loadEvaluator(evaluatorKind, model, meta);
const rows = [];
let completedGames = 0;
let decisiveGames = 0;
let totalPlies = 0;
let maxedGames = 0;
let checkmates = 0;
let policyOverturns = 0;
let selectedOutsidePolicyTop1 = 0;
const startedAt = Date.now();
const log = (msg) => process.stderr.write(`[gumbel-zero evaluator=${evaluatorKind} visits=${visits} k=${candidateCount}] ${msg}\n`);
log(`start games=${games} max_plies=${maxPlies} out=${outPath}`);

for (let game = 0; game < games; game++) {
  let board = openingFens.length ? parseFen(openingFens[Math.floor(rng() * openingFens.length)]) : parseFen(START_FEN);
  const historyFens = [];
  const repetitions = new Map([[fenKey(boardToFen(board)), 1]]);
  const pending = [];
  let terminal = null;
  log(`game ${game + 1}/${games} start`);
  for (let ply = 0; ply < maxPlies; ply++) {
    terminal = terminalWhiteScore(board, repetitions);
    if (terminal) break;
    const policy = new GumbelZeroRootPolicy({ candidateCount, seed: seed + game * 1000003 + ply * 9973, targetTemperature, estimatedQ, minCandidateVisits, qWeight, priorWeight, gumbelScale, visitPenalty });
    const result = await searchRoot(board, evaluator, { visits, cpuct, fpu, batchSize, temperature: 0, historyFens, searchPolicy: policy });
    const entries = emitAllLegal ? result.policy : result.policy.filter((entry) => entry.probability > 0 || entry.candidate || entry.visits > 0);
    const targetEntries = result.policy.filter((entry) => entry.probability > 0).sort((a, b) => b.probability - a.probability);
    const selectedEntry = moveSelection === 'sample-target' ? chooseWeighted(targetEntries, rng) : targetEntries[0];
    const selectedMove = selectedEntry?.move ?? result.move;
    if (!selectedMove) { terminal = { score: 0.5, reason: 'no_move' }; break; }
    const priorTop = result.policy.reduce((best, entry) => entry.prior > best.prior ? entry : best, result.policy[0]);
    const selectedUci = moveToUci(selectedMove);
    const priorTopUci = priorTop ? moveToUci(priorTop.move) : '';
    const candidateQs = entries.filter((entry) => entry.candidate).map((entry) => entry.q);
    const bestQ = candidateQs.length ? Math.max(...candidateQs) : 0;
    const selectedQ = selectedEntry?.q ?? 0;
    const rowPolicy = Object.fromEntries(targetEntries.map((entry) => [moveToUci(entry.move), Number(entry.probability.toFixed(8))]));
    if (selectedUci !== priorTopUci) selectedOutsidePolicyTop1++;
    if (targetEntries.length && selectedUci !== moveToUci(targetEntries[0].move)) policyOverturns++;
    pending.push({
      schema: 'tiny_leela_gumbel_zero_selfplay_v1',
      game_id: `gz${String(game).padStart(6, '0')}`,
      ply,
      fen: boardToFen(board),
      turn: board.turn,
      history_fens: [...historyFens],
      visits: result.visits,
      requested_visits: visits,
      candidate_count: candidateCount,
      evaluator: evaluatorKind,
      policy: rowPolicy,
      selected_move: selectedUci,
      root_value: Number((result.value ?? selectedQ).toFixed(8)),
      root_wdl: [Number(Math.max(0, result.value).toFixed(8)), Number(Math.max(0, 1 - Math.abs(result.value)).toFixed(8)), Number(Math.max(0, -result.value).toFixed(8))],
      policy_entropy: Number(entropy(targetEntries).toFixed(8)),
      search_overturned_policy_top1: selectedUci !== priorTopUci,
      candidate_q_spread: Number(((candidateQs.length ? Math.max(...candidateQs) - Math.min(...candidateQs) : 0)).toFixed(8)),
      selected_regret: Number((bestQ - selectedQ).toFixed(8)),
      candidates: entries.filter((entry) => entry.candidate).sort((a, b) => (a.candidateRank ?? 0) - (b.candidateRank ?? 0)).map((entry) => ({
        move: moveToUci(entry.move),
        action_id: moveToActionId(entry.move),
        prior: Number(entry.prior.toFixed(8)),
        q: Number(entry.q.toFixed(8)),
        visits: entry.visits,
        target_probability: Number((entry.probability ?? 0).toFixed(8)),
        gumbel: entry.gumbel === null ? null : Number(entry.gumbel.toFixed(8)),
        root_score: entry.rootScore === null ? null : Number(entry.rootScore.toFixed(8)),
        regret: Number((bestQ - entry.q).toFixed(8)),
      })),
      legal_count: legalMoves(board).length,
      stats: result.stats,
    });
    historyFens.unshift(boardToFen(board));
    board = makeMove(board, selectedMove);
    const k = fenKey(boardToFen(board));
    repetitions.set(k, (repetitions.get(k) ?? 0) + 1);
    totalPlies++;
    if ((ply + 1) % progressEvery === 0) log(`game ${game + 1}/${games} ply=${ply + 1} move=${selectedUci} rows_pending=${pending.length} elapsed_s=${((Date.now() - startedAt) / 1000).toFixed(1)}`);
  }
  if (!terminal) { terminal = { score: 0.5, reason: 'max_plies' }; maxedGames++; }
  if (terminal.score !== 0.5) decisiveGames++;
  if (terminal.reason === 'checkmate') checkmates++;
  for (const row of pending) rows.push({ ...row, result: resultForTurn(terminal.score, row.turn), white_score: terminal.score, terminal_reason: terminal.reason });
  completedGames++;
  log(`game ${game + 1}/${games} done white_score=${terminal.score} reason=${terminal.reason} rows_total=${rows.length}`);
}

mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, rows.map((row) => JSON.stringify(row)).join('\n') + (rows.length ? '\n' : ''));
console.log(`METRIC gumbel_zero_games=${completedGames}`);
console.log(`METRIC gumbel_zero_positions=${rows.length}`);
console.log(`METRIC gumbel_zero_avg_plies=${(totalPlies / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC gumbel_zero_decisive_rate=${(decisiveGames / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC gumbel_zero_checkmate_rate=${(checkmates / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC gumbel_zero_maxed_rate=${(maxedGames / Math.max(1, completedGames)).toFixed(6)}`);
console.log(`METRIC gumbel_zero_selected_outside_policy_top1_rate=${(selectedOutsidePolicyTop1 / Math.max(1, rows.length)).toFixed(6)}`);
console.log(`METRIC gumbel_zero_policy_overturn_rate=${(policyOverturns / Math.max(1, rows.length)).toFixed(6)}`);
console.log(`METRIC gumbel_zero_output_bytes=${Buffer.byteLength(rows.map((row) => JSON.stringify(row)).join('\n'))}`);
