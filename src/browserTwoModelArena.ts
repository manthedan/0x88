import { boardToFen, parseFen, START_FEN, type BoardState, type Color } from './chess/board.ts';
import { inCheck, legalMoves, makeMove } from './chess/movegen.ts';
import { moveToUci } from './chess/moveCodec.ts';
import { automaticDrawReason } from './chess/drawRules.ts';
import { CachedEvaluator, type Evaluator } from './nn/evaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';
import { collectOrtRuntimeDiagnostics } from './nn/ortRuntime.ts';
import { chooseMove, classicPuctPolicy, montyLitePuctPolicy } from './search/puct.ts';

type Side = 'fp32' | 'fp16';
type ModelSpec = { key: Side; label: string; onnx: string; meta: string; visits: number };
type GameResult = '1-0' | '0-1' | '1/2-1/2';
type GameRow = {
  game: number;
  opening: string;
  white: Side;
  black: Side;
  result: GameResult;
  fp16Score: number;
  reason: string;
  plies: number;
  moves: string[];
  finalFen: string;
  fp32ThinkMs: number;
  fp16ThinkMs: number;
  fp32Moves: number;
  fp16Moves: number;
  fp32AvgMs: number;
  fp16AvgMs: number;
};

const OPENING_FENS: { name: string; fen: string }[] = [
  { name: 'start', fen: START_FEN },
  { name: 'italian', fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3' },
  { name: 'ruy_lopez', fen: 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 2 3' },
  { name: 'sicilian_open', fen: 'rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2' },
  { name: 'french_advance', fen: 'rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3' },
  { name: 'caro_kann_advance', fen: 'rnbqkbnr/pp2pppp/2p5/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3' },
  { name: 'qgd', fen: 'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2' },
  { name: 'kings_indian', fen: 'rnbqkb1r/pppppp1p/5np1/8/2PPP3/8/PP3PPP/RNBQKBNR b KQkq - 0 3' },
  { name: 'english', fen: 'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1' },
  { name: 'reti', fen: 'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1' },
  { name: 'scandinavian', fen: 'rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2' },
  { name: 'pirc', fen: 'rnbqkbnr/ppp1pppp/3p4/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2' },
  { name: 'slav', fen: 'rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq d6 0 3' },
  { name: 'benoni', fen: 'rnbqkbnr/pp1ppppp/8/2p5/2PP4/8/PP2PPPP/RNBQKBNR w KQkq c6 0 2' },
  { name: 'dutch', fen: 'rnbqkbnr/ppppp1pp/8/5p2/2PP4/8/PP2PPPP/RNBQKBNR w KQkq f6 0 2' },
  { name: 'four_knights', fen: 'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 4 4' },
];

const params = new URLSearchParams(location.search);
const base = '/models/onnx_deploy_sweep_20260523';
const models: Record<Side, ModelSpec> = {
  fp32: {
    key: 'fp32',
    label: params.get('fp32Label') ?? 'onnxsim_default_fp32',
    onnx: params.get('fp32Onnx') ?? `${base}/bt4_sampled1b_best_onnxsim_default.onnx`,
    meta: params.get('fp32Meta') ?? `${base}/bt4_sampled1b_best_onnxsim_default.meta.json`,
    visits: intParam('fp32Visits', 128, 1, 4096),
  },
  fp16: {
    key: 'fp16',
    label: params.get('fp16Label') ?? 'onnxsim_fp16_keepio',
    onnx: params.get('fp16Onnx') ?? `${base}/bt4_sampled1b_best_onnxsim_fp16_keepio.onnx`,
    meta: params.get('fp16Meta') ?? `${base}/bt4_sampled1b_best_onnxsim_fp16_keepio.meta.json`,
    visits: intParam('fp16Visits', 176, 1, 4096),
  },
};
const openingCount = intParam('openings', 12, 1, 10000);
const openingOffset = intParam('openingOffset', 0, 0, 1000000);
const openingsUrl = params.get('openingsUrl') ?? params.get('openingsFile') ?? '';
const searchMode = (params.get('mode') ?? params.get('searchMode') ?? 'puct').toLowerCase().replace(/[ -]/g, '_');
const maxPlies = intParam('maxPlies', 96, 2, 240);
const batchSize = intParam('batch', 16, 1, 128);
const cpuct = numParam('cpuct', 1.5);
const fpu = numParam('fpu', 0);
const adjudicateCp = numParam('adjudicatePawns', 1.5);
const cacheEntries = intParam('cacheEntries', 8192, 1, 131072);
const yieldEveryMs = intParam('yieldEveryMs', 16, 0, 1000);

function intParam(name: string, fallback: number, lo: number, hi: number): number {
  const n = Math.floor(Number(params.get(name) ?? fallback));
  return Math.max(lo, Math.min(hi, Number.isFinite(n) ? n : fallback));
}
function numParam(name: string, fallback: number): number {
  const n = Number(params.get(name) ?? fallback);
  return Number.isFinite(n) ? n : fallback;
}
function nowMs() { return typeof performance === 'undefined' ? Date.now() : performance.now(); }
function setStatus(text: string) {
  const el = document.getElementById('status');
  if (el) el.textContent = text;
}
async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}
async function loadModel(spec: ModelSpec): Promise<Evaluator> {
  const meta = await loadJson<SquareFormerMeta>(spec.meta);
  const baseEval = await SquareFormerEvaluator.create(spec.onnx, meta);
  return new CachedEvaluator(baseEval, { maxEntries: cacheEntries, includeHistory: true, includeLegalMoves: true, label: spec.label });
}
function materialPawns(board: BoardState): number {
  const values: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  let score = 0;
  for (const piece of board.squares) {
    if (!piece) continue;
    const v = values[piece[1]] ?? 0;
    score += piece[0] === 'w' ? v : -v;
  }
  return score;
}
function resultScore(result: GameResult, side: Side, white: Side): number {
  if (result === '1/2-1/2') return 0.5;
  const sideWon = (result === '1-0' && white === side) || (result === '0-1' && white !== side);
  return sideWon ? 1 : 0;
}
function terminalOrAdjudicated(board: BoardState, historyFens: string[], ply: number, white: Side): { done: boolean; result?: GameResult; reason?: string } {
  const moves = legalMoves(board);
  if (!moves.length) {
    if (inCheck(board)) {
      const result: GameResult = board.turn === 'w' ? '0-1' : '1-0';
      return { done: true, result, reason: 'checkmate' };
    }
    return { done: true, result: '1/2-1/2', reason: 'stalemate' };
  }
  const draw = automaticDrawReason(board, historyFens);
  if (draw) return { done: true, result: '1/2-1/2', reason: draw };
  if (ply >= maxPlies) {
    const mat = materialPawns(board);
    if (Math.abs(mat) >= adjudicateCp) return { done: true, result: mat > 0 ? '1-0' : '0-1', reason: `material@${maxPlies} ${mat.toFixed(1)} pawns` };
    return { done: true, result: '1/2-1/2', reason: `draw@${maxPlies} material ${mat.toFixed(1)}` };
  }
  return { done: false };
}
function searchPolicyForMode() {
  if (searchMode === 'monty' || searchMode === 'monty_lc0both' || searchMode === 'monty_lc0') return montyLitePuctPolicy;
  return classicPuctPolicy;
}
function searchOptionPatchForMode() {
  if (searchMode === 'monty_lc0both' || searchMode === 'monty_lc0') {
    return { cpuctSchedule: 'lc0-log' as const, fpuStrategy: 'lc0-reduction' as const, fpuReduction: 0.33 };
  }
  return {};
}
async function loadOpenings(): Promise<{ name: string; fen: string }[]> {
  if (!openingsUrl) return OPENING_FENS.slice(openingOffset, openingOffset + openingCount);
  const res = await fetch(openingsUrl);
  if (!res.ok) throw new Error(`${openingsUrl}: ${res.status} ${res.statusText}`);
  const text = await res.text();
  const rows: { name: string; fen: string }[] = [];
  for (const raw of text.split(/\r?\n/)) {
    let line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const hash = line.indexOf(' #');
    if (hash >= 0) line = line.slice(0, hash).trim();
    rows.push({ name: `fen_${rows.length.toString().padStart(4, '0')}`, fen: line });
  }
  return rows.slice(openingOffset, openingOffset + openingCount);
}
async function playGame(gameNo: number, opening: { name: string; fen: string }, white: Side, evaluators: Record<Side, Evaluator>): Promise<GameRow> {
  let board = parseFen(opening.fen);
  let historyFens: string[] = [];
  const moves: string[] = [];
  const think: Record<Side, number> = { fp32: 0, fp16: 0 };
  const moveCounts: Record<Side, number> = { fp32: 0, fp16: 0 };
  for (let ply = 0; ply <= maxPlies; ply++) {
    const status = terminalOrAdjudicated(board, historyFens, ply, white);
    if (status.done) {
      const result = status.result ?? '1/2-1/2';
      return {
        game: gameNo, opening: opening.name, white, black: white === 'fp16' ? 'fp32' : 'fp16', result,
        fp16Score: resultScore(result, 'fp16', white), reason: status.reason ?? 'done', plies: ply, moves, finalFen: boardToFen(board),
        fp32ThinkMs: think.fp32, fp16ThinkMs: think.fp16, fp32Moves: moveCounts.fp32, fp16Moves: moveCounts.fp16,
        fp32AvgMs: think.fp32 / Math.max(1, moveCounts.fp32), fp16AvgMs: think.fp16 / Math.max(1, moveCounts.fp16),
      };
    }
    const side: Side = (board.turn === 'w') === (white === 'fp16') ? 'fp16' : 'fp32';
    const legal = legalMoves(board);
    const t0 = nowMs();
    const search = await chooseMove(board, evaluators[side], {
      visits: models[side].visits,
      batchSize,
      cpuct,
      fpu,
      historyFens,
      rootMoves: legal,
      searchPolicy: searchPolicyForMode(),
      ...searchOptionPatchForMode(),
      yieldEveryMs,
      includePv: false,
    });
    const dt = nowMs() - t0;
    think[side] += dt;
    moveCounts[side] += 1;
    const move = search.move ?? legal[0];
    moves.push(moveToUci(move));
    const prevFen = boardToFen(board);
    board = makeMove(board, move);
    historyFens = [prevFen, ...historyFens].slice(0, 16);
  }
  throw new Error('unreachable');
}
function summarize(rows: GameRow[]) {
  const n = rows.length;
  const fp16Score = rows.reduce((s, r) => s + r.fp16Score, 0);
  const wins = rows.filter((r) => r.fp16Score === 1).length;
  const draws = rows.filter((r) => r.fp16Score === 0.5).length;
  const losses = rows.filter((r) => r.fp16Score === 0).length;
  const fp32Moves = rows.reduce((s, r) => s + r.fp32Moves, 0);
  const fp16Moves = rows.reduce((s, r) => s + r.fp16Moves, 0);
  const fp32Ms = rows.reduce((s, r) => s + r.fp32ThinkMs, 0);
  const fp16Ms = rows.reduce((s, r) => s + r.fp16ThinkMs, 0);
  const scoreRate = fp16Score / Math.max(1, n);
  const elo = scoreRate <= 0 ? -Infinity : scoreRate >= 1 ? Infinity : -400 * Math.log10(1 / scoreRate - 1);
  return { games: n, fp16Score, scoreRate, elo, wins, draws, losses, fp32Visits: models.fp32.visits, fp16Visits: models.fp16.visits, fp32AvgMs: fp32Ms / Math.max(1, fp32Moves), fp16AvgMs: fp16Ms / Math.max(1, fp16Moves), fp32Moves, fp16Moves };
}

export async function run() {
  const startedAt = new Date().toISOString();
  setStatus('loading models…');
  // ORT WebGPU cannot reliably create two sessions concurrently in Chromium
  // ("another WebGPU EP inference session is being created"). Load serially so
  // both models get a fair WebGPU session instead of one silently falling back.
  const fp32Eval = await loadModel(models.fp32);
  const fp16Eval = await loadModel(models.fp16);
  const ortDiagnostics = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
  const evaluators: Record<Side, Evaluator> = { fp32: fp32Eval, fp16: fp16Eval };
  const rows: GameRow[] = [];
  let gameNo = 0;
  const selectedOpenings = await loadOpenings();
  for (const opening of selectedOpenings) {
    for (const white of ['fp16', 'fp32'] as Side[]) {
      gameNo += 1;
      setStatus(`game ${gameNo}/${selectedOpenings.length * 2}: ${opening.name}, white=${white}`);
      rows.push(await playGame(gameNo, opening, white, evaluators));
    }
  }
  const result = { startedAt, finishedAt: new Date().toISOString(), config: { models, openingCount: selectedOpenings.length, openingOffset, openingsUrl: openingsUrl || null, searchMode, maxPlies, batchSize, cpuct, fpu, adjudicateCp, cacheEntries, ortDiagnostics }, summary: summarize(rows), games: rows };
  setStatus(JSON.stringify(result.summary, null, 2));
  return result;
}

const api = { run };
(window as unknown as { tinyLeelaTwoModelArena: typeof api }).tinyLeelaTwoModelArena = api;

const autorun = params.get('autorun') === '1' || params.get('run') === '1';
if (autorun) void run().then((r) => console.log('[two-model-arena]', r)).catch((e) => { console.error(e); setStatus(String(e?.stack ?? e)); });
