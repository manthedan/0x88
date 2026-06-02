import { boardToFen, parseFen, START_FEN, type BoardState } from './chess/board.ts';
import { inCheck, legalMoves, makeMove } from './chess/movegen.ts';
import { moveToUci } from './chess/moveCodec.ts';
import { automaticDrawReason } from './chess/drawRules.ts';
import { BrokeredEvaluator, CachedEvaluator, type Evaluator } from './nn/evaluator.ts';
import { SquareFormerEvaluator, type SquareFormerMeta } from './nn/squareformerEvaluator.ts';
import { collectOrtRuntimeDiagnostics } from './nn/ortRuntime.ts';
import { chooseMove, classicPuctPolicy, montyLitePuctPolicy } from './search/puct.ts';

type ModelSpec = { id: string; label?: string; onnx: string; meta: string; visits?: number; valueWdlAuxHead?: string; valueWdlAuxWeight?: number; valueWdlBlendMode?: 'constant' | 'confidence'; valueWdlBaseTemp?: number; valueWdlAuxTemp?: number };
type PlayerSide = 'a' | 'b';
type GameResult = '1-0' | '0-1' | '1/2-1/2';
type PairJob = { mode: string; a: ModelSpec; b: ModelSpec };
type ArenaCheckpointKind = 'game' | 'pair';
type ArenaCheckpointPayload = {
  kind: ArenaCheckpointKind;
  at: string;
  mode: string;
  pairIndex: number;
  pairTotal: number;
  a: string;
  b: string;
  game?: GameRow;
  pair?: PairResult;
  partialSummary?: ReturnType<typeof summarizePair>;
};
type GameRow = {
  game: number;
  opening: string;
  white: string;
  black: string;
  result: GameResult;
  aScore: number;
  reason: string;
  plies: number;
  moves: string[];
  finalFen: string;
  aThinkMs: number;
  bThinkMs: number;
  aMoves: number;
  bMoves: number;
  aAvgMs: number;
  bAvgMs: number;
};
type PairResult = { mode: string; a: ModelSpec; b: ModelSpec; startedAt: string; finishedAt: string; summary: ReturnType<typeof summarizePair>; games: GameRow[] };

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
const defaultModelBase = params.get('modelBase') ?? '/models/tg_wdl_roundrobin_20260529';
const defaultModels: ModelSpec[] = [
  { id: 'control', label: 'control', onnx: `${defaultModelBase}/h7_lc0rep_control.onnx`, meta: `${defaultModelBase}/h7_lc0rep_control.meta.json` },
  { id: 'mix50', label: 'mix50', onnx: `${defaultModelBase}/tg_mix50_100m.onnx`, meta: `${defaultModelBase}/tg_mix50_100m.meta.json` },
  { id: 'lc0wdl', label: 'lc0wdl', onnx: `${defaultModelBase}/tg_lc0wdl_100m.onnx`, meta: `${defaultModelBase}/tg_lc0wdl_100m.meta.json` },
  { id: 'sf18wdl', label: 'sf18wdl', onnx: `${defaultModelBase}/tg_sf18wdl_100m.onnx`, meta: `${defaultModelBase}/tg_sf18wdl_100m.meta.json` },
];
const visits = intParam('visits', 256, 1, 4096);
const models = parseModels().map((m) => ({ ...m, label: m.label ?? m.id, visits: intFinite(m.visits, visits, 1, 4096) }));
const modes = (params.get('modes') ?? params.get('mode') ?? 'puct,monty_lc0both').split(',').map((s) => s.trim().toLowerCase().replace(/[ -]/g, '_')).filter(Boolean);
const openingCount = intParam('openings', 32, 1, 10000);
const openingOffset = intParam('openingOffset', 0, 0, 1000000);
const openingsUrl = params.get('openingsUrl') ?? params.get('openingsFile') ?? '';
const maxPlies = intParam('maxPlies', 128, 2, 240);
const batchSize = intParam('batch', 64, 1, 256);
const brokerBatchSize = intParam('brokerBatch', 0, 0, 1024);
const brokerWaitMs = intParam('brokerWaitMs', 0, 0, 1000);
const cpuct = numParam('cpuct', 1.5);
const fpu = numParam('fpu', 0);
const adjudicateCp = numParam('adjudicatePawns', 1.5);
const cacheEntries = intParam('cacheEntries', 131072, 1, 1048576);
const yieldEveryMs = intParam('yieldEveryMs', 0, 0, 1000);
const includeMoves = params.get('includeMoves') !== '0';

function intFinite(value: unknown, fallback: number, lo: number, hi: number): number {
  const n = Math.floor(Number(value ?? fallback));
  return Math.max(lo, Math.min(hi, Number.isFinite(n) ? n : fallback));
}
function intParam(name: string, fallback: number, lo: number, hi: number): number { return intFinite(params.get(name), fallback, lo, hi); }
function numParam(name: string, fallback: number): number {
  const n = Number(params.get(name) ?? fallback);
  return Number.isFinite(n) ? n : fallback;
}
function nowMs() { return typeof performance === 'undefined' ? Date.now() : performance.now(); }
function labelOf(model: ModelSpec): string { return model.label ?? model.id; }
function setStatus(text: string) {
  const el = document.getElementById('status');
  if (el) el.textContent = text;
}
function emitCheckpoint(payload: ArenaCheckpointPayload) {
  const hook = (window as unknown as { __tinyLeelaArenaCheckpoint?: (payload: ArenaCheckpointPayload) => void }).__tinyLeelaArenaCheckpoint;
  if (typeof hook !== 'function') return;
  try { hook(payload); } catch (err) { console.warn('[multi-model-arena] checkpoint hook failed', err); }
}
function parseModels(): ModelSpec[] {
  const raw = params.get('modelsJson') ?? params.get('models');
  if (!raw) return defaultModels;
  const decoded = JSON.parse(raw) as ModelSpec[];
  if (!Array.isArray(decoded) || decoded.length < 2) throw new Error('modelsJson must be an array with at least two model specs');
  for (const m of decoded) {
    if (!m.id || !m.onnx || !m.meta) throw new Error(`bad model spec: ${JSON.stringify(m)}`);
  }
  return decoded;
}
function pairJobs(): PairJob[] {
  const jobs: PairJob[] = [];
  for (const mode of modes) {
    for (let i = 0; i < models.length; i++) {
      for (let j = i + 1; j < models.length; j++) jobs.push({ mode, a: models[i], b: models[j] });
    }
  }
  return jobs;
}
async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}
async function loadModel(spec: ModelSpec): Promise<Evaluator> {
  const meta = await loadJson<SquareFormerMeta>(spec.meta);
  const baseEval = await SquareFormerEvaluator.create(spec.onnx, meta);
  const cached = new CachedEvaluator(baseEval, { maxEntries: cacheEntries, includeHistory: true, includeLegalMoves: true, label: labelOf(spec) });
  return brokerBatchSize > 0 ? new BrokeredEvaluator(cached, { maxBatchSize: brokerBatchSize, maxWaitMs: brokerWaitMs, label: `${labelOf(spec)}-broker` }) : cached;
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
function resultScore(result: GameResult, side: PlayerSide, white: PlayerSide): number {
  if (result === '1/2-1/2') return 0.5;
  const sideWon = (result === '1-0' && white === side) || (result === '0-1' && white !== side);
  return sideWon ? 1 : 0;
}
function terminalOrAdjudicated(board: BoardState, historyFens: string[], ply: number): { done: boolean; result?: GameResult; reason?: string } {
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
function searchPolicyForMode(mode: string) {
  if (mode === 'monty' || mode === 'monty_lc0both' || mode === 'monty_lc0') return montyLitePuctPolicy;
  return classicPuctPolicy;
}
function searchOptionPatchForMode(mode: string) {
  if (mode === 'monty_lc0both' || mode === 'monty_lc0') return { cpuctSchedule: 'lc0-log' as const, fpuStrategy: 'lc0-reduction' as const, fpuReduction: 0.33 };
  return {};
}
function searchOptionPatchForModel(model: ModelSpec) {
  return {
    ...(model.valueWdlAuxHead ? { valueWdlAuxHead: model.valueWdlAuxHead } : {}),
    ...(Number.isFinite(Number(model.valueWdlAuxWeight)) ? { valueWdlAuxWeight: Number(model.valueWdlAuxWeight) } : {}),
    ...(model.valueWdlBlendMode ? { valueWdlBlendMode: model.valueWdlBlendMode } : {}),
    ...(Number.isFinite(Number(model.valueWdlBaseTemp)) ? { valueWdlBaseTemp: Number(model.valueWdlBaseTemp) } : {}),
    ...(Number.isFinite(Number(model.valueWdlAuxTemp)) ? { valueWdlAuxTemp: Number(model.valueWdlAuxTemp) } : {}),
  };
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
async function playGame(gameNo: number, opening: { name: string; fen: string }, white: PlayerSide, job: PairJob, evaluators: Map<string, Evaluator>): Promise<GameRow> {
  let board = parseFen(opening.fen);
  let historyFens: string[] = [];
  const moves: string[] = [];
  const think: Record<PlayerSide, number> = { a: 0, b: 0 };
  const moveCounts: Record<PlayerSide, number> = { a: 0, b: 0 };
  for (let ply = 0; ply <= maxPlies; ply++) {
    const status = terminalOrAdjudicated(board, historyFens, ply);
    if (status.done) {
      const result = status.result ?? '1/2-1/2';
      return {
        game: gameNo,
        opening: opening.name,
        white: white === 'a' ? job.a.id : job.b.id,
        black: white === 'a' ? job.b.id : job.a.id,
        result,
        aScore: resultScore(result, 'a', white),
        reason: status.reason ?? 'done',
        plies: ply,
        moves: includeMoves ? moves : [],
        finalFen: includeMoves ? boardToFen(board) : '',
        aThinkMs: think.a,
        bThinkMs: think.b,
        aMoves: moveCounts.a,
        bMoves: moveCounts.b,
        aAvgMs: think.a / Math.max(1, moveCounts.a),
        bAvgMs: think.b / Math.max(1, moveCounts.b),
      };
    }
    const side: PlayerSide = (board.turn === 'w') === (white === 'a') ? 'a' : 'b';
    const model = side === 'a' ? job.a : job.b;
    const evaluator = evaluators.get(model.id);
    if (!evaluator) throw new Error(`evaluator not loaded: ${model.id}`);
    const legal = legalMoves(board);
    const t0 = nowMs();
    const search = await chooseMove(board, evaluator, {
      visits: intFinite(model.visits, visits, 1, 4096),
      batchSize,
      cpuct,
      fpu,
      historyFens,
      rootMoves: legal,
      searchPolicy: searchPolicyForMode(job.mode),
      ...searchOptionPatchForMode(job.mode),
      ...searchOptionPatchForModel(model),
      yieldEveryMs,
      includePv: false,
    });
    const dt = nowMs() - t0;
    think[side] += dt;
    moveCounts[side] += 1;
    const move = search.move ?? legal[0];
    if (includeMoves) moves.push(moveToUci(move));
    const prevFen = boardToFen(board);
    board = makeMove(board, move);
    historyFens = [prevFen, ...historyFens].slice(0, 16);
  }
  throw new Error('unreachable');
}
function summarizePair(rows: GameRow[], job: PairJob) {
  const n = rows.length;
  const aScore = rows.reduce((s, r) => s + r.aScore, 0);
  const wins = rows.filter((r) => r.aScore === 1).length;
  const draws = rows.filter((r) => r.aScore === 0.5).length;
  const losses = rows.filter((r) => r.aScore === 0).length;
  const aMoves = rows.reduce((s, r) => s + r.aMoves, 0);
  const bMoves = rows.reduce((s, r) => s + r.bMoves, 0);
  const aMs = rows.reduce((s, r) => s + r.aThinkMs, 0);
  const bMs = rows.reduce((s, r) => s + r.bThinkMs, 0);
  const scoreRate = aScore / Math.max(1, n);
  const elo = scoreRate <= 0 ? -Infinity : scoreRate >= 1 ? Infinity : -400 * Math.log10(1 / scoreRate - 1);
  return { mode: job.mode, a: job.a.id, b: job.b.id, games: n, aScore, scoreRate, elo, wins, draws, losses, aVisits: job.a.visits ?? visits, bVisits: job.b.visits ?? visits, aAvgMs: aMs / Math.max(1, aMoves), bAvgMs: bMs / Math.max(1, bMoves), aMoves, bMoves };
}
function emptyStanding() { return { games: 0, score: 0, wins: 0, draws: 0, losses: 0 }; }
function addGame(standings: Record<string, ReturnType<typeof emptyStanding>>, modelId: string, score: number) {
  const s = standings[modelId] ??= emptyStanding();
  s.games += 1; s.score += score;
  if (score === 1) s.wins += 1; else if (score === 0.5) s.draws += 1; else s.losses += 1;
}
function summarizeRoundRobin(results: PairResult[]) {
  const byMode: Record<string, { standings: Record<string, ReturnType<typeof emptyStanding>>; pairs: unknown[]; table?: unknown[] }> = {};
  for (const rec of results) {
    const m = byMode[rec.mode] ??= { standings: {}, pairs: [] };
    for (const g of rec.games) {
      addGame(m.standings, rec.a.id, g.aScore);
      addGame(m.standings, rec.b.id, 1 - g.aScore);
    }
    m.pairs.push(rec.summary);
  }
  for (const m of Object.values(byMode)) {
    m.table = Object.entries(m.standings).map(([id, s]) => ({ id, ...s, scoreRate: s.games ? s.score / s.games : 0 })).sort((a, b) => b.scoreRate - a.scoreRate || b.score - a.score);
  }
  return byMode;
}
async function runPair(job: PairJob, idx: number, total: number, openings: { name: string; fen: string }[], evaluators: Map<string, Evaluator>): Promise<PairResult> {
  const startedAt = new Date().toISOString();
  const rows: GameRow[] = [];
  let gameNo = 0;
  for (const opening of openings) {
    for (const white of ['a', 'b'] as PlayerSide[]) {
      gameNo += 1;
      setStatus(`pair ${idx + 1}/${total} ${job.mode} ${job.a.id} vs ${job.b.id}: game ${gameNo}/${openings.length * 2} ${opening.name}, white=${white === 'a' ? job.a.id : job.b.id}`);
      const game = await playGame(gameNo, opening, white, job, evaluators);
      rows.push(game);
      emitCheckpoint({
        kind: 'game',
        at: new Date().toISOString(),
        mode: job.mode,
        pairIndex: idx,
        pairTotal: total,
        a: job.a.id,
        b: job.b.id,
        game,
        partialSummary: summarizePair(rows, job),
      });
    }
  }
  const finishedAt = new Date().toISOString();
  const pair = { mode: job.mode, a: job.a, b: job.b, startedAt, finishedAt, summary: summarizePair(rows, job), games: rows };
  emitCheckpoint({
    kind: 'pair',
    at: finishedAt,
    mode: job.mode,
    pairIndex: idx,
    pairTotal: total,
    a: job.a.id,
    b: job.b.id,
    pair,
    partialSummary: pair.summary,
  });
  return pair;
}
export async function run() {
  const startedAt = new Date().toISOString();
  const jobs = pairJobs();
  setStatus(`loading ${models.length} models serially…`);
  const evaluators = new Map<string, Evaluator>();
  for (let i = 0; i < models.length; i++) {
    setStatus(`loading model ${i + 1}/${models.length}: ${models[i].id}`);
    evaluators.set(models[i].id, await loadModel(models[i]));
  }
  const ortDiagnostics = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
  const openings = await loadOpenings();
  const results: PairResult[] = [];
  for (let i = 0; i < jobs.length; i++) results.push(await runPair(jobs[i], i, jobs.length, openings, evaluators));
  const summary = summarizeRoundRobin(results);
  const evaluatorMetrics = Object.fromEntries([...evaluators.entries()].map(([id, ev]) => [id, typeof (ev as Evaluator & { metrics?: () => unknown }).metrics === 'function' ? (ev as Evaluator & { metrics: () => unknown }).metrics() : null]));
  const result = { startedAt, finishedAt: new Date().toISOString(), config: { models, modes, jobs: jobs.map((j) => ({ mode: j.mode, a: j.a.id, b: j.b.id })), openingCount: openings.length, openingOffset, openingsUrl: openingsUrl || null, visits, maxPlies, batchSize, brokerBatchSize, brokerWaitMs, cpuct, fpu, adjudicateCp, cacheEntries, includeMoves, ortDiagnostics, evaluatorMetrics }, summary, results };
  setStatus(JSON.stringify(summary, null, 2));
  return result;
}

const api = { run };
(window as unknown as { tinyLeelaMultiModelArena: typeof api }).tinyLeelaMultiModelArena = api;

const autorun = params.get('autorun') === '1' || params.get('run') === '1';
if (autorun) void run().then((r) => console.log('[multi-model-arena]', r)).catch((e) => { console.error(e); setStatus(String(e?.stack ?? e)); });
