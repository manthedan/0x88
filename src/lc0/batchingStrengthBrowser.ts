import { type BoardState, boardToFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import {
  BATCHING_STRENGTH_BUILD,
  LQO_BLACK_DRAW_SCORE,
  LQO_BLACK_SEARCH_CONTEMPT_LIMIT,
  LQO_CPUCT,
  LQO_FPU_VALUE,
  LQO_WHITE_DRAW_SCORE,
  LQO_WHITE_SEARCH_CONTEMPT_LIMIT,
  PLAY_DRAW_SCORE,
  PLAY_SEARCH_CONTEMPT_LIMIT,
  type StrengthArm,
  type StrengthMatchSpec,
  strengthMatch,
} from './batchingStrengthMatch.ts';
import { lqoQueenOddsOpenings, SMALL_NET_EPD_PATH, type StrengthOpening, scheduleGame, smallNetOpenings } from './batchingStrengthOpenings.ts';
import { strengthSeed } from './batchingStrengthRandom.ts';
import { BATCHING_STRENGTH_GENERATOR_VERSION, type ScoredGame, uciLineToPgn } from './batchingStrengthScore.ts';
import { swapColorsInput, swapColorsUci } from './bt4Engine.ts';
import { type GameResultCode, gameOutcome } from './engineBattle.ts';
import type { Lc0EvaluatorInput } from './onnxEvaluator.ts';

const PROBE_FENS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
  'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2',
];

export interface StrengthGameRecord extends ScoredGame {
  pair: number;
  opening: string;
  startFen: string;
  queenLessColor: 'w' | 'b' | null;
  reason: string;
  plies: number;
  moves: string[];
  bookMoves: string[];
  whiteName: string;
  blackName: string;
  whiteMoveMs: number[];
  blackMoveMs: number[];
  whiteVisits: number[];
  blackVisits: number[];
  pgn: string;
}

export interface PhysicalProbe {
  arm: string;
  logical: number;
  ortPhysicalBatchSize: number | null;
  expected: number;
  ok: boolean;
}

export interface DeterminismProbe {
  arm: string;
  visits: number;
  move1: string | null;
  move2: string | null;
  match: boolean;
}

export interface StrengthSnapshot {
  build: string;
  generatorVersion: number;
  matchId: string;
  runToken: string;
  label: string;
  phase: 'idle' | 'loading' | 'probing' | 'thinking' | 'between' | 'paused' | 'done' | 'error';
  gamesDone: number;
  gamesTarget: number;
  fatal: boolean;
  error: string | null;
  backends: Record<string, string>;
  probes: PhysicalProbe[];
  determinism: DeterminismProbe[];
  lastResult: string | null;
  gameIndex: number | null;
  ply: number | null;
}

interface SearchPayload {
  move?: string | null;
  visits?: number;
  elapsedMs?: number;
  cancelled?: boolean;
}

interface WorkerClient {
  name: string;
  init(): Promise<string>;
  search(input: Lc0EvaluatorInput, fields: Record<string, unknown>): Promise<SearchPayload>;
  evaluateBatch(inputs: Lc0EvaluatorInput[]): Promise<Array<{ timing?: { ortPhysicalBatchSize?: number } }>>;
  reset(): Promise<void>;
}

class FatalStrengthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FatalStrengthError';
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function createWorker(arm: StrengthArm): WorkerClient {
  let worker: Worker | null = null;
  let seq = 0;
  const pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();

  function ensure(): Worker {
    if (worker) return worker;
    worker = new Worker(new URL('./batchingStrengthWorker.ts', import.meta.url), { type: 'module', name: `batching-${arm.name}` });
    worker.addEventListener('message', (event: MessageEvent) => {
      const message = event.data as { id?: number; type?: string; error?: string };
      if (message.type === 'downloadProgress' || message.type === 'searchProgress') return;
      if (message.id === undefined) return;
      const waiter = pending.get(message.id);
      if (!waiter) return;
      pending.delete(message.id);
      if (message.type === 'error') waiter.reject(new Error(message.error ?? 'search worker error'));
      else waiter.resolve(message);
    });
    worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'search worker crashed');
      for (const waiter of pending.values()) waiter.reject(error);
      pending.clear();
    });
    return worker;
  }

  function post<T>(message: Record<string, unknown>, timeoutMs: number): Promise<T> {
    const id = ++seq;
    const target = ensure();
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(id);
        target.postMessage({ type: 'cancel', target: id });
        reject(new Error(`${arm.name} timed out after ${timeoutMs}ms`));
      }, timeoutMs);
      pending.set(id, {
        resolve: (value) => {
          clearTimeout(timer);
          resolve(value as T);
        },
        reject: (error) => {
          clearTimeout(timer);
          reject(error);
        },
      });
      target.postMessage({ ...message, id });
    });
  }

  return {
    name: arm.name,
    async init() {
      const ready = await post<{ backend?: string; ortFallback?: unknown }>(
        {
          type: 'init',
          modelUrl: arm.modelUrl,
          ep: 'webgpu',
          cacheModel: true,
          evalCacheEntries: 2048,
          physicalBatchMode: arm.physicalBatchMode,
          reportDownloadProgress: false,
          requestPersistentModelStorage: true,
        },
        180000,
      );
      const backend = ready.backend ?? '';
      const folded = backend.toLowerCase();
      if (!folded.includes('webgpu') || folded.includes('wasm') || ready.ortFallback) {
        throw new FatalStrengthError(
          `${arm.name} expected strict WebGPU, got ${backend || 'no backend'}${ready.ortFallback ? ` fallback ${JSON.stringify(ready.ortFallback)}` : ''}`,
        );
      }
      return backend;
    },
    async search(input, fields) {
      const response = await post<{ result?: SearchPayload }>({ type: 'search', input, ...fields, randomSeed: strengthSeed(JSON.stringify(input)) }, 180000);
      return response.result ?? {};
    },
    async evaluateBatch(inputs) {
      const response = await post<{ result?: Array<{ timing?: { ortPhysicalBatchSize?: number } }> }>({ type: 'evaluateBatch', inputs }, 180000);
      return response.result ?? [];
    },
    async reset() {
      await post({ type: 'resetSearch' }, 30000);
    },
  };
}

function budgetOf(spec: StrengthMatchSpec): { visits: number; movetimeMs?: number } {
  if (spec.movetimeMs && spec.movetimeMs > 0) return { visits: spec.visits ?? 100000, movetimeMs: spec.movetimeMs };
  return { visits: spec.visits ?? 32 };
}

function searchFields(spec: StrengthMatchSpec, arm: StrengthArm, color: 'w' | 'b', queenLess: 'w' | 'b' | null): Record<string, unknown> {
  const odds = spec.kind === 'lqo' && queenLess === color;
  const shared = {
    ...budgetOf(spec),
    batchSize: arm.batchSize,
    batchPipelineDepth: 1,
    reuseTree: false,
    earlyStop: 'none',
    temperature: 0,
  };
  if (odds && color === 'b') {
    return {
      ...shared,
      drawScore: LQO_BLACK_DRAW_SCORE,
      searchContemptLimit: LQO_BLACK_SEARCH_CONTEMPT_LIMIT,
      cpuct: LQO_CPUCT,
      fpu: LQO_FPU_VALUE,
    };
  }
  if (odds && color === 'w') {
    return {
      ...shared,
      drawScore: LQO_WHITE_DRAW_SCORE,
      searchContemptLimit: LQO_WHITE_SEARCH_CONTEMPT_LIMIT,
      cpuct: LQO_CPUCT,
      fpu: LQO_FPU_VALUE,
    };
  }
  return { ...shared, drawScore: PLAY_DRAW_SCORE, searchContemptLimit: PLAY_SEARCH_CONTEMPT_LIMIT };
}

async function chooseMove(
  client: WorkerClient,
  positions: BoardState[],
  color: 'w' | 'b',
  spec: StrengthMatchSpec,
  arm: StrengthArm,
  queenLess: 'w' | 'b' | null,
): Promise<{ uci: string | null; visits: number; elapsedMs: number }> {
  const swap = spec.kind === 'lqo' && queenLess === 'b' && color === 'b';
  const input: Lc0EvaluatorInput = { positions };
  const sent = swap ? swapColorsInput(input) : input;
  const started = performance.now();
  const result = await client.search(sent, searchFields(spec, arm, color, queenLess));
  if (result.cancelled) throw new Error(`${arm.name} search cancelled`);
  const elapsedMs = Number.isFinite(result.elapsedMs) ? Number(result.elapsedMs) : performance.now() - started;
  const raw = result.move ?? null;
  const uci = raw && swap ? swapColorsUci(raw) : raw;
  return { uci, visits: result.visits ?? 0, elapsedMs };
}

async function loadOpenings(spec: StrengthMatchSpec, openingCount: number): Promise<StrengthOpening[]> {
  if (spec.kind === 'lqo') return lqoQueenOddsOpenings(openingCount);
  const response = await fetch(SMALL_NET_EPD_PATH);
  if (!response.ok) throw new Error(`failed to load ${SMALL_NET_EPD_PATH}: ${response.status}`);
  return smallNetOpenings(await response.text(), openingCount);
}

export interface BatchingStrengthApi {
  startFromParams(params: URLSearchParams): Promise<void>;
  start(options: {
    matchId: string;
    runToken?: string;
    games?: number;
    startIndex?: number;
    movetimeMs?: number;
    maxPlies?: number;
    openingCount?: number;
  }): Promise<void>;
  pause(): void;
  resume(): void;
  snapshot(): StrengthSnapshot;
  pull(fromIndex: number, limit?: number): { games: StrengthGameRecord[] };
}

export function installBatchingStrengthPage(): BatchingStrengthApi {
  const games: StrengthGameRecord[] = [];
  const control = { pause: false, running: false };
  const snapshot: StrengthSnapshot = {
    build: BATCHING_STRENGTH_BUILD,
    generatorVersion: BATCHING_STRENGTH_GENERATOR_VERSION,
    matchId: '',
    runToken: '',
    label: '',
    phase: 'idle',
    gamesDone: 0,
    gamesTarget: 0,
    fatal: false,
    error: null,
    backends: {},
    probes: [],
    determinism: [],
    lastResult: null,
    gameIndex: null,
    ply: null,
  };

  function render(): void {
    const status = document.querySelector('#bs-status');
    const json = document.querySelector('#bs-json');
    const line = `${snapshot.phase} ${snapshot.matchId} ${snapshot.gamesDone}/${snapshot.gamesTarget}${snapshot.error ? ` ERROR ${snapshot.error}` : ''}`;
    if (status) status.textContent = line;
    if (json) json.textContent = JSON.stringify(snapshot);
  }

  function fail(error: unknown, fatal = true): void {
    snapshot.phase = 'error';
    snapshot.fatal = fatal;
    snapshot.error = error instanceof Error ? error.message : String(error);
    control.running = false;
    render();
  }

  async function probeArm(client: WorkerClient, arm: StrengthArm): Promise<void> {
    const evals = await client.evaluateBatch(PROBE_FENS);
    const physical = Number(evals[0]?.timing?.ortPhysicalBatchSize);
    const expected = arm.expectPhysical === 'logical' ? PROBE_FENS.length : arm.expectPhysical;
    const row: PhysicalProbe = {
      arm: arm.name,
      logical: PROBE_FENS.length,
      ortPhysicalBatchSize: Number.isFinite(physical) ? physical : null,
      expected,
      ok: physical === expected,
    };
    snapshot.probes.push(row);
    render();
    if (!row.ok) throw new FatalStrengthError(`${arm.name} physical batch ${row.ortPhysicalBatchSize} != expected ${expected}`);
    const fen = PROBE_FENS[0];
    const fields = {
      visits: 100,
      batchSize: arm.batchSize,
      batchPipelineDepth: 1,
      reuseTree: false,
      earlyStop: 'none',
      temperature: 0,
      searchContemptLimit: 16,
    };
    const once = await client.search(fen, fields);
    await client.reset();
    const twice = await client.search(fen, fields);
    await client.reset();
    const match = !!once.move && once.move === twice.move && once.visits === fields.visits && twice.visits === fields.visits;
    snapshot.determinism.push({ arm: arm.name, visits: fields.visits, move1: once.move ?? null, move2: twice.move ?? null, match });
    render();
    if (!match) throw new FatalStrengthError(`${arm.name} failed seeded repeat-search probe`);
  }

  async function playGame(
    spec: StrengthMatchSpec,
    openings: StrengthOpening[],
    index: number,
    candidate: WorkerClient,
    baseline: WorkerClient,
  ): Promise<StrengthGameRecord> {
    const schedule = scheduleGame(openings.length, index);
    const opening = openings[schedule.openingIndex];
    const whiteClient = schedule.candidateWhite ? candidate : baseline;
    const blackClient = schedule.candidateWhite ? baseline : candidate;
    const whiteArm = schedule.candidateWhite ? spec.candidate : spec.baseline;
    const blackArm = schedule.candidateWhite ? spec.baseline : spec.candidate;
    await candidate.reset();
    await baseline.reset();
    const positions = opening.positions.slice();
    let board = positions[positions.length - 1];
    const priorFens = positions.slice(0, -1).map((position) => boardToFen(position));
    const bookMoves = [...(opening.moves ?? [])];
    const moves: string[] = [];
    const whiteMoveMs: number[] = [];
    const blackMoveMs: number[] = [];
    const whiteVisits: number[] = [];
    const blackVisits: number[] = [];
    let result: GameResultCode = '1/2-1/2';
    let reason = 'max plies';
    while (moves.length < spec.maxPlies) {
      while (control.pause) {
        snapshot.phase = 'paused';
        render();
        await sleep(150);
      }
      snapshot.phase = 'thinking';
      snapshot.gameIndex = index;
      snapshot.ply = bookMoves.length + moves.length;
      render();
      const outcome = gameOutcome(board, priorFens);
      if (outcome) {
        result = outcome.result;
        reason = outcome.reason;
        break;
      }
      const color = board.turn;
      const client = color === 'w' ? whiteClient : blackClient;
      const arm = color === 'w' ? whiteArm : blackArm;
      const choice = await chooseMove(client, positions, color, spec, arm, opening.queenLessColor);
      const legal = legalMoves(board);
      const move = choice.uci ? legal.find((candidateMove) => moveToUci(candidateMove) === choice.uci) : undefined;
      if (!move) {
        result = color === 'w' ? '0-1' : '1-0';
        reason = choice.uci ? `illegal move ${choice.uci}` : 'resigned';
        break;
      }
      if (color === 'w') {
        whiteMoveMs.push(choice.elapsedMs);
        whiteVisits.push(choice.visits);
      } else {
        blackMoveMs.push(choice.elapsedMs);
        blackVisits.push(choice.visits);
      }
      priorFens.push(boardToFen(board));
      board = makeMove(board, move);
      positions.push(board);
      moves.push(choice.uci!);
    }
    if (reason === 'max plies') {
      const outcome = gameOutcome(board, priorFens);
      if (outcome) {
        result = outcome.result;
        reason = outcome.reason;
      }
    }
    const record: StrengthGameRecord = {
      index,
      pair: schedule.pair,
      candidateWhite: schedule.candidateWhite,
      result,
      opening: opening.name,
      startFen: opening.fen,
      queenLessColor: opening.queenLessColor,
      reason,
      plies: moves.length,
      moves,
      bookMoves,
      whiteName: whiteArm.name,
      blackName: blackArm.name,
      whiteMoveMs,
      blackMoveMs,
      whiteVisits,
      blackVisits,
      pgn: uciLineToPgn({
        startFen: opening.fen,
        moves,
        white: whiteArm.name,
        black: blackArm.name,
        result,
        headers: {
          Opening: opening.name,
          BookMoves: bookMoves.join(' '),
          Match: spec.id,
          QueenLess: opening.queenLessColor ?? 'none',
          Visits: String(spec.visits ?? ''),
          MovetimeMs: String(spec.movetimeMs ?? ''),
        },
      }),
    };
    return record;
  }

  async function run(spec: StrengthMatchSpec, gamesTarget: number, startIndex: number, openings: StrengthOpening[]): Promise<void> {
    snapshot.phase = 'loading';
    snapshot.matchId = spec.id;
    snapshot.label = spec.label;
    snapshot.gamesTarget = gamesTarget;
    snapshot.gamesDone = startIndex;
    render();
    const candidate = createWorker(spec.candidate);
    const baseline = createWorker(spec.baseline);
    try {
      snapshot.backends[spec.candidate.name] = await candidate.init();
      snapshot.backends[spec.baseline.name] = await baseline.init();
    } catch (error) {
      if (error instanceof FatalStrengthError) throw error;
      const message = error instanceof Error ? error.message : String(error);
      throw new FatalStrengthError(message);
    }
    snapshot.phase = 'probing';
    render();
    await probeArm(baseline, spec.baseline);
    await probeArm(candidate, spec.candidate);
    for (let index = startIndex; index < gamesTarget; index++) {
      while (control.pause) {
        snapshot.phase = 'paused';
        render();
        await sleep(150);
      }
      const record = await playGame(spec, openings, index, candidate, baseline);
      games.push(record);
      snapshot.gamesDone = record.index + 1;
      snapshot.lastResult = `${record.index} ${record.result} ${record.reason} ${record.plies}pl`;
      snapshot.phase = 'between';
      render();
    }
    snapshot.phase = 'done';
    snapshot.gameIndex = null;
    snapshot.ply = null;
    control.running = false;
    render();
  }

  const api: BatchingStrengthApi = {
    async startFromParams(params) {
      const matchId = params.get('match') ?? '';
      const games = params.get('games');
      const startIndex = params.get('start');
      const movetimeMs = params.get('movetimeMs');
      const maxPlies = params.get('maxPlies');
      const openingCount = params.get('openingCount');
      await api.start({
        matchId,
        runToken: params.get('runToken') ?? undefined,
        games: games ? Number(games) : undefined,
        startIndex: startIndex ? Number(startIndex) : undefined,
        movetimeMs: movetimeMs ? Number(movetimeMs) : undefined,
        maxPlies: maxPlies ? Number(maxPlies) : undefined,
        openingCount: openingCount ? Number(openingCount) : undefined,
      });
    },
    async start(options) {
      if (control.running) return;
      games.length = 0;
      control.running = true;
      control.pause = false;
      snapshot.error = null;
      snapshot.fatal = false;
      snapshot.probes = [];
      snapshot.determinism = [];
      snapshot.runToken = options.runToken ?? crypto.randomUUID();
      try {
        const spec = structuredClone(strengthMatch(options.matchId));
        if (options.movetimeMs && options.movetimeMs > 0) spec.movetimeMs = options.movetimeMs;
        if (options.maxPlies && options.maxPlies > 0) spec.maxPlies = Math.floor(options.maxPlies);
        const openingCount = Math.max(1, Math.floor(options.openingCount ?? spec.openingCount));
        spec.openingCount = openingCount;
        const gamesTarget = Math.max(1, Math.floor(options.games ?? spec.games));
        const startIndex = Math.max(0, Math.floor(options.startIndex ?? 0));
        const openings = await loadOpenings(spec, openingCount);
        if (openings.length < 2) throw new Error(`only ${openings.length} openings`);
        await run(spec, gamesTarget, startIndex, openings);
      } catch (error) {
        fail(error, error instanceof FatalStrengthError);
      }
    },
    pause() {
      control.pause = true;
      render();
    },
    resume() {
      control.pause = false;
      render();
    },
    snapshot() {
      return snapshot;
    },
    pull(fromIndex, limit = 0) {
      const from = Math.max(0, fromIndex);
      const matched = games.filter((game) => game.index >= from);
      return { games: limit > 0 ? matched.slice(0, limit) : matched };
    },
  };

  window.__batchingStrength = api;
  render();
  return api;
}

declare global {
  interface Window {
    __batchingStrength?: BatchingStrengthApi;
  }
}
