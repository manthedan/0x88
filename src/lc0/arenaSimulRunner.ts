import { boardToFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { BUILTIN_ARENA_OPENINGS } from './arenaOpenings.ts';
import { ArenaSimulBroker } from './arenaSimulBroker.ts';
import { gameOutcome } from './engineBattle.ts';
import type { Lc0EvaluationProvider } from './onnxEvaluator.ts';
import { Lc0PuctSearcher } from './search.ts';

export const ARENA_SIMUL_PROTOCOL = 'arena-simul-1';
export const SIMUL_OPENINGS = [1, 4, 7, 8].map((index) => BUILTIN_ARENA_OPENINGS[index]);
// Deterministic Arena-style search. ScLimit/temperature/Gumbel sampling are off:
// seeds are identity metadata, not a worker-global Math.random override (unsafe across games).
export const SIMUL_SEARCH_OPTIONS = {
  batchSize: 1,
  batchPipelineDepth: 1,
  temperature: 0,
  drawScore: 0,
  searchContemptLimit: 0,
  contemptElo: 0,
  reuseTree: false,
} as const;

export interface SimulGame {
  id: number;
  opening: string;
  bookMoves: string[];
  aIsWhite: boolean;
  seed: number;
  moves: string[];
  visits: number[];
  moveMs: number[];
  result: string;
  reason: string;
  finalFen: string;
}

export interface SimulProgress {
  game: number;
  ply: number;
  completed: number;
  fen: string;
}

export interface SimulTrialOptions {
  concurrency: number;
  visits: number;
  maxPlies: number;
  signal?: AbortSignal;
  onProgress?: (progress: SimulProgress) => void;
  /** Controller owns generations via AbortControllers; cancellation never scores a result. */
  controllers?: Map<number, AbortController>;
}

export function validateSimulTrial(options: SimulTrialOptions): void {
  if (
    ![1, 2, 4, 8].includes(options.concurrency) ||
    !Number.isInteger(options.visits) ||
    options.visits < 1 ||
    options.visits > 128 ||
    !Number.isInteger(options.maxPlies) ||
    options.maxPlies < 1 ||
    options.maxPlies > 240
  )
    throw new Error('Invalid bounded simul trial');
  if (new Set(SIMUL_OPENINGS.map((opening) => opening.fen)).size !== 4) throw new Error('Simul needs four distinct openings');
}

export async function runSimulTrial(provider: Lc0EvaluationProvider, options: SimulTrialOptions) {
  validateSimulTrial(options);
  const broker = new ArenaSimulBroker(provider, options.concurrency, 8, options.concurrency === 1 ? 0 : 1);
  const controllers = options.controllers ?? new Map<number, AbortController>();
  if (controllers.size) throw new Error('Previous game generation still active');
  const games: SimulGame[] = [];
  const cancelled: number[] = [];
  let next = 0;
  let completed = 0;
  let failed = false;
  const abortAll = () => {
    for (const controller of controllers.values()) controller.abort();
  };
  options.signal?.addEventListener('abort', abortAll, { once: true });
  const started = performance.now();

  async function play(id: number): Promise<void> {
    const controller = new AbortController();
    controllers.set(id, controller);
    if (options.signal?.aborted) controller.abort();
    const searcher = new Lc0PuctSearcher(broker.client(String(id), controller.signal));
    const opening = SIMUL_OPENINGS[Math.floor(id / 2)];
    const positions = [...opening.positions!];
    const record: SimulGame = {
      id,
      opening: opening.name,
      bookMoves: [...opening.moves!],
      aIsWhite: id % 2 === 0,
      seed: 1,
      moves: [],
      visits: [],
      moveMs: [],
      result: '',
      reason: '',
      finalFen: opening.fen,
    };
    try {
      for (let ply = 0; ; ply++) {
        controller.signal.throwIfAborted();
        const board = positions.at(-1)!;
        const outcome = gameOutcome(board, positions.slice(0, -1).map(boardToFen));
        if (outcome || ply === options.maxPlies) {
          Object.assign(record, outcome ?? { result: '1/2-1/2', reason: 'max plies' });
          record.finalFen = boardToFen(board);
          games[id] = record;
          completed++;
          options.onProgress?.({ game: id, ply, completed, fen: record.finalFen });
          return;
        }
        const start = performance.now();
        const choice = await searcher.search(
          { positions },
          {
            ...SIMUL_SEARCH_OPTIONS,
            visits: options.visits,
            signal: controller.signal,
          },
        );
        controller.signal.throwIfAborted();
        if (choice.visits !== options.visits) throw new Error(`Game ${id}: fixed visit budget changed`);
        const move = legalMoves(board).find((candidate) => moveToUci(candidate) === choice.move);
        if (!move) throw new Error(`Game ${id}: illegal or missing search move`);
        record.moveMs.push(performance.now() - start);
        record.visits.push(choice.visits);
        record.moves.push(choice.move!);
        positions.push(makeMove(board, move));
        options.onProgress?.({ game: id, ply: ply + 1, completed, fen: boardToFen(positions.at(-1)!) });
      }
    } catch (error) {
      if (!controller.signal.aborted || options.signal?.aborted || failed) throw error;
      cancelled.push(id);
    } finally {
      searcher.resetTree();
      controllers.delete(id);
    }
  }

  const lanes = Array.from({ length: options.concurrency }, async () => {
    while (!failed && next < 8) {
      const id = next++;
      try {
        await play(id);
      } catch (error) {
        failed = true;
        abortAll();
        throw error;
      }
    }
  });
  try {
    // First failure cancels siblings, but await their settlement before disposing the one session.
    await Promise.all(lanes).catch(async (error) => {
      abortAll();
      await Promise.allSettled(lanes);
      throw error;
    });
    const elapsedMs = performance.now() - started;
    const metrics = broker.metrics();
    const latencies = games.flatMap((game) => game.moveMs).sort((a, b) => a - b);
    const visits = games.reduce((sum, game) => sum + game.visits.reduce((a, b) => a + b, 0), 0);
    return {
      concurrency: options.concurrency,
      elapsedMs,
      completed,
      games: games.filter(Boolean),
      cancelled,
      gamesPerHour: (completed * 3_600_000) / elapsedMs,
      evalsPerSecond: (metrics.evaluations * 1000) / elapsedMs,
      visitsPerSecond: (visits * 1000) / elapsedMs,
      p95MoveMs: latencies[Math.ceil(latencies.length * 0.95) - 1] ?? 0,
      broker: metrics,
    };
  } finally {
    options.signal?.removeEventListener('abort', abortAll);
    await broker.close();
  }
}
