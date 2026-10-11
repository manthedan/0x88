import { type BoardState, boardToFen } from '../chess/board.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import type { BrowserUciEngine } from './browserUciEngine.ts';
import { type GameResultCode, gameOutcome } from './engineBattle.ts';
import type { GameTree } from './gameTree.ts';
import type { EngineResourceBroker } from './resourceBroker.ts';

export interface CpuArenaSeat {
  /** Game- and seat-namespaced broker/telemetry identity. */
  id: string;
  name: string;
  engine: BrowserUciEngine;
}
export interface CpuArenaPosition {
  tree: GameTree;
  positions: BoardState[];
  lastUci: string | null;
  ply: number;
}

/** A game has no shared board, TT, worker, or mutable settings with its peers. */
export async function playCpuArenaGame(options: {
  tree: GameTree;
  white: CpuArenaSeat;
  black: CpuArenaSeat;
  broker: EngineResourceBroker;
  signal: AbortSignal;
  beforeMove?: (seat: CpuArenaSeat) => void;
  afterMove?: (seat: CpuArenaSeat, fen: string, move: string | null, elapsedMs: number) => void;
  position?: (position: CpuArenaPosition) => void;
  delay?: () => Promise<void>;
}): Promise<{ result: GameResultCode; reason: string; tree: GameTree }> {
  const { tree, white, black, broker, signal } = options;
  const positions = tree.historyBoards();
  let board = positions[positions.length - 1];
  const priorFens = positions.slice(0, -1).map(boardToFen);
  const cancelled = () => ({ result: '1/2-1/2' as const, reason: 'cancelled', tree });
  for (const seat of [white, black]) {
    await seat.engine.prewarm(signal);
    await seat.engine.newGame(signal);
    if (signal.aborted) return cancelled();
  }
  options.position?.({ tree, positions, lastUci: null, ply: 0 });
  for (let ply = 0; ply < 300; ply++) {
    if (signal.aborted) return cancelled();
    const outcome = gameOutcome(board, priorFens);
    if (outcome) return { ...outcome, tree };
    const seat = board.turn === 'w' ? white : black;
    const lease = await broker.acquire({ engineId: seat.id, signal });
    const fen = boardToFen(board);
    let uci: string | null;
    try {
      options.beforeMove?.(seat);
      const started = performance.now();
      uci = await seat.engine.bestMove(fen, signal);
      if (signal.aborted) return cancelled();
      options.afterMove?.(seat, fen, uci, performance.now() - started);
    } finally {
      lease.release();
    }
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) return { result: board.turn === 'w' ? '0-1' : '1-0', reason: uci ? `illegal ${uci}` : 'resigned', tree };
    priorFens.push(fen);
    board = makeMove(board, move);
    positions.push(board);
    tree.addUci(uci!);
    options.position?.({ tree, positions, lastUci: uci, ply: ply + 1 });
    await options.delay?.();
  }
  return { result: '1/2-1/2', reason: 'max plies', tree };
}
