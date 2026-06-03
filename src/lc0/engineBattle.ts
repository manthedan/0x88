import { boardToFen, parseFen, START_FEN, type BoardState } from '../chess/board.ts';
import { automaticDrawReason } from '../chess/drawRules.ts';
import { inCheck, legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToUci } from '../chess/moveCodec.ts';
import type { Lc0PolicyOnlyPlayer } from './policyOnlyPlayer.ts';
import type { Lc0PuctSearcher } from './search.ts';

/**
 * Minimal engine-vs-engine battle harness. The core game loop is engine-
 * agnostic: any opponent (LC0 policy-only, LC0 fixed-visit search, or an
 * external UCI engine such as Stockfish.wasm) is a BattleEngine. LC0 adapters
 * are provided; a Stockfish opponent is wired by implementing the same
 * interface around its UCI loop.
 */

export type GameResultCode = '1-0' | '0-1' | '1/2-1/2';

export interface BattleEngine {
  name: string;
  /** Choose a UCI move given the full position history (root first). Return null to resign. */
  chooseMove(positions: BoardState[]): Promise<string | null> | string | null;
}

export interface GameResult {
  result: GameResultCode;
  reason: string;
  plies: number;
  moves: string[];
  finalFen: string;
}

export interface PlayGameOptions {
  startFen?: string;
  maxPlies?: number;
  /** Abort between plies (and via an engine that honors the signal). */
  signal?: AbortSignal;
}

function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

function terminalResult(board: BoardState): { result: GameResultCode; reason: string } | null {
  if (legalMoves(board).length === 0) {
    if (inCheck(board)) {
      // The side to move is checkmated; the other side wins.
      return { result: board.turn === 'w' ? '0-1' : '1-0', reason: 'checkmate' };
    }
    return { result: '1/2-1/2', reason: 'stalemate' };
  }
  return null;
}

export async function playGame(white: BattleEngine, black: BattleEngine, options: PlayGameOptions = {}): Promise<GameResult> {
  const maxPlies = options.maxPlies ?? 300;
  let board = parseFen(options.startFen ?? START_FEN);
  const positions: BoardState[] = [board];
  // FENs of positions that occurred strictly before the current board, for
  // threefold-repetition detection.
  const priorFens: string[] = [];
  const moves: string[] = [];

  for (let ply = 0; ply < maxPlies; ply++) {
    if (options.signal?.aborted) return { result: '1/2-1/2', reason: 'cancelled', plies: ply, moves, finalFen: boardToFen(board) };
    const terminal = terminalResult(board);
    if (terminal) return { ...terminal, plies: ply, moves, finalFen: boardToFen(board) };
    const drawReason = automaticDrawReason(board, priorFens);
    if (drawReason) return { result: '1/2-1/2', reason: drawReason, plies: ply, moves, finalFen: boardToFen(board) };

    const engine = board.turn === 'w' ? white : black;
    const legal = legalMoves(board);
    let uci: string | null;
    try {
      uci = await engine.chooseMove(positions);
    } catch (error) {
      if (isAbortError(error)) return { result: '1/2-1/2', reason: 'cancelled', plies: ply, moves, finalFen: boardToFen(board) };
      throw error;
    }
    const move = uci ? legal.find((m) => moveToUci(m) === uci) : undefined;
    if (!move) {
      // A resignation or an illegal/missing move forfeits the game.
      const reason = uci ? `illegal move ${uci}` : 'resigned';
      return { result: board.turn === 'w' ? '0-1' : '1-0', reason, plies: ply, moves, finalFen: boardToFen(board) };
    }

    priorFens.push(boardToFen(board));
    board = makeMove(board, move);
    positions.push(board);
    moves.push(moveToUci(move));
  }
  return { result: '1/2-1/2', reason: 'max plies', plies: maxPlies, moves, finalFen: boardToFen(board) };
}

export interface MatchSummary {
  engineA: string;
  engineB: string;
  games: number;
  /** Games actually played (< games if the match was cancelled). */
  played: number;
  aWins: number;
  bWins: number;
  draws: number;
  /** Score for engine A: win=1, draw=0.5 (cancelled games excluded). */
  aScore: number;
  cancelled: boolean;
  results: GameResult[];
}

export interface RunMatchOptions extends PlayGameOptions {
  /** Called after each completed game, for streaming progress. aIsWhite tells which color A had. */
  onGame?: (index: number, total: number, result: GameResult, aIsWhite: boolean) => void;
}

/**
 * Play a match alternating colors so neither engine keeps the white advantage.
 * Even games: A is white. Odd games: B is white. A cancelled game ends the match.
 */
export async function runMatch(engineA: BattleEngine, engineB: BattleEngine, games: number, options: RunMatchOptions = {}): Promise<MatchSummary> {
  let aWins = 0;
  let bWins = 0;
  let draws = 0;
  let played = 0;
  let cancelled = false;
  const results: GameResult[] = [];
  for (let i = 0; i < games; i++) {
    if (options.signal?.aborted) { cancelled = true; break; }
    const aIsWhite = i % 2 === 0;
    const result = await playGame(aIsWhite ? engineA : engineB, aIsWhite ? engineB : engineA, options);
    if (result.reason === 'cancelled') { cancelled = true; break; }
    results.push(result);
    played += 1;
    if (result.result === '1/2-1/2') draws += 1;
    else {
      const whiteWon = result.result === '1-0';
      if (whiteWon === aIsWhite) aWins += 1;
      else bWins += 1;
    }
    options.onGame?.(i, games, result, aIsWhite);
  }
  return { engineA: engineA.name, engineB: engineB.name, games, played, aWins, bWins, draws, aScore: aWins + draws * 0.5, cancelled, results };
}

// --- LC0 engine adapters ---

export function lc0PolicyBattleEngine(player: Lc0PolicyOnlyPlayer, name = 'lc0-policy'): BattleEngine {
  return {
    name,
    async chooseMove(positions) {
      const choice = await player.chooseMove({ positions });
      return choice.move ?? null;
    },
  };
}

export function lc0SearchBattleEngine(searcher: Lc0PuctSearcher, visits: number, name = `lc0-search-${visits}`): BattleEngine {
  return {
    name,
    async chooseMove(positions) {
      const result = await searcher.search({ positions }, { visits });
      return result.move ?? null;
    },
  };
}
