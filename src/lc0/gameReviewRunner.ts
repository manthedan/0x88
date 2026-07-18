import { parseFen } from '../chess/board.ts';
import { inCheck, legalMoves } from '../chess/movegen.ts';
import { evalBarWhitePercent } from './analysisFormat.ts';
import { reviewGame, type GameReview } from './gameReview.ts';

export interface GameReviewEngineLine {
  scoreCp?: number;
  mateIn?: number;
  pvUci: string[];
}

export interface GameReviewEngine {
  analyze(fen: string, options: { multipv: number; depth: number; signal?: AbortSignal }): Promise<GameReviewEngineLine[]>;
}

export interface GameReviewRunnerNode {
  fen: string;
  san?: string | null;
  uci?: string | null;
}

export interface GameReviewProgress {
  position: number;
  total: number;
}

export interface RunGameReviewOptions {
  nodes: readonly GameReviewRunnerNode[];
  engine: GameReviewEngine;
  depth: number;
  signal?: AbortSignal;
  onProgress?: (progress: GameReviewProgress) => void;
}

export function reviewWinWhite(fen: string, info: Pick<GameReviewEngineLine, 'scoreCp' | 'mateIn'> | undefined): number {
  if (!info) return 0.5;
  const turn = parseFen(fen).turn;
  const whiteCp = info.scoreCp === undefined ? undefined : turn === 'w' ? info.scoreCp : -info.scoreCp;
  const whiteMate = info.mateIn === undefined ? undefined : turn === 'w' ? info.mateIn : -info.mateIn;
  return evalBarWhitePercent(whiteCp, whiteMate) / 100;
}

export async function runGameReview(options: RunGameReviewOptions): Promise<GameReview> {
  if (options.nodes.length < 2) throw new Error('Review needs a game with at least one move');
  const positions = [];
  for (let index = 0; index < options.nodes.length; index += 1) {
    if (options.signal?.aborted) throw new DOMException('Review stopped', 'AbortError');
    options.onProgress?.({ position: index + 1, total: options.nodes.length });
    const fen = options.nodes[index].fen;
    const board = parseFen(fen);
    const legal = legalMoves(board).length;
    if (legal === 0) {
      positions.push({
        winWhite: inCheck(board) ? (board.turn === 'w' ? 0 : 1) : 0.5,
        bestUci: null,
        legalMoves: 0,
      });
      continue;
    }
    const lines = await options.engine.analyze(fen, {
      multipv: 1,
      depth: options.depth,
      signal: options.signal,
    });
    if (options.signal?.aborted) throw new DOMException('Review stopped', 'AbortError');
    if (!lines[0]?.pvUci.length || (lines[0].scoreCp === undefined && lines[0].mateIn === undefined)) {
      throw new Error(`Engine returned no scored line for position ${index + 1}`);
    }
    positions.push({
      winWhite: reviewWinWhite(fen, lines[0]),
      bestUci: lines[0]?.pvUci[0] ?? null,
      legalMoves: legal,
    });
  }
  const moves = options.nodes.slice(1).map((node) => ({ san: node.san ?? '?', uci: node.uci ?? '' }));
  return reviewGame(positions, moves, parseFen(options.nodes[0].fen).turn);
}
