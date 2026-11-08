import { type BoardState, boardToFen, parseFen, START_FEN } from '../chess/board.ts';
import { type Move, moveToUci } from '../chess/moveCodec.ts';
import { gameTreeToPgn } from '../chess/pgn.ts';
import { gameOutcome } from './engineBattle.ts';
import { GameTree } from './gameTree.ts';

/** Only the selected path, never Analysis's other branches or future moves. */
export interface PlayPosition {
  startFen: string;
  moves: string[];
}

export interface ReplayedPlayPosition {
  startFen: string;
  board: BoardState;
  positions: BoardState[];
  moves: Move[];
  sans: string[];
}

export function playPositionFromTree(tree: GameTree): PlayPosition {
  return { startFen: tree.root.fen, moves: tree.lineTo().slice(1).map((node) => moveToUci(node.move!)) };
}

/** Validate untrusted roots and replay every move, retaining neural/repetition history. */
export function replayPlayPosition(value: unknown): ReplayedPlayPosition {
  if (!value || typeof value !== 'object') throw new Error('Missing position');
  const { startFen, moves } = value as Partial<PlayPosition>;
  if (typeof startFen !== 'string' || startFen.length > 200 || startFen.trim().split(/\s+/).length !== 6) throw new Error('Invalid starting FEN');
  if (!Array.isArray(moves) || moves.length > 2000) throw new Error('Invalid move history');
  const tree = new GameTree(startFen);
  for (const uci of moves) {
    if (typeof uci !== 'string' || !/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci) || !tree.addUci(uci)) throw new Error('Illegal move history');
  }
  const path = tree.lineTo();
  const positions = tree.historyBoards();
  return {
    startFen: tree.root.fen,
    board: positions[positions.length - 1],
    positions,
    moves: path.slice(1).map((node) => node.move!),
    sans: path.slice(1).map((node) => node.san!),
  };
}

export function playFromHereUnavailableReason(tree: GameTree): string | null {
  const reason = gameOutcome(tree.currentBoard(), tree.lineTo().slice(0, -1).map((node) => node.fen))?.reason;
  if (reason === 'insufficientMaterial') return 'insufficient material';
  if (reason === 'threefold') return 'threefold repetition';
  if (reason === 'fiftyMove') return 'fifty-move rule';
  return reason ?? null;
}

/** Use the actual parent board, not the path index (custom roots may start at 37... ). */
export function playMoveLabel(before: BoardState, san: string): string {
  return `${before.fullmove}${before.turn === 'w' ? '.' : '…'} ${san}`;
}

export function playMoveRows(positions: BoardState[], sans: string[]): Array<{ number: number; white: number | null; black: number | null }> {
  const rows: Array<{ number: number; white: number | null; black: number | null }> = [];
  for (let i = 0; i < sans.length; i++) {
    const before = positions[i];
    if (!rows.length || before.turn === 'w') rows.push({ number: before.fullmove, white: null, black: null });
    rows[rows.length - 1][before.turn === 'w' ? 'white' : 'black'] = i;
  }
  return rows;
}

/** The practice boundary is a floor, even when the opponent moves first. */
export function playTakebackPly(positions: BoardState[], human: 'w' | 'b', boundary: number): number {
  let ply = positions.length - 1;
  if (ply <= boundary) return ply;
  do { ply--; } while (ply > boundary && positions[ply].turn !== human);
  return ply;
}

/** Shared by Copy, Download and Analyze; header choice follows the game's root, not its current FEN. */
export function playGameToPgn(startFen: string, moves: Move[], tags: Record<string, string> = {}, result = '*'): string {
  const tree = new GameTree(startFen);
  for (const move of moves) tree.addMove(move);
  const root = boardToFen(parseFen(startFen));
  return gameTreeToPgn(tree, { ...tags, ...(root === START_FEN ? {} : { SetUp: '1', FEN: root }) }, result);
}
