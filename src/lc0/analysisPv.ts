import { boardToFen, parseFen } from '../chess/board.ts';
import { type Move, moveToUci } from '../chess/moveCodec.ts';
import { legalMoves, makeMove } from '../chess/movegen.ts';
import { moveToSan } from '../chess/san.ts';
import { type GameNode, GameTree } from './gameTree.ts';

export interface PvStep {
  uci: string;
  san: string;
  label: string;
  fen: string;
}

export interface PvPreview {
  rootFen: string;
  steps: PvStep[];
  /** Engines can report an illegal tail, e.g. after mate. Only the legal prefix is usable. */
  truncated: boolean;
}

/** Validate once and freeze the displayed continuation independently of live engine results. */
export function createPvPreview(rootFen: string, pvUci: readonly string[]): PvPreview {
  let board = parseFen(rootFen);
  const steps: PvStep[] = [];
  for (const uci of pvUci) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) break;
    const san = moveToSan(board, move);
    const label = `${board.fullmove}${board.turn === 'w' ? '.' : '…'} ${san}`;
    board = makeMove(board, move);
    steps.push({ uci, san, label, fen: boardToFen(board) });
  }
  return { rootFen, steps, truncated: steps.length !== pvUci.length };
}

/** Validate the entire requested prefix before touching the tree. Never promote existing branches. */
export function insertPvPrefix(tree: GameTree, root: GameNode, pvUci: readonly string[], count: number): GameNode | null {
  if (tree.current !== root || !Number.isInteger(count) || count < 1 || count > pvUci.length) return null;
  // Identity matters: equal FENs can have different repetition/neural histories.
  const path = tree.lineTo(root);
  if (path[0] !== tree.root || path.some((node, i) => i > 0 && !path[i - 1].children.includes(node))) return null;
  let board = parseFen(root.fen);
  const moves: Move[] = [];
  for (const uci of pvUci.slice(0, count)) {
    const move = legalMoves(board).find((candidate) => moveToUci(candidate) === uci);
    if (!move) return null;
    moves.push(move);
    board = makeMove(board, move);
  }
  for (const move of moves) tree.addMove(move);
  return tree.current;
}
