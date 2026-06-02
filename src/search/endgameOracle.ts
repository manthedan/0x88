import { boardToFen, type BoardState, type Color, type PieceRole } from '../chess/board.ts';
import { inCheck, legalMoves, makeMove } from '../chess/movegen.ts';
import type { Move } from '../chess/moveCodec.ts';

export interface EndgameOracleMove {
  move: Move;
  /** Human-readable source label for UI/telemetry. */
  source: 'KQK endgame oracle';
  /** Forced mate distance in plies after the selected move, when proven within the search horizon. */
  mateInPlies: number | null;
}

interface KqkInfo {
  attacker: Color;
  defender: Color;
  attackerKing: number;
  queen: number;
  defenderKing: number;
}

const KQK_MAX_MATE_PLIES = 3;

type MemoValue = number | null;

type PieceOnSquare = { sq: number; color: Color; role: PieceRole };

function pieces(board: BoardState): PieceOnSquare[] {
  const out: PieceOnSquare[] = [];
  for (let sq = 0; sq < 64; sq++) {
    const piece = board.squares[sq];
    if (!piece) continue;
    out.push({ sq, color: piece[0] as Color, role: piece[1] as PieceRole });
  }
  return out;
}

function detectKqk(board: BoardState): KqkInfo | null {
  const ps = pieces(board);
  if (ps.length !== 3) return null;
  const queens = ps.filter((p) => p.role === 'q');
  const kings = ps.filter((p) => p.role === 'k');
  if (queens.length !== 1 || kings.length !== 2) return null;
  const attacker = queens[0].color;
  const defender = attacker === 'w' ? 'b' : 'w';
  const attackerKing = kings.find((p) => p.color === attacker)?.sq;
  const defenderKing = kings.find((p) => p.color === defender)?.sq;
  if (attackerKing === undefined || defenderKing === undefined) return null;
  return { attacker, defender, attackerKing, queen: queens[0].sq, defenderKing };
}

function isDefenderCheckmated(board: BoardState, attacker: Color, legal = legalMoves(board)): boolean {
  return board.turn !== attacker && legal.length === 0 && inCheck(board, board.turn);
}

function key(board: BoardState): string {
  // KQK has no castling/en-passant/pawns. Halfmove is included because the FE
  // enforces a 100-halfmove automatic draw and late KQK conversions must mate
  // before that counter expires.
  return `${boardToFen(board).split(' ').slice(0, 2).join(' ')} ${Math.min(100, board.halfmove)}`;
}

function moveOrderingScore(board: BoardState, move: Move, attacker: Color): number {
  const child = makeMove(board, move);
  const mateNow = isDefenderCheckmated(child, attacker);
  if (mateNow) return Number.NEGATIVE_INFINITY;
  return fallbackKqkScore(child, attacker);
}

function orderedMoves(board: BoardState, moves: Move[], attacker: Color): Move[] {
  // Attacker tries most mating-shaped moves first; defender tries the most
  // evasive moves first so failed proofs cut off early.
  const sign = board.turn === attacker ? 1 : -1;
  return [...moves].sort((a, b) => sign * (moveOrderingScore(board, a, attacker) - moveOrderingScore(board, b, attacker)));
}

function mateDistance(board: BoardState, attacker: Color, depth: number, memo: Map<string, MemoValue>, proven: Map<string, number>): number | null {
  const legal = legalMoves(board);
  if (isDefenderCheckmated(board, attacker, legal)) return 0;
  if (board.halfmove >= 100 || depth <= 0 || legal.length === 0) return null;

  const stateKey = key(board);
  const provenDistance = proven.get(stateKey);
  if (provenDistance !== undefined) return provenDistance <= depth ? provenDistance : null;

  const memoKey = `${stateKey} d${depth}`;
  if (memo.has(memoKey)) return memo.get(memoKey)!;

  let result: number | null;
  if (board.turn === attacker) {
    result = null;
    for (const move of orderedMoves(board, legal, attacker)) {
      const childDistance = mateDistance(makeMove(board, move), attacker, depth - 1, memo, proven);
      if (childDistance === null) continue;
      const distance = childDistance + 1;
      if (result === null || distance < result) result = distance;
    }
  } else {
    // Defender chooses the longest line and refutes the proof if any legal move
    // escapes mate within the current horizon.
    result = 0;
    for (const move of orderedMoves(board, legal, attacker)) {
      const childDistance = mateDistance(makeMove(board, move), attacker, depth - 1, memo, proven);
      if (childDistance === null) { result = null; break; }
      result = Math.max(result, childDistance + 1);
    }
  }

  memo.set(memoKey, result);
  if (result !== null) proven.set(stateKey, result);
  return result;
}

const fileOf = (sq: number) => sq % 8;
const rankOf = (sq: number) => Math.floor(sq / 8);
const chebyshev = (a: number, b: number) => Math.max(Math.abs(fileOf(a) - fileOf(b)), Math.abs(rankOf(a) - rankOf(b)));

function defenderCanImmediatelyCaptureQueen(board: BoardState, attacker: Color): boolean {
  const info = detectKqk(board);
  if (!info || info.attacker !== attacker || board.turn !== info.defender) return false;
  return legalMoves(board).some((move) => move.to === info.queen && board.squares[move.to] === `${attacker}q`);
}

function fallbackKqkScore(board: BoardState, attacker: Color): number {
  const info = detectKqk(board);
  if (!info) return Number.POSITIVE_INFINITY;
  const defenderEdgeDistance = Math.min(fileOf(info.defenderKing), 7 - fileOf(info.defenderKing), rankOf(info.defenderKing), 7 - rankOf(info.defenderKing));
  const attackerKingDistance = chebyshev(info.attackerKing, info.defenderKing);
  const defenderMobility = board.turn === info.defender ? legalMoves(board).length : 0;
  const checkBonus = board.turn === info.defender && inCheck(board, info.defender) ? -2 : 0;
  // Lower is better for the mating side: drive the king to an edge/corner,
  // keep it boxed, and bring our king close enough to support mate.
  return defenderEdgeDistance * 100 + defenderMobility * 10 + attackerKingDistance + checkBonus + (attacker === info.attacker ? 0 : 10000);
}

export function chooseKingQueenVsKingMove(board: BoardState): EndgameOracleMove | null {
  const info = detectKqk(board);
  if (!info || board.turn !== info.attacker) return null;
  const legal = legalMoves(board);
  if (!legal.length) return null;

  for (const move of legal) {
    if (isDefenderCheckmated(makeMove(board, move), info.attacker)) {
      return { move, source: 'KQK endgame oracle', mateInPlies: 1 };
    }
  }

  let best: { move: Move; mateInPlies: number | null; score: number } | null = null;
  const memo = new Map<string, MemoValue>();
  const proven = new Map<string, number>();
  const orderedRootMoves = orderedMoves(board, legal, info.attacker);
  for (let depth = 2; depth <= KQK_MAX_MATE_PLIES; depth++) {
    for (const move of orderedRootMoves) {
      const child = makeMove(board, move);
      const childDistance = mateDistance(child, info.attacker, depth - 1, memo, proven);
      if (childDistance === null) continue;
      const mateInPlies = childDistance + 1;
      const score = mateInPlies * 100000 + fallbackKqkScore(child, info.attacker);
      if (!best || mateInPlies < (best.mateInPlies ?? Number.POSITIVE_INFINITY) || (mateInPlies === best.mateInPlies && score < best.score)) {
        best = { move, mateInPlies, score };
      }
    }
    if (best && best.mateInPlies !== null) return { move: best.move, source: 'KQK endgame oracle', mateInPlies: best.mateInPlies };
  }

  // KQK is theoretically won, but if the 50-move counter leaves too little room
  // for a proof inside the horizon, still avoid neural dithering by making the
  // most mating-shaped legal move. Never select stalemate, and prefer moves that
  // do not let the defender immediately capture the queen (turning the win into
  // bare kings). This matters for positions where the short mate proof horizon
  // falls back to geometry, e.g. a tempting checking move that simply hangs Q.
  let riskyBest: { move: Move; mateInPlies: number | null; score: number } | null = null;
  for (const move of legal) {
    const child = makeMove(board, move);
    const childLegal = legalMoves(child);
    if (child.turn === info.defender && childLegal.length === 0 && !inCheck(child, info.defender)) continue;
    const hangsQueen = defenderCanImmediatelyCaptureQueen(child, info.attacker);
    const score = fallbackKqkScore(child, info.attacker) + (hangsQueen ? 1_000_000 : 0);
    const candidate = { move, mateInPlies: null, score };
    if (hangsQueen) {
      if (!riskyBest || score < riskyBest.score) riskyBest = candidate;
      continue;
    }
    if (!best || score < best.score) best = candidate;
  }
  const selected = best ?? riskyBest;
  return selected ? { move: selected.move, source: 'KQK endgame oracle', mateInPlies: selected.mateInPlies } : null;
}
