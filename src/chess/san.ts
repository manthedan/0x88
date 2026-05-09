import { cloneBoard, squareName, type BoardState, type PieceRole } from './board.ts';
import { inCheck, legalMoves, makeMove } from './movegen.ts';
import { moveFromUci, moveToUci, type Move } from './moveCodec.ts';

const PIECE_SAN: Record<Exclude<PieceRole, 'p'>, string> = { n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' };
const PROMO_SAN: Record<Exclude<PieceRole, 'p' | 'k'>, string> = { n: 'N', b: 'B', r: 'R', q: 'Q' };

function fileName(square: number) { return squareName(square)[0]; }
function rankName(square: number) { return squareName(square)[1]; }
function sameMove(a: Move, b: Move) { return a.from === b.from && a.to === b.to && a.promotion === b.promotion; }
function legalEquivalent(board: BoardState, move: Move) {
  return legalMoves(board).find((candidate) => sameMove(candidate, move))
    ?? legalMoves(board).find((candidate) => candidate.from === move.from && candidate.to === move.to && !candidate.promotion && !move.promotion)
    ?? null;
}

function checkSuffix(board: BoardState, move: Move) {
  const next = makeMove(board, move);
  if (!inCheck(next, next.turn)) return '';
  return legalMoves(next).length === 0 ? '#' : '+';
}

export function moveToSan(board: BoardState, inputMove: Move): string {
  try {
    const move = legalEquivalent(board, inputMove) ?? inputMove;
    const piece = board.squares[move.from];
    if (!piece) return moveToUci(inputMove);
    const role = piece[1] as PieceRole;
    const from = squareName(move.from);
    const to = squareName(move.to);
    const isCastle = role === 'k' && Math.abs((move.to % 8) - (move.from % 8)) === 2;
    if (isCastle) return `${move.to % 8 === 6 ? 'O-O' : 'O-O-O'}${checkSuffix(board, move)}`;

    const isEp = role === 'p' && move.to === board.epSquare && !board.squares[move.to] && from[0] !== to[0];
    const isCapture = !!board.squares[move.to] || isEp;
    const promotion = move.promotion ? `=${PROMO_SAN[move.promotion]}` : '';
    if (role === 'p') {
      const pawnPrefix = isCapture ? from[0] : '';
      return `${pawnPrefix}${isCapture ? 'x' : ''}${to}${promotion}${checkSuffix(board, move)}`;
    }

    const samePieceAmbiguities = legalMoves(board).filter((candidate) => {
      if (sameMove(candidate, move) || candidate.to !== move.to) return false;
      return board.squares[candidate.from] === piece;
    });
    let disambiguation = '';
    if (samePieceAmbiguities.length) {
      const sameFile = samePieceAmbiguities.some((candidate) => fileName(candidate.from) === fileName(move.from));
      const sameRank = samePieceAmbiguities.some((candidate) => rankName(candidate.from) === rankName(move.from));
      disambiguation = !sameFile ? fileName(move.from) : !sameRank ? rankName(move.from) : from;
    }
    return `${PIECE_SAN[role as Exclude<PieceRole, 'p'>]}${disambiguation}${isCapture ? 'x' : ''}${to}${checkSuffix(board, move)}`;
  } catch {
    return moveToUci(inputMove);
  }
}

export function uciToSan(board: BoardState, uci: string): string {
  try {
    if (!/^[a-h][1-8][a-h][1-8][nbrq]?$/.test(uci)) return uci;
    const parsed = moveFromUci(uci);
    const move = legalEquivalent(board, parsed) ?? parsed;
    return moveToSan(board, move);
  } catch {
    return uci;
  }
}

export function uciLineToSan(board: BoardState, ucis: string[], maxMoves = ucis.length): string {
  let cursor = cloneBoard(board);
  const sans: string[] = [];
  for (const uci of ucis.slice(0, maxMoves)) {
    try {
      const parsed = moveFromUci(uci);
      const move = legalEquivalent(cursor, parsed);
      if (!move) { sans.push(uci); break; }
      sans.push(moveToSan(cursor, move));
      cursor = makeMove(cursor, move);
    } catch {
      sans.push(uci);
      break;
    }
  }
  return sans.join(' ');
}
