// Deterministic host-side legal user moves, not another in-browser engine/worker.
// A shallow material/development search avoids a random pawn pusher getting mated
// before the required ten full moves. This work runs only while it is the user's turn.
import { boardToFen, parseFen } from '../../src/chess/board.ts';
import { moveToUci } from '../../src/chess/moveCodec.ts';
import { inCheck, legalMoves, makeMove } from '../../src/chess/movegen.ts';
import { moveToSan } from '../../src/chess/san.ts';

export function playHumanTurn(status) {
  return /Your move|You are in check/.test(status);
}

const value = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
function score(board) {
  let total = 0;
  for (let i = 0; i < 64; i++) {
    const p = board.squares[i];
    if (!p) continue;
    const file = i % 8,
      rank = p[0] === 'w' ? Math.floor(i / 8) : 7 - Math.floor(i / 8);
    const center = 3.5 - Math.abs(file - 3.5);
    const development =
      p[1] === 'p'
        ? rank * 4 + center * 2
        : ['n', 'b'].includes(p[1])
          ? (rank > 0 ? 12 : 0) + center * 3
          : p[1] === 'k'
            ? rank === 0 && [6, 2].includes(file)
              ? 35
              : -rank * 10
            : 0;
    total += (p[0] === 'w' ? 1 : -1) * (value[p[1]] + development);
  }
  return total * (board.turn === 'w' ? 1 : -1);
}
function search(board, depth, alpha, beta) {
  const moves = legalMoves(board);
  if (!moves.length) return inCheck(board) ? -100000 - depth : 0;
  if (!depth) return score(board);
  moves.sort((a, b) => (value[board.squares[b.to]?.[1]] ?? 0) - (value[board.squares[a.to]?.[1]] ?? 0));
  for (const move of moves) {
    alpha = Math.max(alpha, -search(makeMove(board, move), depth - 1, -beta, -alpha));
    if (alpha >= beta) break;
  }
  return alpha;
}
export function chooseHumanMove(sans) {
  let board = parseFen();
  const seen = new Set([boardToFen(board).split(' ').slice(0, 2).join(' ')]);
  for (const san of sans) {
    const move = legalMoves(board).find((m) => moveToSan(board, m) === san.trim());
    if (!move) throw Error(`Cannot replay committed SAN ${san}`);
    board = makeMove(board, move);
    seen.add(boardToFen(board).split(' ').slice(0, 2).join(' '));
  }
  if (board.turn !== 'w') throw Error('Human chooser requires the committed White turn');
  const moves = legalMoves(board)
    .filter((m) => !m.promotion)
    .sort((a, b) => moveToUci(a).localeCompare(moveToUci(b)));
  let best,
    bestScore = -Infinity;
  for (const move of moves) {
    const next = makeMove(board, move);
    if (!legalMoves(next).length) continue; // Do not deliberately end the measured game before its tenth reply.
    const repeats = seen.has(boardToFen(next).split(' ').slice(0, 2).join(' '));
    const candidate = -search(next, 2, -Infinity, Infinity) - (repeats ? 20 : 0);
    if (candidate > bestScore) {
      best = move;
      bestScore = candidate;
    }
  }
  if (!best) throw Error('Game ended before ten full moves');
  return moveToUci(best);
}
