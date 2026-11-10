import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, START_FEN } from '../src/chess/board.ts';
import { parsePgnGame } from '../src/chess/pgn.ts';
import { gameOutcome } from '../src/lc0/engineBattle.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import {
  playFromHereUnavailableReason, playGameToPgn, playMoveLabel, playMoveRows,
  playPositionFromTree, playTakebackPly, replayPlayPosition,
} from '../src/lc0/playPosition.ts';

function line(moves, startFen = START_FEN) {
  const tree = new GameTree(startFen);
  for (const move of moves) assert.ok(tree.addUci(move), move);
  return tree;
}

const customFen = '4k3/8/8/8/8/8/4P3/4K3 b - - 0 37';

test('position transfer copies only the selected branch through the chosen node without changing Analysis', () => {
  const tree = line(['e2e4', 'e7e5', 'g1f3']);
  const mainlineEnd = tree.current;
  tree.toStart();
  tree.addUci('d2d4');
  const selected = tree.current;
  tree.addUci('d7d5');
  tree.goTo(selected);
  const position = playPositionFromTree(tree);
  assert.deepEqual(position, { startFen: START_FEN, moves: ['d2d4'] });
  const replay = replayPlayPosition(position);
  assert.equal(boardToFen(replay.board), selected.fen);
  assert.equal(tree.current, selected);
  assert.equal(tree.mainlineFrom().at(-1), mainlineEnd);
  replay.moves.pop();
  assert.equal(tree.current, selected);
});

test('history preserves castling and en-passant rights and applies special moves legally', () => {
  const castle = line(['e1g1'], 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 20');
  const replay = replayPlayPosition(playPositionFromTree(castle));
  assert.equal(boardToFen(replay.board), castle.current.fen);
  assert.equal(replay.positions[0].castling, 'KQkq');
  const ep = line(['e5d6'], '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 23');
  assert.equal(boardToFen(replayPlayPosition(playPositionFromTree(ep)).board), ep.current.fen);
});

test('invalid roots, malformed UCI and illegal history are rejected rather than partially loaded', () => {
  for (const value of [null, {}, { startFen: 'nonsense', moves: [] }, { startFen: '8/8/8/8/8/8/8/8 w - - 0 1', moves: [] },
    { startFen: START_FEN, moves: ['e2e5'] }, { startFen: START_FEN, moves: ['e2e4', 'e2e3'] },
    { startFen: START_FEN, moves: ['E2E4'] }, { startFen: START_FEN, moves: Array(2001).fill('e2e4') }]) {
    assert.throws(() => replayPlayPosition(value));
  }
});

test('mate, stalemate and automatic draws have an explicit unavailable reason', () => {
  assert.equal(playFromHereUnavailableReason(line(['f2f3', 'e7e5', 'g2g4', 'd8h4'])), 'checkmate');
  assert.equal(playFromHereUnavailableReason(new GameTree('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1')), 'stalemate');
  assert.equal(playFromHereUnavailableReason(new GameTree('4k3/8/8/8/8/8/8/4K3 w - - 0 1')), 'insufficient material');
  assert.equal(playFromHereUnavailableReason(line(['e2e4'])), null);
});

test('supplied repetition history is retained for Play draw detection', () => {
  const tree = line(['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8']);
  const replay = replayPlayPosition(playPositionFromTree(tree));
  assert.equal(replay.positions.length, 9);
  assert.equal(gameOutcome(replay.board, replay.positions.slice(0, -1).map(boardToFen))?.reason, 'threefold');
});

test('PGN from a standard root includes the whole path and no FEN or SetUp header', () => {
  const replay = replayPlayPosition(playPositionFromTree(line(['e2e4', 'e7e5', 'g1f3'])));
  const pgn = playGameToPgn(replay.startFen, replay.moves, { White: 'You', Black: 'Maia3' });
  assert.doesNotMatch(pgn, /\[(FEN|SetUp) /);
  assert.match(pgn, /1\. e4 e5 2\. Nf3 \*/);
  const parsed = parsePgnGame(pgn);
  parsed.tree.toEnd();
  assert.equal(parsed.tree.current.fen, boardToFen(replay.board));
});

test('custom-root PGN has FEN and SetUp headers with Black-to-move numbering, including an empty record', () => {
  const replay = replayPlayPosition(playPositionFromTree(line(['e8d7', 'e2e4'], customFen)));
  const pgn = playGameToPgn(replay.startFen, replay.moves, {}, '1/2-1/2');
  assert.match(pgn, /\[SetUp "1"\]/);
  assert.ok(pgn.includes(`[FEN "${customFen}"]`));
  assert.match(pgn, /37\.\.\. Kd7 38\. e4 1\/2-1\/2$/);
  const parsed = parsePgnGame(pgn);
  parsed.tree.toEnd();
  assert.equal(parsed.tree.current.fen, boardToFen(replay.board));
  assert.ok(playGameToPgn(customFen, []).includes(`[FEN "${customFen}"]`));
});

test('move rows and browsing labels use custom-root fullmove and turn', () => {
  const replay = replayPlayPosition(playPositionFromTree(line(['e8d7', 'e2e4', 'd7c6'], customFen)));
  assert.deepEqual(playMoveRows(replay.positions, replay.sans), [
    { number: 37, white: null, black: 0 }, { number: 38, white: 1, black: 2 },
  ]);
  assert.equal(playMoveLabel(replay.positions[0], replay.sans[0]), '37… Kd7');
  assert.equal(playMoveLabel(replay.positions[1], replay.sans[1]), '38. e4');
});

test('takeback never crosses the practice boundary for either colour, even after only an engine move', () => {
  const replay = replayPlayPosition(playPositionFromTree(line(['e2e4', 'e7e5', 'g1f3', 'b8c6'])));
  assert.equal(playTakebackPly(replay.positions, 'w', 2), 2);
  assert.equal(playTakebackPly(replay.positions, 'b', 2), 3);
  assert.equal(playTakebackPly(replay.positions.slice(0, 4), 'b', 2), 2);
  assert.equal(playTakebackPly(replay.positions.slice(0, 3), 'b', 2), 2);
});
