import assert from 'node:assert/strict';
import { test } from 'node:test';
import { BUILTIN_ARENA_OPENINGS, parseArenaOpenings, scheduleOpenings } from '../src/lc0/arenaOpenings.ts';
import { START_FEN } from '../src/chess/board.ts';

test('parseArenaOpenings accepts raw FEN and named FEN rows', () => {
  const rows = parseArenaOpenings(`
# comments and blanks are ignored
${START_FEN}
Sicilian | rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3 0 2
French; rnbqkbnr/ppp2ppp/4p3/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3
`);
  assert.equal(rows.length, 3);
  assert.equal(rows[0].name, 'Position 1');
  assert.equal(rows[0].fen, START_FEN);
  assert.equal(rows[1].name, 'Sicilian');
  assert.match(rows[1].fen, /^rnbqkbnr\/pp1ppppp/);
  assert.equal(rows[2].name, 'French');
});

test('built-in arena opening suite has valid normalized FENs', () => {
  assert.ok(BUILTIN_ARENA_OPENINGS.length >= 8);
  assert.ok(BUILTIN_ARENA_OPENINGS.every((opening) => opening.name && opening.fen.split(/\s+/).length === 6));
});

test('built-in opening suite uses canonical sides to move for recognizable lines', () => {
  const byName = new Map(BUILTIN_ARENA_OPENINGS.map((opening) => [opening.name, opening.fen]));
  assert.equal(byName.get('French Advance'), 'rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3');
  assert.equal(byName.get('Caro-Kann Advance'), 'rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3');
  assert.equal(byName.get('Slav Defense'), 'rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3');
  assert.equal(byName.get('Benoni Defense'), 'rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2');
  assert.equal(byName.get('Four Knights'), 'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4');
});

test('scheduleOpenings plays every pairing on every configured position', () => {
  const pairings = [{ white: 'a', black: 'b' }, { white: 'b', black: 'a' }];
  const openings = [{ name: 'Start', fen: START_FEN }, { name: 'French', fen: 'fen2' }];
  const scheduled = scheduleOpenings(pairings, openings);
  assert.deepEqual(scheduled.map((g) => [g.white, g.black, g.opening.name]), [
    ['a', 'b', 'Start'],
    ['b', 'a', 'Start'],
    ['a', 'b', 'French'],
    ['b', 'a', 'French'],
  ]);
});
