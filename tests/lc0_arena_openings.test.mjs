import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boardToFen, START_FEN } from '../src/chess/board.ts';
import { BUILTIN_ARENA_OPENINGS, openingFromUciMoves, parseArenaOpenings, scheduleOpenings } from '../src/lc0/arenaOpenings.ts';

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

test('built-in arena opening suite has valid normalized FENs and true histories', () => {
  assert.ok(BUILTIN_ARENA_OPENINGS.length >= 8);
  assert.ok(BUILTIN_ARENA_OPENINGS.every((opening) => opening.name && opening.fen.split(/\s+/).length === 6));
  assert.ok(BUILTIN_ARENA_OPENINGS.every((opening) => opening.positions?.length === (opening.moves?.length ?? 0) + 1));
  assert.ok(BUILTIN_ARENA_OPENINGS.every((opening) => boardToFen(opening.positions.at(-1)) === opening.fen));
});

test('built-in opening suite uses canonical sides to move for recognizable lines', () => {
  const byName = new Map(BUILTIN_ARENA_OPENINGS.map((opening) => [opening.name, opening.fen]));
  assert.equal(byName.get('French Advance'), 'rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3');
  assert.equal(byName.get('Caro-Kann Advance'), 'rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3');
  assert.equal(byName.get('Slav Defense'), 'rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3');
  assert.equal(byName.get('Benoni Defense'), 'rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 0 2');
  assert.equal(byName.get('Four Knights'), 'r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4');
});

test('parseArenaOpenings accepts UCI and PGN/SAN replays with history boards', () => {
  const rows = parseArenaOpenings(`
Ruy UCI | e2e4 e7e5 g1f3 b8c6 f1b5
Italian PGN | 1. e4 e5 2. Nf3 Nc6 3. Bc4
`);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].moves, ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1b5']);
  assert.equal(rows[0].positions.length, 6);
  assert.equal(rows[0].fen, 'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3');
  assert.deepEqual(rows[1].moves, ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f1c4']);
  assert.equal(rows[1].positions.length, 6);
});

test('parseArenaOpenings rejects malformed PGN/SAN instead of silently skipping tokens', () => {
  assert.throws(() => parseArenaOpenings('Bad | 1. e4 bogus e5'), /illegal or unsupported SAN token bogus/);
  assert.throws(() => parseArenaOpenings('Bad NAG | 1. e4 $bogus e5'), /Malformed PGN NAG token \$bogus/);
});

test('parseArenaOpenings rejects illegal FENs with a specific position error and opening name', () => {
  for (const [fen, error] of [
    ['4k3/8/8/8/8/8/8/K3R3 w - - 0 1', /Opening Bad: Invalid FEN: black king is in check, but that side is not to move/],
    ['4k2r/8/8/8/8/8/8/7K b - - 0 1', /white king is in check, but that side is not to move/],
    ['8/8/8/8/8/8/8/4K3 w - - 0 1', /exactly one black king/],
    ['k3k3/8/8/8/8/8/8/4K3 w - - 0 1', /exactly one black king/],
    ['4k3/8/8/8/8/8/8/P3K3 w - - 0 1', /pawns cannot be on rank 1 or 8/],
  ]) {
    assert.throws(() => parseArenaOpenings(`Bad | ${fen}`), error);
  }
  assert.throws(() => parseArenaOpenings('4k3/8/8/8/8/8/8/K3R3 w - - 0 1'), /Opening Position 1:.*black king is in check/);
});

test('Arena validates a PGN setup FEN before replaying moves that could hide the illegal root', () => {
  const illegal = '4k3/8/8/8/8/8/8/K3R3 w - - 0 1';
  assert.throws(() => parseArenaOpenings(`Bad PGN | [FEN "${illegal}"] 1. Ra2`), /black king is in check/);
  assert.throws(() => openingFromUciMoves('Bad UCI root', ['e1a2'], illegal), /black king is in check/);
});

test('Arena accepts a legal check on the side to move and retains PGN setup history', () => {
  const legal = '4k3/8/8/8/8/8/8/K3R3 b - - 0 1';
  const [fen, pgn] = parseArenaOpenings(`Checked | ${legal}\nPGN | [FEN "${legal}"] 1... Kd7`);
  assert.equal(fen.fen, legal);
  assert.equal(pgn.startFen, legal);
  assert.deepEqual(pgn.moves, ['e8d7']);
  assert.equal(pgn.positions.length, 2);
});

test('scheduleOpenings plays every pairing on every configured position', () => {
  const pairings = [
    { white: 'a', black: 'b' },
    { white: 'b', black: 'a' },
  ];
  const openings = [
    { name: 'Start', fen: START_FEN },
    { name: 'French', fen: 'fen2' },
  ];
  const scheduled = scheduleOpenings(pairings, openings);
  assert.deepEqual(
    scheduled.map((g) => [g.white, g.black, g.opening.name]),
    [
      ['a', 'b', 'Start'],
      ['b', 'a', 'Start'],
      ['a', 'b', 'French'],
      ['b', 'a', 'French'],
    ],
  );
});
