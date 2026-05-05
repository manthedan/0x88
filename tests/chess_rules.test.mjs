import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

function runSnippet(source) {
  return JSON.parse(execFileSync('node', ['--experimental-strip-types', '--input-type=module', '-e', source], { encoding: 'utf8' }));
}

test('legal moves reject king moves onto attacked squares', () => {
  const result = runSnippet(`
    import { parseFen } from './src/chess/board.ts';
    import { inCheck, legalMoves } from './src/chess/movegen.ts';
    import { moveToUci } from './src/chess/moveCodec.ts';
    const fen = 'k3r3/8/8/8/8/8/8/4K3 w - - 0 1';
    console.log(JSON.stringify({ check: inCheck(parseFen(fen)), moves: legalMoves(parseFen(fen)).map(moveToUci).sort() }));
  `);
  assert.equal(result.check, true);
  assert.equal(result.moves.includes('e1e2'), false);
  assert.deepEqual(result.moves, ['e1d1', 'e1d2', 'e1f1', 'e1f2']);
});

test('castling is generated only through unattacked empty transit squares', () => {
  const result = runSnippet(`
    import { parseFen } from './src/chess/board.ts';
    import { legalMoves } from './src/chess/movegen.ts';
    import { moveToUci } from './src/chess/moveCodec.ts';
    const ucis = (fen) => legalMoves(parseFen(fen)).map(moveToUci).sort();
    console.log(JSON.stringify({ clear: ucis('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'), attacked: ucis('r3k2r/8/8/8/8/5r2/8/R3K2R w KQkq - 0 1') }));
  `);
  assert.ok(result.clear.includes('e1g1'));
  assert.ok(result.clear.includes('e1c1'));
  assert.equal(result.attacked.includes('e1g1'), false);
});

test('makeMove updates en passant, en passant capture, and castling rook movement', () => {
  const result = runSnippet(`
    import { parseFen, boardToFen } from './src/chess/board.ts';
    import { makeMove } from './src/chess/movegen.ts';
    const afterDouble = makeMove(parseFen('8/8/8/8/8/8/4P3/4K2k w - - 0 1'), { from: 12, to: 28 });
    const ep = makeMove(parseFen('8/8/8/3pP3/8/8/8/4K2k w - d6 0 1'), { from: 36, to: 43 });
    const castle = makeMove(parseFen('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'), { from: 4, to: 6 });
    console.log(JSON.stringify({ fen: boardToFen(afterDouble), epFrom: ep.squares[35], epTo: ep.squares[43], king: castle.squares[6], rook: castle.squares[5], oldRook: castle.squares[7], castling: castle.castling }));
  `);
  assert.match(result.fen, / e3 /);
  assert.equal(result.epFrom, null);
  assert.equal(result.epTo, 'wp');
  assert.equal(result.king, 'wk');
  assert.equal(result.rook, 'wr');
  assert.equal(result.oldRook, null);
  assert.equal(result.castling.includes('K'), false);
});
