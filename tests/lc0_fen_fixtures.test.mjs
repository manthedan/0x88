import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { parseFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { uciToLc0PolicyIndex } from '../src/lc0/policyMap.ts';

const fixtures = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8'));

test('LC0 FEN-only fixture suite covers agreed edge cases', () => {
  assert.deepEqual(fixtures.map((fixture) => fixture.id), [
    'startpos',
    'black-to-move-after-e4',
    'castling-rights',
    'en-passant-available',
    'promotion-near',
    'rule50-nonzero',
  ]);
});

test('LC0 FEN-only fixtures parse, expose expected legal moves, and map to policy indices', () => {
  for (const fixture of fixtures) {
    const board = parseFen(fixture.fen);
    const legal = new Set(legalMoves(board).map(moveToUci));
    for (const uci of fixture.expectedLegalMoves) {
      assert.equal(legal.has(uci), true, `${fixture.id} expected legal move ${uci}`);
      const isStandardCastle = ['e1g1', 'e1c1', 'e8g8', 'e8c8'].includes(uci);
      assert.equal(typeof uciToLc0PolicyIndex(uci, 0, { standardCastling: isStandardCastle }), 'number', `${fixture.id} LC0 policy index for ${uci}`);
    }
  }
});

test('LC0 FEN-only fixtures include rule50 and en-passant state explicitly', () => {
  const byId = new Map(fixtures.map((fixture) => [fixture.id, parseFen(fixture.fen)]));
  assert.equal(byId.get('rule50-nonzero').halfmove, 37);
  assert.equal(byId.get('en-passant-available').epSquare, 43); // d6, with a1=0.
});
