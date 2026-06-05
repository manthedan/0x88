import test from 'node:test';
import assert from 'node:assert/strict';

import { defaultPgnCollectionName, formatPgnCollectionSummary, sanitizePgnCollectionName } from '../src/lc0/pgnDatabase.ts';

test('PGN collection names are trimmed and bounded', () => {
  assert.equal(sanitizePgnCollectionName('  My   Games  '), 'My Games');
  assert.equal(sanitizePgnCollectionName('   ', 'Fallback'), 'Fallback');
  assert.equal(sanitizePgnCollectionName('x'.repeat(120)).length, 80);
});

test('default PGN collection names include source, user, and day', () => {
  const day = new Date('2026-06-05T12:00:00Z');
  assert.equal(defaultPgnCollectionName('lichess', 'dan', day), 'Lichess dan 2026-06-05');
  assert.equal(defaultPgnCollectionName('chesscom', 'dan', day), 'Chess.com dan 2026-06-05');
  assert.equal(defaultPgnCollectionName('manual', '', day), 'Imported PGN 2026-06-05');
});

test('PGN collection summary is compact and source-aware', () => {
  const label = formatPgnCollectionSummary({
    id: 'a',
    name: 'Blitz sample',
    gameCount: 12,
    source: 'lichess',
    username: 'dan',
    indexedPositionCount: 42,
    createdAt: Date.UTC(2026, 5, 5),
    updatedAt: Date.UTC(2026, 5, 5),
  });
  assert.match(label, /^Blitz sample · 12 games · 42 positions · lichess:dan · /);
});
