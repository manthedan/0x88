import test from 'node:test';
import assert from 'node:assert/strict';

import { defaultPgnCollectionName, formatPgnCollectionSummary, normalizePgnDatabaseBackup, pgnDatabaseBackupFilename, rebuildPgnCollectionIndex, sanitizePgnCollectionName } from '../src/lc0/pgnDatabase.ts';

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

test('PGN database backup filenames are dated', () => {
  assert.equal(pgnDatabaseBackupFilename(new Date('2026-06-05T12:00:00Z')), 'lc0-analysis-pgn-db-2026-06-05.json');
});

test('PGN database backup import normalizes collection records', () => {
  const collections = normalizePgnDatabaseBackup({
    kind: 'lc0-analysis-pgn-database-backup',
    version: 1,
    collections: [{
      id: 'ignored',
      name: '  Backup   Games  ',
      pgn: '[Result "*"]\n\n1. e4 *',
      gameCount: '3',
      source: 'lichess',
      username: ' dan ',
      color: 'white',
      positionIndex: {
        'start w KQkq -': [{ uci: 'e2e4', san: 'e4', count: 3, whiteWins: 1, blackWins: 1, draws: 1 }],
      },
    }],
  });
  assert.equal(collections.length, 1);
  assert.equal(collections[0].name, 'Backup Games');
  assert.equal(collections[0].source, 'lichess');
  assert.equal(collections[0].username, 'dan');
  assert.equal(collections[0].gameCount, 3);
  assert.equal(collections[0].indexedPositionCount, 1);
});

test('PGN database backup import rejects invalid backups', () => {
  assert.throws(() => normalizePgnDatabaseBackup({}), /collections array/);
  assert.throws(() => normalizePgnDatabaseBackup({ collections: [{ name: 'empty', pgn: '' }] }), /has no PGN/);
});

test('PGN database import rebuilds position indexes from raw PGN', () => {
  const rebuilt = rebuildPgnCollectionIndex({
    name: 'Unindexed backup',
    pgn: '[Result "1-0"]\n\n1. e4 e5 1-0',
    gameCount: 0,
    source: 'manual',
    positionIndex: { stale: [{ uci: 'd2d4', san: 'd4', count: 99, whiteWins: 0, blackWins: 0, draws: 0 }] },
  });
  assert.equal(rebuilt.gameCount, 1);
  assert.ok(rebuilt.indexedPositionCount > 0);
  assert.equal(rebuilt.positionIndex.stale, undefined);
  assert.equal(rebuilt.positionIndex['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'][0].san, 'e4');
});
