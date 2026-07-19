import 'fake-indexeddb/auto';
import assert from 'node:assert/strict';
import test from 'node:test';

import { reviewGame } from '../src/lc0/gameReview.ts';
import {
  deletePgnCollection,
  exportPgnDatabaseBackup,
  importPgnDatabaseBackup,
  listPgnCollectionGames,
  loadPgnGame,
  loadPgnGameReview,
  pgnGameReviewKey,
  savePgnCollection,
  savePgnGameReview,
} from '../src/lc0/pgnDatabase.ts';

const DB_NAME = 'lc0-analysis-pgn-database';
const PGN = '[White "Alice"]\n[Black "Bob"]\n[Date "2026.07.19"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 1-0';
const VARIANT_PGN = '[Event "Annotated copy"]\n[White "Alice"]\n[Black "Bob"]\n[Date "2026.07.19"]\n[Result "1-0"]\n\n1. e4 { collection-local note } e5 2. Nf3 1-0';

function deleteDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.deleteDatabase(DB_NAME);
    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
    request.onblocked = () => reject(new Error('Database deletion blocked'));
  });
}

function createLegacyV2Collection() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 2);
    request.onupgradeneeded = () => {
      const store = request.result.createObjectStore('collections', { keyPath: 'id' });
      store.createIndex('updatedAt', 'updatedAt');
      store.createIndex('name', 'name');
      store.put({ id: 'legacy', name: 'Legacy', pgn: PGN, gameCount: 1, source: 'manual', createdAt: 1, updatedAt: 1 });
    };
    request.onerror = () => reject(request.error);
    request.onsuccess = () => { request.result.close(); resolve(); };
  });
}

test('v3 lazily migrates games, deduplicates storage, and persists reviews', async () => {
  await deleteDatabase();
  await createLegacyV2Collection();

  const legacyGames = await listPgnCollectionGames('legacy');
  assert.equal(legacyGames.length, 1);
  assert.equal(legacyGames[0].white, 'Alice');
  assert.match(legacyGames[0].id, /^game-[a-f0-9]{64}$/);

  const duplicateCollection = await savePgnCollection({ name: 'Duplicate source', pgn: `${VARIANT_PGN}\n\n${PGN}`, gameCount: 99, source: 'manual' });
  const duplicateGames = await listPgnCollectionGames(duplicateCollection.id);
  assert.equal(duplicateGames[0].id, legacyGames[0].id);
  assert.equal(duplicateGames[1].id, legacyGames[0].id, 'semantic duplicates retain distinct ordered occurrences');
  assert.match(duplicateGames[0].pgn, /collection-local note/);
  assert.doesNotMatch(duplicateGames[1].pgn, /collection-local note/);
  assert.doesNotMatch((await listPgnCollectionGames('legacy'))[0].pgn, /collection-local note/);
  assert.equal(duplicateCollection.gameCount, 2, 'game count is derived from the structured parse');

  const gameReview = reviewGame(
    [{ winWhite: 0.5, bestUci: 'e2e4' }, { winWhite: 0.55, bestUci: 'e7e5' }, { winWhite: 0.5, bestUci: 'g1f3' }, { winWhite: 0.52, bestUci: null }],
    [{ san: 'e4', uci: 'e2e4' }, { san: 'e5', uci: 'e7e5' }, { san: 'Nf3', uci: 'g1f3' }],
  );
  const reviewKey = pgnGameReviewKey('stockfish-lite', 12);
  await savePgnGameReview({ gameId: legacyGames[0].id, reviewKey, engine: 'stockfish-lite', depth: 12, algorithmVersion: 1, review: gameReview, annotatedPgn: 'annotated' });
  const restored = await loadPgnGameReview(legacyGames[0].id, reviewKey);
  assert.equal(restored.review.accuracy.white, gameReview.accuracy.white);

  const deeperReviewKey = pgnGameReviewKey('stockfish-lite', 16);
  await savePgnGameReview({ gameId: legacyGames[0].id, reviewKey: deeperReviewKey, engine: 'stockfish-lite', depth: 16, algorithmVersion: 1, review: gameReview, annotatedPgn: 'deeper' });
  assert.equal((await loadPgnGameReview(legacyGames[0].id, reviewKey)).depth, 12);
  assert.equal((await loadPgnGameReview(legacyGames[0].id, deeperReviewKey)).depth, 16);

  const listedAgain = await listPgnCollectionGames('legacy');
  assert.equal(listedAgain[0].latestReview.depth, 16);
  const backup = await exportPgnDatabaseBackup(new Date('2026-07-19T12:00:00Z'));
  assert.equal(backup.version, 2);
  assert.equal(backup.reviews.length, 2);

  await deletePgnCollection('legacy');
  assert.ok(await loadPgnGame(legacyGames[0].id), 'deduplicated game remains while another collection references it');
  await deletePgnCollection(duplicateCollection.id);
  assert.equal(await loadPgnGame(legacyGames[0].id), null, 'final membership deletion removes the orphaned game');
  assert.equal(await loadPgnGameReview(legacyGames[0].id, reviewKey), null, 'orphaned reviews are removed with their game');

  await deleteDatabase();
  assert.equal(await importPgnDatabaseBackup(backup), 2);
  assert.equal((await loadPgnGameReview(legacyGames[0].id, reviewKey)).depth, 12, 'backup reviews round-trip through a fresh database');
  await deleteDatabase();
});
