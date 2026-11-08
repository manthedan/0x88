import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';

const originalIdb = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
afterEach(() => {
  if (originalIdb) Object.defineProperty(globalThis, 'indexedDB', originalIdb);
  else delete globalThis.indexedDB;
});
import { START_FEN } from '../src/chess/board.ts';
import { boundedPgnChunks, importLocalPgn, MAX_PGN_BYTES } from '../src/lc0/localPgnImport.ts';
import { openingStatsFromIndex, OPENING_INDEX_VERSION } from '../src/lc0/openingStats.ts';
import { deletePgnCollection, duplicatePgnCollection, exportPgnDatabaseBackup, importPgnDatabaseBackup, listPgnCollections, loadPgnCollection, renamePgnCollection, savePgnCollection, searchPgnCollectionsByPosition } from '../src/lc0/pgnDatabase.ts';

const pgn = '[Result "*"]\n\n1. Nf3 (1. d4 d5) Nf6 2. Ng1 Ng8 3. e4 *';

test('local import reports incremental progress and indexes mainlines only', async () => {
  const progress = [];
  const batch = await importLocalPgn(`${pgn}\n\n[Result "1-0"]\n\n1. e4 e5 1-0`, (done, total) => progress.push([done, total]));
  assert.deepEqual(progress, [[0, 2], [1, 2], [2, 2]]);
  assert.equal(batch.games.length, 2);
  assert.deepEqual(openingStatsFromIndex(batch.positionIndex, START_FEN).map((s) => [s.san, s.count]), [['e4', 1], ['Nf3', 1]]);
});

test('local PGN caps reject before parsing and invalid moves cannot silently become games', async () => {
  assert.throws(() => boundedPgnChunks('x'.repeat(MAX_PGN_BYTES + 1)), /1 MiB/);
  assert.throws(() => boundedPgnChunks(Array(201).fill('[Result "*"]\n1. e4 *').join('\n\n')), /200 games/);
  assert.throws(() => boundedPgnChunks(`1. e4 {${'x'.repeat(32768)}} *`), /32 KiB/);
  await assert.rejects(importLocalPgn('not a game'), /Game 1: Invalid PGN move/);
  await assert.rejects(importLocalPgn('[Event "No moves"]'), /No playable moves/);
  await assert.rejects(importLocalPgn('1. e4 e5 2. e4 *'), /Invalid PGN move/);
});

test('local import cancellation prevents publication', async () => {
  const controller = new AbortController();
  await assert.rejects(importLocalPgn(`${pgn}\n\n${pgn}`, (done) => { if (done === 1) controller.abort(); }, controller.signal), { name: 'AbortError' });
});

// Minimal asynchronous IDB contract for storage tests. Browser verification also exercises real IndexedDB.
function memoryDatabase() {
  const records = new Map();
  const db = {
    close() {},
    transaction() {
      const tx = {
        objectStore() { return {
          get(id) { return request(records.get(id)); },
          getAll() { return request([...records.values()]); },
          put(record) { records.set(record.id, structuredClone(record)); return request(record.id); },
          delete(id) { records.delete(id); return request(undefined); },
        }; },
      };
      setTimeout(() => tx.oncomplete?.(), 0);
      return tx;
    },
  };
  function request(value) {
    const req = {};
    queueMicrotask(() => { req.result = structuredClone(value); req.onsuccess?.(); });
    return req;
  }
  return { records, open() { const req = {}; queueMicrotask(() => { req.result = db; req.onsuccess?.(); }); return req; } };
}

// The database itself contains functions, so its open request must not structuredClone it.
function installMemoryIdb(t) {
  const database = memoryDatabase();
  Object.defineProperty(globalThis, 'indexedDB', { value: { open() {
    const req = {};
    queueMicrotask(() => {
      const connection = database.open();
      connection.onsuccess = () => { req.result = connection.result; req.onsuccess?.(); };
    });
    return req;
  } }, configurable: true });
  return database;
}

test('IndexedDB denied, missing, and blocked failures settle without hanging', async (t) => {
  Object.defineProperty(globalThis, 'indexedDB', { value: undefined, configurable: true });
  await assert.rejects(listPgnCollections(), /unavailable/);
  Object.defineProperty(globalThis, 'indexedDB', { value: { open() { throw new DOMException('Denied', 'SecurityError'); } }, configurable: true });
  await assert.rejects(listPgnCollections(), /Denied/);
  let closed = false;
  Object.defineProperty(globalThis, 'indexedDB', { value: { open() {
    const req = {};
    queueMicrotask(() => req.onblocked());
    setTimeout(() => { req.result = { close() { closed = true; } }; req.onsuccess(); }, 0);
    return req;
  } }, configurable: true });
  await assert.rejects(listPgnCollections(), /blocked by another tab/);
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(closed, true, 'late connection is closed after rejection');
});

test('backup round trip rebuilds poisoned indexes and collection CRUD stays local', async (t) => {
  installMemoryIdb(t);
  const batch = await importLocalPgn(pgn);
  const record = await savePgnCollection({ name: 'Original', pgn, gameCount: 1, positionIndex: batch.positionIndex });
  assert.equal(record.indexVersion, OPENING_INDEX_VERSION);
  const copy = await duplicatePgnCollection(record.id);
  await renamePgnCollection(copy.id, 'Renamed');
  assert.equal((await loadPgnCollection(copy.id)).name, 'Renamed');
  await deletePgnCollection(copy.id);
  const backup = JSON.parse(JSON.stringify(await exportPgnDatabaseBackup()));
  assert.equal(backup.collections.length, 1);
  assert.equal(backup.collections[0].positionIndex, undefined, 'portable backup omits derived indexes');
  backup.collections[0].positionIndex = { bogus: [{ count: 999 }] };
  backup.collections[0].gameCount = 999;
  await deletePgnCollection(record.id);
  assert.equal(await importPgnDatabaseBackup(backup), 1);
  const restored = await listPgnCollections();
  assert.equal(restored.length, 1);
  const loaded = await loadPgnCollection(restored[0].id);
  assert.equal(loaded.gameCount, 1);
  assert.deepEqual(loaded.positionIndex, batch.positionIndex);
  const hits = await searchPgnCollectionsByPosition(START_FEN);
  assert.equal(hits[0].total, 1);
  assert.equal(hits[0].stats[0].unknown, 1);
  const invalid = structuredClone(backup);
  invalid.collections.push({ name: 'Bad', pgn: '1. notamove *' });
  await assert.rejects(importPgnDatabaseBackup(invalid), /Invalid PGN move/);
  assert.equal((await listPgnCollections()).length, 1, 'a failed restore leaves no partial additions');
});
