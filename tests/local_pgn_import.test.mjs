import assert from 'node:assert/strict';
import test, { afterEach } from 'node:test';

const originalIdb = Object.getOwnPropertyDescriptor(globalThis, 'indexedDB');
afterEach(() => {
  if (originalIdb) Object.defineProperty(globalThis, 'indexedDB', originalIdb);
  else delete globalThis.indexedDB;
});

import { START_FEN } from '../src/chess/board.ts';
import { boundedPgnChunks, importLocalPgn, importStoredPgn, MAX_BACKUP_BYTES, MAX_PGN_BYTES } from '../src/lc0/localPgnImport.ts';
import { OPENING_INDEX_VERSION, openingStatsFromIndex } from '../src/lc0/openingStats.ts';
import {
  deletePgnCollection,
  duplicatePgnCollection,
  exportPgnDatabaseBackup,
  importPgnDatabaseBackup,
  listPgnCollections,
  loadPgnCollection,
  renamePgnCollection,
  savePgnCollection,
  searchPgnCollectionsByPosition,
  serializePgnDatabaseBackup,
} from '../src/lc0/pgnDatabase.ts';

const pgn = '[Result "*"]\n\n1. Nf3 (1. d4 d5) Nf6 2. Ng1 Ng8 3. e4 *';

test('local import reports incremental progress and indexes mainlines only', async () => {
  const progress = [];
  const batch = await importLocalPgn(`${pgn}\n\n[Result "1-0"]\n\n1. e4 e5 1-0`, (done, total) => progress.push([done, total]));
  assert.deepEqual(progress, [
    [0, 2],
    [1, 2],
    [2, 2],
  ]);
  assert.equal(batch.games.length, 2);
  assert.deepEqual(
    openingStatsFromIndex(batch.positionIndex, START_FEN).map((s) => [s.san, s.count]),
    [
      ['e4', 1],
      ['Nf3', 1],
    ],
  );
});

test('valid zero-move records are skipped without losing playable games or progress', async () => {
  const raw = `[Event "Aborted"]\n[Result "*"]\n\n*\n\n${pgn}\n\n[Event "Forfeit"]\n\n0-1 {no play}`;
  for (const importer of [importLocalPgn, importStoredPgn]) {
    const progress = [];
    const batch = await importer(raw, (done, total) => progress.push([done, total]));
    assert.equal(batch.games.length, 1);
    assert.equal(batch.skippedGames, 2);
    assert.deepEqual(progress, [
      [0, 3],
      [1, 3],
      [2, 3],
      [3, 3],
    ]);
    assert.deepEqual(batch.positionIndex, (await importLocalPgn(pgn)).positionIndex);
    await assert.rejects(importer('[Result "*"]\n*\n\n[Result "0-1"]\n0-1'), /No playable moves/);
    await assert.rejects(importer(`${raw}\n\n[Event "Broken"]\nnotamove *`), /Game 4: Invalid PGN move/);
    await assert.rejects(importer(`${pgn}\n\n[Event "Missing result"]`), /Game 2: No playable moves/);
  }
});

test('stored zero-move records survive index rebuild, export and restore with playable counts', async () => {
  installMemoryIdb();
  const raw = `[Result "0-1"]\n0-1\n\n${pgn}`;
  const record = await savePgnCollection({ name: 'With forfeit', pgn: raw, gameCount: 2 });
  assert.equal((await searchPgnCollectionsByPosition(START_FEN))[0].total, 1);
  const backup = await exportPgnDatabaseBackup();
  await deletePgnCollection(record.id);
  assert.equal(await importPgnDatabaseBackup(backup), 1);
  const [restored] = await listPgnCollections();
  assert.equal(restored.gameCount, 1);
  assert.deepEqual((await loadPgnCollection(restored.id)).positionIndex, (await importStoredPgn(pgn)).positionIndex);
});

test('local PGN caps reject before parsing and invalid moves cannot silently become games', async () => {
  assert.throws(() => boundedPgnChunks('x'.repeat(MAX_PGN_BYTES + 1)), /1 MiB/);
  assert.throws(() => boundedPgnChunks(Array(201).fill('[Result "*"]\n1. e4 *').join('\n\n')), /200 games/);
  assert.throws(() => boundedPgnChunks(`1. e4 {${'x'.repeat(32768)}} *`), /32 KiB/);
  await assert.rejects(importLocalPgn('not a game'), /Game 1: Invalid PGN move/);
  await assert.rejects(importLocalPgn('[Event "No moves"]'), /No playable moves/);
  await assert.rejects(importLocalPgn('1. e4 e5 2. e4 *'), /Invalid PGN move/);
});

test('malformed comment braces and variations reject instead of hanging or truncating games', async () => {
  await assert.rejects(importLocalPgn('1. e4 } *'), /Unmatched closing PGN comment brace/);
  await assert.rejects(importLocalPgn('1. e4 { unfinished'), /Unclosed PGN comment/);
  await assert.rejects(importLocalPgn('1. e4 ) *'), /Unmatched closing PGN variation/);
  await assert.rejects(importLocalPgn('1. e4 (1. d4 *'), /Unclosed PGN variation/);
});

test('repeated-move RAVs cannot replace or extend the literal mainline', async () => {
  const batch = await importLocalPgn('[Result "*"]\n1. e4 (1. e4 c5) e5 *\n\n[Result "*"]\n1. e4 (1. e4 c5) *');
  const afterE4 = batch.games[0].tree.root.children[0].fen;
  const stats = openingStatsFromIndex(batch.positionIndex, afterE4);
  assert.deepEqual(
    stats.map((s) => [s.san, s.count]),
    [['e5', 1]],
  );
  assert.equal(batch.games[1].tree.root.children[0].children.length, 0);
});

test('multiline brace comments containing PGN annotation directives stay in the same game', async () => {
  const batch = await importLocalPgn('[Result "*"]\n\n1. e4 {\n[%eval 0.2]\n[%clk 0:05:00]\n} e5 *\n\n[Result "1-0"]\n\n1. d4 d5 1-0');
  assert.equal(batch.games.length, 2);
  assert.deepEqual(
    batch.games[0].tree.mainlineFrom().map((node) => node.san),
    ['e4', 'e5'],
  );
});

test('oversized backups are rejected before download with a selected-collection fallback', () => {
  const backup = { kind: 'lc0-analysis-pgn-database-backup', version: 1, exportedAt: '', collections: [{ pgn: 'x'.repeat(MAX_BACKUP_BYTES) }] };
  assert.throws(() => serializePgnDatabaseBackup(backup), /Back up selected collection/);
  backup.collections = [{ pgn }];
  assert.equal(JSON.parse(serializePgnDatabaseBackup(backup)).collections[0].pgn, pgn);
});

test('legacy 201–300-game collections load, rebuild and round-trip through the backup contract', async () => {
  installMemoryIdb();
  for (const count of [201, 300]) {
    const raw = Array(count)
      .fill(`[Result "1-0"]\n\n1. e4 {${'x'.repeat(4096)}} e5 1-0`)
      .join('\n\n');
    assert.throws(() => boundedPgnChunks(raw), count === 300 ? /1 MiB/ : /200 games/, 'new-input bounds remain in force');
    const progress = [];
    const batch = await importStoredPgn(raw, (done, total) => progress.push([done, total]));
    assert.equal(batch.games.length, count);
    assert.deepEqual(progress.at(-1), [count, count]);
    const record = await savePgnCollection({ name: 'Legacy history', pgn: raw, gameCount: count });
    assert.equal((await loadPgnCollection(record.id)).pgn, raw);
    const hits = await searchPgnCollectionsByPosition(START_FEN);
    assert.equal(hits[0].total, count, 'unindexed legacy collections rebuild without the batch cap');
    let ticks = 0;
    const timer = setInterval(() => ticks++, 0);
    let backup;
    try {
      backup = JSON.parse(serializePgnDatabaseBackup(await exportPgnDatabaseBackup()));
    } finally {
      clearInterval(timer);
    }
    assert.ok(ticks >= count, 'export replay validation yields between games');
    await deletePgnCollection(record.id);
    assert.equal(await importPgnDatabaseBackup(backup), 1);
    const [restored] = await listPgnCollections();
    assert.equal(restored.gameCount, count);
    assert.deepEqual((await loadPgnCollection(restored.id)).positionIndex, batch.positionIndex);
    await deletePgnCollection(restored.id);
  }
});

test('stored collection and uploaded backup replay retain the per-game safety bound', async () => {
  installMemoryIdb();
  const oversizedGame = `1. e4 {${'x'.repeat(32768)}} e5 *`;
  const backup = {
    kind: 'lc0-analysis-pgn-database-backup',
    version: 1,
    exportedAt: '',
    collections: [{ name: 'Oversized game', pgn: oversizedGame, gameCount: 1 }],
  };
  await assert.rejects(importStoredPgn(oversizedGame), /32 KiB/);
  await assert.rejects(importPgnDatabaseBackup(backup), /32 KiB/);
  assert.throws(() => serializePgnDatabaseBackup(backup), /32 KiB/);
  assert.equal((await listPgnCollections()).length, 0);
});

test('backup export rejects records that restore cannot legally replay', async () => {
  installMemoryIdb();
  const record = await savePgnCollection({ name: 'Invalid', pgn: '1. e4 e5 2. e4 *', gameCount: 1 });
  assert.doesNotThrow(
    () => serializePgnDatabaseBackup({ kind: 'lc0-analysis-pgn-database-backup', version: 1, exportedAt: '', collections: [record] }),
    'serialization does not repeat legal replay synchronously',
  );
  await assert.rejects(exportPgnDatabaseBackup(), /Invalid PGN move/);
});

test('headerless games split on top-level results before the game-count cap', async () => {
  const batch = await importLocalPgn('1. e4 e5 1-0\n\n1. d4 d5 0-1');
  assert.equal(batch.games.length, 2);
  const stats = openingStatsFromIndex(batch.positionIndex, START_FEN);
  assert.equal(stats.find((s) => s.san === 'e4').whiteWins, 1);
  assert.equal(stats.find((s) => s.san === 'd4').blackWins, 1);
  assert.throws(() => boundedPgnChunks(Array(201).fill('1. e4 e5 *').join('\n')), /200 games/);
  const annotated = await importLocalPgn('1. e4 {1-0} (1. d4 d5 0-1) e5 * {trailing comment}\n1. d4 d5 *');
  assert.equal(annotated.games.length, 2, 'only top-level results split games');
});

test('adjacent numeric annotation glyphs do not change strict-import moves', async () => {
  const batch = await importLocalPgn('[Result "*"]\n\n1.e4$1 e5$2 2.Nf3$10 Nc6 *');
  assert.deepEqual(
    batch.games[0].tree.mainlineFrom().map((node) => node.san),
    ['e4', 'e5', 'Nf3', 'Nc6'],
  );
});

test('CR-only and CRLF PGN line endings preserve semicolon-comment boundaries and games', async () => {
  for (const newline of ['\r', '\r\n']) {
    const raw = ['[Result "1-0"]', '1. e4 ; comment', 'e5 1-0', '[Result "0-1"]', '1. d4 ; another', 'd5 0-1'].join(newline);
    const batch = await importLocalPgn(raw);
    assert.equal(batch.games.length, 2);
    assert.deepEqual(
      batch.games.map((game) => game.tree.mainlineFrom().map((node) => node.san)),
      [
        ['e4', 'e5'],
        ['d4', 'd5'],
      ],
    );
    assert.deepEqual(
      batch.games.map((game) => game.result),
      ['1-0', '0-1'],
    );
  }
});

test('local import cancellation prevents publication', async () => {
  const controller = new AbortController();
  await assert.rejects(
    importLocalPgn(
      `${pgn}\n\n${pgn}`,
      (done) => {
        if (done === 1) controller.abort();
      },
      controller.signal,
    ),
    { name: 'AbortError' },
  );
});

// Minimal asynchronous IDB contract for storage tests. Browser verification also exercises real IndexedDB.
function memoryDatabase() {
  const records = new Map();
  const db = {
    close() {},
    transaction() {
      const tx = {
        objectStore() {
          return {
            get(id) {
              return request(records.get(id));
            },
            getAll() {
              return request([...records.values()]);
            },
            put(record) {
              records.set(record.id, structuredClone(record));
              return request(record.id);
            },
            delete(id) {
              records.delete(id);
              return request(undefined);
            },
          };
        },
      };
      setTimeout(() => tx.oncomplete?.(), 0);
      return tx;
    },
  };
  function request(value) {
    const req = {};
    queueMicrotask(() => {
      req.result = structuredClone(value);
      req.onsuccess?.();
    });
    return req;
  }
  return {
    records,
    open() {
      const req = {};
      queueMicrotask(() => {
        req.result = db;
        req.onsuccess?.();
      });
      return req;
    },
  };
}

// The database itself contains functions, so its open request must not structuredClone it.
function installMemoryIdb() {
  const database = memoryDatabase();
  Object.defineProperty(globalThis, 'indexedDB', {
    value: {
      open() {
        const req = {};
        queueMicrotask(() => {
          const connection = database.open();
          connection.onsuccess = () => {
            req.result = connection.result;
            req.onsuccess?.();
          };
        });
        return req;
      },
    },
    configurable: true,
  });
  return database;
}

test('IndexedDB denied, missing, and blocked failures settle without hanging', async () => {
  Object.defineProperty(globalThis, 'indexedDB', { value: undefined, configurable: true });
  await assert.rejects(listPgnCollections(), /unavailable/);
  Object.defineProperty(globalThis, 'indexedDB', {
    value: {
      open() {
        throw new DOMException('Denied', 'SecurityError');
      },
    },
    configurable: true,
  });
  await assert.rejects(listPgnCollections(), /Denied/);
  let closed = false;
  Object.defineProperty(globalThis, 'indexedDB', {
    value: {
      open() {
        const req = {};
        queueMicrotask(() => req.onblocked());
        setTimeout(() => {
          req.result = {
            close() {
              closed = true;
            },
          };
          req.onsuccess();
        }, 0);
        return req;
      },
    },
    configurable: true,
  });
  await assert.rejects(listPgnCollections(), /blocked by another tab/);
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(closed, true, 'late connection is closed after rejection');
});

test('backup round trip rebuilds poisoned indexes and collection CRUD stays local', async () => {
  installMemoryIdb();
  const batch = await importLocalPgn(pgn);
  const record = await savePgnCollection({ name: 'Original', pgn, gameCount: 1, positionIndex: batch.positionIndex });
  assert.equal(record.indexVersion, OPENING_INDEX_VERSION);
  const copy = await duplicatePgnCollection(record.id);
  await renamePgnCollection(copy.id, 'Renamed');
  assert.equal((await loadPgnCollection(copy.id)).name, 'Renamed');
  await deletePgnCollection(copy.id);
  assert.equal((await exportPgnDatabaseBackup(new Date(), record.id)).collections.length, 1);
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
