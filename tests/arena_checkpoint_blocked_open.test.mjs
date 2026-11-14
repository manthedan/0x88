import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadArenaCheckpoint } from '../src/lc0/arenaCheckpoint.ts';

// An IDB open request cannot be cancelled. After onblocked rejects the app's
// promise, closing the other tab can still deliver onupgradeneeded/onsuccess.
test('a rejected blocked checkpoint open closes its eventual connection', async (t) => {
  let request;
  let closes = 0;
  let transactions = 0;
  replaceGlobal(t, 'indexedDB', {
    open() {
      request = {
        result: {
          createObjectStore() {},
          close() {
            closes++;
          },
          transaction() {
            transactions++;
            throw new Error('rejected open has no reader');
          },
        },
      };
      return request;
    },
  });
  const pending = loadArenaCheckpoint();
  request.onblocked();
  await assert.rejects(pending, /blocked/);
  request.onupgradeneeded();
  request.onsuccess();
  await new Promise(setImmediate);
  assert.equal(closes, 1, 'the abandoned connection would block later database upgrades/deletion');
  assert.equal(transactions, 0);
});

test('a successful checkpoint open retains its connection until the read finishes', async (t) => {
  let closes = 0;
  replaceGlobal(t, 'indexedDB', {
    open() {
      const request = {
        result: {
          close() {
            closes++;
          },
          transaction() {
            assert.equal(closes, 0);
            return {
              objectStore: () => ({
                get() {
                  const read = { result: undefined };
                  queueMicrotask(() => read.onsuccess());
                  return read;
                },
              }),
            };
          },
        },
      };
      queueMicrotask(() => request.onsuccess());
      return request;
    },
  });
  assert.equal(await loadArenaCheckpoint(), null);
  assert.equal(closes, 1);
});

function replaceGlobal(t, key, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  });
}
