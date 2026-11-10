import assert from 'node:assert/strict';
import { test } from 'node:test';
import { loadLc0ModelForOrt } from '../src/lc0/modelCache.ts';

for (const { name, activation, persisted = false, enabled = true, expectedCalls = 0, expectedResult } of [
  { name: 'no transient activation', activation: false },
  { name: 'no userActivation API (including workers)' },
  { name: 'active gesture', activation: true, expectedCalls: 1, expectedResult: true },
  { name: 'already persisted without a gesture', persisted: true, expectedResult: true },
  { name: 'already persisted during a gesture', persisted: true, activation: true, expectedResult: true },
  { name: 'request disabled during a gesture', enabled: false, activation: true },
]) {
  test(`model storage persistence: ${name}`, async () => {
    const previous = { caches: globalThis.caches, fetch: globalThis.fetch, navigator: Object.getOwnPropertyDescriptor(globalThis, 'navigator') };
    let persistedCalls = 0;
    let persistCalls = 0;
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: {
        ...(activation === undefined ? {} : { userActivation: { isActive: activation } }),
        storage: {
          persisted: async () => {
            persistedCalls += 1;
            return persisted;
          },
          persist: async () => {
            persistCalls += 1;
            return true;
          },
        },
      },
    });
    globalThis.caches = { open: async () => ({ match: async () => undefined, put: async () => {}, keys: async () => [] }) };
    globalThis.fetch = async () => new Response('abc');
    try {
      const result = await loadLc0ModelForOrt('https://example.test/model.onnx', {
        cache: true,
        requestPersistentStorage: enabled,
        manifestEntry: {
          file: 'model.onnx',
          url: 'https://example.test/model.onnx',
          sha256: 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',
        },
      });
      assert.equal(persistedCalls, enabled ? 1 : 0);
      assert.equal(persistCalls, expectedCalls);
      assert.equal(result.storagePersistent, expectedResult);
    } finally {
      globalThis.caches = previous.caches;
      globalThis.fetch = previous.fetch;
      if (previous.navigator) Object.defineProperty(globalThis, 'navigator', previous.navigator);
      else delete globalThis.navigator;
    }
  });
}
