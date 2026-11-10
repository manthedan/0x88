import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { enginePins, OUTPUT, serializePins } from '../scripts/write_engine_integrity.mjs';
import { createEngineArtifactLoader } from '../src/lc0/engineArtifacts.ts';

const good = new TextEncoder().encode('trusted engine');
const pin = { sha256: createHash('sha256').update(good).digest('hex'), bytes: good.length };
const loader = createEngineArtifactLoader({ '/engine.wasm': pin });
function at(t, origin) {
  const previous = globalThis.location;
  globalThis.location = new URL(origin);
  t.after(() => {
    globalThis.location = previous;
  });
}

test('generated integrity pins exactly match committed engine manifests', () => {
  assert.equal(readFileSync(OUTPUT, 'utf8'), serializePins(enginePins()));
});
test('good bytes verify at aliases and content-addressed URLs; same-size tampering fails even locally', async (t) => {
  at(t, 'http://localhost:5181');
  assert.equal(await loader.verify('/engine.wasm', good.buffer), good.buffer);
  await loader.verify(`/artifacts/sha256/${pin.sha256}/identity`, good.buffer);
  const bad = good.slice();
  bad[0] ^= 1;
  await assert.rejects(loader.verify('/engine.wasm', bad.buffer), /integrity check failed.*SHA-256 mismatch.*Retry/);
  await assert.rejects(loader.verify('/engine.wasm', new ArrayBuffer(1)), /expected .* bytes/);
});
test('missing pins fail closed before fetching on production and warn only on local origins', async (t) => {
  at(t, 'https://0x88.app');
  let fetches = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    fetches++;
    return new Response(good);
  });
  await assert.rejects(loader.bytes('/unregistered.wasm'), /no trusted SHA-256 pin/);
  assert.equal(fetches, 0);
  const warnings = [];
  t.mock.method(console, 'warn', (message) => warnings.push(message));
  globalThis.location = new URL('http://dev.localhost:5181');
  await loader.bytes('/unregistered.wasm');
  assert.ok(warnings.length);
  assert.equal(fetches, 1);
});
for (const corrupt of [good.map((v) => v ^ 1), new Uint8Array(1)]) {
  test(`a cached ${corrupt.length === 1 ? 'size' : 'hash'} mismatch is replaced once with reload`, async (t) => {
    at(t, 'https://0x88.app');
    const modes = [];
    t.mock.method(globalThis, 'fetch', async (_url, options) => {
      modes.push(options.cache);
      return new Response(modes.length === 1 ? corrupt : good);
    });
    assert.deepEqual(new Uint8Array(await loader.bytes('/engine.wasm')), good);
    assert.deepEqual(modes, ['default', 'reload']);
  });
}
test('persistent corruption fails after one reload and a later clean retry succeeds', async (t) => {
  at(t, 'https://0x88.app');
  const modes = [];
  t.mock.method(globalThis, 'fetch', async (_url, options) => {
    modes.push(options.cache);
    return new Response(modes.length <= 2 ? good.map((v) => v ^ 1) : good);
  });
  await assert.rejects(loader.bytes('/engine.wasm'), /SHA-256 mismatch.*Retry/);
  assert.deepEqual(modes, ['default', 'reload']);
  assert.deepEqual(new Uint8Array(await loader.bytes('/engine.wasm')), good);
  assert.deepEqual(modes, ['default', 'reload', 'default']);
});
