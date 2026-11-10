import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createEngineArtifactLoader } from '../src/lc0/engineArtifacts.ts';

test('engine artifacts never force-cache mutable aliases or retain unverified HTTP bodies', async (t) => {
  const previous = globalThis.location;
  globalThis.location = new URL('http://localhost:5181');
  t.after(() => {
    globalThis.location = previous;
  });
  t.mock.method(console, 'warn', () => {});
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    seen.push([url, options.cache]);
    return new Response('local experiment');
  });
  const loader = createEngineArtifactLoader({});
  await loader.bytes('/reckless/local.wasm');
  assert.deepEqual(seen, [['/reckless/local.wasm', 'no-store']]);
  for (const file of ['emscriptenUciEngine.ts', 'recklessWasiWorker.ts', 'recklessBrowserApiWorker.ts', 'stockfishEngine.ts']) {
    const source = readFileSync(new URL(`../src/lc0/${file}`, import.meta.url), 'utf8');
    assert.doesNotMatch(source, /cache:\s*['"]force-cache['"]/, file);
    assert.match(source, /engineArtifact/, `${file} must use the shared verified loader`);
  }
});
