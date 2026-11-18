import assert from 'node:assert/strict';
import test from 'node:test';
import { clearLc0ModelCacheEntry, loadLc0ModelForOrt } from '../src/lc0/modelCache.ts';
import { createCentipawnOrtSession, isPinnedCentipawnOnnx } from '../src/nn/centipawnModel.ts';

const URL = 'https://centipawn.test/models/s4fix_v_ema.onnx';
const SHA = 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad';
async function environment(t, create, { manifestAvailable = true, channelUrl = null } = {}) {
  const body = new Map(),
    metadata = new Map(),
    requests = [],
    loads = [],
    sessions = [];
  let wire = 'abc',
    writes = 0,
    clears = 0;
  const cache = {
    match: async (req) => body.get(req.url ?? req)?.clone(),
    put: async (req, res) => {
      writes++;
      body.set(req.url ?? req, res.clone());
    },
    delete: async (req) => body.delete(req.url ?? req),
    keys: async () => [...body.keys()].map((url) => new Request(url)),
  };
  const store = {
    get: async (key) => metadata.get(key),
    put: async (value) => metadata.set(value.key, value),
    delete: async (key) => metadata.delete(key),
    list: async () => [...metadata.values()],
  };
  const previousCaches = Object.getOwnPropertyDescriptor(globalThis, 'caches');
  Object.defineProperty(globalThis, 'caches', { configurable: true, value: { open: async () => cache, has: async () => true } });
  t.after(() => (previousCaches ? Object.defineProperty(globalThis, 'caches', previousCaches) : delete globalThis.caches));
  t.mock.method(globalThis, 'fetch', async (input, init) => {
    const url = input.url ?? input;
    requests.push({ url, cache: init?.cache ?? input.cache });
    if (url === '/models/centipawn.manifest.json')
      return manifestAvailable
        ? Response.json({ models: [{ file: 's4fix_v_ema.onnx', url: URL, bytes: 3, sha256: SHA }] })
        : new Response(null, { status: 503 });
    if (url === channelUrl) return Response.json({ schema: 'lc0_browser.artifact_channel_manifest.v1', releaseManifestUrl: '/releases/test.json' });
    if (url === 'https://untrusted-assets.test/releases/test.json')
      return Response.json({
        schema: 'lc0_browser.artifact_release_manifest.v1',
        artifacts: [{ file: 's4fix_v_ema.onnx', logicalUrl: '/models/s4fix_v_ema.onnx', artifactUrl: URL, bytes: 3, sha256: SHA }],
      });
    if (decodeURIComponent(String(url)).split('?')[0] === URL) return new Response(wire);
    return new Response(null, { status: 404 });
  });
  const cacheName = `centipawn-test-${t.name}`;
  const deps = {
    load: async (url, options) => {
      assert.equal(options.manifestUrl, '/models/centipawn.manifest.json');
      assert.equal(options.allowUnverified, false);
      assert.equal(options.requireManifestEntry, true);
      assert.equal(options.cache, true);
      const result = await loadLc0ModelForOrt(url, { ...options, cacheName, metadataStore: store, channelUrl });
      loads.push(result);
      return result;
    },
    clear: async (url) => {
      clears++;
      return clearLc0ModelCacheEntry(url, cacheName, store);
    },
    create: async (input, options) => {
      sessions.push({ input, options });
      return create ? create(input, options, sessions.length) : {};
    },
  };
  return {
    deps,
    body,
    metadata,
    loads,
    sessions,
    requests,
    setWire: (value) => {
      wire = value;
    },
    counts: () => ({ writes, clears }),
  };
}

test('the pinned loader follows the ONNX artifact identity, not metadata or a claimed model ID', () => {
  assert.equal(isPinnedCentipawnOnnx('/models/s4fix_v_ema.onnx?v=1'), true);
  assert.equal(isPinnedCentipawnOnnx('https://assets.0x88.app/models/s4fix_v_ema.onnx'), true);
  assert.equal(isPinnedCentipawnOnnx('/models/%734fix_v_ema.onnx'), true);
  assert.equal(isPinnedCentipawnOnnx('/models/%E9/s4fix_v_ema.onnx'), true, 'invalid UTF-8 in a directory must not bypass a valid shipped basename');
  assert.equal(isPinnedCentipawnOnnx('/models/s4fix_v_ema.meta.json'), false);
  assert.equal(isPinnedCentipawnOnnx('/models/custom.onnx'), false);
});

test('Centipawn uses manifest-pinned SHA-256 and Cache Storage; second creation and lazy fallback read verified cache', async (t) => {
  const env = await environment(t);
  await createCentipawnOrtSession(URL, env.deps);
  await createCentipawnOrtSession(URL, env.deps);
  assert.equal(env.loads[0].verification, 'hashed');
  assert.equal(env.loads[0].sha256Valid, true);
  assert.equal(env.loads[1].cacheStatus, 'hit');
  assert.equal(env.requests.filter((r) => r.url === URL).length, 1);
  assert.equal(env.counts().writes, 1);
  assert.deepEqual(new Uint8Array(env.sessions[0].input), new TextEncoder().encode('abc'));
  assert.equal(env.sessions[0].options.rejectModelParseErrors(), false);
  await env.sessions[0].options.reloadModel();
  assert.equal(env.requests.filter((r) => r.url === URL).length, 1);
  assert.equal(env.sessions[0].options.rejectModelParseErrors(), true, 'fallback parse recovery follows the reloaded provenance');
});

test('Centipawn cannot replace an unavailable app pin with an asset-host-supplied digest', async (t) => {
  const channelUrl = 'https://untrusted-assets.test/channels/test.json';
  const env = await environment(t, undefined, { manifestAvailable: false, channelUrl });
  const channelOnly = await loadLc0ModelForOrt(URL, { cache: false, manifestUrl: '/models/centipawn.manifest.json', channelUrl });
  assert.equal(channelOnly.sha256Valid, true, 'fixture really offers otherwise valid channel-supplied bytes and digest');
  env.requests.length = 0;
  await assert.rejects(createCentipawnOrtSession(URL, env.deps), /App-origin model integrity metadata is unavailable/);
  assert.deepEqual(
    env.requests.map((r) => r.url),
    ['/models/centipawn.manifest.json'],
  );
  assert.equal(env.sessions.length, 0);
  assert.equal(env.counts().writes, 0);
});

test('percent-encoded Centipawn filenames use the app pin while preserving the requested fetch URL', async (t) => {
  const env = await environment(t),
    requested = URL.replace('/s4fix', '/%734fix');
  await createCentipawnOrtSession(requested, env.deps);
  assert.equal(env.loads[0].sha256Valid, true);
  assert.ok(env.requests.some((r) => r.url === requested));
});

test('Centipawn query overrides find the same app pin and preserve the requested verified fetch URL', async (t) => {
  const env = await environment(t),
    requested = `${URL}?v=1`;
  await createCentipawnOrtSession(requested, env.deps);
  assert.equal(env.loads[0].sha256Valid, true);
  assert.equal(env.loads[0].verification, 'hashed');
  assert.equal(env.requests.filter((request) => request.url === requested).length, 1);
  assert.equal(env.sessions.length, 1);
});

test('a queried Centipawn override still rejects a wrong hash before session creation', async (t) => {
  const env = await environment(t);
  env.setWire('bad');
  await assert.rejects(createCentipawnOrtSession(`${URL}?v=2`, env.deps), /sha256 mismatch/i);
  assert.equal(env.sessions.length, 0);
  assert.equal(env.counts().writes, 0);
});

test('Centipawn hash mismatch rejects before session creation or persistent cache writes', async (t) => {
  const env = await environment(t);
  env.setWire('bad');
  await assert.rejects(createCentipawnOrtSession(URL, env.deps), /sha256 mismatch/i);
  assert.equal(env.sessions.length, 0);
  assert.equal(env.counts().writes, 0);
});

for (const corruptFresh of [false, true])
  test(`Centipawn trusted-cache parse repair is bounded and freshly verifies bytes (corrupt fresh=${corruptFresh})`, async (t) => {
    let poison = false;
    const env = await environment(t, (_input, options) => {
      if (poison && options.rejectModelParseErrors()) {
        poison = false;
        throw new Error('invalid protobuf model');
      }
      return {};
    });
    await createCentipawnOrtSession(URL, env.deps);
    // Same length, with verification metadata left untouched: this is the shared
    // loader's intentionally metadata-trusted warm-cache threat model.
    env.body.set(URL, new Response('bad'));
    poison = true;
    if (corruptFresh) {
      env.setWire('bad');
      await assert.rejects(createCentipawnOrtSession(URL, env.deps), /sha256 mismatch/i);
    } else await createCentipawnOrtSession(URL, env.deps);
    assert.equal(env.counts().clears, 1);
    assert.equal(env.requests.filter((r) => r.url === URL && r.cache === 'reload').length, 1);
    assert.equal(env.sessions.length, corruptFresh ? 2 : 3);
    if (!corruptFresh) {
      assert.equal(env.loads.at(-1).verification, 'hashed');
      assert.equal(await env.body.get(URL).clone().text(), 'abc');
    }
  });
