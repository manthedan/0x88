import assert from 'node:assert/strict';
import { createHash, webcrypto } from 'node:crypto';
import test from 'node:test';
import vm from 'node:vm';
import { emscriptenUciWorkerScript } from '../src/lc0/emscriptenUciEngine.ts';
import { ENGINE_ARTIFACT_PINS } from '../src/lc0/engineArtifactPins.generated.ts';
import { stockfishWorkerUrl } from '../src/lc0/stockfishEngine.ts';

const bytes = (s) => new TextEncoder().encode(s);
const files = {
  '/fixture/engine.js': bytes('trusted glue'),
  '/fixture/engine.wasm': Uint8Array.of(0, 97, 115, 109, 1, 0, 0, 0),
  '/fixture/engine.data': bytes('trusted data'),
};
const pins = Object.fromEntries(
  Object.entries(files).map(([path, data]) => [path, { bytes: data.length, sha256: createHash('sha256').update(data).digest('hex') }]),
);
const flush = async () => {
  for (let i = 0; i < 20; i++) await new Promise((resolve) => setTimeout(resolve, 2));
};

function sandbox(tamper) {
  const messages = [];
  let imports = 0;
  let factoryOptions;
  const context = {
    location: new URL('https://0x88.app/worker.js'),
    URL,
    Blob,
    Response,
    TextDecoder,
    Uint8Array,
    WebAssembly,
    crypto: webcrypto,
    fetch: async (url) => {
      const path = new URL(url, 'https://0x88.app').pathname;
      const data = files[path].slice();
      if (path === tamper) data[0] ^= 1;
      return new Response(data);
    },
    importScripts: () => {
      imports++;
      context.Factory = async (options) => {
        factoryOptions = options;
        options.instantiateWasm({}, (instance) => assert(instance instanceof WebAssembly.Instance));
        const preload = options.getPreloadedPackage();
        assert.deepEqual([...new Uint8Array(preload)], [...files['/fixture/engine.data']]);
        assert.throws(() => options.getPreloadedPackage(), /Missing verified.*preload data/);
        options.preRun = [() => preload];
        return options;
      };
      context.onmessage = () => {};
    },
    postMessage: (message) => messages.push(message),
    Worker: class {},
    setTimeout: (fn) => {
      try {
        fn();
      } catch (error) {
        messages.push({ message: error.message });
      }
    },
  };
  context.self = context;
  return { context, messages, imports: () => imports, factoryOptions: () => factoryOptions };
}
function withPins(script) {
  return script.replace(JSON.stringify(ENGINE_ARTIFACT_PINS), JSON.stringify(pins));
}

for (const artifact of ['js', 'wasm', 'data']) {
  test(`Emscripten rejects corrupt ${artifact} before executing glue`, async () => {
    const env = sandbox(`/fixture/engine.${artifact}`);
    vm.runInNewContext(withPins(emscriptenUciWorkerScript({ label: 'Fixture', factoryGlobal: 'Factory' })), env.context);
    env.context.onmessage({
      data: { type: 'init', id: 1, jsUrl: '/fixture/engine.js', wasmUrl: '/fixture/engine.wasm', dataUrl: '/fixture/engine.data', trustedJsUrl: true },
    });
    await flush();
    assert.equal(env.imports(), 0);
    assert.match(env.messages.at(-1)?.message ?? '', /integrity check failed.*SHA-256 mismatch/);
  });
}
test('Emscripten passes only verified WASM and preload bytes to the factory', async () => {
  const env = sandbox();
  vm.runInNewContext(withPins(emscriptenUciWorkerScript({ label: 'Fixture', factoryGlobal: 'Factory' })), env.context);
  env.context.onmessage({
    data: { type: 'init', id: 1, jsUrl: '/fixture/engine.js', wasmUrl: '/fixture/engine.wasm', dataUrl: '/fixture/engine.data', trustedJsUrl: true },
  });
  await flush();
  assert.equal(env.imports(), 1);
  assert.equal(env.messages.at(-1)?.type, 'ready');
  for (const field of ['instantiateWasm', 'getPreloadedPackage', 'preRun']) assert(!Object.hasOwn(env.factoryOptions(), field), field);
});
test('Emscripten starts all asset fetches before the binaries finish', async () => {
  const env = sandbox();
  const binaryGate = Promise.withResolvers();
  const requested = [];
  const fetch = env.context.fetch;
  env.context.fetch = async (url) => {
    const name = new URL(url).pathname;
    requested.push(name);
    if (name.endsWith('.wasm') || name.endsWith('.data')) await binaryGate.promise;
    return fetch(url);
  };
  vm.runInNewContext(withPins(emscriptenUciWorkerScript({ label: 'Fixture', factoryGlobal: 'Factory' })), env.context);
  env.context.onmessage({ data: { type: 'init', id: 1, jsUrl: '/fixture/engine.js', trustedJsUrl: true } });
  await new Promise(setImmediate);
  assert.deepEqual(requested.sort(), ['/fixture/engine.data', '/fixture/engine.js', '/fixture/engine.wasm']);
  assert.equal(env.imports(), 0);
  binaryGate.resolve();
  await flush();
  assert.equal(env.messages.at(-1)?.type, 'ready');
});
test('the Emscripten factory can read the stop flag without a worker message', async () => {
  const env = sandbox();
  vm.runInNewContext(withPins(emscriptenUciWorkerScript({ label: 'Fixture', factoryGlobal: 'Factory' })), env.context);
  const stopBuffer = new SharedArrayBuffer(4);
  env.context.onmessage({ data: { type: 'init', id: 1, jsUrl: '/fixture/engine.js', trustedJsUrl: true, stopBuffer } });
  await flush();
  assert.equal(env.factoryOptions().shouldStop(), 0);
  Atomics.store(new Int32Array(stopBuffer), 0, 1);
  assert.equal(env.factoryOptions().shouldStop(), 1);
});
test('Emscripten default constructors verify the sibling preload requested by glue', async () => {
  const env = sandbox();
  vm.runInNewContext(withPins(emscriptenUciWorkerScript({ label: 'Fixture', factoryGlobal: 'Factory' })), env.context);
  env.context.onmessage({ data: { type: 'init', id: 1, jsUrl: '/fixture/engine.js', trustedJsUrl: true } });
  await flush();
  assert.equal(env.imports(), 1);
  assert.equal(env.messages.at(-1)?.type, 'ready', JSON.stringify(env.messages));
});
for (const artifact of ['js', 'wasm']) {
  test(`Stockfish rejects corrupt ${artifact} before executing glue or starting pthreads`, async () => {
    let script;
    const create = URL.createObjectURL;
    URL.createObjectURL = (blob) => {
      script = blob.text();
      return 'blob:https://0x88.app/fixture';
    };
    try {
      stockfishWorkerUrl('/fixture/engine.js');
    } finally {
      URL.createObjectURL = create;
    }
    const env = sandbox(`/fixture/engine.${artifact}`);
    vm.runInNewContext(withPins(await script), env.context);
    await flush();
    assert.equal(env.imports(), 0);
    assert.match(env.messages.at(-1)?.message ?? '', /integrity check failed.*SHA-256 mismatch/);
  });
}

test('Stockfish supplies verified WASM once and starts pthreads from verified glue bytes', async () => {
  let script;
  const create = URL.createObjectURL;
  URL.createObjectURL = (blob) => {
    script = blob.text();
    return 'blob:https://0x88.app/fixture';
  };
  try {
    stockfishWorkerUrl('/fixture/engine.js');
  } finally {
    URL.createObjectURL = create;
  }
  const env = sandbox();
  const children = [];
  env.context.Worker = class {
    constructor(url, options) {
      children.push({ url, options });
    }
  };
  vm.runInNewContext(withPins(await script), env.context);
  env.context.onmessage({ data: 'uci' }); // the real adapter posts before fetch completes
  await flush();
  assert.equal(env.imports(), 1);
  assert.equal(env.messages.length, 0);
  const response = await env.context.fetch('http://localhost/fixture/engine.wasm');
  assert.equal(response.headers.get('content-type'), 'application/wasm');
  assert.deepEqual([...new Uint8Array(await response.arrayBuffer())], [...files['/fixture/engine.wasm']]);
  await assert.rejects(env.context.fetch('http://localhost/fixture/engine.wasm'), /already consumed/);
  new env.context.Worker('/unverified-worker.js', { name: 'em-pthread' });
  assert.match(children[0].url, /^blob:/);
  assert.equal(children[0].options.name, 'em-pthread');
});
