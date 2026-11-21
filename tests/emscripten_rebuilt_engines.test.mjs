import assert from 'node:assert/strict';
import { webcrypto } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createContext, runInContext } from 'node:vm';
import { emscriptenUciWorkerScript } from '../src/lc0/emscriptenUciEngine.ts';

const MiB = 1024 * 1024;
const tiers = {
  berserk: ['', '-simd128', '-relaxed-simd128'],
  plentychess: ['', '-sse41', '-relaxed-simd128'],
  stormphrax: ['', '-relaxed-simd128'],
};
const unlinked = { plentychess: 'processed.bin', stormphrax: 'undertown.nnue' };
const factoryNames = { berserk: 'Berserk', plentychess: 'PlentyChess', stormphrax: 'Stormphrax' };
const lfsPointerHeader = 'version https://git-lfs.github.com/spec/v1\n';

function fixtureAbsent(path) {
  const stat = statSync(path, { throwIfNoEntry: false });
  return !stat || (stat.size < 1024 && readFileSync(path, 'utf8').startsWith(lfsPointerHeader));
}

test('missing Emscripten fixtures include LFS pointers but not corrupt bodies', (t) => {
  const tmp = mkdtempSync(join(tmpdir(), 'emscripten-lfs-'));
  t.after(() => rmSync(tmp, { recursive: true, force: true }));
  const path = join(tmp, 'fixture.wasm');
  assert.equal(fixtureAbsent(path), true);
  writeFileSync(path, `${lfsPointerHeader}oid sha256:${'0'.repeat(64)}\nsize 128000\n`);
  assert.equal(fixtureAbsent(path), true);
  writeFileSync(path, 'corrupt body');
  assert.equal(fixtureAbsent(path), false);
});

async function startWorker(engine, jsUrl, wasmUrl, dataUrl, stop) {
  const factoryGlobal = factoryNames[engine];
  const source = readFileSync(`public${jsUrl}`, 'utf8');
  const stdout = [];
  let mod;
  let instances = 0;
  const requests = new Set();
  const { promise: ready, resolve, reject } = Promise.withResolvers();
  const context = createContext({
    location: new URL('https://0x88.app/worker.js'),
    WorkerGlobalScope: class {},
    URL,
    Blob,
    Response,
    TextDecoder,
    Uint8Array,
    console,
    performance,
    crypto: webcrypto,
    WebAssembly: Object.assign(Object.create(WebAssembly), {
      Instance: class extends WebAssembly.Instance {
        constructor(module, imports) {
          super(module, imports);
          instances++;
          const memory = Object.values(this.exports).find((value) => value instanceof WebAssembly.Memory);
          assert.equal(memory.buffer.byteLength, 64 * MiB);
          assert.throws(() => memory.grow((512 - 64) * 16 + 1), RangeError);
        }
      },
    }),
    fetch: async (url) => {
      const { pathname } = new URL(url);
      assert(!requests.has(pathname), `worker fetched ${pathname} twice`);
      requests.add(pathname);
      return new Response(readFileSync(`public${pathname}`));
    },
    importScripts() {
      runInContext(source, context);
      const factory = context[factoryGlobal];
      context[factoryGlobal] = async (options) => (mod = await factory(options));
    },
    postMessage(message) {
      if (message.type === 'ready') resolve();
      else if (message.type === 'error') reject(new Error(message.message));
      else if (message.type === 'line' && message.stream === 'stdout') stdout.push(message.line);
    },
  });
  context.self = context;
  runInContext(emscriptenUciWorkerScript({ label: engine, factoryGlobal }), context);
  context.onmessage({ data: { type: 'init', id: 1, jsUrl, wasmUrl, dataUrl, stopBuffer: stop.buffer } });
  await ready;
  assert.deepEqual(requests, new Set([jsUrl, wasmUrl, dataUrl]));
  assert.equal(instances, 1);
  for (const field of ['getPreloadedPackage', 'preRun']) assert(!Object.hasOwn(mod, field), field);
  return { mod, stdout };
}

for (const [engine, suffixes] of Object.entries(tiers)) {
  for (const suffix of suffixes) {
    test(`${engine}${suffix}: rebuilt WASM stops cooperatively, then searches again`, async (t) => {
      const base = `/${engine}/${engine}-emscripten${suffix}`;
      const wasmUrl = `${base}.wasm`;
      const dataUrl = `/${engine}/${engine}-emscripten.data`;
      const missing = [`public${wasmUrl}`, `public${dataUrl}`].find(fixtureAbsent);
      if (missing) return t.skip(`Engine fixture not hydrated: ${missing}`);
      const flag = new Int32Array(new SharedArrayBuffer(4));
      const { mod, stdout } = await startWorker(engine, `${base}.js`, wasmUrl, dataUrl, flag);
      const command = (line) => mod.ccall('command', null, ['string'], [line]);
      if (unlinked[engine]) {
        // The packager exposes the MEMFS create call. It throws EEXIST if the
        // real file is still there; insert an empty file to catch any reread.
        mod.FS_createDataFile('/', unlinked[engine], new Uint8Array(), true, false, false);
      }
      command('uci');
      command(`setoption name Hash value ${suffix ? 128 : 256}`);
      command('ucinewgame');
      command('isready');
      assert(stdout.includes('readyok'));
      command('position startpos');
      Atomics.store(flag, 0, 1);
      command('go depth 8');
      assert.match(stdout.at(-1), /^bestmove [a-h][1-8]/);
      assert(!stdout.some((line) => /^info depth 8 /.test(line)), 'the stop flag must be consumed in depth-limited searches');
      Atomics.store(flag, 0, 0);
      stdout.length = 0;
      command('ucinewgame');
      command('position startpos moves e2e4');
      command('go depth 4');
      assert(stdout.some((line) => /^info depth 4 /.test(line)));
      assert.match(stdout.at(-1), /^bestmove [a-h][1-8]/);
    });
  }
}
