import assert from 'node:assert/strict';
import { copyFileSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';

const require = createRequire(import.meta.url);
const MiB = 1024 * 1024;
const tiers = {
  berserk: ['', '-simd128', '-relaxed-simd128'],
  plentychess: ['', '-sse41', '-relaxed-simd128'],
  stormphrax: ['', '-relaxed-simd128'],
};
const unlinked = { plentychess: 'processed.bin', stormphrax: 'undertown.nnue' };
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

for (const [engine, suffixes] of Object.entries(tiers)) {
  for (const suffix of suffixes) {
    test(`${engine}${suffix}: rebuilt WASM stops cooperatively, then searches again`, async (t) => {
      const base = `${engine}-emscripten${suffix}`;
      const dir = resolve('public', engine);
      const wasm = join(dir, `${base}.wasm`);
      const data = join(dir, `${engine}-emscripten.data`);
      const missing = [wasm, data].find(fixtureAbsent);
      if (missing) return t.skip(`Engine fixture not hydrated: ${missing}`);
      const tmp = mkdtempSync(join(tmpdir(), 'emscripten-uci-'));
      t.after(() => rmSync(tmp, { recursive: true, force: true }));
      const glue = join(tmp, `${base}.cjs`);
      copyFileSync(join(dir, `${base}.js`), glue);
      const compiled = new WebAssembly.Module(readFileSync(wasm));
      const stdout = [];
      const flag = new Int32Array(new SharedArrayBuffer(4));
      let stopReads = 0;
      const mod = await require(glue)({
        instantiateWasm(imports, receiveInstance) {
          const instance = new WebAssembly.Instance(compiled, imports);
          const memory = Object.values(instance.exports).find((value) => value instanceof WebAssembly.Memory);
          assert.equal(memory.buffer.byteLength, 64 * MiB);
          assert.throws(() => memory.grow((512 - 64) * 16 + 1), RangeError);
          return receiveInstance(instance);
        },
        locateFile: () => data,
        print: (line) => stdout.push(line),
        shouldStop: () => {
          const value = Atomics.load(flag, 0);
          stopReads += value;
          return value;
        },
      });
      const command = (line) => mod.ccall('command', null, ['string'], [line]);
      if (unlinked[engine]) {
        // The packager exposes the MEMFS create call. It throws EEXIST if the
        // real file is still there; insert an empty file to catch any reread.
        mod.FS_createDataFile('/', unlinked[engine], new Uint8Array(), true, false, false);
      }
      command('uci');
      command('setoption name Hash value 128');
      command('ucinewgame');
      command('isready');
      assert(stdout.includes('readyok'));
      command('position startpos');
      Atomics.store(flag, 0, 1);
      command('go depth 8');
      assert(stopReads > 0, 'the stop flag must be consumed in depth-limited searches');
      assert.match(stdout.at(-1), /^bestmove [a-h][1-8]/);
      assert(!stdout.some((line) => /^info depth 8 /.test(line)));
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
