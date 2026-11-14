import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { Bt4WorkerSearcher, T3_NET } from '../src/lc0/bt4Engine.ts';

// Execute the actual Arena warmup closure with the actual searcher and a
// controllable message transport, following the existing source-extraction tests.
const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
const start = source.indexOf('  const lc0BigNetWarmup =');
const end = source.indexOf('  const centipawnWarmup =', start);
assert.ok(start >= 0 && end > start);
const warmupSource = stripTypeScriptTypes(source.slice(start, end));

for (const phase of ['before init', 'during init', 'during search', 'no cancellation']) {
  test(`big-net warmup cancellation: ${phase}`, async (t) => {
    const workers = [];
    const messages = [];
    replaceGlobal(
      t,
      'Worker',
      class extends EventTarget {
        constructor() {
          super();
          workers.push(this);
        }
        postMessage(message) {
          messages.push(message);
          if (message.type === 'search' && phase !== 'during search') queueMicrotask(() => this.reply({ id: message.id, result: { visits: 1 } }));
          if (message.type === 'cancel') queueMicrotask(() => this.reply({ id: message.target, result: { visits: 0, cancelled: true } }));
          if (message.type === 'resetSearch') queueMicrotask(() => this.reply({ id: message.id }));
        }
        reply(data) {
          this.dispatchEvent(new MessageEvent('message', { data }));
        }
        terminate() {}
      },
    );
    const searcher = new Bt4WorkerSearcher(T3_NET);
    t.after(() => searcher.dispose());
    const warmup = new Function(
      'bigNetFor',
      'warmupPositions',
      'showDownloadProgress',
      'hideDownloadProgress',
      'arenaCacheEntries',
      `${warmupSource}\nreturn lc0BigNetWarmup('t3');`,
    )(
      () => ({ config: T3_NET, searcher }),
      [parseFen(START_FEN)],
      () => {},
      () => {},
      () => 2048,
    );
    const controller = new AbortController();
    if (phase === 'before init') controller.abort();
    const pending = warmup(controller.signal);
    if (phase === 'during init') controller.abort();
    const init = messages.find((message) => message.type === 'init');
    if (init) workers[0].reply({ id: init.id, backend: 'mock-webgpu' });
    if (phase === 'during search') {
      await new Promise(setImmediate);
      assert.ok(messages.some((message) => message.type === 'search'));
      controller.abort();
    }
    await pending;
    const commands = messages.filter((message) => message.type !== 'init').map((message) => message.type);
    assert.deepEqual(commands, phase === 'no cancellation' ? ['search', 'resetSearch'] : phase === 'during search' ? ['search', 'cancel'] : []);
    if (phase === 'before init') assert.equal(workers.length, 0);
  });
}

function replaceGlobal(t, key, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, key);
  Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  t.after(() => {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  });
}
