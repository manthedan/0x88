import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';
import { parseFen, START_FEN } from '../src/chess/board.ts';
import { Bt4WorkerSearcher, LQO_NET } from '../src/lc0/bt4Engine.ts';

// Execute the production closures, not a reimplementation; UI sinks are spies.
const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
for (const operation of ['warmup', 'move']) {
  for (const cancelled of [true, false]) {
    test(`Arena ${operation} progress stays with its ${cancelled ? 'cancelled' : 'live'} operation`, async (t) => {
      const workers = [];
      const events = [];
      const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
      globalThis.Worker = class extends EventTarget {
        messages = [];
        constructor() {
          super();
          workers.push(this);
        }
        postMessage(message) {
          this.messages.push(message);
          if (message.type === 'search') queueMicrotask(() => this.reply({ id: message.id, result: { visits: 1, move: 'e2e4' } }));
          if (message.type === 'resetSearch') queueMicrotask(() => this.reply({ id: message.id }));
        }
        reply(data) {
          this.dispatchEvent(new MessageEvent('message', { data }));
        }
        terminate() {}
      };
      const searcher = new Bt4WorkerSearcher(LQO_NET);
      t.after(() => {
        searcher.dispose();
        if (previousWorker) Object.defineProperty(globalThis, 'Worker', previousWorker);
        else delete globalThis.Worker;
      });
      const name = operation === 'warmup' ? 'lc0BigNetWarmup' : 'lc0BigNetMove';
      const start = source.indexOf(`  const ${name} =`);
      const end = source.indexOf(operation === 'warmup' ? '  const centipawnWarmup =' : '  const centipawnMove =', start);
      assert.ok(start >= 0 && end > start);
      const factory = new Function(
        'deps',
        `const { bigNetFor, warmupPositions, showDownloadProgress, hideDownloadProgress, arenaCacheEntries, engines, arenaBudgetMode, arenaMovetimeMs, setArenaVisitProgress, recordBt4SearchOutput, clearArenaSearchProgress } = deps;\n${stripTypeScriptTypes(source.slice(start, end))}\nreturn ${name};`,
      )({
        bigNetFor: () => ({ config: LQO_NET, searcher }),
        warmupPositions: [parseFen(START_FEN)],
        showDownloadProgress: (...args) => events.push(['show', ...args]),
        hideDownloadProgress: () => events.push(['hide']),
        arenaCacheEntries: () => 2048,
        engines: new Map(),
        arenaBudgetMode: () => 'visits',
        arenaMovetimeMs: () => 1000,
        setArenaVisitProgress: () => {},
        recordBt4SearchOutput: () => {},
        clearArenaSearchProgress: () => {},
      });
      const controller = new AbortController();
      const pending =
        operation === 'warmup'
          ? factory('lqo')(controller.signal)
          : factory('seat-1', { variant: 'lqo', strength: 1 })([parseFen(START_FEN)], controller.signal);
      const worker = workers[0];
      assert.ok(worker, 'operation is awaiting shared model initialization');
      if (cancelled) {
        controller.abort();
        assert.deepEqual(events.at(-1), ['hide'], "Stop releases this operation's progress display at once");
      }
      // After Stop, a replacement progress surface now belongs to another op.
      events.length = 0;
      worker.reply({ type: 'downloadProgress', loadedBytes: 20, totalBytes: 100 });
      const init = worker.messages.find((message) => message.type === 'init');
      worker.reply({ id: init.id, backend: 'mock-webgpu' });
      await pending;
      assert.deepEqual(
        events,
        cancelled ? [] : [['show', 'Lc0 Queen Odds', 20, 100, 'Downloading'], ['hide']],
        'old download callbacks/finally must not modify the replacement surface',
      );
    });
  }
}
