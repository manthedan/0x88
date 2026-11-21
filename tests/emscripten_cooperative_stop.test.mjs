import assert from 'node:assert/strict';
import test from 'node:test';
import { setImmediate as turn } from 'node:timers/promises';
import { START_FEN } from '../src/chess/board.ts';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';

function setup(t, isolated) {
  const workers = [];
  const globals = ['crossOriginIsolated', 'location', 'Worker'].map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]);
  globalThis.crossOriginIsolated = isolated;
  globalThis.location = new URL('https://0x88.app/app/analysis/');
  globalThis.Worker = class {
    messages = [];
    terminated = false;
    constructor() {
      workers.push(this);
    }
    emit(data) {
      this.onmessage?.({ data });
    }
    line(line) {
      this.emit({ type: 'line', stream: 'stdout', line });
    }
    postMessage(message) {
      this.messages.push(message);
      queueMicrotask(() => {
        if (this.terminated || message.command?.startsWith('go ')) return;
        if (message.command === 'uci') this.line('uciok');
        if (message.command === 'isready') this.line('readyok');
        this.emit({ type: message.type === 'init' ? 'ready' : 'commandDone', id: message.id });
      });
    }
    terminate() {
      this.terminated = true;
    }
  };
  const engine = new BerserkEngine({ depth: 6 });
  t.after(() => {
    engine.dispose();
    for (const [name, descriptor] of globals) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
  });
  return { engine, workers };
}

test('analysis move stepping drains bestmove and commandDone before reusing the same worker', async (t) => {
  const { engine, workers } = setup(t, true);
  await engine.prewarm();
  const worker = workers[0];
  const stop = new Int32Array(worker.messages[0].stopBuffer);
  assert.equal(stop.length, 1);
  const controller = new AbortController();
  const rejected = assert.rejects(engine.analyze(START_FEN, { signal: controller.signal, depth: 40 }), { name: 'AbortError' });
  await turn();
  const oldGo = worker.messages.at(-1);
  assert.equal(oldGo.command, 'go depth 40');
  controller.abort();
  assert.equal(Atomics.load(stop, 0), 1);
  const next = engine.bestMove(START_FEN, undefined, { moves: ['e2e4'] });
  worker.line('info depth 20 score cp 110 pv d2d4');
  worker.line('bestmove d2d4');
  await turn();
  assert.equal(worker.messages.at(-1), oldGo);
  worker.emit({ type: 'commandDone', id: oldGo.id });
  await rejected;
  await turn();
  assert.equal(workers.length, 1);
  assert.equal(worker.terminated, false);
  assert.equal(Atomics.load(stop, 0), 0);
  assert.equal(worker.messages.at(-2).command, `position fen ${START_FEN} moves e2e4`);
  worker.line('info depth 6 score cp 20 pv e7e5');
  worker.line('bestmove e7e5');
  worker.emit({ type: 'commandDone', id: worker.messages.at(-1).id });
  assert.equal(await next, 'e7e5');
  assert.equal(engine.lastInfo()[0].depth, 6);
});

for (const isolated of [false, true]) {
  test(`an Emscripten search falls back to worker retirement ${isolated ? 'after a stop timeout' : 'without shared memory'}`, async (t) => {
    const { engine, workers } = setup(t, isolated);
    await engine.prewarm();
    const controller = new AbortController();
    const rejected = assert.rejects(engine.bestMove(START_FEN, controller.signal), { name: 'AbortError' });
    await turn();
    controller.abort();
    assert.equal(workers[0].terminated, !isolated);
    await rejected;
    assert.equal(workers[0].terminated, true);
    await engine.prewarm();
    assert.equal(workers.length, 2);
  });
}
