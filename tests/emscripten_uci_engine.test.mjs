import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';
import { PlentyChessEngine } from '../src/lc0/plentychessEngine.ts';
import { StormphraxEngine } from '../src/lc0/stormphraxEngine.ts';

const flush = async () => {
  for (let i = 0; i < 10; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

/** Structured-message stand-in for the Emscripten UCI worker. */
class EmscriptenWorker {
  static instances = [];
  static failNextInit = false;
  commands = [];
  holdSearch = false;
  terminated = false;
  failInit = false;
  constructor() {
    EmscriptenWorker.instances.push(this);
    this.failInit = EmscriptenWorker.failNextInit;
    EmscriptenWorker.failNextInit = false;
  }
  postMessage(message) {
    if (message.command) this.commands.push(message.command);
    queueMicrotask(() => {
      if (this.terminated) return;
      if (message.type === 'init' && this.failInit) {
        // A rejected glue fetch: the worker keeps its rejected module promise.
        this.onmessage?.({ data: { type: 'error', id: message.id, message: 'Failed to fetch glue' } });
        return;
      }
      const command = message.command;
      if (command?.startsWith('go ') && this.holdSearch) return;
      const line = command === 'uci' ? 'uciok' : command === 'isready' ? 'readyok' : command?.startsWith('go ') ? 'bestmove e2e4' : null;
      if (line) this.onmessage?.({ data: { type: 'line', stream: 'stdout', line } });
      this.onmessage?.({ data: { type: message.type === 'init' ? 'ready' : 'commandDone', id: message.id } });
    });
  }
  terminate() {
    this.terminated = true;
  }
}

function install(t) {
  const previousWorker = globalThis.Worker;
  const previousLocation = globalThis.location;
  globalThis.Worker = EmscriptenWorker;
  globalThis.location = { href: 'http://localhost:5181/arena/', origin: 'http://localhost:5181' };
  EmscriptenWorker.instances = [];
  EmscriptenWorker.failNextInit = false;
  t.after(() => {
    globalThis.Worker = previousWorker;
    if (previousLocation === undefined) delete globalThis.location;
    else globalThis.location = previousLocation;
  });
}

const ENGINES = [
  ['Berserk', BerserkEngine],
  ['PlentyChess', PlentyChessEngine],
  ['Stormphrax', StormphraxEngine],
];

for (const [label, Engine] of ENGINES) {
  test(`${label} recovers from a failed glue load by replacing the worker`, async (t) => {
    install(t);
    const engine = new Engine({ depth: 2 });
    t.after(() => engine.dispose());
    EmscriptenWorker.failNextInit = true;
    await assert.rejects(engine.prewarm(), /Failed to fetch glue/);
    assert.equal(EmscriptenWorker.instances[0].terminated, true, 'the poisoned worker is discarded');
    assert.equal(await engine.bestMove(START_FEN), 'e2e4');
    assert.equal(EmscriptenWorker.instances.length, 2);
  });

  test(`${label} gives depth searches the long timeout, then terminates the stuck worker`, async (t) => {
    install(t);
    t.mock.timers.enable({ apis: ['setTimeout'] });
    const engine = new Engine({ depth: 30 });
    t.after(() => engine.dispose());
    await engine.prewarm();
    const worker = EmscriptenWorker.instances[0];
    worker.holdSearch = true;
    let outcome = null;
    engine.bestMove(START_FEN).then(
      (move) => (outcome = move),
      (error) => (outcome = error),
    );
    await flush();
    assert.ok(worker.commands.some((command) => command.startsWith('go depth 30')));
    t.mock.timers.tick(61_000);
    await flush();
    assert.equal(outcome, null, 'a deep search is not cut off at the old fixed 60 s');
    t.mock.timers.tick(60_000);
    await flush();
    assert.match(outcome?.message ?? '', /Timed out waiting for bestmove/);
    assert.equal(worker.terminated, true, 'a timed-out synchronous search cannot be stopped, so its worker is terminated');
    assert.equal(await engine.bestMove(START_FEN), 'e2e4');
    assert.equal(EmscriptenWorker.instances.length, 2, 'the next search runs on a fresh worker instead of queueing behind the stuck one');
  });

  test(`${label} sends Hash/Threads once per worker`, async (t) => {
    install(t);
    const engine = new Engine({ depth: 2, hashMb: 64, threads: 1 });
    t.after(() => engine.dispose());
    await engine.bestMove(START_FEN);
    await engine.bestMove(START_FEN);
    await engine.analyze(START_FEN, { multipv: 2 });
    const options = EmscriptenWorker.instances[0].commands.filter((command) => /^setoption name (Hash|Threads) /.test(command));
    assert.deepEqual(options, ['setoption name Hash value 64', 'setoption name Threads value 1']);
  });
}
