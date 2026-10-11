import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';
import { StormphraxEngine } from '../src/lc0/stormphraxEngine.ts';

class UciWorker {
  static instances = [];
  commands = [];
  failCommand = null;
  holdSearch = false;
  postMessage(message) {
    const command = typeof message === 'string' ? message : message.command;
    if (command) this.commands.push(command);
    queueMicrotask(() => {
      if (command === this.failCommand) {
        this.failCommand = null;
        this.onmessage?.({ data: { type: 'error', id: message.id, message: 'option failed' } });
        return;
      }
      const line = command === 'uci' ? 'uciok' : command === 'isready' ? 'readyok' : command?.startsWith('go ') && !this.holdSearch ? 'bestmove e2e4' : null;
      if (line) this.onmessage?.({ data: typeof message === 'string' ? line : { type: 'line', stream: 'stdout', line } });
      if (typeof message !== 'string' && !(this.holdSearch && command?.startsWith('go '))) {
        this.onmessage?.({ data: { type: message.type === 'init' ? 'ready' : 'commandDone', id: message.id } });
      }
    });
  }
  constructor() {
    UciWorker.instances.push(this);
  }
  terminate() {
    this.terminated = true;
  }
}

function withWorkers(fn) {
  return async () => {
    const previousWorker = globalThis.Worker;
    const previousLocation = globalThis.location;
    globalThis.Worker = UciWorker;
    globalThis.location = { href: 'http://localhost:5181/app/play/', origin: 'http://localhost:5181' };
    UciWorker.instances = [];
    try {
      await fn();
    } finally {
      globalThis.Worker = previousWorker;
      if (previousLocation === undefined) delete globalThis.location;
      else globalThis.location = previousLocation;
    }
  };
}

const options = (worker) => worker.commands.filter((command) => /^setoption name (Hash|Threads) /.test(command));
for (const Engine of [BerserkEngine, StormphraxEngine]) {
  test(
    `${Engine.name} dedupes normalized settings across bestMove/analyze and preserves newGame`,
    withWorkers(async () => {
      const engine = new Engine({ hashMb: 64, threads: 1, depth: 2 });
      try {
        await engine.bestMove(START_FEN);
        const worker = UciWorker.instances[0];
        await engine.analyze(START_FEN, { depth: 2 });
        assert.deepEqual(options(worker), ['setoption name Hash value 64', 'setoption name Threads value 1']);
        engine.setOptions({ hashMb: 64.9, threads: 0, depth: 3 });
        await engine.bestMove(START_FEN);
        assert.equal(options(worker).length, 2, 'equivalent clamped settings are not changes');
        engine.setOptions({ hashMb: 32, threads: 2 });
        await engine.analyze(START_FEN);
        assert.deepEqual(options(worker).slice(2), ['setoption name Hash value 32', 'setoption name Threads value 2']);
        await engine.newGame();
        await engine.bestMove(START_FEN);
        assert.equal(worker.commands.filter((command) => command === 'ucinewgame').length, 1);
        assert.equal(options(worker).length, 4, 'newGame clears TT but retains option settings');
        assert.equal(UciWorker.instances.length, 1);
      } finally {
        engine.dispose();
      }
    }),
  );

  test(
    `${Engine.name} retries unacknowledged settings and restores them after worker failure/abort`,
    withWorkers(async () => {
      const engine = new Engine({ hashMb: 64, threads: 1, depth: 2 });
      try {
        await engine.prewarm();
        const worker = UciWorker.instances[0];
        engine.setOptions({ hashMb: 32 });
        worker.failCommand = 'setoption name Hash value 32';
        await assert.rejects(engine.bestMove(START_FEN), /option failed/);
        await engine.bestMove(START_FEN);
        assert.equal(UciWorker.instances.flatMap((instance) => instance.commands).filter((command) => command === 'setoption name Hash value 32').length, 2);
        UciWorker.instances.at(-1).onerror({ message: 'worker failed' });
        await engine.bestMove(START_FEN);
        const replacement = UciWorker.instances.at(-1);
        assert.deepEqual(options(replacement), ['setoption name Hash value 32', 'setoption name Threads value 1']);
        replacement.holdSearch = true;
        const controller = new AbortController();
        const search = engine.bestMove(START_FEN, controller.signal);
        while (
          !replacement.commands.some((command) => command.startsWith('go ')) ||
          replacement.commands.filter((command) => command.startsWith('go ')).length < 2
        ) {
          await new Promise((resolve) => setImmediate(resolve));
        }
        controller.abort();
        await assert.rejects(search, { name: 'AbortError' });
        await engine.bestMove(START_FEN);
        assert.notEqual(UciWorker.instances.at(-1), replacement);
        assert.deepEqual(options(UciWorker.instances.at(-1)), ['setoption name Hash value 32', 'setoption name Threads value 1']);
      } finally {
        engine.dispose();
      }
    }),
  );
}

test(
  'Stockfish dedupes costly Hash/Threads setOptions and restores them on replacement',
  withWorkers(async () => {
    const engine = new StockfishEngine({ hashMb: 64, threads: 1 }, '/stockfish/stockfish-19-lite-single.js');
    try {
      await engine.prewarm();
      const worker = UciWorker.instances[0];
      engine.setOptions({ threads: 1, hashMb: 64 });
      engine.setOptions({ threads: 0, hashMb: 64.9 });
      assert.deepEqual(options(worker), ['setoption name Threads value 1', 'setoption name Hash value 64']);
      engine.setOptions({ threads: 2, hashMb: 32 });
      assert.deepEqual(options(worker).slice(2), ['setoption name Threads value 2', 'setoption name Hash value 32']);
      await engine.newGame();
      engine.setOptions({ threads: 2, hashMb: 32 });
      assert.equal(options(worker).length, 4);
      worker.onerror({ message: 'worker failed', preventDefault() {} });
      await engine.prewarm();
      assert.deepEqual(options(UciWorker.instances[1]), ['setoption name Threads value 2', 'setoption name Hash value 32']);
    } finally {
      engine.dispose();
    }
  }),
);
