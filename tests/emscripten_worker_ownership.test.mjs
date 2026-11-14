import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { BerserkEngine } from '../src/lc0/berserkEngine.ts';
import { PlentyChessEngine } from '../src/lc0/plentychessEngine.ts';
import { StormphraxEngine } from '../src/lc0/stormphraxEngine.ts';

const turn = () => new Promise(setImmediate);

// Structured command ids are unique, but stdout and error events are untagged.
// Exercise real public adapters with callbacks queued before worker retirement.
for (const Engine of [BerserkEngine, PlentyChessEngine, StormphraxEngine]) {
  for (const event of ['line', 'error']) {
    test(`${Engine.name}: retired worker ${event} cannot affect replacement search`, async (t) => {
      const workers = [];
      const previousWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
      const previousLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
      globalThis.location = { href: 'https://audit.test/app/arena/', origin: 'https://audit.test' };
      globalThis.Worker = class {
        messages = [];
        terminated = false;
        constructor() {
          workers.push(this);
        }
        postMessage(message) {
          this.messages.push(message);
          queueMicrotask(() => {
            if (this.terminated) return;
            const command = message.command;
            // A running synchronous search can emit lines before commandDone.
            if (command?.startsWith('go ')) return;
            const line = command === 'uci' ? 'uciok' : command === 'isready' ? 'readyok' : null;
            if (line) this.emit({ type: 'line', stream: 'stdout', line });
            this.emit({ type: message.type === 'init' ? 'ready' : 'commandDone', id: message.id });
          });
        }
        emit(data) {
          this.onmessage?.({ data });
        }
        crash(message) {
          this.onerror?.({ message });
        }
        terminate() {
          this.terminated = true;
        }
      };
      const engine = new Engine({ depth: 2 });
      t.after(() => {
        engine.dispose();
        for (const [name, descriptor] of [
          ['Worker', previousWorker],
          ['location', previousLocation],
        ]) {
          if (descriptor) Object.defineProperty(globalThis, name, descriptor);
          else delete globalThis[name];
        }
      });
      await engine.prewarm();
      const old = workers[0];
      const failed = assert.rejects(engine.bestMove(START_FEN), /first worker failed/);
      await turn();
      assert.ok(old.messages.some((message) => message.command?.startsWith('go ')));
      old.crash('first worker failed');
      await failed;
      assert.equal(old.terminated, true);

      let outcome = 'pending';
      const pending = engine.bestMove(START_FEN).then(
        (move) => {
          outcome = 'resolved';
          return move;
        },
        (error) => {
          outcome = 'rejected';
          return error;
        },
      );
      await turn();
      const current = workers[1];
      const go = current.messages.find((message) => message.command?.startsWith('go '));
      assert.ok(go, 'replacement is searching');
      if (event === 'error') old.crash('late old worker error');
      else {
        old.emit({ type: 'line', stream: 'stdout', line: 'info depth 99 score cp -999 pv d2d4' });
        old.emit({ type: 'line', stream: 'stdout', line: 'bestmove d2d4' });
        // Complete only the current command, not its stdout search result.
        current.emit({ type: 'commandDone', id: go.id });
      }
      await turn();
      assert.equal(outcome, 'pending', 'retired worker settled replacement work');
      assert.equal(current.terminated, false, 'retired worker killed the replacement');
      current.emit({ type: 'line', stream: 'stdout', line: 'bestmove e2e4' });
      current.emit({ type: 'commandDone', id: go.id });
      assert.equal(await pending, 'e2e4');
    });
  }
}
