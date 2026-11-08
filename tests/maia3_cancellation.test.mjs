import assert from 'node:assert/strict';
import test from 'node:test';
import { Maia3BrowserEvaluator } from '../src/lc0/maia3.ts';
import { START_FEN } from '../src/chess/board.ts';

class WorkerStub extends EventTarget {
  terminated = false;
  postMessage(message) { this.lastMessage = message; }
  terminate() { this.terminated = true; }
}

test('Maia worker initialization can be aborted without waiting for ready', async () => {
  const worker = new WorkerStub();
  const controller = new AbortController();
  const pending = Maia3BrowserEvaluator.postInit(worker, 'model', 'auto', controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(worker.terminated, true);
});

test('Maia initialization forwards the report CPU-thread reservation to its worker', async () => {
  const worker = new WorkerStub();
  const pending = Maia3BrowserEvaluator.postInit(worker, 'model', 'wasm', undefined, 1);
  assert.equal(worker.lastMessage.wasmThreads, 1);
  worker.dispatchEvent(new MessageEvent('message', { data: { type: 'ready', id: 0, inputNames: [], outputNames: [], backend: 'wasm' } }));
  await pending;
});

test('terminating a report-owned Maia evaluator rejects in-flight and future inference', async () => {
  const worker = new WorkerStub();
  const evaluator = new Maia3BrowserEvaluator(worker, { url: 'model' }, 1500, 1500, { inputNames: [], outputNames: [], backend: 'test' });
  const pending = evaluator.evaluateConditions(START_FEN, [{ selfElo: 1500, oppoElo: 1500 }]);
  evaluator.terminate();
  await assert.rejects(pending, { name: 'AbortError' });
  await assert.rejects(evaluator.evaluateConditions(START_FEN, [{ selfElo: 1500, oppoElo: 1500 }]), { name: 'AbortError' });
  await evaluator.dispose();
  assert.equal(worker.terminated, true);
});
