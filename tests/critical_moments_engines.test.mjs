import assert from 'node:assert/strict';
import test from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { createCriticalEngines } from '../src/lc0/criticalMomentsEngines.ts';
import { StockfishEngine } from '../src/lc0/stockfishEngine.ts';

test('a timed-out report search drops the disposed Stockfish instance before the next moment', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const engines = [];
  t.mock.method(StockfishEngine.prototype, 'analyze', async function () {
    engines.push(this);
    if (this.disposed) throw new Error('permanently disposed engine reused');
    if (engines.length === 1) return new Promise(() => {});
    return [{ multipv: 1, depth: 12, scoreCp: 10, pvUci: ['e2e4'] }];
  });
  const broker = { register() {}, unregister() {}, acquire: async () => ({ release() {} }) };
  const report = createCriticalEngines(broker, undefined, new AbortController().signal);
  t.after(report.dispose);
  const lane = report.lanes[0];
  const moment = { fen: START_FEN, history: [START_FEN] };
  const first = lane.evaluate(moment, undefined, () => {});
  const rejected = assert.rejects(first, /timed out/);
  await new Promise((resolve) => setImmediate(resolve));
  t.mock.timers.tick(180_000);
  await rejected;
  const result = await lane.evaluate(moment, undefined, () => {});
  assert.equal(result.line.pvUci[0], 'e2e4');
  assert.equal(engines[0].disposed, true);
  assert.notEqual(engines[0], engines[1]);
});
