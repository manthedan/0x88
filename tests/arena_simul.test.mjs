import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { boardToFen } from '../src/chess/board.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { ArenaSimulBroker } from '../src/lc0/arenaSimulBroker.ts';
import { simulGeneration, simulHelps4 } from '../src/lc0/arenaSimulGate.ts';
import { runSimulTrial, SIMUL_OPENINGS, SIMUL_SEARCH_OPTIONS } from '../src/lc0/arenaSimulRunner.ts';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const value = (input) => ({ fen: String(input), wdl: [0.4, 0.2, 0.4], q: 0, mlh: 0, legalPriors: [] });
const signal = () => new AbortController();

function fakeProvider() {
  const calls = [];
  let active = 0;
  return {
    calls,
    evaluate: value,
    async evaluateBatch(inputs) {
      assert.equal(++active, 1, 'one session.run at a time');
      calls.push(inputs);
      await tick();
      active--;
      return inputs.map(value);
    },
  };
}

test('simul coalesces singleton leaves, routes rows in order, and serializes physical calls', async () => {
  const inner = fakeProvider();
  const broker = new ArenaSimulBroker(inner, 4);
  const clients = Array.from({ length: 8 }, (_, id) => broker.client(String(id), signal().signal));
  const results = await Promise.all(clients.map((client, id) => client.evaluate(String(id))));
  assert.deepEqual(inner.calls, [
    ['0', '1', '2', '3'],
    ['4', '5', '6', '7'],
  ]);
  assert.deepEqual(
    results.map((result) => result.fen),
    ['0', '1', '2', '3', '4', '5', '6', '7'],
  );
  assert.equal(broker.metrics().maxResident, 8);
  assert.deepEqual(broker.metrics().batchSizes, { 4: 2 });
  await broker.close();
});

test('simul bounds queue including in-flight slots and forbids multiple leaves per game', async () => {
  const broker = new ArenaSimulBroker(fakeProvider(), 2, 2, 5);
  const a = broker.client('a', signal().signal);
  const pending = a.evaluate('a');
  await assert.rejects(a.evaluate('again'), /one outstanding/);
  await assert.rejects(a.evaluateBatch(['a', 'b']), /search batch 1/);
  const b = broker.client('b', signal().signal).evaluate('b');
  await assert.rejects(broker.client('c', signal().signal).evaluate('c'), /capacity/);
  await Promise.all([pending, b]);
  await broker.close();
});

test('simul cancels queued and in-flight rows independently; stale results cannot resolve a new generation', async () => {
  let finish;
  const inner = {
    evaluate: value,
    evaluateBatch: (inputs) =>
      new Promise((resolve) => {
        finish = () => resolve(inputs.map(value));
      }),
  };
  const broker = new ArenaSimulBroker(inner, 2, 4);
  const a = signal();
  const b = signal();
  const c = signal();
  const pa = broker.client('a-v1', a.signal).evaluate('a');
  const rejectedA = assert.rejects(pa, { name: 'AbortError' });
  const pb = broker.client('b', b.signal).evaluate('b');
  await tick();
  const pc = broker.client('c', c.signal).evaluate('c');
  const rejectedC = assert.rejects(pc, { name: 'AbortError' });
  c.abort();
  a.abort();
  await Promise.all([rejectedA, rejectedC]);
  finish();
  assert.equal((await pb).fen, 'b');
  await assert.rejects(broker.client('a-v1', a.signal).evaluate('late'), { name: 'AbortError' });
  await tick();
  const next = broker.client('a-v2', signal().signal).evaluate('new');
  await new Promise((resolve) => setTimeout(resolve, 10));
  finish();
  assert.equal((await next).fen, 'new');
  assert.equal(broker.metrics().cancelled, 2);
  await broker.close();
});

test('simul has a finite wait for a partial batch and never starves a slower game', async () => {
  const inner = fakeProvider();
  const broker = new ArenaSimulBroker(inner, 2, 8, 1);
  const a = broker.client('fast', signal().signal);
  const b = broker.client('slow', signal().signal);
  const fast = (async () => {
    for (let i = 0; i < 8; i++) await a.evaluate(`a${i}`);
  })();
  const slow = b.evaluate('b');
  assert.equal((await slow).fen, 'b');
  await fast;
  assert.ok(inner.calls[0].includes('b'), 'FIFO prevents a fast game overtaking an admitted slow game');
  assert.equal(broker.metrics().perGame.fast, 8);
  assert.equal(broker.metrics().perGame.slow, 1);
  await broker.close();
});

test('simul rejects all rows on failure/cardinality errors and recovers its queue', async () => {
  let failed = true;
  const broker = new ArenaSimulBroker({ evaluate: value, evaluateBatch: (inputs) => (failed ? [] : inputs.map(value)) }, 2);
  const clients = ['a', 'b'].map((id) => broker.client(id, signal().signal));
  const results = await Promise.allSettled(clients.map((client) => client.evaluate('x')));
  assert.ok(results.every((result) => result.status === 'rejected' && /cardinality/.test(result.reason.message)));
  failed = false;
  assert.equal((await clients[0].evaluate('recovered')).fen, 'recovered');
  await broker.close();
  await assert.rejects(clients[1].evaluate('closed'), { name: 'AbortError' });
});

function deterministicProvider() {
  return {
    evaluate(input) {
      const board = input.prepared?.board ?? input.positions.at(-1);
      const moves = legalMoves(board);
      return {
        fen: boardToFen(board),
        wdl: [0.4, 0.2, 0.4],
        q: 0,
        mlh: 0,
        legalPriors: moves.map((move, index) => ({ uci: moveToUci(move), actionId: moveToActionId(move), index, logit: 0, prior: 1 / moves.length })),
      };
    },
    async evaluateBatch(inputs) {
      await tick();
      return inputs.map((input) => this.evaluate(input));
    },
  };
}

test('simul preserves each game history, logical search order, exact visits and moves when numerics are equivalent', async () => {
  assert.equal(new Set(SIMUL_OPENINGS.map((opening) => opening.fen)).size, 4);
  assert.equal(SIMUL_SEARCH_OPTIONS.batchSize, 1);
  const serial = await runSimulTrial(deterministicProvider(), { concurrency: 1, visits: 8, maxPlies: 5 });
  for (const concurrency of [2, 4, 8]) {
    const simul = await runSimulTrial(deterministicProvider(), { concurrency, visits: 8, maxPlies: 5 });
    assert.deepEqual(
      simul.games.map(({ moveMs, ...game }) => game),
      serial.games.map(({ moveMs, ...game }) => game),
    );
    assert.equal(simul.completed, 8);
    assert.ok(simul.games.every((game) => game.visits.every((visits) => visits === 8)));
    assert.ok(simul.broker.maxResident <= concurrency);
    assert.equal(Object.keys(simul.broker.perGame).length, 8);
  }
});

test('simul per-game cancellation does not score that game or stop siblings; run cancellation rejects', async () => {
  const controllers = new Map();
  let cancelled = false;
  const trial = await runSimulTrial(deterministicProvider(), {
    concurrency: 4,
    visits: 4,
    maxPlies: 3,
    controllers,
    onProgress(progress) {
      if (!cancelled && progress.game === 0) {
        cancelled = true;
        controllers.get(0)?.abort();
      }
    },
  });
  assert.deepEqual(trial.cancelled, [0]);
  assert.equal(trial.completed, 7);
  assert.equal(trial.games.length, 7);
  assert.equal(controllers.size, 0);
  const controller = signal();
  const pending = runSimulTrial(deterministicProvider(), { concurrency: 4, visits: 4, maxPlies: 3, signal: controller.signal, controllers });
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(controllers.size, 0);
});

test('simul trial failure prevents admission of fresh games while siblings cancel', async () => {
  const controllers = new Map();
  const admitted = new Set();
  await assert.rejects(
    runSimulTrial(deterministicProvider(), {
      concurrency: 4,
      visits: 4,
      maxPlies: 4,
      controllers,
      onProgress(progress) {
        for (const id of controllers.keys()) admitted.add(id);
        if (progress.game === 0) throw new Error('Injected move publication failure');
      },
    }),
    /Injected move publication failure/,
  );
  assert.ok([...admitted].every((id) => id < 4));
  assert.equal(controllers.size, 0);
});

test('simul generations cannot collide across gate restarts and N=8 eligibility matches the workload', () => {
  assert.notEqual(simulGeneration('run-a', 0), simulGeneration('run-b', 0));
  assert.notEqual(simulGeneration('run-a', 0), simulGeneration('run-a', 1));
  const result = {
    n: 4,
    visits: 8,
    maxPlies: 160,
    pairs: 3,
    rows: Array.from({ length: 6 }, (_, i) => ({
      control: false,
      concurrency: i % 2 ? 4 : 1,
      gamesPerHour: i % 2 ? 140 : 100,
      completed: 8,
    })),
  };
  assert.equal(simulHelps4([result], 8, 160), true);
  assert.equal(simulHelps4([result], 16, 160), false);
  assert.equal(simulHelps4([result], 8, 240), false);
  assert.equal(simulHelps4([{ ...result, pairs: 1 }], 8, 160), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.slice(0, 4) }], 8, 160), false);
});

test('simul stays lab-only, never autostarts, and uses distinct physical exports without a global RNG override', () => {
  const page = readFileSync(new URL('../src/routes/lab/arena-simul/+page.svelte', import.meta.url), 'utf8');
  const worker = readFileSync(new URL('../src/lc0/arenaSimulWorker.ts', import.meta.url), 'utf8');
  assert.doesNotMatch(page, /params\.get|autostart.*===/);
  assert.doesNotMatch(worker, /Math\.random\s*=/);
  assert.match(worker, /ortPhysicalBatchSize/);
  assert.match(worker, /for \(const input of inputs\) \{\s*signal\.throwIfAborted\(\)/);
  assert.match(worker, /signal\.throwIfAborted\(\);\s*const warmup/);
  assert.match(worker, /message\.generation === generation/);
  for (const name of ['arenaBrowser.ts', 'searchWorkerCore.ts', 'arenaScheduleRunner.ts']) {
    assert.doesNotMatch(readFileSync(new URL(`../src/lc0/${name}`, import.meta.url), 'utf8'), /arenaSimul/);
  }
});
