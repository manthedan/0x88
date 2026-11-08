import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { analyzeSimulRun } from '../scripts/analyze_arena_simul.mjs';
import { boardToFen } from '../src/chess/board.ts';
import { moveToActionId, moveToUci } from '../src/chess/moveCodec.ts';
import { legalMoves } from '../src/chess/movegen.ts';
import { ArenaSimulBroker } from '../src/lc0/arenaSimulBroker.ts';
import {
  measureSimulMemory,
  simulCancelGame,
  simulEquivalentGames,
  simulGeneration,
  simulHelps4,
  simulMemoryPending,
  subscribeSimulMemory,
} from '../src/lc0/arenaSimulGate.ts';
import { simulProbe } from '../src/lc0/arenaSimulProbe.ts';
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
  const games = Array.from({ length: 8 }, (_, id) => ({ id, moves: ['a2a4'], visits: [8], result: '1/2-1/2', reason: 'max plies' }));
  const result = {
    n: 4,
    visits: 8,
    maxPlies: 160,
    pairs: 3,
    rows: Array.from({ length: 7 }, (_, i) => ({
      control: i === 6,
      physical: i % 2 || i === 6 ? 4 : 1,
      modelUrl: `batch${i % 2 || i === 6 ? 4 : 1}.onnx`,
      concurrency: i % 2 ? 4 : 1,
      gamesPerHour: i % 2 ? 140 : 100,
      evalsPerSecond: i % 2 ? 140 : 100,
      completed: 8,
      games,
    })),
  };
  assert.equal(simulHelps4([result], 8, 160), true);
  assert.equal(simulHelps4([result], 16, 160), false);
  assert.equal(simulHelps4([result], 8, 240), false);
  assert.equal(simulHelps4([{ ...result, pairs: 1 }], 8, 160), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.slice(0, 4) }], 8, 160), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.slice(0, 6) }], 8, 160), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.map((row) => ({ ...row, evalsPerSecond: 100 })) }], 8, 160), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.map((row) => (row.control ? { ...row, physical: 1 } : row)) }], 8, 160), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.map((row) => (row.control ? { ...row, modelUrl: 'different.onnx' } : row)) }], 8, 160), false);
  const changed = games.map((game) => ({ ...game, moves: ['b2b4'] }));
  assert.equal(simulEquivalentGames(games, changed), false);
  assert.equal(simulHelps4([{ ...result, rows: result.rows.map((row) => (row.control ? { ...row, games: changed } : row)) }], 8, 160), false);
});

test('simul cancellation button selects a live controller, not stale last-progress metadata', () => {
  const finished = signal();
  const live = signal();
  const controllers = new Map([
    [0, finished],
    [4, live],
  ]);
  controllers.delete(0);
  simulCancelGame(controllers);
  assert.equal(live.signal.aborted, true);
  assert.equal(finished.signal.aborted, false);
  simulCancelGame(new Map());
});

test('simul pending memory guard survives route-style unsubscribe and remount', async () => {
  let finish;
  const oldView = [];
  const unmount = subscribeSimulMemory((pending) => oldView.push(pending));
  const measurement = measureSimulMemory(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      }),
  );
  unmount();
  const newView = [];
  const cleanup = subscribeSimulMemory((pending) => newView.push(pending));
  assert.equal(simulMemoryPending(), true);
  finish({ bytes: 123 });
  await measurement;
  assert.equal(simulMemoryPending(), false);
  assert.deepEqual(oldView, [false, true]);
  assert.deepEqual(newView, [true, false]);
  cleanup();
});

test('simul probe checks Stop between actual physical calls, including batch-1 logical batches', async () => {
  const controller = signal();
  const calls = [];
  const provider = {
    evaluate: async (input) => {
      calls.push(input);
      controller.abort();
      return value(input);
    },
    evaluateBatch: () => {
      throw new Error('Serial probes may not hide multiple physical runs');
    },
  };
  await assert.rejects(simulProbe(provider, ['a', 'b', 'c', 'd'], 1, controller.signal), { name: 'AbortError' });
  assert.deepEqual(calls, ['a']);
  const batched = fakeProvider();
  assert.equal((await simulProbe(batched, ['a', 'b', 'c', 'd'], 4, signal().signal)).length, 4);
  assert.deepEqual(batched.calls, [['a', 'b', 'c', 'd']]);
});

test('simul evidence audit replays legal histories and rejects illegal moves or incomplete trials', async () => {
  const serial = await runSimulTrial(deterministicProvider(), { concurrency: 1, visits: 4, maxPlies: 2 });
  const simul = await runSimulTrial(deterministicProvider(), { concurrency: 4, visits: 4, maxPlies: 2 });
  const run = {
    protocol: 'arena-simul-1',
    n: 4,
    visits: 4,
    maxPlies: 2,
    pairs: 3,
    openings: SIMUL_OPENINGS.map(({ name, fen, moves }) => ({ name, fen, moves })),
    rows: Array.from({ length: 7 }, (_, index) => {
      const trial = index % 2 ? simul : serial;
      const physical = index % 2 || index === 6 ? 4 : 1;
      return {
        ...trial,
        index,
        physical,
        modelUrl: `/models/lc0/t1-256x10-distilled-swa-2432500.batch${physical}.f16.onnx`,
        control: index === 6,
        physicalRuns: trial.broker.batches,
      };
    }),
  };
  assert.equal(analyzeSimulRun(run).rows.length, 7);
  const wrongExport = structuredClone(run);
  wrongExport.rows[6].physical = 1;
  assert.throws(() => analyzeSimulRun(wrongExport), /physical batch export/);
  const wrongModel = structuredClone(run);
  wrongModel.rows[6].modelUrl = 'not-the-same-export.onnx';
  assert.throws(() => analyzeSimulRun(wrongModel), /model identity|exact artifact/);
  const illegal = structuredClone(run);
  illegal.rows[0].games[0].moves[0] = 'a1a8';
  assert.throws(() => analyzeSimulRun(illegal), /illegal move/);
  const incomplete = structuredClone(run);
  incomplete.rows[0].completed = 7;
  assert.throws(() => analyzeSimulRun(incomplete));
  const directory = mkdtempSync(join(tmpdir(), 'arena-simul-export-'));
  try {
    const input = join(directory, 'browser-export.json');
    const output = join(directory, 'summary.json');
    writeFileSync(input, JSON.stringify([run, run]));
    const cli = spawnSync(process.execPath, ['--experimental-strip-types', 'scripts/analyze_arena_simul.mjs', output, input], { encoding: 'utf8' });
    assert.equal(cli.status, 0, cli.stderr);
    const audited = JSON.parse(readFileSync(output, 'utf8'));
    assert.equal(audited.runs.length, 2);
    assert.equal(audited.runs[0].rawSha256, audited.runs[1].rawSha256);
    assert.equal(audited.runs[1].rawRunIndex, 1);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('simul stays lab-only, never autostarts, and uses distinct physical exports without a global RNG override', () => {
  const page = readFileSync(new URL('../src/routes/lab/arena-simul/+page.svelte', import.meta.url), 'utf8');
  const worker = readFileSync(new URL('../src/lc0/arenaSimulWorker.ts', import.meta.url), 'utf8');
  assert.doesNotMatch(page, /params\.get|autostart.*===/);
  assert.doesNotMatch(worker, /Math\.random\s*=/);
  assert.match(worker, /ortPhysicalBatchSize/);
  assert.ok(worker.indexOf('if (trial.completed !== 8)') < worker.indexOf("post({ type: 'trial', row })"));
  assert.doesNotMatch(worker, /liveBytes|peakBytes|buffer\.destroy/);
  assert.match(worker, /control \? await sampleMemory\(signal\)/);
  assert.match(page, /busy \|\| memoryPending/);
  assert.match(worker, /for \(const input of inputs\) \{\s*signal\.throwIfAborted\(\)/);
  assert.match(worker, /signal\.throwIfAborted\(\);\s*const warmup/);
  assert.match(worker, /message\.generation === generation/);
  for (const name of ['arenaBrowser.ts', 'searchWorkerCore.ts', 'arenaScheduleRunner.ts']) {
    assert.doesNotMatch(readFileSync(new URL(`../src/lc0/${name}`, import.meta.url), 'utf8'), /arenaSimul/);
  }
});
