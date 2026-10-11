import assert from 'node:assert/strict';
import { test } from 'node:test';
import { playCpuArenaGame } from '../src/lc0/arenaCpuGame.ts';
import { arenaCpuOnlyPool, arenaGameConcurrency, runArenaSchedule } from '../src/lc0/arenaScheduleRunner.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { EngineResourceBroker } from '../src/lc0/resourceBroker.ts';
import { TournamentStandings } from '../src/lc0/tournament.ts';

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test('CPU concurrency is conservative and never includes neural/WASM-fallback seats', () => {
  assert.equal(arenaCpuOnlyPool(['sf', 'berserk', 'reckless']), true);
  for (const family of ['lc0', 'centipawn', 'maia3', 'future-engine']) assert.equal(arenaCpuOnlyPool(['sf', family]), false);
  assert.equal(arenaCpuOnlyPool(['sf']), false);
  for (const deviceMemory of [undefined, 0, 2, 4, NaN]) assert.equal(arenaGameConcurrency({ hardwareConcurrency: 10, deviceMemory }, '3'), 1);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 10, deviceMemory: 16 }), 2);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 10, deviceMemory: 16 }, '3'), 3);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 10, deviceMemory: 8 }, '3'), 2);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 4, deviceMemory: 8 }, '3'), 1);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 6, deviceMemory: 8 }, '3'), 2);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 6, deviceMemory: 8 }), 1);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 10, deviceMemory: 16 }, '1'), 1);
  assert.equal(arenaGameConcurrency({ hardwareConcurrency: 10, deviceMemory: 16 }, 'oops'), 1);
});

test('bounded games complete out of order; indexed export and standings equal K=1', async () => {
  const run = async (concurrency) => {
    let active = 0;
    let peak = 0;
    const records = [];
    const completion = [];
    const standings = new TournamentStandings([
      { id: 'a', name: 'A' },
      { id: 'b', name: 'B' },
      { id: 'c', name: 'C' },
    ]);
    const games = [
      ['a', 'b', '1-0'],
      ['b', 'c', '0-1'],
      ['c', 'a', '1/2-1/2'],
    ];
    await runArenaSchedule({
      count: 3,
      concurrency,
      signal: new AbortController().signal,
      play: async (index) => {
        active++;
        peak = Math.max(peak, active);
        await delay(index === 0 ? 30 : 1);
        active--;
        return games[index];
      },
      complete: (index, game) => {
        completion.push(index);
        records.push({ index, pgn: JSON.stringify(game) });
        records.sort((a, b) => a.index - b.index);
        standings.record(...game);
      },
    });
    return {
      peak,
      completion,
      pgn: records.map((r) => r.pgn).join('\n'),
      table: standings.table().map(({ perOpponent, ...row }) => ({ ...row, perOpponent: [...perOpponent].sort() })),
    };
  };
  const serial = await run(1);
  const parallel = await run(2);
  assert.equal(serial.peak, 1);
  assert.equal(parallel.peak, 2);
  assert.notDeepEqual(serial.completion, parallel.completion);
  assert.equal(serial.pgn, parallel.pgn);
  assert.deepEqual(serial.table, parallel.table);
});

test('stop cancels every per-game signal, waits for cleanup, and starts no more games', async () => {
  const parent = new AbortController();
  const children = [];
  let cleanups = 0;
  let completions = 0;
  const pending = runArenaSchedule({
    count: 10,
    concurrency: 3,
    signal: parent.signal,
    play: async (_, signal) => {
      children.push(signal);
      await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }));
      await delay(5);
      cleanups++;
      return 'cancelled';
    },
    complete: () => completions++,
  });
  parent.abort();
  await pending;
  assert.equal(children.length, 3);
  assert.equal(new Set(children).size, 3);
  assert.ok(children.every((signal) => signal.aborted));
  assert.equal(cleanups, 3);
  assert.equal(completions, 0);
});

test('one failure cancels siblings and propagates only after their cleanup', async () => {
  let cleaned = false;
  const failure = new Error('worker failed');
  await assert.rejects(
    runArenaSchedule({
      count: 5,
      concurrency: 2,
      signal: new AbortController().signal,
      play: async (index, signal) => {
        if (index === 0) {
          await delay(1);
          throw failure;
        }
        await new Promise((resolve) => signal.addEventListener('abort', resolve, { once: true }));
        await delay(5);
        cleaned = true;
        return null;
      },
      complete: () => assert.fail('failed game must not count'),
    }),
    (error) => error === failure,
  );
  assert.equal(cleaned, true);
});

test('pre-aborted tournaments do not create workers', async () => {
  const controller = new AbortController();
  controller.abort();
  await runArenaSchedule({ count: 4, concurrency: 2, signal: controller.signal, play: () => assert.fail(), complete: () => assert.fail() });
});

test('CPU game resets both private engines, has a local board, and releases namespaced leases', async () => {
  const broker = new EngineResourceBroker({ policy: 'bounded', maxConcurrentCpu: 2 });
  const makeSeat = (id) => {
    const commands = [];
    broker.register(id, { resourceClass: 'cpu', maxThreads: 1 });
    return {
      id,
      name: id,
      commands,
      engine: {
        prewarm: async () => commands.push('prewarm'),
        newGame: async () => commands.push('newGame'),
        bestMove: async () => {
          commands.push('move');
          return 'f7g7';
        },
      },
    };
  };
  const first = [makeSeat('game:0:w'), makeSeat('game:0:b')];
  const second = [makeSeat('game:1:w'), makeSeat('game:1:b')];
  const trees = [0, 1].map(() => new GameTree('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'));
  const results = await Promise.all(
    [first, second].map(([white, black], i) => playCpuArenaGame({ tree: trees[i], white, black, broker, signal: new AbortController().signal })),
  );
  assert.equal(results[0].result, '1-0');
  assert.equal(results[1].result, '1-0');
  assert.notEqual(results[0].tree, results[1].tree);
  for (const [white, black] of [first, second]) {
    assert.deepEqual(white.commands, ['prewarm', 'newGame', 'move']);
    assert.deepEqual(black.commands, ['prewarm', 'newGame']);
  }
  assert.deepEqual(broker.snapshot().activeLeases, []);
  assert.deepEqual(broker.snapshot().queuedEngineIds, []);
});

test('CPU move failure does not leak its lease', async () => {
  const broker = new EngineResourceBroker();
  broker.register('white', { resourceClass: 'cpu', maxThreads: 1 });
  const white = {
    id: 'white',
    name: 'white',
    engine: {
      prewarm: async () => {},
      newGame: async () => {},
      bestMove: async () => {
        throw new Error('search error');
      },
    },
  };
  await assert.rejects(playCpuArenaGame({ tree: new GameTree(), white, black: white, broker, signal: new AbortController().signal }), /search error/);
  assert.deepEqual(broker.snapshot().activeLeases, []);
});
