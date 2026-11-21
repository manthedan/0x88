import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { ARENA_BIG_NET_MEMORY_REASON, arenaBigNetPairingBlocked, arenaBigNetSeatBlocked } from '../src/lc0/arenaMemoryGuard.ts';
import { arenaCpuOnlyPool } from '../src/lc0/arenaScheduleRunner.ts';
import { BigNetSessionPool } from '../src/lc0/bigNetSessionPool.ts';
import { BIG_NETS } from '../src/lc0/bt4Engine.ts';
import { DisposableVariantPool } from '../src/lc0/disposableVariantPool.ts';
import { PHONE_LARGE_NET_REASON, phoneSafeEngineVariant } from '../src/lc0/phoneLargeNetPolicy.ts';

const big = (variant) => ({ family: 'lc0', variant, strength: 100 });
const cpu = { family: 'sf', variant: 'lite', strength: 8 };
const small = big('small');

for (const a of Object.keys(BIG_NETS)) {
  for (const b of Object.keys(BIG_NETS)) {
    test(`constrained Arena blocks ${a} + ${b}, including mirror games`, () => {
      for (const mode of ['match', 'round-robin', 'gauntlet']) {
        assert.equal(arenaBigNetPairingBlocked(mode, [big(a), big(b), cpu], true), true);
        assert.equal(arenaBigNetPairingBlocked(mode, [big(a), big(b), cpu], false), false);
        assert.equal(arenaBigNetSeatBlocked(mode, [big(a), cpu], 1, true), true);
      }
    });
  }
}

test('pairings, not roster counts, determine the block', () => {
  const seats = [cpu, big('bt4'), big('t3'), big('lqo')];
  assert.equal(arenaBigNetPairingBlocked('round-robin', seats, true), true);
  assert.equal(arenaBigNetPairingBlocked('gauntlet', seats, true), false);
  assert.equal(arenaBigNetSeatBlocked('gauntlet', seats, 0, true), true);
  for (const index of [1, 2, 3]) assert.equal(arenaBigNetSeatBlocked('gauntlet', seats, index, true), false);
  assert.equal(arenaBigNetPairingBlocked('match', seats, true), false);
  assert.equal(arenaBigNetSeatBlocked('match', seats, 2, true), false, 'inactive seats cannot affect match options');
  assert.equal(arenaBigNetPairingBlocked('round-robin', [small, cpu, big('bt4')], true), false);
  assert.equal(arenaBigNetPairingBlocked('match', [cpu, { ...cpu, variant: 'bt4' }], true), false);
  assert.equal(arenaBigNetPairingBlocked('match', [], true), false);
});

const source = readFileSync(new URL('../src/lc0/arenaBrowser.ts', import.meta.url), 'utf8');
function extract(start, end, wrap = (text) => text) {
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from);
  return stripTypeScriptTypes(wrap(source.slice(from, to)));
}

test('constrained Arena reuses the shared workers even with a one-thread CPU pool', () => {
  let constrained = true;
  const isolatedCpuPool = new Function(
    'activeSeatRows',
    'arenaCpuOnlyPool',
    'stockfishThreadsPlanned',
    'isMemoryConstrainedDevice',
    `${extract('function isolatedCpuPool(', 'function selectedArenaConcurrency(')}\nreturn isolatedCpuPool;`,
  )(
    () => [cpu, cpu],
    arenaCpuOnlyPool,
    () => 1,
    () => constrained,
  );
  assert.equal(isolatedCpuPool(), false);
  constrained = false;
  assert.equal(isolatedCpuPool(), true, 'desktop can still use private workers');
});

test('Arena start entry blocks restored/URL/programmatic pairings before acquiring assets or searchers', async () => {
  const nodes = new Map();
  const el = (id) => {
    if (!nodes.has(id)) nodes.set(id, { textContent: '' });
    return nodes.get(id);
  };
  for (const mode of ['match', 'round-robin', 'gauntlet']) {
    for (const pairBlocked of [true, false]) {
      let cleared = false;
      const deps = {
        running: false,
        startPending: false,
        abort: null,
        mountAbort: new AbortController(),
        abortArenaPickerConsent() {},
        setArenaStartDisabled() {},
        syncArenaPhase() {},
        checkpointWrites: Promise.resolve(),
        isStaleMount: () => false,
        checkpointUnreadable: false,
        checkpointCompatibility: async () => 'test',
        normalizeArenaControls: () => ({}),
        ARENA_CONTROL_IDS: [],
        el,
        syncSeatRowsFromDom() {},
        arenaBigNetPairingBlocked: (mode, seats) => arenaBigNetPairingBlocked(mode, seats, true),
        arenaTournamentMode: () => mode,
        seatRows: [big('bt4'), pairBlocked ? big('t3') : cpu],
        ARENA_BIG_NET_MEMORY_REASON,
        PHONE_LARGE_NET_REASON,
        phoneSafeEngineVariant: (family, variant) => phoneSafeEngineVariant(family, variant, true, false),
        renderSeatSelectors() {},
        showPhoneLargeNetFallback() {},
        clearStartPending() {
          cleared = true;
        },
        ensureSelectedRecklessAssetsAvailable() {
          assert.fail('blocked setup reached asset checks');
        },
      };
      const start = new Function(
        'deps',
        `let { ${Object.keys(deps).join(', ')} } = deps;\n${extract('async function startMatchUnchecked(', '\nasync function startMatch(')}\nreturn startMatchUnchecked;`,
      )(deps);
      await start();
      assert.equal(cleared, true);
      assert.equal(el('message').textContent, pairBlocked ? ARENA_BIG_NET_MEMORY_REASON : PHONE_LARGE_NET_REASON);
    }
  }
});

test('safe gauntlet warmup uses only each game on phones; desktop still warms the roster once', async () => {
  for (const constrained of [true, false]) {
    const calls = [];
    const pool = new BigNetSessionPool(
      (config) => ({ config, dispose() {}, isInitializedFor: () => true }),
      undefined,
      undefined,
      () => constrained,
    );
    pool.acquire('lqo'); // An earlier page/game may have retained a different net.
    const bigNetFor = new Function(
      'BIG_NETS',
      'acquireBigNetSearcherAsync',
      'isMemoryConstrainedDevice',
      'arenaCacheEntries',
      `${extract('async function bigNetFor(', '\nlet runtimeIsolation')}\nreturn bigNetFor;`,
    )(
      BIG_NETS,
      (key, exclusive) => pool.acquireAsync(key, exclusive),
      () => false,
      () => 0,
    );
    const deps = {
      seatIds: ['sf:lite:8', 'lc0:bt4:100', 'lc0:t3:100'],
      matchAbort: new AbortController(),
      minimumWarmupVisibleMs: 400,
      isMemoryConstrainedDevice: () => constrained,
      isStaleMount: () => false,
      mountSignal: new AbortController().signal,
      pendingIndices: [0, 1],
      schedule: [
        { whiteId: '0', blackId: '1' },
        { whiteId: '2', blackId: '0' },
      ],
      byPid: new Map([
        ['0', { index: 0, engine: { id: 'sf:lite:8' } }],
        ['1', { index: 1, engine: { id: 'lc0:bt4:100' } }],
        ['2', { index: 2, engine: { id: 'lc0:t3:100' } }],
      ]),
      seatRows: [cpu, big('bt4'), big('t3')],
      sharesArenaCpuWorker: () => false,
      disposeStockfish() {},
      disposeUnusedUciEngines() {},
      async warmUpSelectedEngines(ids) {
        calls.push(ids);
        for (const id of ids) {
          if (!id.startsWith('lc0:')) continue;
          await bigNetFor(id.split(':')[1], new AbortController().signal);
          if (constrained) assert.equal(Object.keys(BIG_NETS).filter((key) => pool.has(key)).length, 1);
        }
      },
    };
    const warmup = extract(
      '      const constrained = isMemoryConstrainedDevice();\n      if (!constrained) {',
      '        resetLc0SearchTrees(seatIds);\n        for (const key of ARENA_BIG_NET_KEYS)',
      (text) => `async function run() { let enginesReady = false;\n${text}\n} }`,
    );
    const run = new Function('deps', `const { ${Object.keys(deps).join(', ')} } = deps;\n${warmup}\nreturn run;`)(deps);
    await run();
    assert.deepEqual(
      calls,
      constrained
        ? [
            ['sf:lite:8', 'lc0:bt4:100'],
            ['lc0:t3:100', 'sf:lite:8'],
          ]
        : [deps.seatIds],
    );
    pool.disposeAllNow();
  }
});

test('constrained Arena retires CPUs outside the next pairing and disposes the final pairing on exit', async () => {
  const resident = new Set();
  const families = ['stormphrax', 'plentychess', 'berserk', 'reckless', 'viridithas'];
  const pools = Object.fromEntries(
    families.map((family) => [
      family,
      new DisposableVariantPool(
        (key) => key,
        () => {
          resident.add(family);
          return { dispose: () => resident.delete(family) };
        },
      ),
    ]),
  );
  const seatIds = families.slice(0, 3);
  const schedule = [
    { whiteId: 'stormphrax', blackId: 'plentychess' },
    { whiteId: 'plentychess', blackId: 'berserk' },
  ];
  const matchAbort = new AbortController();
  let warmups = 0;
  const deps = {
    matchAbort,
    abort: matchAbort,
    checkpointWrites: Promise.resolve(),
    isMemoryConstrainedDevice: () => true,
    isStaleMount: () => false,
    mountSignal: matchAbort.signal,
    schedule,
    seatIds,
    minimumWarmupVisibleMs: 0,
    pendingIndices: [0, 1],
    byPid: new Map(seatIds.map((id, index) => [id, { index, engine: { id } }])),
    seatRows: seatIds.map((id) => ({ family: id, variant: id })),
    sharesArenaCpuWorker: () => false,
    isolatedCpuEngines: new Set(),
    isolatedCpuSeats: new Map(),
    async warmUpSelectedEngines(ids) {
      assert.deepEqual([...resident], warmups++ === 0 ? [] : ['plentychess'], 'evict before warming the next pair');
      for (const id of ids) pools[id].getOrCreate(id);
    },
  };
  for (const family of families) {
    const prefix = family === 'plentychess' ? 'plentyChess' : family;
    deps[`${prefix}Engines`] = pools[family];
    deps[`${prefix}VariantForKey`] = (key) => key;
  }
  const warmup = extract(
    '      const constrained = isMemoryConstrainedDevice();\n      if (!constrained) {',
    '        resetLc0SearchTrees(seatIds);\n        for (const key of ARENA_BIG_NET_KEYS)',
    (text) => `async function warmup() { let enginesReady = false;\n${text}\n} }`,
  );
  const match = source.slice(source.indexOf('  let enginesReady = false;'), source.indexOf('\nasync function startMatch('));
  const cleanup = match.slice(match.lastIndexOf('  } finally {'), match.indexOf('    const ownsMatchState = abort === matchAbort;'));
  const run = new Function(
    'deps',
    `const { ${Object.keys(deps).join(', ')} } = deps;
    let stockfishLite = null, stockfishFull = null;
    ${extract('function disposeStockfish(', 'function availableRecklessVariants(')}
    ${extract('function disposeUnusedUciEngines(', 'function reconcileArenaNeuralSessions(')}
    ${extract('function disposeArenaCpuEngines(', 'function disposeRuntimeResources(')}
    ${warmup}
    return async function() { try { await warmup(); ${cleanup} } };`,
  )(deps);
  await run();
  assert.equal(warmups, 2);
  assert.deepEqual([...resident], []);
});

test('constrained Arena uses two private seats only for pairings that would share a CPU worker', async () => {
  const seatRows = [
    { ...cpu, strength: 6 },
    { ...cpu, strength: 14 },
    { ...cpu, variant: 'full' },
  ];
  const seatIds = ['lite-6', 'lite-14', 'full'];
  const schedule = [
    { whiteId: '0', blackId: '1' },
    { whiteId: '0', blackId: '2' },
    { whiteId: '1', blackId: '0' },
  ];
  const privateGames = [];
  const sharedWarmups = [];
  const deps = {
    seatRows,
    seatIds,
    schedule,
    byPid: new Map(seatIds.map((id, index) => [String(index), { index, engine: { id } }])),
    pendingIndices: [0, 1, 2],
    minimumWarmupVisibleMs: 0,
    matchAbort: new AbortController(),
    mountSignal: new AbortController().signal,
    isMemoryConstrainedDevice: () => true,
    isStaleMount: () => false,
    el: () => ({}),
    completeGame() {},
    async runIsolatedCpuTournament(_schedule, pendingIndices) {
      privateGames.push(...pendingIndices);
    },
    async warmUpSelectedEngines(ids) {
      sharedWarmups.push(ids);
    },
    disposeStockfish() {},
    disposeUnusedUciEngines() {},
  };
  const warmup = extract(
    '      const constrained = isMemoryConstrainedDevice();\n      if (!constrained) {',
    '        resetLc0SearchTrees(seatIds);\n        for (const key of ARENA_BIG_NET_KEYS)',
    (text) => `async function run() { let enginesReady = false;\n${text}\n} }`,
  );
  const run = new Function(
    'deps',
    `const { ${Object.keys(deps).join(', ')} } = deps;\n${extract('function arenaCpuWorkerKey(', 'function createIsolatedCpuEngine(')}\n${warmup}\nreturn run;`,
  )(deps);
  await run();
  assert.deepEqual(privateGames, [0, 2], 'both color assignments use the existing independent seat workers');
  assert.deepEqual(sharedWarmups, [['lite-6', 'full']], 'different variants still reuse the shared pool');
});

test('Arena diagnostic rendering never acquires or replaces a resident net', () => {
  const deps = {
    diagnosticEngineIds: () => ['lc0:bt4:100', 'lc0:t3:100'],
    isolatedCpuSeats: new Map(),
    rowForEngineId: (id) => big(id.split(':')[1]),
    engines: new Map(),
    isLc0BigNetVariant: (variant) => ['bt4', 't3'].includes(variant),
    BIG_NETS,
    peekBigNetSearcher: () => null,
    budgetText: () => '100ms/move',
    arenaCacheEntries: () => 0,
    bt4Telemetry: new Map(),
  };
  const diagnostics = new Function(
    'deps',
    `const { ${Object.keys(deps).join(', ')} } = deps;\n${extract('function engineRuntimeDiagnosticsText()', '\nfunction lc0TreeTelemetrySummary(')}\n${extract('function engineSearchDiagnosticsText(', '\nfunction diagnosticsHtml(')}\nreturn [engineRuntimeDiagnosticsText(), engineSearchDiagnosticsText()];`,
  )(deps);
  assert.match(diagnostics[0], /lazy worker/);
  assert.match(diagnostics[1], /search waiting/);
});
