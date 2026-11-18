import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { PendingCpuNewGames } from '../src/lc0/cpuEngineResidency.ts';
import { createPlayEngines } from '../src/lc0/play/playEngines.ts';
import { RecklessEngine } from '../src/lc0/recklessEngine.ts';

function extracted(path, start, end) {
  const source = readFileSync(new URL(path, import.meta.url), 'utf8');
  const from = source.indexOf(start);
  const to = source.indexOf(end, from);
  assert.ok(from >= 0 && to > from, `${start} not found in ${path}`);
  return stripTypeScriptTypes(source.slice(from, to));
}

test('Arena resets every participating CPU engine per game, Stockfish and Reckless included', () => {
  const sfLite = { name: 'sf-lite' };
  const sfFull = { name: 'sf-full' };
  const reckless = new Map();
  const getRecklessFor = (variant) => {
    if (!reckless.has(variant)) reckless.set(variant, { name: `reckless-${variant}` });
    return reckless.get(variant);
  };
  const single = (name) => () => ({ name });
  const dependencies = {
    stockfishEngineFor: (kind) => (kind === 'full' ? sfFull : sfLite),
    recklessVariantForKey: (key) => ({ key }),
    recklessVariantAssetStatus: (variant) => (variant.key === 'missing' ? 'missing' : 'available'),
    getRecklessFor,
    getViridithasFor: single('viridithas'),
    getBerserkFor: single('berserk'),
    getPlentyChessFor: single('plentychess'),
    getStormphraxFor: single('stormphrax'),
  };
  const select = new Function(
    ...Object.keys(dependencies),
    `${extracted('../src/lc0/arenaBrowser.ts', 'function cpuEnginesToResetForGame(', 'function recklessMissingAssetMessage(')}\nreturn cpuEnginesToResetForGame;`,
  )(...Object.values(dependencies));
  const names = (rows) => [...select(rows)].map((engine) => engine.name);
  assert.deepEqual(
    names([
      { family: 'sf', variant: 'lite' },
      { family: 'reckless', variant: 'default' },
    ]),
    ['sf-lite', 'reckless-default'],
  );
  assert.deepEqual(
    names([
      { family: 'sf', variant: 'full' },
      { family: 'sf', variant: 'full' },
    ]),
    ['sf-full'],
    'mirror seats share a worker',
  );
  assert.deepEqual(names([{ family: 'reckless', variant: 'missing' }, { family: 'lc0', variant: 'small' }, undefined]), []);
  assert.deepEqual(
    names([
      { family: 'reckless', variant: 'custom' },
      { family: 'berserk', variant: 'x' },
    ]),
    ['reckless-custom', 'berserk'],
  );
});

test('Play marks resident engines at every game start and resets each before its next move', async () => {
  const log = [];
  const engines = createPlayEngines({
    loadCentipawn: () => assert.fail('unused'),
    createCpuEngine: (option) => ({
      setOptions() {},
      bestMove: async () => log.push(`${option.id}:move`) && null,
      newGame: async () => log.push(`${option.id}:reset`),
    }),
  });
  const engineUi = { current: () => () => false, mounted: () => true, note() {}, clearNote() {}, progress() {}, hideProgress() {} };
  const sf = { id: 'sf-lite', label: 'Stockfish', family: 'sf', variant: 'lite' };
  const berserk = { id: 'berserk', label: 'Berserk', family: 'berserk', variant: 'berserk' };
  // Play's production game-start and engine-move paths, linked to the real engine host.
  let option = sf;
  const play = new Function(
    'deps',
    `const { ${['engines', 'ctxSelectedEngine', 'ctxStrengthFor', 'SF_LEVELS', 'uciGameHistory', 'replayPlayPosition', 'startFenFor', 'parseFen']} } = deps;
    ${extracted('../src/lc0/playBrowser.ts', 'async function ctxRequestEngineMove(', 'async function ctxVerifySelectedEngineReady(')}
    ${extracted('../src/lc0/playBrowser.ts', 'function ctxResetStartPosition(', '/** Restored preferences')}
    return { move: ctxRequestEngineMove, resetStart: ctxResetStartPosition };`,
  )({
    engines,
    ctxSelectedEngine: () => option,
    ctxStrengthFor: () => 1,
    SF_LEVELS: [{ skill: 0 }],
    uciGameHistory: () => ({ fen: 'fen', moves: [] }),
    replayPlayPosition: () => ({}),
    startFenFor: () => 'fen',
    parseFen: (fen) => ({ fen }),
  });
  const ctx = { opponent: { active: { level: 0 } }, positions: [], practicePosition: null, humanColor: 'w', engineUi };
  const move = (next) => {
    option = next;
    return play.move(ctx, new AbortController().signal);
  };
  await move(sf);
  await move(berserk);
  assert.deepEqual(log, ['sf-lite:move', 'berserk:move'], 'a fresh engine is not reset');
  log.length = 0;
  play.resetStart(ctx);
  await engines.cpuEngineFor(sf, engineUi);
  assert.deepEqual(log, [], 'the readiness probe leaves the reset to the first move');
  await move(sf);
  await move(sf);
  await move(berserk);
  assert.deepEqual(
    log,
    ['sf-lite:reset', 'sf-lite:move', 'sf-lite:move', 'berserk:reset', 'berserk:move'],
    'each resident engine resets once, before its next move',
  );
});

test('PendingCpuNewGames sends newGame once and retries a reset that did not finish', async () => {
  const pending = new PendingCpuNewGames();
  let calls = 0;
  let fail = true;
  const engine = {
    newGame: async () => {
      calls += 1;
      if (fail) throw new DOMException('aborted', 'AbortError');
    },
  };
  await pending.resetIfPending('sf', engine);
  assert.equal(calls, 0, 'a fresh engine is not reset');
  pending.markAll(['sf', 'old']);
  await assert.rejects(pending.resetIfPending('sf', engine), { name: 'AbortError' });
  fail = false;
  await pending.resetIfPending('sf', engine);
  await pending.resetIfPending('sf', engine);
  assert.equal(calls, 2);
  pending.forget('old');
  assert.equal(pending.has('old'), false);
});

test('Reckless newGame sends ucinewgame to a persistent worker and skips one-shot runs', async () => {
  const previous = globalThis.crossOriginIsolated;
  try {
    for (const [isolated, expected] of [
      [true, [['ucinewgame', 'isready']]],
      [false, []],
    ]) {
      globalThis.crossOriginIsolated = isolated;
      const engine = new RecklessEngine();
      const sent = [];
      engine.runCommandsUntil = async (commands) => {
        sent.push(commands);
        return { exitCode: 0, stdout: ['readyok'], stderr: [] };
      };
      await engine.newGame();
      assert.deepEqual(sent, expected, `crossOriginIsolated=${isolated}`);
      engine.dispose();
    }
  } finally {
    if (previous === undefined) delete globalThis.crossOriginIsolated;
    else globalThis.crossOriginIsolated = previous;
  }
});
