import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { boardToFen, parseFen, START_FEN } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { acknowledgeInputTransfer, encodePlayHandoff, PLAY_HANDOFF_KEY, takePlayHandoff } from '../src/lc0/analysisHandoff.ts';
import { transitionPendingInput } from '../src/lc0/pendingInput.ts';
import { replayPlayPosition } from '../src/lc0/playPosition.ts';
import { createWorkspaceAutosave } from '../src/lc0/workspaceAutosave.ts';

const source = readFileSync(new URL('../src/lc0/playBrowser.ts', import.meta.url), 'utf8');
function playHarness(run) {
  const values = new Map();
  const faults = { write: true, clear: false };
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem(key, raw) {
      if (faults.write) throw Error('quota');
      values.set(key, raw);
    },
    removeItem(key) {
      if (faults.clear && key === '0x88-play-workspace') throw Error('denied');
      values.delete(key);
    },
  };
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storage });
  const ctx = {
    pendingInput: null,
    practicePosition: null,
    moves: [],
    positions: [parseFen(START_FEN)],
    startFen: START_FEN,
    started: false,
    humanColor: 'w',
    orientation: 'white',
    gameOver: null,
    opponent: { active: { engineId: 'fake', color: 'white' } },
  };
  const position = { startFen: START_FEN, moves: ['d2d4', 'd7d5'] };
  const raw = encodePlayHandoff(position);
  values.set(PLAY_HANDOFF_KEY, raw);
  const apply = source.slice(source.indexOf('function ctxApplyPlayHandoffForMount('), source.indexOf('function ctxPrepareOrStartEngine('));
  const init = source.slice(
    source.indexOf('  // Only a decoded storage position handoff'),
    source.indexOf('  const validEngineIds =', source.indexOf('function ctxInit(')),
  );
  const autosave = source.slice(source.indexOf("  ctx.autosave = createWorkspaceAutosave('play'"), source.indexOf("  trackListener(ctx, window, 'pagehide'"));
  const handler = source.slice(source.indexOf("  trackListener(ctx, el('useStartingPosition')"), source.indexOf("  trackListener(ctx, el('changeOpponent')"));
  const reset = source.slice(source.indexOf('function ctxResetGameStateForMount('), source.indexOf('function ctxInit('));
  const newGame = source.slice(source.indexOf('function ctxNewGame('), source.indexOf('function ctxTakeback('));
  const render =
    source.slice(source.indexOf('function ctxRender(ctx:'), source.indexOf('  ctx.viewPly = normalizeViewPly', source.indexOf('function ctxRender(ctx:'))) +
    '}';
  const lifecycle = source.includes('function ctxUpdatePendingInput(')
    ? source.slice(source.indexOf('function ctxUpdatePendingInput('), source.indexOf('/** Import before any autosaved-game restore;'))
    : '';
  const dependencies = {
    ctx,
    takePlayHandoff,
    acknowledgeInputTransfer,
    transitionPendingInput,
    createWorkspaceAutosave,
    START_FEN,
    parseFen,
    boardToFen,
    moveToUci,
    replayPlayPosition,
  };
  const code = `
    let useStartingPosition;
    const el = () => ({}), trackListener = (ctx, el, type, fn) => {useStartingPosition = fn;};
    const startFenFor = () => START_FEN, ctxSelectedEngine = () => ({}), ctxRefreshEngineOptions = () => {};
    const ctxCancelEngineTurn = () => {}, ctxCurrentPlayPreferences = () => ctx.opponent.active;
    const committedPlayOpponent = active => ({active}), ctxSetOpponent = (ctx, opponent) => {ctx.opponent = opponent;};
    const ctxSetEngineNote = () => {}, ctxDisarmResign = () => {}, ctxDisarmNewGame = () => {}, ctxPrepareOrStartEngine = () => {};
    const ctxPersistPlayPreferences = () => {}, ctxRestoreSelectedRuntimeWarning = () => {}, evictInactiveCpuEngines = () => {};
    const lc0Searcher = null, BIG_NET_KEYS = [];
    ${stripTypeScriptTypes(lifecycle + apply + reset + newGame + render + init + autosave + handler)}
    ctxResetGameStateForMount(ctx); ctxRender(ctx); ctx.autosave.flush();
    return {useStartingPosition: () => useStartingPosition(), newGame: () => ctxNewGame(ctx), render: () => ctxRender(ctx)};
  `;
  try {
    const controller = new Function(...Object.keys(dependencies), code)(...Object.values(dependencies));
    run({ ...controller, ctx, values, faults, raw });
  } finally {
    ctx.autosave?.pause('storage');
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
    else delete globalThis.localStorage;
  }
}

test('Play Use starting position before the first successful save retires the discarded handoff', () => {
  playHarness((h) => {
    assert.equal(h.values.get(PLAY_HANDOFF_KEY), h.raw);
    h.useStartingPosition();
    assert.equal(h.values.has(PLAY_HANDOFF_KEY), false);
    assert.equal(h.ctx.practicePosition, null);
    assert.equal(h.values.has('0x88-play-workspace'), false, 'Reload opens an empty setup');
  });
});

for (const action of ['useStartingPosition', 'newGame']) {
  test(`Play ${action} explicitly retires a handoff even if workspace clear fails`, () => {
    playHarness((h) => {
      h.faults.clear = true;
      h[action]();
      assert.equal(h.values.has(PLAY_HANDOFF_KEY), false);
    });
  });
}

test('Play later autosave acknowledges its captured input and protects a newer transfer', () => {
  for (const newer of [false, true])
    playHarness((h) => {
      if (newer) h.values.set(PLAY_HANDOFF_KEY, 'newer-transfer');
      h.faults.write = false;
      h.render();
      h.ctx.autosave.flush();
      assert.equal(h.values.get(PLAY_HANDOFF_KEY), newer ? 'newer-transfer' : undefined);
      assert.deepEqual(JSON.parse(h.values.get('0x88-play-workspace')).data.moves, ['d2d4', 'd7d5']);
    });
});
