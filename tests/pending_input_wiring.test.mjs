import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { describePgnWarning, gameTreeToPgn, parseFirstPgnGame } from '../src/chess/pgn.ts';
import { acknowledgeInputTransfer, takeAnalysisHandoff } from '../src/lc0/analysisHandoff.ts';
import { analysisUrlFen } from '../src/lc0/analysisUrlInput.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { transitionPendingInput } from '../src/lc0/pendingInput.ts';
import * as autosave from '../src/lc0/workspaceAutosave.ts';

const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
function analysisHarness(query, run) {
  const values = new Map();
  const faults = { write: false, remove: false };
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem(key, value) {
      if (faults.write) throw Error('quota');
      values.set(key, value);
    },
    removeItem(key) {
      if (faults.remove) throw Error('denied');
      values.delete(key);
    },
  };
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: storage });
  const url = new URL('https://example.test/app/analysis/?dev&' + query + '#board');
  const elements = new Map();
  const el = (key) => {
    if (!elements.has(key))
      elements.set(key, {
        value: '',
        hidden: true,
        textContent: '',
        focus() {},
        addEventListener(event, handler) {
          this[event] = handler;
        },
      });
    return elements.get(key);
  };
  const dependencies = {
    ...autosave,
    GameTree,
    START_FEN,
    describePgnWarning,
    gameTreeToPgn,
    parseFirstPgnGame,
    takeAnalysisHandoff,
    acknowledgeInputTransfer,
    analysisUrlFen,
    transitionPendingInput,
    url,
    el,
  };
  const state = source.slice(source.indexOf('let tree = new GameTree(START_FEN)'), source.indexOf('let searcher:'));
  const mount = source.slice(
    source.indexOf('  workspaceAutosave = createWorkspaceAutosave('),
    source.indexOf('  analysisWorkerProgressMessages = 0;', source.indexOf('  workspaceAutosave = createWorkspaceAutosave(')),
  );
  const reset = source.slice(source.indexOf("  el('reset').addEventListener"), source.indexOf("  el('loadPgn').addEventListener"));
  const load = source.slice(source.indexOf('function loadFen()'), source.indexOf('async function copyFen()'));
  const previousHandler = source.slice(
    source.indexOf("  el('restorePreviousAnalysis').addEventListener"),
    source.indexOf("  el('analysisRestoreDismiss').addEventListener"),
  );
  const code = `
    const location = url, params = new URLSearchParams(url.search), inputEl = el;
    let lastAnalysisMountHref = url.href;
    const history = {state: {}, replaceState(state, title, next) {url.href = next.href;}};
    const reviewMetadata = new Map(), reviewSignature = () => '';
    const renderPgnImportNotice = () => {}, lineCache = new Map(), completeAnalysisKeys = new Set();
    const cancelActiveAnalysisRun = () => {};
    const afterNavigation = () => saveAnalysisWorkspace();
    ${stripTypeScriptTypes(state + mount + reset + load + previousHandler)}
    return {restore: restoreAnalysisWorkspace, save: saveAnalysisWorkspace, flush: () => workspaceAutosave.flush(),
      reset: () => el('reset').click(),
      loadFen: raw => {el('fenInput').value = raw; loadFen();},
      loadPgn: raw => {el('pgnInput').value = raw; loadPgn();},
      previous: () => el('restorePreviousAnalysis').click(), tree: () => tree, dispose: () => workspaceAutosave.pause('storage')};
  `;
  let controller;
  try {
    controller = new Function(...Object.keys(dependencies), code)(...Object.values(dependencies));
    run({ ...controller, values, faults, url, storage });
  } finally {
    controller?.dispose();
    if (previous) Object.defineProperty(globalThis, 'localStorage', previous);
    else delete globalThis.localStorage;
  }
}

test('Analysis URL is acknowledged by a later autosave after initial quota failure, preserving edits on reload', () => {
  analysisHarness('pgn=' + encodeURIComponent('1. e4 *'), (h) => {
    h.faults.write = true;
    h.restore();
    assert.equal(h.url.searchParams.has('pgn'), true);
    h.faults.write = false;
    h.tree().toEnd();
    h.tree().addUci('e7e5');
    h.save();
    h.flush();
    assert.equal(h.url.searchParams.has('pgn'), false);
    assert.equal(h.url.searchParams.has('dev'), true);
    assert.equal(h.url.hash, '#board');
    const saved = autosave.readWorkspace('analysis', autosave.decodeAnalysisSnapshot, h.storage);
    assert.equal(saved.tree.mainlineFrom().length, 2, 'Reload restores the edits instead of reimporting the one-ply link');
  });
});

test('Analysis START_FEN URL is acknowledged when its empty autosave clear succeeds', () => {
  analysisHarness('fen=' + encodeURIComponent(START_FEN), (h) => {
    h.restore();
    assert.equal(h.url.searchParams.has('fen'), false);
    assert.equal(h.values.has('0x88-analysis-workspace'), false);
  });
});

test('Analysis empty clear failure retains the URL until a later clear succeeds', () => {
  analysisHarness('fen=' + encodeURIComponent(START_FEN), (h) => {
    h.faults.remove = true;
    h.restore();
    assert.equal(h.url.searchParams.has('fen'), true);
    h.faults.remove = false;
    h.save();
    assert.equal(h.url.searchParams.has('fen'), false);
  });
});

test('Analysis URL supersedes and retires the captured older handoff after saving', () => {
  for (const query of ['pgn=' + encodeURIComponent('1. d4 *'), 'fen=' + encodeURIComponent(START_FEN)]) {
    analysisHarness(query, (h) => {
      const key = '0x88-analysis-handoff-pgn-v1';
      h.values.set(key, JSON.stringify({ pgn: '1. e4 *', at: Date.now() }));
      h.restore();
      assert.equal(h.values.has(key), false, 'The old handoff must not replace the URL-selected workspace on reload');
      assert.equal(h.url.searchParams.has('pgn') || h.url.searchParams.has('fen'), false);
    });
  }
});

test('Analysis Reset explicitly retires a handoff after a failed save even when clearing storage fails', () => {
  analysisHarness('', (h) => {
    const key = '0x88-analysis-handoff-pgn-v1';
    h.values.set(key, JSON.stringify({ pgn: '1. e4 *', at: Date.now() }));
    h.faults.write = true;
    h.restore();
    assert.equal(h.values.has(key), true);
    // Only workspace removal fails; handoff retirement remains available.
    const remove = h.storage.removeItem;
    h.storage.removeItem = (k) => {
      if (k === '0x88-analysis-workspace') throw Error('denied');
      remove(k);
    };
    h.reset();
    assert.equal(h.values.has(key), false, 'Reset must not resurrect the discarded game on reload');
    assert.equal(h.tree().root.children.length, 0);
  });
});

test('Analysis acknowledgement never retires a newer handoff', () => {
  analysisHarness('pgn=' + encodeURIComponent('1. d4 *'), (h) => {
    const key = '0x88-analysis-handoff-pgn-v1';
    h.values.set(key, JSON.stringify({ pgn: '1. e4 *', at: Date.now() }));
    h.faults.write = true;
    h.restore();
    const newer = JSON.stringify({ pgn: '1. c4 *', at: Date.now() });
    h.values.set(key, newer);
    h.faults.write = false;
    h.save();
    h.flush();
    assert.equal(h.values.get(key), newer);
  });
});

for (const action of ['reset', 'loadFen', 'loadPgn', 'previous']) {
  test(`Analysis ${action} retires pending handoff and URL after failed saving`, () => {
    analysisHarness('pgn=' + encodeURIComponent('1. d4 *'), (h) => {
      const key = '0x88-analysis-handoff-pgn-v1';
      h.values.set(key, JSON.stringify({ pgn: '1. e4 *', at: Date.now() }));
      h.faults.write = true;
      h.restore();
      if (action === 'previous') {
        h.values.set(
          autosave.PREVIOUS_ANALYSIS_KEY,
          JSON.stringify({ kind: 'analysis', version: 1, savedAt: 1, data: { tree: autosave.serializeTree(new GameTree()), headers: {}, result: '*' } }),
        );
        h.faults.write = false;
      }
      h[action](action === 'loadFen' ? START_FEN : '1. c4 *');
      assert.equal(h.values.has(key), false);
      assert.equal(h.url.searchParams.has('pgn'), false);
    });
  });
}
