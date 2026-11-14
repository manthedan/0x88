import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { analysisOperationId, initialAnalysisRunState, transitionAnalysis } from '../src/lc0/analysisRunState.ts';
import { cancelHiddenTabWork, initialHiddenTabPause, transitionHiddenTab } from '../src/lc0/hiddenTabPause.ts';
import { boardToFen, START_FEN } from '../src/chess/board.ts';
import { analysisResultCacheKey, cachedAnalysisSelection, cachedReviewPosition } from '../src/lc0/analysisCache.ts';
import { evalBarWhitePercent, stockfishAnalysisLines } from '../src/lc0/analysisFormat.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

const row = { family: 'sf', variant: 'lite', strength: 12 };
const settings = { multiPv: 1, runtime: '', cpu: 'balanced/2/sf-lite' };
const key = (r = row, s = settings, history = [START_FEN]) => analysisResultCacheKey(START_FEN, history, r, s);

test('review cache identity requires exact engine, depth, history, MultiPV and runtime settings', () => {
  assert.equal(key(), key({ ...row }, { ...settings }, [START_FEN]));
  assert.notEqual(key(), key({ ...row, variant: 'full' }));
  assert.notEqual(key(), key({ ...row, strength: 13 }));
  assert.notEqual(key(), key({ ...row, family: 'berserk' }));
  assert.notEqual(key(), key(row, { ...settings, multiPv: 3 }));
  assert.notEqual(key(), key(row, { ...settings, runtime: 'wasm' }));
  assert.notEqual(key(), key(row, { ...settings, cpu: 'eco/1/sf-lite' }));
  assert.notEqual(key(), key(row, settings, [START_FEN, START_FEN]));
});

test('completed selections restore the right history after a same-FEN transposition', () => {
  const histories = [
    ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'],
    ['g1f3', 'g8f6', 'f3h4', 'f6h5', 'h4f3', 'h5f6', 'f3g1', 'f6g8'],
  ].map((moves) => {
    const game = new GameTree();
    for (const move of moves) game.addUci(move);
    return game;
  });
  assert.equal(histories[0].current.fen, histories[1].current.fen);
  const keys = histories.map((game) => analysisResultCacheKey(game.current.fen, game.historyBoards().map(boardToFen), row, settings));
  assert.notEqual(keys[0], keys[1]);
  const a = stockfishAnalysisLines([{ multipv: 1, depth: 12, scoreCp: 0, pvUci: ['g1f3'] }], histories[0].current.fen, 'SF Lite');
  const b = stockfishAnalysisLines([{ multipv: 1, depth: 12, scoreCp: 16, pvUci: ['e2e4'] }], histories[1].current.fen, 'SF Lite');
  const cache = new Map([
    [keys[0], a],
    [keys[1], b],
  ]);
  assert.deepEqual(cachedAnalysisSelection(cache, [keys[0]]), a);
  assert.deepEqual(cachedAnalysisSelection(cache, [keys[1]]), b);
  assert.deepEqual(cachedAnalysisSelection(cache, [keys[1], keys[0]]), [...b, ...a]);
  cache.delete(keys[0]);
  assert.equal(cachedAnalysisSelection(cache, [keys[0]]), undefined, 'a completed marker must not hide an evicted result');
  assert.equal(cachedAnalysisSelection(cache, [keys[1], keys[0]]), undefined, 'do not restore a partial engine selection');
});

test('cached review scores preserve White perspective on Black-to-move boards', () => {
  const fen = START_FEN.replace(' w ', ' b ');
  for (const score of [{ scoreCp: 120 }, { mateIn: 3 }, { mateIn: -2 }]) {
    const infos = [{ multipv: 1, depth: 12, pvUci: ['e7e5'], ...score }];
    const lines = stockfishAnalysisLines(infos, fen, 'SF Lite');
    const cached = cachedReviewPosition(lines, 20);
    assert.deepEqual(cached, {
      winWhite: evalBarWhitePercent(score.scoreCp === undefined ? undefined : -score.scoreCp, score.mateIn === undefined ? undefined : -score.mateIn) / 100,
      bestUci: 'e7e5',
      legalMoves: 20,
      ...(score.scoreCp === undefined ? {} : { scoreCp: -score.scoreCp }),
      ...(score.mateIn === undefined ? {} : { mateIn: -score.mateIn }),
    });
  }
});

test('review ignores errors, missing top lines and unscored cache entries', () => {
  assert.equal(cachedReviewPosition(undefined, 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 2, scoreCp: 10 }], 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 1, error: 'failed', scoreCp: 10 }], 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 1 }], 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 1, scoreCp: NaN }], 20), undefined);
  assert.equal(cachedReviewPosition([{ multipv: 1, mateIn: Infinity }], 20), undefined);
});

function reviewHandoffPage() {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const slice = (start, end) => {
    const from = source.indexOf(start), to = source.indexOf(end, from);
    assert.ok(from >= 0 && to > from);
    return source.slice(from, to);
  };
  const code = ts.transpile(`
    const { initialAnalysisRunState, transitionAnalysis, analysisOperationId, initialHiddenTabPause, cancelHiddenTabWork, transitionHiddenTab } = helpers;
    let state = initialAnalysisRunState(), id = 0;
    const effects = [];
    const dispatchAnalysis = (event) => {
      const next = transitionAnalysis(state, { ...event, nextId: ++id });
      state = next.state; effects.push(...next.effects);
      return Promise.resolve();
    };
    const tree = new GameTree(); tree.addUci('e2e4');
    const mainlineNodes = () => tree.lineTo();
    const reviewSignature = () => 'mainline';
    let reviewAbort = null, reviewRunSignature = '', lastReview = null, lastReviewReviewer = null;
    let lastReviewNodes = [], lastReviewSignature = '', retry = null, retryReviewer = null;
    let analysisAbort = null, hiddenTab = initialHiddenTabPause(false);
    let retryModeGeneration = 0, retryHumanGeneration = 0, orientation = 'white';
    let retryOriginalOrientation, retryOriginalAuto, retryOriginalShapes = [], restoreRetryPanels = null;
    const elements = new Map();
    const el = (id) => elements.get(id) ?? (elements.set(id, {
      textContent: '', hidden: false, checked: true, value: id === 'retrySide' ? 'w' : '',
      toggleAttribute() {}, focus() {}, querySelector() { return {}; }, scrollTop: 0,
    }), elements.get(id));
    const inputEl = el, selectEl = el;
    const workspace = { dataset: {} }, document = { querySelector: () => workspace };
    const criticalPanel = { invalidate() {}, sync() {}, isBusy: () => false };
    const ground = { state: { drawable: { shapes: [] } }, setShapes() {} }, keyboardBoard = { focus() {} };
    const maia3PanelTask = { invalidate() {} };
    let maia3PanelTimer = null;
    const maia3PanelEvaluator = null, maia3PanelLoading = false, maia3PanelStatusText = '';
    const retireAnalysisRunCentipawnKeys = () => {}, clearAnalysisSearchProgress = () => {};
    let failChoice = false, failSetup = false;
    const closePvPreview = () => {}, hidePromotionOverlay = () => {};
    const syncBrokerParticipants = () => { if (failSetup) throw new Error('setup failed'); };
    const quarantineRetryPanels = () => () => {}, renderMistakeRetry = () => {};
    const renderAll = () => {}, renderAnalysisStatus = () => {}, scheduleMaia3Panel = () => {};
    const activeEngineRows = () => [], reviewer = { engine: {}, row: { family: 'sf', variant: 'lite', strength: 12 }, label: 'SF Lite', depth: 12 };
    const reviewEngineChoice = () => { if (failChoice) throw new Error('unsupported reviewer'); return reviewer; };
    const retryProblems = () => [{}];
    class MistakeRetry { cancel() {} }
    const clearReviewState = () => {};
    const engineAnalysisCacheKey = (_fen, _row, node) => node.id;
    const engineLineCache = new Map(mainlineNodes().map((node) => [node.id, [{ multipv: 1, scoreCp: 20, pvUci: ['e2e4'] }]]));
    const cachedReviewPosition = helpers.cachedReviewPosition;
    const parseFen = () => ({ turn: 'w' }), legalMoves = () => Array(20);
    const terminalReviewPosition = () => null, moveToUci = () => 'e2e4';
    const reviewGame = () => ({ moves: [{ class: 'mistake', ply: 1 }] });
    const retrySideFromTags = () => 'w', reviewMetadata = new Map();
    const renderReview = () => {}, analysisEngineName = (label) => label;
    const matchMedia = () => ({ matches: false });
    ${slice('function stopGameReview(', 'function clearReviewIfMainlineChanged(')}
    ${slice('function startMistakeRetry(', 'function renderMistakeRetry(')}
    ${slice('function exitMistakeRetry(', 'async function runGameReview(')}
    ${slice('async function runGameReview(', 'function reviewedPgn(')}
    ${slice('function applyHiddenTab(', 'function installHiddenTabPause(')}
    return {
      review: runGameReview, stopReview: stopGameReview, startRetry: startMistakeRetry, exitRetry: exitMistakeRetry,
      send: dispatchAnalysis, state: () => state, effects, el,
      hideWithWork: () => { hiddenTab = transitionHiddenTab(hiddenTab, true, ['analysis', 'maia']).state; },
      hidden: () => hiddenTab, visibility: applyHiddenTab,
      failChoice: (value) => { failChoice = value; }, failSetup: (value) => { failSetup = value; },
    };
  `, { target: ts.ScriptTarget.ES2022 });
  return new Function('GameTree', 'helpers', code)(GameTree, { initialAnalysisRunState, transitionAnalysis, analysisOperationId, initialHiddenTabPause, cancelHiddenTabWork, transitionHiddenTab, cachedReviewPosition });
}

for (const cancel of [false, true]) {
  test(`review ${cancel ? 'cancellation' : 'completion'} releases admission without draining reviewer maintenance`, async () => {
    const page = reviewHandoffPage();
    await page.send({ type: 'REQUEST', intent: { requestId: 100, force: true }, delayMs: 0 });
    const operationId = analysisOperationId(page.state().execution);
    await page.send({ type: 'SETTINGS_CHANGED', revision: 1, intent: { requestId: 101, force: true }, request: true, maintenance: { reloadLc0: true, disposeUnused: true } });
    const review = page.review();
    assert.deepEqual(page.state().modes, ['review']);
    assert.equal(page.state().wanted, null);
    await page.send({ type: 'SETTLED', operationId });
    assert.equal(page.effects.some((effect) => effect.type === 'maintain'), false);
    if (cancel) page.stopReview();
    await review;
    assert.deepEqual(page.state().modes, []);
    assert.equal(page.state().execution.tag, 'idle');
    assert.deepEqual(page.state().maintenance, { reloadLc0: true, disposeUnused: true });
    assert.equal(page.effects.some((effect) => effect.type === 'maintain'), false);
  });
}

test('review setup failure never leaves review admission held', async () => {
  const page = reviewHandoffPage();
  await page.send({ type: 'REQUEST', intent: { requestId: 100, force: true }, delayMs: 0 });
  page.failChoice(true);
  await assert.rejects(page.review(), /unsupported reviewer/);
  // As before the reducer, the handoff cancels the search; only the mode is released.
  assert.deepEqual(page.state().modes, []);
  assert.equal(page.state().execution.tag, 'idle');
  page.failChoice(false);
  page.failSetup(true);
  await assert.rejects(page.review(), /setup failed/);
  assert.deepEqual(page.state().modes, []);
  page.failSetup(false);
  await page.send({ type: 'REQUEST', intent: { requestId: 101, force: false }, delayMs: 0 });
  assert.equal(page.state().execution.tag, 'preparing');
});

test('retry handoff clears hidden analysis/Maia resumes and restores auto without restarting search', async () => {
  const page = reviewHandoffPage();
  await page.review();
  await page.send({ type: 'REQUEST', intent: { requestId: 100, force: true }, delayMs: 250 });
  page.hideWithWork();
  page.startRetry();
  assert.deepEqual(page.state().modes, ['retry']);
  assert.equal(page.state().wanted, null);
  assert.deepEqual(page.hidden().suspended, []);
  assert.equal(page.el('autoAnalyze').checked, false);
  page.visibility(true);
  page.visibility(false);
  assert.deepEqual(page.state().modes, ['retry']);
  assert.equal(page.state().visible, true);
  assert.deepEqual(page.hidden().suspended, []);
  page.exitRetry(false);
  assert.deepEqual(page.state().modes, []);
  assert.equal(page.el('autoAnalyze').checked, true);
  assert.equal(page.state().execution.tag, 'idle');
  assert.equal(page.effects.some((effect) => effect.type === 'search'), false);
});

test('review retains original finite evals for export and ignores a corrupt mate field', () => {
  const position = cachedReviewPosition([{ multipv: 1, scoreCp: 25, mateIn: NaN, pvUci: [] }], 12);
  assert.equal(position.winWhite, evalBarWhitePercent(25, undefined) / 100);
  assert.equal(position.scoreCp, 25);
  assert.equal(position.bestUci, null);
  assert.equal(position.mateIn, undefined);
});
