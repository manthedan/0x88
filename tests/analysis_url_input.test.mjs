import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { parseFirstPgnGame } from '../src/chess/pgn.ts';
import { analysisUrlFen } from '../src/lc0/analysisUrlInput.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { transitionPendingInput } from '../src/lc0/pendingInput.ts';
import { createWorkspaceAutosave, hasExplicitWorkspaceInput, serializeTree, WORKSPACE_MAX_CHARS } from '../src/lc0/workspaceAutosave.ts';

test('missing and valid URL FENs initialize safely', () => {
  assert.deepEqual(analysisUrlFen(null), { fen: START_FEN, error: null });
  assert.deepEqual(analysisUrlFen(START_FEN), { fen: START_FEN, error: null });
});

test('malformed and illegal URL roots fall back with an actionable problem', () => {
  for (const raw of ['', 'not-a-fen', '8/8/8/8/8/8/8/8 w - - 0 1', '4k3/8/8/8/8/8/8/P3K3 w - - 0 1']) {
    const result = analysisUrlFen(raw);
    assert.equal(result.fen, START_FEN, raw);
    assert.match(result.error, /Could not load linked FEN: Invalid FEN/);
  }
});

test('invalid links preserve the saved tree and attach their visible problem to the restored workspace', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const restore = source.slice(source.indexOf('function restoreAnalysisWorkspace()'), source.indexOf('let searcher:'));
  for (const saved of [null, { tree: new GameTree(), headers: { Event: 'Saved' }, result: '*' }]) {
    for (const query of ['fen=not-a-fen', 'fen=', 'pgn=invalid']) {
      const elements = new Map(['restorePreviousAnalysis', 'pgnInput', 'analysisRestoreNotice'].map((key) => [key, { value: '', hidden: true }]));
      const dependencies = { GameTree, START_FEN, analysisUrlFen, hasExplicitWorkspaceInput, elements, saved, query };
      const code = `
        let tree = new GameTree(), analysisGameTree = tree, workspaceHeaders = {}, workspaceResult = '*', notice;
        const params = new URLSearchParams(query), location = {search: '?' + query};
        const takeAnalysisHandoff = () => null, hasPreviousAnalysis = () => false;
        const parseFirstPgnGame = () => { throw Error('Invalid PGN game'); };
        const preservePreviousAnalysis = () => { throw Error('Invalid input must not back up or replace'); };
        const clearAnalysisWorkspace = () => { throw Error('Invalid input must not clear'); };
        const readWorkspace = () => saved, decodeAnalysisSnapshot = () => {};
        const el = key => elements.get(key), inputEl = el;
        const reviewMetadata = new Map(), reviewSignature = () => '';
        const gameTreeToPgn = () => 'Saved PGN';
        const renderPgnImportNotice = text => { notice = {tree, text}; };
        ${stripTypeScriptTypes(restore)}
        restoreAnalysisWorkspace();
        return {tree, notice};
      `;
      const state = new Function(...Object.keys(dependencies), code)(...Object.values(dependencies));
      if (saved) assert.equal(state.tree, saved.tree);
      else assert.equal(state.tree.root.fen, START_FEN);
      assert.equal(state.notice.tree, state.tree, 'Automatic status rendering must retain the error after restoring a saved tree');
      assert.match(state.notice.text, /Could not load linked analysis/);
    }
  }
});

test('linked PGN rejection and backup failure remain visible after fallback status rendering', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const restore = source.slice(source.indexOf('function restoreAnalysisWorkspace()'), source.indexOf('let searcher:'));
  const notice = source.slice(source.indexOf('let pgnImportNotice:'), source.indexOf('/** Labelled run state'));
  for (const pgn of ['not-a-game', '1. d4 d5 *']) {
    const saved = { tree: new GameTree(), headers: { Event: 'Saved' }, result: '*' };
    saved.tree.addUci('e2e4');
    const elements = new Map(
      ['restorePreviousAnalysis', 'pgnInput', 'analysisRestoreNotice', 'pgnImportNotice'].map((id) => [id, { textContent: '', hidden: true }]),
    );
    const dependencies = { GameTree, START_FEN, analysisUrlFen, hasExplicitWorkspaceInput, parseFirstPgnGame, saved, elements, pgn };
    const code = `
      let tree = new GameTree(), analysisGameTree = tree, workspaceHeaders = {}, workspaceResult = '*';
      const params = new URLSearchParams({pgn}), location = {search: '?' + params};
      const takeAnalysisHandoff = () => null, hasPreviousAnalysis = () => false;
      const preservePreviousAnalysis = () => false;
      const clearAnalysisWorkspace = () => { throw Error('Must not clear after rejection'); };
      const readWorkspace = () => saved, decodeAnalysisSnapshot = () => {};
      const el = id => elements.get(id), inputEl = el;
      const reviewMetadata = new Map(), reviewSignature = () => '', gameTreeToPgn = () => 'Saved PGN';
      ${stripTypeScriptTypes(notice + restore)}
      restoreAnalysisWorkspace();
      for (let i = 0; i < 3; i++) renderPgnImportNotice();
      return {tree, text: el('pgnImportNotice').textContent};
    `;
    const actual = new Function(...Object.keys(dependencies), code)(...Object.values(dependencies));
    assert.equal(actual.tree, saved.tree);
    assert.match(actual.text, /Could not load linked analysis/);
    assert.match(actual.text, pgn.startsWith('1.') ? /Export it before replacing/ : /PGN|game/i);
  }
});

test('linked inputs stay in the URL until a recoverable snapshot is written', () => {
  const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
  const lifecycle = source.slice(source.indexOf('let pendingAnalysisInput:'), source.indexOf('function saveAnalysisWorkspace()'));
  const save = source.slice(source.indexOf('function saveAnalysisWorkspace()'), source.indexOf('function clearAnalysisWorkspace('));
  const restore = source.slice(source.indexOf('function restoreAnalysisWorkspace()'), source.indexOf('let searcher:'));
  for (const mode of ['quota', 'too-large', 'success']) {
    const values = new Map();
    const storage = {
      getItem: (key) => values.get(key) ?? null,
      setItem(key, value) {
        if (mode === 'quota') throw Error('quota');
        values.set(key, value);
      },
      removeItem: (key) => values.delete(key),
    };
    let commit = () => {};
    const saver = createWorkspaceAutosave('analysis', storage, 400, undefined, undefined, (type) => commit(type));
    const linked = mode === 'too-large' ? '{' + 'x'.repeat(WORKSPACE_MAX_CHARS) + '} 1. e4 *' : '1. e4 *';
    const url = new URL('https://example.test/app/analysis/?dev&pgn=' + encodeURIComponent(linked) + '#board');
    const dependencies = {
      GameTree,
      START_FEN,
      analysisUrlFen,
      hasExplicitWorkspaceInput,
      parseFirstPgnGame,
      serializeTree,
      saver,
      url,
      transitionPendingInput,
      setCommit: (fn) => {
        commit = fn;
      },
    };
    const code = `
      let tree = new GameTree(), analysisGameTree = tree, workspaceHeaders = {}, workspaceResult = '*', lastAnalysisMountHref;
      const workspaceAutosave = saver, reviewMetadata = new Map(), reviewSignature = () => '';
      const params = new URLSearchParams(url.search), location = url;
      const history = {state: {}, replaceState(state, title, next) {url.href = next.href;}};
      const takeAnalysisHandoff = () => null, hasPreviousAnalysis = () => false, preservePreviousAnalysis = () => true;
      const clearAnalysisWorkspace = () => saver.clear(), readWorkspace = () => null, decodeAnalysisSnapshot = () => {};
      const el = () => ({}), inputEl = el, renderPgnImportNotice = () => {};
      ${stripTypeScriptTypes(lifecycle + save + restore)}
      setCommit(type => updatePendingAnalysisInput({type}));
      restoreAnalysisWorkspace();
    `;
    new Function(...Object.keys(dependencies), code)(...Object.values(dependencies));
    assert.equal(url.searchParams.has('pgn'), mode !== 'success', mode);
    assert.equal(values.has('0x88-analysis-workspace'), mode === 'success', mode);
    assert.equal(url.searchParams.has('dev'), true);
    assert.equal(url.hash, '#board');
    saver.clear();
  }
});
