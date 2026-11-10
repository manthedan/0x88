import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { analysisUrlFen } from '../src/lc0/analysisUrlInput.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { hasExplicitWorkspaceInput } from '../src/lc0/workspaceAutosave.ts';

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
