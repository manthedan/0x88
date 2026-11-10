import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { START_FEN } from '../src/chess/board.ts';
import { GameTree } from '../src/lc0/gameTree.ts';
import { createWorkspaceAutosave, serializeTree } from '../src/lc0/workspaceAutosave.ts';

const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');

test('restoring an identical mainline retires review nodes from the discarded tree', () => {
  const oldTree = new GameTree();
  oldTree.addUci('e2e4');
  const restoredTree = new GameTree();
  restoredTree.addUci('e2e4');
  const handler = source.slice(
    source.indexOf("el('restorePreviousAnalysis').addEventListener"),
    source.indexOf("el('analysisRestoreDismiss').addEventListener"),
  );
  const dependencies = {
    oldTree,
    restoredTree,
    serializeTree,
  };
  const code = `
    let tree = oldTree, analysisGameTree = tree, lastReviewNodes = [tree.root, tree.current];
    let workspaceHeaders = {}, workspaceResult = '*', handler;
    const workspaceAutosave = {flush() {}}, restorePreviousAnalysis = () => ({tree: restoredTree, headers: {}, result: '*'});
    const cancelActiveAnalysisRun = () => {}, reviewMetadata = new Map(), reviewSignature = () => '';
    const lineCache = new Map(), completeAnalysisKeys = new Set(), renderPgnImportNotice = () => {};
    const el = () => ({addEventListener(event, fn) {handler = fn;}, focus() {}}), inputEl = el;
    const gameTreeToPgn = () => '';
    function afterNavigation() {
      if (analysisGameTree !== tree) {analysisGameTree = tree; lastReviewNodes = [];}
    }
    ${stripTypeScriptTypes(handler)}
    handler();
    assert.equal(lastReviewNodes.length, 0, 'Review graph must not retain old GameNodes');
    assert.doesNotThrow(() => serializeTree(tree));
  `;
  new Function('assert', ...Object.keys(dependencies), code)(assert, ...Object.values(dependencies));
});

test('serialization failure preserves the last autosave, including an empty tree with a stale current node', () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value), removeItem: (key) => values.delete(key) };
  const saver = createWorkspaceAutosave('analysis', storage);
  const tree = new GameTree();
  tree.addUci('e2e4');
  saver.schedule({ tree: serializeTree(tree), headers: {}, result: '*' });
  saver.flush();
  const previous = values.get('0x88-analysis-workspace');
  tree.root.children = [];
  const save = source.slice(source.indexOf('function saveAnalysisWorkspace()'), source.indexOf('function clearAnalysisWorkspace()'));
  const dependencies = { tree, workspaceAutosave: saver, workspaceHeaders: {}, workspaceResult: '*', reviewMetadata: new Map(), serializeTree, START_FEN };
  new Function(...Object.keys(dependencies), `${stripTypeScriptTypes(save)}; saveAnalysisWorkspace();`)(...Object.values(dependencies));
  saver.flush();
  assert.equal(values.get('0x88-analysis-workspace'), previous);
});
