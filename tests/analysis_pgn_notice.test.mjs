import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { compile } from 'svelte/compiler';
import ts from 'typescript';
import { describePgnWarning, parseFirstPgnGame } from '../src/chess/pgn.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

// Run the actual import/status functions without mounting the page or loading engines.
const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const template = readFileSync(new URL('../src/routes/app/analysis/+page.svelte', import.meta.url), 'utf8');
const statusStart = source.includes('let pgnImportNotice:') ? source.indexOf('let pgnImportNotice:') : source.indexOf('function renderAnalysisStatus()');
const statusFunctions = source.slice(statusStart, source.indexOf('function activeEngineNames()'));
const loadFunction = source.slice(source.indexOf('function loadPgn()'), source.indexOf('async function copyFen()'));

function controller() {
  const elements = new Map();
  const element = () => ({
    id: '',
    textContent: '',
    dataset: {},
    disabled: false,
    style: {},
    closest: () => null,
    setAttribute() {},
    after(child) {
      elements.set(child.id, child);
    },
    remove() {
      elements.delete(this.id);
    },
  });
  for (const id of ['message', 'pgnImportNotice', 'analysisStatus', 'stop', 'analyze', 'analyzeLabel', 'pgnInput']) elements.set(id, { ...element(), id });
  const document = { getElementById: (id) => elements.get(id) ?? null, createElement: element };
  const code = ts.transpile(
    `
    let tree = new GameTree(), workspaceHeaders = {}, workspaceResult = '*';
    const el = id => document.getElementById(id), inputEl = el;
    const reviewMetadata = new Map(), lineCache = new Map(), completeAnalysisKeys = new Set();
    const retry = false, reviewSignature = () => '', clearAnalysisWorkspace = () => {};
    const analysisEngineName = engine => engine;
    ${statusFunctions}
    function afterNavigation() {
      // A ready engine starts synchronously, before the browser can paint the load message.
      el('message').textContent = 'Analyzing (ready engine, 1 lines)…';
      renderAnalysisStatus();
    }
    ${loadFunction}
    return {
      load(raw) { inputEl('pgnInput').value = raw; loadPgn(); },
      engineStatus(text) { el('message').textContent = text; renderAnalysisStatus(); },
      navigate() { tree.addUci('e2e4'); renderAnalysisStatus(); },
      replaceTree() { tree = new GameTree(); renderAnalysisStatus(); },
    };
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  const api = new Function('document', 'GameTree', 'parseFirstPgnGame', 'describePgnWarning', code)(document, GameTree, parseFirstPgnGame, describePgnWarning);
  return { ...api, notice: () => document.getElementById('pgnImportNotice')?.textContent ?? '', elements };
}

test('PGN truncation warning stays visible through automatic analysis status updates', () => {
  const api = controller();
  api.load('1. e4 e5 2. Nf3 bad Nc6 *');
  assert.match(api.notice(), /Stopped at move 2\.\.\.: bad/);
  for (const text of ['Loading engine…', 'Analyzing…', 'Analysis complete · ready engine']) {
    api.engineStatus(text);
    assert.match(api.notice(), /the rest of that line was not loaded/);
  }
  api.navigate();
  assert.equal(api.notice(), '');
});

test('ignored-game warning stays visible until another game is loaded', () => {
  const api = controller();
  api.load('1. e4 *\n1. d4 *');
  assert.match(api.notice(), /Only the first of 2 games was loaded/);
  api.engineStatus('Analyzing…');
  assert.match(api.notice(), /Only the first of 2 games was loaded/);
  api.load('1. d4 d5 *');
  assert.equal(api.notice(), '');
  api.load('1. e4 bad *');
  assert.match(api.notice(), /Stopped/);
  api.replaceTree();
  assert.equal(api.notice(), '', 'loading a FEN also retires the old notice');
  api.load('1. e4 bad *');
  api.load('not PGN');
  assert.equal(api.notice(), '', 'a new load attempt retires the previous import notice');
});

test('long malformed command import leaves a concise persistent warning', () => {
  const api = controller();
  api.load(`1. e4 [%clk ${'Nf3 Nf6 Ng1 Ng8 '.repeat(1000)}`);
  assert.ok(api.notice().length < 250);
  assert.match(api.notice(), /… \(unclosed bracket command\)/);
});

test('persistent PGN notice uses the template element and clears it without removing it', () => {
  const api = controller();
  const notice = api.elements.get('pgnImportNotice');
  api.load('1. e4 bad *');
  assert.equal(api.elements.get('pgnImportNotice'), notice);
  api.navigate();
  assert.equal(api.elements.get('pgnImportNotice'), notice);
  assert.equal(notice.textContent, '');
});

test('template notice is scoped, accessible, bounded and placed across the mobile grid', () => {
  assert.match(template, /<div id="pgnImportNotice"[^>]*role="status"[^>]*aria-live="polite"[^>]*><\/div>/);
  const { css } = compile(template, { generate: 'client' });
  assert.match(css.code, /#pgnImportNotice[^{}]*\{[^{}]*max-height:44px;[^{}]*overflow:auto/);
  assert.match(css.code, /#pgnImportNotice[^{}]*:empty[^{}]*\{display:none\}/);
  assert.match(css.code, /#pgnImportNotice[^{}]*\{grid-column:1 \/ -1; grid-row:calc\(5 \+ var\(--restore-row\)\)\}/);
});

function exportController() {
  const elements = new Map(
    ['message', 'pgnInput', 'analysisRestoreNotice', 'restorePreviousAnalysis'].map((id) => [id, { value: 'previous PGN', textContent: '', hidden: true }]),
  );
  const restore = source.slice(source.indexOf('function restoreAnalysisWorkspace()'), source.indexOf('let searcher:'));
  const copy = source.slice(source.indexOf('async function copyPgn()'), source.indexOf('function closePvPreview()'));
  const code = ts.transpile(
    `
    let tree = new GameTree(), analysisGameTree = tree, workspaceHeaders = {}, workspaceResult = '*';
    const reviewMetadata = new Map(), reviewSignature = () => 'signature';
    const el = id => elements.get(id), inputEl = el;
    const takeAnalysisHandoff = () => null, hasExplicitWorkspaceInput = () => false, hasPreviousAnalysis = () => false;
    const location = {search: '', hash: ''}, decodeAnalysisSnapshot = () => {}, params = new Map();
    const readWorkspace = () => ({ tree: new GameTree(), headers: {}, result: '*' });
    const renderPgnImportNotice = text => { el('message').textContent = text; };
    const pgnResultForExport = (headers, result) => ({ headers, result });
    const gameTreeToPgn = () => { throw new Error('export failed'); };
    ${restore}
    ${copy}
    return { restoreAnalysisWorkspace, copyPgn, metadata: () => reviewMetadata.get(tree) };
  `,
    { target: ts.ScriptTarget.ESNext, module: ts.ModuleKind.ESNext },
  );
  return { ...new Function('elements', 'GameTree', code)(elements, GameTree), elements };
}

test('workspace restore survives an unexpected export error and finishes restoring metadata', () => {
  const api = exportController();
  assert.doesNotThrow(() => api.restoreAnalysisWorkspace());
  assert.equal(api.metadata().signature, 'signature');
  assert.equal(api.elements.get('analysisRestoreNotice').hidden, false);
  assert.match(api.elements.get('message').textContent, /PGN export failed.*export failed/);
});

test('copy PGN reports export errors without changing the existing input', () => {
  const api = exportController();
  assert.doesNotThrow(() => api.copyPgn());
  assert.equal(api.elements.get('pgnInput').value, 'previous PGN');
  assert.match(api.elements.get('message').textContent, /PGN export failed.*export failed/);
});
