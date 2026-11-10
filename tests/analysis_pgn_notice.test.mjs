import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import { describePgnWarning, parseFirstPgnGame } from '../src/chess/pgn.ts';
import { GameTree } from '../src/lc0/gameTree.ts';

// Run the actual import/status functions without mounting the page or loading engines.
const source = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const statusStart = source.includes('let pgnImportNotice:') ? source.indexOf('let pgnImportNotice:') : source.indexOf('function renderAnalysisStatus()');
const statusFunctions = source.slice(statusStart, source.indexOf('function activeEngineNames()'));
const loadFunction = source.slice(source.indexOf('function loadPgn()'), source.indexOf('function copyFen()'));

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
  for (const id of ['message', 'analysisStatus', 'stop', 'analyze', 'analyzeLabel', 'pgnInput']) elements.set(id, { ...element(), id });
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
  return { ...api, notice: () => document.getElementById('pgnImportNotice')?.textContent ?? '' };
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
