import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import test from 'node:test';

// The end of Analysis init(), after the startup asset checks have been awaited.
const text = readFileSync(new URL('../src/lc0/analysisBrowser.ts', import.meta.url), 'utf8');
const initStart = text.indexOf('async function init(mountSignal: AbortSignal)');
const anchor = '  if (!isV0DeployProfile()) void refreshCentipawnHybridManifestStatus();\n';
const tailStart = text.indexOf(anchor, initStart) + anchor.length;
const tail = text.slice(tailStart, text.indexOf('\n}\n', tailStart));

function startup({ autoAnalyze, smallLc0 }) {
  const calls = [];
  const element = () => ({
    textContent: '',
    disabled: true,
    toggleAttribute(_name, on) {
      this.disabled = on;
    },
  });
  const elements = new Map([
    ['analyze', element()],
    ['message', element()],
  ]);
  const deps = {
    automaticAnalysisAllowed: () => true,
    activeEngineRows: () => [smallLc0 ? { family: 'lc0', variant: 'small' } : { family: 'sf', variant: 'lite' }],
    isLc0BigNetVariant: () => false,
    inputEl: (id) => (assert.equal(id, 'autoAnalyze'), { checked: autoAnalyze }),
    el: (id) => elements.get(id),
    selectEl: () => ({ disabled: true }),
    loadLc0Backend: async (runAutoAnalyze) => calls.push(`backend:${runAutoAnalyze}`),
    analyzeCurrent: async () => calls.push('analyze'),
    mountSignal: new AbortController().signal,
  };
  const body = stripTypeScriptTypes(`async function run(){${tail}\n}`);
  const run = new Function(...Object.keys(deps), `return (${body})();`);
  return run(...Object.values(deps)).then(() => ({ calls, analyzeDisabled: elements.get('analyze').disabled }));
}

test('a saved setup without Lc0 Small honours Auto-analyze turned off during startup checks', async () => {
  assert.ok(tailStart > initStart, 'init() startup tail found');
  const off = await startup({ autoAnalyze: false, smallLc0: false });
  assert.deepEqual(off.calls, [], 'no automatic request');
  assert.equal(off.analyzeDisabled, false, 'manual Analyze is enabled');
  const on = await startup({ autoAnalyze: true, smallLc0: false });
  assert.deepEqual(on.calls, ['analyze']);
});

test('a saved setup with Lc0 Small still loads its backend; the loader applies the checkbox', async () => {
  for (const autoAnalyze of [false, true]) assert.deepEqual((await startup({ autoAnalyze, smallLc0: true })).calls, ['backend:true']);
});
