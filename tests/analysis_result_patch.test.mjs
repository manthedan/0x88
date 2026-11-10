import assert from 'node:assert/strict';
import test from 'node:test';
import { patchAnalysisResults } from '../src/lc0/analysisPresentation.ts';

function fixture({ focus = false, oldKey = 'fen|sf|1', newKey = oldKey, left = 10, right = 60 } = {}) {
  let markup = 'before';
  const oldScroll = { dataset: { resultScrollKey: oldKey }, scrollLeft: 280 };
  const newScroll = { dataset: { resultScrollKey: newKey }, scrollLeft: 0, getBoundingClientRect: () => ({ left: 0, right: 100 }) };
  const oldTarget = { dataset: { focusKey: 'move|8' } };
  let focusOptions;
  const newTarget = {
    dataset: oldTarget.dataset,
    focus: (options) => {
      focusOptions = options;
    },
    closest: () => newScroll,
    getBoundingClientRect: () => ({ left, right }),
  };
  const container = {
    get innerHTML() {
      return markup;
    },
    set innerHTML(value) {
      markup = value;
    },
    contains: (node) => node === oldTarget,
    querySelectorAll: (selector) => {
      if (selector === '[data-result-scroll-key]') return [markup === 'before' ? oldScroll : newScroll];
      if (selector === '[data-focus-key]') return [newTarget];
      return [];
    },
  };
  const previousDocument = globalThis.document;
  globalThis.document = { activeElement: focus ? oldTarget : null };
  try {
    patchAnalysisResults(container, 'after');
  } finally {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
  return { newScroll, focusOptions };
}

test('live result updates retain inline PV scroll by position, engine and line', () => {
  assert.equal(fixture().newScroll.scrollLeft, 280);
  assert.equal(fixture({ newKey: 'different-fen|sf|1' }).newScroll.scrollLeft, 0);
});

test('restored PV move focus stays visible without scrolling the entire page', () => {
  const visible = fixture({ focus: true });
  assert.deepEqual(visible.focusOptions, { preventScroll: true });
  assert.equal(visible.newScroll.scrollLeft, 280);
  assert.equal(fixture({ focus: true, left: -20, right: 30 }).newScroll.scrollLeft, 260);
  assert.equal(fixture({ focus: true, left: 80, right: 130 }).newScroll.scrollLeft, 310);
});
