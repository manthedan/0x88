import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { createBrowserDom } from '../src/lc0/browserDom.ts';

class Root {
  id = 'main';
  nodes = new Map();
  queries = 0;
  contains(node) {
    return node === this || [...this.nodes.values()].includes(node);
  }
  querySelector(selector) {
    this.queries++;
    const id = JSON.parse(selector.slice(4, -1));
    return this.nodes.get(id) ?? null;
  }
}

test('route lookups survive document detachment and cannot write to the incoming route', (t) => {
  const root = new Root(),
    oldMessage = { textContent: '' },
    nextMessage = { textContent: 'incoming' };
  root.nodes.set('message', oldMessage);
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'document');
  Object.defineProperty(globalThis, 'document', { configurable: true, value: { getElementById: () => nextMessage } });
  t.after(() => (previous ? Object.defineProperty(globalThis, 'document', previous) : delete globalThis.document));
  const dom = createBrowserDom(),
    release = dom.mount(root);
  assert.equal(dom.el('main'), root);
  assert.equal(dom.el('message'), oldMessage);
  const queries = root.queries;
  // Svelte removes the root from document before invoking teardown. The scope
  // deliberately does not use document.getElementById or node.isConnected.
  dom.el('message').textContent = 'retired';
  assert.equal(root.queries, queries, 'an unchanged host uses the cache');
  assert.equal(nextMessage.textContent, 'incoming');
  assert.equal(oldMessage.textContent, 'retired');
  release();
  assert.equal(dom.maybeEl('message'), null);
  assert.throws(() => dom.el('message'), /Missing #message/);
});

test('dynamic replacement and retry placeholders replace stale cached hosts', () => {
  const root = new Root(),
    dom = createBrowserDom();
  const release = dom.mount(root),
    original = {},
    placeholder = {};
  root.nodes.set('humanPanel', original);
  assert.equal(dom.el('humanPanel'), original);
  root.nodes.set('humanPanel', placeholder);
  assert.equal(dom.el('humanPanel'), placeholder);
  root.nodes.set('humanPanel', original);
  assert.equal(dom.el('humanPanel'), original);
  assert.equal(dom.maybeEl('modelLoadProgress'), null);
  const generated = {};
  root.nodes.set('modelLoadProgress', generated);
  assert.equal(dom.el('modelLoadProgress'), generated, 'misses must not be cached');
  release();
});

test('scopes are independent and an old cleanup cannot release a remounted root', () => {
  const a = createBrowserDom(),
    b = createBrowserDom(),
    first = new Root(),
    next = new Root();
  first.nodes.set('message', {});
  next.nodes.set('message', {});
  const releaseOld = a.mount(first),
    releaseB = b.mount(next),
    releaseNew = a.mount(next);
  releaseOld();
  assert.equal(a.el('message'), next.nodes.get('message'));
  releaseNew();
  assert.equal(a.maybeEl('message'), null);
  assert.equal(b.el('message'), next.nodes.get('message'));
  releaseB();
  assert.throws(() => a.mount(null), /Missing #main/);
});

for (const controller of ['play', 'analysis', 'arena'])
  test(`${controller} captures its root at mount and releases it after synchronous teardown`, () => {
    const text = readFileSync(new URL(`../src/lc0/${controller}Browser.ts`, import.meta.url), 'utf8');
    assert.match(text, /const pageDom = createBrowserDom\(\)/);
    assert.doesNotMatch(text, /import\s*\{[^}]*\bel\b[^}]*\}\s*from '\.\/browserDom.ts'/);
    const mount = text.slice(text.indexOf(`export function mount${controller[0].toUpperCase() + controller.slice(1)}Browser`));
    assert.match(mount, /const releaseDom = pageDom.mount\(/);
    assert.match(mount, /finally\s*\{\s*releaseDom\(\)/);
    assert.deepEqual(
      [...text.matchAll(/document\.getElementById\('([^']+)'\)/g)].map((match) => match[1]),
      ['main'],
      'only mount may look up a route element in the document',
    );
  });
