import assert from 'node:assert/strict';
import { test } from 'node:test';
import { classifyChunkLoadError, showRouteControllerFailure } from '../src/lib/chunkLoadError.ts';

test('a missing route chunk has a working reload action without blaming a deploy', (t) => {
  const reload = t.mock.fn();
  const location = Object.getOwnPropertyDescriptor(globalThis, 'location');
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { reload } });
  t.after(() => location ? Object.defineProperty(globalThis, 'location', location) : delete globalThis.location);
  const button = {};
  const node = { ownerDocument: { createElement: () => button }, append: t.mock.fn() };
  showRouteControllerFailure(node, new TypeError('Importing a module script failed.'));
  assert.match(node.textContent, /Reload this page/);
  assert.doesNotMatch(node.textContent, /updated/);
  assert.deepEqual(node.append.mock.calls[0].arguments, [' ', button]);
  assert.equal(button.textContent, 'Reload page');
  button.onclick();
  assert.equal(reload.mock.callCount(), 1);

  showRouteControllerFailure(node, new Error('bad controller state'));
  assert.equal(node.textContent, 'Page failed to initialize: bad controller state');
  assert.equal(node.append.mock.callCount(), 1);
});

test('offline is reported for chunk failures but not unrelated exceptions', (t) => {
  Object.defineProperty(navigator, 'onLine', { configurable: true, value: false });
  t.after(() => delete navigator.onLine);
  const error = new TypeError('error loading dynamically imported module');
  assert.equal(classifyChunkLoadError(error).kind, 'offline');
  assert.equal(classifyChunkLoadError(new Error('Unsupported engine')), undefined);
});
