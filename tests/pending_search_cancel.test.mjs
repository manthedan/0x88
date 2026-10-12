import assert from 'node:assert/strict';
import test from 'node:test';
import { noteSearchCancel, takeSearchCancel } from '../src/lc0/pendingSearchCancel.ts';

test('a cancel that arrives before the search starts is consumed by that search', () => {
  const pending = [];
  noteSearchCancel(pending, 4);
  assert.equal(takeSearchCancel(pending, 4), true);
  assert.deepEqual(pending, []);
  assert.equal(takeSearchCancel(pending, 4), false);
});

test('a cancel for a different id does not stop the search that is starting', () => {
  const pending = [];
  noteSearchCancel(pending, 4);
  assert.equal(takeSearchCancel(pending, 5), false);
  assert.deepEqual(pending, [4]);
});

test('duplicate cancels are stored once and the pending list stays bounded', () => {
  const pending = [];
  noteSearchCancel(pending, 1);
  noteSearchCancel(pending, 1);
  assert.deepEqual(pending, [1]);
  for (let id = 2; id <= 40; id += 1) noteSearchCancel(pending, id);
  assert.equal(pending.length, 32);
  assert.equal(pending[0], 9);
  assert.equal(pending.at(-1), 40);
});
