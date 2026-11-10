import assert from 'node:assert/strict';
import { test } from 'node:test';
import { transitionPendingInput } from '../src/lc0/pendingInput.ts';

const url = { urlKeys: ['fen', 'pgn'], transfers: [] };
const handoff = { urlKeys: [], transfers: [{ key: 'handoff', id: 'captured-transfer' }] };
const both = { ...url, transfers: handoff.transfers };

// The table is the lifecycle contract; failures keep the input reloadable at any age.
for (const [name, before, event, pending, acknowledged] of [
  ['apply URL', null, { type: 'applied', input: url }, url, null],
  ['apply handoff', null, { type: 'applied', input: handoff }, handoff, null],
  ['supersede handoff with URL', handoff, { type: 'applied', input: url }, both, null],
  ['failed save', both, { type: 'save-failed' }, both, null],
  ['failed clear', both, { type: 'clear-failed' }, both, null],
  ['successful save', both, { type: 'saved' }, null, both],
  ['durably empty workspace', both, { type: 'cleared' }, null, both],
  ['explicit discard or replacement', both, { type: 'discarded' }, null, both],
  ['save without input', null, { type: 'saved' }, null, null],
  ['clear without input', null, { type: 'cleared' }, null, null],
  ['discard without input', null, { type: 'discarded' }, null, null],
  [
    'capture newer transfer too',
    handoff,
    { type: 'applied', input: { urlKeys: [], transfers: [{ key: 'handoff', id: 'newer' }] } },
    { urlKeys: [], transfers: [...handoff.transfers, { key: 'handoff', id: 'newer' }] },
    null,
  ],
]) {
  test(`pending input transition: ${name}`, () => {
    const original = JSON.stringify({ before, event });
    assert.deepEqual(transitionPendingInput(before, event), { pending, acknowledged });
    assert.equal(JSON.stringify({ before, event }), original, 'Transitions must not mutate their inputs');
  });
}

test('later success acknowledges once after any number of failures', () => {
  let pending = both;
  for (let i = 0; i < 3; i++) pending = transitionPendingInput(pending, { type: 'save-failed' }).pending;
  const saved = transitionPendingInput(pending, { type: 'saved' });
  assert.deepEqual(saved.acknowledged, both);
  assert.equal(transitionPendingInput(saved.pending, { type: 'cleared' }).acknowledged, null);
});
