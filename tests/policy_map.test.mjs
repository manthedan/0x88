import assert from 'node:assert/strict';
import { test } from 'node:test';
import { POLICY_INDEX, POLICY_MAP, POLICY_MOVES, POLICY_SIZE, moveToPolicyIndex } from '../src/chess/policyMap.ts';
import { moveFromUci } from '../src/chess/moveCodec.ts';

test('fixed policy map is stable and covers common/promotional moves', () => {
  assert.equal(POLICY_MAP, 'uci_queen_knight_promo_v1');
  assert.equal(POLICY_SIZE, 1968);
  for (const uci of ['e2e4', 'g1f3', 'e1g1', 'e7e8q', 'a2b1n', 'h7h8r']) {
    assert.equal(typeof POLICY_INDEX.get(uci), 'number', uci);
    assert.equal(moveToPolicyIndex(moveFromUci(uci)), POLICY_INDEX.get(uci));
  }
  assert.deepEqual([...POLICY_MOVES].sort(), POLICY_MOVES);
});
