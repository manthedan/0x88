import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  POLICY_INDEX,
  actionIdToMove,
  assertCanonicalMoveEncoding,
  chessBenchAvClassToMove,
  moveFromUci,
  moveToActionId,
  moveToChessBenchAvClass,
  moveToResidualPolicyIndex,
  moveToSquareformerPolicyIndex,
  moveToUci,
} from '../src/chess/moveEncodings.ts';

const promos = [undefined, 'n', 'b', 'r', 'q'];

function uciOf(move) { return moveToUci(move); }

test('canonical action-id mapping roundtrips every source/destination/promotion candidate', () => {
  for (let from = 0; from < 64; from++) {
    for (let to = 0; to < 64; to++) {
      for (const promotion of promos) {
        const move = promotion ? { from, to, promotion } : { from, to };
        assert.equal(uciOf(actionIdToMove(moveToActionId(move))), uciOf(move));
      }
    }
  }
});

test('ChessBench and SquareFormer AV encodings are the same compact-20480 map', () => {
  for (const uci of ['a1a8', 'h8a1', 'e2e4', 'e7e8q', 'e7e8r', 'e7e8b', 'e7e8n', 'a2b1q', 'h7h8n']) {
    const move = moveFromUci(uci);
    assert.equal(moveToSquareformerPolicyIndex(move), moveToChessBenchAvClass(move));
    assert.equal(uciOf(chessBenchAvClassToMove(moveToChessBenchAvClass(move))), uci);
    assertCanonicalMoveEncoding(move);
  }
});

test('residual policy helper matches policy index map', () => {
  for (const [uci, index] of POLICY_INDEX.entries()) {
    assert.equal(moveToResidualPolicyIndex(moveFromUci(uci)), index, uci);
  }
});
