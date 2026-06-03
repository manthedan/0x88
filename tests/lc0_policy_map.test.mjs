import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import {
  LC0_FLIP_TRANSFORM,
  LC0_MIRROR_TRANSFORM,
  LC0_POLICY_INDEX,
  LC0_POLICY_MAP,
  LC0_POLICY_MOVES,
  LC0_POLICY_SIZE,
  LC0_TRANSPOSE_TRANSFORM,
  lc0PolicyIndexToUci,
  moveToLc0PolicyIndex,
  transformPolicyUci,
  uciToLc0PolicyIndex,
} from '../src/lc0/policyMap.ts';
import { moveFromUci } from '../src/chess/moveCodec.ts';

function lc0SourceMoveStrings() {
  const source = readFileSync('../repos/lc0/src/neural/encoder.cc', 'utf8');
  const match = source.match(/const char\* kMoveStrs\[\]\s*=\s*\{([\s\S]*?)\};/);
  assert.ok(match, 'kMoveStrs[] exists in LC0 encoder.cc');
  return [...match[1].matchAll(/"([a-h][1-8][a-h][1-8][qrb]?)"/g)].map((m) => m[1]);
}

test('LC0 generated policy table matches encoder.cc kMoveStrs', () => {
  assert.equal(LC0_POLICY_MAP, 'lc0_1858_from_encoder_cc');
  assert.equal(LC0_POLICY_SIZE, 1858);
  assert.deepEqual([...LC0_POLICY_MOVES], lc0SourceMoveStrings());
});

test('LC0 policy map exposes known startpos prior indices', () => {
  assert.equal(uciToLc0PolicyIndex('d2d4'), 293);
  assert.equal(uciToLc0PolicyIndex('g1f3'), 159);
  assert.equal(uciToLc0PolicyIndex('e2e4'), 322);
  assert.equal(uciToLc0PolicyIndex('c2c4'), 264);
  assert.equal(uciToLc0PolicyIndex('g2g3'), 374);
  assert.equal(lc0PolicyIndexToUci(293), 'd2d4');
  assert.equal(moveToLc0PolicyIndex(moveFromUci('d2d4')), 293);
});

test('LC0 policy table raw index roundtrips every generated move', () => {
  assert.equal(new Set(LC0_POLICY_MOVES).size, LC0_POLICY_SIZE);
  for (let i = 0; i < LC0_POLICY_MOVES.length; i++) {
    const uci = LC0_POLICY_MOVES[i];
    assert.equal(LC0_POLICY_INDEX.get(uci), i, uci);
    assert.equal(lc0PolicyIndexToUci(i), uci, uci);
  }
});

test('LC0 policy handles knight promotions like LC0 MoveToNNIndex packing', () => {
  assert.equal(LC0_POLICY_INDEX.has('a7a8n'), false);
  assert.equal(uciToLc0PolicyIndex('a7a8n'), uciToLc0PolicyIndex('a7a8'));
  assert.equal(uciToLc0PolicyIndex('a7b8n'), uciToLc0PolicyIndex('a7b8'));
  assert.equal(uciToLc0PolicyIndex('h7h8n'), uciToLc0PolicyIndex('h7h8'));
  assert.equal(uciToLc0PolicyIndex('a7a8q'), 1792);
});

test('LC0 policy castling normalization is explicit because raw UCI is ambiguous', () => {
  assert.equal(uciToLc0PolicyIndex('e1g1'), LC0_POLICY_INDEX.get('e1g1'));
  assert.equal(uciToLc0PolicyIndex('e1g1', 0, { standardCastling: true }), LC0_POLICY_INDEX.get('e1h1'));
  assert.equal(uciToLc0PolicyIndex('e1c1', 0, { standardCastling: true }), LC0_POLICY_INDEX.get('e1a1'));
  assert.equal(uciToLc0PolicyIndex('e8g8', 0, { standardCastling: true }), LC0_POLICY_INDEX.get('e8h8'));
  assert.equal(uciToLc0PolicyIndex('e8c8', 0, { standardCastling: true }), LC0_POLICY_INDEX.get('e8a8'));
  assert.equal(lc0PolicyIndexToUci(LC0_POLICY_INDEX.get('e1h1'), 0, { standardCastling: true }), 'e1g1');
});

test('LC0 policy transforms mirror encoder.cc square transform and invert selected moves', () => {
  const transforms = [
    0,
    LC0_FLIP_TRANSFORM,
    LC0_MIRROR_TRANSFORM,
    LC0_FLIP_TRANSFORM | LC0_MIRROR_TRANSFORM,
    LC0_TRANSPOSE_TRANSFORM,
    LC0_TRANSPOSE_TRANSFORM | LC0_FLIP_TRANSFORM,
    LC0_TRANSPOSE_TRANSFORM | LC0_MIRROR_TRANSFORM,
    LC0_TRANSPOSE_TRANSFORM | LC0_FLIP_TRANSFORM | LC0_MIRROR_TRANSFORM,
  ];
  for (const transform of transforms) {
    const idx = uciToLc0PolicyIndex('d2d4', transform);
    assert.equal(typeof idx, 'number', `d2d4 transform ${transform}`);
    assert.equal(lc0PolicyIndexToUci(idx, transform), 'd2d4');
  }
  assert.equal(transformPolicyUci('d2d4', LC0_FLIP_TRANSFORM), 'e2e4');
  assert.equal(transformPolicyUci('d2d4', LC0_MIRROR_TRANSFORM), 'd7d5');
  assert.equal(transformPolicyUci('d2d4', LC0_TRANSPOSE_TRANSFORM), 'e7e5');
});
