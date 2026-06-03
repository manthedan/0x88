import assert from 'node:assert/strict';
import { test } from 'node:test';
import { bestMoveShapes, searchShapes } from '../src/lc0/boardArrows.ts';

test('bestMoveShapes returns one green arrow for a UCI move', () => {
  assert.deepEqual(bestMoveShapes('d2d4'), [{ orig: 'd2', dest: 'd4', brush: 'green' }]);
});

test('bestMoveShapes ignores the promotion suffix and rejects junk', () => {
  assert.deepEqual(bestMoveShapes('a7a8q'), [{ orig: 'a7', dest: 'a8', brush: 'green' }]);
  assert.deepEqual(bestMoveShapes(undefined), []);
  assert.deepEqual(bestMoveShapes('e2'), []);
});

test('searchShapes draws the best move green and other MultiPV roots blue', () => {
  const shapes = searchShapes('d2d4', [['d2d4', 'd7d5'], ['g1f3', 'g8f6'], ['c2c4']]);
  assert.deepEqual(shapes, [
    { orig: 'd2', dest: 'd4', brush: 'green' },
    { orig: 'g1', dest: 'f3', brush: 'blue' },
    { orig: 'c2', dest: 'c4', brush: 'blue' },
  ]);
});

test('searchShapes does not duplicate the best move as a blue alternative', () => {
  const shapes = searchShapes('d2d4', [['d2d4', 'd7d5'], ['d2d4']]);
  assert.deepEqual(shapes, [{ orig: 'd2', dest: 'd4', brush: 'green' }]);
});

test('searchShapes with no multiPv is just the best move', () => {
  assert.deepEqual(searchShapes('e2e4'), [{ orig: 'e2', dest: 'e4', brush: 'green' }]);
  assert.deepEqual(searchShapes(undefined), []);
});
