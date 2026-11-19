import assert from 'node:assert/strict';
import test from 'node:test';
import { decodeMaia3Value, maia3WinProbability } from '../src/lc0/maia3.ts';

// Maia3's logits_value is [Loss, Draw, Win] (upstream processOutputsMaia3), the reverse of LC0's [W, D, L].
const LOSING_ROW = Float32Array.of(4, 0, -4);

test('decodeMaia3Value reads the model row as Loss, Draw, Win', () => {
  const { logits, probabilities } = decodeMaia3Value(LOSING_ROW);
  assert.deepEqual(logits, { win: -4, draw: 0, loss: 4 });
  assert.ok(probabilities.loss > 0.98 && probabilities.win < 0.001, JSON.stringify(probabilities));
  assert.ok(Math.abs(probabilities.win + probabilities.draw + probabilities.loss - 1) < 1e-12);
});

test('decoded value is named fields, so callers cannot index it as W,D,L', () => {
  const { probabilities } = decodeMaia3Value(LOSING_ROW);
  assert.equal(Array.isArray(probabilities), false);
  assert.equal(probabilities[0], undefined);
});

test('maia3WinProbability is the side-to-move score flipped to White', () => {
  const evaluation = (side) => ({ fen: `8/8/8/8/8/8/8/K6k ${side} - - 0 1`, valueProbabilities: decodeMaia3Value(LOSING_ROW).probabilities });
  assert.ok(maia3WinProbability(evaluation('w')) < 0.02);
  assert.ok(maia3WinProbability(evaluation('b')) > 0.98);
});

test('a malformed value row decodes to NaN and scores as even', () => {
  const { probabilities } = decodeMaia3Value([1, 2]);
  assert.ok(Number.isNaN(probabilities.win));
  // softmax would turn a non-finite logit into a plausible uniform 1/3 split
  for (const row of [
    [Number.NaN, 0, 0],
    [0, Number.POSITIVE_INFINITY, 0],
  ]) {
    const decoded = decodeMaia3Value(row).probabilities;
    assert.ok([decoded.win, decoded.draw, decoded.loss].every(Number.isNaN), JSON.stringify(decoded));
  }
  assert.equal(maia3WinProbability({ fen: '8/8/8/8/8/8/8/K6k w - - 0 1', valueProbabilities: probabilities }), 0.5);
});
