import assert from 'node:assert/strict';
import { test } from 'node:test';
import { START_FEN, parseFen } from '../src/chess/board.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';

test('SquareFormer evaluator requires named policy and WDL outputs', async () => {
  const meta = {
    kind: 'squareformer_v2',
    input_dim: 35,
    input_format: 'float_onehot_rules',
    policy_size: 20480,
    history_plies: 2,
  };
  const session = { run: async () => ({ first: { data: new Float32Array(20480) }, wdl: { data: new Float32Array([0, 0, 0]) } }) };
  const evaluator = new SquareFormerEvaluator(session, meta);
  await assert.rejects(() => evaluator.evaluate(parseFen(START_FEN)), /missing required tensor 'policy'/);
});
