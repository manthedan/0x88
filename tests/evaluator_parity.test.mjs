import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';

test('browser-style bytes evaluator matches Node path evaluator', { skip: !existsSync('public/models/chessformer_v1_100m_e3_single.onnx') }, () => {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/evaluator_byte_vs_path_parity.mjs'], { encoding: 'utf8' });
  assert.match(output, /METRIC evaluator_path_byte_parity_positions=4/);
  assert.match(output, /METRIC evaluator_path_byte_parity_max_policy_diff=0/);
});

test('CNN and SquareFormer adapters share legal action-id policy contract', () => {
  const output = execFileSync('node', ['--experimental-strip-types', 'eval/adapter_policy_key_parity_check.mjs'], { encoding: 'utf8' });
  assert.match(output, /METRIC adapter_policy_key_structural_positions=4/);
});
