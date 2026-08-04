import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const kernelRoot = new URL('../public/runtimes/squareformer-tvm-hybrid/bt4-anneal-muon-best/v1/squareformer-ffn/', import.meta.url);
const summary = JSON.parse(readFileSync(new URL('summary.json', kernelRoot), 'utf8'));
const expectedLabels = [
  'attn_qkv_64x128x384',
  'attn_out_64x128x128',
  'ffn_dense1_64x128x256',
  'ffn_dense2_64x256x128',
  'policy_proj_64x128x256',
  'policy_pair_proj_64x128x128',
  'ffn_dense1_gelu_64x128x256',
  'ffn_dense2_residual_64x256x128',
  'attn_out_residual_64x128x128',
];

test('SquareFormer TVM kernel manifest covers every staged WGSL variant', () => {
  assert.deepEqual(
    summary.map((entry) => entry.label),
    expectedLabels,
  );
  for (const entry of summary) {
    assert.equal(entry.m, 64);
    assert.ok(entry.k > 0 && entry.n > 0);
    assert.ok(entry.dispatch.x > 0 && entry.dispatch.y > 0 && entry.dispatch.z > 0);
    assert.deepEqual(entry.bindings.slice(0, 5), ['output', 'w', 'x', 'podArgs', 'b']);
    assert.equal(entry.bindings.includes('skip'), entry.epilogue === 'bias-residual');
  }
});

test('every staged TVM WGSL source matches manifest size and entry point contract', () => {
  for (const entry of summary) {
    const source = readFileSync(new URL(entry.file, kernelRoot));
    assert.equal(source.byteLength, entry.bytes, `${entry.file} byte size`);
    const text = source.toString('utf8');
    assert.match(text, /@compute\s+@workgroup_size\(/, `${entry.file} compute annotation`);
    assert.match(text, /fn\s+matmul_kernel\b/, `${entry.file} entry point`);
    assert.match(text, /var<storage,\s*read_write>/, `${entry.file} output storage binding`);
  }
});
