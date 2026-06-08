import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import test from 'node:test';

const generatedUrl = new URL('../src/lc0/generated/tvmPackedF16Wgsl.ts', import.meta.url);

test('LC0 TVM packed-f16 WGSL generated module is fresh', () => {
  const result = spawnSync(process.execPath, ['scripts/generate_lc0_tvm_wgsl_kernels.mjs', '--check'], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8',
  });
  assert.equal(result.status, 0, result.stderr || result.stdout);
});

test('LC0 TVM packed-f16 WGSL generated metadata covers all kernels', () => {
  const source = readFileSync(generatedUrl, 'utf8');
  const names = [...source.matchAll(/"name": "([A-Z0-9_]+)"/g)].map((match) => match[1]);
  assert.deepEqual(names, [
    'ATTENTION_BLOCK_QKV_TVM_PACKED_F16_WGSL',
    'ATTENTION_OUTPUT_PROJ_TVM_PACKED_F16_WGSL',
    'FFN_DENSE1_TVM_PACKED_F16_WGSL',
    'FFN_DENSE2_TVM_PACKED_F16_WGSL',
  ]);
  assert.equal([...source.matchAll(/"sha256": "[0-9a-f]{64}"/g)].length, names.length);
  assert.equal([...source.matchAll(/"bytes": [1-9][0-9]*/g)].length, names.length);
});
