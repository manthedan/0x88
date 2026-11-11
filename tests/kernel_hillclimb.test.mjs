import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import test from 'node:test';
import { CANDIDATES, WGSL_HEADS_GATE } from '../scripts/kernel_hillclimb/candidates.mjs';
import { classify, devicePeaks, MATMUL_RE, matchMatmuls, PATHS } from '../scripts/kernel_hillclimb/roofline.mjs';
import { gemmKernel, subgroupMatrixGemmKernel } from '../lab/kernel-hillclimb/wgsl-kernels.mjs';

// The lab modules read location.search at import time.
globalThis.location ??= { search: '' };
const { aggregateProfile } = await import('../lab/kernel-hillclimb/gpu-profiler.mjs');
const { summarize } = await import('../lab/kernel-hillclimb/common.mjs');

const peaks = { flops: 3e12, bw: 100e9, ridge: 30, gemmCeilingPlain: 2e12 };

test('every candidate points at a lab page and declares a known gate kind', () => {
  for (const [name, c] of Object.entries(CANDIDATES)) {
    assert.ok(existsSync(`lab/kernel-hillclimb/${c.page}.html`), `${name}: missing page ${c.page}`);
    assert.ok(['in-page', 'command', 'none'].includes(c.gate.kind), `${name}: gate kind ${c.gate.kind}`);
    if (c.gate.kind === 'in-page') assert.match(c.gate.query, /^parity=/, name);
    if (c.gate.command) assert.ok(c.gate.command.includes('--port') && !c.gate.command.includes('5291'), `${name}: command gate must pin an allowed port`);
  }
  assert.equal(WGSL_HEADS_GATE.kind, 'command');
  for (const name of Object.keys(PATHS)) assert.ok(CANDIDATES[name], `roofline path ${name} has no candidate`);
});

test('classify separates launch, compute, bandwidth and underused kernels', () => {
  assert.equal(classify({ meanUs: 3, achievedFlops: 0, achievedBw: 1e9, ai: 0 }, peaks).bound, 'launch');
  assert.equal(classify({ meanUs: 500, achievedFlops: 2e12, achievedBw: 10e9, ai: 200 }, peaks).bound, 'compute');
  assert.equal(classify({ meanUs: 500, achievedFlops: 0, achievedBw: 80e9, ai: 0 }, peaks).bound, 'bandwidth');
  assert.equal(classify({ meanUs: 300, achievedFlops: 3e9, achievedBw: 4e9, ai: 1 }, peaks).bound, 'underused');
});

test('matchMatmuls pairs dispatches with ONNX matmuls by bound bytes, without reuse', () => {
  const nodes = [
    { op: 'MatMul', flops: 100, bytes: 1000 },
    { op: 'MatMul', flops: 200, bytes: 4000 },
    { op: 'Add', flops: 5, bytes: 1000 },
  ];
  const { matched, unmatchedNodes } = matchMatmuls(
    [
      { i: 0, kname: 'Gemm', bindBytes: 4100 },
      { i: 1, kname: 'MatMul', bindBytes: 1010 },
    ],
    nodes,
  );
  assert.deepEqual(
    matched.map((m) => m.node.flops),
    [200, 100],
  );
  assert.equal(unmatchedNodes, 0);
  assert.ok(MATMUL_RE.test('fused_matmul2_add1_kernel') && !MATMUL_RE.test('Softmax'));
});

test('aggregateProfile reports busy, envelope and gap per invoke', () => {
  const rows = [
    { invoke: 1, name: 'A', ns: 1000, beginNs: 0, endNs: 1000, workgroups: [1, 1, 1], bindBytes: 8, index: 0 },
    { invoke: 1, name: 'B', ns: 2500, beginNs: 1500, endNs: 4000, workgroups: [2, 1, 1], bindBytes: 8, index: 1 },
    { invoke: 2, name: 'A', ns: 1000, beginNs: 10000, endNs: 11000, workgroups: [1, 1, 1], bindBytes: 8, index: 2 },
  ];
  const agg = aggregateProfile({ rows }, { invokes: 2 });
  assert.equal(agg.perInvoke[0].busyMs, 0.0035);
  assert.equal(agg.perInvoke[0].envelopeMs, 0.004);
  assert.ok(Math.abs(agg.perInvoke[0].gapMs - 0.0005) < 1e-12);
  assert.equal(agg.kernels[0].name, 'B');
  assert.equal(agg.kernels.find((k) => k.name === 'A').dispatchesPerInvoke, 1);
});

test('summarize flags noisy samples with the perf canary MAD rule', () => {
  assert.equal(summarize([10, 10, 10, 10.1, 9.9]).noisy, false);
  assert.equal(summarize([10, 14, 7, 12, 9]).noisy, true);
});

test('devicePeaks takes medians over repeats and splits plain vs subgroup-matrix GEMM', () => {
  const run = (f, sg) => ({
    peaks: { fmaF32Tflops: f, fmaF16Tflops: f - 0.5, readGBps: 90, copyGBps: 80, bestGemmTflops: { 'square 2048x2048x2048': sg } },
    results: { gemm: [{ shape: 'square 2048x2048x2048', tflops: 1.4 }], subgroupMatrix: { rows: [{ shape: 'square 2048x2048x2048', tflops: sg }] } },
  });
  const p = devicePeaks({ results: [run(3.2, 2.9), run(3.4, 3.0), run(3.0, 2.8)] });
  assert.equal(p.flops, 3.2e12);
  assert.equal(p.bw, 90e9);
  assert.equal(p.gemmByShape['square 2048x2048x2048'].plainTflops, 1.4);
  assert.equal(p.gemmByShape['square 2048x2048x2048'].sgmatrixTflops, 2.9);
});

test('GEMM generators reject non-tiling shapes and size the grid', () => {
  assert.deepEqual(gemmKernel({ M: 512, N: 1024, K: 1024, storage: 'f16', accum: 'f32', tm4: 1, tn4: 2 }).dispatch, [8, 8, 1]);
  assert.throws(() => gemmKernel({ M: 100, N: 64, K: 16, storage: 'f32', accum: 'f32', tm4: 1, tn4: 1 }));
  const sg = subgroupMatrixGemmKernel({ M: 64, N: 64, K: 64, type: 'f16', resultType: 'f16', mm: 8, nn: 8, kk: 8, sg: 32, rt: 4, ct: 4 });
  assert.deepEqual(sg.dispatch, [2, 2, 1]);
  assert.match(sg.code, /@workgroup_size\(32\)/);
});
