import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import test from 'node:test';
import { compareCentipawn, compareLc0, ortParity, profileComplete } from '../lab/kernel-hillclimb/gates.mjs';
import { fmaKernel, gemmKernel, subgroupMatrixGemmKernel } from '../lab/kernel-hillclimb/wgsl-kernels.mjs';
import { CANDIDATES, WGSL_HEADS_GATE } from '../scripts/kernel_hillclimb/candidates.mjs';
import { analyzeCandidate, classify, devicePeaks, gridConsistent, MATMUL_RE, matchMatmuls, PATHS } from '../scripts/kernel_hillclimb/roofline.mjs';

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

test('matchMatmuls attaches FLOPs by bytes and grid shape', () => {
  const nodes = [
    { op: 'MatMul', flops: 100, bytes: 1000, mnk: { batch: 1, M: 8, N: 32, K: 8 } },
    { op: 'MatMul', flops: 200, bytes: 4000, mnk: { batch: 1, M: 64, N: 64, K: 8 } },
    { op: 'Add', flops: 5, bytes: 1000 },
  ];
  const { matched } = matchMatmuls(
    [
      { i: 0, kname: 'Gemm', wg: '2x2x1', bindBytes: 4100 },
      { i: 1, kname: 'MatMul', wg: '1x1x1', bindBytes: 1010 },
    ],
    nodes,
    { absTol: 0 },
  );
  assert.deepEqual(
    matched.map((m) => m.node.flops),
    [200, 100],
  );
  assert.ok(MATMUL_RE.test('fused_matmul2_add1_kernel') && !MATMUL_RE.test('Softmax'));
});

// Node classes and dispatch classes from the real BT4-it332 b8 graph / profile.
const bt4Node = (bytes, flops, batch, M, N, K, count) => Array.from({ length: count }, () => ({ op: 'MatMul', bytes, flops, mnk: { batch, M, N, K } }));
const BT4_NODES = [
  ...bt4Node(4194304, 1073741824, 1, 512, 1024, 1024, 63),
  ...bt4Node(4194304, 67108864, 256, 64, 64, 32, 15),
  ...bt4Node(4194304, 67108864, 256, 64, 32, 64, 15),
  ...bt4Node(4325376, 536870912, 8, 32, 4096, 256, 15),
  ...bt4Node(4329472, 33554432, 1, 8, 8192, 256, 15),
  ...bt4Node(5767168, 1610612736, 1, 512, 1536, 1024, 16),
];
const dispatchesOf = (spec) => spec.flatMap(([kname, wg, bindBytes, n]) => Array.from({ length: n }, () => ({ kname, wg, bindBytes })));

test('matchMatmuls does not give the BT4 attention batch-matmuls the 1-GFLOP projection FLOPs', () => {
  const ds = dispatchesOf([
    ['Gemm', '32x16x1', 4196432, 63],
    ['MatMul', '2x2x256', 4194432, 15],
    ['MatMul', '1x2x256', 4194432, 15],
    ['MatMul', '128x8x1', 4325504, 15],
    ['Gemm', '256x1x1', 4345936, 15],
    ['Gemm', '48x16x1', 6294608, 16],
  ]).map((d, i) => ({ ...d, i }));
  const { matched } = matchMatmuls(ds, BT4_NODES);
  const flopsOf = (kname, wg) => new Set(matched.filter((m) => m.kname === kname && m.wg === wg).map((m) => m.node?.flops ?? null));
  assert.deepEqual([...flopsOf('Gemm', '32x16x1')], [1073741824]);
  assert.deepEqual([...flopsOf('MatMul', '2x2x256')], [67108864]);
  assert.deepEqual([...flopsOf('MatMul', '1x2x256')], [67108864]);
  assert.deepEqual([...flopsOf('Gemm', '256x1x1')], [33554432]);
  assert.deepEqual([...flopsOf('Gemm', '48x16x1')], [1610612736]);
  // 128x8x1 is the batch-8 32x4096 matmul (rows 8*32 = 8 tiles of 32), not the
  // byte-identical 512x1024 projection (that one is 32x16 tiles).
  assert.deepEqual([...flopsOf('MatMul', '128x8x1')], [536870912]);
  // With only lenient tiles to go on, the dispatch count separates candidates.
  const lenient = matchMatmuls(dispatchesOf([['MatMul', '256x4x1', 4325504, 15]]), BT4_NODES);
  assert.equal(lenient.matched[0].node.flops, 536870912);
  assert.equal(lenient.matched[0].resolvedBy, 'bytes+grid+count');
  // No kernel may exceed the device FMA peak through a mis-attached FLOP count.
  for (const d of matched) if (d.node) assert.ok(d.node.flops / (d.bindBytes / 93e9) < 3.3e12 * 50, 'sanity');
});

test('matchMatmuls leaves an unresolvable byte collision unassigned instead of guessing', () => {
  const nodes = [...bt4Node(4194304, 1073741824, 1, 512, 1024, 1024, 4), ...bt4Node(4194304, 67108864, 1, 512, 1024, 64, 4)];
  const ds = dispatchesOf([['MatMul', '32x16x1', 4194304, 2]]).map((d, i) => ({ ...d, i }));
  const { matched, ambiguousClasses } = matchMatmuls(ds, nodes);
  assert.equal(matched[0].node, null);
  assert.equal(matched[0].ambiguous, true);
  assert.equal(ambiguousClasses.length, 1);
});

test('gridConsistent uses z as the batch and folds batch into rows otherwise', () => {
  assert.ok(gridConsistent('2x2x256', { batch: 256, M: 64, N: 64, K: 32 }));
  assert.ok(!gridConsistent('2x2x256', { batch: 1, M: 512, N: 1024, K: 1024 }));
  assert.ok(gridConsistent('128x8x1', { batch: 8, M: 32, N: 4096, K: 256 }));
  assert.ok(!gridConsistent('32x16x1', { batch: 1, M: 8, N: 8192, K: 256 }));
  assert.ok(gridConsistent('indirect', { batch: 1, M: 8, N: 8, K: 8 }));
});

// ---- broken-candidate gates: each must FAIL ----
const ref = { '/output/policy': Float32Array.from({ length: 64 }, (_, i) => Math.sin(i)), '/output/wdl': Float32Array.of(0.5, 0.3, 0.2) };
const limits = { '/output/policy': 0.06, '/output/wdl': 0.006 };

test('ORT parity passes the reference and fails scaled, NaN, truncated or missing outputs', () => {
  assert.equal(ortParity(ref, ref, limits).pass, true);
  const scaled = { '/output/policy': ref['/output/policy'].map((v) => v * 1.25), '/output/wdl': ref['/output/wdl'] };
  assert.deepEqual(ortParity(scaled, ref, limits).failures, ['/output/policy']);
  const nanWdl = { ...ref, '/output/wdl': Float32Array.of(Number.NaN, 0.3, 0.2) };
  assert.deepEqual(ortParity(nanWdl, ref, limits).failures, ['/output/wdl']);
  const short = { ...ref, '/output/wdl': Float32Array.of(0.5, 0.3) };
  assert.equal(ortParity(short, ref, limits).pass, false);
  assert.equal(ortParity({ '/output/policy': ref['/output/policy'] }, ref, limits).pass, false);
  assert.equal(ortParity(ref, ref, null).pass, false, 'no limits, no pass');
});

const lc0Row = (priors, wdl) => ({ legalPriors: Object.entries(priors).map(([uci, prior]) => ({ uci, prior })), wdl });
const lc0Ref = [lc0Row({ e2e4: 0.5, d2d4: 0.3, g1f3: 0.2 }, [0.4, 0.4, 0.2]), lc0Row({ a2a3: 0.302, b2b3: 0.298, c2c3: 0.4 }, [0.1, 0.8, 0.1])];
const lc0Limits = { tieEpsilon: 0.01, maxPriorAbs: 0.03, maxWdlAbs: 0.01 };

test('Lc0 parity passes near-identical rows and tolerates reference-distribution ties', () => {
  assert.equal(compareLc0(lc0Ref, lc0Ref, lc0Limits).pass, true);
  // c2c3 vs a2a3 is NOT a tie in the reference (gap 0.098), but a flip from
  // c2c3 to a2a3 within prior tolerance would need >0.03 movement; so flip the
  // genuine near-tie instead: second row's best stays c2c3; first row unchanged.
  const tieRef = [lc0Row({ e2e4: 0.401, d2d4: 0.399, g1f3: 0.2 }, [0.4, 0.4, 0.2])];
  const tieCand = [lc0Row({ e2e4: 0.398, d2d4: 0.402, g1f3: 0.2 }, [0.4, 0.4, 0.2])];
  const r = compareLc0(tieCand, tieRef, lc0Limits);
  assert.equal(r.pass, true);
  assert.equal(r.tieTolerated.length, 1);
});

test('Lc0 parity fails NaN WDL, shifted WDL, missing moves and flattened policies', () => {
  const nanWdl = [lc0Row({ e2e4: 0.5, d2d4: 0.3, g1f3: 0.2 }, [Number.NaN, 0.4, 0.2]), lc0Ref[1]];
  assert.equal(compareLc0(nanWdl, lc0Ref, lc0Limits).pass, false);
  const shifted = [lc0Row({ e2e4: 0.5, d2d4: 0.3, g1f3: 0.2 }, [0.5, 0.3, 0.2]), lc0Ref[1]];
  assert.equal(compareLc0(shifted, lc0Ref, lc0Limits).pass, false, 'WDL off by 0.1 must fail even with the same best move');
  const missing = [lc0Row({ e2e4: 0.5, d2d4: 0.5 }, [0.4, 0.4, 0.2]), lc0Ref[1]];
  assert.equal(compareLc0(missing, lc0Ref, lc0Limits).pass, false);
  // A flattened (near-uniform) candidate: its own top-2 gap is tiny, so a gap
  // measured in the candidate would excuse the flip; the reference gap (0.2) does not.
  const flat = [lc0Row({ e2e4: 0.333, d2d4: 0.334, g1f3: 0.333 }, [0.4, 0.4, 0.2]), lc0Ref[1]];
  const r = compareLc0(flat, lc0Ref, { ...lc0Limits, maxPriorAbs: 1 });
  assert.equal(r.pass, false);
  assert.equal(r.flips[0].referencePriorGap.toFixed(3), '0.200');
  assert.throws(() => compareLc0(lc0Ref, lc0Ref, { tieEpsilon: 0 }), /required/);
});

test('Centipawn parity fails NaN values and missing moves', () => {
  const row = (p, wdl) => ({ policy: new Map(Object.entries(p)), wdl });
  const refRows = [row({ e2e4: 0.6, d2d4: 0.4 }, [0.3, 0.4, 0.3])];
  assert.equal(compareCentipawn(refRows, refRows).pass, true);
  assert.equal(compareCentipawn([row({ e2e4: 0.6, d2d4: 0.4 }, [Number.NaN, 0.4, 0.3])], refRows).pass, false);
  assert.equal(compareCentipawn([row({ e2e4: 1 }, [0.3, 0.4, 0.3])], refRows).pass, false);
  assert.equal(compareCentipawn([row({ e2e4: 0.6, d2d4: 0.4, g1f3: 0 }, [0.3, 0.4, 0.3])], refRows).pass, false);
});

test('profiles that dropped dispatches are incomplete', () => {
  assert.equal(profileComplete({ dropped: 0, passMode: { dropped: 0 } }), true);
  assert.equal(profileComplete({ dropped: 12 }), false);
  assert.equal(profileComplete({ dropped: 0, passMode: { dropped: 3 } }), false);
  const agg = aggregateProfile({ rows: [], dropped: 7 }, { invokes: 1 });
  assert.equal(agg.complete, false);
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

test('devicePeaks tolerates a missing f16 figure and skips failed runs', () => {
  const noF16 = { ok: true, peaks: { fmaF32Tflops: 3, fmaF16Tflops: null, readGBps: 90, copyGBps: null, bestGemmTflops: {} }, results: {} };
  const failed = { ok: false, peaks: { fmaF32Tflops: 99, readGBps: 999 }, results: {} };
  const p = devicePeaks({ results: [noF16, failed] });
  assert.equal(p.flops, 3e12);
  assert.equal(p.bw, 90e9);
  assert.throws(() => devicePeaks({ results: [failed] }), /no ok peak runs/);
  const invalidGemm = { ...noF16, results: { gemm: [{ shape: 's', tflops: 9, check: { valid: false } }] } };
  assert.equal(devicePeaks({ results: [invalidGemm] }).gemmByShape.s, undefined);
});

test('roofline drops failed and quantized repeats, and zero durations stay unresolved', () => {
  const rep = (ok, quantized, us) => ({
    ok,
    candidate: { runtime: 'ort-webgpu', batch: 1, model: 'm' },
    timing: { runMs: { median: 1 } },
    profile: {
      timestampsQuantized: quantized,
      dropped: 0,
      submitsPerInvoke: 1,
      kernelMsPerInvoke: 0.5,
      perInvoke: [{ envelopeMs: 0.6, dispatches: 1 }],
      sequence: [['Softmax', us, '1x1x1', 4096]],
    },
  });
  const run = { results: [rep(false, false, 50), rep(true, true, 65.536), rep(true, false, 40)] };
  const a = analyzeCandidate('x', run, null, peaks);
  assert.equal(a.repeatsUsed, 1);
  assert.deepEqual(
    a.rejected.map((r) => r.repeat),
    [0, 1],
  );
  assert.equal(a.kernels[0].meanUs, 40);
  const zero = analyzeCandidate('x', { results: [rep(true, false, 0)] }, null, peaks);
  assert.equal(zero.kernels[0].bound, 'unresolved');
  assert.equal(zero.kernels[0].gbps, null);
  assert.match(analyzeCandidate('x', { results: [rep(false, false, 5)] }, null, peaks).error, /not ok/);
});

test('GEMM generators reject non-tiling shapes and size the grid', () => {
  assert.deepEqual(gemmKernel({ M: 512, N: 1024, K: 1024, storage: 'f16', accum: 'f32', tm4: 1, tn4: 2 }).dispatch, [8, 8, 1]);
  assert.throws(() => gemmKernel({ M: 100, N: 64, K: 16, storage: 'f32', accum: 'f32', tm4: 1, tn4: 1 }));
  const sg = subgroupMatrixGemmKernel({ M: 64, N: 64, K: 64, type: 'f16', resultType: 'f16', mm: 8, nn: 8, kk: 8, sg: 32, rt: 4, ct: 4 });
  assert.deepEqual(sg.dispatch, [2, 2, 1]);
  assert.match(sg.code, /@workgroup_size\(32\)/);
});

test('GEMM A-tile load writes whole workgroup vectors only (no cross-invocation component writes)', () => {
  for (const tm4 of [1, 2]) {
    const { code } = gemmKernel({ M: 512, N: 512, K: 64, storage: 'f16', accum: 'f32', tm4, tn4: 1 });
    assert.doesNotMatch(code, /As\[[^\]]*\]\s*\[/, 'component write into As');
    assert.doesNotMatch(code, /As\[[^\]]*\]\.[xyzw]\s*=/, 'swizzle write into As');
    assert.match(code, new RegExp(`if \\(li < ${64 * tm4}u\\)`));
  }
  assert.throws(() => gemmKernel({ M: 512, N: 512, K: 64, storage: 'f32', accum: 'f32', tm4: 8, tn4: 1 }));
});

test('FMA peak kernel uses per-lane constants and stores every invocation', () => {
  const code = fmaKernel('f32', 4);
  assert.match(code, /P: array<vec4<f32>, 2>/);
  assert.match(code, /let b = vec4<f32>\(P\[0\]\)/);
  assert.match(code, /\n  outBuf\[gid\.x\] = vec4<f32>\(s\);/);
  assert.doesNotMatch(code, /gid\.x == 0u/);
});
