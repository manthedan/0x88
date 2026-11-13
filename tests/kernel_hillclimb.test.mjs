import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import test from 'node:test';
import { FUSED_ATTN_PARITY_TOL, fusedAttnDiffStats, fusedAttnGate } from '../lab/kernel-hillclimb/fused-attn-kernels.mjs';
import {
  abIncompleteProfiles,
  compareCentipawn,
  compareLc0,
  fixtureChunks,
  legalPolicyParity,
  MAIA3_PARITY_CONDITIONS,
  maia3ParityRows,
  manifestCheck,
  ortParity,
  profileComplete,
  profileInvokeCap,
  runChunked,
} from '../lab/kernel-hillclimb/gates.mjs';
import { fmaKernel, gemmKernel, subgroupMatrixGemmKernel } from '../lab/kernel-hillclimb/wgsl-kernels.mjs';
import { CANDIDATES, WGSL_HEADS_GATE } from '../scripts/kernel_hillclimb/candidates.mjs';
import {
  analyzeCandidate,
  classify,
  devicePeaks,
  gridConsistent,
  MATMUL_RE,
  matchMatmuls,
  PATHS,
  pathPrecision,
  roofFor,
  toMarkdown,
} from '../scripts/kernel_hillclimb/roofline.mjs';
import { encodeLc0Classical112 } from '../src/lc0/encoder112.ts';

// The lab modules read location.search at import time.
globalThis.location ??= { search: '' };
const { aggregateProfile, sequenceMeanUs } = await import('../lab/kernel-hillclimb/gpu-profiler.mjs');
const { abExtraModels, roundOrder, summarize } = await import('../lab/kernel-hillclimb/common.mjs');

const peaks = { flops: 3e12, flopsByPrecision: { f32: 3e12, f16: 3e12 }, bw: 100e9, ridge: 30, gemmCeilingPlain: 2e12 };

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
  const roof = roofFor(peaks, 'f32');
  assert.equal(classify({ meanUs: 3, achievedFlops: 0, achievedBw: 1e9, ai: 0 }, peaks, roof).bound, 'launch');
  assert.equal(classify({ meanUs: 500, achievedFlops: 2e12, achievedBw: 10e9, ai: 200 }, peaks, roof).bound, 'compute');
  assert.equal(classify({ meanUs: 500, achievedFlops: 0, achievedBw: 80e9, ai: 0 }, peaks, roof).bound, 'bandwidth');
  assert.equal(classify({ meanUs: 300, achievedFlops: 3e9, achievedBw: 4e9, ai: 1 }, peaks, roof).bound, 'underused');
  assert.throws(() => classify({ meanUs: 300, achievedFlops: 3e9, achievedBw: 4e9, ai: 1 }, peaks), /roof/);
});

test('each kernel is judged against the roof of its own precision', () => {
  // f16 twice as fast as f32: an f32 kernel at 80% of the f32 roof is
  // compute-bound, though it is only 40% of the larger (f16) roof.
  const p2 = { flops: 20e12, flopsByPrecision: { f32: 10e12, f16: 20e12 }, bw: 100e9, ridge: 200 };
  const k = { meanUs: 500, achievedFlops: 8e12, achievedBw: 20e9, ai: 400 };
  assert.equal(classify(k, p2, roofFor(p2, 'f32')).bound, 'compute');
  assert.equal(classify(k, p2, roofFor(p2, 'f16')).bound, 'underused');
  // Ridge per precision: AI 150 is past the f32 ridge (100) but not the f16 one (200).
  const mid = { meanUs: 500, achievedFlops: 6e12, achievedBw: 40e9, ai: 150 };
  assert.equal(classify(mid, p2, roofFor(p2, 'f32')).bound, 'compute');
  assert.equal(roofFor(p2, 'f16').ridge, 200);
  assert.throws(() => roofFor({ ...p2, flopsByPrecision: { f32: 10e12, f16: null } }, 'f16'), /no f16/);
  assert.equal(pathPrecision('tvmjs-centipawn'), 'f32');
  assert.equal(pathPrecision('ort-bt4'), 'f16');
  assert.equal(pathPrecision('new', { manifest: '/runtimes/x/f32/v2/manifest.json' }), 'f32');
  assert.throws(() => pathPrecision('new', { model: '/models/x.onnx' }), /cannot tell/);
  for (const [name, meta] of Object.entries(PATHS)) assert.ok(['f16', 'f32'].includes(meta.precision), name);
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

test('abProfile invokes are capped to the timestamp capacity', () => {
  // Shipped Lc0 Small: 356 dispatches/invoke; 50 invokes would need 17,800 of 16,384 slots.
  assert.deepEqual(profileInvokeCap(50, 356, 16384), { invokes: 46, capped: true });
  assert.deepEqual(profileInvokeCap(8, 744, 16384), { invokes: 8, capped: false });
  assert.deepEqual(profileInvokeCap(46, 356, 16384), { invokes: 46, capped: false });
  assert.deepEqual(profileInvokeCap(5, 20000, 16384), { invokes: 0, capped: true });
  assert.deepEqual(profileInvokeCap(5, 0, 16384), { invokes: 5, capped: false }); // unknown: completeness still gates
});

test('incomplete auxiliary ab profiles are named and fail the gate', () => {
  const full = (invokes = 8) => ({ invokes, requestedInvokes: invokes, capped: false, complete: true, dropped: 0 });
  const ab = {
    a: { pass: { dropped: 0 } },
    b: { model: '/m/shipped.onnx', pass: { dropped: 0 }, profile: full() },
    extra: [{ key: 'c1', model: '/m/c1.onnx', pass: { dropped: 0 }, profile: full() }],
  };
  assert.deepEqual(abIncompleteProfiles(ab), []);
  assert.deepEqual(abIncompleteProfiles(null), []);
  assert.deepEqual(abIncompleteProfiles({ ...ab, b: { ...ab.b, profile: null } }), []); // abProfile=0
  const overflow = { ...ab, b: { ...ab.b, profile: { ...full(50), complete: false, dropped: 1416 } } };
  assert.deepEqual(abIncompleteProfiles(overflow), [
    { arm: 'b', model: '/m/shipped.onnx', profile: 'dispatch', invokes: 50, requestedInvokes: 50, dropped: 1416 },
  ]);
  const extraNoRoom = { ...ab, extra: [{ ...ab.extra[0], profile: { invokes: 0, requestedInvokes: 5, capped: true, complete: false, dropped: null } }] };
  assert.deepEqual(
    abIncompleteProfiles(extraNoRoom).map((e) => [e.arm, e.profile, e.invokes]),
    [['c1', 'dispatch', 0]],
  );
  const passDrop = { ...ab, a: { pass: { dropped: 3 } } };
  assert.deepEqual(abIncompleteProfiles(passDrop), [{ arm: 'a', model: null, profile: 'pass', dropped: 3 }]);
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
  assert.equal(p.flopsByPrecision.f32, 3.2e12);
  assert.equal(p.flopsByPrecision.f16, 2.7e12);
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
    candidate: { runtime: 'ort-webgpu', batch: 1, model: 'm.f16.onnx' },
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

test('roofline rejects the whole report when the command gate failed', () => {
  const rep = {
    ok: true,
    candidate: { runtime: 'ort-webgpu', batch: 1, model: 'm.f16.onnx' },
    timing: { runMs: { median: 1 } },
    profile: {
      dropped: 0,
      submitsPerInvoke: 1,
      kernelMsPerInvoke: 0.5,
      perInvoke: [{ envelopeMs: 0.6, dispatches: 1 }],
      sequence: [['Softmax', 40, '1x1x1', 4096]],
    },
  };
  const gate = { inPagePass: true, commandGate: { command: 'npm run gate', pass: false, tail: 'FAIL' } };
  const bad = analyzeCandidate('broken', { ok: false, gate, results: [rep, rep] }, null, peaks);
  assert.match(bad.error, /command gate failed/);
  assert.equal(bad.rejected.length, 2);
  const passing = analyzeCandidate('fine', { ok: true, gate: { ...gate, commandGate: { ...gate.commandGate, pass: true } }, results: [rep] }, null, peaks);
  assert.equal(passing.error, undefined);
  const md = toMarkdown({ peaks, candidates: [bad] });
  assert.match(md, /broken: REJECTED, command gate failed/);
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

const CORPUS = JSON.parse(readFileSync('fixtures/lc0/fen_only.json', 'utf8')).map((f) => f.fen);
const RULE50_PLANE = 109;

test('parity chunks cover every fixture, including the only nonzero rule50 one', () => {
  const rule50 = CORPUS.map((fen) => encodeLc0Classical112(fen).planes[RULE50_PLANE * 64]);
  const nonzero = rule50.flatMap((v, i) => (v ? [i] : []));
  assert.equal(nonzero.length, 1, 'fixture corpus should hold exactly one nonzero rule50 position');
  for (const batch of [1, 4, 8, 16]) {
    const chunks = fixtureChunks(CORPUS, batch);
    assert.ok(chunks.every((c) => c.fens.length === batch));
    assert.deepEqual(
      chunks.flatMap((c) => c.fens.slice(0, c.real)),
      CORPUS,
      `batch ${batch}`,
    );
  }
});

test('a candidate that zeroes the rule50 plane fails full-corpus parity', async () => {
  // Fake "models": the moves-left head reads the rule50 plane; the broken
  // candidate zeroes it before the forward pass.
  const model = (zeroRule50) => async (fens) => {
    const mlh = new Float32Array(fens.length);
    const policy = new Float32Array(fens.length * 4);
    fens.forEach((fen, i) => {
      const planes = Float32Array.from(encodeLc0Classical112(fen).planes);
      if (zeroRule50) planes.fill(0, RULE50_PLANE * 64, (RULE50_PLANE + 1) * 64);
      let occ = 0;
      for (let k = 0; k < 104 * 64; k++) occ += planes[k];
      mlh[i] = 0.1 * planes[RULE50_PLANE * 64] + 0.01 * occ;
      policy.fill(occ / 64, i * 4, i * 4 + 4);
    });
    return { '/output/mlh': mlh, '/output/policy': policy };
  };
  const limits = { '/output/policy': 0.06, '/output/mlh': 0.3 };
  for (const batch of [1, 8]) {
    const cand = await runChunked(CORPUS, batch, model(true));
    const ref = await runChunked(CORPUS, batch, model(false));
    assert.equal(ref['/output/mlh'].length, CORPUS.length);
    const full = ortParity(cand, ref, limits);
    assert.equal(full.pass, false, `batch ${batch}`);
    assert.deepEqual(full.failures, ['/output/mlh']);
    // The old timing-feed parity (first `batch` fixtures, cycled) missed it.
    const timingFeeds = Array.from({ length: batch }, (_, i) => CORPUS[i % CORPUS.length]);
    assert.equal(ortParity(await model(true)(timingFeeds), await model(false)(timingFeeds), limits).pass, true, `timing feed b${batch}`);
  }
});

test('runChunked keeps only real rows and rejects outputs without a batch dimension', async () => {
  const out = await runChunked(['a', 'b', 'c'], 2, async (fens) => ({ y: Float32Array.from(fens.flatMap((f) => [f.charCodeAt(0), 0])) }));
  assert.deepEqual([...out.y], [97, 0, 98, 0, 99, 0]);
  await assert.rejects(
    runChunked(['a'], 2, async () => ({ y: new Float32Array(3) })),
    /not a multiple/,
  );
});

test('manifestCheck flags a silent v3-subgroups -> v2 fallback', () => {
  const base = 'http://127.0.0.1:5241/lab/kernel-hillclimb/tvmjs-profile.html';
  const v3 = '/runtimes/centipawn-tvmjs-webgpu/x/f32/v3-subgroups/manifest.json';
  const v2 = 'http://127.0.0.1:5241/runtimes/centipawn-tvmjs-webgpu/x/f32/v2-shape-k16/manifest.json';
  assert.equal(manifestCheck(v3, `http://127.0.0.1:5241${v3}`, base).ok, true);
  const fb = manifestCheck(v3, v2, base);
  assert.equal(fb.ok, false);
  assert.equal(fb.effective, '/runtimes/centipawn-tvmjs-webgpu/x/f32/v2-shape-k16/manifest.json');
});

test('swapping the top-2 legal logits passes the logit tripwire but fails the legal-policy gate', () => {
  // Two rows of 6 policy slots; legal moves are slots 0, 2, 3, 5.
  const rows = [{ indices: [0, 2, 3, 5], ucis: ['e2e4', 'd2d4', 'g1f3', 'c2c4'] }, { indices: [0, 2, 3, 5] }];
  const ref = Float32Array.from([1, -9, 0, -1, -9, -2, 0.5, -9, 0.2, -1, -9, -3]);
  const limits = { maxProbAbs: 0.05, tieEpsilon: 0.01, topK: 3 };
  const same = legalPolicyParity(
    Float32Array.from(ref, (x) => x + 0.001),
    ref,
    rows,
    limits,
  );
  assert.equal(same.pass, true);
  assert.equal(same.bestMoveMatches, 2);
  const swapped = Float32Array.from(ref);
  [swapped[0], swapped[2]] = [ref[2], ref[0]];
  assert.equal(ortParity({ logits_move: swapped }, { logits_move: ref }, { logits_move: 1.3 }).pass, true, 'raw-logit tripwire alone misses the swap');
  const g = legalPolicyParity(swapped, ref, rows, limits);
  assert.equal(g.pass, false);
  assert.ok(g.problems.some((p) => /row 0: best move d2d4 vs reference e2e4/.test(p)));
  assert.ok(g.problems.some((p) => /row 1: /.test(p)) === false, 'row 1 is untouched');
  // Every row is checked, not just rows before the first failure.
  const both = Float32Array.from(swapped);
  [both[6], both[8]] = [swapped[8], swapped[6]];
  assert.ok(legalPolicyParity(both, ref, rows, limits).problems.some((p) => /row 1: best move/.test(p)));
  // A near-tie in the REFERENCE (gap <= tieEpsilon) may flip.
  const tieRef = Float32Array.from(ref);
  tieRef[2] = tieRef[0] - 0.01; // ~0.3% prior gap
  const tieSwap = Float32Array.from(tieRef);
  [tieSwap[0], tieSwap[2]] = [tieRef[2], tieRef[0]];
  const tie = legalPolicyParity(tieSwap, tieRef, rows, limits);
  assert.equal(tie.pass, true);
  assert.equal(tie.tieTolerated.length, 1);
  // Swapping the 2nd and 3rd moves (best move intact) fails the top-3 order check.
  const order = Float32Array.from(ref);
  [order[2], order[3]] = [ref[3], ref[2]];
  assert.ok(legalPolicyParity(order, ref, rows, { ...limits, maxProbAbs: 1 }).problems.some((p) => /order d2d4 > g1f3 reversed/.test(p)));
});

test('a Maia3 candidate that swaps elo_self and elo_oppo fails parity', async () => {
  // Fake Maia3: the value logit depends on the rating difference.
  const model = (swap) => async (rows) => ({
    logits_value: Float32Array.from(
      rows.flatMap((r) => {
        const [self, oppo] = swap ? [r.eloOppo, r.eloSelf] : [r.eloSelf, r.eloOppo];
        return [0, 0, (self - oppo) / 400];
      }),
    ),
  });
  const limits = { logits_value: 0.06 };
  const rows = maia3ParityRows(CORPUS);
  assert.equal(rows.length, CORPUS.length * MAIA3_PARITY_CONDITIONS.length);
  assert.ok(MAIA3_PARITY_CONDITIONS.some(([a, b]) => a !== b));
  for (const batch of [1, 11]) {
    const full = ortParity(await runChunked(rows, batch, model(true)), await runChunked(rows, batch, model(false)), limits);
    assert.equal(full.pass, false, `batch ${batch}`);
  }
  // The old feeds gave both inputs the same 1100 + 100 * i array: the swap was invisible.
  const old = CORPUS.map((fen, i) => ({ fen, eloSelf: 1100 + 100 * (i % 1), eloOppo: 1100 + 100 * (i % 1) }));
  assert.equal(ortParity(await runChunked(old, 1, model(true)), await runChunked(old, 1, model(false)), limits).pass, true);
});

test('an allowlisted drift flip passes only for its own position and move pair', () => {
  const fenA = 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1';
  const row = (fen) => ({ fen, indices: [0, 1, 2], ucis: ['h1h8', 'a1a8', 'e1d1'] });
  // Reference 0.535 / 0.465 between the leading pair (gap 0.07), then a tail move.
  const logits = (p) => Float32Array.from(p.map(Math.log));
  const ref = logits([0.5, 0.435, 0.065]);
  const swapped = logits([0.435, 0.5, 0.065]);
  const allowedFlips = [{ fen: fenA, moves: ['h1h8', 'a1a8'], maxGap: 0.1 }];
  const limits = { maxProbAbs: 0.08, tieEpsilon: 0.01, topK: 3, allowedFlips };
  const ok = legalPolicyParity(swapped, ref, [row(fenA)], limits);
  assert.equal(ok.pass, true);
  assert.equal(ok.tieTolerated[0].why, 'allowlisted');
  // The same swap in another position fails (best move and top-3 order).
  const other = legalPolicyParity(swapped, ref, [row('4k3/8/8/8/8/8/8/R3K2R w Q - 0 1')], limits);
  assert.equal(other.pass, false);
  assert.ok(other.problems.some((p) => /best move a1a8 vs reference h1h8/.test(p)));
  assert.ok(other.problems.some((p) => /order h1h8 > a1a8 reversed/.test(p)));
  // A different pair in the allowlisted position fails too.
  const ref2 = logits([0.5, 0.065, 0.435]);
  const swap2 = logits([0.435, 0.065, 0.5]);
  assert.equal(legalPolicyParity(swap2, ref2, [row(fenA)], limits).pass, false);
  // So does the allowlisted pair beyond its maxGap.
  assert.equal(legalPolicyParity(logits([0.3, 0.65, 0.05]), logits([0.65, 0.3, 0.05]), [row(fenA)], { ...limits, maxProbAbs: 1 }).pass, false);
  // Without the allowlist (blanket 0.01 epsilon), the measured flip fails.
  assert.equal(legalPolicyParity(swapped, ref, [row(fenA)], { ...limits, allowedFlips: [] }).pass, false);
});

test('roundOrder rotates every arm through every slot (two arms alternate AB, BA)', () => {
  assert.deepEqual(
    [0, 1, 2].map((r) => roundOrder(['a', 'b'], r).join('')),
    ['ab', 'ba', 'ab'],
  );
  const arms = ['a', 'b', 'c1', 'c2'];
  const slots = arms.map(() => new Set());
  for (let r = 0; r < arms.length; r++) {
    for (const [slot, arm] of roundOrder(arms, r).entries()) slots[slot].add(arm);
  }
  for (const s of slots) assert.equal(s.size, arms.length);
});

test('abExtraModels parses, dedupes and rejects cross-origin or excess arms', () => {
  assert.deepEqual(abExtraModels(null), []);
  assert.deepEqual(abExtraModels(''), []);
  assert.deepEqual(abExtraModels('/m/a.onnx, /m/b.onnx,/m/a.onnx,'), ['/m/a.onnx', '/m/b.onnx']);
  assert.throws(() => abExtraModels('https://evil.example/a.onnx'), /same-origin/);
  assert.throws(() => abExtraModels('//evil.example/a.onnx'), /same-origin/);
  assert.throws(() => abExtraModels(Array.from({ length: 9 }, (_, i) => `/m/${i}.onnx`).join(',')), /max 8/);
});

test('sequenceMeanUs averages per dispatch position and refuses mismatched invokes', () => {
  const row = (invoke, name, ns) => ({ invoke, name, ns });
  const rows = [row(0, 'Warm', 1), row(1, 'Gemm', 10000), row(1, 'Add', 2000), row(2, 'Gemm', 14000), row(2, 'Add', 4000)];
  assert.deepEqual(sequenceMeanUs(rows, 2), [12, 3]);
  assert.equal(sequenceMeanUs([row(1, 'Gemm', 1), row(2, 'MatMul', 1)], 2), null);
  assert.equal(sequenceMeanUs([row(1, 'Gemm', 1), row(2, 'Gemm', 1), row(2, 'Add', 1)], 2), null);
  assert.equal(sequenceMeanUs([], 1), null);
});

test('fused-attn gate: measured 1-ULP f16 differences pass, zeroed/rescaled/NaN outputs fail, f16 variants never gate', () => {
  // Worst measured f32-accumulate elements (BT4 L0/L7/L14 on M4 and RTX 3090): exactly 1 f16 ULP.
  const ref = Float32Array.from([1.337, -0.263, 2.449, 0, 1e-5, -5.9]);
  const oneUlp = Float32Array.from([1.337 + 2 ** -10, -0.263 + 2 ** -12, 2.449 - 2 ** -9, 2 ** -24, 1e-5, -5.9 + 2 ** -8]);
  const rows = (cand) => [{ layer: 14, vsWasm: fusedAttnDiffStats(cand, ref) }];
  const ok = fusedAttnGate('f32r1', 'f32', rows(oneUlp));
  assert.deepEqual(ok, { gated: true, pass: true, problems: [] });
  assert.ok(rows(oneUlp)[0].vsWasm.worstTolRatio < 0.5, 'measured worst element keeps >2x headroom');

  const zero = fusedAttnGate('f32r1', 'f32', rows(new Float32Array(ref.length)));
  assert.equal(zero.pass, false);
  assert.match(zero.problems[0], /f32r1 L14: 4 elements over/);
  assert.equal(fusedAttnGate('f32r2', 'f32', rows(ref.map((x) => x * 1.01))).pass, false, '1% rescale fails');
  const nan = fusedAttnGate('f32r1', 'f32', rows(Float32Array.from(ref, (x, i) => (i === 2 ? Number.NaN : x))));
  assert.equal(nan.pass, false);
  assert.match(nan.problems.join(), /1 non-finite/);
  assert.equal(fusedAttnGate('f32r1', 'f32', []).pass, false, 'no rows is not a pass');
  // A non-finite reference or truth must fail rather than make the allowance NaN (NaN > 1 is false).
  const nanRef = fusedAttnGate('f32r1', 'f32', [{ layer: 3, vsWasm: fusedAttnDiffStats(Float32Array.of(0), Float32Array.of(Number.NaN), Float32Array.of(1)) }]);
  assert.equal(nanRef.pass, false);
  assert.match(nanRef.problems.join(), /L3: 1 non-finite reference\/truth/);
  const infRef = [{ layer: 3, vsWasm: fusedAttnDiffStats(Float32Array.of(0), Float32Array.of(Number.POSITIVE_INFINITY)) }];
  assert.equal(fusedAttnGate('f32r1', 'f32', infRef).pass, false);
  const nanTruth = [{ layer: 3, vsWasm: fusedAttnDiffStats(Float32Array.of(1), Float32Array.of(1), Float32Array.of(Number.NaN)) }];
  assert.equal(fusedAttnGate('f32r1', 'f32', nanTruth).pass, false);

  // BT4 L14 index 374874: WASM 5.26e-4, kernel 9.365e-4, f64 9.372e-4. The reference's own error is allowed;
  // without the f64 truth the same element would be blamed on the kernel.
  const wasm = Float32Array.from([5.26e-4, 1.337]);
  const truth = Float32Array.from([9.372e-4, 1.337]);
  const kernel = Float32Array.from([9.365e-4, 1.337]);
  const withTruth = (cand) => [{ layer: 14, vsWasm: fusedAttnDiffStats(cand, wasm, truth) }];
  assert.equal(fusedAttnGate('f32r1', 'f32', withTruth(kernel)).pass, true);
  assert.equal(fusedAttnGate('f32r1', 'f32', [{ layer: 14, vsWasm: fusedAttnDiffStats(kernel, wasm) }]).pass, false);
  assert.equal(fusedAttnGate('f32r1', 'f32', withTruth(new Float32Array(2))).pass, false, 'zeros fail even with the truth allowance');

  const f16 = fusedAttnGate('f16r1', 'f16', rows(new Float32Array(ref.length)));
  assert.deepEqual(f16, { gated: false, pass: null, problems: [] });
  assert.equal(FUSED_ATTN_PARITY_TOL.rtol, 2 ** -9);
});
