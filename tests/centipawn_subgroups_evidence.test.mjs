import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { statistics, summarizeRun } from '../scripts/summarize_centipawn_subgroups.mjs';
const root = 'lab/centipawn-tvmjs-subgroups-results/';
const load = (name) => JSON.parse(readFileSync(root + name + '.json'));

for (const name of ['run1', 'run2', 'run3', 'run4', 'edge-parity'])
  test(`saved Centipawn subgroup evidence is internally valid: ${name}`, () => {
    const run = load(name);
    assert.equal(run.ok, true);
    assert.equal(run.batch, 16);
    assert.equal(run.adapter.subgroupMinSize, 32);
    assert.equal(run.adapter.subgroupMaxSize, 32);
    assert.equal(run.lifecycle, 'PASS');
    assert.deepEqual(run.gpuErrors, []);
    for (const row of Object.values(run.parity)) {
      assert.ok(row.maxPolicyAbs <= row.tolerance);
      assert.ok(row.maxWdlAbs <= row.tolerance);
    }
    for (let k = 0; k < 2; k++) {
      assert.equal(run.invokes[k].length, run.repeats);
      for (const row of run.invokes[k]) {
        assert.ok(row.busyMs > 0 && row.envelopeMs > 0 && row.wallMs > 0);
        assert.equal(row.passes, 161);
        assert.ok(row.zeroPasses >= 0 && row.zeroPasses < row.passes);
        if (row.kernels) {
          assert.ok(Math.abs(Object.values(row.kernels).reduce((sum, k) => sum + k.gpuMs, 0) - row.busyMs) < 1e-9);
          assert.equal(
            Object.values(row.kernels).reduce((sum, k) => sum + k.count, 0),
            row.passes,
          );
        }
      }
      assert.deepEqual(run.summary[k].invokeBusyMs, statistics(run.invokes[k].map((r) => r.busyMs)));
      assert.equal(
        run.artifacts[k].manifest.models[0].sha256,
        k === 0 ? 'fd3d8635c8368338c3a0d1b57a725a70b4f8a74f8d130ee831848c641cba75b5' : '8bd117093f1914d60a5e7b89dbcb768c5a91130c346246fa7e9ebd763213b6c9',
      );
    }
    for (const row of run.search) {
      assert.equal(row.results[0].move, row.results[1].move);
      assert.deepEqual(row.results[0].pv, row.results[1].pv);
      assert.equal(row.results[0].visits, row.results[1].visits);
    }
    for (const row of run.ortSearch ?? []) {
      const tvm = run.search.find((s) => s.fixture === row.fixture).results[0];
      assert.equal(row.move, tvm.move);
      assert.deepEqual(row.pv, tvm.pv);
    }
  });
test('production browser routing and hardened lifecycle smoke pass', () => {
  const routing = load('routing').result;
  assert.equal(routing.ok, true);
  assert.deepEqual(
    routing.rows.map((row) => row.requestedFeatures),
    [[['subgroups']], [[]], [[]]],
  );
  assert.match(routing.rows[0].manifestUrl, /v3-subgroups\/manifest.json$/);
  for (const row of routing.rows.slice(1)) {
    assert.match(row.manifestUrl, /v2-shape-k16\/manifest.json$/);
    assert.equal(row.runtimeConfigId, 'centipawn-tvmjs-webgpu-f32-v2-shape-k16');
  }
  const smoke = load('final-smoke').result;
  assert.equal(smoke.ok, true);
  assert.equal(smoke.lifecycle, 'PASS');
  assert.deepEqual(smoke.gpuErrors, []);
});

test('summary is recomputed from retained interleaved samples', () => {
  const summary = load('summary');
  for (let n = 1; n <= 4; n++) assert.deepEqual(summary[`run${n}`], JSON.parse(JSON.stringify(summarizeRun(load(`run${n}`)))));
});
test('existing raw Centipawn parity gate passes for all four outputs', () => {
  for (const name of ['raw-parity-v2', 'raw-parity-v3']) {
    const run = load(name).result;
    assert.equal(run.ok, true);
    assert.deepEqual(run.gpuErrors, []);
    assert.deepEqual(Object.keys(run.parity).sort(), ['hidden', 'policy', 'q', 'wdl']);
    for (const row of Object.values(run.parity)) {
      assert.equal(row.allFinite, true);
      assert.equal(row.length, row.expectedLength);
      assert.ok(row.maxAbsDiff <= 0.001);
    }
  }
});
test('WGSL inspection changes only four reduction bodies and records actual staged bytes', () => {
  const inspection = load('wgsl-inspection');
  assert.equal(inspection.kernels.length, 46);
  assert.ok(inspection.kernels.every((k) => k.controlMatchesV2Normalized));
  const changed = inspection.kernels.filter((k) => !k.bodyMatchesV2Normalized);
  assert.deepEqual(
    changed.map((k) => k.kernel),
    ['fused_matmul12_add9_kernel', 'layer_norm_kernel', 'layer_norm1_kernel', 'softmax_kernel'],
  );
  const wasm = readFileSync(
    'public/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v3-subgroups/bt4-soap-rem-c19000-final.batch16.f32.webgpu.tvmjs-wasm.probe.tvmjs.wasm',
  );
  for (const kernel of changed) {
    const bytes = readFileSync(`${root}wgsl/${kernel.kernel}.wgsl`);
    assert.ok(wasm.includes(bytes));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), kernel.v3Sha256);
    assert.deepEqual(Object.keys(kernel.subgroupOps), ['subgroupShuffleDown']);
    assert.equal(kernel.workgroup, '64, 1, 1');
  }
});
