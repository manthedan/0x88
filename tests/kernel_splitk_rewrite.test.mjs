import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { skipUnlessAssets } from './lib/required_assets.mjs';

// scripts/kernel_hillclimb/splitk_rewrite.py on a two-MatMul toy graph: a
// token-row projection whose weight sits behind a per-column DequantizeLinear
// (as in the QDQ8 Lc0 exports) and an M = 1 GEMV, each followed by a bias Add.
const root = fileURLToPath(new URL('..', import.meta.url));
const python = resolve(root, '.venv-onnx/bin/python');
const script = 'scripts/kernel_hillclimb/splitk_rewrite.py';

function run(args) {
  return spawnSync(python, ['-I', ...args], { cwd: root, encoding: 'utf8' });
}

const BUILD = (path) =>
  [
    'import numpy as np, onnx',
    'from onnx import TensorProto, helper, numpy_helper',
    'rng = np.random.default_rng(1)',
    'x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [8, 64])',
    'g = helper.make_tensor_value_info("g", TensorProto.FLOAT, [1, 128])',
    'y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [8, 32])',
    'y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 32])',
    'inits = [',
    '  numpy_helper.from_array(rng.integers(-127, 128, (64, 32)).astype(np.int8), "w1_q"),',
    '  numpy_helper.from_array((rng.random(32) / 64 + 1e-3).astype(np.float32), "w1_scale"),',
    '  numpy_helper.from_array(rng.standard_normal(32).astype(np.float32), "b1"),',
    '  numpy_helper.from_array(rng.standard_normal((128, 32)).astype(np.float32), "w2"),',
    '  numpy_helper.from_array(rng.standard_normal(32).astype(np.float32), "b2"),',
    ']',
    'nodes = [',
    '  helper.make_node("DequantizeLinear", ["w1_q", "w1_scale"], ["w1"], name="/proj/dq", axis=1),',
    '  helper.make_node("MatMul", ["x", "w1"], ["p1"], name="/proj/w"),',
    '  helper.make_node("Add", ["p1", "b1"], ["y1"], name="/proj/bias"),',
    '  helper.make_node("MatMul", ["g", "w2"], ["p2"], name="/gemv/w"),',
    '  helper.make_node("Add", ["p2", "b2"], ["y2"], name="/gemv/bias"),',
    ']',
    'graph = helper.make_graph(nodes, "splitk-toy", [x, g], [y1, y2], inits)',
    `onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)]), ${JSON.stringify(path)})`,
  ].join('\n');

const OPS = (path) =>
  [
    'import collections, json, onnx',
    `m = onnx.load(${JSON.stringify(path)})`,
    'dims = {t.name: list(t.dims) for t in m.graph.initializer}',
    'print(json.dumps({"ops": collections.Counter(n.op_type for n in m.graph.node), "dims": dims}))',
  ].join('\n');

test('splitk_rewrite include / gemv-reduce / max-s / dispatch-probe options', async (t) => {
  const missing = skipUnlessAssets(python);
  if (missing) return t.skip(missing);
  if (run(['-c', 'import numpy, onnx, onnxruntime']).status !== 0) return t.skip('numpy/onnx/onnxruntime unavailable in .venv-onnx');
  const dir = mkdtempSync(join(tmpdir(), 'splitk-rewrite-test-'));
  try {
    const src = join(dir, 'src.onnx');
    const made = run(['-c', BUILD(src)]);
    assert.equal(made.status, 0, made.stderr);
    const rewrite = (name, ...opts) => {
      const dst = join(dir, `${name}.onnx`);
      const summaryPath = join(dir, `${name}.json`);
      const r = run([script, src, dst, '--min-weight', '0', '--min-inner', '8', '--check', '--summary', summaryPath, ...opts]);
      assert.equal(r.status, 0, r.stderr);
      const ops = run(['-c', OPS(dst)]);
      assert.equal(ops.status, 0, ops.stderr);
      return { summary: JSON.parse(readFileSync(summaryPath, 'utf8')), ...JSON.parse(ops.stdout) };
    };

    // Default: both MatMuls split; the DQ'd int8 weight is re-dimensioned to [S, K/S, N].
    const all = rewrite('all');
    assert.deepEqual(
      all.summary.rewritten.map((r) => [r.node, r.M, r.K, r.N, r.S]),
      [
        ['/proj/w', 8, 64, 32, 8],
        ['/gemv/w', 1, 128, 32, 16],
      ],
    );
    assert.deepEqual(all.dims.w1_q, [8, 8, 32]);
    assert.deepEqual(all.dims.w2, [16, 8, 32]);
    assert.equal(all.ops.ReduceSum, 2);
    for (const v of Object.values(all.summary.cpuCheck)) assert.ok(v < 1e-4, `max-abs ${v}`);

    // --include restricts the rewrite; --gemv-reduce matmul reduces the GEMV with ones[1,S] instead of ReduceSum.
    const gemv = rewrite('gemv', '--include', '^/gemv/', '--gemv-reduce', 'matmul');
    assert.deepEqual(
      gemv.summary.rewritten.map((r) => r.node),
      ['/gemv/w'],
    );
    assert.deepEqual(gemv.dims.w1_q, [64, 32]);
    assert.equal(gemv.ops.ReduceSum, undefined);
    assert.equal(gemv.ops.MatMul, 3);
    for (const v of Object.values(gemv.summary.cpuCheck)) assert.ok(v < 1e-4, `max-abs ${v}`);

    // --max-s caps the split factor.
    const capped = rewrite('capped', '--max-s', '4');
    assert.deepEqual(
      capped.summary.rewritten.map((r) => r.S),
      [4, 4],
    );

    // --dispatch-probe: no split, weights untouched, 2N Neg after each selected MatMul's bias Add, exact outputs.
    const probe = rewrite('probe', '--dispatch-probe', '2');
    assert.deepEqual(
      probe.summary.rewritten.map((r) => [r.node, r.S, r.probeAfter, r.probeDispatches]),
      [
        ['/proj/w', 1, 'Add', 4],
        ['/gemv/w', 1, 'Add', 4],
      ],
    );
    assert.deepEqual(probe.dims.w1_q, [64, 32]);
    assert.deepEqual(probe.dims.w2, [128, 32]);
    assert.equal(probe.ops.Neg, 8);
    assert.equal(probe.ops.MatMul, 2);
    assert.equal(probe.ops.ReduceSum, undefined);
    for (const v of Object.values(probe.summary.cpuCheck)) assert.equal(v, 0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
