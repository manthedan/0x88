import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createOrtSession, setRequestedOrtExecutionProviderForCurrentThread, Tensor } from '../src/nn/ortRuntime.ts';

const root = fileURLToPath(new URL('..', import.meta.url));
const python = resolve(root, '.venv-onnx/bin/python');

function run(args) {
  return spawnSync(python, args, { cwd: root, encoding: 'utf8' });
}

test('dynamic MatMulInteger transformer emits an executable integer graph', async (t) => {
  if (!existsSync(python)) {
    t.skip('project ONNX virtual environment is unavailable');
    return;
  }
  const dependencyCheck = run(['-c', 'import numpy, onnx']);
  if (dependencyCheck.status !== 0) {
    t.skip('numpy and onnx are unavailable in the project virtual environment');
    return;
  }

  const directory = mkdtempSync(join(tmpdir(), 'lc0-mmi-test-'));
  const source = join(directory, 'source.onnx');
  const candidate = join(directory, 'candidate.onnx');
  const report = join(directory, 'report.json');
  try {
    const create = run([
      '-c',
      [
        'import numpy as np, onnx',
        'from onnx import TensorProto, helper, numpy_helper',
        `x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])`,
        `y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3])`,
        `w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 10, "w")`,
        `node = helper.make_node("MatMul", ["x", "w"], ["y"], name="projection")`,
        `graph = helper.make_graph([node], "mmi-test", [x], [y], [w])`,
        `onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), ${JSON.stringify(source)})`,
      ].join('; '),
    ]);
    assert.equal(create.status, 0, create.stderr);

    const transform = run(['scripts/lc0_quantize_onnx_dynamic_mmi.py', '--in', source, '--out', candidate, '--report', report, '--min-elements', '1']);
    assert.equal(transform.status, 0, transform.stderr);

    const summary = JSON.parse(readFileSync(report, 'utf8')).summary;
    assert.equal(summary.selectedMatMuls, 1);
    assert.equal(summary.uniqueQuantizedWeights, 1);
    assert.equal(summary.sharedActivationQuantizers, 1);
    assert.equal(summary.removedFloatWeights, 1);
    assert.equal(summary.operatorCounts.DynamicQuantizeLinear, 1);
    assert.equal(summary.operatorCounts.MatMulInteger, 1);
    assert.equal(summary.operatorCounts.Cast, 1);
    assert.equal(summary.operatorCounts.Mul, 2);

    const inspect = run([
      '-c',
      [
        'import json, onnx',
        `model = onnx.load(${JSON.stringify(candidate)})`,
        'onnx.checker.check_model(model, full_check=True)',
        'print(json.dumps({"inputs": [x.name for x in model.graph.input], "outputs": [x.name for x in model.graph.output], "initializers": [x.name for x in model.graph.initializer]}))',
      ].join('; '),
    ]);
    assert.equal(inspect.status, 0, inspect.stderr);
    const graph = JSON.parse(inspect.stdout);
    assert.deepEqual(graph.inputs, ['x']);
    assert.deepEqual(graph.outputs, ['y']);
    assert.equal(graph.initializers.includes('w'), false);
    assert.equal(graph.initializers.includes('w__mmi_q8'), true);

    setRequestedOrtExecutionProviderForCurrentThread('wasm');
    const sourceSession = await createOrtSession(readFileSync(source));
    const candidateSession = await createOrtSession(readFileSync(candidate));
    try {
      const input = new Tensor('float32', Float32Array.from([-1.25, -0.5, 0.25, 1.5, 1.75, 0.5, -0.75, -1.5]), [1, 2, 4]);
      const baseline = await sourceSession.run({ x: input });
      const transformed = await candidateSession.run({ x: input });
      const expected = Array.from(baseline.y.data);
      const actual = Array.from(transformed.y.data);
      assert.equal(actual.length, expected.length);
      const maxAbsError = Math.max(...expected.map((value, index) => Math.abs(Number(value) - Number(actual[index]))));
      assert.ok(maxAbsError < 0.03, `max absolute error ${maxAbsError} exceeds tolerance`);
    } finally {
      await sourceSession.release();
      await candidateSession.release();
      setRequestedOrtExecutionProviderForCurrentThread(null);
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test('dynamic MatMulInteger transformer can retain smolgen MatMuls in f32', (t) => {
  if (!existsSync(python)) {
    t.skip('project ONNX virtual environment is unavailable');
    return;
  }
  const dependencyCheck = run(['-c', 'import numpy, onnx']);
  if (dependencyCheck.status !== 0) {
    t.skip('numpy and onnx are unavailable in the project virtual environment');
    return;
  }

  const directory = mkdtempSync(join(tmpdir(), 'lc0-mmi-smolgen-test-'));
  const source = join(directory, 'source.onnx');
  const candidate = join(directory, 'candidate.onnx');
  const report = join(directory, 'report.json');
  try {
    const create = run([
      '-c',
      [
        'import numpy as np, onnx',
        'from onnx import TensorProto, helper, numpy_helper',
        `x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])`,
        `smol = helper.make_tensor_value_info("smol", TensorProto.FLOAT, [1, 3])`,
        `ffn = helper.make_tensor_value_info("ffn", TensorProto.FLOAT, [1, 3])`,
        `smol_w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 10, "/encoder0/smolgen/dense1/w/w")`,
        `ffn_w = numpy_helper.from_array(np.arange(12, dtype=np.float32).reshape(4, 3) / 20, "/encoder0/ffn/dense1/w/w")`,
        `smol_node = helper.make_node("MatMul", ["x", smol_w.name], ["smol"], name="/encoder0/smolgen/dense1/w")`,
        `ffn_node = helper.make_node("MatMul", ["x", ffn_w.name], ["ffn"], name="/encoder0/ffn/dense1/w")`,
        `graph = helper.make_graph([smol_node, ffn_node], "mmi-smolgen-test", [x], [smol, ffn], [smol_w, ffn_w])`,
        `onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), ${JSON.stringify(source)})`,
      ].join('; '),
    ]);
    assert.equal(create.status, 0, create.stderr);

    const transform = run([
      'scripts/lc0_quantize_onnx_dynamic_mmi.py',
      '--in',
      source,
      '--out',
      candidate,
      '--report',
      report,
      '--min-elements',
      '1',
      '--keep-smolgen-f32',
    ]);
    assert.equal(transform.status, 0, transform.stderr);

    const summary = JSON.parse(readFileSync(report, 'utf8')).summary;
    assert.equal(summary.keepSmolgenF32, true);
    assert.equal(summary.selectedMatMuls, 1);
    assert.equal(summary.skippedMatMuls.exclude_filter, 1);
    assert.equal(summary.operatorCounts.MatMul, 1);
    assert.equal(summary.operatorCounts.MatMulInteger, 1);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
