import assert from 'node:assert/strict';
import test from 'node:test';
import { Lc0WholeOnnxWebgpuEvaluator } from '../src/lc0/wholeOnnxWebgpuEvaluator.ts';

const FENS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
  'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
  '6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1',
];
const F16_HALF = 0x3800;
const F16_QUARTER = 0x3400;
const F16_TEN = 0x4900;

function f16Bytes(count, bitsAt) {
  const out = new Uint16Array(count);
  for (let i = 0; i < count; i++) out[i] = bitsAt(i);
  return new Uint8Array(out.buffer);
}

/**
 * Fake TVMJS runtime that mirrors the parts of the real one this evaluator
 * depends on: a single scope stack whose endScope() disposes everything
 * attached to the top scope, PackedFunc/Tensor return values auto-attached to
 * the current scope, detachFromCurrentScope() throwing for unknown objects, and
 * GPU->CPU copies that only land in the CPU tensor at device.sync().
 */
function fakeRuntime(batch, { syncGate, failCopyAt } = {}) {
  const state = {
    live: new Set(),
    scopes: [],
    invokes: 0,
    writeAfterFree: 0,
    disposedBeforeReadback: 0,
    runtimeDisposed: 0,
    deviceDestroyed: 0,
    pendingCopies: [],
  };
  const attach = (obj) => {
    if (!state.scopes.length) throw new Error('Must call beginScope to use functions that returns TVM objects');
    state.scopes.at(-1).push(obj);
    return obj;
  };
  const track = (obj) => {
    obj.disposed = false;
    state.live.add(obj);
    obj.dispose = () => {
      if (obj.readbackPending) state.disposedBeforeReadback += 1;
      obj.disposed = true;
      state.live.delete(obj);
    };
    return obj;
  };
  const device = {
    async sync() {
      if (syncGate) await syncGate();
      for (const copy of state.pendingCopies.splice(0)) copy();
    },
  };
  let copies = 0;
  const tensor = (shape, dtype, bytes) => {
    const t = track({ shape, dtype, bytes, device });
    t.copyFromRawBytes = (raw) => {
      assert.equal(t.disposed, false, 'copyFromRawBytes on a disposed tensor');
      t.bytes = raw.slice();
    };
    t.copyFrom = (source) => {
      assert.equal(source.disposed, false, 'copy from a disposed tensor');
      if (++copies === failCopyAt) throw new Error('copy enqueue failed');
      t.readbackPending = true;
      state.pendingCopies.push(() => {
        if (t.disposed) state.writeAfterFree += 1;
        else t.bytes = source.bytes.slice();
        t.readbackPending = false;
      });
    };
    t.toRawBytes = () => {
      if (t.disposed) throw new Error('Object has already been disposed');
      return t.bytes;
    };
    return t;
  };
  const outputs = [
    { shape: [batch, 1858], bytes: () => f16Bytes(batch * 1858, () => 0) },
    { shape: [batch, 3], bytes: () => f16Bytes(batch * 3, (i) => (i % 3 === 0 ? F16_HALF : F16_QUARTER)) },
    { shape: [batch, 1], bytes: () => f16Bytes(batch, () => F16_TEN) },
  ];
  const functions = {
    set_input(_name, input) {
      assert.equal(input.disposed, false);
      assert.deepEqual(input.shape, [batch, 112, 8, 8]);
    },
    invoke_stateful() {
      state.invokes += 1;
    },
    get_output(_name, index) {
      const out = outputs[index.value];
      return attach(tensor(out.shape, 'float16', out.bytes()));
    },
  };
  const vmMod = {
    getFunction(name) {
      const fn = (...args) => functions[name](...args);
      return attach(track(fn));
    },
  };
  const runtime = {
    beginScope() {
      state.scopes.push([]);
    },
    endScope() {
      if (!state.scopes.length) throw new Error('tvm.endScope called when the stack is empty.');
      for (const obj of state.scopes.pop()) obj?.dispose();
    },
    detachFromCurrentScope(obj) {
      const scope = state.scopes.at(-1);
      const index = scope.indexOf(obj);
      if (index < 0) throw new Error('Cannot find obj in the current auto conversion pool');
      scope[index] = undefined;
      return obj;
    },
    dispose() {
      state.runtimeDisposed += 1;
    },
    cpu: () => 'cpu',
    webgpu: () => 'webgpu',
    scalar: (value, dtype) => ({ value, dtype }),
    empty(shape, dtype) {
      attach(track({ kind: 'shape-tuple' }));
      return attach(tensor(shape, dtype, new Uint8Array(0)));
    },
  };
  const gpuDevice = {
    destroy() {
      state.deviceDestroyed += 1;
    },
  };
  return { runtime, vmMod, gpuDevice, state };
}

function createEvaluator(batch, options) {
  const fake = fakeRuntime(batch, options);
  // create() opens the long-lived root scope before building the VM.
  fake.runtime.beginScope();
  const evaluator = new Lc0WholeOnnxWebgpuEvaluator(fake.runtime, fake.gpuDevice, fake.vmMod, batch, [], {}, undefined);
  return { evaluator, ...fake };
}

function assertDecoded(rows, count) {
  assert.equal(rows.length, count);
  for (const row of rows) {
    assert.deepEqual(row.wdl, [0.5, 0.25, 0.25]);
    assert.equal(row.mlh, 10);
    assert.ok(row.legalPriors.length > 0);
  }
}

test('whole-model evaluator releases every per-batch TVM object after each evaluation', async () => {
  const { evaluator, state } = createEvaluator(4);
  const baseline = state.live.size;
  for (let i = 0; i < 3; i++) {
    assertDecoded(await evaluator.evaluateBatch([...FENS, ...FENS.slice(0, 2)]), 6);
    assert.equal(state.live.size, baseline, `live TVM objects after evaluation ${i}`);
  }
  assertDecoded([await evaluator.evaluate(FENS[0])], 1);
  const sequence = await evaluator.evaluateBatchSequence([FENS, FENS.slice(0, 3)]);
  assertDecoded(sequence[0], 4);
  assertDecoded(sequence[1], 3);
  assert.equal(state.live.size, baseline);
  assert.equal(state.scopes.length, 1, 'only the create()-time root scope stays open');
  assert.equal(state.writeAfterFree, 0);
  assert.equal(state.invokes, 2 * 3 + 1 + 2);
});

test('whole-model evaluator keeps concurrent evaluations independent of each other', async () => {
  const { evaluator, state } = createEvaluator(4, { syncGate: () => new Promise((resolve) => setTimeout(resolve, 0)) });
  const baseline = state.live.size;
  const [a, b, c] = await Promise.all([evaluator.evaluateBatch(FENS), evaluator.evaluateBatchSequence([FENS, FENS]), evaluator.evaluate(FENS[1])]);
  assertDecoded(a, 4);
  assertDecoded(b[0], 4);
  assertDecoded(b[1], 4);
  assert.equal(c.fen, FENS[1]);
  assert.equal(state.writeAfterFree, 0);
  assert.equal(state.live.size, baseline);
  assert.equal(state.scopes.length, 1);
});

test('whole-model evaluator drains submitted batches before rejecting a later invalid FEN', async () => {
  const { evaluator, state } = createEvaluator(4, { syncGate: () => new Promise((resolve) => setTimeout(resolve, 0)) });
  const baseline = state.live.size;
  await assert.rejects(evaluator.evaluateBatchSequence([FENS, ['invalid FEN']]), /FEN/);
  assert.equal(state.disposedBeforeReadback, 0, 'staging allocations must remain live until deferred writes finish');
  assert.equal(state.pendingCopies.length, 0, 'readbacks settle even when submission throws');
  assert.equal(state.writeAfterFree, 0);
  assert.equal(state.live.size, baseline);
  assertDecoded(await evaluator.evaluateBatch(FENS), 4);
});

test('whole-model evaluator drains partial staging copies when enqueueing throws', async () => {
  for (const sequence of [false, true]) {
    const { evaluator, state } = createEvaluator(4, { failCopyAt: 2 });
    const baseline = state.live.size;
    await assert.rejects(sequence ? evaluator.evaluateBatchSequence([FENS]) : evaluator.evaluateBatch(FENS), /copy enqueue failed/);
    assert.equal(state.disposedBeforeReadback, 0);
    assert.equal(state.writeAfterFree, 0);
    assert.equal(state.pendingCopies.length, 0);
    assert.equal(state.live.size, baseline);
  }
});

test('whole-model evaluator retains destinations if draining readbacks fails', async () => {
  const { evaluator, state } = createEvaluator(4, {
    syncGate: async () => {
      throw new Error('device lost');
    },
  });
  await assert.rejects(evaluator.evaluateBatch(FENS), /device lost/);
  await assert.rejects(evaluator.evaluateBatchSequence([FENS, FENS]), /device lost/);
  await assert.rejects(evaluator.evaluateBatchSequence([FENS, ['invalid FEN']]), /device lost/);
  assert.equal(state.disposedBeforeReadback, 0);
  evaluator.dispose();
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(state.runtimeDisposed, 0, 'pending WASM destinations must outlive the runtime too');
  for (const copy of state.pendingCopies.splice(0)) copy();
  assert.equal(state.writeAfterFree, 0);
  assert.equal(state.scopes.length, 1);
});

function replaceGlobal(name, value) {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, name);
  Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  return () => {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  };
}

test('whole-model create rejects odd physical batches whose WDL/MLH outputs cannot be read back', async (t) => {
  const manifest = { models: [1, 4, 8].map((batch) => ({ batch, wasm: `b${batch}.wasm` })), files: [] };
  const restore = [
    replaceGlobal('location', { href: 'https://lab.test/app/analysis', origin: 'https://lab.test' }),
    replaceGlobal('fetch', async () => new Response(JSON.stringify(manifest), { status: 200 })),
  ];
  t.after(() => {
    for (const undo of restore) undo();
  });
  const manifestUrl = '/runtimes/lc0-tvmjs-webgpu/t1/f16/v1/manifest.json';
  await assert.rejects(Lc0WholeOnnxWebgpuEvaluator.create({ manifestUrl, batch: 1 }), /batch 1 is unsupported.*\(6\/2 bytes\).*even batch \(4, 8\)/);
  // An even batch passes the readback check and fails later on the (absent) bundle metadata.
  await assert.rejects(Lc0WholeOnnxWebgpuEvaluator.create({ manifestUrl, batch: 4 }), /Runtime bundle integrity metadata missing/);
});

test('whole-model evaluator defers runtime and device release until in-flight work settles', async () => {
  let releaseSync;
  let syncStarted;
  const started = new Promise((resolve) => {
    syncStarted = resolve;
  });
  const blocked = new Promise((resolve) => {
    releaseSync = resolve;
  });
  const { evaluator, state } = createEvaluator(4, {
    syncGate: async () => {
      syncStarted();
      await blocked;
    },
  });
  const evaluation = evaluator.evaluateBatch(FENS);
  await started;
  evaluator.dispose();
  evaluator.dispose();
  await assert.rejects(evaluator.evaluate(FENS[0]), /disposed/);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(state.runtimeDisposed, 0);
  assert.equal(state.deviceDestroyed, 0);
  releaseSync();
  assertDecoded(await evaluation, 4);
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(state.runtimeDisposed, 1);
  assert.equal(state.deviceDestroyed, 1);
  assert.equal(state.scopes.length, 0, 'root scope released');
  assert.equal(state.writeAfterFree, 0);
});
