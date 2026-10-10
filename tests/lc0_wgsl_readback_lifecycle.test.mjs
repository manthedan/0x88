import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';

// Execute the real runtime methods with controlled buffer/queue promises.
// Node has no WebGPU: omit device/pack construction, not lifecycle logic.
const source = await readFile(new URL('../src/lc0/wgslMatmulAddProbe.ts', import.meta.url), 'utf8');
const classSource = source.slice(source.indexOf('class Lc0WebHybridRuntime {'), source.indexOf('\nexport async function runLc0WebHybridEvaluation'));
const Runtime = runInNewContext(`${stripTypeScriptTypes(classSource)}; Lc0WebHybridRuntime`, {
  nowMs: () => performance.now(),
});

const headHelpers = source.slice(
  source.indexOf('async function mapWgslPolicyValueHeadOutputs('),
  source.indexOf('\nasync function runCachedWgslPolicyValueHeads('),
);
const f32Helper = source.slice(source.indexOf('async function readF32OutputOnce('), source.indexOf('\nasync function measureGpuTimestampMs('));
const readers = runInNewContext(
  `${stripTypeScriptTypes(headHelpers + '\n' + f32Helper)}; ({ mapWgslPolicyValueHeadOutputs, mapWgslLegalPriorsOutputs, readF32OutputOnce })`,
  {
    nowMs: () => performance.now(),
    gpuGlobals: () => ({ GPUMapMode: { READ: 1 } }),
    DEFAULT_POLICY_MAPPED_OUTPUTS: 1858,
    WGSL_GPU_LEGAL_OUTPUT_FLOATS: 768,
  },
);

for (const name of Object.keys(readers)) {
  test(`${name} unmaps after a failed range read and permits reuse`, async () => {
    let mapped = false;
    let firstRead = true;
    let unmaps = 0;
    const buffer = {
      async mapAsync() {
        if (mapped) throw new Error('double map');
        mapped = true;
      },
      getMappedRange() {
        if (firstRead) {
          firstRead = false;
          throw new Error('range read failed');
        }
        return new ArrayBuffer(7444);
      },
      unmap() {
        mapped = false;
        unmaps += 1;
      },
    };
    const device = { createCommandEncoder: () => ({ copyBufferToBuffer() {}, finish() {} }), queue: { submit() {} } };
    const read = () => (name === 'readF32OutputOnce' ? readers[name](device, {}, buffer, 3) : readers[name]({ headsReadbackBuffer: buffer }, []));
    await assert.rejects(read(), /range read failed/);
    assert.equal(mapped, false, 'persistent readback must not remain mapped after an error');
    await read();
    assert.equal(mapped, false);
    assert.equal(unmaps, 2);
  });
}

function gate() {
  let resolve;
  const promise = new Promise((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

function submitted(slot, mapPromise = Promise.resolve(), queuePromise = Promise.resolve()) {
  let mapped = false;
  let unmapped = 0;
  const buffer = {
    getMappedRange() {
      throw new Error('range read failed');
    },
    unmap() {
      mapped = false;
      unmapped += 1;
    },
  };
  return {
    deferredReadbackSlot: slot,
    readbackBuffer: buffer,
    readbackMapState: {
      startedAt: 0,
      promise: mapPromise.then(() => {
        mapped = true;
      }),
    },
    queueState: { startedAt: 0, promise: queuePromise },
    get mapped() {
      return mapped;
    },
    get unmapped() {
      return unmapped;
    },
  };
}

function runtime() {
  return new Runtime({ headBackend: 'wgsl' });
}

test('failed second deferred submission drains and releases the first pending map before returning', async () => {
  const r = runtime();
  const mapping = gate();
  const first = submitted(0, mapping.promise);
  r.ensureWgslDeferredBatchSlots = () => [];
  r.ensureWgslDeferredBatchUploadBuffer = () => ({});
  r.ensureWgslDeferredReadbackBuffer = () => ({});
  r.wgslDeferredReadbackCapacity = 1;
  let calls = 0;
  r.submitWgslBatch = async () => {
    if (++calls === 1) return first;
    throw new Error('invalid second input');
  };
  let settled = false;
  const work = r.evaluateWgslBatchesDeferredReadback([['valid'], ['invalid']], {}).finally(() => {
    settled = true;
  });
  const rejected = assert.rejects(work, /invalid second input/);
  await new Promise((resolve) => setImmediate(resolve));
  const returnedBeforeMap = settled;
  mapping.resolve();
  await rejected;
  assert.equal(returnedBeforeMap, false, 'error must not return while prior map is pending');
  assert.equal(first.mapped, false);
  assert.equal(first.unmapped, 1);
  assert.equal(r.wgslDeferredReadbackInUse.size, 0, 'the next sequence can reuse both slots');
});

test('batch finalization waits for mapping and unmaps even when the queue fence rejects first', async () => {
  const r = runtime();
  const mapping = gate();
  const batch = submitted(0, mapping.promise, Promise.reject(new Error('device lost')));
  r.wgslDeferredReadbackInUse.add(0);
  let settled = false;
  const work = r.finishWgslBatch(batch, {}, 'deferred-double-buffered').finally(() => {
    settled = true;
  });
  const rejected = assert.rejects(work, /device lost/);
  await new Promise((resolve) => setImmediate(resolve));
  const returnedBeforeMap = settled;
  mapping.resolve();
  await rejected;
  assert.equal(returnedBeforeMap, false);
  assert.equal(batch.mapped, false);
  assert.equal(batch.unmapped, 1);
  assert.equal(r.wgslDeferredReadbackInUse.size, 0);
});

test('error cleanup unmaps a successful map despite a rejected queue fence', async () => {
  const r = runtime();
  const batch = submitted(0, Promise.resolve(), Promise.reject(new Error('device lost')));
  await r.cleanupStartedReadbackMap(batch.readbackBuffer, batch.readbackMapState, batch.queueState);
  assert.equal(batch.mapped, false);
  assert.equal(batch.unmapped, 1);
});

test('batch finalization releases a mapped buffer after a mapped-range read throws', async () => {
  const r = runtime();
  r.wgslHeadReadbackBytes = () => 16;
  const batch = submitted(0);
  batch.inputs = ['valid'];
  r.wgslDeferredReadbackInUse.add(0);
  await assert.rejects(r.finishWgslBatch(batch, {}, 'deferred-double-buffered'), /range read failed/);
  assert.equal(batch.mapped, false);
  assert.equal(batch.unmapped, 1);
  assert.equal(r.wgslDeferredReadbackInUse.size, 0);
});
