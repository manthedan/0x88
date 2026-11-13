import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { defaultLc0OrtIoMode, lc0GpuIoSpecs } from '../src/lc0/onnxEvaluator.ts';
import { OrtGpuIoBinding, ortGpuIoLayout, ortGraphCaptureSessionOptions, pinOrtGpuBufferImport } from '../src/nn/ortGpuIo.ts';

const INPUT = { name: '/input/planes', type: 'float16', dims: [1, 112, 8, 8] };
const OUTPUTS = [
  { name: '/output/policy', type: 'float16', dims: [1, 1858] },
  { name: '/output/wdl', type: 'float16', dims: [1, 3] },
  { name: '/output/mlh', type: 'float16', dims: [1, 1] },
];

test('ortGpuIoLayout packs outputs at 16-byte aligned offsets', () => {
  const { slots, stagingBytes } = ortGpuIoLayout(OUTPUTS);
  assert.deepEqual(
    slots.map((s) => [s.elements, s.bytes, s.paddedBytes, s.offset]),
    [
      [1858, 3716, 3728, 0],
      [3, 6, 16, 3728],
      [1, 2, 16, 3744],
    ],
  );
  assert.equal(stagingBytes, 3760);
});

// What ORT-web 1.27's webgpuRegisterBuffer / webgpuUnregisterBuffer do with a
// caller's GPUBuffer (dist/ort-wasm-simd-threaded.jspi.mjs): a [handle, refcount]
// record under a private symbol; the handle is released when the count reaches 0.
const ORT_META = Symbol('gpuBufferMetadata');
let nextHandle = 1;
const liveHandles = new Set();
function ortRegister(buffer) {
  const meta = buffer[ORT_META];
  if (meta) {
    meta[1]++;
    return meta[0];
  }
  const handle = nextHandle++;
  liveHandles.add(handle);
  buffer[ORT_META] = [handle, 1];
  return handle;
}
function ortUnregister(buffer) {
  const meta = buffer[ORT_META];
  if (!meta) throw new Error('Buffer is not registered');
  meta[1]--;
  if (meta[1] === 0) {
    liveHandles.delete(meta[0]);
    delete buffer[ORT_META];
  }
}

/**
 * Fake device + session: the "GPU" output buffers are filled by session.run from
 * the uploaded input, so the test sees whether each invoke reads its own outputs.
 * run() registers and unregisters every gpu tensor the way ORT does.
 */
function fakeGpu({ failRun = false, captureBinding = false } = {}) {
  const log = [];
  const buffers = [];
  const makeBuffer = (size, usage) => {
    const buffer = {
      size,
      usage,
      bytes: new Uint8Array(size),
      mapped: false,
      destroyed: false,
      async mapAsync() {
        log.push('mapAsync');
        await Promise.resolve();
        this.mapped = true;
      },
      getMappedRange() {
        assert.ok(this.mapped, 'getMappedRange on an unmapped buffer');
        return this.bytes.buffer;
      },
      unmap() {
        this.mapped = false;
      },
      destroy() {
        this.destroyed = true;
      },
    };
    buffers.push(buffer);
    return buffer;
  };
  const device = {
    createBuffer: ({ size, usage }) => makeBuffer(size, usage),
    createCommandEncoder: () => {
      const copies = [];
      return {
        copyBufferToBuffer: (src, srcOff, dst, dstOff, size) => copies.push({ src, srcOff, dst, dstOff, size }),
        finish: () => copies,
      };
    },
    queue: {
      writeBuffer(buffer, offset, data, dataOffset, size) {
        log.push('writeBuffer');
        buffer.bytes.set(new Uint8Array(data, dataOffset, size), offset);
      },
      submit(cmds) {
        log.push('submit');
        for (const copies of cmds) for (const c of copies) c.dst.bytes.set(c.src.bytes.subarray(c.srcOff, c.srcOff + c.size), c.dstOff);
      },
    },
  };
  const Tensor = { fromGpuBuffer: (buffer, options) => ({ buffer, ...options }) };
  const session = {
    runs: 0,
    async run(feeds, fetches) {
      log.push('run');
      this.runs += 1;
      const tensors = [...Object.values(feeds), ...Object.values(fetches)];
      const handles = tensors.map((t) => ortRegister(t.buffer));
      // Graph capture binds IO once: later runs must see the first run's handles, still live.
      this.boundHandles ??= handles;
      try {
        await Promise.resolve();
        if (failRun) throw new Error('ort failed');
        if (captureBinding) {
          for (const h of this.boundHandles) assert.ok(liveHandles.has(h), 'captured run on a released handle');
        }
        // output value = first input element (+ slot index), as f16 bit patterns
        const seed = new Uint16Array(feeds['/input/planes'].buffer.bytes.buffer)[0];
        Object.values(fetches).forEach((t, i) => {
          new Uint16Array(t.buffer.bytes.buffer).fill(seed + i);
        });
        return fetches;
      } finally {
        for (const t of tensors) ortUnregister(t.buffer);
      }
    },
  };
  return { device, Tensor, session, log, buffers };
}

/**
 * A captured ORT-web 1.27 session (wasm-core-impl.ts + session-handler-inference.ts),
 * reduced to what decides native OrtValue lifetimes:
 * - IO is bound on the first run only; OrtRunWithBinding returns one new native
 *   OrtValue copy per output on every run.
 * - First run: the fetches are "pre-allocated outputs", so the copies are released.
 *   Later runs: they are not, so each copy is kept for a returned tensor's dispose(),
 *   and InferenceSession.run returns the caller's fetch tensors instead: the copies leak.
 * - No fetches: ORT allocates the outputs once (bound, reused by replays) and returns
 *   gpu-buffer tensors whose dispose() releases that run's copies.
 */
function fakeCapturedSession(gpu) {
  const native = new Set();
  let nextValue = 1;
  let owned = null;
  const session = {
    runs: 0,
    native,
    async run(feeds, fetches) {
      this.runs += 1;
      const first = this.runs === 1;
      const tensors = [...Object.values(feeds), ...Object.values(fetches ?? {})];
      for (const t of tensors) ortRegister(t.buffer);
      try {
        await Promise.resolve();
        const seed = new Uint16Array(feeds['/input/planes'].buffer.bytes.buffer)[0];
        if (!fetches) owned ??= OUTPUTS.map((o) => gpu.device.createBuffer({ size: o.dims.reduce((a, b) => a * b, 1) * 2 + 16, usage: 0x84 }));
        const targets = fetches ? Object.values(fetches).map((t) => t.buffer) : owned;
        targets.forEach((buffer, i) => {
          new Uint16Array(buffer.bytes.buffer).fill(seed + i);
        });
        const copies = OUTPUTS.map(() => {
          const id = nextValue++;
          native.add(id);
          return id;
        });
        if (fetches) {
          if (first) for (const id of copies) native.delete(id);
          return fetches;
        }
        return Object.fromEntries(
          OUTPUTS.map((o, i) => [o.name, { location: 'gpu-buffer', gpuBuffer: owned[i], type: o.type, dims: o.dims, dispose: () => native.delete(copies[i]) }]),
        );
      } finally {
        for (const t of tensors) ortUnregister(t.buffer);
      }
    },
  };
  return session;
}

test("captured sessions: preallocated fetches leak ORT's per-run output copies, session-owned outputs do not", async () => {
  const input = (v) => new Uint16Array(112 * 64).fill(v);
  const leaky = fakeGpu();
  const leakySession = fakeCapturedSession(leaky);
  const a = new OrtGpuIoBinding(leakySession, leaky.device, leaky.Tensor, INPUT, OUTPUTS);
  for (const v of [1, 2, 3, 4]) await a.run(input(v));
  assert.equal(leakySession.native.size, 3 * 3, 'three leaked copies per replay (run 1 is released)');

  const gpu = fakeGpu();
  const session = fakeCapturedSession(gpu);
  const b = new OrtGpuIoBinding(session, gpu.device, gpu.Tensor, INPUT, OUTPUTS, { outputMode: 'session' });
  assert.equal(b.outputMode, 'session');
  for (const v of [5, 6, 7, 8]) {
    const out = await b.run(input(v));
    assert.equal(out['/output/policy'][0], v);
    assert.equal(out['/output/wdl'][2], v + 1);
    assert.equal(out['/output/mlh'][0], v + 2);
  }
  assert.equal(session.native.size, 0, 'every returned output copy is disposed');
  await b.dispose();
  // our own buffers (input + staging) are destroyed; ORT-owned outputs belong to the session
  assert.equal(gpu.buffers.filter((buf) => buf.destroyed).length, 2);
});

test('session-owned outputs: wrong output shape is refused and the copies are still released', async () => {
  const gpu = fakeGpu();
  const session = fakeCapturedSession(gpu);
  const wrong = OUTPUTS.map((o, i) => (i === 0 ? { ...o, dims: [1, 1857] } : o));
  const binding = new OrtGpuIoBinding(session, gpu.device, gpu.Tensor, INPUT, wrong, { outputMode: 'session' });
  await assert.rejects(binding.run(new Uint16Array(112 * 64)), /expected float16 \[1,1857\]/);
  assert.equal(session.native.size, 0);
});

test('the evaluator gives captured sessions session-owned outputs', () => {
  const source = readFileSync(new URL('../src/lc0/onnxEvaluator.ts', import.meta.url), 'utf8');
  assert.match(source, /outputMode: mode === 'gpu-capture' \? 'session' : 'preallocated'/);
});

test('OrtGpuIoBinding: one upload, one run, one submit and one map per invoke, outputs per invoke', async () => {
  const gpu = fakeGpu();
  const binding = new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, INPUT, OUTPUTS);
  const input = (v) => new Uint16Array(112 * 64).fill(v);
  // Concurrent calls are serialized: each sees its own input's outputs.
  const [a, b] = await Promise.all([binding.run(input(7)), binding.run(input(9))]);
  assert.deepEqual([a['/output/policy'].length, a['/output/wdl'].length, a['/output/mlh'].length], [1858, 3, 1]);
  assert.equal(a['/output/policy'][0], 7);
  assert.equal(a['/output/wdl'][2], 8);
  assert.equal(a['/output/mlh'][0], 9);
  assert.equal(b['/output/policy'][1857], 9);
  assert.equal(b['/output/mlh'][0], 11);
  assert.ok(a['/output/policy'] instanceof Uint16Array);
  assert.deepEqual(gpu.log.slice(0, 8), ['writeBuffer', 'run', 'submit', 'mapAsync', 'writeBuffer', 'run', 'submit', 'mapAsync']);
  // A Float16Array view of the same f16 bits (what ort.Tensor re-wraps them as in browsers) is accepted as-is.
  if (typeof Float16Array === 'function') {
    const bits = input(0x3c00); // 1.0
    const c = await binding.run(new Float16Array(bits.buffer));
    assert.equal(c['/output/policy'][0], 0x3c00);
  }
  await binding.dispose();
  assert.ok(gpu.buffers.every((buf) => buf.destroyed));
  await assert.rejects(binding.run(input(1)), /disposed/);
});

test("pinOrtGpuBufferImport keeps ORT's import of a caller-owned buffer alive across runs", () => {
  const gpu = fakeGpu();
  const plain = gpu.device.createBuffer({ size: 16, usage: 0x80 });
  const h0 = ortRegister(plain);
  ortUnregister(plain);
  assert.equal(liveHandles.has(h0), false, 'unpinned: ORT releases the handle after the run');
  assert.notEqual(ortRegister(plain), h0, 'unpinned: the next run gets a new handle');

  const pinnedBuf = gpu.device.createBuffer({ size: 16, usage: 0x80 });
  const pinned = pinOrtGpuBufferImport(pinnedBuf);
  assert.equal(pinned(), false);
  const h = ortRegister(pinnedBuf);
  ortUnregister(pinnedBuf);
  assert.equal(pinned(), true);
  assert.ok(liveHandles.has(h));
  assert.equal(ortRegister(pinnedBuf), h, 'later runs reuse the same handle');
  ortUnregister(pinnedBuf);
  assert.ok(liveHandles.has(h));
  // methods and plain properties still work through the proxied prototype
  pinnedBuf.extra = 1;
  assert.equal(pinnedBuf.extra, 1);
  assert.ok(Object.hasOwn(pinnedBuf, 'extra'));
});

test('graph capture (IO bound once) needs the pinned handles', async () => {
  const unpinned = fakeGpu({ captureBinding: true });
  const a = new OrtGpuIoBinding(unpinned.session, unpinned.device, unpinned.Tensor, INPUT, OUTPUTS, { pin: false });
  await a.run(new Uint16Array(112 * 64).fill(3));
  await assert.rejects(a.run(new Uint16Array(112 * 64).fill(4)), /released handle/);

  const pinned = fakeGpu({ captureBinding: true });
  const b = new OrtGpuIoBinding(pinned.session, pinned.device, pinned.Tensor, INPUT, OUTPUTS);
  for (const v of [3, 4, 5]) assert.equal((await b.run(new Uint16Array(112 * 64).fill(v)))['/output/policy'][0], v);
});

test('OrtGpuIoBinding refuses to continue if ORT did not register the buffers as expected', async () => {
  const gpu = fakeGpu();
  gpu.session.run = async (_feeds, fetches) => fetches; // never registers
  const binding = new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, INPUT, OUTPUTS);
  await assert.rejects(binding.run(new Uint16Array(112 * 64)), /did not register/);
});

test('OrtGpuIoBinding rejects wrong input size or type, and a failed run does not wedge the queue', async () => {
  const gpu = fakeGpu({ failRun: true });
  const binding = new OrtGpuIoBinding(gpu.session, gpu.device, gpu.Tensor, INPUT, OUTPUTS);
  await assert.rejects(binding.run(new Uint16Array(10)), /bytes, expected/);
  await assert.rejects(binding.run(new Float32Array(112 * 32)), /must be f16 bits/);
  await assert.rejects(binding.run(new Uint16Array(112 * 64)), /ort failed/);
  await assert.rejects(binding.run(new Uint16Array(112 * 64)), /ort failed/);
  assert.equal(gpu.session.runs, 2);
});

test('lc0GpuIoSpecs accepts fixed-shape float models only', () => {
  const session = (inputShape, outputType = 'float16') => ({
    inputNames: ['/input/planes'],
    outputNames: ['/output/policy', '/output/wdl', '/output/mlh'],
    inputMetadata: [{ name: '/input/planes', isTensor: true, type: 'float16', shape: inputShape }],
    outputMetadata: [
      { name: '/output/policy', isTensor: true, type: outputType, shape: [1, 1858] },
      { name: '/output/wdl', isTensor: true, type: 'float16', shape: [1, 3] },
      { name: '/output/mlh', isTensor: true, type: 'float16', shape: [1, 1] },
    ],
  });
  const ok = lc0GpuIoSpecs(session([1, 112, 8, 8]));
  assert.deepEqual(ok.input, INPUT);
  assert.deepEqual(ok.outputs, OUTPUTS);
  assert.match(lc0GpuIoSpecs(session(['batch', 112, 8, 8])).reason, /dynamic shape/);
  assert.match(lc0GpuIoSpecs(session([1, 112, 8, 8], 'int64')).reason, /type int64/);
});

test('graph capture is requested with the object-form webgpu provider', () => {
  // ORT-web forwards enableGraphCapture to the native EP only for an object entry;
  // the string form 'webgpu' runs uncaptured (356 bind groups per invoke on Lc0 Small).
  assert.deepEqual(ortGraphCaptureSessionOptions(), { enableGraphCapture: true, executionProviders: [{ name: 'webgpu' }] });
});

test('search worker defaults the device-selected Lc0 Small model to captured GPU IO', () => {
  assert.equal(defaultLc0OrtIoMode(true), 'gpu-capture');
  assert.equal(defaultLc0OrtIoMode(false), 'cpu');
  const worker = readFileSync(new URL('../src/lc0/searchWorkerCore.ts', import.meta.url), 'utf8');
  assert.match(worker, /ioMode: message\.ortIoMode \?\? defaultLc0OrtIoMode\(message\.deviceModelSelection === true\)/);
});
