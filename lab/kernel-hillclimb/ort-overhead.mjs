// ORT WebGPU per-invoke CPU/sync overhead: where the non-GPU time of one
// Lc0 b1 invoke goes, and paired A/B of IO variants that change only the
// plumbing (docs/ort_invoke_overhead_2026-11.md).
//
// Variants (each its own session on the same model bytes and dequant fold):
//   cpu      shipped: CPU input tensor, CPU outputs (ORT MemcpyToHost per output)
//   gpuout   preferredOutputLocation=gpu-buffer, outputs read with getData() in parallel
//   iobind   GPU input buffer (writeBuffer) + preallocated GPU outputs, one staging map (src/nn/ortGpuIo.ts)
//   capture  graph capture (ORT replays its recorded dispatches; ortGraphCaptureSessionOptions) with a GPU
//            input buffer and ORT-owned GPU outputs, one staging map (OrtGpuIoBinding outputMode 'session')
//   capture-prealloc  graph capture with preallocated output fetches: leaks native OrtValues (see leak=N)
//   capture-jsflag  iobind + enableGraphCapture with the string EP form: JS-side capture constraints only
//   bisection only: iobind-nopin / capture-nopin (no pinOrtGpuBufferImport), gpuin / capture-ortout
//            (GPU input buffer, ORT-allocated GPU outputs read with getData())
//
// Gate: every variant's raw output bytes must equal the cpu variant's bytes on
// every fixture (plumbing, not math: any difference fails the run).
//
// Query params:
//   model=URL (default shipped Lc0 Small b1 qdq8)   variants=cpu,gpuout,iobind,capture
//   warmup=N (10)  rounds=N (20)  block=N (5)  traceInvokes=N (20, 0 disables)
//   gpuBusy=1 (pass-mode GPU busy per variant)  evaluator=1 (Lc0OnnxEvaluator phase timings on cpu and the GPU-IO variants)
//   leak=N (0): N more invokes per variant with a tap on ORT's wasm module (ort-wasm-tap.mjs):
//            leaked native output handles per invoke and wasm heap growth

import { parseFen } from '../../src/chess/board.ts';
import { encodeLc0Classical112 } from '../../src/lc0/encoder112.ts';
import { Lc0OnnxEvaluator, prepareLc0EvaluatorInput } from '../../src/lc0/onnxEvaluator.ts';
import { f32ToF16Bits } from '../../src/nn/float16.ts';
import { OrtGpuIoBinding, ortGraphCaptureSessionOptions, pinOrtGpuBufferImport } from '../../src/nn/ortGpuIo.ts';
import * as ort from '../../src/nn/ortRuntime.ts';
import { intParam, listParam, log, params, runMain, runtimeDeviceIdentity, summarize } from './common.mjs';
import { installGpuProfiler, passModeProfile } from './gpu-profiler.mjs';
import { installOrtWasmTap } from './ort-wasm-tap.mjs';

const DEFAULT_MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';
const INPUT = '/input/planes';

// ---- WebGPU API timeline tracer (prototype patches; installed before any device exists) ----
const trace = { armed: false, events: [] };
function patch(proto, method, wrap) {
  if (!proto || typeof proto[method] !== 'function') return;
  const original = proto[method];
  proto[method] = wrap(original);
}
function installTracer() {
  const rec = (kind, extra) => {
    if (trace.armed) trace.events.push({ t: performance.now(), kind, ...extra });
  };
  patch(
    GPUQueue.prototype,
    'submit',
    (o) =>
      function submit(buffers) {
        rec('submit', { n: buffers?.length ?? 0 });
        return o.call(this, buffers);
      },
  );
  patch(
    GPUQueue.prototype,
    'writeBuffer',
    (o) =>
      function writeBuffer(...args) {
        rec('writeBuffer');
        return o.apply(this, args);
      },
  );
  patch(
    GPUBuffer.prototype,
    'mapAsync',
    (o) =>
      function mapAsync(...args) {
        const id = trace.events.length;
        rec('mapAsync', { id });
        const p = o.apply(this, args);
        if (!trace.armed) return p;
        return p.then(
          (v) => (rec('mapResolved', { id }), v),
          (e) => {
            rec('mapRejected', { id });
            throw e;
          },
        );
      },
  );
  patch(
    GPUDevice.prototype,
    'createBindGroup',
    (o) =>
      function createBindGroup(d) {
        rec('bindGroup');
        return o.call(this, d);
      },
  );
  patch(
    GPUDevice.prototype,
    'createBuffer',
    (o) =>
      function createBuffer(d) {
        rec('createBuffer', { size: d?.size, usage: d?.usage });
        return o.call(this, d);
      },
  );
  patch(
    GPUComputePassEncoder.prototype,
    'dispatchWorkgroups',
    (o) =>
      function dispatchWorkgroups(...args) {
        rec('dispatch');
        return o.apply(this, args);
      },
  );
  patch(
    GPUCommandEncoder.prototype,
    'copyBufferToBuffer',
    (o) =>
      function copyBufferToBuffer(...args) {
        rec('copy');
        return o.apply(this, args);
      },
  );
}

// Bisection aid: log when a buffer we own is destroyed by someone else (ORT releasing its import).
const watched = new WeakMap();
const destroyedWatched = [];
let tearingDown = false;
function watchDestroy(buffer, owner) {
  watched.set(buffer, owner);
}
function installDestroyWatch() {
  patch(
    GPUBuffer.prototype,
    'destroy',
    (o) =>
      function destroy() {
        const owner = watched.get(this);
        if (owner && !tearingDown) {
          destroyedWatched.push({ owner, size: this.size });
          console.warn(`ort-overhead: ${owner} buffer (${this.size} B) destroyed by the runtime`);
        }
        return o.call(this);
      },
  );
}

/** One invoke's timeline, relative to the call. */
function analyzeInvoke(events, t0, t1) {
  const ev = events.map((e) => ({ ...e, t: e.t - t0 }));
  const firstMapIdx = ev.findIndex((e) => e.kind === 'mapAsync');
  const before = firstMapIdx < 0 ? ev : ev.slice(0, firstMapIdx);
  const submitsBeforeMap = before.filter((e) => e.kind === 'submit');
  const maps = ev.filter((e) => e.kind === 'mapAsync');
  const resolves = ev.filter((e) => e.kind === 'mapResolved');
  const count = (k) => ev.filter((e) => e.kind === k).length;
  const firstGpuCall = ev.find((e) => e.kind !== 'mapResolved');
  return {
    wallMs: t1 - t0,
    dispatches: count('dispatch'),
    submits: count('submit'),
    bindGroups: count('bindGroup'),
    writeBuffers: count('writeBuffer'),
    createBuffers: count('createBuffer'),
    copies: count('copy'),
    maps: maps.length,
    // t of the first WebGPU call: JS + ORT pre-run (input copy into the heap, binding) before any GPU API use.
    firstGpuCallMs: firstGpuCall?.t ?? null,
    // CPU time until the last compute submit before the first readback: ORT's per-kernel encode on the critical path.
    lastComputeSubmitMs: submitsBeforeMap.at(-1)?.t ?? null,
    firstMapCallMs: maps[0]?.t ?? null,
    firstMapResolvedMs: resolves[0]?.t ?? null,
    lastMapResolvedMs: resolves.at(-1)?.t ?? null,
    // Readback chain: first map call to last resolve (GPU tail + every extra serial round trip).
    readbackChainMs: maps.length && resolves.length ? resolves.at(-1).t - maps[0].t : null,
    // After the last resolve: output copy out of the heap, tensor wrap, JS.
    afterLastResolveMs: resolves.length ? t1 - t0 - resolves.at(-1).t : null,
    mapGapsMs: maps.slice(1).map((m, i) => m.t - (resolves[i]?.t ?? Number.NaN)),
  };
}

function summarizeTraces(rows) {
  const keys = Object.keys(rows[0] ?? {}).filter((k) => typeof rows[0][k] === 'number');
  const out = {};
  for (const k of keys) out[k] = summarize(rows.map((r) => r[k]).filter((v) => v !== null && Number.isFinite(v)));
  return out;
}

// ---- inputs ----
async function fixtureFens() {
  const fixtures = await (await fetch('/fixtures/lc0/fen_only.json')).json();
  return fixtures.map((f) => f.fen).filter(Boolean);
}

function f16Planes(fen) {
  const planes = encodeLc0Classical112(fen).planes;
  const out = new Uint16Array(planes.length);
  for (let i = 0; i < planes.length; i++) out[i] = f32ToF16Bits(planes[i]);
  return out;
}

function outputSpecs(session) {
  return session.outputNames.map((name) => {
    const meta = session.outputMetadata.find((m) => m.name === name);
    const dims = (meta?.shape ?? []).map((d) => (typeof d === 'number' ? d : 1));
    return { name, type: meta?.type === 'float16' ? 'float16' : 'float32', dims };
  });
}

function bytesOf(arr) {
  return new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
}

function sameBytes(a, b) {
  const x = bytesOf(a);
  const y = bytesOf(b);
  if (x.length !== y.length) return false;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
  return true;
}

// ---- variants ----
async function makeVariant(name, model, device) {
  const t0 = performance.now();
  if (name === 'cpu') {
    const session = await ort.createOrtSession(model);
    const inputType = session.inputMetadata.find((m) => m.name === INPUT)?.type;
    if (inputType !== 'float16') throw new Error(`ort-overhead: expected a float16 input, got ${inputType}`);
    return {
      name,
      session,
      createMs: performance.now() - t0,
      invoke: async (f16) => {
        const outputs = await session.run({ [INPUT]: new ort.Tensor('float16', f16, [1, 112, 8, 8]) });
        const out = {};
        for (const [k, t] of Object.entries(outputs)) out[k] = t.data;
        return out;
      },
    };
  }
  if (name === 'gpuout') {
    const session = await ort.createOrtSession(model, { webgpuOnly: true, webgpuSessionOptions: { preferredOutputLocation: 'gpu-buffer' } });
    return {
      name,
      session,
      createMs: performance.now() - t0,
      invoke: async (f16) => {
        const outputs = await session.run({ [INPUT]: new ort.Tensor('float16', f16, [1, 112, 8, 8]) });
        const entries = await Promise.all(Object.entries(outputs).map(async ([k, t]) => [k, await ort.ortTensorData(t)]));
        return Object.fromEntries(entries);
      },
    };
  }
  if (['iobind', 'capture', 'capture-prealloc', 'iobind-nopin', 'capture-nopin', 'capture-jsflag'].includes(name)) {
    const capture = name.startsWith('capture');
    // capture-jsflag: enableGraphCapture with the string provider form, which never reaches the EP (see ortGraphCaptureSessionOptions).
    const captureOptions = name === 'capture-jsflag' ? { enableGraphCapture: true } : ortGraphCaptureSessionOptions();
    const webgpuSessionOptions = { preferredOutputLocation: 'gpu-buffer', ...(capture ? captureOptions : {}) };
    const session = await ort.createOrtSession(model, { webgpuOnly: true, webgpuSessionOptions });
    const binding = new OrtGpuIoBinding(
      session,
      device ?? (await ort.env.webgpu.device),
      ort.Tensor,
      { name: INPUT, type: 'float16', dims: [1, 112, 8, 8] },
      outputSpecs(session),
      { pin: !name.endsWith('-nopin'), outputMode: name === 'capture' ? 'session' : 'preallocated' },
    );
    for (const buffer of [binding.inputBuffer, ...binding.outputBuffers]) watchDestroy(buffer, name);
    return { name, session, binding, createMs: performance.now() - t0, invoke: (f16) => binding.run(f16) };
  }
  // Bisection variants: GPU input buffer with ORT-allocated GPU outputs read by getData().
  if (name === 'gpuin' || name === 'capture-ortout') {
    const webgpuSessionOptions = { preferredOutputLocation: 'gpu-buffer', ...(name === 'capture-ortout' ? ortGraphCaptureSessionOptions() : {}) };
    const session = await ort.createOrtSession(model, { webgpuOnly: true, webgpuSessionOptions });
    const dev = device ?? (await ort.env.webgpu.device);
    const inputBuffer = dev.createBuffer({ size: 112 * 64 * 2, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    pinOrtGpuBufferImport(inputBuffer);
    watchDestroy(inputBuffer, name);
    const feeds = { [INPUT]: ort.Tensor.fromGpuBuffer(inputBuffer, { dataType: 'float16', dims: [1, 112, 8, 8] }) };
    return {
      name,
      session,
      createMs: performance.now() - t0,
      invoke: async (f16) => {
        dev.queue.writeBuffer(inputBuffer, 0, f16);
        const outputs = await session.run(feeds);
        const entries = await Promise.all(Object.entries(outputs).map(async ([k, t]) => [k, await ort.ortTensorData(t)]));
        return Object.fromEntries(entries);
      },
    };
  }
  throw new Error(`unknown variant ${name}`);
}

// Median spacing between leaked handle addresses (OrtValue pointers): heap bytes per leaked object.
function handleStride(handles) {
  const sorted = handles.map(Number).sort((x, y) => x - y);
  const gaps = [];
  for (let i = 1; i < sorted.length; i++) gaps.push(sorted[i] - sorted[i - 1]);
  return gaps.length ? median(gaps) : null;
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

runMain(async () => {
  installTracer();
  installDestroyWatch();
  const profiler = installGpuProfiler();
  ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
  const model = params.get('model') ?? DEFAULT_MODEL;
  const names = listParam('variants', ['cpu', 'gpuout', 'iobind', 'capture']);
  if (names[0] !== 'cpu') throw new Error('variants must start with cpu (the reference)');
  const warmup = intParam('warmup', 10, 0, 1000);
  const rounds = intParam('rounds', 20, 2, 1000);
  const block = intParam('block', 5, 1, 100);
  const traceInvokes = intParam('traceInvokes', 20, 0, 1000);
  const leakInvokes = intParam('leak', 0, 0, 1_000_000);
  // Before the first session: ORT loads its glue on the first session create.
  const tap = leakInvokes > 0 ? installOrtWasmTap(ort.env) : null;
  const fens = await fixtureFens();
  const inputs = fens.map(f16Planes);

  const variants = [];
  for (const name of names) {
    const v = await makeVariant(name, model, variants.length ? await ort.env.webgpu.device : null);
    variants.push(v);
    log('session', { variant: name, createMs: v.createMs.toFixed(0), backend: ort.describeOrtBackendConfig() });
  }
  const prof = profiler.lastDevice();
  for (const v of variants) for (let i = 0; i < warmup; i++) await v.invoke(inputs[i % inputs.length]);

  // Gate: byte-identical outputs on every fixture, twice per variant (also catches stale-buffer reuse).
  const reference = [];
  for (const f of inputs) reference.push(await variants[0].invoke(f));
  const exactness = {};
  for (const v of variants) {
    const mismatches = [];
    for (let pass = 0; pass < 2; pass++) {
      // Second pass in reverse order: a variant returning the previous invoke's outputs fails.
      const order = pass ? [...inputs.keys()].reverse() : [...inputs.keys()];
      for (const i of order) {
        const out = await v.invoke(inputs[i]);
        for (const [k, ref] of Object.entries(reference[i])) if (!out[k] || !sameBytes(out[k], ref)) mismatches.push({ fixture: i, output: k, pass });
      }
    }
    exactness[v.name] = { fixtures: inputs.length, passes: 2, identical: mismatches.length === 0, mismatches: mismatches.slice(0, 10) };
  }
  log('exactness', exactness);
  const runtimeDestroyed = destroyedWatched.slice();
  log('buffers destroyed by the runtime before timing', runtimeDestroyed);

  // Native leak check: N invokes per variant; output handles still live afterwards were never released.
  let leak = null;
  if (tap) {
    leak = {};
    for (const v of variants) {
      const before = tap.snapshot();
      for (let i = 0; i < leakInvokes; i++) await v.invoke(inputs[i % inputs.length]);
      const after = tap.snapshot();
      leak[v.name] = {
        invokes: leakInvokes,
        leakedHandles: after.liveOutputHandles - before.liveOutputHandles,
        leakedHandlesPerInvoke: (after.liveOutputHandles - before.liveOutputHandles) / leakInvokes,
        outputHandlesRead: after.gotData - before.gotData,
        handlesReleased: after.released - before.released,
        heapGrowthBytes: after.heapBytes - before.heapBytes,
        heapBytesAfter: after.heapBytes,
        leakedHandleStrideBytes: handleStride([...after.live].filter((h) => !before.live.has(h))),
      };
      log('leak', v.name, leak[v.name]);
    }
  }

  // Timeline breakdown per variant.
  const traces = {};
  if (traceInvokes > 0) {
    for (const v of variants) {
      const rows = [];
      for (let i = 0; i < traceInvokes; i++) {
        trace.events = [];
        trace.armed = true;
        const t0 = performance.now();
        await v.invoke(inputs[i % inputs.length]);
        const t1 = performance.now();
        trace.armed = false;
        rows.push(analyzeInvoke(trace.events, t0, t1));
      }
      traces[v.name] = { summary: summarizeTraces(rows), sample: rows[Math.floor(rows.length / 2)] };
      log('trace', v.name, {
        wall: traces[v.name].summary.wallMs.median,
        lastComputeSubmit: traces[v.name].summary.lastComputeSubmitMs?.median,
        readbackChain: traces[v.name].summary.readbackChainMs?.median,
        submits: traces[v.name].summary.submits.median,
        maps: traces[v.name].summary.maps.median,
      });
    }
  }

  // GPU busy / envelope / idle-gap per variant (runtime's own passes timestamped).
  const gpu = {};
  if (params.get('gpuBusy') !== '0' && prof?.supported) {
    for (const v of variants) {
      let i = 0;
      gpu[v.name] = await passModeProfile(prof, () => v.invoke(inputs[i++ % inputs.length]), 20);
      log('gpu', v.name, gpu[v.name]);
    }
  }

  // Evaluator phase timings (Lc0OnnxEvaluator, the product's NN call): on the cpu
  // session and, when present, on the GPU-IO variants through the same binding.
  let evaluatorPhases = null;
  if (params.get('evaluator') !== '0') {
    evaluatorPhases = {};
    // Search-shaped input: prepared legal moves and cache key, as Lc0PuctSearcher passes them.
    const prepared = fens.map((fen) => prepareLc0EvaluatorInput(parseFen(fen)));
    for (const v of variants) {
      if (v.name !== 'cpu' && !v.binding) continue;
      const evaluator = new Lc0OnnxEvaluator(
        v.session,
        {},
        v.binding ? { binding: v.binding, mode: v.name.startsWith('capture') ? 'gpu-capture' : 'gpu' } : null,
      );
      const rows = [];
      for (let i = 0; i < 40; i++) {
        const t0 = performance.now();
        const e = await evaluator.evaluate(prepared[i % prepared.length]);
        const wall = performance.now() - t0;
        if (i >= 5) rows.push({ wall, ...Object.fromEntries(Object.entries(e.timing).filter(([, x]) => typeof x === 'number')) });
      }
      evaluatorPhases[v.name] = summarizeTraces(rows);
      log('evaluator', v.name, Object.fromEntries(Object.entries(evaluatorPhases[v.name]).map(([k, s]) => [k, s.median])));
    }
  }

  // Paired ABAB: per round, `block` invokes of every variant (rotating order); ratios vs cpu per round.
  const perRound = Object.fromEntries(variants.map((v) => [v.name, []]));
  let k = 0;
  for (let r = 0; r < rounds; r++) {
    const order = variants.map((_, i) => variants[(i + r) % variants.length]);
    for (const v of order) {
      const ms = [];
      for (let i = 0; i < block; i++) {
        const f = inputs[k++ % inputs.length];
        const t = performance.now();
        await v.invoke(f);
        ms.push(performance.now() - t);
      }
      perRound[v.name].push(median(ms));
    }
  }
  const ab = {};
  for (const v of variants) {
    ab[v.name] = {
      wallMs: summarize(perRound[v.name]),
      ratioVsCpu: summarize(perRound[v.name].map((x, i) => x / perRound.cpu[i])),
    };
  }
  log('ab', Object.fromEntries(Object.entries(ab).map(([n, x]) => [n, { wall: x.wallMs.median, ratio: x.ratioVsCpu.median }])));

  const identity = await runtimeDeviceIdentity(prof);
  tearingDown = true;
  for (const v of variants) {
    await ort.releaseOrtSession(v.session);
    await v.binding?.dispose();
  }
  const exact = Object.values(exactness).every((e) => e.identical);
  // Every variant except the known-leaky capture-prealloc must release all its output handles.
  const leakFree = !leak || Object.entries(leak).every(([name, l]) => name === 'capture-prealloc' || l.leakedHandles === 0);
  return {
    bench: 'ort-overhead',
    ok: exact && leakFree && !profiler.state.errors.length,
    leakFree,
    device: identity,
    candidate: { runtime: 'ort-webgpu', model, variants: names, createMs: Object.fromEntries(variants.map((v) => [v.name, v.createMs])) },
    params: { warmup, rounds, block, traceInvokes, leakInvokes, fixtures: fens.length },
    exactness,
    runtimeDestroyed,
    leak,
    traces,
    gpu,
    evaluatorPhases,
    ab,
    errors: profiler.state.errors,
  };
});
