// Fused 64-token attention feasibility page (docs/bignet_fused_attention_2026-11.md).
//
// Inputs are real activations of the big nets' attention cores (Q/K/V after the
// bias Add, smolgen bias), captured by scripts/kernel_hillclimb/fused_attn_data.py
// and staged by build_lab.mjs under /models/fused-attn/<net>/. For each kernel
// variant and for ORT WebGPU running the same core (the exported subgraph, and the
// MultiHeadAttention rewrite of it) the page reports:
//   - parity: max-abs vs ORT WASM running the exported subgraph (the shipped
//     reference runtime) and vs a float64 numpy reference, per captured layer
//   - GPU time: per-dispatch timestamps (gpu-profiler split mode) for the custom
//     kernels; per-invoke kernel sum for the ORT sessions (same profiler)
//
// Query params:
//   net=bt4|t3 (bt4)   variants=f32r1,f32r2,f16r1,f16r2   iters=N (40)  ortIters=N (10)
//   handoff=1 also measures the cross-runtime boundary of a hybrid split: ORT runs
//             the MHA subgraph with GPU-resident inputs/outputs (preferredOutputLocation
//             'gpu-buffer') on its own device, and the custom kernel consumes ORT's
//             output buffer on that same device (ort.env.webgpu.device). Reports wall
//             per session.run in a 15-run chain vs the same GPU work, and the chain
//             with a custom dispatch after every run.
//   debugZeroOutput=1 negative check: the parity readback clears the output buffer
//             instead of dispatching the kernel, so the gate must report ok:false.
//
// Gate (`ok`): every f32-accumulate kernel variant must match ORT WASM per element
// within atol 2^-12 + rtol 2^-9 * |wasm| (>= 2 f16 ULPs) + |wasm - f64| (the reference's
// own error; see FUSED_ATTN_PARITY_TOL in fused-attn-kernels.mjs), with no non-finite
// outputs. WebGPU validation errors throw (page error, bench.mjs fails). f16-accumulate variants and the ORT WebGPU sessions are reported
// but diagnostic only (`gate.diagnostic`). A failed gate publishes ok:false with
// `gate.problems` (status "done (gate failed)"; bench.mjs exits 1).
import * as ort from '../../src/nn/ortRuntime.ts';
import { intParam, listParam, log, params, runMain, runtimeDeviceIdentity, summarize } from './common.mjs';
import {
  fusedAttnDiffStats as diffStats,
  FUSED_ATTN_PARITY_TOL,
  fusedAttentionBytes,
  fusedAttentionFlops,
  fusedAttentionKernel,
  fusedAttnGate,
} from './fused-attn-kernels.mjs';
import { aggregateProfile, installGpuProfiler, passModeProfile } from './gpu-profiler.mjs';

const VARIANTS = {
  f32r1: { acc: 'f32', rows: 1 },
  f32r2: { acc: 'f32', rows: 2 },
  f16r1: { acc: 'f16', rows: 1 },
  f16r2: { acc: 'f16', rows: 2 },
};

function f16BitsToF32(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}
const f16ToF32Array = (u16) => Float32Array.from(u16, f16BitsToF32);

async function fetchBin(url, Type) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Type(await r.arrayBuffer());
}

runMain(async () => {
  const net = params.get('net') ?? 'bt4';
  const iters = intParam('iters', 40, 3, 2000);
  const ortIters = intParam('ortIters', 10, 1, 200);
  const variantNames = listParam('variants', Object.keys(VARIANTS));
  const debugZeroOutput = params.get('debugZeroOutput') === '1';
  const base = `/models/fused-attn/${net}/`;
  const manifest = await (await fetch(`${base}manifest.json`)).json();
  const { batch, heads, depth, dModel, scale, inputNames, outputName } = manifest;
  const shapeInfo = { batch, heads, depth, tokens: 64 };
  log('manifest', { net, batch, heads, depth, dModel, scale, layers: manifest.layers.map((l) => l.layer) });

  const layers = [];
  for (const L of manifest.layers) {
    const f = L.files;
    layers.push({
      layer: L.layer,
      q: await fetchBin(base + f.q, Uint16Array),
      k: await fetchBin(base + f.k, Uint16Array),
      v: await fetchBin(base + f.v, Uint16Array),
      bias: await fetchBin(base + f.bias, Uint16Array),
      ref: await fetchBin(base + f.ref, Float32Array),
      pyStats: L.stats,
    });
  }
  const feedsFor = (L) => ({
    [inputNames.q]: new ort.Tensor('float16', L.q, [batch * 64, dModel]),
    [inputNames.k]: new ort.Tensor('float16', L.k, [batch * 64, dModel]),
    [inputNames.v]: new ort.Tensor('float16', L.v, [batch * 64, dModel]),
    [inputNames.bias]: new ort.Tensor('float16', L.bias, [batch, heads, 64, 64]),
  });
  const outF32 = async (t) => {
    const data = t.location && t.location !== 'cpu' ? await t.getData() : t.data;
    return data instanceof Uint16Array ? f16ToF32Array(data) : Float32Array.from(data);
  };

  const profiler = installGpuProfiler();

  // 1. ORT WebGPU on the same subgraph: exported ops, and the MultiHeadAttention rewrite.
  // (First, so the product session factory configures ORT's env before the direct WASM session.)
  const ortResults = {};
  const ortOuts = {};
  {
    ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
    for (const [key, file] of Object.entries(manifest.subModels)) {
      const session = await ort.createOrtSession(`${base}${file}`);
      const backend = ort.describeOrtBackendConfig();
      if (!/webgpu/.test(backend)) throw new Error(`ORT ${key} subgraph did not run on WebGPU (${backend})`);
      const prof = profiler.lastDevice();
      ortOuts[key] = [];
      for (const L of layers) ortOuts[key].push(await outF32((await session.run(feedsFor(L)))[outputName]));
      const feeds = feedsFor(layers[0]);
      for (let i = 0; i < 3; i++) await session.run(feeds);
      prof.arm();
      for (let i = 0; i < ortIters; i++) {
        prof.nextInvoke();
        await session.run(feeds);
      }
      const agg = aggregateProfile(await prof.collect(), { invokes: ortIters });
      if (!agg.complete) throw new Error(`ORT ${key}: profiler dropped dispatches`);
      const perInvokeUs = agg.perInvoke.map((p) => p.busyMs * 1000);
      // Split mode puts every dispatch in its own timestamped pass, which inflates chains of
      // small dispatches; pass mode times ORT's own passes (the busy time that matters).
      const passMode = await passModeProfile(prof, () => session.run(feeds), ortIters);
      ortResults[key] = {
        backend,
        parity: null,
        kernelUsPerInvoke: summarize(perInvokeUs),
        passModeBusyUs: passMode.busyMs * 1000,
        passModeWallMs: passMode.wallMs,
        dispatchesPerInvoke: agg.perInvoke[0]?.dispatches,
        kernels: agg.kernels.map((k) => ({
          name: k.name,
          dispatchesPerInvoke: k.dispatchesPerInvoke,
          usPerInvoke: k.msPerInvoke * 1000,
          workgroups: k.workgroups,
          workgroupSize: k.workgroupSize,
        })),
      };
      log(`ort ${key}`, { us: ortResults[key].kernelUsPerInvoke.median, dispatches: ortResults[key].dispatchesPerInvoke });
      await ort.releaseOrtSession(session);
    }
  }

  // 2. Reference: ORT WASM on the exported subgraph.
  ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
  const wasm = await ort.InferenceSession.create(`${base}${manifest.subModels.orig}`, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
  for (const L of layers) L.wasm = await outF32((await wasm.run(feedsFor(L)))[outputName]);
  await wasm.release();
  const wasmVsRef = layers.map((L) => ({ layer: L.layer, ...diffStats(L.wasm, L.ref) }));
  log('wasm vs f64', wasmVsRef);

  for (const [key, outs] of Object.entries(ortOuts)) {
    ortResults[key].parity = layers.map((L, i) => ({ layer: L.layer, vsWasm: diffStats(outs[i], L.wasm), vsF64: diffStats(outs[i], L.ref) }));
    log(
      `ort ${key} parity`,
      ortResults[key].parity.map((p) => [p.layer, p.vsWasm.maxAbs, p.vsF64.maxAbs]),
    );
  }

  const handoff = params.get('handoff') === '1' ? await handoffProbe(base, manifest, layers[0], profiler) : null;

  // 3. Custom fused kernels on our own device (instrumented by the same profiler).
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter?.features.has('shader-f16')) throw new Error('shader-f16 unavailable');
  // rows=2 variants need ks/vs (16 KiB) plus the cross-half reduction scratch, so ask for the
  // adapter's workgroup-storage limit (32 KiB on Apple and NVIDIA) instead of the 16 KiB default.
  const device = await adapter.requestDevice({
    requiredFeatures: ['shader-f16'],
    requiredLimits: { maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize },
  });
  const prof = profiler.lastDevice();
  if (prof.device !== device) throw new Error('profiler did not instrument the lab device');
  const gpuErrors = profiler.state.errors;
  const mk = (data, usage) => {
    const buf = device.createBuffer({ size: Math.ceil(data.byteLength / 16) * 16, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  };
  const outBytes = batch * 64 * dModel * 2;
  // clearBuffer (debugZeroOutput) needs COPY_DST; the timed configuration keeps the plain usage.
  const outUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | (debugZeroOutput ? GPUBufferUsage.COPY_DST : 0);
  const outBuf = device.createBuffer({ size: outBytes, usage: outUsage });
  const readBuf = device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uni, 0, new Float32Array([scale, 0, 0, 0]));
  const layerBufs = layers.map((L) => ({
    q: mk(L.q, GPUBufferUsage.STORAGE),
    k: mk(L.k, GPUBufferUsage.STORAGE),
    v: mk(L.v, GPUBufferUsage.STORAGE),
    bias: mk(L.bias, GPUBufferUsage.STORAGE),
  }));
  const kernels = {};
  for (const name of variantNames) {
    const cfg = VARIANTS[name];
    if (!cfg) throw new Error(`unknown variant ${name}`);
    device.pushErrorScope('validation');
    const code = fusedAttentionKernel({ heads, depth, acc: cfg.acc, rows: cfg.rows });
    const module = device.createShaderModule({ code, label: `fused_attn_${name}` });
    const info = await module.getCompilationInfo();
    const errs = info.messages.filter((m) => m.type === 'error');
    if (errs.length) throw new Error(`${name} WGSL: ${errs.map((m) => `${m.lineNum}:${m.message}`).join('; ')}`);
    const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label: `fused_attn_${name}` });
    const groups = layerBufs.map((lb) =>
      device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [lb.q, lb.k, lb.v, lb.bias, outBuf, uni].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
    const scopeErr = await device.popErrorScope();
    if (scopeErr) throw new Error(`${name}: ${scopeErr.message}`);
    const dispatch = (enc, g) => {
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, g);
      pass.dispatchWorkgroups(batch * heads);
      pass.end();
    };
    // Parity per layer.
    const parity = [];
    for (const [i, L] of layers.entries()) {
      const enc = device.createCommandEncoder();
      if (debugZeroOutput) enc.clearBuffer(outBuf);
      else dispatch(enc, groups[i]);
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
      device.queue.submit([enc.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      const out = f16ToF32Array(new Uint16Array(readBuf.getMappedRange().slice(0)));
      readBuf.unmap();
      const vsWasm = diffStats(out, L.wasm, L.ref);
      const w = vsWasm.worstIndex;
      parity.push({ layer: L.layer, vsWasm, vsF64: diffStats(out, L.ref), atWorst: { index: w, kernel: out[w], wasm: L.wasm[w], f64: L.ref[w] } });
    }
    // Timing: `iters` dispatches, each its own timestamped pass (split mode), 15 per
    // submit like one invoke's worth of encoders.
    for (let w = 0; w < 2; w++) {
      const enc = device.createCommandEncoder();
      for (let i = 0; i < 15; i++) dispatch(enc, groups[0]);
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    prof.arm();
    for (let done = 0; done < iters; ) {
      prof.nextInvoke();
      const enc = device.createCommandEncoder();
      const n = Math.min(15, iters - done);
      for (let i = 0; i < n; i++) dispatch(enc, groups[0]);
      device.queue.submit([enc.finish()]);
      done += n;
    }
    await device.queue.onSubmittedWorkDone();
    const collected = await prof.collect();
    const us = collected.rows.filter((r) => r.invoke >= 1).map((r) => r.ns / 1000);
    if (us.length !== iters) throw new Error(`${name}: timed ${us.length} of ${iters} dispatches`);
    const t = summarize(us);
    kernels[name] = {
      ...cfg,
      gate: fusedAttnGate(name, cfg.acc, parity),
      workgroupSize: 64 * cfg.rows,
      workgroups: batch * heads,
      parity,
      dispatchUs: t,
      gbPerS: fusedAttentionBytes(shapeInfo) / (t.median * 1e3),
      gflopPerS: fusedAttentionFlops(shapeInfo) / (t.median * 1e3),
      timestampsQuantized: collected.timestampsQuantized,
    };
    log(`kernel ${name}`, { us: t.median, mad: t.mad, parity: parity.map((p) => [p.layer, p.vsWasm.maxAbs, p.vsF64.maxAbs]) });
  }
  if (gpuErrors.length) throw new Error(`WebGPU errors: ${gpuErrors.join(' | ')}`);
  const identity = await runtimeDeviceIdentity(prof);
  const gatedVariants = variantNames.filter((n) => kernels[n].gate.gated);
  const gateProblems = gatedVariants.flatMap((n) => kernels[n].gate.problems);
  if (!gatedVariants.length) gateProblems.push('no f32-accumulate variant ran; nothing gates (add f32r1 or f32r2 to variants)');
  const gate = {
    policy: 'f32-accumulate kernels vs ORT WASM, per element |kernel - wasm| <= atol + rtol*|wasm| + |wasm - f64|, no non-finite outputs',
    tolerance: FUSED_ATTN_PARITY_TOL,
    gated: gatedVariants,
    diagnostic: [...variantNames.filter((n) => !kernels[n].gate.gated), ...Object.keys(ortResults).map((k) => `ort.${k}`)],
    debugZeroOutput,
    problems: gateProblems,
    pass: gateProblems.length === 0,
  };
  if (!gate.pass) log('GATE_FAILED', gateProblems);
  return {
    bench: 'fused-attn',
    ok: gate.pass,
    gate,
    device: identity,
    net,
    shape: { ...shapeInfo, dModel, scale, bytesPerDispatch: fusedAttentionBytes(shapeInfo), flopsPerDispatch: fusedAttentionFlops(shapeInfo) },
    params: { iters, ortIters, variants: variantNames },
    pyStats: layers.map((L) => ({ layer: L.layer, ...L.pyStats })),
    wasmVsF64: wasmVsRef,
    ort: ortResults,
    handoff,
    kernels,
  };
});

/**
 * Hybrid-split boundary cost: chains of 15 ORT session.run calls with GPU-resident
 * IO (one per encoder in a per-layer split), with and without a custom-kernel
 * dispatch on ORT's device consuming each run's output buffer.
 */
async function handoffProbe(base, manifest, L, profiler) {
  const { batch, heads, depth, dModel, scale, inputNames, outputName } = manifest;
  const session = await ort.InferenceSession.create(`${base}${manifest.subModels.mha}`, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
    preferredOutputLocation: 'gpu-buffer',
  });
  const device = ort.env.webgpu.device;
  if (!(device instanceof GPUDevice)) throw new Error('handoff: ort.env.webgpu.device is not a GPUDevice');
  const prof = profiler.state.devices.find((d) => d.device === device) ?? null;
  const upload = (data) => {
    const buf = device.createBuffer({
      size: Math.ceil(data.byteLength / 16) * 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  };
  const bufs = { q: upload(L.q), k: upload(L.k), v: upload(L.v), bias: upload(L.bias) };
  const feeds = {
    [inputNames.q]: ort.Tensor.fromGpuBuffer(bufs.q, { dataType: 'float16', dims: [batch * 64, dModel] }),
    [inputNames.k]: ort.Tensor.fromGpuBuffer(bufs.k, { dataType: 'float16', dims: [batch * 64, dModel] }),
    [inputNames.v]: ort.Tensor.fromGpuBuffer(bufs.v, { dataType: 'float16', dims: [batch * 64, dModel] }),
    [inputNames.bias]: ort.Tensor.fromGpuBuffer(bufs.bias, { dataType: 'float16', dims: [batch, heads, 64, 64] }),
  };
  const module = device.createShaderModule({ code: fusedAttentionKernel({ heads, depth, acc: 'f32', rows: 1 }), label: 'fused_attn_handoff' });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uni, 0, new Float32Array([scale, 0, 0, 0]));
  const sink = device.createBuffer({ size: batch * 64 * dModel * 2, usage: GPUBufferUsage.STORAGE });
  const customOn = (qBuf) => {
    // The custom kernel reads ORT's output as its Q (any [B*64, D] f16 buffer works for timing).
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [qBuf, bufs.k, bufs.v, bufs.bias, sink, uni].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(batch * heads);
    pass.end();
    device.queue.submit([enc.finish()]);
  };
  const chain = async (withCustom) => {
    const t0 = performance.now();
    let cpuRunMs = 0;
    for (let i = 0; i < 15; i++) {
      const c0 = performance.now();
      const out = (await session.run(feeds))[outputName];
      cpuRunMs += performance.now() - c0;
      if (withCustom) customOn(out.gpuBuffer);
      out.dispose();
    }
    await device.queue.onSubmittedWorkDone();
    return { wallMs: performance.now() - t0, cpuRunMs };
  };
  for (let i = 0; i < 3; i++) {
    await chain(false);
    await chain(true);
  }
  const rounds = intParam('handoffRounds', 12, 2, 200);
  const ortOnly = [];
  const withCustom = [];
  const runCall = [];
  for (let r = 0; r < rounds; r++) {
    for (const k of r % 2 ? ['c', 'o'] : ['o', 'c']) {
      const res = await chain(k === 'c');
      (k === 'c' ? withCustom : ortOnly).push(res.wallMs / 15);
      runCall.push(res.cpuRunMs / 15);
    }
  }
  // GPU busy of one run (pass mode, the runtime's own passes) for the same chain.
  let gpuBusyUsPerRun = null;
  if (prof?.supported) {
    prof.arm({ mode: 'pass' });
    for (let i = 0; i < 15; i++) {
      prof.nextInvoke();
      (await session.run(feeds))[outputName].dispose();
    }
    await device.queue.onSubmittedWorkDone();
    const agg = aggregateProfile(await prof.collect(), { invokes: 15 });
    gpuBusyUsPerRun = summarize(agg.perInvoke.map((p) => p.busyMs * 1000));
  }
  await session.release();
  for (const b of Object.values(bufs)) b.destroy();
  const out = {
    sameDevice: true,
    perRunWallMsOrtOnly: summarize(ortOnly),
    perRunWallMsWithCustom: summarize(withCustom),
    sessionRunCallMs: summarize(runCall),
    gpuBusyUsPerRun,
  };
  log('handoff', {
    ortOnly: out.perRunWallMsOrtOnly.median,
    withCustom: out.perRunWallMsWithCustom.median,
    runCall: out.sessionRunCallMs.median,
    gpuUs: gpuBusyUsPerRun?.median,
  });
  return out;
}
