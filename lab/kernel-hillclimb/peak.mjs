// Device peak / practical-ceiling microbenchmarks for WebGPU.
// Query params:
//   tests=fma,bandwidth,gemm,sgmatrix,overhead (default all)
//   trials=N (default 7)   gemmShapes=MxNxK,... (default: real model shapes)
// Publishes window.__kernelBenchResult (bench: 'peak').
import { deviceIdentity, intParam, listParam, log, requestAdapter, runMain, summarize } from './common.mjs';
import { copyKernel, fmaFlopsPerThread, fmaKernel, gemmKernel, readKernel, subgroupMatrixGemmKernel, tinyKernel } from './wgsl-kernels.mjs';

const DEFAULT_GEMM_SHAPES = [
  // [label, M, N, K]; M = 64 tokens * batch for token-parallel layers.
  ['t1-b1-qkv 64x256x256', 64, 256, 256],
  ['t1-b16-qkv 1024x256x256', 1024, 256, 256],
  ['t1-b16-ffn1 1024x1024x256', 1024, 1024, 256],
  ['t1-b16-ffn2 1024x256x1024', 1024, 256, 1024],
  ['cp-b16-qkv 1024x384x128', 1024, 384, 128],
  ['bt4-b8-qkv 512x1024x1024', 512, 1024, 1024],
  ['bt4-b8-ffn1 512x1536x1024', 512, 1536, 1024],
  ['bt4-b8-ffn2 512x1024x1536', 512, 1024, 1536],
  ['square 2048x2048x2048', 2048, 2048, 2048],
];

async function getDevice() {
  const adapter = await requestAdapter();
  const want = ['timestamp-query', 'shader-f16', 'subgroups', 'chromium-experimental-subgroup-matrix'];
  const requiredFeatures = want.filter((f) => adapter.features.has(f));
  const requiredLimits = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
  };
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  device.addEventListener('uncapturederror', (e) => log('uncapturederror', String(e.error?.message ?? e)));
  if (!device.features.has('timestamp-query')) throw new Error('timestamp-query required for peak measurements');
  return { adapter, device };
}

async function compile(device, code, label) {
  const module = device.createShaderModule({ code, label });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((m) => m.type === 'error');
  if (errors.length) throw new Error(`${label}: WGSL compile failed: ${errors.map((m) => `${m.lineNum}:${m.linePos} ${m.message}`).join('; ')}`);
  device.pushErrorScope('validation');
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' }, label });
  const err = await device.popErrorScope();
  if (err) throw new Error(`${label}: pipeline: ${err.message}`);
  return pipeline;
}

function bindGroup(device, pipeline, resources) {
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: resources.map((buffer, binding) => ({ binding, resource: { buffer } })) });
}

/**
 * Time `reps` back-to-back dispatches inside ONE timestamped pass per trial;
 * returns per-dispatch GPU ns samples (one per trial) and wall ms samples.
 */
async function timeDispatches(device, pipeline, group, dispatch, { reps, trials, warmup = 2 }) {
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 * trials });
  const resolve = device.createBuffer({ size: 16 * trials, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * trials, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encodeRun = (writes) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(writes ? { timestampWrites: writes } : {});
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    for (let r = 0; r < reps; r++) pass.dispatchWorkgroups(...dispatch);
    pass.end();
    return enc.finish();
  };
  for (let i = 0; i < warmup; i++) {
    device.queue.submit([encodeRun(null)]);
    await device.queue.onSubmittedWorkDone();
  }
  const wall = [];
  for (let t = 0; t < trials; t++) {
    const cb = encodeRun({ querySet: qs, beginningOfPassWriteIndex: 2 * t, endOfPassWriteIndex: 2 * t + 1 });
    const t0 = performance.now();
    device.queue.submit([cb]);
    await device.queue.onSubmittedWorkDone();
    wall.push((performance.now() - t0) / reps);
  }
  const enc = device.createCommandEncoder();
  enc.resolveQuerySet(qs, 0, 2 * trials, resolve, 0);
  enc.copyBufferToBuffer(resolve, 0, read, 0, 16 * trials);
  device.queue.submit([enc.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const ticks = new BigUint64Array(read.getMappedRange().slice(0));
  read.unmap();
  const gpuNs = [];
  for (let t = 0; t < trials; t++) gpuNs.push(Number(ticks[2 * t + 1] - ticks[2 * t]) / reps);
  qs.destroy();
  resolve.destroy();
  read.destroy();
  return { gpuNs, wallMs: wall };
}

function randomBuffer(device, bytes, type) {
  const buf = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const n = bytes / (type === 'f16' ? 2 : 4);
  const chunk = 1 << 20;
  for (let off = 0; off < n; off += chunk) {
    const len = Math.min(chunk, n - off);
    if (type === 'f16') {
      const a = new Uint16Array(len);
      // f16 values in roughly [-0.5, 0.5): sign + exponent 13..14 + random mantissa.
      for (let i = 0; i < len; i++) a[i] = ((Math.random() < 0.5 ? 0x8000 : 0) | ((13 + (i & 1)) << 10) | ((Math.random() * 1024) | 0)) & 0xffff;
      device.queue.writeBuffer(buf, off * 2, a);
    } else {
      const a = new Float32Array(len);
      for (let i = 0; i < len; i++) a[i] = Math.random() - 0.5;
      device.queue.writeBuffer(buf, off * 4, a);
    }
  }
  return buf;
}

async function fmaTests(device, trials) {
  const out = [];
  const threads = 4096 * 256;
  const iters = 256;
  const outBuf = device.createBuffer({ size: threads * 16, usage: GPUBufferUsage.STORAGE });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uni, 0, new Float32Array([0.999, 0.001, 0, 0]));
  for (const type of ['f32', 'f16']) {
    if (type === 'f16' && !device.features.has('shader-f16')) continue;
    const pipeline = await compile(device, fmaKernel(type, iters), `fma_${type}`);
    const group = bindGroup(device, pipeline, [outBuf, uni]);
    const t = await timeDispatches(device, pipeline, group, [threads / 256, 1, 1], { reps: 4, trials });
    const flops = threads * fmaFlopsPerThread(iters);
    const s = summarize(t.gpuNs);
    const row = { test: `fma_${type}`, flopsPerDispatch: flops, gpuNs: s, tflops: flops / s.median / 1e3, tflopsBest: flops / s.min / 1e3 };
    log(row.test, `${row.tflops.toFixed(3)} TFLOP/s (best ${row.tflopsBest.toFixed(3)})`);
    out.push(row);
  }
  outBuf.destroy();
  uni.destroy();
  return out;
}

async function bandwidthTests(device, trials) {
  const out = [];
  const bytes = 256 * 1024 * 1024;
  const src = randomBuffer(device, bytes, 'f32');
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(uni, 0, new Uint32Array([bytes / 16, 0, 0, 0]));
  const read = await compile(device, readKernel(), 'bw_read');
  const rt = await timeDispatches(device, read, bindGroup(device, read, [src, dst, uni]), [2048, 1, 1], { reps: 4, trials });
  const rs = summarize(rt.gpuNs);
  out.push({ test: 'read_vec4_f32', bytesPerDispatch: bytes, gpuNs: rs, gbps: bytes / rs.median, gbpsBest: bytes / rs.min });
  log('read', `${(bytes / rs.median).toFixed(1)} GB/s`);
  const half = bytes / 2;
  device.queue.writeBuffer(uni, 0, new Uint32Array([half / 16, 0, 0, 0]));
  const copy = await compile(device, copyKernel(), 'bw_copy');
  const ct = await timeDispatches(device, copy, bindGroup(device, copy, [src, dst, uni]), [2048, 1, 1], { reps: 4, trials });
  const cs = summarize(ct.gpuNs);
  // copy moves `half` bytes in and `half` bytes out.
  out.push({ test: 'copy_vec4_f32', bytesPerDispatch: 2 * half, gpuNs: cs, gbps: (2 * half) / cs.median, gbpsBest: (2 * half) / cs.min });
  log('copy', `${((2 * half) / cs.median).toFixed(1)} GB/s (read+write)`);
  // copyBufferToBuffer, timed by a wall clock over several copies.
  const walls = [];
  for (let t = 0; t < trials; t++) {
    const enc = device.createCommandEncoder();
    for (let r = 0; r < 4; r++) enc.copyBufferToBuffer(src, 0, dst, 0, bytes);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    walls.push((performance.now() - t0) / 4);
  }
  const ws = summarize(walls);
  out.push({ test: 'copyBufferToBuffer_wall', bytesPerCopy: 2 * bytes, wallMs: ws, gbps: (2 * bytes) / ws.median / 1e6 });
  log('copyBufferToBuffer', `${((2 * bytes) / ws.median / 1e6).toFixed(1)} GB/s (read+write, wall)`);
  src.destroy();
  dst.destroy();
  uni.destroy();
  return out;
}

function gemmVariants(M, N) {
  const v = [];
  for (const [storage, accum] of [
    ['f32', 'f32'],
    ['f16', 'f16'],
    ['f16', 'f32'],
  ])
    for (const [tm4, tn4] of [
      [1, 1],
      [2, 2],
      [1, 2],
      [2, 1],
    ])
      if (M % (64 * tm4) === 0 && N % (64 * tn4) === 0) v.push({ storage, accum, tm4, tn4 });
  return v;
}

async function gemmTests(device, trials, shapes) {
  const out = [];
  for (const [label, M, N, K] of shapes) {
    const flops = 2 * M * N * K;
    const reps = Math.max(1, Math.min(64, Math.round(2e9 / flops)));
    const buffers = {};
    for (const type of ['f32', 'f16']) {
      if (type === 'f16' && !device.features.has('shader-f16')) continue;
      const size = type === 'f16' ? 2 : 4;
      buffers[type] = {
        A: randomBuffer(device, M * K * size, type),
        B: randomBuffer(device, K * N * size, type),
        C: device.createBuffer({ size: M * N * size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
      };
    }
    for (const variant of gemmVariants(M, N)) {
      const b = buffers[variant.storage];
      if (!b) continue;
      const name = `gemm_${variant.storage}acc${variant.accum}_${variant.tm4 * 4}x${variant.tn4 * 4}`;
      try {
        const { code, dispatch } = gemmKernel({ M, N, K, ...variant });
        const pipeline = await compile(device, code, name);
        const t = await timeDispatches(device, pipeline, bindGroup(device, pipeline, [b.A, b.B, b.C]), dispatch, { reps, trials });
        const s = summarize(t.gpuNs);
        const bytes = (M * K + K * N + M * N) * (variant.storage === 'f16' ? 2 : 4);
        out.push({
          test: 'gemm',
          shape: label,
          M,
          N,
          K,
          ...variant,
          kernel: name,
          flops,
          gpuNs: s,
          tflops: flops / s.median / 1e3,
          minBytes: bytes,
          gbpsAtMinBytes: bytes / s.median,
        });
      } catch (error) {
        out.push({ test: 'gemm', shape: label, M, N, K, ...variant, kernel: name, error: String(error.message ?? error) });
      }
    }
    const best = out.filter((r) => r.shape === label && r.tflops).sort((a, b) => b.tflops - a.tflops)[0];
    if (best) log('gemm', label, `${best.tflops.toFixed(3)} TFLOP/s best=${best.kernel} ${(best.gpuNs.median / 1e3).toFixed(1)}us`);
    for (const b of Object.values(buffers)) for (const buf of Object.values(b)) buf.destroy();
  }
  return out;
}

async function subgroupMatrixTests(device, adapter, trials, shapes) {
  const configs = [...(adapter.info.subgroupMatrixConfigs ?? [])].map((c) => ({
    componentType: c.componentType,
    resultComponentType: c.resultComponentType,
    M: c.M,
    N: c.N,
    K: c.K,
  }));
  const sg = adapter.info.subgroupMinSize;
  if (!device.features.has('chromium-experimental-subgroup-matrix')) return { available: false, configs, rows: [] };
  if (sg !== adapter.info.subgroupMaxSize) return { available: true, configs, rows: [], skipped: 'variable subgroup size' };
  const rows = [];
  for (const cfg of configs) {
    for (const [label, M, N, K] of shapes) {
      const flops = 2 * M * N * K;
      const reps = Math.max(1, Math.min(64, Math.round(2e9 / flops)));
      for (const [rt, ct] of [
        [2, 2],
        [4, 4],
        [4, 2],
      ]) {
        const name = `sgmatrix_${cfg.componentType}to${cfg.resultComponentType}_${cfg.M}x${cfg.N}x${cfg.K}_${rt}x${ct}`;
        try {
          const { code, dispatch } = subgroupMatrixGemmKernel({
            M,
            N,
            K,
            type: cfg.componentType,
            resultType: cfg.resultComponentType,
            mm: cfg.M,
            nn: cfg.N,
            kk: cfg.K,
            sg,
            rt,
            ct,
          });
          const pipeline = await compile(device, code, name);
          const size = cfg.componentType === 'f16' ? 2 : 4;
          const A = randomBuffer(device, M * K * size, cfg.componentType);
          const B = randomBuffer(device, K * N * size, cfg.componentType);
          const C = device.createBuffer({ size: M * N * (cfg.resultComponentType === 'f16' ? 2 : 4), usage: GPUBufferUsage.STORAGE });
          const t = await timeDispatches(device, pipeline, bindGroup(device, pipeline, [A, B, C]), dispatch, { reps, trials });
          const s = summarize(t.gpuNs);
          rows.push({ test: 'sgmatrix_gemm', shape: label, M, N, K, config: cfg, rt, ct, kernel: name, flops, gpuNs: s, tflops: flops / s.median / 1e3 });
          A.destroy();
          B.destroy();
          C.destroy();
        } catch (error) {
          rows.push({ test: 'sgmatrix_gemm', shape: label, M, N, K, config: cfg, rt, ct, kernel: name, error: String(error.message ?? error).slice(0, 600) });
        }
      }
      const best = rows.filter((r) => r.shape === label && r.config === cfg && r.tflops).sort((a, b) => b.tflops - a.tflops)[0];
      if (best) log('sgmatrix', label, `${best.tflops.toFixed(3)} TFLOP/s ${best.kernel}`);
    }
  }
  return { available: true, configs, rows };
}

/** Launch / sync overheads: what a tiny kernel costs per dispatch, pass, submit, readback. */
async function overheadTests(device, trials) {
  const pipeline = await compile(device, tinyKernel(), 'tiny');
  const buf = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const group = bindGroup(device, pipeline, [buf]);
  const n = 1000;
  const samePass = await timeDispatches(device, pipeline, group, [1, 1, 1], { reps: n, trials });
  // One pass per dispatch on one encoder, timestamped first-begin..last-end.
  const qs = device.createQuerySet({ type: 'timestamp', count: 2 });
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const passEnvelope = [];
  const passWall = [];
  for (let t = 0; t < trials; t++) {
    const enc = device.createCommandEncoder();
    for (let i = 0; i < n; i++) {
      const writes = i === 0 ? { querySet: qs, beginningOfPassWriteIndex: 0 } : i === n - 1 ? { querySet: qs, endOfPassWriteIndex: 1 } : undefined;
      const pass = enc.beginComputePass(writes ? { timestampWrites: writes } : {});
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    enc.resolveQuerySet(qs, 0, 2, resolve, 0);
    const read = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(resolve, 0, read, 0, 16);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    passWall.push((performance.now() - t0) / n);
    await read.mapAsync(GPUMapMode.READ);
    const ticks = new BigUint64Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    passEnvelope.push(Number(ticks[1] - ticks[0]) / n);
  }
  // Submit + onSubmittedWorkDone round trip with one tiny dispatch.
  const submitRtt = [];
  for (let t = 0; t < 50; t++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    submitRtt.push(performance.now() - t0);
  }
  // Dispatch + copy + mapAsync readback round trip (what an eval readback costs).
  const readback = [];
  const staging = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  for (let t = 0; t < 50; t++) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(buf, 0, staging, 0, 256);
    const t0 = performance.now();
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    staging.unmap();
    readback.push(performance.now() - t0);
  }
  // CPU cost to encode one dispatch in its own pass.
  const enc = device.createCommandEncoder();
  const t0 = performance.now();
  for (let i = 0; i < n; i++) {
    const pass = enc.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(1);
    pass.end();
  }
  const encodeUs = ((performance.now() - t0) * 1e3) / n;
  device.queue.submit([enc.finish()]);
  await device.queue.onSubmittedWorkDone();
  const result = {
    tinyDispatchSamePassGpuNs: summarize(samePass.gpuNs),
    tinyDispatchOwnPassGpuNs: summarize(passEnvelope),
    tinyDispatchOwnPassWallUs: summarize(passWall.map((v) => v * 1e3)),
    submitRoundTripMs: summarize(submitRtt),
    dispatchCopyMapReadbackMs: summarize(readback),
    cpuEncodeUsPerPassDispatch: encodeUs,
  };
  log('overhead', {
    samePassNs: result.tinyDispatchSamePassGpuNs.median,
    ownPassNs: result.tinyDispatchOwnPassGpuNs.median,
    submitRttMs: result.submitRoundTripMs.median,
    readbackMs: result.dispatchCopyMapReadbackMs.median,
    encodeUs,
  });
  buf.destroy();
  staging.destroy();
  qs.destroy();
  resolve.destroy();
  return result;
}

function parseShapes() {
  const raw = listParam('gemmShapes', null);
  if (!raw) return DEFAULT_GEMM_SHAPES;
  return raw.map((s) => {
    const [M, N, K] = s.split('x').map(Number);
    return [`${M}x${N}x${K}`, M, N, K];
  });
}

runMain(async () => {
  const tests = new Set(listParam('tests', ['fma', 'bandwidth', 'gemm', 'sgmatrix', 'overhead']));
  const trials = intParam('trials', 7, 3, 200);
  const shapes = parseShapes();
  const { adapter, device } = await getDevice();
  const identity = await deviceIdentity(adapter, device);
  log('device', identity.adapter);
  const result = { bench: 'peak', ok: true, device: identity, params: { tests: [...tests], trials, gemmShapes: shapes }, results: {} };
  if (tests.has('fma')) result.results.fma = await fmaTests(device, trials);
  if (tests.has('bandwidth')) result.results.bandwidth = await bandwidthTests(device, trials);
  if (tests.has('overhead')) result.results.overhead = await overheadTests(device, trials);
  if (tests.has('gemm')) result.results.gemm = await gemmTests(device, trials, shapes);
  if (tests.has('sgmatrix')) result.results.subgroupMatrix = await subgroupMatrixTests(device, adapter, trials, shapes);
  // Headline peaks: best measured values (the practical roofline used by roofline.mjs).
  const fma = result.results.fma ?? [];
  const bw = result.results.bandwidth ?? [];
  const gemmRows = [...(result.results.gemm ?? []), ...(result.results.subgroupMatrix?.rows ?? [])].filter((r) => r.tflops);
  result.peaks = {
    fmaF32Tflops: fma.find((r) => r.test === 'fma_f32')?.tflops ?? null,
    fmaF16Tflops: fma.find((r) => r.test === 'fma_f16')?.tflops ?? null,
    readGBps: bw.find((r) => r.test === 'read_vec4_f32')?.gbps ?? null,
    copyGBps: bw.find((r) => r.test === 'copy_vec4_f32')?.gbps ?? null,
    bestGemmTflops: Object.fromEntries(
      shapes.map(([label]) => [label, gemmRows.filter((r) => r.shape === label).sort((a, b) => b.tflops - a.tflops)[0]?.tflops ?? null]),
    ),
    bestGemmKernel: Object.fromEntries(
      shapes.map(([label]) => [label, gemmRows.filter((r) => r.shape === label).sort((a, b) => b.tflops - a.tflops)[0]?.kernel ?? null]),
    ),
  };
  device.destroy();
  return result;
});
