// Self-contained WebGPU / WASM device probe for the kernel hill-climb.
// Contract: sets window.__kernelBenchResult (object) or window.__kernelBenchError (string).
// Query params:
//   bench=0        skip GPU microbenchmarks (adapter/feature probe only)
//   ms=2000        total microbench budget in ms (split FMA / bandwidth)
//   power=high-performance|low-power   requestAdapter powerPreference
//   mib=64         bandwidth buffer size in MiB (each of src/dst)

const q = new URLSearchParams(location.search);
const BENCH = q.get('bench') !== '0';
const BUDGET_MS = Number(q.get('ms') ?? 2000);
const POWER = q.get('power') ?? 'high-performance';
const BW_MIB = Number(q.get('mib') ?? 64);

const out = document.getElementById('out');
const log = (s) => { out.textContent += `\n${s}`; console.log(s); };

const WANT_FEATURES = [
  'shader-f16', 'subgroups', 'timestamp-query', 'chromium-experimental-subgroup-matrix',
  'chromium-experimental-timestamp-query-inside-passes', 'chromium-experimental-subgroups',
  'chromium-experimental-shader-int64',
];

function plain(obj) {
  // Copy own + prototype getters (GPUAdapterInfo / GPUSupportedLimits are interface objects).
  const o = {};
  if (!obj) return o;
  for (let p = obj; p && p !== Object.prototype; p = Object.getPrototypeOf(p)) {
    for (const k of Object.getOwnPropertyNames(p)) {
      if (k === 'constructor' || k in o) continue;
      let v; try { v = obj[k]; } catch { continue; }
      if (typeof v === 'function') continue;
      o[k] = Array.isArray(v) ? v.map((x) => (typeof x === 'object' && x ? plain(x) : x))
        : typeof v === 'object' && v !== null ? (v instanceof Set ? [...v] : plain(v)) : v;
    }
  }
  return o;
}

function wasmFeatures() {
  const v = (bytes) => { try { return WebAssembly.validate(new Uint8Array(bytes)); } catch { return false; } };
  return {
    // byte sequences from GoogleChromeLabs/wasm-feature-detect
    simd: v([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]),
    relaxedSimd: v([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]),
    threads: typeof SharedArrayBuffer === 'function' && self.crossOriginIsolated === true,
    crossOriginIsolated: self.crossOriginIsolated === true,
    hardwareConcurrency: navigator.hardwareConcurrency,
  };
}

async function probe() {
  const result = {
    probe: 'device-probe', version: 1, startedAt: new Date().toISOString(),
    userAgent: navigator.userAgent, wasm: wasmFeatures(), webgpu: null, bench: null,
  };
  if (!navigator.gpu) { result.webgpu = { available: false, reason: 'navigator.gpu missing' }; return result; }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: POWER });
  if (!adapter) { result.webgpu = { available: false, reason: 'requestAdapter() returned null' }; return result; }
  const features = [...adapter.features].sort();
  const info = plain(adapter.info);
  const limits = plain(adapter.limits);
  const wgslLanguageFeatures = navigator.gpu.wgslLanguageFeatures ? [...navigator.gpu.wgslLanguageFeatures].sort() : null;
  const want = Object.fromEntries(WANT_FEATURES.map((f) => [f, adapter.features.has(f)]));
  result.webgpu = {
    available: true, isFallbackAdapter: adapter.isFallbackAdapter ?? info.isFallbackAdapter ?? null,
    preferredCanvasFormat: navigator.gpu.getPreferredCanvasFormat?.(),
    info, features, want, limits, wgslLanguageFeatures,
  };
  log(`adapter: ${info.vendor} / ${info.architecture} / ${info.description}`);
  log(`features: ${features.join(', ')}`);

  // Device with every requestable feature we care about, max limits we use.
  const requiredFeatures = WANT_FEATURES.filter((f) => adapter.features.has(f));
  const requiredLimits = {};
  for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize', 'maxComputeWorkgroupStorageSize',
    'maxComputeInvocationsPerWorkgroup', 'maxComputeWorkgroupSizeX']) requiredLimits[k] = adapter.limits[k];
  let device;
  try {
    device = await adapter.requestDevice({ requiredFeatures, requiredLimits });
  } catch (e) {
    result.webgpu.deviceError = String(e);
    return result;
  }
  result.webgpu.deviceFeatures = [...device.features].sort();
  if (device.adapterInfo) result.webgpu.deviceAdapterInfo = plain(device.adapterInfo);
  const lost = { v: null }; device.lost.then((l) => { lost.v = `${l.reason}: ${l.message}`; });
  device.addEventListener?.('uncapturederror', (ev) => log(`uncapturederror: ${ev.error.message}`));

  // Smoke-compile subgroup / f16 shaders to prove the features actually work.
  result.webgpu.shaderChecks = await shaderChecks(device);

  if (BENCH) {
    try { result.bench = await microbench(device); } catch (e) { result.bench = { error: String(e?.stack ?? e) }; }
  }
  if (lost.v) result.webgpu.deviceLost = lost.v;
  device.destroy();
  result.finishedAt = new Date().toISOString();
  return result;
}

async function shaderChecks(device) {
  const checks = {};
  const tryCompile = async (name, code) => {
    device.pushErrorScope('validation');
    const m = device.createShaderModule({ code });
    const ci = await m.getCompilationInfo();
    const err = await device.popErrorScope();
    const msgs = ci.messages.filter((x) => x.type === 'error').map((x) => x.message);
    checks[name] = err || msgs.length ? { ok: false, error: err?.message ?? msgs.join('; ') } : { ok: true };
  };
  if (device.features.has('shader-f16')) await tryCompile('f16', `enable f16;
    @group(0) @binding(0) var<storage, read_write> b: array<f16>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) { b[g.x] = fma(b[g.x], 2.0h, 1.0h); }`);
  if (device.features.has('subgroups')) await tryCompile('subgroups', `enable subgroups;
    @group(0) @binding(0) var<storage, read_write> b: array<f32>;
    @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u, @builtin(subgroup_size) s: u32) { b[g.x] = subgroupAdd(b[g.x]) + f32(s); }`);
  if (device.features.has('chromium-experimental-subgroup-matrix')) {
    await tryCompile('subgroupMatrixF32', `enable chromium_experimental_subgroup_matrix;
      @group(0) @binding(0) var<storage, read_write> b: array<f32>;
      @compute @workgroup_size(64) fn main() {
        let m = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(&b, 0, false, 8);
        let n = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(&b, 64, false, 8);
        var r = subgroup_matrix_result<f32, 8, 8>();
        r = subgroupMatrixMultiplyAccumulate(m, n, r);
        subgroupMatrixStore(&b, 128, r, false, 8);
      }`);
    if (device.features.has('shader-f16')) await tryCompile('subgroupMatrixF16', `enable f16; enable chromium_experimental_subgroup_matrix;
      @group(0) @binding(0) var<storage, read_write> b: array<f16>;
      @group(0) @binding(1) var<storage, read_write> c: array<f32>;
      @compute @workgroup_size(64) fn main() {
        let m = subgroupMatrixLoad<subgroup_matrix_left<f16, 16, 16>>(&b, 0, false, 16);
        let n = subgroupMatrixLoad<subgroup_matrix_right<f16, 16, 16>>(&b, 256, false, 16);
        var r = subgroup_matrix_result<f32, 16, 16>();
        r = subgroupMatrixMultiplyAccumulate(m, n, r);
        subgroupMatrixStore(&c, 0, r, false, 16);
      }`);
  }
  return checks;
}

// ---------------------------------------------------------------- microbench
async function microbench(device) {
  const hasTs = device.features.has('timestamp-query');
  const timer = makeTimer(device, hasTs);
  const res = { timing: hasTs ? 'timestamp-query' : 'wall-clock(onSubmittedWorkDone)', budgetMs: BUDGET_MS };
  const t0 = performance.now();
  res.fmaF32 = await fmaBench(device, timer, 'f32', BUDGET_MS * 0.4);
  if (device.features.has('shader-f16')) res.fmaF16 = await fmaBench(device, timer, 'f16', BUDGET_MS * 0.15);
  res.bandwidth = await bwBench(device, timer, BUDGET_MS * 0.4);
  res.totalMs = +(performance.now() - t0).toFixed(1);
  return res;
}

function makeTimer(device, hasTs) {
  let qs, resolveBuf, readBuf;
  if (hasTs) {
    qs = device.createQuerySet({ type: 'timestamp', count: 2 });
    resolveBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    readBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }
  // Runs encodeFn(pass) inside one compute pass; returns {gpuMs, wallMs}.
  return async (encodeFn) => {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass(hasTs ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {});
    encodeFn(pass);
    pass.end();
    if (hasTs) { enc.resolveQuerySet(qs, 0, 2, resolveBuf, 0); enc.copyBufferToBuffer(resolveBuf, 0, readBuf, 0, 16); }
    const w0 = performance.now();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    const wallMs = performance.now() - w0;
    let gpuMs = null;
    if (hasTs) {
      await readBuf.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
      gpuMs = Number(t[1] - t[0]) / 1e6;
      if (!(gpuMs > 0)) gpuMs = null; // quantized / invalid
    }
    return { gpuMs, wallMs };
  };
}

function stats(xs) {
  const s = [...xs].sort((a, b) => a - b);
  return { n: s.length, best: s[0], median: s[Math.floor(s.length / 2)], worst: s[s.length - 1] };
}

async function fmaBench(device, timer, type, budgetMs) {
  const T = type; const V = `vec4<${T}>`;
  const code = `${T === 'f16' ? 'enable f16;' : ''}
  struct P { iters: u32 }
  @group(0) @binding(0) var<storage, read_write> outb: array<${V}>;
  @group(0) @binding(1) var<uniform> p: P;
  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) g: vec3u) {
    let s = ${T}(f32(g.x & 255u) * 1e-3);
    var a0 = ${V}(s); var a1 = ${V}(s + ${T}(0.1)); var a2 = ${V}(s + ${T}(0.2)); var a3 = ${V}(s + ${T}(0.3));
    var a4 = ${V}(s + ${T}(0.4)); var a5 = ${V}(s + ${T}(0.5)); var a6 = ${V}(s + ${T}(0.6)); var a7 = ${V}(s + ${T}(0.7));
    let m = ${V}(${T}(0.999)); let c = ${V}(${T}(0.001));
    for (var i = 0u; i < p.iters; i++) {
      a0 = fma(a0, m, c); a1 = fma(a1, m, c); a2 = fma(a2, m, c); a3 = fma(a3, m, c);
      a4 = fma(a4, m, c); a5 = fma(a5, m, c); a6 = fma(a6, m, c); a7 = fma(a7, m, c);
    }
    outb[g.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  }`;
  const WG = 8192, threads = WG * 256;
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module: device.createShaderModule({ code }), entryPoint: 'main' } });
  const outBuf = device.createBuffer({ size: threads * (T === 'f16' ? 8 : 16), usage: GPUBufferUsage.STORAGE });
  const uni = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outBuf } }, { binding: 1, resource: { buffer: uni } }] });
  const run = (iters) => {
    device.queue.writeBuffer(uni, 0, new Uint32Array([iters, 0, 0, 0]));
    return timer((pass) => { pass.setPipeline(pipeline); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(WG); });
  };
  // Calibrate: grow iters until one dispatch takes >= 15 ms (cap 60 ms to stay far from watchdogs).
  let iters = 32, r = await run(iters);
  while ((r.gpuMs ?? r.wallMs) < 15 && iters < (1 << 20)) { iters *= 2; r = await run(iters); }
  const flopsPerDispatch = threads * iters * 8 * 4 * 2;
  const gpu = [], wall = [];
  const end = performance.now() + budgetMs;
  while (performance.now() < end || gpu.length + wall.length < 6) {
    r = await run(iters); wall.push(r.wallMs); if (r.gpuMs) gpu.push(r.gpuMs);
    if (wall.length >= 200) break;
  }
  const best = gpu.length ? Math.min(...gpu) : Math.min(...wall);
  const med = (gpu.length ? stats(gpu) : stats(wall)).median;
  outBuf.destroy(); uni.destroy();
  return {
    type: T, threads, iters, flopsPerDispatch, reps: wall.length,
    gpuMs: gpu.length ? stats(gpu) : null, wallMs: stats(wall),
    tflopsBest: +(flopsPerDispatch / best / 1e9).toFixed(3), tflopsMedian: +(flopsPerDispatch / med / 1e9).toFixed(3),
  };
}

async function bwBench(device, timer, budgetMs) {
  const maxBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
  const bytes = Math.min(BW_MIB * 1024 * 1024, maxBytes - (maxBytes % 4096));
  const n = bytes / 16;
  const WGS = 256; const groups = Math.ceil(n / WGS);
  const gx = Math.min(groups, 32768), gy = Math.ceil(groups / gx);
  const code = `
  @group(0) @binding(0) var<storage, read> src: array<vec4f>;
  @group(0) @binding(1) var<storage, read_write> dst: array<vec4f>;
  @compute @workgroup_size(${WGS})
  fn copy(@builtin(global_invocation_id) g: vec3u) {
    let i = g.x + g.y * ${gx * WGS}u;
    if (i < ${n}u) { dst[i] = src[i] * 1.0001; }
  }`;
  const module = device.createShaderModule({ code });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'copy' } });
  const src = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE });
  const bg = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: src } }, { binding: 1, resource: { buffer: dst } }] });
  const K = 8; // dispatches per pass
  const run = () => timer((pass) => { pass.setPipeline(pipeline); pass.setBindGroup(0, bg); for (let k = 0; k < K; k++) pass.dispatchWorkgroups(gx, gy); });
  await run(); // warm
  const gpu = [], wall = [];
  const end = performance.now() + budgetMs;
  while (performance.now() < end || wall.length < 5) {
    const r = await run(); wall.push(r.wallMs); if (r.gpuMs) gpu.push(r.gpuMs);
    if (wall.length >= 200) break;
  }
  const moved = 2 * bytes * K; // read + write
  const best = gpu.length ? Math.min(...gpu) : Math.min(...wall);
  const med = (gpu.length ? stats(gpu) : stats(wall)).median;
  src.destroy(); dst.destroy();
  return {
    kernel: 'vec4f copy*scale (read+write)', bufferMiB: bytes / 1048576, dispatchesPerPass: K, reps: wall.length,
    gpuMs: gpu.length ? stats(gpu) : null, wallMs: stats(wall),
    gbpsBest: +(moved / best / 1e6).toFixed(1), gbpsMedian: +(moved / med / 1e6).toFixed(1),
  };
}

probe().then((r) => {
  window.__kernelBenchResult = r;
  log(`done: ${JSON.stringify(r.bench ? { fma: r.bench.fmaF32?.tflopsBest, bw: r.bench.bandwidth?.gbpsBest } : r.webgpu?.want)}`);
}, (e) => {
  window.__kernelBenchError = String(e?.stack ?? e);
  log(`error: ${window.__kernelBenchError}`);
});
