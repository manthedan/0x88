import { parseFen } from '../src/chess/board.ts';
import { moveToUci } from '../src/chess/moveCodec.ts';
import { searchRoot } from '../src/search/puct.ts';
import { SquareformerTvmjsWebgpuEvaluator } from '../src/nn/squareformerTvmjsWebgpuEvaluator.ts';
import { SquareFormerEvaluator } from '../src/nn/squareformerEvaluator.ts';
import * as ort from '../src/nn/ortRuntime.ts';
import { createBrowserSquareformerRuntimeEvaluator } from '../src/nn/browserRuntimeEvaluator.ts';

const root = '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/';
const variants = ['v2-shape-k16', 'v3-subgroups'];
const params = new URLSearchParams(location.search);
const repeats = Number(params.get('repeats') ?? 31);
const searchRepeats = Number(params.get('searchRepeats') ?? 7);
const visits = Number(params.get('visits') ?? 128);
const searchFixtureOffset = Number(params.get('searchFixtureOffset') ?? 0);
const searchFixtureCount = Number(params.get('searchFixtureCount') ?? 4);
const orderOffset = params.get('order') === 'ba' ? 1 : 0;
const order = (i) => ((i + orderOffset) % 2 ? [1, 0] : [0, 1]);
const log = (value) => {
  document.querySelector('#log').textContent += `${JSON.stringify(value)}\n`;
};
const hash = async (bytes) => Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), (x) => x.toString(16).padStart(2, '0')).join('');
const stats = (values) => {
  const s = [...values].sort((a, b) => a - b);
  const quantile = (q) => s[Math.floor((s.length - 1) * q)];
  return {
    count: s.length,
    median: s.length % 2 ? s[s.length >> 1] : (s[(s.length >> 1) - 1] + s[s.length >> 1]) / 2,
    p10: quantile(0.1),
    p90: quantile(0.9),
    min: s[0],
    max: s.at(-1),
  };
};
function compare(a, b) {
  if (a.length !== b.length) throw new Error('Evaluation length mismatch');
  let policy = 0,
    wdl = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i].policy.size !== b[i].policy.size) throw new Error('Legal policy size mismatch');
    for (const [k, p] of a[i].policy) {
      const q = b[i].policy.get(k);
      if (!Number.isFinite(p) || !Number.isFinite(q)) throw new Error('Invalid legal policy');
      policy = Math.max(policy, Math.abs(p - q));
    }
    for (let j = 0; j < 3; j++) {
      if (!Number.isFinite(a[i].wdl[j]) || !Number.isFinite(b[i].wdl[j])) throw new Error('Invalid WDL');
      wdl = Math.max(wdl, Math.abs(a[i].wdl[j] - b[i].wdl[j]));
    }
  }
  if (policy > 0.001 || wdl > 0.001) throw new Error(`Parity failed: ${policy}/${wdl}`);
  return { maxPolicyAbs: policy, maxWdlAbs: wdl, tolerance: 0.001 };
}

// Diagnostic-only API instrumentation: real timestamp queries on each actual
// TVM compute pass, without rewriting shaders or modifying the runtime bundle.
// Search samples disable timestamp writes/resolution entirely.
function installProfiler() {
  const original = GPUAdapter.prototype.requestDevice;
  const devices = [];
  const errors = [];
  GPUAdapter.prototype.requestDevice = async function (descriptor = {}) {
    if (!this.features.has('timestamp-query')) throw new Error('timestamp-query required for GPU timings');
    const device = await original.call(this, { ...descriptor, requiredFeatures: [...new Set([...(descriptor.requiredFeatures ?? []), 'timestamp-query'])] });
    device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
    device.lost.then((info) => {
      if (info.reason !== 'destroyed') errors.push(`Device lost: ${info.message}`);
    });
    const shaders = [];
    const createShader = device.createShaderModule.bind(device);
    device.createShaderModule = (descriptor) => {
      shaders.push({ code: descriptor.code });
      return createShader(descriptor);
    };
    const pipelineNames = new WeakMap();
    for (const method of ['createComputePipeline', 'createComputePipelineAsync']) {
      const create = device[method].bind(device);
      device[method] = (descriptor) => {
        const pipeline = create(descriptor);
        if (method.endsWith('Async'))
          return pipeline.then((p) => {
            pipelineNames.set(p, descriptor.compute.entryPoint);
            return p;
          });
        pipelineNames.set(pipeline, descriptor.compute.entryPoint);
        return pipeline;
      };
    }
    const profile = { device, shaders, active: null };
    const createEncoder = device.createCommandEncoder.bind(device);
    device.createCommandEncoder = (...args) => {
      const encoder = createEncoder(...args);
      const begin = encoder.beginComputePass.bind(encoder);
      encoder.beginComputePass = (descriptor = {}) => {
        if (profile.active) {
          const p = profile.active;
          if (p.count + 2 > 2048) throw new Error('Timestamp query capacity exceeded');
          descriptor = { ...descriptor, timestampWrites: { querySet: p.querySet, beginningOfPassWriteIndex: p.count++, endOfPassWriteIndex: p.count++ } };
        }
        const pass = begin(descriptor);
        if (profile.active) {
          const p = profile.active,
            index = p.count / 2 - 1;
          const setPipeline = pass.setPipeline.bind(pass);
          pass.setPipeline = (pipeline) => {
            p.names[index] = pipelineNames.get(pipeline);
            setPipeline(pipeline);
          };
        }
        return pass;
      };
      return encoder;
    };
    devices.push(profile);
    return device;
  };
  return {
    devices,
    errors,
    restore() {
      GPUAdapter.prototype.requestDevice = original;
    },
  };
}
async function gpuInvoke(evaluator, profile, boards) {
  const device = profile.device;
  const querySet = device.createQuerySet({ type: 'timestamp', count: 2048 });
  const resolveBuffer = device.createBuffer({ size: 16384, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16384, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const p = { querySet, count: 0, names: [] };
  try {
    profile.active = p;
    const start = performance.now();
    await evaluator.evaluateBatch(boards);
    const wallMs = performance.now() - start;
    profile.active = null;
    if (!p.count) throw new Error('No GPU timestamps');
    const encoder = device.createCommandEncoder();
    encoder.resolveQuerySet(querySet, 0, p.count, resolveBuffer, 0);
    encoder.copyBufferToBuffer(resolveBuffer, 0, read, 0, p.count * 8);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const ts = new BigUint64Array(read.getMappedRange(), 0, p.count);
    let busyNs = 0n,
      zeroPasses = 0;
    const kernels = {};
    for (let i = 0; i < p.count; i += 2) {
      if (ts[i + 1] < ts[i]) throw new Error('Reversed GPU timestamp');
      if (ts[i + 1] === ts[i]) zeroPasses++;
      const ns = ts[i + 1] - ts[i];
      busyNs += ns;
      const name = p.names[i / 2];
      if (!name) throw new Error('Missing timestamp pipeline identity');
      const k = (kernels[name] ??= { count: 0, gpuMs: 0, zeroPasses: 0 });
      k.count++;
      k.gpuMs += Number(ns) / 1e6;
      k.zeroPasses += ns === 0n ? 1 : 0;
    }
    if (busyNs <= 0n || ts[p.count - 1] <= ts[0]) throw new Error('Nonpositive whole-invoke GPU timestamp');
    const row = { wallMs, busyMs: Number(busyNs) / 1e6, envelopeMs: Number(ts[p.count - 1] - ts[0]) / 1e6, passes: p.count / 2, zeroPasses, kernels };
    read.unmap();
    return row;
  } finally {
    profile.active = null;
    read.destroy();
    resolveBuffer.destroy();
    querySet.destroy();
  }
}
async function shaderEvidence(profile) {
  return Promise.all(
    profile.shaders.map(async ({ code }) => ({
      kernel: code.match(/fn (\w+)\(/)?.[1],
      sha256: await hash(new TextEncoder().encode(code)),
      subgroupOps: [...new Set(code.match(/\bsubgroup\w+(?=\()/g) ?? [])],
      workgroup: code.match(/@workgroup_size\(([^)]+)\)/)?.[1],
    })),
  );
}
async function run() {
  if (
    ![repeats, searchRepeats, visits, searchFixtureCount].every((n) => Number.isSafeInteger(n) && n > 0) ||
    !Number.isSafeInteger(searchFixtureOffset) ||
    searchFixtureOffset < 0
  )
    throw new Error('Invalid sample counts');
  if (!crossOriginIsolated) throw new Error('Cross-origin isolation required');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const info = adapter.info;
  if (!adapter.features.has('subgroups') || info.subgroupMinSize !== 32 || info.subgroupMaxSize !== 32)
    throw new Error('This measurement requires fixed size-32 subgroups; do not time fallback as v3');
  const meta = await (await fetch('/models/bt4_soap_rem_c19000_final.meta.json')).json();
  const fixtures = await (await fetch('/fixtures/lc0/fen_only.json')).json();
  const boards = fixtures.slice(0, 16).map((f) => parseFen(f.fen));
  while (boards.length < 16) boards.push(boards[boards.length - 1]);
  const profiler = installProfiler();
  const evaluators = [];
  let reference;
  try {
    for (const v of variants) {
      const evaluator = await SquareformerTvmjsWebgpuEvaluator.create(`${root}${v}/manifest.json`, meta);
      if (!evaluator.manifestUrl.endsWith(`/${v}/manifest.json`)) throw new Error('Unexpected runtime fallback');
      evaluators.push(evaluator);
    }
    ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
    reference = await SquareFormerEvaluator.create('/models/bt4_soap_rem_c19000_final.onnx', meta);
    const evals = [];
    for (const e of [...evaluators, reference]) evals.push(await e.evaluateBatch(boards));
    const parity = { v3VsV2: compare(evals[1], evals[0]), v2VsOrt: compare(evals[0], evals[2]), v3VsOrt: compare(evals[1], evals[2]) };
    log({ parity });
    for (let i = 0; i < 5; i++) for (const e of evaluators) await e.evaluateBatch(boards);
    const invokes = [[], []];
    for (let i = 0; i < repeats; i++) {
      for (const k of order(i)) invokes[k].push(await gpuInvoke(evaluators[k], profiler.devices[k], boards));
    }
    const search = [],
      ortSearch = [];
    const searchFixtures = fixtures.slice(searchFixtureOffset, searchFixtureOffset + searchFixtureCount);
    if (!searchFixtures.length) throw new Error('No search fixtures selected');
    for (const fixture of searchFixtures) {
      const b = parseFen(fixture.fen);
      // Warm search code/allocations outside measured samples.
      for (const e of evaluators) await searchRoot(b, e, { visits, batchSize: 16, temperature: 0, includePv: true, pvDepth: 8 });
      for (let i = 0; i < searchRepeats; i++) {
        const results = [];
        for (const k of order(i)) {
          const start = performance.now();
          const r = await searchRoot(b, evaluators[k], { visits, batchSize: 16, temperature: 0, includePv: true, pvDepth: 8 });
          results[k] = {
            wallMs: performance.now() - start,
            move: r.move ? moveToUci(r.move) : null,
            pv: r.principalVariation?.map((p) => moveToUci(p.move)),
            visits: r.visits,
            stats: r.stats,
          };
        }
        const same = results[0].move === results[1].move && JSON.stringify(results[0].pv) === JSON.stringify(results[1].pv);
        if (!same) throw new Error(`Search parity failed: ${JSON.stringify(results)}`);
        search.push({ fixture: fixture.id, repeat: i, results });
      }
      const r = await searchRoot(b, reference, { visits, batchSize: 16, temperature: 0, includePv: true, pvDepth: 8 });
      const expected = search.at(-1).results[0];
      const actual = {
        fixture: fixture.id,
        move: r.move ? moveToUci(r.move) : null,
        pv: r.principalVariation?.map((p) => moveToUci(p.move)),
        visits: r.visits,
      };
      ortSearch.push(actual);
      if (actual.move !== expected.move || JSON.stringify(actual.pv) !== JSON.stringify(expected.pv)) throw new Error('ORT search move/PV mismatch');
      log({ fixture: search.at(-1).fixture, move: expected.move, pv: expected.pv });
    }
    const shaders = await Promise.all(profiler.devices.slice(0, 2).map(shaderEvidence));
    // Real destroy-during-flight, reject subsequent calls, recreate and run.
    for (let k = 0; k < 2; k++) {
      let settled = false;
      const flight = evaluators[k].evaluateBatch(boards);
      void flight.then(
        () => {
          settled = true;
        },
        () => {
          settled = true;
        },
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
      if (settled) throw new Error('Lifecycle probe did not overlap an in-flight evaluation');
      evaluators[k].destroy();
      await flight;
      const lost = await profiler.devices[k].device.lost;
      if (lost.reason !== 'destroyed') throw new Error('Lifecycle device was not destroyed');
      await evaluators[k].evaluate(boards[0]).then(
        () => {
          throw new Error('Destroyed evaluator accepted work');
        },
        (error) => {
          if (!String(error).includes('destroyed')) throw error;
        },
      );
      const recreated = await SquareformerTvmjsWebgpuEvaluator.create(`${root}${variants[k]}/manifest.json`, meta);
      compare(await recreated.evaluateBatch(boards), evals[k]);
      recreated.destroy();
    }
    if (profiler.errors.length) throw new Error(profiler.errors.join('\n'));
    const artifacts = await Promise.all(
      variants.map(async (v) => {
        const url = `${root}${v}/manifest.json`;
        const bytes = await (await fetch(url)).arrayBuffer();
        return { version: v, manifestSha256: await hash(bytes), manifest: JSON.parse(new TextDecoder().decode(bytes)) };
      }),
    );
    const scriptUrl = document.querySelector('script[type="module"][src]').src;
    const executedBundle = { url: scriptUrl, sha256: await hash(await (await fetch(scriptUrl)).arrayBuffer()) };
    const result = {
      schema: 'centipawn-tvmjs-subgroups.v1',
      ok: true,
      executedBundle,
      orderOffset,
      date: new Date().toISOString(),
      browser: navigator.userAgent,
      adapter: { vendor: info.vendor, architecture: info.architecture, subgroupMinSize: info.subgroupMinSize, subgroupMaxSize: info.subgroupMaxSize },
      batch: 16,
      repeats,
      searchRepeats,
      visits,
      searchFixtureOffset,
      searchFixtureCount,
      fixtures: fixtures.slice(0, 16),
      parity,
      lifecycle: 'PASS',
      gpuErrors: profiler.errors,
      artifacts,
      shaders,
      invokes,
      search,
      ortSearch,
      summary: variants.map((v, k) => ({
        variant: v,
        invokeBusyMs: stats(invokes[k].map((r) => r.busyMs)),
        invokeEnvelopeMs: stats(invokes[k].map((r) => r.envelopeMs)),
        invokeWallMs: stats(invokes[k].map((r) => r.wallMs)),
        searchWallMs: stats(search.map((r) => r.results[k].wallMs)),
      })),
    };
    window.subgroupsResult = result;
    log({ summary: result.summary, ok: true });
  } finally {
    for (const e of evaluators) e.destroy();
    reference?.destroy();
    ort.setRequestedOrtExecutionProviderForCurrentThread(null);
    profiler.restore();
  }
}
// Browser integration gate for the factory's selected artifact/audit identity.
window.checkSubgroupRouting = async () => {
  const original = navigator.gpu.requestAdapter.bind(navigator.gpu);
  const rows = [],
    evaluations = [];
  try {
    for (const mode of ['candidate', 'variable-size-fallback', 'default']) {
      const requestedFeatures = [];
      let device;
      navigator.gpu.requestAdapter = async (...args) => {
        const adapter = await original(...args);
        return {
          features: adapter.features,
          limits: adapter.limits,
          info: mode === 'variable-size-fallback' ? { subgroupMinSize: 32, subgroupMaxSize: 64 } : adapter.info,
          requestDevice: async (descriptor) => {
            requestedFeatures.push(descriptor.requiredFeatures);
            device = await adapter.requestDevice(descriptor);
            return device;
          },
        };
      };
      const result = await createBrowserSquareformerRuntimeEvaluator(
        {
          onnx: '/models/bt4_soap_rem_c19000_final.onnx',
          meta: '/models/bt4_soap_rem_c19000_final.meta.json',
          runtime: mode === 'default' ? 'auto' : 'tvmjs-webgpu',
          ...(mode === 'default' ? {} : { manifestUrl: `${root}v3-subgroups/manifest.json` }),
        },
        { fallback: false },
      );
      try {
        const expected = mode === 'candidate' ? 'v3-subgroups' : 'v2-shape-k16';
        if (!result.manifestUrl.endsWith(`/${expected}/manifest.json`) || !result.runtimeConfigId.endsWith(expected))
          throw new Error('Factory selected identity mismatch');
        if (JSON.stringify(requestedFeatures) !== JSON.stringify([mode === 'candidate' ? ['subgroups'] : []]))
          throw new Error('Unexpected device feature request');
        evaluations.push(await result.evaluator.evaluate(parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')));
        rows.push({ mode, manifestUrl: result.manifestUrl, runtimeConfigId: result.runtimeConfigId, requestedFeatures });
      } finally {
        result.evaluator.destroy();
        if ((await device.lost).reason !== 'destroyed') throw new Error('Routing probe device was not destroyed');
      }
    }
    const result = { ok: true, rows, parity: compare(evaluations, [evaluations[0], evaluations[0], evaluations[0]]) };
    window.subgroupsRoutingResult = result;
    return result;
  } finally {
    navigator.gpu.requestAdapter = original;
  }
};

document.querySelector('#run').onclick = async () => {
  const button = document.querySelector('#run');
  button.disabled = true;
  window.subgroupsError = undefined;
  window.subgroupsResult = undefined;
  try {
    await run();
  } catch (error) {
    window.subgroupsError = String(error.stack ?? error);
    log({ error: window.subgroupsError });
  } finally {
    button.disabled = false;
  }
};
