// ORT WebGPU (native C++ EP, the shipped path) timing + per-kernel profile.
//
// Uses the product session factory (src/nn/ortRuntime.ts createOrtSession:
// dequant fold on, graphOptimizationLevel 'all', built-mode /ort/ glue) with
// the EP forced to 'webgpu' so a silent WASM fallback cannot be mistaken for a
// GPU number. ORT 1.27's env.webgpu.profiling.ondata is JSEP-only, so kernel
// attribution comes from the runtime-agnostic WebGPU API profiler.
//
// Query params:
//   preset=lc0-small|lc0-small-b8|t3|bt4|lqo|maia3|maia3-grid  (or model=URL & batch=N)
//   warmup=N (5)  iters=N (30)  profileInvokes=N (5, 0 disables)
//   parity=wasm   also run the same inputs on an ORT WASM session of the REFERENCE model
//                 and gate max abs diffs (fails the run on mismatch)
//   referenceModel=URL  reference for parity (default: the preset's shipped model,
//                 so a candidate passed via model= is judged against the original)
//   ortDequantFold=0  run the graph as shipped (QDQ dequant per invoke)
//   ab=PRESET     paired comparison: also load PRESET's model and interleave
//                 A/B blocks (ABAB...) in this page session; reports the
//                 per-position wall ratio and GPU-busy ratio per round, which
//                 holds up under host load where absolute wall times do not
//   abRounds=N (20)  abBlock=N (3 invokes per arm per round)
import { parseFen } from '../../src/chess/board.ts';
import { encodeLc0Classical112 } from '../../src/lc0/encoder112.ts';
import { boardToMaia3Tokens } from '../../src/lc0/maia3.ts';
import * as ort from '../../src/nn/ortRuntime.ts';
import { deviceIdentity, intParam, log, params, runMain, summarize } from './common.mjs';
import { ortParity, profileComplete, runChunked } from './gates.mjs';
import { aggregateProfile, installGpuProfiler, passModeProfile } from './gpu-profiler.mjs';

const PRESETS = {
  'lc0-small': {
    model: '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx',
    batch: 1,
    family: 't1-256x10',
    shipped: 'Lc0 Small (Play/Analysis/Arena default), search batch 1',
    // Re-derived over the full corpus (baseline policy 0.0996, WDL 0.0027,
    // MLH 0.966); the old 0.06/0.3 were 2x a start-position-only baseline.
    parityMaxAbs: { '/output/policy': 0.2, '/output/wdl': 0.006, '/output/mlh': 2 },
  },
  'lc0-small-b8': {
    model: '/models/lc0/t1-256x10-distilled-swa-2432500.batch8.f16.qdq8.onnx',
    batch: 8,
    family: 't1-256x10',
    shipped: 'not shipped: t1 qdq8 b8 export (batching what-if)',
    parityMaxAbs: { '/output/policy': 0.2, '/output/wdl': 0.006, '/output/mlh': 0.8 },
  },
  t3: {
    model: '/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx',
    batch: 8,
    family: 't3-512x15x16h',
    shipped: 'Lc0 t3-512 (Arena/Analysis), batch 8',
    parityMaxAbs: { '/output/policy': 0.15, '/output/wdl': 0.007, '/output/mlh': 0.5 },
  },
  bt4: {
    model: '/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch8.f16.qdq8.onnx',
    batch: 8,
    family: 'BT4-1024x15x32h',
    shipped: 'Lc0 BT4-it332 (Play/Analysis/Arena), batch 8',
    parityMaxAbs: { '/output/policy': 0.6, '/output/wdl': 0.006, '/output/mlh': 2 },
  },
  lqo: {
    model: '/models/lc0/lqo_v2.f16.qdq8.onnx',
    batch: 8,
    family: 'lqo_v2',
    shipped: 'Leela Queen Odds (Play), batch 8',
    parityMaxAbs: { '/output/policy': 2.0, '/output/wdl': 0.02, '/output/mlh': 0.3 },
  },
  maia3: {
    model: '/models/maia3/maia3_simplified.qdq8.onnx',
    batch: 1,
    family: 'maia3',
    shipped: 'Maia3 single eval (Play move)',
    // Full-corpus baseline: move 0.641, value 0.031 (old limits were start-position-only).
    parityMaxAbs: { logits_move: 1.3, logits_value: 0.06 },
  },
  'maia3-grid': {
    model: '/models/maia3/maia3_simplified.qdq8.onnx',
    batch: 11,
    family: 'maia3',
    shipped: 'Maia3 Elo-grid evaluateConditions (11 conditions)',
    parityMaxAbs: { logits_move: 0.9, logits_value: 0.2 },
  },
};

function f32ToF16(values) {
  const out = new Uint16Array(values.length);
  const f = new Float32Array(1);
  const u = new Uint32Array(f.buffer);
  for (let i = 0; i < values.length; i++) {
    f[0] = values[i];
    const x = u[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 0xff) - 127 + 15;
    const mant = x & 0x7fffff;
    if (exp <= 0) out[i] = sign;
    else if (exp >= 31) out[i] = sign | 0x7c00;
    else out[i] = sign | (exp << 10) | Math.round(mant / 8192);
  }
  return out;
}

async function fixtureCorpus() {
  const fixtures = await (await fetch('/fixtures/lc0/fen_only.json')).json();
  const fens = fixtures.map((f) => f.fen).filter(Boolean);
  if (!fens.length) throw new Error('ort-profile: empty fixture corpus');
  return fens;
}

async function loadBoards(n) {
  const fens = await fixtureCorpus();
  return Array.from({ length: n }, (_, i) => fens[i % fens.length]);
}

async function buildFeeds(session, batch, explicitFens = null) {
  const fens = explicitFens ?? (await loadBoards(batch));
  if (fens.length !== batch) throw new Error(`buildFeeds: ${fens.length} positions for batch ${batch}`);
  const feeds = {};
  for (const meta of session.inputMetadata) {
    const name = meta.name;
    const type = meta.type ?? 'float32';
    let data;
    let shape;
    if (/planes/.test(name)) {
      data = new Float32Array(batch * 112 * 64);
      for (const [i, fen] of fens.entries()) data.set(encodeLc0Classical112(fen).planes, i * 112 * 64);
      shape = [batch, 112, 8, 8];
    } else if (name === 'tokens') {
      data = new Float32Array(batch * 64 * 12);
      for (const [i, fen] of fens.entries()) data.set(boardToMaia3Tokens(parseFen(fen)), i * 64 * 12);
      shape = [batch, 64, 12];
    } else if (/elo/.test(name)) {
      data = Float32Array.from({ length: batch }, (_, i) => 1100 + 100 * i);
      shape = [batch];
    } else throw new Error(`ort-profile: no input generator for ${name} (${type})`);
    feeds[name] = type === 'float16' ? new ort.Tensor('float16', f32ToF16(data), shape) : new ort.Tensor('float32', data, shape);
  }
  return { feeds, fens };
}

async function outputsToArrays(outputs) {
  const out = {};
  for (const [name, tensor] of Object.entries(outputs)) {
    const data = tensor.location && tensor.location !== 'cpu' ? await tensor.getData() : tensor.data;
    out[name] = data instanceof Uint16Array && tensor.type === 'float16' ? Float32Array.from(data, f16BitsToF32) : Float32Array.from(data);
  }
  return out;
}

function f16BitsToF32(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/**
 * Interleaved A/B in one page session: per round, `block` timed invokes of A
 * then of B (order alternates each round), plus pass-mode GPU busy for both
 * arms. Ratios are A per position / B per position, one per round.
 */
async function pairedComparison(prof, a, bPresetName) {
  const bPreset = PRESETS[bPresetName];
  if (!bPreset) throw new Error(`ab: unknown preset ${bPresetName}`);
  const rounds = intParam('abRounds', 20, 2, 500);
  const block = intParam('abBlock', 3, 1, 100);
  const bSession = await ort.createOrtSession(bPreset.model);
  const b = { session: bSession, feeds: (await buildFeeds(bSession, bPreset.batch)).feeds, batch: bPreset.batch, name: bPresetName };
  for (let i = 0; i < 5; i++) await b.session.run(b.feeds);
  const arms = { a, b };
  const wall = { a: [], b: [] };
  const busy = { a: [], b: [] };
  const wallRatio = [];
  const busyRatio = [];
  for (let r = 0; r < rounds; r++) {
    const order = r % 2 ? ['b', 'a'] : ['a', 'b'];
    const roundWall = {};
    const roundBusy = {};
    for (const k of order) {
      const arm = arms[k];
      const ms = [];
      for (let i = 0; i < block; i++) {
        const t = performance.now();
        await arm.session.run(arm.feeds);
        ms.push(performance.now() - t);
      }
      roundWall[k] = median(ms) / arm.batch;
      if (prof?.supported) roundBusy[k] = (await passModeProfile(prof, () => arm.session.run(arm.feeds), block)).busyMs / arm.batch;
    }
    wall.a.push(roundWall.a);
    wall.b.push(roundWall.b);
    wallRatio.push(roundWall.a / roundWall.b);
    if (roundBusy.a > 0 && roundBusy.b > 0) {
      busy.a.push(roundBusy.a);
      busy.b.push(roundBusy.b);
      busyRatio.push(roundBusy.a / roundBusy.b);
    }
  }
  await ort.releaseOrtSession(bSession);
  const out = {
    a: { name: a.name, batch: a.batch },
    b: { name: b.name, batch: b.batch, model: bPreset.model },
    rounds,
    block,
    perPositionWallMs: { a: summarize(wall.a), b: summarize(wall.b) },
    wallRatioAoverB: summarize(wallRatio),
    perPositionGpuBusyMs: busy.a.length ? { a: summarize(busy.a), b: summarize(busy.b) } : null,
    gpuBusyRatioAoverB: busyRatio.length ? summarize(busyRatio) : null,
  };
  log('ab', { wallRatio: out.wallRatioAoverB.median, gpuBusyRatio: out.gpuBusyRatioAoverB?.median });
  return out;
}

runMain(async () => {
  const presetName = params.get('preset') ?? (params.get('model') ? null : 'lc0-small');
  const preset = presetName ? PRESETS[presetName] : null;
  if (presetName && !preset) throw new Error(`unknown preset ${presetName}; known: ${Object.keys(PRESETS).join(', ')}`);
  const model = params.get('model') ?? preset.model;
  const batch = intParam('batch', preset?.batch ?? 1, 1, 256);
  const warmup = intParam('warmup', 5, 0, 1000);
  const iters = intParam('iters', 30, 3, 10000);
  const profileInvokes = intParam('profileInvokes', 5, 0, 200);
  const profiler = installGpuProfiler();
  ort.setRequestedOrtExecutionProviderForCurrentThread(params.get('ep') === 'wasm' ? 'wasm' : 'webgpu');
  const t0 = performance.now();
  const session = await ort.createOrtSession(model);
  const sessionCreateMs = performance.now() - t0;
  const backend = ort.describeOrtBackendConfig();
  log('session', { model, batch, sessionCreateMs, backend, fold: ort.lastOrtDequantFoldSummary() });
  const prof = profiler.lastDevice();
  const { feeds, fens } = await buildFeeds(session, batch);
  for (let i = 0; i < warmup; i++) await session.run(feeds);
  // Headline: unprofiled invokes (runtime's own passes, no timestamp writes).
  const runMs = [];
  for (let i = 0; i < iters; i++) {
    const s = performance.now();
    await session.run(feeds);
    runMs.push(performance.now() - s);
  }
  const timing = summarize(runMs);
  log('session.run ms', timing);
  const ab = params.get('ab') ? await pairedComparison(prof, { session, feeds, batch, name: presetName ?? model }, params.get('ab')) : null;
  let profile = null;
  if (profileInvokes > 0 && prof?.supported) {
    prof.arm();
    const profiledRunMs = [];
    for (let i = 0; i < profileInvokes; i++) {
      prof.nextInvoke();
      const s = performance.now();
      await session.run(feeds);
      profiledRunMs.push(performance.now() - s);
    }
    const submits = prof.submits;
    const collected = await prof.collect();
    profile = aggregateProfile(collected, { invokes: profileInvokes });
    profile.profiledRunMs = summarize(profiledRunMs);
    profile.submitsPerInvoke = submits / profileInvokes;
    profile.timestampsQuantized = collected.timestampsQuantized;
    profile.errors = collected.errors;
    profile.sequence = collected.rows
      .filter((r) => r.invoke === 1)
      .map((r) => [r.name, Math.round(r.ns / 10) / 100, r.workgroups?.join('x') ?? 'indirect', r.bindBytes]);
    profile.shaders = prof.shaders.map(({ code, ...rest }) => rest);
    log('profile', {
      kernelMsPerInvoke: profile.kernelMsPerInvoke,
      dispatches: profile.perInvoke[0]?.dispatches,
      submitsPerInvoke: profile.submitsPerInvoke,
      top: profile.kernels.slice(0, 6).map((k) => `${k.name}:${k.msPerInvoke.toFixed(3)}`),
    });
    // Same invokes with the runtime's own passes timestamped (no split): true GPU busy vs idle.
    profile.passMode = await passModeProfile(prof, () => session.run(feeds), Math.max(profileInvokes, 10));
    log('passMode', profile.passMode);
  } else if (profileInvokes > 0)
    profile = { supported: false, reason: prof ? 'timestamp-query unavailable' : 'ORT created no instrumented WebGPU device (WASM fallback?)' };
  let parity = null;
  if (params.get('parity') === 'wasm') {
    // Parity covers the whole fixture corpus in physical-batch chunks, not the
    // timing feeds (b1 timing feeds are only the start position).
    const corpus = await fixtureCorpus();
    const gpuOut = await runChunked(corpus, batch, async (chunk) => outputsToArrays(await session.run((await buildFeeds(session, batch, chunk)).feeds)));
    ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
    // The reference is the ORIGINAL model (preset's shipped export unless
    // referenceModel= says otherwise), never the candidate URL: comparing a
    // candidate with itself on another EP would pass broken weights.
    const referenceModel = params.get('referenceModel') ?? preset?.model ?? null;
    // Thresholds: ~2x the shipped baseline's max-abs over the 10-position
    // fixture corpus (M4, Chrome 157), i.e. a regression tripwire for f16 accumulation-order noise,
    // not a numerical proof (npm test pins ORT WASM to native lc0).
    const limits = preset?.parityMaxAbs ?? null;
    if (!referenceModel) {
      parity = { reference: null, candidateModel: model, pass: false, failures: ['no reference model: pass preset= or referenceModel='] };
    } else {
      const ref = await ort.InferenceSession.create(referenceModel, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      // Same chunking as the candidate (the reference takes the same feed shapes).
      const refOut = await runChunked(corpus, batch, async (chunk) => outputsToArrays(await ref.run((await buildFeeds(ref, batch, chunk)).feeds)));
      parity = {
        reference: 'ORT WASM, reference model, no dequant fold',
        fixtures: corpus.length,
        chunks: Math.ceil(corpus.length / batch),
        referenceModel,
        candidateModel: model,
        candidateIsReference: referenceModel === model,
        ...ortParity(gpuOut, refOut, limits),
      };
      await ref.release();
    }
    log('parity', parity);
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const identity = await deviceIdentity(adapter, prof?.device ?? null);
  await ort.releaseOrtSession(session);
  const gpuUsed = /webgpu/.test(backend) && !!prof;
  return {
    bench: 'ort-profile',
    ok: gpuUsed && !profiler.state.errors.length && profileComplete(profile) && (parity ? parity.pass : true),
    device: identity,
    candidate: {
      runtime: 'ort-webgpu',
      preset: presetName,
      model,
      batch,
      dequantFold: ort.lastOrtDequantFoldSummary(),
      backend,
      shipped: preset?.shipped ?? null,
      family: preset?.family ?? null,
    },
    params: { warmup, iters, profileInvokes, fens },
    timing: { sessionCreateMs, runMs: timing, perPositionMs: timing.median / batch },
    profile,
    parity,
    ab,
    errors: profiler.state.errors,
  };
});
