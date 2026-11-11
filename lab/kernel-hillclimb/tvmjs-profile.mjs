// TVMJS WebGPU timing + per-kernel profile through the production evaluators.
//
// Query params:
//   target=centipawn|lc0-t1|lc0-t3|lc0-bt4   (default centipawn, the shipped TVMJS path)
//   manifest=/runtimes/...manifest.json       override the target's default manifest
//   batch=N   physical batch (lc0 targets; centipawn manifests are fixed b16)
//   warmup=N (5)  iters=N (30)  profileInvokes=N (5, 0 disables)
//   parity=ort  compare evaluator outputs with the ORT path on the same positions
//   tieEpsilon=X  (lc0 parity) best-move flips whose REFERENCE prior gap <= X are tolerated
//   maxPriorAbs=X maxWdlAbs=X  (lc0 parity) override the target's max-abs limits
import { parseFen } from '../../src/chess/board.ts';
import { Lc0OnnxEvaluator } from '../../src/lc0/onnxEvaluator.ts';
import { Lc0WholeOnnxWebgpuEvaluator } from '../../src/lc0/wholeOnnxWebgpuEvaluator.ts';
import * as ort from '../../src/nn/ortRuntime.ts';
import { SquareFormerEvaluator } from '../../src/nn/squareformerEvaluator.ts';
import { SquareformerTvmjsWebgpuEvaluator } from '../../src/nn/squareformerTvmjsWebgpuEvaluator.ts';
import { deviceIdentity, intParam, log, params, runMain, summarize } from './common.mjs';
import { compareCentipawn, compareLc0, manifestCheck, profileComplete } from './gates.mjs';
import { aggregateProfile, installGpuProfiler, passModeProfile } from './gpu-profiler.mjs';

const LC0 = '/runtimes/lc0-tvmjs-webgpu';
const TARGETS = {
  centipawn: {
    kind: 'centipawn',
    manifest: '/runtimes/centipawn-tvmjs-webgpu/bt4-soap-rem-c19000-final/f32/v2-shape-k16/manifest.json',
    batch: 16,
    meta: '/models/bt4_soap_rem_c19000_final.meta.json',
    ortModel: '/models/bt4_soap_rem_c19000_final.onnx',
    shipped: 'Centipawn default runtime (TVMJS v2-shape-k16, f32, physical b16)',
  },
  'lc0-t1': {
    kind: 'lc0',
    manifest: `${LC0}/t1-256x10-distilled-swa-2432500/f16/v1/manifest.json`,
    batch: 8,
    ortModel: '/models/lc0/t1-256x10-distilled-swa-2432500.batch{batch}.f16.onnx',
    // ~4x the max-abs observed vs ORT (prior 0.0063, WDL 0.0024; M4, Nov 2026).
    maxPriorAbs: 0.03,
    maxWdlAbs: 0.01,
    shipped: 'Lc0 small TVM whole-model option (non-default; hidden in the v0 deploy profile)',
  },
  'lc0-t3': {
    kind: 'lc0',
    manifest: `${LC0}/t3-512x15x16h-distill-swa-2767500/f16/v1/manifest.json`,
    batch: 8,
    ortModel: '/models/lc0/t3-512x15x16h-distill-swa-2767500.batch{batch}.f16.onnx',
    maxPriorAbs: 0.05,
    maxWdlAbs: 0.01,
    shipped: 'research lane (not wired to product)',
  },
  'lc0-bt4': {
    kind: 'lc0',
    manifest: `${LC0}/BT4-1024x15x32h-swa-6147500-policytune-332/f16/v1/manifest.json`,
    batch: 8,
    ortModel: '/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch{batch}.f16.onnx',
    // ~4x observed (prior 0.012, WDL 0.0022).
    maxPriorAbs: 0.05,
    maxWdlAbs: 0.01,
    shipped: 'research lane (not wired to product)',
  },
};

async function fixtureCorpus() {
  const fixtures = await (await fetch('/fixtures/lc0/fen_only.json')).json();
  return fixtures.map((f) => f.fen);
}

async function fixtureFens(n) {
  const fens = await fixtureCorpus();
  return Array.from({ length: n }, (_, i) => fens[i % fens.length]);
}

runMain(async () => {
  const targetName = params.get('target') ?? 'centipawn';
  const target = TARGETS[targetName];
  if (!target) throw new Error(`unknown target ${targetName}; known: ${Object.keys(TARGETS).join(', ')}`);
  const manifest = params.get('manifest') ?? target.manifest;
  const batch = intParam('batch', target.batch, 1, 64);
  const warmup = intParam('warmup', 5, 0, 1000);
  const iters = intParam('iters', 30, 3, 10000);
  const profileInvokes = intParam('profileInvokes', 5, 0, 200);
  const tieEpsilonRaw = params.get('tieEpsilon');
  const tieEpsilon = tieEpsilonRaw === null ? null : Number(tieEpsilonRaw);
  const profiler = installGpuProfiler();
  const fens = await fixtureFens(batch);
  const t0 = performance.now();
  let evaluator;
  let evaluate;
  if (target.kind === 'centipawn') {
    const meta = await (await fetch(target.meta)).json();
    evaluator = await SquareformerTvmjsWebgpuEvaluator.create(manifest, meta);
    const boards = fens.map((fen) => parseFen(fen));
    evaluate = () => evaluator.evaluateBatch(boards);
    target.metaJson = meta;
  } else {
    evaluator = await Lc0WholeOnnxWebgpuEvaluator.create({ manifestUrl: manifest, batch });
    evaluate = () => evaluator.evaluateBatch(fens);
  }
  const createMs = performance.now() - t0;
  const prof = profiler.lastDevice();
  // The Centipawn evaluator silently swaps v3-subgroups for v2-shape-k16 on
  // adapters without fixed 32-wide subgroups; record what actually ran.
  const manifestRun = manifestCheck(manifest, evaluator.manifestUrl ?? manifest, location.href);
  log('evaluator', { target: targetName, manifest, effectiveManifest: manifestRun.effective, batch, createMs });
  if (!manifestRun.ok) log('MANIFEST FALLBACK', manifestRun);
  for (let i = 0; i < warmup; i++) await evaluate();
  const runMs = [];
  for (let i = 0; i < iters; i++) {
    const s = performance.now();
    await evaluate();
    runMs.push(performance.now() - s);
  }
  const timing = summarize(runMs);
  log('evaluateBatch ms', timing);
  let profile = null;
  if (profileInvokes > 0 && prof?.supported) {
    prof.arm();
    const profiledRunMs = [];
    for (let i = 0; i < profileInvokes; i++) {
      prof.nextInvoke();
      const s = performance.now();
      await evaluate();
      profiledRunMs.push(performance.now() - s);
    }
    const submits = prof.submits;
    const collected = await prof.collect();
    profile = aggregateProfile(collected, { invokes: profileInvokes });
    profile.profiledRunMs = summarize(profiledRunMs);
    profile.submitsPerInvoke = submits / profileInvokes;
    profile.timestampsQuantized = collected.timestampsQuantized;
    profile.sequence = collected.rows
      .filter((r) => r.invoke === 1)
      .map((r) => [r.name, Math.round(r.ns / 10) / 100, r.workgroups?.join('x') ?? 'indirect', r.bindBytes]);
    profile.shaders = prof.shaders.map(({ code, ...rest }) => rest);
    log('profile', {
      kernelMsPerInvoke: profile.kernelMsPerInvoke,
      dispatches: profile.perInvoke[0]?.dispatches,
      top: profile.kernels.slice(0, 6).map((k) => `${k.name}:${k.msPerInvoke.toFixed(3)}`),
    });
    // Same invokes with the runtime's own passes timestamped (no split): true GPU busy vs idle.
    profile.passMode = await passModeProfile(prof, evaluate, Math.max(profileInvokes, 10));
    log('passMode', profile.passMode);
  } else if (profileInvokes > 0) profile = { supported: false, reason: 'timestamp-query unavailable' };
  let parity = null;
  if (params.get('parity') === 'ort') {
    // Whole fixture corpus, not the timing batch (b8 timing feeds drop the
    // tail fixtures, including the only nonzero rule50 position). The
    // evaluators split it into physical batches and pad the last one.
    const corpus = await fixtureCorpus();
    if (target.kind === 'centipawn') {
      const boards = corpus.map((f) => parseFen(f));
      const tvmOut = await evaluator.evaluateBatch(boards);
      const ref = await SquareFormerEvaluator.create(target.ortModel, target.metaJson);
      parity = {
        reference: `ORT ${ort.describeOrtBackendConfig()} ${target.ortModel}`,
        fixtures: corpus.length,
        ...compareCentipawn(tvmOut, await ref.evaluateBatch(boards)),
      };
    } else {
      const tvmOut = await evaluator.evaluateBatch(corpus);
      const ortModel = target.ortModel.replace('{batch}', '1');
      const ref = await Lc0OnnxEvaluator.create(ortModel);
      const refOut = [];
      for (const fen of corpus) refOut.push(await ref.evaluate(fen));
      const maxPriorAbs = params.get('maxPriorAbs') === null ? target.maxPriorAbs : Number(params.get('maxPriorAbs'));
      const maxWdlAbs = params.get('maxWdlAbs') === null ? target.maxWdlAbs : Number(params.get('maxWdlAbs'));
      parity = {
        reference: `ORT ${ort.describeOrtBackendConfig()} ${ortModel}`,
        fixtures: corpus.length,
        ...compareLc0(tvmOut, refOut, { tieEpsilon, maxPriorAbs, maxWdlAbs }),
      };
    }
    log('parity', parity);
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  const identity = await deviceIdentity(adapter, prof?.device ?? null);
  evaluator.dispose?.();
  return {
    bench: 'tvmjs-profile',
    ok: manifestRun.ok && !profiler.state.errors.length && profileComplete(profile) && (parity ? parity.pass : true),
    device: identity,
    // `manifest` is what ran (roofline reads its MAC table from it);
    // `requestedManifest` is what was asked for.
    candidate: {
      runtime: 'tvmjs-webgpu',
      target: targetName,
      manifest: manifestRun.effective,
      requestedManifest: manifestRun.requested,
      manifestFallback: !manifestRun.ok,
      batch,
      shipped: target.shipped,
    },
    params: { warmup, iters, profileInvokes, fens },
    timing: { createMs, runMs: timing, perPositionMs: timing.median / batch },
    profile,
    parity,
    errors: manifestRun.ok
      ? profiler.state.errors
      : [...profiler.state.errors, `manifest fallback: requested ${manifestRun.requested}, ran ${manifestRun.effective}`],
  };
});
