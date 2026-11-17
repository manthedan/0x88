// Optional ort-profile mode: time the actual production evaluator's synchronized
// b1 IO path. Capture parity observes the existing binding's CPU readback only
// after timing, then applies the page's unchanged full-corpus shipping gates.
import { parseFen } from '../../src/chess/board.ts';
import { LQO_NET } from '../../src/lc0/bt4Engine.ts';
import { encodeLc0Classical112 } from '../../src/lc0/encoder112.ts';
import { createLc0OnnxEvaluatorFromUrl } from '../../src/lc0/lc0SmallModelSelection.ts';
import { legalPolicyPriors } from '../../src/lc0/onnxEvaluator.ts';
import { f16BitsToF32, f32ToF16Bits } from '../../src/nn/float16.ts';
import { OrtGpuIoBinding } from '../../src/nn/ortGpuIo.ts';
import * as ort from '../../src/nn/ortRuntime.ts';
import { runtimeDeviceIdentity, summarize } from './common.mjs';
import { legalPolicyParity, ortParity } from './gates.mjs';

const values = (data) => (data instanceof Uint16Array ? Array.from(data, f16BitsToF32) : Array.from(data));
function joinRows(rows) {
  return Object.fromEntries(Object.keys(rows[0]).map((name) => [name, rows.flatMap((row) => row[name])]));
}
export async function profileLc0Io({ params, preset, model, batch, warmup, iters }) {
  const mode = params.get('lc0Io');
  if (params.get('preset') !== 'lqo' || model !== LQO_NET.modelUrl || batch !== 1 || !['cpu', 'gpu-capture'].includes(mode)) {
    throw Error('lc0Io benchmark requires the shipped lqo model, batch=1, and cpu or gpu-capture');
  }
  if (params.has('ab') || Number(params.get('profileInvokes') ?? 5) !== 0) throw Error('lc0Io mode requires profileInvokes=0 and no paired/profile mode');
  if (params.get('parity') === 'wasm' && mode !== 'gpu-capture') throw Error('lc0Io full raw parity requires gpu-capture');
  const corpus = (await (await fetch('/fixtures/lc0/fen_only.json')).json()).map((row) => row.fen).filter(Boolean);
  if (!corpus.length) throw Error('empty fixture corpus');
  ort.setRequestedOrtExecutionProviderForCurrentThread('webgpu');
  ort.setOrtDequantFoldForCurrentThread(ort.ortDequantFoldEnabled(LQO_NET.dequantFold));
  if (mode === 'gpu-capture' && LQO_NET.freeDimensionOverrides?.batch !== 1) throw Error('this build has no product batch-one capture policy');
  const start = performance.now();
  const loaded = await createLc0OnnxEvaluatorFromUrl(model, {
    deviceSelection: true,
    load: { cache: true, allowUnverified: false },
    evaluator: { ioMode: mode, ...(mode === 'gpu-capture' ? { freeDimensionOverrides: LQO_NET.freeDimensionOverrides } : {}) },
  });
  const evaluator = loaded.evaluator;
  const sessionCreateMs = performance.now() - start;
  const backend = ort.describeOrtBackendConfig();
  const fixedBatchSize = evaluator.fixedBatchSize;
  if (evaluator.ioMode !== mode || backend !== 'webgpu->webgpu' || (mode === 'gpu-capture' && fixedBatchSize !== 1)) {
    await evaluator.dispose();
    throw Error(`unexpected effective IO/placement ${evaluator.ioMode}, ${backend}, b${fixedBatchSize}`);
  }
  let timing,
    parity = null;
  try {
    for (let i = 0; i < warmup; i++) await evaluator.evaluate(corpus[0]);
    const times = [];
    for (let i = 0; i < iters; i++) {
      const t = performance.now();
      await evaluator.evaluate(corpus[0]);
      times.push(performance.now() - t);
    }
    timing = summarize(times);
    if (params.get('parity') === 'wasm') {
      const captured = [];
      const original = OrtGpuIoBinding.prototype.run;
      OrtGpuIoBinding.prototype.run = async function (input) {
        const out = await original.call(this, input);
        captured.push(Object.fromEntries(Object.entries(out).map(([name, data]) => [name, values(data)])));
        return out;
      };
      try {
        await evaluator.evaluateBatch(corpus);
      } finally {
        OrtGpuIoBinding.prototype.run = original;
      }
      if (captured.length !== corpus.length) throw Error('expected one physical b1 readback per corpus row');
      await evaluator.dispose();
      // Same unfurled reference and gate as ort-profile's ordinary parity path.
      ort.setRequestedOrtExecutionProviderForCurrentThread('wasm');
      const reference = await ort.InferenceSession.create(model, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
      const refRows = [];
      try {
        for (const fen of corpus) {
          const planes = encodeLc0Classical112(fen).planes;
          const feeds = { '/input/planes': new ort.Tensor('float16', Uint16Array.from(planes, f32ToF16Bits), [1, 112, 8, 8]) };
          const out = await reference.run(feeds);
          try {
            refRows.push(Object.fromEntries(Object.entries(out).map(([name, tensor]) => [name, values(tensor.data)])));
          } finally {
            ort.releaseOrtGpuOutputs(out);
            for (const tensor of Object.values(feeds)) tensor.dispose();
          }
        }
      } finally {
        await reference.release();
      }
      const candidate = joinRows(captured),
        ref = joinRows(refRows);
      const raw = ortParity(candidate, ref, preset.parityMaxAbs);
      const rows = corpus.map((fen) => {
        const legal = legalPolicyPriors(parseFen(fen), new Float32Array(1858), 1);
        return { fen, indices: legal.map((move) => move.index), ucis: legal.map((move) => move.uci) };
      });
      const legal = legalPolicyParity(candidate['/output/policy'], ref['/output/policy'], rows, preset.legalPolicy);
      parity = {
        fixtures: corpus.length,
        physicalBatchSize: 1,
        reference: 'ORT WASM, shipped bytes, no app fold',
        ...raw,
        legalPolicy: legal,
        pass: raw.pass && legal.pass,
      };
    }
  } finally {
    await evaluator.dispose();
  }
  return {
    bench: 'ort-profile-lc0-io',
    ok: parity ? parity.pass : true,
    device: await runtimeDeviceIdentity(),
    candidate: {
      preset: 'lqo',
      model,
      batch: 1,
      ioMode: mode,
      fixedBatchSize,
      backend,
      dequantFold: ort.lastOrtDequantFoldSummary(),
      modelLoad: loaded.modelLoad,
    },
    timing: { sessionCreateMs, runMs: timing, perPositionMs: timing.median },
    parity,
  };
}
