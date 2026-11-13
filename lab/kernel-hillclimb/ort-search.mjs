// Lc0 Small search throughput per ORT IO mode, in the product search worker
// (docs/ort_invoke_overhead_2026-11.md). One worker per mode, same model and
// device rule as the product (deviceModelSelection: Apple gets split-K), fixed
// visits, eval cache cleared before every search so each mode runs the same
// NN evaluations. Paired per (round, position); the search results must match
// exactly across modes (outputs are byte-identical, so the trees are too).
//
// Query params:
//   worker=URL     the built search worker (dist-client/_app/immutable/workers/searchWorker-*.js); required
//   model=URL      default the shipped Lc0 Small b1 qdq8
//   deviceModelSelection=1 (0 loads exactly model=)
//   modes=cpu,gpu-capture   (first mode is the reference)
//   visits=256  rounds=3  positions=10 (fixture FENs)  warmupSearches=2
import { intParam, listParam, log, params, runMain, summarize } from './common.mjs';

const DEFAULT_MODEL = '/models/lc0/t1-256x10-distilled-swa-2432500.batch1.f16.qdq8.onnx';

class WorkerClient {
  constructor(url, name) {
    this.worker = new Worker(url, { type: 'module', name });
    this.nextId = 1;
    this.pending = new Map();
    this.worker.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === 'searchProgress' || msg.type === 'downloadProgress') return;
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.error));
      else p.resolve(msg);
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(`worker error: ${e.message}`));
      this.pending.clear();
    };
  }
  request(message) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...message, id });
    });
  }
  terminate() {
    this.worker.terminate();
  }
}

function searchSignature(result) {
  return JSON.stringify({ move: result.move, visits: result.visits, value: result.value, children: result.children, pv: result.pv });
}

runMain(async () => {
  const workerUrl = params.get('worker');
  if (!workerUrl) throw new Error('ort-search: worker= (built searchWorker URL) is required');
  const model = params.get('model') ?? DEFAULT_MODEL;
  const deviceModelSelection = params.get('deviceModelSelection') !== '0';
  const modes = listParam('modes', ['cpu', 'gpu-capture']);
  const visits = intParam('visits', 256, 1, 100000);
  const rounds = intParam('rounds', 3, 1, 100);
  const warmupSearches = intParam('warmupSearches', 2, 0, 100);
  const fixtures = await (await fetch('/fixtures/lc0/fen_only.json')).json();
  const fens = fixtures
    .map((f) => f.fen)
    .filter(Boolean)
    .slice(0, intParam('positions', 10, 1, 1000));

  const clients = {};
  const ready = {};
  for (const mode of modes) {
    const client = new WorkerClient(workerUrl, `ort-search-${mode}`);
    const msg = await client.request({
      type: 'init',
      modelUrl: model,
      deviceModelSelection,
      ep: 'webgpu',
      cacheModel: false,
      evalCacheEntries: 2048,
      ...(mode === 'cpu' ? {} : { ortIoMode: mode }),
    });
    clients[mode] = client;
    ready[mode] = { backend: msg.backend, servedModelUrl: msg.servedModelUrl, ortIoMode: msg.ortIoMode ?? null };
    log('ready', mode, ready[mode]);
    if (mode !== 'cpu' && msg.ortIoMode !== mode) throw new Error(`ort-search: ${mode} worker fell back to ${msg.ortIoMode}`);
  }

  const search = async (mode, fen) => {
    await clients[mode].request({ type: 'resetSearch', clearEvalCache: true });
    const t0 = performance.now();
    const msg = await clients[mode].request({ type: 'search', input: fen, visits, batchSize: 1, reuseTree: false });
    return { wallMs: performance.now() - t0, elapsedMs: msg.result.elapsedMs, signature: searchSignature(msg.result), stats: msg.result.stats };
  };

  for (const mode of modes) for (let i = 0; i < warmupSearches; i++) await search(mode, fens[i % fens.length]);

  const elapsed = Object.fromEntries(modes.map((m) => [m, []]));
  const ratios = Object.fromEntries(modes.slice(1).map((m) => [m, []]));
  const mismatches = [];
  let evals = null;
  for (let r = 0; r < rounds; r++) {
    for (const [p, fen] of fens.entries()) {
      const order = modes.map((_, i) => modes[(i + r + p) % modes.length]);
      const row = {};
      for (const mode of order) row[mode] = await search(mode, fen);
      for (const mode of modes) elapsed[mode].push(row[mode].elapsedMs);
      for (const mode of modes.slice(1)) {
        ratios[mode].push(row[mode].elapsedMs / row[modes[0]].elapsedMs);
        if (row[mode].signature !== row[modes[0]].signature) mismatches.push({ round: r, position: p, mode });
      }
      evals ??= row[modes[0]].stats ?? null;
    }
    log('round', r, Object.fromEntries(modes.slice(1).map((m) => [m, summarize(ratios[m]).median])));
  }
  for (const client of Object.values(clients)) client.terminate();
  const out = {
    bench: 'ort-search',
    ok: mismatches.length === 0,
    candidate: { model, deviceModelSelection, modes, ready },
    params: { visits, rounds, positions: fens.length, warmupSearches },
    searchMs: Object.fromEntries(modes.map((m) => [m, summarize(elapsed[m])])),
    // < 1: the mode is faster than the reference at the same visits.
    ratioVsReference: Object.fromEntries(modes.slice(1).map((m) => [m, summarize(ratios[m])])),
    mismatches: mismatches.slice(0, 20),
    identicalSearches: mismatches.length === 0,
    sampleStats: evals,
  };
  log('result', out.ratioVsReference);
  return out;
});
