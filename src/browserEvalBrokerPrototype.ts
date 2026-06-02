import * as ort from './nn/ortRuntime.ts';

type Meta = { token_features?: number; history_plies?: number; onnx_fixed_legal_moves?: number; max_legal_moves?: number };
type EvalResult = { key: string; q: number; checksum: number };
type Pending = { key: string; tokens: BigInt64Array; legal: BigInt64Array; enqueuedAt: number; resolve: (x: EvalResult) => void; reject: (err: unknown) => void };
type Percentiles = { p50: number; p95: number; max: number };
type BrokerStats = {
  logicalRequests: number;
  cacheHits: number;
  inflightHits: number;
  enqueued: number;
  evalRuns: number;
  positionsEvaluated: number;
  batchSizes: Record<string, number>;
  queueWaitMs: number[];
  runMs: number[];
  maxQueueDepth: number;
};
type Result = {
  startedAt: string;
  finishedAt?: string;
  model: string;
  meta: string;
  ortEp: string;
  config: Record<string, number | string>;
  diagnostics?: ort.OrtRuntimeDiagnostics;
  stats?: BrokerStats & { queueWait: Percentiles; run: Percentiles; totalMs: number; logicalReqPerSec: number; nnPosPerSec: number; cacheHitRate: number; inflightHitRate: number; nnReductionFactor: number; avgBatch: number };
  baseline?: { requests: number; totalMs: number; msPerRequest: number; reqPerSec: number; checksum: number };
  errors: string[];
};

declare global {
  interface Window {
    tinyLeelaEvalBrokerPrototype?: { run: () => Promise<Result>; state: Result };
  }
}

const params = new URLSearchParams(location.search);
const modelPath = params.get('onnx') ?? '/models/bt4_sampled1b_best.onnx';
const metaPath = params.get('meta') ?? '/models/bt4_sampled1b_best.meta.json';
const statusEl = document.getElementById('status')!;
const diagnosticsEl = document.getElementById('diagnostics')!;
const metricsEl = document.getElementById('metrics')!;
const batchRowsEl = document.getElementById('batchRows')!;
const jsonEl = document.getElementById('json')!;

function intParam(name: string, fallback: number): number {
  const raw = params.get(name);
  const n = raw === null ? fallback : Number(raw);
  return Number.isFinite(n) ? Math.max(0, Math.floor(n)) : fallback;
}

function floatParam(name: string, fallback: number): number {
  const raw = params.get(name);
  const n = raw === null ? fallback : Number(raw);
  return Number.isFinite(n) ? n : fallback;
}

async function loadJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  return await res.json() as T;
}

function hash32(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

function makePosition(key: string, tokenStride: number, legalWidth: number): { tokens: BigInt64Array; legal: BigInt64Array } {
  const seed = hash32(key);
  const tokens = new BigInt64Array(64 * tokenStride);
  const legal = new BigInt64Array(legalWidth);
  for (let sq = 0; sq < 64; sq++) {
    const base = sq * tokenStride;
    for (let f = 0; f < tokenStride; f++) tokens[base + f] = BigInt((seed + sq * 17 + f * 31) & 15);
  }
  for (let j = 0; j < legalWidth; j++) legal[j] = BigInt((seed + j * 37) % 20480);
  return { tokens, legal };
}

function copyIntoBatch(items: Pending[], tokenStride: number, legalWidth: number): { feeds: Record<string, ort.Tensor> } {
  const batch = items.length;
  const tokens = new BigInt64Array(batch * 64 * tokenStride);
  const legal = new BigInt64Array(batch * legalWidth);
  for (let b = 0; b < batch; b++) {
    tokens.set(items[b].tokens, b * 64 * tokenStride);
    legal.set(items[b].legal, b * legalWidth);
  }
  return {
    feeds: {
      tokens: new ort.Tensor('int64', tokens, [batch, 64, tokenStride]),
      legal_action_ids: new ort.Tensor('int64', legal, [batch, legalWidth]),
    },
  };
}

function outputFor(outputs: Record<string, ort.Tensor>, key: string, batchIndex: number, legalWidth: number): EvalResult {
  let q = 0;
  let checksum = 0;
  const qData = outputs.q?.data;
  if (qData instanceof Float32Array && batchIndex < qData.length) q = qData[batchIndex];
  const wdl = outputs.wdl?.data;
  if (wdl instanceof Float32Array) {
    const base = batchIndex * 3;
    for (let i = 0; i < 3 && base + i < wdl.length; i++) checksum += wdl[base + i] * (i + 1);
  }
  const av = outputs.action_values?.data;
  if (av instanceof Float32Array) {
    const base = batchIndex * legalWidth;
    for (let i = 0; i < Math.min(8, legalWidth) && base + i < av.length; i++) checksum += av[base + i] * (i + 4);
  }
  return { key, q, checksum };
}

function percentile(values: number[]): Percentiles {
  if (!values.length) return { p50: 0, p95: 0, max: 0 };
  const xs = [...values].sort((a, b) => a - b);
  const at = (p: number) => xs[Math.min(xs.length - 1, Math.max(0, Math.floor(p * (xs.length - 1))))];
  return { p50: at(0.5), p95: at(0.95), max: xs[xs.length - 1] };
}

class EvalBroker {
  private pending: Pending[] = [];
  private inflight = new Map<string, Promise<EvalResult>>();
  private cache = new Map<string, EvalResult>();
  private timer: number | undefined;
  private flushing = false;
  readonly stats: BrokerStats = { logicalRequests: 0, cacheHits: 0, inflightHits: 0, enqueued: 0, evalRuns: 0, positionsEvaluated: 0, batchSizes: {}, queueWaitMs: [], runMs: [], maxQueueDepth: 0 };

  constructor(private session: ort.InferenceSession, private tokenStride: number, private legalWidth: number, private batchTarget: number, private maxWaitMs: number, private cacheCap: number) {}

  request(key: string): Promise<EvalResult> {
    this.stats.logicalRequests++;
    const cached = this.cache.get(key);
    if (cached) {
      this.stats.cacheHits++;
      // refresh LRU
      this.cache.delete(key);
      this.cache.set(key, cached);
      return Promise.resolve(cached);
    }
    const active = this.inflight.get(key);
    if (active) {
      this.stats.inflightHits++;
      return active;
    }
    const { tokens, legal } = makePosition(key, this.tokenStride, this.legalWidth);
    const promise = new Promise<EvalResult>((resolve, reject) => {
      this.pending.push({ key, tokens, legal, enqueuedAt: performance.now(), resolve, reject });
      this.stats.enqueued++;
      this.stats.maxQueueDepth = Math.max(this.stats.maxQueueDepth, this.pending.length);
      this.schedule();
    });
    this.inflight.set(key, promise);
    return promise;
  }

  private schedule() {
    if (this.pending.length >= this.batchTarget) {
      void this.flush();
      return;
    }
    if (this.timer === undefined) this.timer = window.setTimeout(() => { this.timer = undefined; void this.flush(); }, this.maxWaitMs);
  }

  private remember(result: EvalResult) {
    if (this.cacheCap <= 0) return;
    this.cache.set(result.key, result);
    while (this.cache.size > this.cacheCap) {
      const first = this.cache.keys().next().value as string | undefined;
      if (first === undefined) break;
      this.cache.delete(first);
    }
  }

  async drain(): Promise<void> {
    while (this.pending.length || this.flushing) {
      await this.flush();
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }

  private async flush(): Promise<void> {
    if (this.flushing) return;
    if (this.timer !== undefined) { window.clearTimeout(this.timer); this.timer = undefined; }
    this.flushing = true;
    try {
      while (this.pending.length) {
        const items = this.pending.splice(0, this.batchTarget);
        const now = performance.now();
        for (const item of items) this.stats.queueWaitMs.push(now - item.enqueuedAt);
        const { feeds } = copyIntoBatch(items, this.tokenStride, this.legalWidth);
        const t0 = performance.now();
        const outputs = await this.session.run(feeds);
        const runMs = performance.now() - t0;
        this.stats.runMs.push(runMs);
        this.stats.evalRuns++;
        this.stats.positionsEvaluated += items.length;
        this.stats.batchSizes[String(items.length)] = (this.stats.batchSizes[String(items.length)] ?? 0) + 1;
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          const result = outputFor(outputs, item.key, i, this.legalWidth);
          this.remember(result);
          this.inflight.delete(item.key);
          item.resolve(result);
        }
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    } catch (err) {
      const items = this.pending.splice(0);
      for (const item of items) {
        this.inflight.delete(item.key);
        item.reject(err);
      }
      throw err;
    } finally {
      this.flushing = false;
      if (this.pending.length) this.schedule();
    }
  }
}

function makeRng(seed0: number): () => number {
  let seed = seed0 >>> 0;
  return () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 0x100000000;
  };
}

function chooseKey(rng: () => number, workerId: number, step: number, hotProb: number, hotKeys: number, keySpace: number, openings: number): string {
  // Root/opening repetitions intentionally mimic many games searching nearby opening trees.
  if (step < 12 && rng() < 0.80) return `opening:${step}:${Math.floor(rng() * openings)}`;
  if (rng() < hotProb) return `hot:${Math.floor(rng() * hotKeys)}`;
  return `cold:${workerId}:${step}:${Math.floor(rng() * keySpace)}`;
}

async function virtualWorker(workerId: number, broker: EvalBroker, requests: number, burst: number, hotProb: number, hotKeys: number, keySpace: number, openings: number): Promise<number> {
  const rng = makeRng(0x9e3779b9 ^ workerId);
  let checksum = 0;
  for (let step = 0; step < requests; step += burst) {
    const ps: Promise<EvalResult>[] = [];
    for (let j = 0; j < burst && step + j < requests; j++) ps.push(broker.request(chooseKey(rng, workerId, step + j, hotProb, hotKeys, keySpace, openings)));
    const rs = await Promise.all(ps);
    for (const r of rs) checksum += r.q + r.checksum * 0.001;
    if ((step & 31) === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }
  return checksum;
}

async function runBaseline(session: ort.InferenceSession, tokenStride: number, legalWidth: number, requests: number): Promise<Result['baseline']> {
  let checksum = 0;
  const t0 = performance.now();
  for (let i = 0; i < requests; i++) {
    const { tokens, legal } = makePosition(`baseline:${i}`, tokenStride, legalWidth);
    const outputs = await session.run({ tokens: new ort.Tensor('int64', tokens, [1, 64, tokenStride]), legal_action_ids: new ort.Tensor('int64', legal, [1, legalWidth]) });
    checksum += outputFor(outputs, `baseline:${i}`, 0, legalWidth).checksum;
  }
  const totalMs = performance.now() - t0;
  return { requests, totalMs, msPerRequest: totalMs / requests, reqPerSec: requests / (totalMs / 1000), checksum };
}

function diagnosticSummary(diag?: ort.OrtRuntimeDiagnostics): string {
  if (!diag) return 'diagnostics pending';
  const adapter = diag.adapter ? (diag.adapter.ok ? diag.adapter.summary ?? 'adapter ok' : `adapter unavailable: ${diag.adapter.error}`) : 'adapter not probed';
  const attempts = diag.sessionAttempts.map((a) => `${a.providers.join('+')}:${a.ok ? 'ok' : 'fail'}:${a.ms.toFixed(0)}ms`).join(' | ') || 'none';
  return `requested=${diag.requestedEp} resolved=${diag.resolvedExecutionProviders.join(',')} webgpu=${diag.webgpuAvailable ? 1 : 0} wasmThreads=${diag.wasm.numThreads ?? '?'} adapter=${adapter} sessions=${attempts}`;
}

function render(state: Result) {
  diagnosticsEl.textContent = diagnosticSummary(state.diagnostics);
  const s = state.stats;
  const metrics: [string, string][] = [];
  if (s) {
    metrics.push(['logical req/s', s.logicalReqPerSec.toFixed(0)]);
    metrics.push(['NN pos/s', s.nnPosPerSec.toFixed(0)]);
    metrics.push(['logical requests', String(s.logicalRequests)]);
    metrics.push(['NN positions', String(s.positionsEvaluated)]);
    metrics.push(['reduction factor', `${s.nnReductionFactor.toFixed(2)}×`]);
    metrics.push(['cache hits', `${(s.cacheHitRate * 100).toFixed(1)}%`]);
    metrics.push(['in-flight hits', `${(s.inflightHitRate * 100).toFixed(1)}%`]);
    metrics.push(['avg batch', s.avgBatch.toFixed(1)]);
    metrics.push(['run p50/p95', `${s.run.p50.toFixed(2)} / ${s.run.p95.toFixed(2)} ms`]);
    metrics.push(['wait p50/p95', `${s.queueWait.p50.toFixed(2)} / ${s.queueWait.p95.toFixed(2)} ms`]);
  }
  if (state.baseline) metrics.push(['baseline batch1', `${state.baseline.msPerRequest.toFixed(2)} ms/req`]);
  metricsEl.innerHTML = metrics.map(([k, v]) => `<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
  batchRowsEl.innerHTML = s ? Object.entries(s.batchSizes).sort((a, b) => Number(a[0]) - Number(b[0])).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('') : '';
  jsonEl.textContent = JSON.stringify(state, null, 2);
}

const config = {
  workers: intParam('workers', 96),
  requestsPerWorker: intParam('requestsPerWorker', 200),
  burst: intParam('burst', 4),
  batchTarget: intParam('batchTarget', 64),
  maxWaitMs: floatParam('maxWaitMs', 1.5),
  cacheCap: intParam('cacheCap', 50000),
  hotProb: floatParam('hotProb', 0.72),
  hotKeys: intParam('hotKeys', 2048),
  keySpace: intParam('keySpace', 1000000),
  openings: intParam('openings', 64),
  baselineRequests: intParam('baselineRequests', 64),
};

const state: Result = { startedAt: new Date().toISOString(), model: modelPath, meta: metaPath, ortEp: params.get('ortEp') ?? params.get('ep') ?? 'webgpu', config, errors: [] };

async function run(): Promise<Result> {
  state.startedAt = new Date().toISOString();
  state.finishedAt = undefined;
  state.errors = [];
  state.stats = undefined;
  state.baseline = undefined;
  statusEl.textContent = 'loading meta and diagnostics';
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  const meta = await loadJson<Meta>(metaPath);
  const tokenStride = Number(meta.token_features ?? ((meta.history_plies ?? 8) + 9));
  const legalWidth = Number(meta.onnx_fixed_legal_moves ?? meta.max_legal_moves ?? 128);
  statusEl.textContent = `creating ORT session; tokenStride=${tokenStride} legalWidth=${legalWidth}`;
  const session = await ort.createOrtSession(modelPath);
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  if (config.baselineRequests > 0) {
    statusEl.textContent = `running batch-1 baseline (${config.baselineRequests} requests)`;
    state.baseline = await runBaseline(session, tokenStride, legalWidth, config.baselineRequests);
    render(state);
  }
  statusEl.textContent = `running broker workload: ${config.workers} workers × ${config.requestsPerWorker} requests`;
  const broker = new EvalBroker(session, tokenStride, legalWidth, config.batchTarget, config.maxWaitMs, config.cacheCap);
  const t0 = performance.now();
  const checksums = await Promise.all(Array.from({ length: config.workers }, (_, i) => virtualWorker(i, broker, config.requestsPerWorker, config.burst, config.hotProb, config.hotKeys, config.keySpace, config.openings)));
  await broker.drain();
  const totalMs = performance.now() - t0;
  const stats = broker.stats;
  const logicalReqPerSec = stats.logicalRequests / (totalMs / 1000);
  const nnPosPerSec = stats.positionsEvaluated / (stats.runMs.reduce((a, b) => a + b, 0) / 1000);
  state.stats = {
    ...stats,
    queueWait: percentile(stats.queueWaitMs),
    run: percentile(stats.runMs),
    totalMs,
    logicalReqPerSec,
    nnPosPerSec,
    cacheHitRate: stats.cacheHits / Math.max(1, stats.logicalRequests),
    inflightHitRate: stats.inflightHits / Math.max(1, stats.logicalRequests),
    nnReductionFactor: stats.logicalRequests / Math.max(1, stats.positionsEvaluated),
    avgBatch: stats.positionsEvaluated / Math.max(1, stats.evalRuns),
  };
  // Prevent optimizing away result-dependent work in devtools timelines.
  state.config.checksum = checksums.reduce((a, b) => a + b, 0).toFixed(5);
  state.finishedAt = new Date().toISOString();
  statusEl.textContent = `done in ${totalMs.toFixed(1)}ms`;
  render(state);
  return state;
}

window.tinyLeelaEvalBrokerPrototype = { run, state };
(document.getElementById('run') as HTMLButtonElement).onclick = () => { void run(); };
(document.getElementById('copyJson') as HTMLButtonElement).onclick = () => navigator.clipboard?.writeText(JSON.stringify(state, null, 2));
if (params.get('autorun') === '1') void run();
