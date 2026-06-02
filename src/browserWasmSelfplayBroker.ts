import * as ort from './nn/ortRuntime.ts';

type Meta = { token_features?: number; history_plies?: number; onnx_fixed_legal_moves?: number; max_legal_moves?: number; policy_size?: number; attack_summary_feature_count?: number; attack_summary_schema?: string | null };
type BridgeExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  max_games: () => number;
  max_slots: () => number;
  max_token_stride: () => number;
  max_legal_width: () => number;
  max_attack_features?: () => number;
  token_ptr: () => number;
  legal_ptr: () => number;
  attack_summary_ptr?: () => number;
  slot_hash_lo_ptr: () => number;
  slot_hash_hi_ptr: () => number;
  slot_game_ptr: () => number;
  slot_move_code_ptr: () => number;
  slot_legal_count_ptr: () => number;
  slot_legal_move_code_ptr: () => number;
  path_move_ptr: () => number;
  max_path_moves: () => number;
  game_fen_ptr?: () => number;
  write_game_fen?: (gameIdx: number) => number;
  reset_games: (gameCount: number, maxPlies: number, openingCount: number, openingDepth: number, seed: number) => number;
  game_count: () => number;
  active_games: () => number;
  game_done: (gameIdx: number) => number;
  game_ply: (gameIdx: number) => number;
  prepare_game_to_slots: (gameIdx: number, startSlot: number, tokenStride: number, legalWidth: number, slotLimit: number) => number;
  prepare_game_path_to_slots: (gameIdx: number, pathLen: number, startSlot: number, tokenStride: number, legalWidth: number, slotLimit: number) => number;
  prepare_game_path_position_slot: (gameIdx: number, pathLen: number, slotIdx: number, tokenStride: number, legalWidth: number) => number;
  apply_slot: (slotIdx: number) => number;
};
type EdgeEval = { moveCode: number; prior: number; policyLogit: number; av?: number; rank?: number; regret?: number };
type EvalResult = { key: string; q: number; checksum: number; edges?: EdgeEval[] };
type Pending = { key: string; slot: number; enqueuedAt: number; resolve: (x: EvalResult) => void; reject: (err: unknown) => void };
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
type SelfplayRow = { game: number; ply: number; fen?: string; playedMoveCode: number; playedUci?: string; visits: number; temperature: number; noiseAlpha: number; noiseFrac: number; topMoves: { moveCode: number; uci?: string; visits: number; q: number; prior: number }[] };
const BROKER_USED_OUTPUTS = ['policy', 'wdl', 'action_values', 'rank_scores', 'regrets'] as const;

function selectedBrokerOutputs(session: ort.InferenceSession): string[] | undefined {
  const names = (session as unknown as { outputNames?: string[] }).outputNames;
  if (!Array.isArray(names)) return undefined;
  const fetches = BROKER_USED_OUTPUTS.filter((name) => names.includes(name));
  return fetches.length ? fetches : undefined;
}

type Result = {
  startedAt: string;
  finishedAt?: string;
  model: string;
  meta: string;
  wasm: string;
  ortEp: string;
  config: Record<string, number | string>;
  diagnostics?: ort.OrtRuntimeDiagnostics;
  selfplayRows?: SelfplayRow[];
  stats?: BrokerStats & {
    queueWait: Percentiles;
    run: Percentiles;
    totalMs: number;
    games: number;
    completedGames: number;
    pliesPlayed: number;
    avgPliesPerGame: number;
    leafRequestsPerSec: number;
    pliesPerSec: number;
    nnPosPerSec: number;
    cacheHitRate: number;
    inflightHitRate: number;
    nnReductionFactor: number;
    avgBatch: number;
    searchMode: string;
    puctVisits?: number;
    puctMaxDepth?: number;
    puctExpansions?: number;
    puctLeafEvals?: number;
    puctAvgRootVisits?: number;
    puctLeafWaves?: number;
    puctReuseTree?: number;
    gumbelTopK?: number;
  };
  errors: string[];
};

declare global {
  interface Window {
    tinyLeelaWasmSelfplayBroker?: { run: () => Promise<Result>; state: Result };
  }
}

const params = new URLSearchParams(location.search);
const modelPath = params.get('onnx') ?? '/models/bt4_sampled1b_best.onnx';
const metaPath = params.get('meta') ?? '/models/bt4_sampled1b_best.meta.json';
const wasmPath = params.get('wasm') ?? '/rust_bridge/tl_wasm_selfplay_bridge.wasm';
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

function moveCodeToUci(moveCode: number): string {
  const from = moveCode & 63;
  const to = (moveCode >> 6) & 63;
  const promo = (moveCode >> 12) & 7;
  const square = (idx: number) => String.fromCharCode(97 + (idx % 8)) + String(Math.floor(idx / 8) + 1);
  const suffix = promo === 1 ? 'n' : promo === 2 ? 'b' : promo === 3 ? 'r' : promo === 4 ? 'q' : '';
  return `${square(from)}${square(to)}${suffix}`;
}

function currentFen(wasm: BridgeExports, gameIdx: number): string | undefined {
  if (!wasm.write_game_fen || !wasm.game_fen_ptr) return undefined;
  const len = wasm.write_game_fen(gameIdx);
  if (!len) return undefined;
  return new TextDecoder().decode(new Uint8Array(wasm.memory.buffer, wasm.game_fen_ptr(), len));
}

async function loadBridge(url: string): Promise<BridgeExports> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status} ${res.statusText}`);
  const bytes = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const exports = instance.exports as BridgeExports;
  for (const name of ['memory', 'token_ptr', 'legal_ptr', 'reset_games', 'prepare_game_to_slots', 'apply_slot']) {
    if (!(name in exports)) throw new Error(`WASM bridge missing export ${name}; exports=${Object.keys(exports).join(',')}`);
  }
  return exports;
}

function percentile(values: number[]): Percentiles {
  if (!values.length) return { p50: 0, p95: 0, max: 0 };
  const xs = [...values].sort((a, b) => a - b);
  const at = (p: number) => xs[Math.min(xs.length - 1, Math.max(0, Math.floor(p * (xs.length - 1))))];
  return { p50: at(0.5), p95: at(0.95), max: xs[xs.length - 1] };
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

class EvalBroker {
  private pending: Pending[] = [];
  private inflight = new Map<string, Promise<EvalResult>>();
  private cache = new Map<string, EvalResult>();
  private timer: number | undefined;
  private flushing = false;
  readonly stats: BrokerStats = { logicalRequests: 0, cacheHits: 0, inflightHits: 0, enqueued: 0, evalRuns: 0, positionsEvaluated: 0, batchSizes: {}, queueWaitMs: [], runMs: [], maxQueueDepth: 0 };

  constructor(private session: ort.InferenceSession, private wasm: BridgeExports, private tokenStride: number, private legalWidth: number, private policySize: number, private attackFeatureCount: number, private batchTarget: number, private maxWaitMs: number, private cacheCap: number) {}

  request(key: string, slot: number): Promise<EvalResult> {
    this.stats.logicalRequests++;
    const cached = this.cache.get(key);
    if (cached) {
      this.stats.cacheHits++;
      this.cache.delete(key);
      this.cache.set(key, cached);
      return Promise.resolve(cached);
    }
    const active = this.inflight.get(key);
    if (active) {
      this.stats.inflightHits++;
      return active;
    }
    const promise = new Promise<EvalResult>((resolve, reject) => {
      this.pending.push({ key, slot, enqueuedAt: performance.now(), resolve, reject });
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

  private makeFeeds(items: Pending[]): Record<string, ort.Tensor> {
    const batch = items.length;
    const memory = this.wasm.memory.buffer;
    const tokenBase = this.wasm.token_ptr();
    const legalBase = this.wasm.legal_ptr();
    const maxTokenStride = this.wasm.max_token_stride();
    const maxLegalWidth = this.wasm.max_legal_width();
    const srcTokens = new BigInt64Array(memory, tokenBase, this.wasm.max_slots() * 64 * maxTokenStride);
    const srcLegal = new BigInt64Array(memory, legalBase, this.wasm.max_slots() * maxLegalWidth);
    const tokens = new BigInt64Array(batch * 64 * this.tokenStride);
    const legal = new BigInt64Array(batch * this.legalWidth);
    const feeds: Record<string, ort.Tensor> = {
      tokens: new ort.Tensor('int64', tokens, [batch, 64, this.tokenStride]),
      legal_action_ids: new ort.Tensor('int64', legal, [batch, this.legalWidth]),
    };
    let attack: Float32Array | undefined;
    let srcAttack: Float32Array | undefined;
    let maxAttackFeatures = 0;
    if (this.attackFeatureCount > 0) {
      if (!this.wasm.attack_summary_ptr || !this.wasm.max_attack_features) throw new Error('WASM bridge does not expose TG attack_summary buffers');
      maxAttackFeatures = this.wasm.max_attack_features();
      if (this.attackFeatureCount > maxAttackFeatures) throw new Error(`attackFeatureCount=${this.attackFeatureCount} exceeds bridge max=${maxAttackFeatures}`);
      srcAttack = new Float32Array(memory, this.wasm.attack_summary_ptr(), this.wasm.max_slots() * 64 * maxAttackFeatures);
      attack = new Float32Array(batch * 64 * this.attackFeatureCount);
      feeds.attack_summary = new ort.Tensor('float32', attack, [batch, 64, this.attackFeatureCount]);
    }
    for (let b = 0; b < batch; b++) {
      const slot = items[b].slot;
      for (let sq = 0; sq < 64; sq++) {
        const srcOff = slot * 64 * maxTokenStride + sq * maxTokenStride;
        const dstOff = b * 64 * this.tokenStride + sq * this.tokenStride;
        tokens.set(srcTokens.subarray(srcOff, srcOff + this.tokenStride), dstOff);
        if (attack && srcAttack) {
          const srcAttackOff = slot * 64 * maxAttackFeatures + sq * maxAttackFeatures;
          const dstAttackOff = b * 64 * this.attackFeatureCount + sq * this.attackFeatureCount;
          attack.set(srcAttack.subarray(srcAttackOff, srcAttackOff + this.attackFeatureCount), dstAttackOff);
        }
      }
      const legalOff = slot * maxLegalWidth;
      legal.set(srcLegal.subarray(legalOff, legalOff + this.legalWidth), b * this.legalWidth);
    }
    return feeds;
  }

  private outputForItem(outputs: Record<string, ort.Tensor>, item: Pending, batchIndex: number): EvalResult {
    const result = outputFor(outputs, item.key, batchIndex, this.legalWidth);
    const policy = outputs.policy?.data;
    if (!(policy instanceof Float32Array) || this.policySize <= 0) return result;
    const memory = this.wasm.memory.buffer;
    const maxLegalWidth = this.wasm.max_legal_width();
    const legalBase = this.wasm.legal_ptr();
    const srcLegal = new BigInt64Array(memory, legalBase, this.wasm.max_slots() * maxLegalWidth);
    const moveCodes = new Uint32Array(memory, this.wasm.slot_legal_move_code_ptr(), this.wasm.max_slots() * maxLegalWidth);
    const legalCounts = new Uint32Array(memory, this.wasm.slot_legal_count_ptr(), this.wasm.max_slots());
    const actionValues = outputs.action_values?.data instanceof Float32Array ? outputs.action_values.data : undefined;
    const ranks = outputs.rank_scores?.data instanceof Float32Array ? outputs.rank_scores.data : undefined;
    const regrets = outputs.regrets?.data instanceof Float32Array ? outputs.regrets.data : undefined;
    const count = Math.min(this.legalWidth, legalCounts[item.slot] ?? 0);
    if (count <= 0) return result;
    const raw: EdgeEval[] = [];
    let maxLogit = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < count; j++) {
      const policyIndex = Number(srcLegal[item.slot * maxLegalWidth + j]);
      const logit = policyIndex >= 0 && policyIndex < this.policySize ? policy[batchIndex * this.policySize + policyIndex] : -100;
      maxLogit = Math.max(maxLogit, logit);
      const avOff = batchIndex * this.legalWidth + j;
      raw.push({
        moveCode: moveCodes[item.slot * maxLegalWidth + j],
        prior: 0,
        policyLogit: logit,
        ...(actionValues ? { av: actionValues[avOff] } : {}),
        ...(ranks ? { rank: ranks[avOff] } : {}),
        ...(regrets ? { regret: regrets[avOff] } : {}),
      });
    }
    let denom = 0;
    for (const edge of raw) denom += Math.exp(edge.policyLogit - maxLogit);
    for (const edge of raw) edge.prior = Math.exp(edge.policyLogit - maxLogit) / Math.max(1e-12, denom);
    result.edges = raw;
    return result;
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
        const feeds = this.makeFeeds(items);
        const t0 = performance.now();
        const fetches = selectedBrokerOutputs(this.session);
        const outputs = fetches ? await this.session.run(feeds, fetches) : await this.session.run(feeds);
        const runMs = performance.now() - t0;
        this.stats.runMs.push(runMs);
        this.stats.evalRuns++;
        this.stats.positionsEvaluated += items.length;
        this.stats.batchSizes[String(items.length)] = (this.stats.batchSizes[String(items.length)] ?? 0) + 1;
        for (let i = 0; i < items.length; i++) {
          const item = items[i];
          const result = this.outputForItem(outputs, item, i);
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

function slotKeyViews(wasm: BridgeExports): { lo: Uint32Array; hi: Uint32Array; game: Uint32Array; moveCode: Uint32Array } {
  return {
    lo: new Uint32Array(wasm.memory.buffer, wasm.slot_hash_lo_ptr(), wasm.max_slots()),
    hi: new Uint32Array(wasm.memory.buffer, wasm.slot_hash_hi_ptr(), wasm.max_slots()),
    game: new Uint32Array(wasm.memory.buffer, wasm.slot_game_ptr(), wasm.max_slots()),
    moveCode: new Uint32Array(wasm.memory.buffer, wasm.slot_move_code_ptr(), wasm.max_slots()),
  };
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
    metrics.push(['games complete', `${s.completedGames}/${s.games}`]);
    metrics.push(['plies played', String(s.pliesPlayed)]);
    metrics.push(['plies/s', s.pliesPerSec.toFixed(1)]);
    metrics.push(['leaf req/s', s.leafRequestsPerSec.toFixed(0)]);
    metrics.push(['NN pos/s', s.nnPosPerSec.toFixed(0)]);
    metrics.push(['leaf requests', String(s.logicalRequests)]);
    metrics.push(['NN positions', String(s.positionsEvaluated)]);
    metrics.push(['reduction factor', `${s.nnReductionFactor.toFixed(2)}×`]);
    metrics.push(['cache hits', `${(s.cacheHitRate * 100).toFixed(1)}%`]);
    metrics.push(['in-flight hits', `${(s.inflightHitRate * 100).toFixed(1)}%`]);
    metrics.push(['avg batch', s.avgBatch.toFixed(1)]);
    metrics.push(['search mode', s.searchMode]);
    if (s.puctVisits !== undefined) metrics.push(['PUCT visits/move', String(s.puctVisits)]);
    if (s.puctExpansions !== undefined) metrics.push(['PUCT expansions', String(s.puctExpansions)]);
    if (s.puctLeafEvals !== undefined) metrics.push(['PUCT leaf evals', String(s.puctLeafEvals)]);
    if (s.puctLeafWaves !== undefined) metrics.push(['PUCT leaf waves/drain', String(s.puctLeafWaves)]);
    metrics.push(['run p50/p95', `${s.run.p50.toFixed(2)} / ${s.run.p95.toFixed(2)} ms`]);
    metrics.push(['wait p50/p95', `${s.queueWait.p50.toFixed(2)} / ${s.queueWait.p95.toFixed(2)} ms`]);
  }
  metricsEl.innerHTML = metrics.map(([k, v]) => `<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
  batchRowsEl.innerHTML = s ? Object.entries(s.batchSizes).sort((a, b) => Number(a[0]) - Number(b[0])).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('') : '';
  jsonEl.textContent = JSON.stringify(state, null, 2);
}

type SlotCandidate = { slot: number; key: string; moveCode: number };
type SearchNode = { path: number[]; expanded: boolean; visits: number; pendingEval: number; children: SearchEdge[] };
type SearchEdge = { moveCode: number; prior: number; visits: number; valueSum: number; pending: number; child?: SearchNode };
type PuctCounters = { expansions: number; leafEvals: number; rootVisits: number };

function makeNode(path: number[]): SearchNode {
  return { path, expanded: false, visits: 0, pendingEval: 0, children: [] };
}

function writePath(wasm: BridgeExports, path: number[]) {
  if (path.length > wasm.max_path_moves()) throw new Error(`path length ${path.length} exceeds WASM max_path_moves=${wasm.max_path_moves()}`);
  const ptr = wasm.path_move_ptr();
  const dst = new Uint32Array(wasm.memory.buffer, ptr, wasm.max_path_moves());
  for (let i = 0; i < path.length; i++) dst[i] = path[i];
}

function slotKey(keys: ReturnType<typeof slotKeyViews>, slot: number): string {
  return `bt4:${keys.hi[slot].toString(16).padStart(8, '0')}${keys.lo[slot].toString(16).padStart(8, '0')}`;
}

function preparePathCandidates(wasm: BridgeExports, gameIdx: number, path: number[], startSlot: number, tokenStride: number, legalWidth: number): SlotCandidate[] {
  writePath(wasm, path);
  const count = wasm.prepare_game_path_to_slots(gameIdx, path.length, startSlot, tokenStride, legalWidth, wasm.max_slots() - startSlot);
  const keys = slotKeyViews(wasm);
  const out: SlotCandidate[] = [];
  for (let j = 0; j < count; j++) {
    const slot = startSlot + j;
    out.push({ slot, key: slotKey(keys, slot), moveCode: keys.moveCode[slot] });
  }
  return out;
}

function expandNode(wasm: BridgeExports, gameIdx: number, node: SearchNode, startSlot: number, tokenStride: number, legalWidth: number, counters: PuctCounters): number {
  if (node.expanded) return 0;
  const candidates = preparePathCandidates(wasm, gameIdx, node.path, startSlot, tokenStride, legalWidth);
  const prior = candidates.length ? 1 / candidates.length : 0;
  node.children = candidates.map((c) => ({ moveCode: c.moveCode, prior, visits: 0, valueSum: 0, pending: 0 }));
  node.expanded = true;
  counters.expansions++;
  return candidates.length;
}

function selectEdge(node: SearchNode, cpuct: number, pendingVirtualLoss = 0): SearchEdge | undefined {
  if (!node.children.length) return undefined;
  const parentVisits = Math.max(1, node.visits + node.children.reduce((a, e) => a + e.pending * pendingVirtualLoss, 0));
  let best = node.children[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const edge of node.children) {
    const childPending = edge.child?.pendingEval ?? 0;
    const pending = edge.pending + childPending;
    const effectiveVisits = edge.visits + pending * pendingVirtualLoss;
    const q = edge.visits > 0 ? edge.valueSum / edge.visits : 0;
    const pendingPenalty = pendingVirtualLoss > 0 && pending > 0 ? pendingVirtualLoss * pending : 0;
    const pessimisticQ = pendingPenalty > 0 ? ((edge.valueSum - pendingPenalty) / Math.max(1, effectiveVisits)) : q;
    const u = cpuct * edge.prior * Math.sqrt(parentVisits) / (1 + effectiveVisits);
    const score = pessimisticQ + u;
    if (score > bestScore) { bestScore = score; best = edge; }
  }
  return best;
}

function backupPuct(root: SearchNode, selected: { edge: SearchEdge; nodeDepth: number }[], leafDepth: number, leafQ: number) {
  root.visits++;
  for (const step of selected) {
    const valueFromNode = ((leafDepth - step.nodeDepth) & 1) === 0 ? leafQ : -leafQ;
    step.edge.visits++;
    step.edge.valueSum += valueFromNode;
    if (step.edge.child) step.edge.child.visits++;
  }
}

function reservePendingLeaf(node: SearchNode, selected: { edge: SearchEdge; nodeDepth: number }[]) {
  node.pendingEval++;
  for (const step of selected) step.edge.pending++;
}

function releasePendingLeaf(node: SearchNode, selected: { edge: SearchEdge; nodeDepth: number }[]) {
  node.pendingEval = Math.max(0, node.pendingEval - 1);
  for (const step of selected) step.edge.pending = Math.max(0, step.edge.pending - 1);
}

function preparePositionEval(wasm: BridgeExports, broker: EvalBroker, gameIdx: number, path: number[], slot: number, tokenStride: number, legalWidth: number): Promise<EvalResult> | undefined {
  writePath(wasm, path);
  const legalCount = wasm.prepare_game_path_position_slot(gameIdx, path.length, slot, tokenStride, legalWidth);
  if (legalCount === 0 && path.length > 0) return undefined;
  const keys = slotKeyViews(wasm);
  return broker.request(slotKey(keys, slot), slot);
}

function normalizeEdgePriors(edges: SearchEdge[]) {
  let sum = 0;
  for (const e of edges) sum += Math.max(0, e.prior);
  if (sum <= 0) {
    const p = edges.length ? 1 / edges.length : 0;
    for (const e of edges) e.prior = p;
  } else {
    for (const e of edges) e.prior = Math.max(0, e.prior) / sum;
  }
}

function expandNodeFromEval(node: SearchNode, result: EvalResult, counters: PuctCounters) {
  if (node.expanded) return;
  const edges = result.edges ?? [];
  node.children = edges.map((e) => {
    const avBias = typeof e.av === 'number' && Number.isFinite(e.av) ? Math.max(-1, Math.min(1, e.av)) : 0;
    const regretBoost = typeof e.regret === 'number' && Number.isFinite(e.regret) ? Math.max(-2, Math.min(2, -e.regret)) * 0.02 : 0;
    return { moveCode: e.moveCode, prior: Math.max(1e-6, e.prior + 0.03 * avBias + regretBoost), visits: 0, valueSum: 0, pending: 0 };
  });
  normalizeEdgePriors(node.children);
  node.expanded = true;
  counters.expansions++;
}

function rng(seed: number): () => number {
  let x = seed >>> 0;
  return () => {
    x = (x * 1664525 + 1013904223) >>> 0;
    return x / 0x100000000;
  };
}

function sampleGammaAlpha(alpha: number, rand: () => number): number {
  if (alpha <= 0) return 0;
  if (alpha < 1) return sampleGammaAlpha(alpha + 1, rand) * Math.pow(Math.max(1e-12, rand()), 1 / alpha);
  const d = alpha - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let u = 0; let v = 0; let x = 0;
    do {
      const a = Math.max(1e-12, rand());
      const b = Math.max(1e-12, rand());
      x = Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    u = rand();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(Math.max(1e-12, u)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function applyDirichletNoise(edges: SearchEdge[], alpha: number, frac: number, rand: () => number) {
  if (!edges.length || alpha <= 0 || frac <= 0) return;
  const noise = edges.map(() => sampleGammaAlpha(alpha, rand));
  const sum = noise.reduce((a, b) => a + b, 0) || 1;
  for (let i = 0; i < edges.length; i++) edges[i].prior = (1 - frac) * edges[i].prior + frac * (noise[i] / sum);
  normalizeEdgePriors(edges);
}

function chooseRootEdge(root: SearchNode, temperature: number, rand: () => number): SearchEdge | undefined {
  if (!root.children.length) return undefined;
  if (temperature <= 1e-6) return root.children.reduce((a, b) => (b.visits > a.visits ? b : a), root.children[0]);
  const weights = root.children.map((e) => Math.pow(Math.max(0, e.visits), 1 / temperature));
  let sum = weights.reduce((a, b) => a + b, 0);
  if (sum <= 0) {
    sum = root.children.reduce((a, e) => a + e.prior, 0);
    let r = rand() * sum;
    for (const e of root.children) { r -= e.prior; if (r <= 0) return e; }
    return root.children[root.children.length - 1];
  }
  let r = rand() * sum;
  for (let i = 0; i < root.children.length; i++) { r -= weights[i]; if (r <= 0) return root.children[i]; }
  return root.children[root.children.length - 1];
}

function sampleGumbel(rand: () => number): number {
  return -Math.log(-Math.log(Math.max(1e-12, Math.min(1 - 1e-12, rand()))));
}

function rebaseSubtree(node: SearchNode, drop: number, seen = new Set<SearchNode>()) {
  if (seen.has(node)) return;
  seen.add(node);
  node.path = node.path.slice(Math.min(drop, node.path.length));
  node.pendingEval = 0;
  for (const edge of node.children) {
    edge.pending = 0;
    if (edge.child) rebaseSubtree(edge.child, drop, seen);
  }
}

async function runPolicyPuctMoveBatch(
  wasm: BridgeExports,
  broker: EvalBroker,
  gameIds: number[],
  tokenStride: number,
  legalWidth: number,
  visits: number,
  cpuct: number,
  maxDepth: number,
  counters: PuctCounters,
  temperature: number,
  noiseAlpha: number,
  noiseFrac: number,
  rand: () => number,
  rows: SelfplayRow[],
  maxRows: number,
  leafWaves: number,
  pendingVirtualLoss: number,
  rootStore: Map<number, SearchNode> | undefined,
): Promise<number> {
  const roots = new Map<number, SearchNode>();
  let slot = 0;
  const rootJobs: { g: number; root: SearchNode; promise?: Promise<EvalResult> }[] = [];
  for (const g of gameIds) {
    const root = rootStore?.get(g) ?? makeNode([]);
    roots.set(g, root);
    if (!root.expanded) rootJobs.push({ g, root, promise: preparePositionEval(wasm, broker, g, [], slot++, tokenStride, legalWidth) });
  }
  await broker.drain();
  for (const job of rootJobs) {
    if (!job.promise) continue;
    expandNodeFromEval(job.root, await job.promise, counters);
  }
  for (const root of roots.values()) applyDirichletNoise(root.children, noiseAlpha, noiseFrac, rand);

  const wavesPerDrain = Math.max(1, Math.floor(leafWaves));
  for (let visit = 0; visit < visits; visit += wavesPerDrain) {
    slot = 0;
    const scheduled: { root: SearchNode; node: SearchNode; selected: { edge: SearchEdge; nodeDepth: number }[]; leafDepth: number; promise?: Promise<EvalResult> }[] = [];
    const waves = Math.min(wavesPerDrain, visits - visit);
    for (let wave = 0; wave < waves; wave++) {
      for (const g of gameIds) {
        const root = roots.get(g)!;
        let node = root;
        const selected: { edge: SearchEdge; nodeDepth: number }[] = [];
        for (let depth = 0; depth < maxDepth; depth++) {
          if (!node.expanded) break;
          const edge = selectEdge(node, cpuct, pendingVirtualLoss);
          if (!edge) break;
          selected.push({ edge, nodeDepth: node.path.length });
          edge.child ??= makeNode([...node.path, edge.moveCode]);
          node = edge.child;
          if (!node.expanded || node.path.length >= maxDepth) break;
        }
        reservePendingLeaf(node, selected);
        scheduled.push({ root, node, selected, leafDepth: node.path.length, promise: preparePositionEval(wasm, broker, g, node.path, slot++, tokenStride, legalWidth) });
        if (slot >= wasm.max_slots() - 1) { await broker.drain(); slot = 0; }
      }
    }
    await broker.drain();
    for (const item of scheduled) {
      releasePendingLeaf(item.node, item.selected);
      if (!item.promise) continue;
      const result = await item.promise;
      expandNodeFromEval(item.node, result, counters);
      backupPuct(item.root, item.selected, item.leafDepth, result.q);
      counters.rootVisits++;
      counters.leafEvals++;
    }
    if (((visit / wavesPerDrain) & 7) === 7) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  let applied = 0;
  slot = 0;
  for (const g of gameIds) {
    const root = roots.get(g)!;
    const chosen = chooseRootEdge(root, temperature, rand);
    if (!chosen) continue;
    const candidates = preparePathCandidates(wasm, g, [], slot, tokenStride, legalWidth);
    slot += candidates.length;
    const rootMove = candidates.find((c) => c.moveCode === chosen.moveCode);
    if (rows.length < maxRows) {
      const topMoves = [...root.children]
        .sort((a, b) => b.visits - a.visits)
        .slice(0, 8)
        .map((e) => ({ moveCode: e.moveCode, uci: moveCodeToUci(e.moveCode), visits: e.visits, q: e.visits ? e.valueSum / e.visits : 0, prior: e.prior }));
      rows.push({ game: g, ply: wasm.game_ply(g), fen: currentFen(wasm, g), playedMoveCode: chosen.moveCode, playedUci: moveCodeToUci(chosen.moveCode), visits: root.visits, temperature, noiseAlpha, noiseFrac, topMoves });
    }
    if (rootMove && wasm.apply_slot(rootMove.slot)) {
      applied++;
      if (rootStore) {
        const nextRoot = chosen.child ?? makeNode([]);
        rebaseSubtree(nextRoot, 1);
        rootStore.set(g, nextRoot);
      }
    } else if (rootStore) {
      rootStore.delete(g);
    }
    if (slot >= wasm.max_slots() - 256) slot = 0;
  }
  return applied;
}

async function runGumbelRootMoveBatch(
  wasm: BridgeExports,
  broker: EvalBroker,
  gameIds: number[],
  tokenStride: number,
  legalWidth: number,
  topK: number,
  policyScale: number,
  temperature: number,
  noiseAlpha: number,
  noiseFrac: number,
  rand: () => number,
  rows: SelfplayRow[],
  maxRows: number,
  counters: PuctCounters,
): Promise<number> {
  let slot = 0;
  const roots: { g: number; root: SearchNode; promise?: Promise<EvalResult> }[] = [];
  for (const g of gameIds) {
    const root = makeNode([]);
    roots.push({ g, root, promise: preparePositionEval(wasm, broker, g, [], slot++, tokenStride, legalWidth) });
  }
  await broker.drain();
  for (const r of roots) {
    if (!r.promise) continue;
    expandNodeFromEval(r.root, await r.promise, counters);
    applyDirichletNoise(r.root.children, noiseAlpha, noiseFrac, rand);
  }

  slot = 0;
  const jobs: { g: number; root: SearchNode; edge: SearchEdge; slot: number; promise: Promise<EvalResult> }[] = [];
  for (const r of roots) {
    const root = r.root;
    const ranked = root.children
      .map((edge) => ({ edge, gumbel: sampleGumbel(rand), key: Math.log(Math.max(1e-12, edge.prior)) + sampleGumbel(rand) }))
      .sort((a, b) => b.key - a.key)
      .slice(0, Math.max(1, Math.min(topK, root.children.length)));
    const candidates = preparePathCandidates(wasm, r.g, [], slot, tokenStride, legalWidth);
    slot += candidates.length;
    const byMove = new Map(candidates.map((c) => [c.moveCode, c]));
    for (const item of ranked) {
      const candidate = byMove.get(item.edge.moveCode);
      if (!candidate) continue;
      jobs.push({ g: r.g, root, edge: item.edge, slot: candidate.slot, promise: broker.request(candidate.key, candidate.slot) });
    }
    if (slot >= wasm.max_slots() - 512) { await broker.drain(); slot = 0; }
  }
  await broker.drain();

  const byGame = new Map<number, { root: SearchNode; edge: SearchEdge; result: EvalResult }[]>();
  for (const job of jobs) {
    const result = await job.promise;
    job.edge.visits++;
    job.edge.valueSum += -result.q;
    job.root.visits++;
    counters.leafEvals++;
    const list = byGame.get(job.g) ?? [];
    list.push({ root: job.root, edge: job.edge, result });
    byGame.set(job.g, list);
  }

  let applied = 0;
  slot = 0;
  for (const [g, items] of byGame) {
    let best = items[0];
    let bestScore = Number.NEGATIVE_INFINITY;
    for (const item of items) {
      const q = -item.result.q;
      const score = q + policyScale * Math.log(Math.max(1e-12, item.edge.prior));
      if (score > bestScore) { bestScore = score; best = item; }
    }
    if (temperature > 1e-6 && items.length > 1) {
      const weights = items.map((item) => Math.exp(((-item.result.q) + policyScale * Math.log(Math.max(1e-12, item.edge.prior))) / temperature));
      let sum = weights.reduce((a, b) => a + b, 0);
      let r = rand() * sum;
      for (let i = 0; i < items.length; i++) { r -= weights[i]; if (r <= 0) { best = items[i]; break; } }
    }
    const candidates = preparePathCandidates(wasm, g, [], slot, tokenStride, legalWidth);
    slot += candidates.length;
    const rootMove = candidates.find((c) => c.moveCode === best.edge.moveCode);
    if (rows.length < maxRows) {
      const topMoves = [...items]
        .sort((a, b) => (-b.result.q + policyScale * Math.log(Math.max(1e-12, b.edge.prior))) - (-a.result.q + policyScale * Math.log(Math.max(1e-12, a.edge.prior))))
        .slice(0, 8)
        .map((item) => ({ moveCode: item.edge.moveCode, uci: moveCodeToUci(item.edge.moveCode), visits: item.edge.visits, q: -item.result.q, prior: item.edge.prior }));
      rows.push({ game: g, ply: wasm.game_ply(g), fen: currentFen(wasm, g), playedMoveCode: best.edge.moveCode, playedUci: moveCodeToUci(best.edge.moveCode), visits: items.length, temperature, noiseAlpha, noiseFrac, topMoves });
    }
    if (rootMove && wasm.apply_slot(rootMove.slot)) applied++;
    counters.rootVisits += items.length;
    if (slot >= wasm.max_slots() - 256) slot = 0;
  }
  return applied;
}

async function runPuctMoveBatch(
  wasm: BridgeExports,
  broker: EvalBroker,
  gameIds: number[],
  tokenStride: number,
  legalWidth: number,
  visits: number,
  cpuct: number,
  maxDepth: number,
  counters: PuctCounters,
): Promise<number> {
  const roots = new Map<number, SearchNode>();
  let slot = 0;
  for (const g of gameIds) {
    const root = makeNode([]);
    const used = expandNode(wasm, g, root, slot, tokenStride, legalWidth, counters);
    slot += used;
    roots.set(g, root);
    if (slot >= wasm.max_slots() - 256) slot = 0;
  }

  for (let visit = 0; visit < visits; visit++) {
    slot = 0;
    const scheduled: { root: SearchNode; selected: { edge: SearchEdge; nodeDepth: number }[]; leafDepth: number; promise: Promise<EvalResult> | undefined }[] = [];
    for (const g of gameIds) {
      const root = roots.get(g)!;
      let node = root;
      const selected: { edge: SearchEdge; nodeDepth: number }[] = [];
      for (let depth = 0; depth < maxDepth; depth++) {
        if (!node.expanded) {
          const used = expandNode(wasm, g, node, slot, tokenStride, legalWidth, counters);
          slot += used;
        }
        const edge = selectEdge(node, cpuct);
        if (!edge) break;
        selected.push({ edge, nodeDepth: node.path.length });
        const childPath = [...node.path, edge.moveCode];
        if (edge.visits === 0 || childPath.length >= maxDepth) {
          const parentPath = childPath.slice(0, -1);
          const candidates = preparePathCandidates(wasm, g, parentPath, slot, tokenStride, legalWidth);
          slot += candidates.length;
          const leaf = candidates.find((c) => c.moveCode === edge.moveCode);
          const promise = leaf ? broker.request(leaf.key, leaf.slot) : undefined;
          if (promise) counters.leafEvals++;
          scheduled.push({ root, selected, leafDepth: childPath.length, promise });
          edge.child ??= makeNode(childPath);
          break;
        }
        edge.child ??= makeNode(childPath);
        node = edge.child;
      }
      if (slot >= wasm.max_slots() - 512) {
        await broker.drain();
        slot = 0;
      }
    }
    await broker.drain();
    for (const item of scheduled) {
      const result = item.promise ? await item.promise : { q: 0 } as EvalResult;
      backupPuct(item.root, item.selected, item.leafDepth, result.q);
      counters.rootVisits++;
    }
    if ((visit & 3) === 3) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  let applied = 0;
  slot = 0;
  for (const g of gameIds) {
    const root = roots.get(g)!;
    const best = root.children.reduce<SearchEdge | undefined>((acc, e) => {
      if (!acc) return e;
      if (e.visits !== acc.visits) return e.visits > acc.visits ? e : acc;
      const q = e.visits > 0 ? e.valueSum / e.visits : Number.NEGATIVE_INFINITY;
      const aq = acc.visits > 0 ? acc.valueSum / acc.visits : Number.NEGATIVE_INFINITY;
      return q > aq ? e : acc;
    }, undefined);
    if (!best) continue;
    const candidates = preparePathCandidates(wasm, g, [], slot, tokenStride, legalWidth);
    slot += candidates.length;
    const rootMove = candidates.find((c) => c.moveCode === best.moveCode);
    if (rootMove && wasm.apply_slot(rootMove.slot)) applied++;
    if (slot >= wasm.max_slots() - 256) slot = 0;
  }
  return applied;
}

const config = {
  games: intParam('games', 96),
  maxPlies: intParam('maxPlies', 40),
  openingCount: intParam('openingCount', 24),
  openingDepth: intParam('openingDepth', 4),
  batchTarget: intParam('batchTarget', 64),
  maxWaitMs: floatParam('maxWaitMs', 1.5),
  cacheCap: intParam('cacheCap', 100000),
  seed: intParam('seed', 12648430),
  mode: params.get('mode') ?? 'puct-policy',
  searchVisits: intParam('searchVisits', 16),
  cpuct: floatParam('cpuct', 1.35),
  maxDepth: intParam('maxDepth', 8),
  temperature: floatParam('temperature', 0.85),
  rootNoiseAlpha: floatParam('rootNoiseAlpha', 0.30),
  rootNoiseFrac: floatParam('rootNoiseFrac', 0.25),
  leafWaves: intParam('leafWaves', 1),
  pendingVirtualLoss: floatParam('pendingVirtualLoss', 16),
  reuseTree: intParam('reuseTree', 1),
  gumbelTopK: intParam('gumbelTopK', 16),
  gumbelPolicyScale: floatParam('gumbelPolicyScale', 0.25),
  maxSelfplayRows: intParam('maxSelfplayRows', 256),
};

const state: Result = { startedAt: new Date().toISOString(), model: modelPath, meta: metaPath, wasm: wasmPath, ortEp: params.get('ortEp') ?? params.get('ep') ?? 'webgpu', config, errors: [] };

async function run(): Promise<Result> {
  state.startedAt = new Date().toISOString();
  state.finishedAt = undefined;
  state.errors = [];
  state.stats = undefined;
  statusEl.textContent = 'loading meta/wasm and diagnostics';
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);
  const [meta, wasm] = await Promise.all([loadJson<Meta>(metaPath), loadBridge(wasmPath)]);
  const tokenStride = Number(meta.token_features ?? ((meta.history_plies ?? 8) + 9));
  const legalWidth = Number(meta.onnx_fixed_legal_moves ?? meta.max_legal_moves ?? 128);
  const policySize = Number(meta.policy_size ?? 20480);
  const attackFeatureCount = Number(meta.attack_summary_feature_count ?? 0);
  if (tokenStride > wasm.max_token_stride()) throw new Error(`tokenStride=${tokenStride} exceeds wasm max=${wasm.max_token_stride()}`);
  if (legalWidth > wasm.max_legal_width()) throw new Error(`legalWidth=${legalWidth} exceeds wasm max=${wasm.max_legal_width()}`);
  if (attackFeatureCount > 0) {
    if (meta.attack_summary_schema !== 'threatgraph_square_summary_v1') throw new Error(`unsupported attack_summary_schema=${meta.attack_summary_schema}; self-play bridge currently supports TG1 only`);
    if (!wasm.max_attack_features || !wasm.attack_summary_ptr) throw new Error('WASM bridge missing TG1 attack_summary exports; rebuild tl_wasm_selfplay_bridge');
    if (attackFeatureCount > wasm.max_attack_features()) throw new Error(`attackFeatureCount=${attackFeatureCount} exceeds wasm max=${wasm.max_attack_features()}`);
  }
  (config as Record<string, number | string>).attackFeatureCount = attackFeatureCount;
  if (config.games > wasm.max_games()) throw new Error(`games=${config.games} exceeds wasm max=${wasm.max_games()}`);
  statusEl.textContent = `creating ORT session; tokenStride=${tokenStride} legalWidth=${legalWidth} attackFeatures=${attackFeatureCount}`;
  const session = await ort.createOrtSession(modelPath);
  state.diagnostics = await ort.collectOrtRuntimeDiagnostics({ probeAdapter: state.ortEp !== 'wasm' });
  render(state);

  const broker = new EvalBroker(session, wasm, tokenStride, legalWidth, policySize, attackFeatureCount, config.batchTarget, config.maxWaitMs, config.cacheCap);
  const active0 = wasm.reset_games(config.games, config.maxPlies, config.openingCount, config.openingDepth, config.seed);
  const games = wasm.game_count();
  let pliesPlayed = 0;
  const t0 = performance.now();
  statusEl.textContent = `running Rust-WASM self-play: ${games} games (${active0} active)`;

  const puctCounters: PuctCounters = { expansions: 0, leafEvals: 0, rootVisits: 0 };
  const selfplayRows: SelfplayRow[] = [];
  const searchRand = rng(Number(config.seed) ^ 0x51f15eED);
  const policyRootStore = Number(config.reuseTree) ? new Map<number, SearchNode>() : undefined;
  while (wasm.active_games() > 0) {
    const activeGameIds = Array.from({ length: games }, (_, g) => g).filter((g) => !wasm.game_done(g));
    if (!activeGameIds.length) break;
    if (config.mode === 'gumbel-root') {
      pliesPlayed += await runGumbelRootMoveBatch(
        wasm,
        broker,
        activeGameIds,
        tokenStride,
        legalWidth,
        Number(config.gumbelTopK),
        Number(config.gumbelPolicyScale),
        Number(config.temperature),
        Number(config.rootNoiseAlpha),
        Number(config.rootNoiseFrac),
        searchRand,
        selfplayRows,
        Number(config.maxSelfplayRows),
        puctCounters,
      );
    } else if (config.mode === 'puct-policy') {
      pliesPlayed += await runPolicyPuctMoveBatch(
        wasm,
        broker,
        activeGameIds,
        tokenStride,
        legalWidth,
        Number(config.searchVisits),
        Number(config.cpuct),
        Number(config.maxDepth),
        puctCounters,
        Number(config.temperature),
        Number(config.rootNoiseAlpha),
        Number(config.rootNoiseFrac),
        searchRand,
        selfplayRows,
        Number(config.maxSelfplayRows),
        Number(config.leafWaves),
        Number(config.pendingVirtualLoss),
        policyRootStore,
      );
    } else if (config.mode === 'puct') {
      pliesPlayed += await runPuctMoveBatch(
        wasm,
        broker,
        activeGameIds,
        tokenStride,
        legalWidth,
        Number(config.searchVisits),
        Number(config.cpuct),
        Number(config.maxDepth),
        puctCounters,
      );
    } else {
      let slot = 0;
      const byGame = new Map<number, { slot: number; promise: Promise<EvalResult> }[]>();
      for (const g of activeGameIds) {
        const count = wasm.prepare_game_to_slots(g, slot, tokenStride, legalWidth, wasm.max_slots() - slot);
        if (count === 0) continue;
        // Rust Vec allocations inside prepare_game_to_slots may grow WASM memory and detach old JS views.
        // Recreate hash views after each prepare before reading slot keys.
        const keys = slotKeyViews(wasm);
        const arr: { slot: number; promise: Promise<EvalResult> }[] = [];
        for (let j = 0; j < count; j++) {
          const s = slot + j;
          arr.push({ slot: s, promise: broker.request(slotKey(keys, s), s) });
        }
        byGame.set(g, arr);
        slot += count;
        if (slot >= wasm.max_slots() - 256) break;
      }
      if (byGame.size === 0) break;
      await broker.drain();
      for (const [, candidates] of byGame) {
        const results = await Promise.all(candidates.map((c) => c.promise));
        let best = 0;
        let bestScore = Number.NEGATIVE_INFINITY;
        for (let i = 0; i < results.length; i++) {
          // Child q is from child side-to-move perspective, so parent prefers lower child q.
          const score = -results[i].q + 1e-9 * results[i].checksum;
          if (score > bestScore) { bestScore = score; best = i; }
        }
        if (wasm.apply_slot(candidates[best].slot)) pliesPlayed++;
      }
    }
    statusEl.textContent = `running ${config.mode}: active=${wasm.active_games()} plies=${pliesPlayed} evals=${broker.stats.logicalRequests}`;
    if ((pliesPlayed & 127) === 0) await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const totalMs = performance.now() - t0;
  const stats = broker.stats;
  const completedGames = Array.from({ length: games }, (_, g) => wasm.game_done(g)).filter(Boolean).length;
  const nnRunSeconds = stats.runMs.reduce((a, b) => a + b, 0) / 1000;
  state.selfplayRows = selfplayRows;
  state.stats = {
    ...stats,
    queueWait: percentile(stats.queueWaitMs),
    run: percentile(stats.runMs),
    totalMs,
    games,
    completedGames,
    pliesPlayed,
    avgPliesPerGame: pliesPlayed / Math.max(1, games),
    leafRequestsPerSec: stats.logicalRequests / (totalMs / 1000),
    pliesPerSec: pliesPlayed / (totalMs / 1000),
    nnPosPerSec: stats.positionsEvaluated / Math.max(1e-9, nnRunSeconds),
    cacheHitRate: stats.cacheHits / Math.max(1, stats.logicalRequests),
    inflightHitRate: stats.inflightHits / Math.max(1, stats.logicalRequests),
    nnReductionFactor: stats.logicalRequests / Math.max(1, stats.positionsEvaluated),
    avgBatch: stats.positionsEvaluated / Math.max(1, stats.evalRuns),
    searchMode: String(config.mode),
    puctVisits: String(config.mode).startsWith('puct') ? Number(config.searchVisits) : undefined,
    puctMaxDepth: String(config.mode).startsWith('puct') ? Number(config.maxDepth) : undefined,
    puctExpansions: String(config.mode).startsWith('puct') ? puctCounters.expansions : undefined,
    puctLeafEvals: String(config.mode).startsWith('puct') ? puctCounters.leafEvals : undefined,
    puctAvgRootVisits: String(config.mode).startsWith('puct') ? puctCounters.rootVisits / Math.max(1, pliesPlayed) : undefined,
    puctLeafWaves: String(config.mode) === 'puct-policy' ? Number(config.leafWaves) : undefined,
    puctReuseTree: String(config.mode) === 'puct-policy' ? Number(config.reuseTree) : undefined,
    gumbelTopK: String(config.mode) === 'gumbel-root' ? Number(config.gumbelTopK) : undefined,
  };
  state.finishedAt = new Date().toISOString();
  statusEl.textContent = `done in ${totalMs.toFixed(1)}ms`;
  render(state);
  return state;
}

window.tinyLeelaWasmSelfplayBroker = { run, state };
(document.getElementById('run') as HTMLButtonElement).onclick = () => { void run(); };
(document.getElementById('copyJson') as HTMLButtonElement).onclick = () => navigator.clipboard?.writeText(JSON.stringify(state, null, 2));
if (params.get('autorun') === '1') void run();
