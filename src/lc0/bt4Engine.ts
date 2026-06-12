// Lc0 big-net (BT4-it332, t3-512 distill) browser integration.
//
// Big nets are large and attention-heavy, so they are handled with care:
//   - WebGPU-gated: callers must check `probeBt4Support()` before exposing them.
//   - Lazy: the dedicated search worker is only created/initialized on first use
//     (the first init fetches the model, cached afterwards by the Cache API).
//   - Off the main thread: search runs entirely inside searchWorker.ts so heavy
//     inference never blocks the UI.
//   - Disposable: `dispose()` terminates the worker, freeing model memory.
import { collectOrtRuntimeDiagnostics, requestedOrtExecutionProvider } from '../nn/ortRuntime.ts';
import type { Lc0EvaluatorInput } from './onnxEvaluator.ts';

export interface BigNetConfig {
  key: 'bt4' | 't3' | 'lqo';
  name: string;
  modelUrl: string;
  approxMb: number;
  recommendedBatchSize: number;
  recommendedPipelineDepth: number;
  /** Short export description for the one-time load warning. */
  exportNote: string;
  /**
   * Play-page visit ladder when running on the wasm-CPU fallback (per-level,
   * 5 entries). Sized from measured wasm eval latency (lqo ~210ms/eval,
   * t3 ~1.1s, BT4 ~1.5s) to keep worst-case moves in the seconds range.
   */
  wasmLevels: number[];
}

export const BT4_NET: BigNetConfig = {
  key: 'bt4',
  name: 'BT4-it332',
  modelUrl: '/models/lc0/BT4-1024x15x32h-swa-6147500-policytune-332.batch4.f16.onnx',
  approxMb: 353,
  recommendedBatchSize: 4,
  recommendedPipelineDepth: 1,
  exportNote: 'policytune-332 batch-4 f16 export',
  wasmLevels: [2, 3, 4, 6, 8],
};

// b8 is t3's measured sweet spot (b16 regressed 119 -> 140 ms at v16);
// see docs/lc0_t3_qdq_webnn_2026-06-10.md.
export const T3_NET: BigNetConfig = {
  key: 't3',
  name: 't3-512 distill',
  modelUrl: '/models/lc0/t3-512x15x16h-distill-swa-2767500.batch8.f16.onnx',
  approxMb: 163,
  recommendedBatchSize: 8,
  recommendedPipelineDepth: 1,
  exportNote: 'distill-swa-2767500 batch-8 f16 export',
  wasmLevels: [2, 4, 6, 8, 12],
};

// LeelaQueenOdds v2 (github.com/notune/LeelaQueenOdds): the public net behind
// the Lichess queen-odds bot, fine-tuned to win against humans from a queen
// down. Note: it evaluates the queen-odds start as equal — odds play only,
// not a general analysis net. The Lichess bot runs it with search-contempt
// (ScLimit) at 12-15k nodes; our PUCT applies drawScore + searchContemptLimit
// (see SearchOptions) scaled down to browser visit budgets.
export const LQO_NET: BigNetConfig = {
  key: 'lqo',
  name: 'Queen Odds',
  modelUrl: '/models/lc0/lqo_v2.f16.onnx',
  approxMb: 189,
  recommendedBatchSize: 8,
  recommendedPipelineDepth: 1,
  exportNote: 'LeelaQueenOdds v2 f16 export (dynamic batch)',
  wasmLevels: [4, 8, 16, 32, 64],
};

export const BIG_NETS: Record<BigNetConfig['key'], BigNetConfig> = { bt4: BT4_NET, t3: T3_NET, lqo: LQO_NET };

export const BT4_MODEL_NAME = BT4_NET.name;
export const BT4_MODEL_URL = BT4_NET.modelUrl;
export const BT4_APPROX_MB = BT4_NET.approxMb;
export const BT4_RECOMMENDED_SEARCH_BATCH_SIZE = BT4_NET.recommendedBatchSize;
export const BT4_RECOMMENDED_BATCH_PIPELINE_DEPTH = BT4_NET.recommendedPipelineDepth;

export interface Bt4SearchResult {
  fen: string;
  move?: string | null;
  visits: number;
  value: number;
  children: { uci: string; visits: number; q: number }[];
  pv: string[];
  multiPv?: string[][];
  stats?: { evalCalls?: number; cacheHits?: number };
  elapsedMs?: number;
  cancelled?: boolean;
}

export interface Bt4SearchOptions {
  visits?: number;
  movetimeMs?: number;
  multiPv?: number;
  reuseTree?: boolean;
  batchSize?: number;
  batchPipelineDepth?: number;
  evalCacheEntries?: number;
  /** WDL draw contempt for the searching side ([-1,1], 0 = off); see SearchOptions.drawScore. */
  drawScore?: number;
  /** Monty-style Elo-diff contempt ([-1000,1000], 0 = off); see SearchOptions.contemptElo. */
  contemptElo?: number;
  cpuct?: number;
  /** ScLimit-style search contempt (opponent visit budget, 0 = off). */
  searchContemptLimit?: number;
}

export type Bt4AssetStatus = 'unknown' | 'present' | 'missing';

let supportProbe: Promise<boolean> | null = null;
let supportedCached: boolean | null = null;
const assetProbes = new Map<string, Promise<Bt4AssetStatus>>();
const assetStatuses = new Map<string, Bt4AssetStatus>();

/** WebGPU usable for BT4? Cached after the first probe. */
export async function probeBt4Support(): Promise<boolean> {
  if (supportProbe) return supportProbe;
  supportProbe = (async () => {
    try {
      // An explicit ?ortEp=wasm override counts as "no WebGPU" so the UI and
      // visit ladders match the engine's actual (forced) backend.
      if (requestedOrtExecutionProvider() === 'wasm') {
        supportedCached = false;
        return supportedCached;
      }
      const diag = await collectOrtRuntimeDiagnostics({ probeAdapter: true });
      supportedCached = diag.webgpuAvailable === true && diag.adapter?.ok !== false;
    } catch {
      supportedCached = false;
    }
    return supportedCached;
  })();
  return supportProbe;
}

/** Last probed support result without re-probing (false until probed). */
export function bt4SupportedSync(): boolean {
  return supportedCached === true;
}

/** Last probed local model asset result without re-probing. */
export function bigNetAssetStatusSync(config: BigNetConfig = BT4_NET): Bt4AssetStatus {
  return assetStatuses.get(config.modelUrl) ?? 'unknown';
}

export function bt4AssetStatusSync(): Bt4AssetStatus {
  return bigNetAssetStatusSync(BT4_NET);
}

/** Browser-served big-net ONNX asset availability. Cached after the first probe. */
export async function checkBigNetAsset(config: BigNetConfig = BT4_NET, onStatus?: () => void): Promise<Bt4AssetStatus> {
  const existing = assetProbes.get(config.modelUrl);
  if (existing) return existing;
  const probe = (async () => {
    let status: Bt4AssetStatus;
    try {
      const response = await fetch(config.modelUrl, { method: 'HEAD', cache: 'no-store' });
      status = response.ok ? 'present' : 'missing';
    } catch {
      status = 'missing';
    }
    assetStatuses.set(config.modelUrl, status);
    onStatus?.();
    return status;
  })();
  assetProbes.set(config.modelUrl, probe);
  return probe;
}

export async function checkBt4Asset(onStatus?: () => void): Promise<Bt4AssetStatus> {
  return checkBigNetAsset(BT4_NET, onStatus);
}

/** A memory caution string when the device reports limited RAM, else null. */
export function bigNetMemoryCaution(config: BigNetConfig = BT4_NET): string | null {
  const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  if (typeof mem === 'number' && mem > 0 && mem <= 4) {
    return `This device reports ~${mem}GB RAM; the Lc0 ${config.name} net (~${config.approxMb}MB) may strain memory.`;
  }
  return null;
}

/** Human-readable one-time warning shown before the first big-net load. */
export function bigNetLoadWarning(config: BigNetConfig = BT4_NET): string {
  const caution = bigNetMemoryCaution(config);
  return `Lc0 ${config.name} is a large net (~${config.approxMb}MB download, cached after first load) and needs WebGPU. Arena uses the ${config.exportNote} with search tree reuse and leaf batching.${caution ? ` ${caution}` : ''}`;
}

export function bt4MemoryCaution(): string | null { return bigNetMemoryCaution(BT4_NET); }
export function bt4LoadWarning(): string { return bigNetLoadWarning(BT4_NET); }

/**
 * A lazily-initialized, worker-backed Lc0 big-net searcher. One instance owns
 * one searchWorker (one resident net). Call `dispose()` to free its memory.
 */
export class Bt4WorkerSearcher {
  readonly config: BigNetConfig;
  private worker: Worker | null = null;
  private ready = false;
  private initPromise: Promise<string> | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private activeSearchId: number | null = null;
  private configuredEvalCacheEntries = -1;
  backend = '';
  /** Model download progress during the first init (not called on reloads). */
  onDownloadProgress: ((loadedBytes: number, totalBytes?: number) => void) | null = null;

  constructor(config: BigNetConfig = BT4_NET) {
    this.config = config;
  }

  get loaded(): boolean {
    return this.ready;
  }

  private post<T>(message: Record<string, unknown>, onId?: (id: number) => void): Promise<T> {
    if (!this.worker) return Promise.reject(new Error('Lc0 BT4 worker unavailable'));
    const id = ++this.seq;
    onId?.(id);
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.worker!.postMessage({ ...message, id });
    });
  }

  /** Create + init the worker on first call (fetches ~353MB, then cached). */
  async init(options: { evalCacheEntries?: number } = {}): Promise<string> {
    const evalCacheEntries = Math.max(0, Math.floor(Number(options.evalCacheEntries ?? 0) || 0));
    if (this.ready && this.configuredEvalCacheEntries === evalCacheEntries) return this.backend;
    if (this.ready && this.configuredEvalCacheEntries !== evalCacheEntries) this.dispose();
    if (this.initPromise) return this.initPromise;
    this.initPromise = (async () => {
      if (!this.worker) {
        this.worker = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module' });
        this.worker.addEventListener('message', (event: MessageEvent) => {
          const message = event.data as { id: number; type: string; error?: string; loadedBytes?: number; totalBytes?: number };
          if (message.type === 'downloadProgress') {
            this.onDownloadProgress?.(message.loadedBytes ?? 0, message.totalBytes);
            return;
          }
          const pending = this.pending.get(message.id);
          if (!pending) return;
          this.pending.delete(message.id);
          if (message.type === 'error') pending.reject(new Error(message.error ?? 'Lc0 BT4 worker error'));
          else pending.resolve(message);
        });
        this.worker.addEventListener('error', (event) => {
          for (const pending of this.pending.values()) pending.reject(new Error(event.message || 'Lc0 BT4 worker error'));
          this.pending.clear();
        });
      }
      // WebGPU-first with wasm-CPU fallback: 'auto' resolves to
      // ['webgpu','wasm'] when an adapter is usable and ['wasm'] otherwise,
      // so big nets stay playable (slowly) without WebGPU — callers read
      // `backend` and scale visit budgets via config.wasmLevels. An explicit
      // ?ortEp=wasm page override forces the CPU path end to end.
      const ep = requestedOrtExecutionProvider() === 'wasm' ? 'wasm' : 'auto';
      const ready = await this.post<{ backend: string }>({ type: 'init', modelUrl: this.config.modelUrl, ep, cacheModel: false, evalCacheEntries, reportDownloadProgress: true });
      this.ready = true;
      this.configuredEvalCacheEntries = evalCacheEntries;
      this.backend = ready.backend;
      return this.backend;
    })();
    try {
      return await this.initPromise;
    } catch (error) {
      // Allow a later retry after a failed init.
      this.initPromise = null;
      throw error;
    }
  }

  async search(input: Lc0EvaluatorInput, options: Bt4SearchOptions): Promise<Bt4SearchResult> {
    await this.init({ evalCacheEntries: options.evalCacheEntries });
    const response = await this.post<{ result: Bt4SearchResult }>(
      {
        type: 'search',
        input,
        visits: options.visits,
        movetimeMs: options.movetimeMs,
        multiPv: options.multiPv,
        batchSize: Math.max(1, Math.floor(Number(options.batchSize ?? this.config.recommendedBatchSize) || this.config.recommendedBatchSize)),
        batchPipelineDepth: Math.max(1, Math.floor(Number(options.batchPipelineDepth ?? this.config.recommendedPipelineDepth) || this.config.recommendedPipelineDepth)),
        reuseTree: options.reuseTree,
        drawScore: options.drawScore,
        contemptElo: options.contemptElo,
        cpuct: options.cpuct,
        searchContemptLimit: options.searchContemptLimit,
      },
      (id) => { this.activeSearchId = id; },
    );
    return response.result;
  }

  async resetTree(): Promise<void> {
    if (!this.ready) return;
    await this.post({ type: 'resetSearch' });
  }

  cancel(): void {
    if (this.activeSearchId !== null && this.worker) this.worker.postMessage({ type: 'cancel', target: this.activeSearchId });
  }

  /** Terminate the worker and free the resident BT4 net. */
  dispose(): void {
    this.cancel();
    this.worker?.terminate();
    this.worker = null;
    this.ready = false;
    this.initPromise = null;
    this.configuredEvalCacheEntries = -1;
    for (const pending of this.pending.values()) pending.reject(new Error('Lc0 BT4 worker disposed'));
    this.pending.clear();
    this.activeSearchId = null;
  }
}
