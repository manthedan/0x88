// Lc0 BT4 (1024x15x32h attention net, ~353MB f16) browser integration.
//
// BT4 is large and attention-heavy, so it is handled with care:
//   - WebGPU-gated: callers must check `probeBt4Support()` before exposing it.
//   - Lazy: the dedicated search worker is only created/initialized on first use
//     (the first init fetches ~353MB, cached afterwards by the Cache API).
//   - Off the main thread: search runs entirely inside searchWorker.ts so heavy
//     inference never blocks the UI.
//   - Disposable: `dispose()` terminates the worker, freeing model memory.
import { collectOrtRuntimeDiagnostics } from '../nn/ortRuntime.ts';
import type { Lc0EvaluatorInput } from './onnxEvaluator.ts';

export const BT4_MODEL_URL = '/models/lc0/BT4-1024x15x32h-swa-6147500.batch1.f16.onnx';
export const BT4_APPROX_MB = 353;

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
}

export type Bt4AssetStatus = 'unknown' | 'present' | 'missing';

let supportProbe: Promise<boolean> | null = null;
let supportedCached: boolean | null = null;
let assetProbe: Promise<Bt4AssetStatus> | null = null;
let assetStatus: Bt4AssetStatus = 'unknown';

/** WebGPU usable for BT4? Cached after the first probe. */
export async function probeBt4Support(): Promise<boolean> {
  if (supportProbe) return supportProbe;
  supportProbe = (async () => {
    try {
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

/** Last probed local BT4 model asset result without re-probing. */
export function bt4AssetStatusSync(): Bt4AssetStatus {
  return assetStatus;
}

/** Browser-served BT4 ONNX asset availability. Cached after the first probe. */
export async function checkBt4Asset(onStatus?: () => void): Promise<Bt4AssetStatus> {
  if (assetProbe) return assetProbe;
  assetProbe = (async () => {
    try {
      const response = await fetch(BT4_MODEL_URL, { method: 'HEAD', cache: 'no-store' });
      assetStatus = response.ok ? 'present' : 'missing';
    } catch {
      assetStatus = 'missing';
    }
    onStatus?.();
    return assetStatus;
  })();
  return assetProbe;
}

/** A memory caution string when the device reports limited RAM, else null. */
export function bt4MemoryCaution(): string | null {
  const mem = (navigator as unknown as { deviceMemory?: number }).deviceMemory;
  if (typeof mem === 'number' && mem > 0 && mem <= 4) {
    return `This device reports ~${mem}GB RAM; the Lc0 BT4 net (~${BT4_APPROX_MB}MB) may strain memory.`;
  }
  return null;
}

/** Human-readable one-time warning shown before the first BT4 load. */
export function bt4LoadWarning(): string {
  const caution = bt4MemoryCaution();
  return `Lc0 BT4 is a large net (~${BT4_APPROX_MB}MB download, cached after first load) and needs WebGPU.${caution ? ` ${caution}` : ''}`;
}

/**
 * A lazily-initialized, worker-backed Lc0 BT4 searcher. One instance owns one
 * searchWorker (one resident BT4 net). Call `dispose()` to free its memory.
 */
export class Bt4WorkerSearcher {
  private worker: Worker | null = null;
  private ready = false;
  private initPromise: Promise<string> | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  private activeSearchId: number | null = null;
  backend = '';

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
  async init(): Promise<string> {
    if (this.ready) return this.backend;
    if (this.initPromise) return this.initPromise;
    this.initPromise = (async () => {
      if (!this.worker) {
        this.worker = new Worker(new URL('./searchWorker.ts', import.meta.url), { type: 'module' });
        this.worker.addEventListener('message', (event: MessageEvent) => {
          const message = event.data as { id: number; type: string; error?: string };
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
      // BT4 is WebGPU-only by policy; never fall back to WASM for this net.
      const ready = await this.post<{ backend: string }>({ type: 'init', modelUrl: BT4_MODEL_URL, ep: 'webgpu', cacheModel: false });
      this.ready = true;
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
    await this.init();
    const response = await this.post<{ result: Bt4SearchResult }>(
      {
        type: 'search',
        input,
        visits: options.visits,
        movetimeMs: options.movetimeMs,
        multiPv: options.multiPv,
        batchSize: 1,
        reuseTree: options.reuseTree,
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
    for (const pending of this.pending.values()) pending.reject(new Error('Lc0 BT4 worker disposed'));
    this.pending.clear();
    this.activeSearchId = null;
  }
}
