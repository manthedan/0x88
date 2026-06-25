import { isTrustedExecutableAssetUrl, resolvePublicAssetUrl } from './assetUrls.ts';
import type { BrowserUciAnalysisOptions, BrowserUciEngine, BrowserUciInfoLine, BrowserUciRuntimeStatus } from './browserUciEngine.ts';
import { supportsWasmRelaxedSimd } from './wasmFeatures.ts';

/**
 * Stockfish "lite" opponent for the engine battle. Drives the single-threaded
 * stockfish-18-lite WASM build (served from /stockfish/) over its UCI worker
 * protocol and exposes a move provider compatible with engineBattle.
 *
 * Strength is deliberately tunable (depth / Skill Level) so the small LC0 net
 * has a competitive opponent rather than full-strength Stockfish.
 */

export interface StockfishOptions {
  /** Fixed search depth (default). Lower = weaker / faster. */
  depth?: number;
  /** Alternative to depth: fixed think time in ms. */
  movetimeMs?: number;
  /** UCI Skill Level 0-20; lower handicaps Stockfish with weaker/blundering play. */
  skillLevel?: number;
  /** UCI Threads. Threaded WASM builds require cross-origin isolation. */
  threads?: number;
}

export type StockfishFlavor = 'lite-single' | 'single' | 'lite-threaded' | 'threaded';

export const DEFAULT_STOCKFISH_FLAVOR: StockfishFlavor = 'lite-single';
export const STOCKFISH_LITE_SINGLE_URL = resolvePublicAssetUrl('/stockfish/stockfish-18-lite-single.js');
export const STOCKFISH_LITE_SINGLE_RELAXED_URL = resolvePublicAssetUrl('/stockfish/stockfish-18-lite-single-relaxed.js');

export function defaultStockfishUrl(): string {
  return supportsWasmRelaxedSimd() ? STOCKFISH_LITE_SINGLE_RELAXED_URL : STOCKFISH_LITE_SINGLE_URL;
}

export const DEFAULT_STOCKFISH_URL = defaultStockfishUrl();

export function normalizeStockfishFlavor(raw: string | null | undefined): StockfishFlavor {
  const value = String(raw ?? '').toLowerCase().replace(/[ _]/g, '-');
  if (value === 'single' || value === 'full-single' || value === 'full') return 'single';
  if (value === 'lite-threaded' || value === 'lite-threads' || value === 'lite-multi') return 'lite-threaded';
  if (value === 'threaded' || value === 'full-threaded' || value === 'threads' || value === 'multi') return 'threaded';
  return 'lite-single';
}

export function stockfishFlavorRequiresIsolation(flavor: StockfishFlavor): boolean {
  return flavor === 'lite-threaded' || flavor === 'threaded';
}

export function stockfishFlavorLabel(flavor: StockfishFlavor): string {
  switch (flavor) {
    case 'single': return 'Stockfish full single';
    case 'lite-threaded': return 'Stockfish lite threaded';
    case 'threaded': return 'Stockfish full threaded';
    default: return 'Stockfish lite single';
  }
}

export function stockfishFlavorUrl(flavor: StockfishFlavor): string {
  switch (flavor) {
    case 'single': return resolvePublicAssetUrl('/stockfish/stockfish-18-single.js');
    case 'lite-threaded': {
      const url = resolvePublicAssetUrl('/stockfish/stockfish-18-lite.js');
      // The pthread builds derive helper-worker URLs from self.location. That
      // is incompatible with the cross-origin blob wrapper used for R2-hosted
      // Stockfish scripts, so hosted builds fall back to the single-threaded
      // artifact until a dedicated pthread wrapper is promoted.
      return sameOriginUrl(url) ? url : defaultStockfishUrl();
    }
    case 'threaded': {
      const url = resolvePublicAssetUrl('/stockfish/stockfish-18.js');
      return sameOriginUrl(url) ? url : defaultStockfishUrl();
    }
    default: return defaultStockfishUrl();
  }
}

function sameOriginUrl(raw: string): boolean {
  try {
    if (typeof location === 'undefined') return true;
    return new URL(raw, location.href).origin === location.origin;
  } catch {
    return false;
  }
}

function stockfishWasmUrl(jsUrl: string): string {
  return jsUrl.replace(/\.js(?:[?#].*)?$/, '.wasm');
}

export function stockfishWorkerUrl(jsUrl: string): { url: string; objectUrl?: string } {
  if (sameOriginUrl(jsUrl)) return { url: jsUrl };
  if (!isTrustedExecutableAssetUrl(jsUrl)) throw new Error(`Refusing untrusted Stockfish worker URL: ${jsUrl}`);
  const wasmUrl = stockfishWasmUrl(jsUrl);
  // nmrugg/stockfish.js reads self.location.hash as "<wasm-url>" for the UCI
  // worker. The ",worker" suffix is reserved for pthread helper workers and
  // makes the top-level UCI worker skip its onmessage initialization.
  const script = `importScripts(${JSON.stringify(jsUrl)});`;
  const objectUrl = URL.createObjectURL(new Blob([script], { type: 'text/javascript' }));
  return { url: `${objectUrl}#${encodeURIComponent(wasmUrl)}`, objectUrl };
}

/** Parse a UCI `bestmove` line into a UCI move, or null for `(none)`/no match. */
export function parseBestMove(line: string): string | null {
  const match = line.match(/^bestmove\s+(\S+)/);
  if (!match) return null;
  return match[1] === '(none)' ? null : match[1];
}

/** Build the UCI `go` command for the configured limit (movetime wins over depth). */
export function stockfishGoCommand(options: StockfishOptions): string {
  if (options.movetimeMs && options.movetimeMs > 0) return `go movetime ${Math.floor(options.movetimeMs)}`;
  return `go depth ${Math.max(1, Math.floor(options.depth ?? 4))}`;
}

function skillLevelCommand(skillLevel: number): string {
  return `setoption name Skill Level value ${Math.max(0, Math.min(20, Math.floor(skillLevel)))}`;
}

function threadsCommand(threads: number): string {
  return `setoption name Threads value ${Math.max(1, Math.min(32, Math.floor(threads)))}`;
}

export interface StockfishInfoLine extends BrowserUciInfoLine {}

const STOCKFISH_ABORT_DRAIN_TIMEOUT_MS = 1500;

/** Parse a UCI `info ... multipv K ... score ... pv ...` line, or null if it lacks a PV. */
export function parseStockfishInfo(line: string): StockfishInfoLine | null {
  if (!line.startsWith('info ') || !line.includes(' pv ')) return null;
  const multipv = Number(line.match(/\bmultipv (\d+)/)?.[1] ?? '1');
  const depth = Number(line.match(/\bdepth (\d+)/)?.[1] ?? '0');
  const cp = line.match(/\bscore cp (-?\d+)/);
  const mate = line.match(/\bscore mate (-?\d+)/);
  const nodes = line.match(/\bnodes (\d+)/);
  const nps = line.match(/\bnps (\d+)/);
  const pv = line.match(/ pv (.+)$/)?.[1].trim().split(/\s+/) ?? [];
  return {
    multipv,
    depth,
    scoreCp: cp ? Number(cp[1]) : undefined,
    mateIn: mate ? Number(mate[1]) : undefined,
    ...(nodes ? { nodes: Number(nodes[1]) } : {}),
    ...(nps ? { nps: Number(nps[1]) } : {}),
    pvUci: pv,
  };
}

function abortError(): Error {
  const error = new Error('Stockfish search aborted');
  error.name = 'AbortError';
  return error;
}

export class StockfishEngine implements BrowserUciEngine {
  readonly name = 'stockfish-lite';
  private worker: Worker | null = null;
  private readyPromise: Promise<void> | null = null;
  private resolveMove: ((uci: string | null) => void) | null = null;
  private rejectMove: ((error: Error) => void) | null = null;
  private resolveAnalyze: ((lines: StockfishInfoLine[]) => void) | null = null;
  private rejectAnalyze: ((error: Error) => void) | null = null;
  private resolveReady: (() => void) | null = null;
  private rejectReady: ((error: Error) => void) | null = null;
  private analyzeLines: Map<number, StockfishInfoLine> | null = null;
  private lastInfoLines: StockfishInfoLine[] = [];
  private queueTail: Promise<void> = Promise.resolve();
  private options: StockfishOptions;
  private readonly url: string;
  private workerObjectUrl: string | null = null;

  constructor(options: StockfishOptions = {}, url: string = defaultStockfishUrl()) {
    this.options = options;
    this.url = url;
  }

  setOptions(next: StockfishOptions): void {
    this.options = { ...this.options, ...next };
    if (this.worker && next.skillLevel !== undefined) this.worker.postMessage(skillLevelCommand(next.skillLevel));
    if (this.worker && next.threads !== undefined) this.worker.postMessage(threadsCommand(next.threads));
  }

  private applyOptions(): void {
    if (this.options.skillLevel !== undefined) this.worker?.postMessage(skillLevelCommand(this.options.skillLevel));
    if (this.options.threads !== undefined) this.worker?.postMessage(threadsCommand(this.options.threads));
  }

  private async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.queueTail;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    this.queueTail = previous.then(() => gate, () => gate);
    await previous.catch(() => undefined);
    try {
      return await fn();
    } finally {
      release();
    }
  }

  private revokeWorkerObjectUrl(): void {
    if (!this.workerObjectUrl) return;
    URL.revokeObjectURL(this.workerObjectUrl);
    this.workerObjectUrl = null;
  }

  private failActive(error: Error): void {
    const rejectMove = this.rejectMove;
    const rejectAnalyze = this.rejectAnalyze;
    const rejectReady = this.rejectReady;
    this.resolveMove = null;
    this.rejectMove = null;
    this.resolveAnalyze = null;
    this.rejectAnalyze = null;
    this.resolveReady = null;
    this.rejectReady = null;
    this.analyzeLines = null;
    this.lastInfoLines = [];
    this.worker?.terminate();
    this.worker = null;
    this.revokeWorkerObjectUrl();
    this.readyPromise = null;
    rejectMove?.(error);
    rejectAnalyze?.(error);
    rejectReady?.(error);
  }

  private init(): Promise<void> {
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = new Promise<void>((resolve, reject) => {
      try {
        const workerUrl = stockfishWorkerUrl(this.url);
        this.workerObjectUrl = workerUrl.objectUrl ?? null;
        const worker = new Worker(workerUrl.url);
        this.worker = worker;
        worker.onmessage = (event: MessageEvent) => {
          const line = typeof event.data === 'string' ? event.data : String(event.data);
          if (line === 'uciok') { this.applyOptions(); worker.postMessage('isready'); return; }
          if (line === 'readyok') {
            if (this.resolveReady) {
              const resolveReady = this.resolveReady;
              this.resolveReady = null;
              this.rejectReady = null;
              resolveReady();
              return;
            }
            resolve();
            return;
          }
          if (line.startsWith('info ') && this.analyzeLines) {
            const info = parseStockfishInfo(line);
            if (info) this.analyzeLines.set(info.multipv, info);
            return;
          }
          if (line.startsWith('bestmove')) {
            if (this.resolveAnalyze) {
              const resolveAnalyze = this.resolveAnalyze;
              const lines = [...(this.analyzeLines?.values() ?? [])].sort((a, b) => a.multipv - b.multipv);
              this.lastInfoLines = lines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
              this.resolveAnalyze = null;
              this.rejectAnalyze = null;
              this.analyzeLines = null;
              resolveAnalyze(lines);
              return;
            }
            if (this.resolveMove) {
              const resolveMove = this.resolveMove;
              const lines = [...(this.analyzeLines?.values() ?? [])].sort((a, b) => a.multipv - b.multipv);
              this.lastInfoLines = lines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
              this.resolveMove = null;
              this.rejectMove = null;
              this.analyzeLines = null;
              resolveMove(parseBestMove(line));
            }
          }
        };
        worker.onerror = (event) => {
          const error = new Error(event.message || 'Stockfish worker error');
          this.failActive(error);
          reject(error);
        };
        worker.postMessage('uci');
      } catch (error) {
        this.revokeWorkerObjectUrl();
        reject(error as Error);
      }
    });
    return this.readyPromise;
  }

  private waitReady(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (!this.worker) return Promise.reject(new Error('Stockfish worker was not initialized'));
    return new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        this.resolveReady = null;
        this.rejectReady = null;
        reject(abortError());
      };
      this.resolveReady = () => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      };
      this.rejectReady = (error) => {
        signal?.removeEventListener('abort', onAbort);
        reject(error);
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      this.worker!.postMessage('isready');
    });
  }

  /** Start the worker and complete the UCI/isready handshake before a real search. */
  async prewarm(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      await this.init();
      if (signal?.aborted) throw abortError();
    });
  }

  /** Reset Stockfish state/hash for a fresh game and wait for `readyok`. */
  async newGame(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      await this.init();
      if (!this.worker || signal?.aborted) throw abortError();
      this.worker.postMessage('ucinewgame');
      await this.waitReady(signal);
      this.lastInfoLines = [];
    });
  }

  runtimeStatus(): BrowserUciRuntimeStatus {
    return {
      mode: this.worker ? 'worker' : 'idle',
      persistentAvailable: false,
      persistentDisabled: false,
      forceOneShot: false,
      workerUrl: this.url,
    };
  }

  runtimeLabel(): string {
    return this.worker ? 'worker ready' : 'worker idle';
  }

  maxThreads(): number {
    if (!/stockfish-18(?:-lite)?\.js(?:[?#].*)?$/.test(this.url)) return 1;
    if (typeof location === 'undefined') return /^[/.]/.test(this.url) ? 32 : 1;
    return sameOriginUrl(this.url) ? 32 : 1;
  }

  /** Last parsed UCI info/PV lines from `bestMove` or `analyze`, sorted by MultiPV rank. */
  lastInfo(): StockfishInfoLine[] {
    return this.lastInfoLines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  /** Best move for a FEN. Aborting sends `stop`, so Stockfish returns its current best. */
  async search(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.bestMove(fen, signal);
  }

  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      await this.init();
      if (!this.worker || signal?.aborted) return null;
      this.lastInfoLines = [];
      this.analyzeLines = new Map();
      this.worker.postMessage('setoption name MultiPV value 1');
      return new Promise<string | null>((resolve, reject) => {
        let abortTimer: ReturnType<typeof setTimeout> | null = null;
        const cleanup = () => {
          signal?.removeEventListener('abort', onAbort);
          if (abortTimer) clearTimeout(abortTimer);
          abortTimer = null;
        };
        const onAbort = () => {
          this.worker?.postMessage('stop');
          abortTimer = setTimeout(() => this.failActive(abortError()), STOCKFISH_ABORT_DRAIN_TIMEOUT_MS);
        };
        // Clean up the abort listener whether the move completes or is aborted, so
        // listeners don't accumulate on the battle-lifetime signal (one per ply).
        this.resolveMove = (uci) => {
          cleanup();
          this.rejectMove = null;
          resolve(uci);
        };
        this.rejectMove = (error) => {
          cleanup();
          this.analyzeLines = null;
          reject(error);
        };
        signal?.addEventListener('abort', onAbort);
        this.worker!.postMessage(`position fen ${fen}`);
        this.worker!.postMessage(stockfishGoCommand(this.options));
      });
    });
  }

  /** MultiPV analysis of a FEN: returns one info line per PV, sorted by rank. */
  async analyze(fen: string, opts: BrowserUciAnalysisOptions = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      await this.init();
      if (!this.worker || opts.signal?.aborted) return [];
      const multipv = Math.max(1, Math.floor(opts.multipv ?? 1));
      this.lastInfoLines = [];
      this.worker.postMessage(`setoption name MultiPV value ${multipv}`);
      return new Promise<StockfishInfoLine[]>((resolve, reject) => {
        this.analyzeLines = new Map();
        let abortTimer: ReturnType<typeof setTimeout> | null = null;
        const cleanup = () => {
          opts.signal?.removeEventListener('abort', onAbort);
          if (abortTimer) clearTimeout(abortTimer);
          abortTimer = null;
        };
        const onAbort = () => {
          this.worker?.postMessage('stop');
          abortTimer = setTimeout(() => this.failActive(abortError()), STOCKFISH_ABORT_DRAIN_TIMEOUT_MS);
        };
        this.resolveAnalyze = (lines) => {
          cleanup();
          this.rejectAnalyze = null;
          resolve(lines);
        };
        this.rejectAnalyze = (error) => {
          cleanup();
          this.analyzeLines = null;
          reject(error);
        };
        opts.signal?.addEventListener('abort', onAbort);
        this.worker!.postMessage(`position fen ${fen}`);
        this.worker!.postMessage(stockfishGoCommand({ depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs }));
      });
    });
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.revokeWorkerObjectUrl();
    this.readyPromise = null;
    this.resolveMove = null;
    this.rejectMove = null;
    this.resolveAnalyze = null;
    this.rejectAnalyze = null;
    this.resolveReady = null;
    this.rejectReady = null;
    this.analyzeLines = null;
    this.lastInfoLines = [];
  }
}
