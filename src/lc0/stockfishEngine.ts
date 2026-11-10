import { isTrustedExecutableAssetUrl, resolvePublicAssetUrl } from './assetUrls.ts';
import {
  type BrowserUciAnalysisOptions,
  type BrowserUciEngine,
  type BrowserUciInfoLine,
  type BrowserUciRuntimeStatus,
  type BrowserUciSearchOptions,
  uciPositionCommand,
} from './browserUciEngine.ts';
import { engineArtifactWorkerScript, resolveEngineArtifactUrl } from './engineArtifacts.ts';
import { supportsWasmRelaxedSimd } from './wasmFeatures.ts';

/**
 * Stockfish "lite" opponent for the engine battle. Drives the single-threaded
 * stockfish-19-lite WASM build (served from /stockfish/) over its UCI worker
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
  /** UCI Hash (transposition table) size in MB; defaults to DEFAULT_STOCKFISH_HASH_MB. */
  hashMb?: number;
}

/**
 * Transposition-table size applied when a caller does not pass `hashMb`.
 *
 * Stockfish's own built-in default is 16 MB, which throttles anything past
 * roughly depth 12 — the analysis surface routinely searches deeper than
 * that. 64 MB is the conservative step up: every Stockfish artifact we ship
 * is built with INITIAL_MEMORY=128 MB and MAXIMUM_MEMORY=2 GB, so a 64 MB
 * table is a modest default. The full embedded network plus TT can require
 * heap growth; the lite network leaves more startup headroom. It stays small enough
 * that arena runs, which can keep several engine workers alive at once,
 * do not multiply into a mobile-browser OOM.
 */
export const DEFAULT_STOCKFISH_HASH_MB = 64;

export type StockfishFlavor = 'lite-single' | 'single' | 'lite-threaded' | 'threaded';

export const DEFAULT_STOCKFISH_FLAVOR: StockfishFlavor = 'lite-single';
export const STOCKFISH_LITE_SINGLE_URL = resolvePublicAssetUrl('/stockfish/stockfish-19-lite-single.js');
export const STOCKFISH_LITE_SINGLE_RELAXED_URL = resolvePublicAssetUrl('/stockfish/stockfish-19-lite-single-relaxed.js');
// The production app shell intentionally omits this large pair: v0 resolves both
// the JS and its sibling WASM to the configured R2 asset origin.
export const STOCKFISH_SINGLE_URL = resolvePublicAssetUrl('/stockfish/stockfish-19-single.js');
// Threaded builds must retain a same-origin worker location so their pthreads
// resolve back to the native Stockfish script instead of our cross-origin bootstrap.
export const STOCKFISH_LITE_THREADED_URL = '/stockfish/stockfish-19-lite.js';
export const STOCKFISH_THREADED_URL = '/stockfish/stockfish-19.js';

export function defaultStockfishUrl(): string {
  return supportsWasmRelaxedSimd() ? STOCKFISH_LITE_SINGLE_RELAXED_URL : STOCKFISH_LITE_SINGLE_URL;
}

export const DEFAULT_STOCKFISH_URL = defaultStockfishUrl();

export function normalizeStockfishFlavor(raw: string | null | undefined): StockfishFlavor {
  const value = String(raw ?? '')
    .toLowerCase()
    .replace(/[ _]/g, '-');
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
    case 'single':
      return 'Stockfish full single';
    case 'lite-threaded':
      return 'Stockfish lite threaded';
    case 'threaded':
      return 'Stockfish full threaded';
    default:
      return 'Stockfish lite single';
  }
}

export function stockfishFlavorUrl(flavor: StockfishFlavor): string {
  switch (flavor) {
    case 'single':
      return STOCKFISH_SINGLE_URL;
    case 'lite-threaded':
      return STOCKFISH_LITE_THREADED_URL;
    case 'threaded':
      return STOCKFISH_THREADED_URL;
    default:
      return defaultStockfishUrl();
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

function pinnedStockfishAsset(url: string): string {
  try {
    return resolveEngineArtifactUrl(url);
  } catch {
    return url;
  }
}

/** Verified glue is also the pthread entry point; children reuse the compiled
 * module and never fetch an unverified second copy of the script. */
export function stockfishWorkerUrl(jsUrl: string): { url: string; objectUrl?: string } {
  if (!sameOriginUrl(jsUrl) && !isTrustedExecutableAssetUrl(jsUrl)) throw new Error(`Refusing untrusted Stockfish worker URL: ${jsUrl}`);
  const resolvedJsUrl = new URL(jsUrl, typeof location === 'undefined' ? 'http://localhost/' : location.href).href;
  const wasmUrl = stockfishWasmUrl(resolvedJsUrl);
  const script = `
${engineArtifactWorkerScript()}
const pending = [];
self.onmessage = (event) => pending.push(event);
(async () => {
  let [glue, wasm] = await Promise.all([
    engineArtifacts.bytes(${JSON.stringify(pinnedStockfishAsset(resolvedJsUrl))}),
    engineArtifacts.bytes(${JSON.stringify(pinnedStockfishAsset(wasmUrl))}),
  ]);
  const glueUrl = URL.createObjectURL(new Blob([glue], { type: 'text/javascript' }));
  // Stockfish's facade fetches the WASM URL encoded in the worker hash.
  // Supply our verified bytes, preserving its streaming instantiate interface.
  const networkFetch = self.fetch;
  self.fetch = async (url, options) => {
    if (new URL(url, self.location.href).href !== new URL(${JSON.stringify(wasmUrl)}, self.location.href).href) return networkFetch(url, options);
    if (!wasm) throw new Error('Verified Stockfish WASM was already consumed');
    const response = new Response(wasm, { headers: { 'Content-Type': 'application/wasm' } });
    wasm = null;
    return response;
  };
  glue = null;
  const NativeWorker = self.Worker;
  self.Worker = class extends NativeWorker {
    constructor(url, options) { super(glueUrl, options); }
  };
  self.onmessage = null;
  importScripts(glueUrl);
  for (const event of pending) self.onmessage(event);
})().catch((error) => { setTimeout(() => { throw error; }); });
`;
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

// Stockfish caps Hash at 2048 MB on 32-bit targets (wasm32), and silently
// ignores a spin value outside an option's declared range, so clamp here.
function hashCommand(hashMb: number): string {
  return `setoption name Hash value ${Math.max(1, Math.min(2048, Math.floor(hashMb)))}`;
}

export interface StockfishInfoLine extends BrowserUciInfoLine {}

const STOCKFISH_ABORT_DRAIN_TIMEOUT_MS = 1500;
// SF19's async instantiateWasm hook can reject without a worker error event.
// Bound both UCI handshakes, allowing time to download the large full network.
export const STOCKFISH_INIT_TIMEOUT_MS = 120_000;

/** Parse a UCI `info ... multipv K ... score ... pv ...` line, or null if it lacks a PV. */
export function parseStockfishInfo(line: string): StockfishInfoLine | null {
  if (!line.startsWith('info ') || !line.includes(' pv ')) return null;
  const multipv = Number(line.match(/\bmultipv (\d+)/)?.[1] ?? '1');
  const depth = Number(line.match(/\bdepth (\d+)/)?.[1] ?? '0');
  const cp = line.match(/\bscore cp (-?\d+)/);
  const mate = line.match(/\bscore mate (-?\d+)/);
  const nodes = line.match(/\bnodes (\d+)/);
  const nps = line.match(/\bnps (\d+)/);
  const pv =
    line
      .match(/ pv (.+)$/)?.[1]
      .trim()
      .split(/\s+/) ?? [];
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
  private rejectInit: ((error: Error) => void) | null = null;
  private analyzeLines: Map<number, StockfishInfoLine> | null = null;
  private lastInfoLines: StockfishInfoLine[] = [];
  private queueTail: Promise<void> = Promise.resolve();
  private disposed = false;
  private options: StockfishOptions;
  private readonly url: string;
  private workerObjectUrl: string | null = null;
  private initTimer: ReturnType<typeof setTimeout> | null = null;
  private appliedOptions = new Map<string, string>();

  constructor(options: StockfishOptions = {}, url: string = defaultStockfishUrl()) {
    this.options = options;
    this.url = url;
  }

  setOptions(next: StockfishOptions): void {
    this.options = { ...this.options, ...next };
    if (this.worker && next.skillLevel !== undefined) this.worker.postMessage(skillLevelCommand(next.skillLevel));
    if (next.threads !== undefined) this.sendOption('Threads', threadsCommand(next.threads));
    if (next.hashMb !== undefined) this.sendOption('Hash', hashCommand(next.hashMb));
  }

  private sendOption(name: string, command: string): void {
    if (!this.worker || this.appliedOptions.get(name) === command) return;
    // Stockfish reallocates/clears the TT on Hash and rebuilds the pool on
    // Threads, including equal values. These settings survive ucinewgame.
    this.worker.postMessage(command);
    this.appliedOptions.set(name, command);
  }

  private applyOptions(): void {
    if (this.options.skillLevel !== undefined) this.worker?.postMessage(skillLevelCommand(this.options.skillLevel));
    if (this.options.threads !== undefined) this.sendOption('Threads', threadsCommand(this.options.threads));
    // Every worker must receive the surface budget (or our default), rather
    // than keep its 16 MB built-in default. Send after Threads because changing
    // the thread count also reallocates the TT.
    this.sendOption('Hash', hashCommand(this.options.hashMb ?? DEFAULT_STOCKFISH_HASH_MB));
  }

  private async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.queueTail;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.queueTail = previous.then(
      () => gate,
      () => gate,
    );
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

  private clearInitTimer(): void {
    if (this.initTimer !== null) clearTimeout(this.initTimer);
    this.initTimer = null;
  }

  private failActive(error: Error): void {
    this.clearInitTimer();
    const rejectMove = this.rejectMove;
    const rejectAnalyze = this.rejectAnalyze;
    const rejectReady = this.rejectReady;
    const rejectInit = this.rejectInit;
    this.resolveMove = null;
    this.rejectMove = null;
    this.resolveAnalyze = null;
    this.rejectAnalyze = null;
    this.resolveReady = null;
    this.rejectReady = null;
    this.rejectInit = null;
    this.analyzeLines = null;
    this.lastInfoLines = [];
    this.worker?.terminate();
    this.worker = null;
    this.appliedOptions.clear();
    this.revokeWorkerObjectUrl();
    this.readyPromise = null;
    rejectMove?.(error);
    rejectAnalyze?.(error);
    rejectReady?.(error);
    rejectInit?.(error);
  }

  private init(): Promise<void> {
    if (this.disposed) return Promise.reject(abortError());
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = new Promise<void>((resolve, reject) => {
      this.rejectInit = reject;
      this.initTimer = setTimeout(() => {
        this.failActive(new Error('Stockfish initialization timed out'));
      }, STOCKFISH_INIT_TIMEOUT_MS);
      try {
        const workerUrl = stockfishWorkerUrl(this.url);
        this.workerObjectUrl = workerUrl.objectUrl ?? null;
        const worker = new Worker(workerUrl.url);
        this.worker = worker;
        worker.onmessage = (event: MessageEvent) => {
          const line = typeof event.data === 'string' ? event.data : String(event.data);
          if (line === 'uciok') {
            this.applyOptions();
            worker.postMessage('isready');
            return;
          }
          if (line === 'readyok') {
            if (this.resolveReady) {
              const resolveReady = this.resolveReady;
              this.resolveReady = null;
              this.rejectReady = null;
              resolveReady();
              return;
            }
            this.clearInitTimer();
            this.rejectInit = null;
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
          // The failure is propagated through the engine promises below; stop
          // the browser from also reporting the same handled worker event as
          // an uncaught global error.
          event.preventDefault?.();
          const error = new Error(event.message || 'Stockfish worker error');
          this.failActive(error);
          reject(error);
        };
        worker.postMessage('uci');
      } catch (error) {
        this.failActive(error as Error);
        reject(error as Error);
      }
    }).catch((error) => {
      // A synchronous constructor failure occurs before readyPromise is assigned.
      // Clear its rejected cache here as well, so the next request can retry.
      this.readyPromise = null;
      throw error;
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
    if (!/stockfish-(?:18|19)(?:-lite)?(?:-relaxed)?\.js(?:[?#].*)?$/.test(this.url)) return 1;
    if (typeof location === 'undefined') return /^[/.]/.test(this.url) ? 32 : 1;
    return sameOriginUrl(this.url) || isTrustedExecutableAssetUrl(this.url) ? 32 : 1;
  }

  /** Last parsed UCI info/PV lines from `bestMove` or `analyze`, sorted by MultiPV rank. */
  lastInfo(): StockfishInfoLine[] {
    return this.lastInfoLines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  /** Best move for a FEN. Aborting sends `stop`, so Stockfish returns its current best. */
  async search(fen: string, signal?: AbortSignal, opts?: BrowserUciSearchOptions): Promise<string | null> {
    return this.bestMove(fen, signal, opts);
  }

  async bestMove(fen: string, signal?: AbortSignal, opts: BrowserUciSearchOptions = {}): Promise<string | null> {
    return this.runExclusive(async () => {
      const position = uciPositionCommand(fen, opts.moves);
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
        this.worker!.postMessage(position);
        this.worker!.postMessage(stockfishGoCommand(this.options));
      });
    });
  }

  /** MultiPV analysis of a FEN: returns one info line per PV, sorted by rank. */
  async analyze(fen: string, opts: BrowserUciAnalysisOptions = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      await this.init();
      if (!this.worker || opts.signal?.aborted) return [];
      const position = uciPositionCommand(fen, opts.moves);
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
        this.worker!.postMessage(position);
        this.worker!.postMessage(stockfishGoCommand({ depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs }));
      });
    });
  }

  dispose(): void {
    this.disposed = true;
    // Reject active and queued work before terminating the worker so callers
    // cannot retain the engine queue or recreate a worker after teardown.
    this.failActive(abortError());
  }
}
