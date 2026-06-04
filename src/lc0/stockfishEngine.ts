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
export const DEFAULT_STOCKFISH_URL = '/stockfish/stockfish-18-lite-single.js';

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
    case 'single': return '/stockfish/stockfish-18-single.js';
    case 'lite-threaded': return '/stockfish/stockfish-18-lite.js';
    case 'threaded': return '/stockfish/stockfish-18.js';
    default: return DEFAULT_STOCKFISH_URL;
  }
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

export interface StockfishInfoLine {
  multipv: number;
  depth: number;
  scoreCp?: number;
  mateIn?: number;
  pvUci: string[];
}

/** Parse a UCI `info ... multipv K ... score ... pv ...` line, or null if it lacks a PV. */
export function parseStockfishInfo(line: string): StockfishInfoLine | null {
  if (!line.startsWith('info ') || !line.includes(' pv ')) return null;
  const multipv = Number(line.match(/\bmultipv (\d+)/)?.[1] ?? '1');
  const depth = Number(line.match(/\bdepth (\d+)/)?.[1] ?? '0');
  const cp = line.match(/\bscore cp (-?\d+)/);
  const mate = line.match(/\bscore mate (-?\d+)/);
  const pv = line.match(/ pv (.+)$/)?.[1].trim().split(/\s+/) ?? [];
  return {
    multipv,
    depth,
    scoreCp: cp ? Number(cp[1]) : undefined,
    mateIn: mate ? Number(mate[1]) : undefined,
    pvUci: pv,
  };
}

export class StockfishEngine {
  readonly name = 'stockfish-lite';
  private worker: Worker | null = null;
  private readyPromise: Promise<void> | null = null;
  private resolveMove: ((uci: string | null) => void) | null = null;
  private rejectMove: ((error: Error) => void) | null = null;
  private resolveAnalyze: ((lines: StockfishInfoLine[]) => void) | null = null;
  private rejectAnalyze: ((error: Error) => void) | null = null;
  private analyzeLines: Map<number, StockfishInfoLine> | null = null;
  private queueTail: Promise<void> = Promise.resolve();
  private options: StockfishOptions;
  private readonly url: string;

  constructor(options: StockfishOptions = {}, url: string = DEFAULT_STOCKFISH_URL) {
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

  private failActive(error: Error): void {
    const rejectMove = this.rejectMove;
    const rejectAnalyze = this.rejectAnalyze;
    this.resolveMove = null;
    this.rejectMove = null;
    this.resolveAnalyze = null;
    this.rejectAnalyze = null;
    this.analyzeLines = null;
    this.worker?.terminate();
    this.worker = null;
    this.readyPromise = null;
    rejectMove?.(error);
    rejectAnalyze?.(error);
  }

  private init(): Promise<void> {
    if (this.readyPromise) return this.readyPromise;
    this.readyPromise = new Promise<void>((resolve, reject) => {
      try {
        const worker = new Worker(this.url);
        this.worker = worker;
        worker.onmessage = (event: MessageEvent) => {
          const line = typeof event.data === 'string' ? event.data : String(event.data);
          if (line === 'uciok') { this.applyOptions(); worker.postMessage('isready'); return; }
          if (line === 'readyok') { resolve(); return; }
          if (line.startsWith('info ') && this.analyzeLines) {
            const info = parseStockfishInfo(line);
            if (info) this.analyzeLines.set(info.multipv, info);
            return;
          }
          if (line.startsWith('bestmove')) {
            if (this.resolveAnalyze) {
              const resolveAnalyze = this.resolveAnalyze;
              const lines = [...(this.analyzeLines?.values() ?? [])].sort((a, b) => a.multipv - b.multipv);
              this.resolveAnalyze = null;
              this.rejectAnalyze = null;
              this.analyzeLines = null;
              resolveAnalyze(lines);
              return;
            }
            if (this.resolveMove) {
              const resolveMove = this.resolveMove;
              this.resolveMove = null;
              this.rejectMove = null;
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
        reject(error as Error);
      }
    });
    return this.readyPromise;
  }

  /** Best move for a FEN. Aborting sends `stop`, so Stockfish returns its current best. */
  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      await this.init();
      if (!this.worker || signal?.aborted) return null;
      return new Promise<string | null>((resolve, reject) => {
        const onAbort = () => this.worker?.postMessage('stop');
        // Clean up the abort listener whether the move completes or is aborted, so
        // listeners don't accumulate on the battle-lifetime signal (one per ply).
        this.resolveMove = (uci) => {
          signal?.removeEventListener('abort', onAbort);
          this.rejectMove = null;
          resolve(uci);
        };
        this.rejectMove = (error) => {
          signal?.removeEventListener('abort', onAbort);
          reject(error);
        };
        signal?.addEventListener('abort', onAbort);
        this.worker!.postMessage(`position fen ${fen}`);
        this.worker!.postMessage(stockfishGoCommand(this.options));
      });
    });
  }

  /** MultiPV analysis of a FEN: returns one info line per PV, sorted by rank. */
  async analyze(fen: string, opts: { multipv?: number; depth?: number; movetimeMs?: number; signal?: AbortSignal } = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      await this.init();
      if (!this.worker || opts.signal?.aborted) return [];
      const multipv = Math.max(1, Math.floor(opts.multipv ?? 1));
      this.worker.postMessage(`setoption name MultiPV value ${multipv}`);
      return new Promise<StockfishInfoLine[]>((resolve, reject) => {
        this.analyzeLines = new Map();
        const onAbort = () => this.worker?.postMessage('stop');
        this.resolveAnalyze = (lines) => {
          opts.signal?.removeEventListener('abort', onAbort);
          this.rejectAnalyze = null;
          resolve(lines);
        };
        this.rejectAnalyze = (error) => {
          opts.signal?.removeEventListener('abort', onAbort);
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
    this.readyPromise = null;
    this.resolveMove = null;
    this.rejectMove = null;
    this.resolveAnalyze = null;
    this.rejectAnalyze = null;
    this.analyzeLines = null;
  }
}
