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
}

export const DEFAULT_STOCKFISH_URL = '/stockfish/stockfish-18-lite-single.js';

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

export class StockfishEngine {
  readonly name = 'stockfish-lite';
  private worker: Worker | null = null;
  private readyPromise: Promise<void> | null = null;
  private resolveMove: ((uci: string | null) => void) | null = null;
  private options: StockfishOptions;
  private readonly url: string;

  constructor(options: StockfishOptions = {}, url: string = DEFAULT_STOCKFISH_URL) {
    this.options = options;
    this.url = url;
  }

  setOptions(next: StockfishOptions): void {
    this.options = { ...this.options, ...next };
    if (this.worker && next.skillLevel !== undefined) this.worker.postMessage(skillLevelCommand(next.skillLevel));
  }

  private applyOptions(): void {
    if (this.options.skillLevel !== undefined) this.worker?.postMessage(skillLevelCommand(this.options.skillLevel));
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
          if (line.startsWith('bestmove') && this.resolveMove) {
            const resolveMove = this.resolveMove;
            this.resolveMove = null;
            resolveMove(parseBestMove(line));
          }
        };
        worker.onerror = (event) => reject(new Error(event.message || 'Stockfish worker error'));
        worker.postMessage('uci');
      } catch (error) {
        reject(error as Error);
      }
    });
    return this.readyPromise;
  }

  /** Best move for a FEN. Aborting sends `stop`, so Stockfish returns its current best. */
  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    await this.init();
    if (!this.worker || signal?.aborted) return null;
    return new Promise<string | null>((resolve) => {
      const onAbort = () => this.worker?.postMessage('stop');
      // Clean up the abort listener whether the move completes or is aborted, so
      // listeners don't accumulate on the battle-lifetime signal (one per ply).
      this.resolveMove = (uci) => {
        signal?.removeEventListener('abort', onAbort);
        resolve(uci);
      };
      signal?.addEventListener('abort', onAbort);
      this.worker!.postMessage(`position fen ${fen}`);
      this.worker!.postMessage(stockfishGoCommand(this.options));
    });
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.readyPromise = null;
    this.resolveMove = null;
  }
}
