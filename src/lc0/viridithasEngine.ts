import { parseBestMove, parseStockfishInfo, type StockfishInfoLine } from './stockfishEngine.ts';
import type { RecklessOptions } from './recklessEngine.ts';

export const DEFAULT_VIRIDITHAS_WASM_URL = '/viridithas/viridithas.wasm';

interface RunResult {
  stdout: string[];
  stderr: string[];
  exitCode: number;
}

type Pending = { resolve: (result: RunResult) => void; reject: (error: Error) => void };

function abortError(): Error {
  const error = new Error('Viridithas search aborted');
  error.name = 'AbortError';
  return error;
}

function goCommand(options: RecklessOptions): string {
  if (options.movetimeMs && options.movetimeMs > 0) return `go movetime ${Math.floor(options.movetimeMs)}`;
  return `go depth ${Math.max(1, Math.floor(options.depth ?? 4))}`;
}

function hashCommand(hashMb: number): string {
  return `setoption name Hash value ${Math.max(1, Math.min(1024, Math.floor(hashMb)))}`;
}

/** Experimental one-shot WASI adapter for patched Viridithas. */
export class ViridithasEngine {
  readonly name = 'viridithas-wasi';
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private queueTail: Promise<void> = Promise.resolve();
  private lastInfoLines: StockfishInfoLine[] = [];

  constructor(private options: RecklessOptions = {}, private readonly wasmUrl = DEFAULT_VIRIDITHAS_WASM_URL) {}

  setOptions(next: RecklessOptions): void {
    this.options = { ...this.options, ...next };
  }

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(new URL('./recklessWasiWorker.ts', import.meta.url), { type: 'module', name: 'viridithas-wasi' });
    worker.onmessage = (event: MessageEvent) => {
      const message = event.data as { type: string; id: number; stdout?: string[]; stderr?: string[]; exitCode?: number; error?: string };
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.type === 'error') pending.reject(new Error(message.error ?? 'Viridithas WASI worker error'));
      else pending.resolve({ stdout: message.stdout ?? [], stderr: message.stderr ?? [], exitCode: message.exitCode ?? 0 });
    };
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Viridithas WASI worker error'));
    this.worker = worker;
    return worker;
  }

  private rejectAllAndDispose(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    this.disposeWorker();
  }

  private disposeWorker(): void {
    this.worker?.terminate();
    this.worker = null;
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

  private runCommands(commands: string[], signal?: AbortSignal): Promise<RunResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    const id = ++this.seq;
    const worker = this.ensureWorker();
    return new Promise<RunResult>((resolve, reject) => {
      const onAbort = () => {
        this.pending.delete(id);
        this.disposeWorker();
        reject(abortError());
      };
      this.pending.set(id, {
        resolve: (result) => {
          signal?.removeEventListener('abort', onAbort);
          resolve(result);
        },
        reject: (error) => {
          signal?.removeEventListener('abort', onAbort);
          reject(error);
        },
      });
      signal?.addEventListener('abort', onAbort, { once: true });
      worker.postMessage({ type: 'run', id, wasmUrl: this.wasmUrl, executableName: 'viridithas', commands });
    });
  }

  private searchCommands(fen: string, options: RecklessOptions, multipv = 1): string[] {
    return [
      'uci',
      'isready',
      hashCommand(options.hashMb ?? this.options.hashMb ?? 16),
      'setoption name Threads value 1',
      `setoption name MultiPV value ${Math.max(1, Math.floor(multipv))}`,
      `position fen ${fen}`,
      goCommand(options),
    ];
  }

  private parseInfo(stdout: string[]): StockfishInfoLine[] {
    const latest = new Map<number, StockfishInfoLine>();
    for (const line of stdout) {
      const info = parseStockfishInfo(line);
      if (info) latest.set(info.multipv, info);
    }
    return [...latest.values()].sort((a, b) => a.multipv - b.multipv);
  }

  lastInfo(): StockfishInfoLine[] {
    return this.lastInfoLines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  runtimeLabel(): string {
    return 'one-shot';
  }

  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      const result = await this.runCommands(this.searchCommands(fen, this.options, 1), signal);
      if (result.exitCode !== 0) throw new Error(`Viridithas exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return parseBestMove(result.stdout.find((line) => line.startsWith('bestmove')) ?? '');
    });
  }

  async analyze(fen: string, opts: { multipv?: number; depth?: number; movetimeMs?: number; signal?: AbortSignal } = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      const result = await this.runCommands(this.searchCommands(fen, { ...this.options, depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs ?? this.options.movetimeMs }, opts.multipv ?? 1), opts.signal);
      if (result.exitCode !== 0) throw new Error(`Viridithas exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return this.lastInfo();
    });
  }

  dispose(): void {
    for (const pending of this.pending.values()) pending.reject(abortError());
    this.pending.clear();
    this.disposeWorker();
    this.lastInfoLines = [];
  }
}
