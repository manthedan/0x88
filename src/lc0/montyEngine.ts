import type { BrowserUciAnalysisOptions, BrowserUciEngine, BrowserUciRuntimeStatus } from './browserUciEngine.ts';
import { parseBestMove, parseStockfishInfo, type StockfishInfoLine } from './stockfishEngine.ts';

export const DEFAULT_MONTY_WASM_URL = '/monty/monty.wasm';

/**
 * Monty's raw networks are NOT embedded in the wasm (the pair is ~950MB raw).
 * The wasm32-wasip1 build opens them by canonical name from the WASI
 * preopened cwd; the worker fetches these URLs and exposes them as in-memory
 * preopened files (see recklessWasiWorker.ts).
 */
export const MONTY_VALUE_NET_FILE = 'nn-09da29a4b6ed.network';
export const MONTY_POLICY_NET_FILE = 'nn-6e49a41bd7c0.network';
export const DEFAULT_MONTY_NET_BASE_URL = '/models/monty';

export interface MontyOptions {
  /** MCTS node budget per move (Monty may overshoot slightly between reports). */
  nodes?: number;
  /** Average-depth cap; the node budget is the more natural MCTS knob. */
  depth?: number;
  movetimeMs?: number;
  hashMb?: number;
  /**
   * Monty's calibrated contempt: the assumed Elo difference vs the opponent
   * (positive = we are stronger and press harder), validated across ±1000.
   */
  contempt?: number;
}

interface RunResult {
  stdout: string[];
  stderr: string[];
  exitCode: number;
}

type Pending = { resolve: (result: RunResult) => void; reject: (error: Error) => void };
type PersistentPending = Pending & {
  stdout: string[];
  stderr: string[];
  resolveWhenLine: (line: string, stream: 'stdout' | 'stderr') => boolean;
};

type SharedInput = {
  buffer: SharedArrayBuffer;
  control: Int32Array;
  data: Uint8Array;
  capacity: number;
};

const SHARED_STDIN_HEADER_INTS = 4;
const SHARED_STDIN_HEADER_BYTES = SHARED_STDIN_HEADER_INTS * Int32Array.BYTES_PER_ELEMENT;
const PERSISTENT_STDIN_CAPACITY_BYTES = 64 * 1024;

function abortError(): Error {
  const error = new Error('Monty search aborted');
  error.name = 'AbortError';
  return error;
}

function createSharedInput(capacity = PERSISTENT_STDIN_CAPACITY_BYTES): SharedInput {
  const buffer = new SharedArrayBuffer(SHARED_STDIN_HEADER_BYTES + capacity);
  const control = new Int32Array(buffer, 0, SHARED_STDIN_HEADER_INTS);
  const data = new Uint8Array(buffer, SHARED_STDIN_HEADER_BYTES);
  Atomics.store(control, 3, capacity);
  return { buffer, control, data, capacity };
}

function closeSharedInput(input: SharedInput): void {
  Atomics.store(input.control, 2, 1);
  Atomics.notify(input.control, 1);
}

function writeSharedInput(input: SharedInput, text: string): void {
  const bytes = new TextEncoder().encode(text);
  const readPos = Atomics.load(input.control, 0);
  let writePos = Atomics.load(input.control, 1);
  if (writePos - readPos + bytes.byteLength > input.capacity) throw new Error('Monty persistent stdin buffer is full');
  let offset = 0;
  while (offset < bytes.byteLength) {
    const ringOffset = writePos % input.capacity;
    const n = Math.min(bytes.byteLength - offset, input.capacity - ringOffset);
    input.data.set(bytes.subarray(offset, offset + n), ringOffset);
    offset += n;
    writePos += n;
  }
  Atomics.store(input.control, 1, writePos);
  Atomics.notify(input.control, 1);
}

export function canUsePersistentMontyWasi(): boolean {
  return typeof SharedArrayBuffer !== 'undefined' && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
}

/**
 * WASI adapter for the patched Monty (single-thread MCTS with calibrated
 * contempt). Persistent mode keeps the engine (and its ~1GB of loaded
 * networks plus the search tree) alive between moves; one-shot mode re-runs
 * the module per request, which re-reads the cached nets into a fresh wasm
 * instance — correct but slower, so persistent is strongly preferred.
 */
export class MontyEngine implements BrowserUciEngine {
  readonly name = 'monty-wasi';
  /** Reports detached-network download/copy progress (urls are per net file). */
  onDownloadProgress: ((url: string, loadedBytes: number, totalBytes: number) => void) | null = null;
  private worker: Worker | null = null;
  private workerMode: 'oneshot' | 'persistent' | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private persistentPending: PersistentPending | null = null;
  private sharedInput: SharedInput | null = null;
  private persistentDisabled = false;
  private persistentInitialized = false;
  private persistentHashCommand: string | null = null;
  private persistentContemptCommand: string | null = null;
  private persistentMultipvCommand: string | null = null;
  private queueTail: Promise<void> = Promise.resolve();
  private lastInfoLines: StockfishInfoLine[] = [];

  constructor(
    private options: MontyOptions = {},
    private readonly wasmUrl = DEFAULT_MONTY_WASM_URL,
    private readonly netBaseUrl = DEFAULT_MONTY_NET_BASE_URL,
  ) {}

  setOptions(next: MontyOptions): void {
    this.options = { ...this.options, ...next };
  }

  private preopenFiles(): { name: string; url: string }[] {
    return [
      { name: MONTY_VALUE_NET_FILE, url: `${this.netBaseUrl}/${MONTY_VALUE_NET_FILE}` },
      { name: MONTY_POLICY_NET_FILE, url: `${this.netBaseUrl}/${MONTY_POLICY_NET_FILE}` },
    ];
  }

  private handleProgressMessage(message: { type: string; url?: string; loadedBytes?: number; totalBytes?: number }): boolean {
    if (message.type !== 'preopen-progress') return false;
    this.onDownloadProgress?.(message.url ?? '', message.loadedBytes ?? 0, message.totalBytes ?? 0);
    return true;
  }

  private ensureOneShotWorker(): Worker {
    if (this.worker && this.workerMode === 'oneshot') return this.worker;
    this.disposeWorker();
    const worker = new Worker(new URL('./recklessWasiWorker.ts', import.meta.url), { type: 'module', name: 'monty-wasi' });
    worker.onmessage = (event: MessageEvent) => {
      const message = event.data as { type: string; id: number; stdout?: string[]; stderr?: string[]; exitCode?: number; error?: string; url?: string; loadedBytes?: number; totalBytes?: number };
      if (this.handleProgressMessage(message)) return;
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.type === 'error') pending.reject(new Error(message.error ?? 'Monty WASI worker error'));
      else pending.resolve({ stdout: message.stdout ?? [], stderr: message.stderr ?? [], exitCode: message.exitCode ?? 0 });
    };
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Monty WASI worker error'));
    this.worker = worker;
    this.workerMode = 'oneshot';
    return worker;
  }

  private ensurePersistentWorker(): Worker {
    if (this.worker && this.workerMode === 'persistent' && this.sharedInput) return this.worker;
    this.disposeWorker();
    const worker = new Worker(new URL('./recklessWasiWorker.ts', import.meta.url), { type: 'module', name: 'monty-wasi-persistent' });
    const sharedInput = createSharedInput();
    worker.onmessage = (event: MessageEvent) => this.handlePersistentMessage(event.data as { type: string; stream?: 'stdout' | 'stderr'; line?: string; exitCode?: number; error?: string; url?: string; loadedBytes?: number; totalBytes?: number });
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Monty persistent WASI worker error'));
    this.worker = worker;
    this.workerMode = 'persistent';
    this.sharedInput = sharedInput;
    worker.postMessage({ type: 'start-persistent', wasmUrl: this.wasmUrl, inputBuffer: sharedInput.buffer, executableName: 'monty', preopenFiles: this.preopenFiles() });
    return worker;
  }

  private handlePersistentMessage(message: { type: string; stream?: 'stdout' | 'stderr'; line?: string; exitCode?: number; error?: string; url?: string; loadedBytes?: number; totalBytes?: number }): void {
    if (this.handleProgressMessage(message)) return;
    if (message.type === 'persistent-ready') return;
    if (message.type === 'persistent-line') {
      const active = this.persistentPending;
      if (!active || !message.line) return;
      if (message.stream === 'stderr') active.stderr.push(message.line);
      else {
        active.stdout.push(message.line);
        if (message.line === 'uciok') this.persistentInitialized = true;
      }
      if (active.resolveWhenLine(message.line, message.stream ?? 'stdout')) this.resolvePersistent({ stdout: active.stdout, stderr: active.stderr, exitCode: 0 });
      return;
    }
    if (message.type === 'persistent-error') {
      this.rejectAllAndDispose(new Error(message.error ?? 'Monty persistent WASI worker error'));
      return;
    }
    if (message.type === 'persistent-exit') this.rejectAllAndDispose(new Error(`Monty persistent WASI process exited with ${message.exitCode ?? 0}`));
  }

  private resolvePersistent(result: RunResult): void {
    const active = this.persistentPending;
    if (!active) return;
    this.persistentPending = null;
    active.resolve(result);
  }

  private rejectAllAndDispose(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    if (this.persistentPending) {
      this.persistentPending.reject(error);
      this.persistentPending = null;
    }
    this.disposeWorker();
  }

  private disposeWorker(): void {
    if (this.sharedInput) closeSharedInput(this.sharedInput);
    this.worker?.terminate();
    this.worker = null;
    this.workerMode = null;
    this.sharedInput = null;
    this.persistentInitialized = false;
    this.persistentHashCommand = null;
    this.persistentContemptCommand = null;
    this.persistentMultipvCommand = null;
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

  private runOneShotCommands(commands: string[], signal?: AbortSignal): Promise<RunResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    const id = ++this.seq;
    const worker = this.ensureOneShotWorker();
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
      worker.postMessage({ type: 'run', id, wasmUrl: this.wasmUrl, executableName: 'monty', commands, preopenFiles: this.preopenFiles() });
    });
  }

  private optimizePersistentCommands(commands: string[]): string[] {
    const out: string[] = [];
    for (const command of commands) {
      if (command === 'uci' && this.persistentInitialized) continue;
      if (command.startsWith('setoption name Hash value ')) {
        if (command === this.persistentHashCommand) continue;
        this.persistentHashCommand = command;
      } else if (command.startsWith('setoption name Contempt value ')) {
        if (command === this.persistentContemptCommand) continue;
        this.persistentContemptCommand = command;
      } else if (command.startsWith('setoption name MultiPV value ')) {
        if (command === this.persistentMultipvCommand) continue;
        this.persistentMultipvCommand = command;
      }
      out.push(command);
    }
    return out;
  }

  private runPersistentCommands(
    commands: string[],
    signal?: AbortSignal,
    resolveWhenLine: (line: string, stream: 'stdout' | 'stderr') => boolean = (line, stream) => stream === 'stdout' && line.startsWith('bestmove'),
  ): Promise<RunResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    this.ensurePersistentWorker();
    const sharedInput = this.sharedInput;
    if (!sharedInput) return Promise.reject(new Error('Monty persistent stdin was not initialized'));
    return new Promise<RunResult>((resolve, reject) => {
      let active: PersistentPending;
      const onAbort = () => {
        if (this.persistentPending !== active) return;
        this.persistentPending = null;
        this.disposeWorker();
        reject(abortError());
      };
      active = {
        stdout: [],
        stderr: [],
        resolve: (result) => {
          signal?.removeEventListener('abort', onAbort);
          resolve(result);
        },
        reject: (error) => {
          signal?.removeEventListener('abort', onAbort);
          reject(error);
        },
        resolveWhenLine,
      };
      this.persistentPending = active;
      signal?.addEventListener('abort', onAbort, { once: true });
      try {
        writeSharedInput(sharedInput, `${this.optimizePersistentCommands(commands).join('\n')}\n`);
      } catch (error) {
        signal?.removeEventListener('abort', onAbort);
        this.persistentPending = null;
        reject(error as Error);
      }
    });
  }

  private async runCommandsUntil(commands: string[], signal?: AbortSignal, resolveWhenLine?: (line: string, stream: 'stdout' | 'stderr') => boolean): Promise<RunResult> {
    if (!this.persistentDisabled && canUsePersistentMontyWasi()) {
      try {
        return await this.runPersistentCommands(commands, signal, resolveWhenLine);
      } catch (error) {
        if ((error as Error).name === 'AbortError') throw error;
        this.persistentDisabled = true;
        this.disposeWorker();
        return this.runOneShotCommands(commands, signal);
      }
    }
    return this.runOneShotCommands(commands, signal);
  }

  private goCommand(options: MontyOptions): string {
    if (options.movetimeMs && options.movetimeMs > 0) return `go movetime ${Math.floor(options.movetimeMs)}`;
    if (options.nodes && options.nodes > 0) return `go nodes ${Math.floor(options.nodes)}`;
    if (options.depth && options.depth > 0) return `go depth ${Math.floor(options.depth)}`;
    return 'go nodes 1000';
  }

  private setupCommands(options: MontyOptions, multipv = 1): string[] {
    return [
      'uci',
      'isready',
      `setoption name Hash value ${Math.max(1, Math.floor(options.hashMb ?? this.options.hashMb ?? 64))}`,
      `setoption name Contempt value ${Math.max(-1000, Math.min(1000, Math.floor(options.contempt ?? this.options.contempt ?? 0)))}`,
      `setoption name MultiPV value ${Math.max(1, Math.floor(multipv))}`,
    ];
  }

  private searchCommands(fen: string, options: MontyOptions, multipv = 1): string[] {
    return [
      ...this.setupCommands(options, multipv),
      fen === 'startpos' ? 'position startpos' : `position fen ${fen}`,
      this.goCommand(options),
    ];
  }

  private parseInfo(stdout: string[]): StockfishInfoLine[] {
    const latest = new Map<number, StockfishInfoLine>();
    for (const line of stdout) {
      const stockfishStyle = parseStockfishInfo(line);
      if (stockfishStyle) {
        latest.set(stockfishStyle.multipv, stockfishStyle);
        continue;
      }
      if (!line.startsWith('info ')) continue;
      const multipv = Number(line.match(/\bmultipv (\d+)/)?.[1] ?? '1');
      const depth = Number(line.match(/\bdepth (\d+)/)?.[1] ?? '0');
      const cp = line.match(/\bscore cp (-?\d+)/);
      const mate = line.match(/\bscore mate (-?\d+)/);
      const nodes = line.match(/\bnodes (\d+)/);
      const nps = line.match(/\bnps (\d+)/);
      const pvUci = line.match(/ pv (.+)$/)?.[1].trim().split(/\s+/).filter(Boolean) ?? [];
      latest.set(multipv, {
        multipv,
        depth,
        scoreCp: cp ? Number(cp[1]) : undefined,
        mateIn: mate ? Number(mate[1]) : undefined,
        ...(nodes ? { nodes: Number(nodes[1]) } : {}),
        ...(nps ? { nps: Number(nps[1]) } : {}),
        pvUci,
      });
    }
    return [...latest.values()].sort((a, b) => a.multipv - b.multipv);
  }

  lastInfo(): StockfishInfoLine[] {
    return this.lastInfoLines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  runtimeStatus(): BrowserUciRuntimeStatus {
    return {
      mode: this.workerMode ?? 'idle',
      persistentAvailable: canUsePersistentMontyWasi(),
      persistentDisabled: this.persistentDisabled,
      wasmUrl: this.wasmUrl,
    };
  }

  runtimeLabel(): string {
    if (this.workerMode === 'persistent') return 'persistent';
    if (this.workerMode === 'oneshot') return canUsePersistentMontyWasi() && this.persistentDisabled ? 'one-shot fallback' : 'one-shot';
    return canUsePersistentMontyWasi() ? 'persistent available' : 'one-shot fallback';
  }

  /** Start the engine and load the detached networks before the first search. */
  async prewarm(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      const result = await this.runCommandsUntil(['uci', 'isready'], signal, (line, stream) => stream === 'stdout' && line === 'readyok');
      if (result.exitCode !== 0) throw new Error(`Monty prewarm exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
    });
  }

  async newGame(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      this.lastInfoLines = [];
      if (this.workerMode !== 'persistent') return;
      const result = await this.runCommandsUntil(['ucinewgame', 'isready'], signal, (line, stream) => stream === 'stdout' && line === 'readyok');
      if (result.exitCode !== 0) throw new Error(`Monty new game exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
    });
  }

  async search(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.bestMove(fen, signal);
  }

  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      const result = await this.runCommandsUntil(this.searchCommands(fen, this.options, 1), signal);
      if (result.exitCode !== 0) throw new Error(`Monty exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return parseBestMove(result.stdout.find((line) => line.startsWith('bestmove')) ?? '');
    });
  }

  async analyze(fen: string, opts: BrowserUciAnalysisOptions & { nodes?: number; contempt?: number } = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      const options = {
        ...this.options,
        nodes: opts.nodes ?? this.options.nodes,
        depth: opts.depth ?? this.options.depth,
        movetimeMs: opts.movetimeMs ?? this.options.movetimeMs,
        contempt: opts.contempt ?? this.options.contempt,
      };
      const result = await this.runCommandsUntil(this.searchCommands(fen, options, opts.multipv ?? 1), opts.signal);
      if (result.exitCode !== 0) throw new Error(`Monty exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return this.lastInfo();
    });
  }

  dispose(): void {
    const error = abortError();
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    if (this.persistentPending) {
      this.persistentPending.reject(error);
      this.persistentPending = null;
    }
    this.disposeWorker();
    this.lastInfoLines = [];
  }
}
