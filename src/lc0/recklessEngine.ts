import { parseBestMove, parseStockfishInfo, type StockfishInfoLine } from './stockfishEngine.ts';

export interface RecklessOptions {
  /** Fixed search depth. Lower = faster. */
  depth?: number;
  /** Alternative to depth: fixed think time in ms. */
  movetimeMs?: number;
  /** Transposition-table size in MB. */
  hashMb?: number;
}

export interface RecklessRuntimeOptions {
  /** Benchmark/debug knob: force the old one-shot WASI path even when SAB persistence is available. */
  forceOneShot?: boolean;
  /** Benchmark/debug knob: fail instead of silently falling back when persistent startup/search errors. */
  disablePersistentFallback?: boolean;
}

export interface RecklessRuntimeStatus {
  mode: 'idle' | 'oneshot' | 'persistent';
  persistentAvailable: boolean;
  persistentDisabled: boolean;
  forceOneShot: boolean;
  wasmUrl: string;
}

export const DEFAULT_RECKLESS_WASM_URL = '/reckless/reckless.wasm';

const SHARED_STDIN_HEADER_INTS = 4;
const SHARED_STDIN_HEADER_BYTES = SHARED_STDIN_HEADER_INTS * Int32Array.BYTES_PER_ELEMENT;
const PERSISTENT_STDIN_CAPACITY_BYTES = 64 * 1024;

function goCommand(options: RecklessOptions): string {
  if (options.movetimeMs && options.movetimeMs > 0) return `go movetime ${Math.floor(options.movetimeMs)}`;
  return `go depth ${Math.max(1, Math.floor(options.depth ?? 4))}`;
}

function hashCommand(hashMb: number): string {
  return `setoption name Hash value ${Math.max(1, Math.min(1024, Math.floor(hashMb)))}`;
}

interface RunResult {
  stdout: string[];
  stderr: string[];
  exitCode: number;
}

type Pending = { resolve: (result: RunResult) => void; reject: (error: Error) => void };
type PersistentPending = Pending & { stdout: string[]; stderr: string[]; onAbort?: () => void };

type SharedInput = {
  buffer: SharedArrayBuffer;
  control: Int32Array;
  data: Uint8Array;
  capacity: number;
};

function abortError(): Error {
  const error = new Error('Reckless search aborted');
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
  if (writePos - readPos + bytes.byteLength > input.capacity) {
    // The command batches are tiny; if this ever happens the reader is wedged.
    throw new Error('Reckless persistent stdin buffer is full');
  }
  let offset = 0;
  while (offset < bytes.byteLength) {
    const ringOffset = writePos % input.capacity;
    const n = Math.min(bytes.byteLength - offset, input.capacity - ringOffset);
    input.data.set(bytes.subarray(offset, offset + n), ringOffset);
    offset += n;
    writePos += n;
  }
  Atomics.store(input.control, 1, writePos);
  // Wake the WASI fd_read loop if it is waiting for more bytes.
  Atomics.notify(input.control, 1);
}

export function canUsePersistentRecklessWasi(): boolean {
  return typeof SharedArrayBuffer !== 'undefined' && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
}

/**
 * Browser adapter for Reckless' Rust/WASI build.
 *
 * In an isolated browser runtime, this starts one persistent patched WASI UCI
 * process and feeds commands through a tiny SharedArrayBuffer-backed stdin. That
 * preserves Reckless' hash across searches and avoids repeated WASI process and
 * NNUE initialization. Non-isolated browsers fall back to one-shot argv searches.
 */
export class RecklessEngine {
  readonly name = 'reckless-wasi';
  private worker: Worker | null = null;
  private workerMode: 'oneshot' | 'persistent' | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private persistentPending: PersistentPending | null = null;
  private sharedInput: SharedInput | null = null;
  private persistentDisabled = false;
  private persistentHashCommand: string | null = null;
  private persistentThreadsCommand: string | null = null;
  private persistentMultipvCommand: string | null = null;
  private queueTail: Promise<void> = Promise.resolve();
  private options: RecklessOptions;
  private readonly wasmUrl: string;
  private readonly runtimeOptions: RecklessRuntimeOptions;
  private lastInfoLines: StockfishInfoLine[] = [];

  constructor(options: RecklessOptions = {}, wasmUrl = DEFAULT_RECKLESS_WASM_URL, runtimeOptions: RecklessRuntimeOptions = {}) {
    this.options = options;
    this.wasmUrl = wasmUrl;
    this.runtimeOptions = runtimeOptions;
  }

  setOptions(next: RecklessOptions): void {
    this.options = { ...this.options, ...next };
  }

  private ensureOneShotWorker(): Worker {
    if (this.worker && this.workerMode === 'oneshot') return this.worker;
    this.disposeWorker();
    const worker = new Worker(new URL('./recklessWasiWorker.ts', import.meta.url), { type: 'module', name: 'reckless-wasi' });
    worker.onmessage = (event: MessageEvent) => {
      const message = event.data as { type: string; id: number; stdout?: string[]; stderr?: string[]; exitCode?: number; error?: string };
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.type === 'error') pending.reject(new Error(message.error ?? 'Reckless WASI worker error'));
      else pending.resolve({ stdout: message.stdout ?? [], stderr: message.stderr ?? [], exitCode: message.exitCode ?? 0 });
    };
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Reckless WASI worker error'));
    this.worker = worker;
    this.workerMode = 'oneshot';
    return worker;
  }

  private ensurePersistentWorker(): Worker {
    if (this.worker && this.workerMode === 'persistent' && this.sharedInput) return this.worker;
    this.disposeWorker();
    const worker = new Worker(new URL('./recklessWasiWorker.ts', import.meta.url), { type: 'module', name: 'reckless-wasi-persistent' });
    const sharedInput = createSharedInput();
    worker.onmessage = (event: MessageEvent) => this.handlePersistentMessage(event.data as { type: string; stream?: 'stdout' | 'stderr'; line?: string; exitCode?: number; error?: string });
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Reckless persistent WASI worker error'));
    this.worker = worker;
    this.workerMode = 'persistent';
    this.sharedInput = sharedInput;
    worker.postMessage({ type: 'start-persistent', wasmUrl: this.wasmUrl, inputBuffer: sharedInput.buffer });
    return worker;
  }

  private handlePersistentMessage(message: { type: string; stream?: 'stdout' | 'stderr'; line?: string; exitCode?: number; error?: string }): void {
    if (message.type === 'persistent-ready') return;
    if (message.type === 'persistent-line') {
      const active = this.persistentPending;
      if (!active || !message.line) return;
      if (message.stream === 'stderr') active.stderr.push(message.line);
      else active.stdout.push(message.line);
      if (message.stream === 'stdout' && message.line.startsWith('bestmove')) {
        this.resolvePersistent({ stdout: active.stdout, stderr: active.stderr, exitCode: 0 });
      }
      return;
    }
    if (message.type === 'persistent-error') {
      this.rejectAllAndDispose(new Error(message.error ?? 'Reckless persistent WASI worker error'));
      return;
    }
    if (message.type === 'persistent-exit') {
      const error = new Error(`Reckless persistent WASI process exited with ${message.exitCode ?? 0}`);
      this.rejectAllAndDispose(error);
    }
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
    this.persistentHashCommand = null;
    this.persistentThreadsCommand = null;
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
      worker.postMessage({ type: 'run', id, wasmUrl: this.wasmUrl, commands });
    });
  }

  private optimizePersistentCommands(commands: string[]): string[] {
    const out: string[] = [];
    for (const command of commands) {
      if (command.startsWith('setoption name Hash value ')) {
        if (command === this.persistentHashCommand) continue;
        this.persistentHashCommand = command;
      } else if (command.startsWith('setoption name Threads value ')) {
        if (command === this.persistentThreadsCommand) continue;
        this.persistentThreadsCommand = command;
      } else if (command.startsWith('setoption name MultiPV value ')) {
        if (command === this.persistentMultipvCommand) continue;
        this.persistentMultipvCommand = command;
      }
      out.push(command);
    }
    return out;
  }

  private runPersistentCommands(commands: string[], signal?: AbortSignal): Promise<RunResult> {
    if (signal?.aborted) return Promise.reject(abortError());
    this.ensurePersistentWorker();
    const sharedInput = this.sharedInput;
    if (!sharedInput) return Promise.reject(new Error('Reckless persistent stdin was not initialized'));
    return new Promise<RunResult>((resolve, reject) => {
      const onAbort = () => {
        this.persistentPending = null;
        this.disposeWorker();
        reject(abortError());
      };
      this.persistentPending = {
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
        onAbort,
      };
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

  private async runCommands(commands: string[], signal?: AbortSignal): Promise<RunResult> {
    if (!this.runtimeOptions.forceOneShot && !this.persistentDisabled && canUsePersistentRecklessWasi()) {
      try {
        return await this.runPersistentCommands(commands, signal);
      } catch (error) {
        if ((error as Error).name === 'AbortError' || this.runtimeOptions.disablePersistentFallback) throw error;
        this.persistentDisabled = true;
        this.disposeWorker();
        return this.runOneShotCommands(commands, signal);
      }
    }
    return this.runOneShotCommands(commands, signal);
  }

  private searchCommands(fen: string, options: RecklessOptions, multipv = 1): string[] {
    return [
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

  runtimeStatus(): RecklessRuntimeStatus {
    return {
      mode: this.workerMode ?? 'idle',
      persistentAvailable: canUsePersistentRecklessWasi(),
      persistentDisabled: this.persistentDisabled,
      forceOneShot: this.runtimeOptions.forceOneShot === true,
      wasmUrl: this.wasmUrl,
    };
  }

  runtimeLabel(): string {
    const status = this.runtimeStatus();
    if (status.forceOneShot) return 'one-shot forced';
    if (status.mode === 'persistent') return 'persistent';
    if (status.mode === 'oneshot') return status.persistentAvailable && status.persistentDisabled ? 'one-shot fallback' : 'one-shot';
    return status.persistentAvailable ? 'persistent available' : 'one-shot fallback';
  }

  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      const result = await this.runCommands(this.searchCommands(fen, this.options, 1), signal);
      if (result.exitCode !== 0) throw new Error(`Reckless exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return parseBestMove(result.stdout.find((line) => line.startsWith('bestmove')) ?? '');
    });
  }

  async analyze(fen: string, opts: { multipv?: number; depth?: number; movetimeMs?: number; signal?: AbortSignal } = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      const result = await this.runCommands(this.searchCommands(fen, { ...this.options, depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs ?? this.options.movetimeMs }, opts.multipv ?? 1), opts.signal);
      if (result.exitCode !== 0) throw new Error(`Reckless exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return this.lastInfo();
    });
  }

  dispose(): void {
    for (const pending of this.pending.values()) pending.reject(abortError());
    this.pending.clear();
    if (this.persistentPending) {
      this.persistentPending.reject(abortError());
      this.persistentPending = null;
    }
    this.disposeWorker();
    this.lastInfoLines = [];
  }
}
