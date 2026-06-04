import { parseBestMove, parseStockfishInfo, type StockfishInfoLine } from './stockfishEngine.ts';

export interface RecklessOptions {
  /** Fixed search depth. Lower = faster. */
  depth?: number;
  /** Alternative to depth: fixed think time in ms. */
  movetimeMs?: number;
  /** Transposition-table size in MB. */
  hashMb?: number;
}

export interface RecklessBrowserApiLoadStatus {
  phase: string;
  url?: string;
  nnueUrl?: string;
  loadedBytes?: number;
  totalBytes?: number;
  elapsedMs?: number;
}

export interface RecklessRuntimeOptions {
  /** Runtime backend. WASI/UCI is the stable default; browser-api is an experimental direct-call path. */
  backend?: 'wasi' | 'browser-api';
  /** Benchmark/debug knob: force the old one-shot WASI path even when SAB persistence is available. */
  forceOneShot?: boolean;
  /** Benchmark/debug knob: fail instead of silently falling back when persistent startup/search errors. */
  disablePersistentFallback?: boolean;
  /** Optional external NNUE asset URL for browser-api builds that do not embed network data. */
  nnueUrl?: string;
  /** Called when the browser API worker reports load/progress status. */
  onStatus?: () => void;
}

export interface RecklessRuntimeStatus {
  mode: 'idle' | 'oneshot' | 'persistent' | 'browser-api';
  persistentAvailable: boolean;
  persistentDisabled: boolean;
  forceOneShot: boolean;
  wasmUrl: string;
  nnueUrl?: string;
  browserApiLoad?: RecklessBrowserApiLoadStatus;
}

export const DEFAULT_RECKLESS_WASM_URL = '/reckless/reckless.wasm';

const SHARED_STDIN_HEADER_INTS = 4;
const SHARED_STDIN_HEADER_BYTES = SHARED_STDIN_HEADER_INTS * Int32Array.BYTES_PER_ELEMENT;
const PERSISTENT_STDIN_CAPACITY_BYTES = 64 * 1024;
const PERSISTENT_ABORT_GRACE_MS = 100;

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

interface BrowserApiSearchLine {
  multipv: number;
  depth: number;
  scoreCp: number | null;
  mateIn: number | null;
  nodes: number;
  nps: number;
  pv: string[];
}

interface BrowserApiSearchResult {
  bestmove: string | null;
  elapsedMs: number;
  lines: BrowserApiSearchLine[];
}

type Pending = { resolve: (result: RunResult) => void; reject: (error: Error) => void };
type BrowserApiPending = { resolve: (result: BrowserApiSearchResult | null) => void; reject: (error: Error) => void };
type BrowserApiWorkerMessage = { type: string; id: number; result?: BrowserApiSearchResult; error?: string } & Partial<RecklessBrowserApiLoadStatus>;
type PersistentPending = Pending & {
  stdout: string[];
  stderr: string[];
  resolveWhenLine: (line: string, stream: 'stdout' | 'stderr') => boolean;
  aborted?: boolean;
  abortTimer?: ReturnType<typeof setTimeout>;
};

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

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

export function formatRecklessBrowserApiLoadStatus(status: RecklessBrowserApiLoadStatus | undefined): string {
  if (!status) return '';
  const parts = [`load ${status.phase}`];
  if (status.loadedBytes !== undefined && status.totalBytes !== undefined && status.totalBytes > 0) {
    const pct = Math.min(100, Math.max(0, (status.loadedBytes / status.totalBytes) * 100));
    parts.push(`${formatBytes(status.loadedBytes)}/${formatBytes(status.totalBytes)} ${pct.toFixed(0)}%`);
  } else if (status.loadedBytes !== undefined) {
    parts.push(formatBytes(status.loadedBytes));
  }
  if (status.elapsedMs !== undefined) parts.push(`${status.elapsedMs.toFixed(0)}ms`);
  return parts.join(' · ');
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
  private workerMode: 'oneshot' | 'persistent' | 'browser-api' | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private browserApiPending = new Map<number, BrowserApiPending>();
  private persistentPending: PersistentPending | null = null;
  private sharedInput: SharedInput | null = null;
  private persistentDisabled = false;
  private persistentHashCommand: string | null = null;
  private persistentThreadsCommand: string | null = null;
  private persistentMultipvCommand: string | null = null;
  private persistentMinimalCommand: string | null = null;
  private persistentPositionCommand: string | null = null;
  private queueTail: Promise<void> = Promise.resolve();
  private options: RecklessOptions;
  private readonly wasmUrl: string;
  private readonly runtimeOptions: RecklessRuntimeOptions;
  private browserApiLoadStatus: RecklessBrowserApiLoadStatus | null = null;
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

  private ensureBrowserApiWorker(): Worker {
    if (this.worker && this.workerMode === 'browser-api') return this.worker;
    this.disposeWorker();
    const worker = new Worker(new URL('./recklessBrowserApiWorker.ts', import.meta.url), { type: 'module', name: 'reckless-browser-api' });
    this.browserApiLoadStatus = null;
    worker.onmessage = (event: MessageEvent) => {
      const message = event.data as BrowserApiWorkerMessage;
      if (message.type === 'status') {
        this.browserApiLoadStatus = {
          phase: message.phase ?? 'unknown',
          url: message.url,
          nnueUrl: message.nnueUrl,
          loadedBytes: message.loadedBytes,
          totalBytes: message.totalBytes,
          elapsedMs: message.elapsedMs,
        };
        this.runtimeOptions.onStatus?.();
        return;
      }
      const pending = this.browserApiPending.get(message.id);
      if (!pending) return;
      this.browserApiPending.delete(message.id);
      if (message.type === 'error') pending.reject(new Error(message.error ?? 'Reckless browser API worker error'));
      else pending.resolve(message.result ?? null);
    };
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Reckless browser API worker error'));
    this.worker = worker;
    this.workerMode = 'browser-api';
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
      if (active.resolveWhenLine(message.line, message.stream ?? 'stdout')) {
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
    if (active.abortTimer) clearTimeout(active.abortTimer);
    if (active.aborted) active.reject(abortError());
    else active.resolve(result);
  }

  private rejectAllAndDispose(error: Error): void {
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear();
    for (const pending of this.browserApiPending.values()) pending.reject(error);
    this.browserApiPending.clear();
    if (this.persistentPending) {
      if (this.persistentPending.abortTimer) clearTimeout(this.persistentPending.abortTimer);
      this.persistentPending.reject(error);
      this.persistentPending = null;
    }
    this.disposeWorker();
  }

  private disposeWorker(): void {
    if (this.sharedInput) closeSharedInput(this.sharedInput);
    if (this.workerMode === 'browser-api') this.worker?.postMessage({ type: 'dispose' });
    this.worker?.terminate();
    this.worker = null;
    this.workerMode = null;
    this.sharedInput = null;
    this.persistentHashCommand = null;
    this.persistentThreadsCommand = null;
    this.persistentMultipvCommand = null;
    this.persistentMinimalCommand = null;
    this.persistentPositionCommand = null;
    if (this.workerMode !== 'browser-api') this.browserApiLoadStatus = null;
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
      } else if (command.startsWith('setoption name Minimal value ')) {
        if (command === this.persistentMinimalCommand) continue;
        this.persistentMinimalCommand = command;
      } else if (command.startsWith('position ')) {
        if (command === this.persistentPositionCommand) continue;
        this.persistentPositionCommand = command;
      } else if (command === 'ucinewgame') {
        this.persistentPositionCommand = null;
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
    if (!sharedInput) return Promise.reject(new Error('Reckless persistent stdin was not initialized'));
    return new Promise<RunResult>((resolve, reject) => {
      let active: PersistentPending;
      const onAbort = () => {
        if (this.persistentPending !== active) return;
        active.aborted = true;
        try {
          writeSharedInput(sharedInput, 'stop\n');
        } catch {
          this.persistentPending = null;
          this.disposeWorker();
          reject(abortError());
          return;
        }
        active.abortTimer = setTimeout(() => {
          if (this.persistentPending !== active) return;
          this.persistentPending = null;
          this.disposeWorker();
          reject(abortError());
        }, PERSISTENT_ABORT_GRACE_MS);
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

  private runBrowserApiMessage(
    message: Omit<{ type: 'prewarm' | 'new-game' | 'search'; id: number; wasmUrl: string; hashMb?: number; fen?: string; depth?: number; movetimeMs?: number; multipv?: number }, 'id' | 'wasmUrl'>,
    signal?: AbortSignal,
  ): Promise<BrowserApiSearchResult | null> {
    if (signal?.aborted) return Promise.reject(abortError());
    const id = ++this.seq;
    const worker = this.ensureBrowserApiWorker();
    return new Promise<BrowserApiSearchResult | null>((resolve, reject) => {
      const onAbort = () => {
        this.browserApiPending.delete(id);
        this.disposeWorker();
        reject(abortError());
      };
      this.browserApiPending.set(id, {
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
      worker.postMessage({ ...message, id, wasmUrl: this.wasmUrl, nnueUrl: this.runtimeOptions.nnueUrl, hashMb: this.options.hashMb ?? 16 });
    });
  }

  private browserApiResultToInfo(result: BrowserApiSearchResult | null): StockfishInfoLine[] {
    return (result?.lines ?? []).map((line) => ({
      multipv: line.multipv,
      depth: line.depth,
      ...(line.scoreCp === null ? {} : { scoreCp: line.scoreCp }),
      ...(line.mateIn === null ? {} : { mateIn: line.mateIn }),
      nodes: line.nodes,
      nps: line.nps,
      pvUci: [...line.pv],
    }));
  }

  private async runCommandsUntil(
    commands: string[],
    signal?: AbortSignal,
    resolveWhenLine?: (line: string, stream: 'stdout' | 'stderr') => boolean,
  ): Promise<RunResult> {
    if (!this.runtimeOptions.forceOneShot && !this.persistentDisabled && canUsePersistentRecklessWasi()) {
      try {
        return await this.runPersistentCommands(commands, signal, resolveWhenLine);
      } catch (error) {
        if ((error as Error).name === 'AbortError' || this.runtimeOptions.disablePersistentFallback) throw error;
        this.persistentDisabled = true;
        this.disposeWorker();
        return this.runOneShotCommands(commands, signal);
      }
    }
    return this.runOneShotCommands(commands, signal);
  }

  private runCommands(commands: string[], signal?: AbortSignal): Promise<RunResult> {
    return this.runCommandsUntil(commands, signal);
  }

  private searchCommands(fen: string, options: RecklessOptions, multipv = 1): string[] {
    return [
      hashCommand(options.hashMb ?? this.options.hashMb ?? 16),
      'setoption name Threads value 1',
      `setoption name MultiPV value ${Math.max(1, Math.floor(multipv))}`,
      'setoption name Minimal value true',
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

  /**
   * Start and initialize the persistent WASI/UCI process before the first real
   * search. This hides worker creation, wasm instantiation, and UCI `isready`
   * latency when cross-origin isolation allows the SharedArrayBuffer path.
   */
  async prewarm(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      if (this.runtimeOptions.backend === 'browser-api') {
        await this.runBrowserApiMessage({ type: 'prewarm' }, signal);
        return;
      }
      if (this.runtimeOptions.forceOneShot || this.persistentDisabled || !canUsePersistentRecklessWasi()) return;
      try {
        const result = await this.runCommandsUntil(['uci', 'isready'], signal, (line, stream) => stream === 'stdout' && line === 'readyok');
        if (result.exitCode !== 0) throw new Error(`Reckless prewarm exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      } catch (error) {
        if ((error as Error).name === 'AbortError' || this.runtimeOptions.disablePersistentFallback) throw error;
        this.persistentDisabled = true;
        this.disposeWorker();
      }
    });
  }

  /**
   * Reset Reckless' game/search state and wait for readiness. Useful for
   * benchmarks that should avoid persistent transposition-table reuse between
   * timed searches.
   */
  async newGame(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      if (this.runtimeOptions.backend === 'browser-api') {
        await this.runBrowserApiMessage({ type: 'new-game' }, signal);
        this.lastInfoLines = [];
        return;
      }
      const result = await this.runCommandsUntil(['ucinewgame', 'isready'], signal, (line, stream) => stream === 'stdout' && line === 'readyok');
      if (result.exitCode !== 0) throw new Error(`Reckless new game exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = [];
    });
  }

  runtimeStatus(): RecklessRuntimeStatus {
    return {
      mode: this.workerMode ?? 'idle',
      persistentAvailable: canUsePersistentRecklessWasi(),
      persistentDisabled: this.persistentDisabled,
      forceOneShot: this.runtimeOptions.forceOneShot === true,
      wasmUrl: this.wasmUrl,
      nnueUrl: this.runtimeOptions.nnueUrl,
      browserApiLoad: this.browserApiLoadStatus ?? undefined,
    };
  }

  runtimeLabel(): string {
    const status = this.runtimeStatus();
    if (status.mode === 'browser-api') return 'browser API';
    if (this.runtimeOptions.backend === 'browser-api') return 'browser API available';
    if (status.forceOneShot) return 'one-shot forced';
    if (status.mode === 'persistent') return 'persistent';
    if (status.mode === 'oneshot') return status.persistentAvailable && status.persistentDisabled ? 'one-shot fallback' : 'one-shot';
    return status.persistentAvailable ? 'persistent available' : 'one-shot fallback';
  }

  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      if (this.runtimeOptions.backend === 'browser-api') {
        const result = await this.runBrowserApiMessage({ type: 'search', fen, depth: this.options.depth, movetimeMs: this.options.movetimeMs, multipv: 1 }, signal);
        this.lastInfoLines = this.browserApiResultToInfo(result);
        return result?.bestmove ?? this.lastInfoLines[0]?.pvUci[0] ?? null;
      }
      const result = await this.runCommands(this.searchCommands(fen, this.options, 1), signal);
      if (result.exitCode !== 0) throw new Error(`Reckless exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return parseBestMove(result.stdout.find((line) => line.startsWith('bestmove')) ?? '');
    });
  }

  async analyze(fen: string, opts: { multipv?: number; depth?: number; movetimeMs?: number; signal?: AbortSignal } = {}): Promise<StockfishInfoLine[]> {
    return this.runExclusive(async () => {
      if (this.runtimeOptions.backend === 'browser-api') {
        const result = await this.runBrowserApiMessage(
          { type: 'search', fen, depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs ?? this.options.movetimeMs, multipv: opts.multipv ?? 1 },
          opts.signal,
        );
        this.lastInfoLines = this.browserApiResultToInfo(result);
        return this.lastInfo();
      }
      const result = await this.runCommands(this.searchCommands(fen, { ...this.options, depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs ?? this.options.movetimeMs }, opts.multipv ?? 1), opts.signal);
      if (result.exitCode !== 0) throw new Error(`Reckless exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = this.parseInfo(result.stdout);
      return this.lastInfo();
    });
  }

  dispose(): void {
    for (const pending of this.pending.values()) pending.reject(abortError());
    this.pending.clear();
    for (const pending of this.browserApiPending.values()) pending.reject(abortError());
    this.browserApiPending.clear();
    if (this.persistentPending) {
      this.persistentPending.reject(abortError());
      this.persistentPending = null;
    }
    this.disposeWorker();
    this.lastInfoLines = [];
  }
}
