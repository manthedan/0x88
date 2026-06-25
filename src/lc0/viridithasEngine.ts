import type { BrowserUciEngine } from './browserUciEngine.ts';
import { parseBestMove, parseStockfishInfo, type StockfishInfoLine } from './stockfishEngine.ts';
import type { RecklessOptions } from './recklessEngine.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';

export const DEFAULT_VIRIDITHAS_WASM_URL = resolvePublicAssetUrl('/viridithas/viridithas.wasm');

interface RunResult {
  stdout: string[];
  stderr: string[];
  exitCode: number;
}

export interface ViridithasBatchSearchResult {
  bestMove: string | null;
  info: StockfishInfoLine | null;
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

export interface ViridithasRuntimeOptions {
  /** Benchmark/debug knob: force one-shot argv execution even when persistent stdin is available. */
  forceOneShot?: boolean;
  /** Benchmark/debug knob: fail instead of silently falling back when persistent startup/search errors. */
  disablePersistentFallback?: boolean;
}

export interface ViridithasRuntimeStatus {
  mode: 'idle' | 'oneshot' | 'persistent';
  persistentAvailable: boolean;
  persistentDisabled: boolean;
  forceOneShot: boolean;
  wasmUrl: string;
}

const SHARED_STDIN_HEADER_INTS = 4;
const SHARED_STDIN_HEADER_BYTES = SHARED_STDIN_HEADER_INTS * Int32Array.BYTES_PER_ELEMENT;
const PERSISTENT_STDIN_CAPACITY_BYTES = 64 * 1024;

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
  if (writePos - readPos + bytes.byteLength > input.capacity) throw new Error('Viridithas persistent stdin buffer is full');
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

export function canUsePersistentViridithasWasi(): boolean {
  return typeof SharedArrayBuffer !== 'undefined' && (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
}

/** Experimental WASI adapter for patched Viridithas. */
export class ViridithasEngine implements BrowserUciEngine {
  readonly name = 'viridithas-wasi';
  private worker: Worker | null = null;
  private workerMode: 'oneshot' | 'persistent' | null = null;
  private seq = 0;
  private pending = new Map<number, Pending>();
  private persistentPending: PersistentPending | null = null;
  private sharedInput: SharedInput | null = null;
  private persistentDisabled = false;
  private persistentInitialized = false;
  private persistentHashCommand: string | null = null;
  private persistentThreadsCommand: string | null = null;
  private persistentMultipvCommand: string | null = null;
  private queueTail: Promise<void> = Promise.resolve();
  private lastInfoLines: StockfishInfoLine[] = [];

  constructor(private options: RecklessOptions = {}, private readonly wasmUrl = DEFAULT_VIRIDITHAS_WASM_URL, private readonly runtimeOptions: ViridithasRuntimeOptions = {}) {}

  setOptions(next: RecklessOptions): void {
    this.options = { ...this.options, ...next };
  }

  private ensureOneShotWorker(): Worker {
    if (this.worker && this.workerMode === 'oneshot') return this.worker;
    this.disposeWorker();
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
    this.workerMode = 'oneshot';
    return worker;
  }

  private ensurePersistentWorker(): Worker {
    if (this.worker && this.workerMode === 'persistent' && this.sharedInput) return this.worker;
    this.disposeWorker();
    const worker = new Worker(new URL('./recklessWasiWorker.ts', import.meta.url), { type: 'module', name: 'viridithas-wasi-persistent' });
    const sharedInput = createSharedInput();
    worker.onmessage = (event: MessageEvent) => this.handlePersistentMessage(event.data as { type: string; stream?: 'stdout' | 'stderr'; line?: string; exitCode?: number; error?: string });
    worker.onerror = (event) => this.rejectAllAndDispose(new Error(event.message || 'Viridithas persistent WASI worker error'));
    this.worker = worker;
    this.workerMode = 'persistent';
    this.sharedInput = sharedInput;
    worker.postMessage({ type: 'start-persistent', wasmUrl: this.wasmUrl, inputBuffer: sharedInput.buffer, executableName: 'viridithas' });
    return worker;
  }

  private handlePersistentMessage(message: { type: string; stream?: 'stdout' | 'stderr'; line?: string; exitCode?: number; error?: string }): void {
    if (message.type === 'persistent-ready') return;
    if (message.type === 'persistent-line') {
      const active = this.persistentPending;
      if (!active || !message.line) return;
      if (message.stream === 'stderr') active.stderr.push(message.line);
      else {
        active.stdout.push(message.line);
        // Only the UCI handshake flips Viridithas out of pretty-print mode; a
        // prior ucinewgame/isready sync must not make later searches skip `uci`.
        if (message.line === 'uciok') this.persistentInitialized = true;
      }
      if (active.resolveWhenLine(message.line, message.stream ?? 'stdout')) this.resolvePersistent({ stdout: active.stdout, stderr: active.stderr, exitCode: 0 });
      return;
    }
    if (message.type === 'persistent-error') {
      this.rejectAllAndDispose(new Error(message.error ?? 'Viridithas persistent WASI worker error'));
      return;
    }
    if (message.type === 'persistent-exit') this.rejectAllAndDispose(new Error(`Viridithas persistent WASI process exited with ${message.exitCode ?? 0}`));
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
      worker.postMessage({ type: 'run', id, wasmUrl: this.wasmUrl, executableName: 'viridithas', commands });
    });
  }

  private optimizePersistentCommands(commands: string[]): string[] {
    const out: string[] = [];
    for (const command of commands) {
      if (command === 'uci' && this.persistentInitialized) continue;
      if (command.startsWith('setoption name Hash value ')) {
        if (command === this.persistentHashCommand) continue;
        this.persistentHashCommand = command;
      } else if (command.startsWith('setoption name Threads value ')) {
        if (command === this.persistentThreadsCommand) continue;
        this.persistentThreadsCommand = command;
      } else if (command.startsWith('setoption name MultiPV value ')) {
        if (command === this.persistentMultipvCommand) continue;
        this.persistentMultipvCommand = command;
      } else if (command === 'ucinewgame') {
        // Hash contents changed but option state did not.
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
    if (!sharedInput) return Promise.reject(new Error('Viridithas persistent stdin was not initialized'));
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
    if (!this.runtimeOptions.forceOneShot && !this.persistentDisabled && canUsePersistentViridithasWasi()) {
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

  private setupCommands(options: RecklessOptions, multipv = 1): string[] {
    return [
      'uci',
      'isready',
      hashCommand(options.hashMb ?? this.options.hashMb ?? 16),
      'setoption name Threads value 1',
      `setoption name MultiPV value ${Math.max(1, Math.floor(multipv))}`,
    ];
  }

  private searchCommands(fen: string, options: RecklessOptions, multipv = 1): string[] {
    return [
      ...this.setupCommands(options, multipv),
      `position fen ${fen}`,
      goCommand(options),
    ];
  }

  private batchSearchCommands(fens: string[], options: RecklessOptions, multipv = 1, clearHashBetweenSearches = true): string[] {
    const commands = this.setupCommands(options, multipv);
    for (const fen of fens) {
      if (clearHashBetweenSearches) commands.push('ucinewgame');
      commands.push(`position fen ${fen}`, goCommand(options));
    }
    return commands;
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

  private parseBatchResults(stdout: string[]): ViridithasBatchSearchResult[] {
    const results: ViridithasBatchSearchResult[] = [];
    let chunk: string[] = [];
    for (const line of stdout) {
      chunk.push(line);
      if (!line.startsWith('bestmove')) continue;
      results.push({
        bestMove: parseBestMove(line),
        info: this.parseInfo(chunk)[0] ?? null,
      });
      chunk = [];
    }
    return results;
  }

  lastInfo(): StockfishInfoLine[] {
    return this.lastInfoLines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  /**
   * Start and initialize the persistent WASI/UCI process before the first real
   * search when isolation allows it. One-shot mode has nothing useful to keep
   * warm, so it returns after checking the requested runtime policy.
   */
  async prewarm(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      if (this.runtimeOptions.forceOneShot || this.persistentDisabled || !canUsePersistentViridithasWasi()) return;
      try {
        const result = await this.runCommandsUntil(['uci', 'isready'], signal, (line, stream) => stream === 'stdout' && line === 'readyok');
        if (result.exitCode !== 0) throw new Error(`Viridithas prewarm exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      } catch (error) {
        if ((error as Error).name === 'AbortError' || this.runtimeOptions.disablePersistentFallback) throw error;
        this.persistentDisabled = true;
        this.disposeWorker();
      }
    });
  }

  runtimeStatus(): ViridithasRuntimeStatus {
    return {
      mode: this.workerMode ?? 'idle',
      persistentAvailable: canUsePersistentViridithasWasi(),
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

  async newGame(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      if (this.runtimeOptions.forceOneShot || this.persistentDisabled || !canUsePersistentViridithasWasi()) {
        this.lastInfoLines = [];
        return;
      }
      const commands = this.persistentInitialized ? ['ucinewgame', 'isready'] : ['uci', 'isready', 'ucinewgame', 'isready'];
      const result = await this.runCommandsUntil(commands, signal, (line, stream) => stream === 'stdout' && line === 'readyok');
      if (result.exitCode !== 0) throw new Error(`Viridithas new game exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      this.lastInfoLines = [];
    });
  }

  async search(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.bestMove(fen, signal);
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

  async bestMovesBatch(fens: string[], signal?: AbortSignal, opts: { clearHashBetweenSearches?: boolean } = {}): Promise<ViridithasBatchSearchResult[]> {
    return this.runExclusive(async () => {
      const result = await this.runOneShotCommands(this.batchSearchCommands(fens, this.options, 1, opts.clearHashBetweenSearches ?? true), signal);
      if (result.exitCode !== 0) throw new Error(`Viridithas exited with ${result.exitCode}: ${result.stderr.join('\n')}`);
      const searches = this.parseBatchResults(result.stdout);
      if (searches.length !== fens.length) throw new Error(`Viridithas batch returned ${searches.length} bestmove line(s) for ${fens.length} position(s)`);
      this.lastInfoLines = searches.map((search) => search.info).filter((info): info is StockfishInfoLine => info !== null);
      return searches.map((search) => ({ bestMove: search.bestMove, info: search.info ? { ...search.info, pvUci: [...search.info.pvUci] } : null }));
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
