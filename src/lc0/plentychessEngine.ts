import type { BrowserUciAnalysisOptions, BrowserUciEngine, BrowserUciInfoLine, BrowserUciRuntimeStatus } from './browserUciEngine.ts';
import { parseBestMove, parseStockfishInfo } from './stockfishEngine.ts';

export interface PlentyChessOptions {
  /** Fixed search depth. */
  depth?: number;
  /** Alternative to depth: fixed think time in ms. */
  movetimeMs?: number;
  /** UCI Hash size in MiB. */
  hashMb?: number;
  /** PlentyChess is currently built as synchronous single-thread Emscripten; keep at 1 unless a pthread build is promoted. */
  threads?: number;
}

export interface PlentyChessInfoLine extends BrowserUciInfoLine {}

export const DEFAULT_PLENTYCHESS_EMSCRIPTEN_JS_URL = '/plentychess/plentychess-emscripten.js';

export function plentyChessGoCommand(options: PlentyChessOptions): string {
  if (options.movetimeMs && options.movetimeMs > 0) return `go movetime ${Math.floor(options.movetimeMs)}`;
  return `go depth ${Math.max(1, Math.floor(options.depth ?? 4))}`;
}

function hashCommand(hashMb: number): string {
  return `setoption name Hash value ${Math.max(2, Math.min(33554432, Math.floor(hashMb)))}`;
}

function threadsCommand(threads: number): string {
  return `setoption name Threads value ${Math.max(1, Math.min(2048, Math.floor(threads)))}`;
}

function abortError(message = 'PlentyChess search aborted'): Error {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

function workerScript(): string {
  return String.raw`
let factory = null;
let modulePromise = null;
let engine = null;
function postError(id, error) {
  self.postMessage({ type: 'error', id, message: error && error.message ? error.message : String(error) });
}
function resolveUrl(url) {
  return new URL(url, self.location.href).href;
}
async function init(id, jsUrl) {
  if (!modulePromise) {
    const resolvedJsUrl = resolveUrl(jsUrl);
    importScripts(resolvedJsUrl);
    factory = self.PlentyChess || (typeof PlentyChess !== 'undefined' ? PlentyChess : null);
    if (!factory) throw new Error('PlentyChess Emscripten factory was not found after importScripts()');
    modulePromise = factory({
      locateFile(file) { return new URL(file, resolvedJsUrl).href; },
      print(line) { self.postMessage({ type: 'line', line: String(line), stream: 'stdout' }); },
      printErr(line) { self.postMessage({ type: 'line', line: String(line), stream: 'stderr' }); },
    }).then((mod) => {
      engine = mod;
      return mod;
    });
  }
  await modulePromise;
  self.postMessage({ type: 'ready', id });
}
self.onmessage = (event) => {
  const message = event.data || {};
  const id = message.id;
  Promise.resolve().then(async () => {
    if (message.type === 'init') {
      await init(id, message.jsUrl);
      return;
    }
    if (message.type === 'command') {
      await init(id, message.jsUrl);
      engine.ccall('command', null, ['string'], [String(message.command)]);
      self.postMessage({ type: 'commandDone', id });
      return;
    }
    throw new Error('Unknown PlentyChess worker message type: ' + message.type);
  }).catch((error) => postError(id, error));
};
`;
}

type WorkerResponse =
  | { type: 'ready'; id: number }
  | { type: 'commandDone'; id: number }
  | { type: 'line'; line: string; stream: 'stdout' | 'stderr' }
  | { type: 'error'; id?: number; message: string };

interface Waiter<T> {
  resolve: (value: T) => void;
  reject: (error: Error) => void;
  cleanup?: () => void;
}

interface LineWaiter extends Waiter<string> {
  startIndex: number;
  predicate: (line: string) => boolean;
  label: string;
}

export class PlentyChessEngine implements BrowserUciEngine {
  readonly name = 'plentychess-emscripten';
  private worker: Worker | null = null;
  private workerObjectUrl: string | null = null;
  private nextId = 1;
  private pending = new Map<number, Waiter<void>>();
  private lineWaiters: LineWaiter[] = [];
  private stdout: string[] = [];
  private stderr: string[] = [];
  private options: PlentyChessOptions;
  private initialized = false;
  private queueTail: Promise<void> = Promise.resolve();
  private lastInfoLines: PlentyChessInfoLine[] = [];
  private readonly jsUrl: string;

  constructor(options: PlentyChessOptions = {}, jsUrl: string = DEFAULT_PLENTYCHESS_EMSCRIPTEN_JS_URL) {
    this.options = options;
    this.jsUrl = jsUrl;
  }

  setOptions(next: PlentyChessOptions): void {
    this.options = { ...this.options, ...next };
  }

  private runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.queueTail;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    this.queueTail = previous.then(() => gate, () => gate);
    return previous.catch(() => undefined).then(async () => {
      try {
        return await fn();
      } finally {
        release();
      }
    });
  }

  private makeWorker(): Worker {
    if (this.worker) return this.worker;
    this.workerObjectUrl = URL.createObjectURL(new Blob([workerScript()], { type: 'text/javascript' }));
    const worker = new Worker(this.workerObjectUrl);
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handleMessage(event.data);
    worker.onerror = (event) => this.failAll(new Error(event.message || 'PlentyChess worker error'));
    this.worker = worker;
    return worker;
  }

  private handleMessage(message: WorkerResponse): void {
    if (message.type === 'line') {
      if (message.stream === 'stderr') this.stderr.push(message.line);
      else this.stdout.push(message.line);
      this.resolveLineWaiters();
      return;
    }
    if (message.type === 'ready' || message.type === 'commandDone') {
      const waiter = this.pending.get(message.id);
      if (!waiter) return;
      this.pending.delete(message.id);
      waiter.cleanup?.();
      waiter.resolve();
      return;
    }
    if (message.type === 'error') {
      const error = new Error(message.message);
      if (message.id !== undefined && this.pending.has(message.id)) {
        const waiter = this.pending.get(message.id)!;
        this.pending.delete(message.id);
        waiter.cleanup?.();
        waiter.reject(error);
      } else {
        this.failAll(error);
      }
    }
  }

  private resolveLineWaiters(): void {
    const remaining: LineWaiter[] = [];
    for (const waiter of this.lineWaiters) {
      let matched: string | null = null;
      for (let i = waiter.startIndex; i < this.stdout.length; i += 1) {
        if (waiter.predicate(this.stdout[i])) {
          matched = this.stdout[i];
          break;
        }
      }
      if (matched !== null) {
        waiter.cleanup?.();
        waiter.resolve(matched);
      } else {
        remaining.push(waiter);
      }
    }
    this.lineWaiters = remaining;
  }

  private resolvedJsUrl(): string {
    return new URL(this.jsUrl, location.href).href;
  }

  private request(type: 'init' | 'command', payload: Record<string, unknown> = {}, signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return Promise.reject(abortError());
    const worker = this.makeWorker();
    const id = this.nextId++;
    return new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        this.pending.delete(id);
        this.restartAfterAbort();
        reject(abortError());
      };
      const cleanup = () => signal?.removeEventListener('abort', onAbort);
      this.pending.set(id, { resolve, reject, cleanup });
      signal?.addEventListener('abort', onAbort, { once: true });
      worker.postMessage({ type, id, jsUrl: this.resolvedJsUrl(), ...payload });
    });
  }

  private waitForLine(predicate: (line: string) => boolean, label: string, startIndex: number, signal?: AbortSignal, timeoutMs = 20000): Promise<string> {
    if (signal?.aborted) return Promise.reject(abortError());
    for (let i = startIndex; i < this.stdout.length; i += 1) if (predicate(this.stdout[i])) return Promise.resolve(this.stdout[i]);
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.lineWaiters = this.lineWaiters.filter((entry) => entry !== waiter);
        reject(new Error(`Timed out waiting for ${label}; last PlentyChess lines: ${this.stdout.slice(-8).join(' | ')}`));
      }, timeoutMs);
      const onAbort = () => {
        clearTimeout(timer);
        this.lineWaiters = this.lineWaiters.filter((entry) => entry !== waiter);
        this.restartAfterAbort();
        reject(abortError());
      };
      const waiter: LineWaiter = {
        startIndex,
        predicate,
        label,
        resolve: (line) => resolve(line),
        reject,
        cleanup: () => {
          clearTimeout(timer);
          signal?.removeEventListener('abort', onAbort);
        },
      };
      signal?.addEventListener('abort', onAbort, { once: true });
      this.lineWaiters.push(waiter);
    });
  }

  private failAll(error: Error): void {
    for (const waiter of this.pending.values()) {
      waiter.cleanup?.();
      waiter.reject(error);
    }
    for (const waiter of this.lineWaiters) {
      waiter.cleanup?.();
      waiter.reject(error);
    }
    this.pending.clear();
    this.lineWaiters = [];
    this.terminateWorker();
  }

  private terminateWorker(): void {
    this.worker?.terminate();
    this.worker = null;
    if (this.workerObjectUrl) URL.revokeObjectURL(this.workerObjectUrl);
    this.workerObjectUrl = null;
    this.initialized = false;
  }

  private restartAfterAbort(): void {
    this.terminateWorker();
    this.pending.clear();
    this.lineWaiters = [];
  }

  private async sendCommand(command: string, signal?: AbortSignal): Promise<void> {
    await this.request('command', { command }, signal);
  }

  private applyOptions(signal?: AbortSignal): Promise<void> {
    const commands: string[] = [];
    if (this.options.hashMb !== undefined) commands.push(hashCommand(this.options.hashMb));
    if (this.options.threads !== undefined) commands.push(threadsCommand(this.options.threads));
    return commands.reduce((promise, command) => promise.then(() => this.sendCommand(command, signal)), Promise.resolve());
  }

  async prewarm(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      await this.request('init', {}, signal);
      if (this.initialized) return;
      const uciStart = this.stdout.length;
      await this.sendCommand('uci', signal);
      await this.waitForLine((line) => line === 'uciok', 'uciok', uciStart, signal);
      await this.applyOptions(signal);
      const readyStart = this.stdout.length;
      await this.sendCommand('isready', signal);
      await this.waitForLine((line) => line === 'readyok', 'readyok', readyStart, signal);
      this.initialized = true;
    });
  }

  async newGame(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(async () => {
      await this.prewarmUnlocked(signal);
      await this.sendCommand('ucinewgame', signal);
      const readyStart = this.stdout.length;
      await this.sendCommand('isready', signal);
      await this.waitForLine((line) => line === 'readyok', 'readyok after ucinewgame', readyStart, signal);
      this.lastInfoLines = [];
    });
  }

  private async prewarmUnlocked(signal?: AbortSignal): Promise<void> {
    await this.request('init', {}, signal);
    if (this.initialized) return;
    const uciStart = this.stdout.length;
    await this.sendCommand('uci', signal);
    await this.waitForLine((line) => line === 'uciok', 'uciok', uciStart, signal);
    await this.applyOptions(signal);
    const readyStart = this.stdout.length;
    await this.sendCommand('isready', signal);
    await this.waitForLine((line) => line === 'readyok', 'readyok', readyStart, signal);
    this.initialized = true;
  }

  async search(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.bestMove(fen, signal);
  }

  async bestMove(fen: string, signal?: AbortSignal): Promise<string | null> {
    return this.runExclusive(async () => {
      await this.prewarmUnlocked(signal);
      this.lastInfoLines = [];
      await this.applyOptions(signal);
      await this.sendCommand('setoption name MultiPV value 1', signal);
      await this.sendCommand(`position fen ${fen}`, signal);
      const searchStart = this.stdout.length;
      const commandPromise = this.sendCommand(plentyChessGoCommand(this.options), signal);
      try {
        const bestLine = await this.waitForLine((line) => line.startsWith('bestmove '), 'bestmove', searchStart, signal, 60000);
        await commandPromise;
        this.lastInfoLines = this.collectInfoLines(searchStart);
        return parseBestMove(bestLine);
      } catch (error) {
        commandPromise.catch(() => undefined);
        throw error;
      }
    });
  }

  async analyze(fen: string, opts: BrowserUciAnalysisOptions = {}): Promise<PlentyChessInfoLine[]> {
    return this.runExclusive(async () => {
      await this.prewarmUnlocked(opts.signal);
      const multipv = Math.max(1, Math.floor(opts.multipv ?? 1));
      this.lastInfoLines = [];
      await this.applyOptions(opts.signal);
      await this.sendCommand(`setoption name MultiPV value ${multipv}`, opts.signal);
      await this.sendCommand(`position fen ${fen}`, opts.signal);
      const searchStart = this.stdout.length;
      const commandPromise = this.sendCommand(plentyChessGoCommand({ ...this.options, depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs }), opts.signal);
      try {
        await this.waitForLine((line) => line.startsWith('bestmove '), 'analysis bestmove', searchStart, opts.signal, 60000);
        await commandPromise;
        this.lastInfoLines = this.collectInfoLines(searchStart);
        return this.lastInfo();
      } catch (error) {
        commandPromise.catch(() => undefined);
        throw error;
      }
    });
  }

  private collectInfoLines(startIndex: number): PlentyChessInfoLine[] {
    const byPv = new Map<number, PlentyChessInfoLine>();
    for (let i = startIndex; i < this.stdout.length; i += 1) {
      const parsed = parseStockfishInfo(this.stdout[i]);
      if (parsed) byPv.set(parsed.multipv, parsed);
    }
    return [...byPv.values()].sort((a, b) => a.multipv - b.multipv).map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  lastInfo(): PlentyChessInfoLine[] {
    return this.lastInfoLines.map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  runtimeStatus(): BrowserUciRuntimeStatus {
    return {
      mode: this.worker ? 'emscripten-worker' : 'idle',
      persistentAvailable: true,
      persistentDisabled: false,
      forceOneShot: false,
      workerUrl: this.jsUrl,
    };
  }

  runtimeLabel(): string {
    if (this.worker && this.initialized) return 'Emscripten worker ready';
    if (this.worker) return 'Emscripten worker loading';
    return 'Emscripten worker idle';
  }

  dispose(): void {
    this.failAll(abortError('PlentyChess engine disposed'));
    this.stdout = [];
    this.stderr = [];
    this.lastInfoLines = [];
  }
}
