import { isTrustedExecutableAssetUrl } from './assetUrls.ts';
import {
  type BrowserUciAnalysisOptions,
  type BrowserUciEngine,
  type BrowserUciInfoLine,
  type BrowserUciRuntimeStatus,
  uciPositionCommand,
} from './browserUciEngine.ts';
import { resolveEmscriptenAssetUrl } from './emscriptenLocateFile.ts';
import { parseBestMove, parseStockfishInfo } from './stockfishEngine.ts';

/**
 * Shared adapter for the synchronous single-thread Emscripten UCI ports
 * (Berserk, PlentyChess, Stormphrax). Each build exposes a MODULARIZE'd
 * factory whose module takes one UCI line per `ccall('command', ...)` and
 * prints its replies; the per-engine files only supply names and URLs.
 */
export interface EmscriptenUciOptions {
  /** Fixed search depth. */
  depth?: number;
  /** Alternative to depth: fixed think time in ms. */
  movetimeMs?: number;
  /** UCI Hash size in MiB. */
  hashMb?: number;
  /** The ports are built as synchronous single-thread Emscripten; keep at 1 unless a pthread build is promoted. */
  threads?: number;
}

export interface EmscriptenUciEngineConfig {
  /** `BrowserUciEngine.name`, e.g. `stormphrax-emscripten`. */
  name: string;
  /** Engine name used in error messages. */
  label: string;
  /** Global defined by the MODULARIZE'd glue (its EXPORT_NAME). */
  factoryGlobal: string;
  /** Appended to glue/module load failures so a missing artifact names its rebuild command. */
  missingArtifactHint?: string;
}

export function emscriptenGoCommand(options: EmscriptenUciOptions): string {
  if (options.movetimeMs && options.movetimeMs > 0) return `go movetime ${Math.floor(options.movetimeMs)}`;
  return `go depth ${Math.max(1, Math.floor(options.depth ?? 4))}`;
}

export function emscriptenSearchTimeoutMs(options: EmscriptenUciOptions): number {
  // The timer starts before the worker receives `go`; leave room for message
  // dispatch and final UCI output above the requested move time.
  if (options.movetimeMs && options.movetimeMs > 0) return Math.max(60000, Math.floor(options.movetimeMs) + 15000);
  return 120000;
}

function hashCommand(hashMb: number): string {
  return `setoption name Hash value ${Math.max(2, Math.min(33554432, Math.floor(hashMb)))}`;
}

function threadsCommand(threads: number): string {
  return `setoption name Threads value ${Math.max(1, Math.min(2048, Math.floor(threads)))}`;
}

/** Output lines kept between exclusive sections, for timeout diagnostics only. */
const RETAINED_OUTPUT_LINES = 16;

function abortError(message: string): Error {
  const error = new Error(message);
  error.name = 'AbortError';
  return error;
}

export function emscriptenUciWorkerScript(config: Pick<EmscriptenUciEngineConfig, 'label' | 'factoryGlobal' | 'missingArtifactHint'>): string {
  // The factory global is spliced in as an identifier (a `var` declared by
  // importScripts may not be a property of `self` in every engine).
  if (!/^[A-Za-z_$][\w$]*$/.test(config.factoryGlobal)) throw new Error(`Invalid Emscripten factory global: ${config.factoryGlobal}`);
  const factory = config.factoryGlobal;
  const label = JSON.stringify(config.label);
  // A missing artifact indicates an incomplete checkout or deploy. When the
  // engine has a rebuild hint, carry it into the worker so the failure
  // identifies the affected URLs instead of a bare "NetworkError importing script".
  return String.raw`
const resolveEmscriptenAssetUrl = ${resolveEmscriptenAssetUrl.toString()};
const ENGINE_LABEL = ${label};
const MISSING_ARTIFACT_HINT = ${JSON.stringify(config.missingArtifactHint ?? null)};
function artifactLoadError(error, urls) {
  if (!MISSING_ARTIFACT_HINT) return error;
  const detail = error && error.message ? error.message : String(error);
  return new Error(ENGINE_LABEL + ' artifacts failed to load (' + urls.filter(Boolean).join(', ') + '): ' + detail + ' — ' + MISSING_ARTIFACT_HINT);
}
let factory = null;
let modulePromise = null;
let engine = null;
let artifactUrls = [];
function postError(id, error) {
  self.postMessage({ type: 'error', id, message: error && error.message ? error.message : String(error) });
}
function resolveUrl(url) {
  return new URL(url, self.location.href).href;
}
async function importEmscriptenGlue(resolvedJsUrl, trustedJsUrl) {
  const sameOrigin = new URL(resolvedJsUrl).origin === self.location.origin;
  if (sameOrigin) {
    importScripts(resolvedJsUrl);
    return;
  }
  if (!trustedJsUrl) throw new Error('Refusing to execute untrusted ' + ENGINE_LABEL + ' Emscripten glue: ' + resolvedJsUrl);
  const response = await fetch(resolvedJsUrl, { mode: 'cors', credentials: 'omit', cache: 'force-cache' });
  if (!response.ok) throw new Error('Failed to fetch ' + ENGINE_LABEL + ' Emscripten glue: ' + response.status + ' ' + resolvedJsUrl);
  const source = await response.text();
  const blobUrl = URL.createObjectURL(new Blob([source + '\n//# sourceURL=' + resolvedJsUrl], { type: 'text/javascript' }));
  try {
    importScripts(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}
async function init(id, jsUrl, wasmUrl, dataUrl, trustedJsUrl) {
  if (!modulePromise) {
    const resolvedJsUrl = resolveUrl(jsUrl);
    const resolvedWasmUrl = wasmUrl ? resolveUrl(wasmUrl) : null;
    const resolvedDataUrl = dataUrl ? resolveUrl(dataUrl) : null;
    artifactUrls = [resolvedJsUrl, resolvedWasmUrl, resolvedDataUrl];
    try {
      await importEmscriptenGlue(resolvedJsUrl, trustedJsUrl);
    } catch (error) {
      throw artifactLoadError(error, [resolvedJsUrl]);
    }
    factory = self.${factory} || (typeof ${factory} !== 'undefined' ? ${factory} : null);
    if (!factory) throw new Error(ENGINE_LABEL + ' Emscripten factory was not found after importScripts()');
    modulePromise = factory({
      locateFile(file) {
        return resolveEmscriptenAssetUrl(file, { jsUrl: resolvedJsUrl, wasmUrl: resolvedWasmUrl, dataUrl: resolvedDataUrl });
      },
      print(line) { self.postMessage({ type: 'line', line: String(line), stream: 'stdout' }); },
      printErr(line) { self.postMessage({ type: 'line', line: String(line), stream: 'stderr' }); },
    }).then((mod) => {
      engine = mod;
      return mod;
    });
  }
  try {
    await modulePromise;
  } catch (error) {
    throw artifactLoadError(error, artifactUrls);
  }
  self.postMessage({ type: 'ready', id });
}
self.onmessage = (event) => {
  const message = event.data || {};
  const id = message.id;
  Promise.resolve().then(async () => {
    if (message.type === 'init') {
      await init(id, message.jsUrl, message.wasmUrl, message.dataUrl, message.trustedJsUrl);
      return;
    }
    if (message.type === 'command') {
      await init(id, message.jsUrl, message.wasmUrl, message.dataUrl, message.trustedJsUrl);
      engine.ccall('command', null, ['string'], [String(message.command)]);
      self.postMessage({ type: 'commandDone', id });
      return;
    }
    throw new Error('Unknown ' + ENGINE_LABEL + ' worker message type: ' + message.type);
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

export class EmscriptenUciEngine<Options extends EmscriptenUciOptions = EmscriptenUciOptions> implements BrowserUciEngine {
  readonly name: string;
  private readonly config: EmscriptenUciEngineConfig;
  private worker: Worker | null = null;
  private workerObjectUrl: string | null = null;
  private nextId = 1;
  private pending = new Map<number, Waiter<void>>();
  private lineWaiters: LineWaiter[] = [];
  private stdout: string[] = [];
  private stderr: string[] = [];
  private options: Options;
  private initialized = false;
  private appliedOptions = new Map<string, string>();
  private queueTail: Promise<void> = Promise.resolve();
  private disposed = false;
  private lastInfoLines: BrowserUciInfoLine[] = [];
  private readonly jsUrl: string;
  private readonly wasmUrl?: string;
  private readonly dataUrl?: string;

  constructor(config: EmscriptenUciEngineConfig, options: Options, jsUrl: string, wasmUrl?: string, dataUrl?: string) {
    this.config = config;
    this.name = config.name;
    this.options = options;
    this.jsUrl = jsUrl;
    this.wasmUrl = wasmUrl;
    this.dataUrl = dataUrl;
  }

  setOptions(next: Options): void {
    this.options = { ...this.options, ...next };
  }

  private abortError(message = `${this.config.label} search aborted`): Error {
    return abortError(message);
  }

  private disposedError(): Error {
    return this.abortError(`${this.config.label} engine disposed`);
  }

  private runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.queueTail;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.queueTail = previous.then(
      () => gate,
      () => gate,
    );
    return previous
      .catch(() => undefined)
      .then(async () => {
        try {
          // Work queued before dispose() must not recreate (and leak) a worker.
          if (this.disposed) throw this.disposedError();
          return await fn();
        } finally {
          this.trimOutput();
          release();
        }
      });
  }

  /**
   * Line indices are only held within one exclusive section, so once it has
   * settled the transcript can be dropped instead of retaining every `info`
   * line until dispose().
   */
  private trimOutput(): void {
    if (this.lineWaiters.length) return;
    if (this.stdout.length > RETAINED_OUTPUT_LINES) this.stdout = this.stdout.slice(-RETAINED_OUTPUT_LINES);
    if (this.stderr.length > RETAINED_OUTPUT_LINES) this.stderr = this.stderr.slice(-RETAINED_OUTPUT_LINES);
  }

  private makeWorker(): Worker {
    if (this.worker) return this.worker;
    if (this.disposed) throw this.disposedError();
    this.workerObjectUrl = URL.createObjectURL(new Blob([emscriptenUciWorkerScript(this.config)], { type: 'text/javascript' }));
    const worker = new Worker(this.workerObjectUrl);
    worker.onmessage = (event: MessageEvent<WorkerResponse>) => this.handleMessage(event.data);
    worker.onerror = (event) => this.failAll(new Error(event.message || `${this.config.label} worker error`));
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
      // Initialization and synchronous UCI failures may leave Emscripten's
      // module promise or engine state poisoned. Reject all work and recreate
      // the worker on the next request rather than caching that failed state.
      this.failAll(new Error(message.message));
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

  private trustedJsUrl(): boolean {
    return isTrustedExecutableAssetUrl(this.resolvedJsUrl());
  }

  private resolvedWasmUrl(): string | undefined {
    return this.wasmUrl ? new URL(this.wasmUrl, location.href).href : undefined;
  }

  private resolvedDataUrl(): string | undefined {
    return this.dataUrl ? new URL(this.dataUrl, location.href).href : undefined;
  }

  private request(type: 'init' | 'command', payload: Record<string, unknown> = {}, signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) return Promise.reject(this.abortError());
    const worker = this.makeWorker();
    const id = this.nextId++;
    return new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        this.pending.delete(id);
        this.restartAfterAbort();
        reject(this.abortError());
      };
      const cleanup = () => signal?.removeEventListener('abort', onAbort);
      this.pending.set(id, { resolve, reject, cleanup });
      signal?.addEventListener('abort', onAbort, { once: true });
      worker.postMessage({
        type,
        id,
        jsUrl: this.resolvedJsUrl(),
        wasmUrl: this.resolvedWasmUrl(),
        dataUrl: this.resolvedDataUrl(),
        trustedJsUrl: this.trustedJsUrl(),
        ...payload,
      });
    });
  }

  private waitForLine(predicate: (line: string) => boolean, label: string, startIndex: number, signal?: AbortSignal, timeoutMs = 20000): Promise<string> {
    if (signal?.aborted) return Promise.reject(this.abortError());
    for (let i = startIndex; i < this.stdout.length; i += 1) if (predicate(this.stdout[i])) return Promise.resolve(this.stdout[i]);
    return new Promise<string>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.lineWaiters = this.lineWaiters.filter((entry) => entry !== waiter);
        waiter.cleanup?.();
        const error = new Error(`Timed out waiting for ${label}; last ${this.config.label} lines: ${this.stdout.slice(-8).join(' | ')}`);
        // A synchronous Emscripten search cannot process `stop`; terminate it
        // so the next command starts from a known-clean worker.
        this.failAll(error);
        reject(error);
      }, timeoutMs);
      const onAbort = () => {
        clearTimeout(timer);
        this.lineWaiters = this.lineWaiters.filter((entry) => entry !== waiter);
        this.restartAfterAbort();
        reject(this.abortError());
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
    this.appliedOptions.clear();
  }

  private restartAfterAbort(): void {
    this.terminateWorker();
    this.pending.clear();
    this.lineWaiters = [];
  }

  private async sendCommand(command: string, signal?: AbortSignal): Promise<void> {
    await this.request('command', { command }, signal);
  }

  private async applyOptions(signal?: AbortSignal): Promise<void> {
    // Berserk's Hash handler calls TTInit (clearing the TT) even for an
    // unchanged size, and Stormphrax's TTable::resize marks an unchanged size
    // pending so finalize clears it; PlentyChess skips equal sizes but still
    // pays a worker round trip. Cache acknowledged, normalized commands only
    // for this worker.
    const commands = new Map<string, string>();
    if (this.options.hashMb !== undefined) commands.set('Hash', hashCommand(this.options.hashMb));
    if (this.options.threads !== undefined) commands.set('Threads', threadsCommand(this.options.threads));
    for (const [name, command] of commands) {
      if (this.appliedOptions.get(name) === command) continue;
      const worker = this.worker;
      await this.sendCommand(command, signal);
      if (this.worker === worker) this.appliedOptions.set(name, command);
    }
  }

  async prewarm(signal?: AbortSignal): Promise<void> {
    return this.runExclusive(() => this.prewarmUnlocked(signal));
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
      const commandPromise = this.sendCommand(emscriptenGoCommand(this.options), signal);
      try {
        const bestLine = await this.waitForLine(
          (line) => line.startsWith('bestmove '),
          'bestmove',
          searchStart,
          signal,
          emscriptenSearchTimeoutMs(this.options),
        );
        await commandPromise;
        this.lastInfoLines = this.collectInfoLines(searchStart);
        return parseBestMove(bestLine);
      } catch (error) {
        commandPromise.catch(() => undefined);
        throw error;
      }
    });
  }

  async analyze(fen: string, opts: BrowserUciAnalysisOptions = {}): Promise<BrowserUciInfoLine[]> {
    return this.runExclusive(async () => {
      await this.prewarmUnlocked(opts.signal);
      const multipv = Math.max(1, Math.floor(opts.multipv ?? 1));
      this.lastInfoLines = [];
      await this.applyOptions(opts.signal);
      await this.sendCommand(`setoption name MultiPV value ${multipv}`, opts.signal);
      await this.sendCommand(uciPositionCommand(fen, opts.moves), opts.signal);
      const searchStart = this.stdout.length;
      const searchOptions = { ...this.options, depth: opts.depth ?? this.options.depth, movetimeMs: opts.movetimeMs };
      const commandPromise = this.sendCommand(emscriptenGoCommand(searchOptions), opts.signal);
      try {
        await this.waitForLine((line) => line.startsWith('bestmove '), 'analysis bestmove', searchStart, opts.signal, emscriptenSearchTimeoutMs(searchOptions));
        await commandPromise;
        this.lastInfoLines = this.collectInfoLines(searchStart);
        return this.lastInfo();
      } catch (error) {
        commandPromise.catch(() => undefined);
        throw error;
      }
    });
  }

  private collectInfoLines(startIndex: number): BrowserUciInfoLine[] {
    const byPv = new Map<number, BrowserUciInfoLine>();
    for (let i = startIndex; i < this.stdout.length; i += 1) {
      const parsed = parseStockfishInfo(this.stdout[i]);
      if (parsed) byPv.set(parsed.multipv, parsed);
    }
    return [...byPv.values()].sort((a, b) => a.multipv - b.multipv).map((entry) => ({ ...entry, pvUci: [...entry.pvUci] }));
  }

  lastInfo(): BrowserUciInfoLine[] {
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
    this.disposed = true;
    this.failAll(this.disposedError());
    this.stdout = [];
    this.stderr = [];
    this.lastInfoLines = [];
  }
}
