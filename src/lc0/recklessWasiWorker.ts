import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory, Fd, wasi } from '@bjorn3/browser_wasi_shim';

/** Extra file fetched over HTTP and exposed to the engine via the WASI preopened cwd (e.g. Monty's detached networks). */
type PreopenFileSpec = { name: string; url: string; expectedBytes?: number };

type OneShotWorkerRequest = {
  type: 'run';
  id: number;
  wasmUrl: string;
  executableName?: string;
  commands: string[];
  preopenFiles?: PreopenFileSpec[];
};

type PersistentWorkerRequest = {
  type: 'start-persistent';
  wasmUrl: string;
  inputBuffer: SharedArrayBuffer;
  executableName?: string;
  preopenFiles?: PreopenFileSpec[];
};

type WorkerRequest = OneShotWorkerRequest | PersistentWorkerRequest;

type WorkerResponse =
  | { type: 'result'; id: number; stdout: string[]; stderr: string[]; exitCode: number }
  | { type: 'error'; id: number; error: string }
  | { type: 'persistent-ready' }
  | { type: 'persistent-line'; stream: 'stdout' | 'stderr'; line: string }
  | { type: 'persistent-exit'; exitCode: number }
  | { type: 'persistent-error'; error: string }
  | { type: 'preopen-progress'; url: string; loadedBytes: number; totalBytes: number };

const moduleCache = new Map<string, Promise<WebAssembly.Module>>();
const inFlightPreopenBytes = new Map<string, Promise<ArrayBuffer>>();
const verifiedPreopenBytes = new Map<string, ArrayBuffer>();
const INITIAL_PREOPEN_CAPACITY = 64 * 1024;
export const MAX_PREOPEN_FILE_BYTES = 1024 * 1024 * 1024;
export const PREOPEN_BYTE_CACHE_BUDGET = 64 * 1024 * 1024;
let verifiedPreopenByteCount = 0;
let verifiedPreopenByteBudget = PREOPEN_BYTE_CACHE_BUDGET;
const SHARED_STDIN_HEADER_INTS = 4;
const SHARED_STDIN_HEADER_BYTES = SHARED_STDIN_HEADER_INTS * Int32Array.BYTES_PER_ELEMENT;

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function isUsefulUciStdoutLine(line: string): boolean {
  return line === 'uciok' || line === 'readyok' || line.startsWith('bestmove') || line.startsWith('info ');
}

function lineCollector(lines: string[] | null, onLine?: (line: string) => void, keepLine: (line: string) => boolean = () => true): ConsoleStdout {
  const decoder = new TextDecoder('utf-8', { fatal: false });
  let pending = '';
  return new ConsoleStdout((chunk) => {
    pending += decoder.decode(chunk, { stream: true });
    const split = pending.split(/\r?\n/);
    pending = split.pop() ?? '';
    for (const line of split) {
      if (!keepLine(line)) continue;
      lines?.push(line);
      onLine?.(line);
    }
  });
}

class SharedStdin extends Fd {
  private readonly control: Int32Array;
  private readonly data: Uint8Array;
  private readonly capacity: number;

  constructor(buffer: SharedArrayBuffer) {
    super();
    this.control = new Int32Array(buffer, 0, SHARED_STDIN_HEADER_INTS);
    this.data = new Uint8Array(buffer, SHARED_STDIN_HEADER_BYTES);
    this.capacity = Atomics.load(this.control, 3) || this.data.byteLength;
  }

  fd_fdstat_get(): { ret: number; fdstat: wasi.Fdstat } {
    const fdstat = new wasi.Fdstat(wasi.FILETYPE_CHARACTER_DEVICE, 0);
    fdstat.fs_rights_base = BigInt(wasi.RIGHTS_FD_READ);
    return { ret: wasi.ERRNO_SUCCESS, fdstat };
  }

  fd_filestat_get(): { ret: number; filestat: wasi.Filestat } {
    return { ret: wasi.ERRNO_SUCCESS, filestat: new wasi.Filestat(0n, wasi.FILETYPE_CHARACTER_DEVICE, 0n) };
  }

  fd_read(size: number): { ret: number; data: Uint8Array } {
    while (true) {
      const readPos = Atomics.load(this.control, 0);
      const writePos = Atomics.load(this.control, 1);
      const available = writePos - readPos;
      if (available > 0) {
        const offset = readPos % this.capacity;
        const n = Math.min(size, available, this.capacity - offset);
        const out = new Uint8Array(n);
        out.set(this.data.subarray(offset, offset + n));
        Atomics.store(this.control, 0, readPos + n);
        Atomics.notify(this.control, 0);
        return { ret: wasi.ERRNO_SUCCESS, data: out };
      }
      if (Atomics.load(this.control, 2) !== 0) return { ret: wasi.ERRNO_SUCCESS, data: new Uint8Array() };
      Atomics.wait(this.control, 1, writePos, 1000);
    }
  }
}

async function fetchAndCompileModule(wasmUrl: string): Promise<WebAssembly.Module> {
  const response = await fetch(wasmUrl);
  if (!response.ok) throw new Error(`failed to fetch WASI module ${wasmUrl}: HTTP ${response.status}`);

  // Use streaming compilation when the server advertises an acceptable wasm MIME
  // type, but keep an ArrayBuffer fallback for dev/static servers that do not.
  if (typeof WebAssembly.compileStreaming === 'function') {
    try {
      return await WebAssembly.compileStreaming(response.clone());
    } catch (error) {
      console.warn('Reckless WASM compileStreaming failed; falling back to ArrayBuffer compile', error);
    }
  }
  return WebAssembly.compile(await response.arrayBuffer());
}

async function compileModule(wasmUrl: string): Promise<WebAssembly.Module> {
  let cached = moduleCache.get(wasmUrl);
  if (!cached) {
    cached = fetchAndCompileModule(wasmUrl);
    moduleCache.set(wasmUrl, cached);
    cached.catch(() => moduleCache.delete(wasmUrl));
  }
  return cached;
}

function decodedLengthHeader(response: Response): { name: string; raw: string } | undefined {
  const decodedLength = response.headers.get('x-artifact-content-length');
  if (decodedLength !== null) return { name: 'x-artifact-content-length', raw: decodedLength };
  const contentEncoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  if (contentEncoding && contentEncoding !== 'identity') return undefined;
  const contentLength = response.headers.get('content-length');
  return contentLength === null ? undefined : { name: 'content-length', raw: contentLength };
}

function parsePositiveSafeInteger(raw: string): number | undefined {
  if (!/^\d+$/.test(raw)) return undefined;
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function validatedExpectedBytes(url: string, expectedBytes: number | undefined): number | undefined {
  if (expectedBytes === undefined) return undefined;
  if (!Number.isSafeInteger(expectedBytes) || expectedBytes <= 0) {
    throw new Error(`invalid expected byte length for preopen asset ${url}: ${String(expectedBytes)}`);
  }
  if (expectedBytes > MAX_PREOPEN_FILE_BYTES) {
    throw new Error(`expected byte length for preopen asset ${url} exceeds the ${MAX_PREOPEN_FILE_BYTES}-byte hard maximum: ${expectedBytes}`);
  }
  return expectedBytes;
}

function validatedDecodedLength(response: Response, url: string, expectedBytes: number | undefined): number | undefined {
  const header = decodedLengthHeader(response);
  if (!header) return undefined;
  const decodedBytes = parsePositiveSafeInteger(header.raw);
  if (decodedBytes === undefined) {
    throw new Error(`invalid decoded byte length metadata for preopen asset ${url}: ${header.name}=${JSON.stringify(header.raw)}`);
  }
  if (decodedBytes > MAX_PREOPEN_FILE_BYTES) {
    throw new Error(`decoded byte length metadata for preopen asset ${url} exceeds the ${MAX_PREOPEN_FILE_BYTES}-byte hard maximum: ${decodedBytes}`);
  }
  if (expectedBytes !== undefined && decodedBytes !== expectedBytes) {
    throw new Error(`decoded byte length metadata mismatch for preopen asset ${url}: got ${decodedBytes}, expected ${expectedBytes}`);
  }
  return decodedBytes;
}

function preopenLimit(expectedBytes: number | undefined, decodedBytes: number | undefined): number {
  return expectedBytes ?? decodedBytes ?? MAX_PREOPEN_FILE_BYTES;
}

function preopenProgressTotal(expectedBytes: number | undefined, decodedBytes: number | undefined): number {
  return expectedBytes ?? decodedBytes ?? 0;
}

function preopenDedupeKey(url: string, expectedBytes: number | undefined): string {
  return `${url}\n${expectedBytes ?? ''}`;
}

function cachedPreopenBytes(cacheKey: string): ArrayBuffer | undefined {
  const cached = verifiedPreopenBytes.get(cacheKey);
  if (!cached) return undefined;
  verifiedPreopenBytes.delete(cacheKey);
  verifiedPreopenBytes.set(cacheKey, cached);
  return cached;
}

function cacheVerifiedPreopenBytes(cacheKey: string, bytes: ArrayBuffer): void {
  if (bytes.byteLength === 0 || bytes.byteLength > verifiedPreopenByteBudget) return;
  const replaced = verifiedPreopenBytes.get(cacheKey);
  if (replaced) {
    verifiedPreopenBytes.delete(cacheKey);
    verifiedPreopenByteCount -= replaced.byteLength;
  }
  while (verifiedPreopenByteCount + bytes.byteLength > verifiedPreopenByteBudget) {
    const oldestKey = verifiedPreopenBytes.keys().next().value as string | undefined;
    if (oldestKey === undefined) break;
    const oldest = verifiedPreopenBytes.get(oldestKey);
    verifiedPreopenBytes.delete(oldestKey);
    verifiedPreopenByteCount -= oldest?.byteLength ?? 0;
  }
  verifiedPreopenBytes.set(cacheKey, bytes);
  verifiedPreopenByteCount += bytes.byteLength;
}

function preopenLengthMismatch(url: string, actualBytes: number, expectedBytes: number): Error {
  return new Error(`preopen asset ${url} decoded byte length mismatch: got ${actualBytes}, expected ${expectedBytes}`);
}

function preopenOverflow(url: string, requiredBytes: number, limitBytes: number): Error {
  return new Error(`preopen asset ${url} exceeds its ${limitBytes}-byte download limit: received at least ${requiredBytes} decoded bytes`);
}

function initialPreopenCapacity(decodedBytes: number | undefined, limitBytes: number): number {
  return decodedBytes ?? Math.min(INITIAL_PREOPEN_CAPACITY, limitBytes);
}

function growPreopenBuffer(bytes: Uint8Array<ArrayBufferLike>, requiredBytes: number, loadedBytes: number, limitBytes: number): Uint8Array<ArrayBuffer> {
  let capacity = Math.max(bytes.byteLength, INITIAL_PREOPEN_CAPACITY);
  while (capacity < requiredBytes) capacity = Math.min(limitBytes, Math.max(requiredBytes, capacity * 2));
  const grown = new Uint8Array(capacity);
  grown.set(bytes.subarray(0, loadedBytes));
  return grown;
}

export async function fetchPreopenBytes(url: string, rawExpectedBytes?: number): Promise<ArrayBuffer> {
  const expectedBytes = validatedExpectedBytes(url, rawExpectedBytes);
  const dedupeKey = preopenDedupeKey(url, expectedBytes);
  const cached = cachedPreopenBytes(dedupeKey);
  if (cached) {
    post({ type: 'preopen-progress', url, loadedBytes: cached.byteLength, totalBytes: expectedBytes ?? cached.byteLength });
    return cached;
  }
  const existing = inFlightPreopenBytes.get(dedupeKey);
  if (existing) return existing;
  const request = (async () => {
      const response = await fetch(url, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`failed to fetch preopen asset ${url}: HTTP ${response.status}`);
      const decodedBytes = validatedDecodedLength(response, url, expectedBytes);
      const limitBytes = preopenLimit(expectedBytes, decodedBytes);
      const totalBytes = preopenProgressTotal(expectedBytes, decodedBytes);
      if (!response.body) {
        if (expectedBytes !== undefined) throw preopenLengthMismatch(url, 0, expectedBytes);
        throw new Error(`preopen asset ${url} returned no response body`);
      }
      const reader = response.body.getReader();
      let bytes: Uint8Array<ArrayBuffer> = new Uint8Array(initialPreopenCapacity(decodedBytes, limitBytes));
      let loadedBytes = 0;
      let lastReport = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const requiredBytes = loadedBytes + value.byteLength;
        if (requiredBytes > limitBytes) {
          await reader.cancel().catch(() => undefined);
          throw preopenOverflow(url, requiredBytes, limitBytes);
        }
        if (requiredBytes > bytes.byteLength) bytes = growPreopenBuffer(bytes, requiredBytes, loadedBytes, limitBytes);
        bytes.set(value, loadedBytes);
        loadedBytes += value.byteLength;
        const now = Date.now();
        if (now - lastReport > 250) {
          lastReport = now;
          post({ type: 'preopen-progress', url, loadedBytes, totalBytes });
        }
      }
      if (expectedBytes !== undefined && loadedBytes !== expectedBytes) {
        throw preopenLengthMismatch(url, loadedBytes, expectedBytes);
      }
      if (decodedBytes !== undefined && loadedBytes !== decodedBytes) {
        throw preopenLengthMismatch(url, loadedBytes, decodedBytes);
      }
      post({ type: 'preopen-progress', url, loadedBytes, totalBytes: totalBytes || loadedBytes });
      const buffer = bytes.buffer as ArrayBuffer;
      const verified = loadedBytes === bytes.byteLength ? buffer : buffer.slice(0, loadedBytes);
      if (expectedBytes !== undefined || decodedBytes !== undefined) cacheVerifiedPreopenBytes(dedupeKey, verified);
      return verified;
  })();
  inFlightPreopenBytes.set(dedupeKey, request);
  try {
    return await request;
  } finally {
    if (inFlightPreopenBytes.get(dedupeKey) === request) inFlightPreopenBytes.delete(dedupeKey);
  }
}

async function buildPreopenDirectory(preopenFiles: PreopenFileSpec[] | undefined): Promise<PreopenDirectory> {
  const entries = new Map<string, File>();
  for (const spec of preopenFiles ?? []) {
    entries.set(spec.name, new File(await fetchPreopenBytes(spec.url, spec.expectedBytes), { readonly: true }));
  }
  return new PreopenDirectory('.', entries);
}

export const recklessWasiWorkerInternalsForTests = {
  resetPreopenByteCache(byteBudget = PREOPEN_BYTE_CACHE_BUDGET): void {
    if (!Number.isSafeInteger(byteBudget) || byteBudget < 0) throw new Error(`invalid preopen byte cache budget: ${String(byteBudget)}`);
    verifiedPreopenBytes.clear();
    verifiedPreopenByteCount = 0;
    verifiedPreopenByteBudget = byteBudget;
  },
  preopenByteCacheState(): { byteBudget: number; totalBytes: number; keys: string[]; inFlightCount: number } {
    return {
      byteBudget: verifiedPreopenByteBudget,
      totalBytes: verifiedPreopenByteCount,
      keys: [...verifiedPreopenBytes.keys()],
      inFlightCount: inFlightPreopenBytes.size,
    };
  },
};

async function runWasiUci(wasmUrl: string, executableName: string, commands: string[], preopenFiles?: PreopenFileSpec[]): Promise<{ stdout: string[]; stderr: string[]; exitCode: number }> {
  const stdout: string[] = [];
  const stderr: string[] = [];
  const wasiInstance = new WASI(
    [executableName, ...commands],
    [],
    [
      new OpenFile(new File([])),
      lineCollector(stdout, undefined, isUsefulUciStdoutLine),
      lineCollector(stderr),
      await buildPreopenDirectory(preopenFiles),
    ],
    { debug: false },
  );
  const instance = await WebAssembly.instantiate(await compileModule(wasmUrl), {
    wasi_snapshot_preview1: wasiInstance.wasiImport,
  });
  const exitCode = wasiInstance.start(instance as WebAssembly.Instance & { exports: { memory: WebAssembly.Memory; _start: () => unknown } });
  return { stdout, stderr, exitCode };
}

async function runPersistentWasiUci(wasmUrl: string, inputBuffer: SharedArrayBuffer, executableName = 'reckless', preopenFiles?: PreopenFileSpec[]): Promise<void> {
  const wasiInstance = new WASI(
    [executableName],
    [],
    [
      new SharedStdin(inputBuffer),
      lineCollector(null, (line) => post({ type: 'persistent-line', stream: 'stdout', line }), isUsefulUciStdoutLine),
      lineCollector(null, (line) => post({ type: 'persistent-line', stream: 'stderr', line })),
      await buildPreopenDirectory(preopenFiles),
    ],
    { debug: false },
  );
  const instance = await WebAssembly.instantiate(await compileModule(wasmUrl), {
    wasi_snapshot_preview1: wasiInstance.wasiImport,
  });
  post({ type: 'persistent-ready' });
  const exitCode = wasiInstance.start(instance as WebAssembly.Instance & { exports: { memory: WebAssembly.Memory; _start: () => unknown } });
  post({ type: 'persistent-exit', exitCode });
}

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  if (message.type === 'run') {
    void runWasiUci(message.wasmUrl, message.executableName ?? 'reckless', message.commands, message.preopenFiles)
      .then((result) => post({ type: 'result', id: message.id, ...result }))
      .catch((error) => post({ type: 'error', id: message.id, error: (error as Error).message }));
    return;
  }
  if (message.type === 'start-persistent') {
    void runPersistentWasiUci(message.wasmUrl, message.inputBuffer, message.executableName, message.preopenFiles)
      .catch((error) => post({ type: 'persistent-error', error: (error as Error).message }));
  }
});
