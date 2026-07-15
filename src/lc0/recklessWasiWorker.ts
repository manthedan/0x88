import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory, Fd, wasi } from '@bjorn3/browser_wasi_shim';

/** Extra file fetched over HTTP and exposed to the engine via the WASI preopened cwd (e.g. Monty's detached networks). */
type PreopenFileSpec = { name: string; url: string };

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
const INITIAL_PREOPEN_CAPACITY = 64 * 1024;
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

function positiveSafeIntegerHeader(response: Response, name: string): number | undefined {
  const raw = response.headers.get(name);
  if (!raw || !/^\d+$/.test(raw)) return undefined;
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function responseDecodedLength(response: Response): number | undefined {
  const decodedLength = positiveSafeIntegerHeader(response, 'x-artifact-content-length');
  if (decodedLength) return decodedLength;
  const contentEncoding = response.headers.get('content-encoding')?.trim().toLowerCase();
  if (contentEncoding && contentEncoding !== 'identity') return undefined;
  return positiveSafeIntegerHeader(response, 'content-length');
}

function growPreopenBuffer(bytes: Uint8Array<ArrayBufferLike>, requiredBytes: number, loadedBytes: number): Uint8Array<ArrayBuffer> {
  let capacity = Math.max(bytes.byteLength, INITIAL_PREOPEN_CAPACITY);
  while (capacity < requiredBytes) capacity = Math.max(requiredBytes, capacity * 2);
  const grown = new Uint8Array(capacity);
  grown.set(bytes.subarray(0, loadedBytes));
  return grown;
}

export async function fetchPreopenBytes(url: string): Promise<ArrayBuffer> {
  const existing = inFlightPreopenBytes.get(url);
  if (existing) return existing;
  const request = (async () => {
      const response = await fetch(url, { cache: 'force-cache' });
      if (!response.ok) throw new Error(`failed to fetch preopen asset ${url}: HTTP ${response.status}`);
      const totalBytes = responseDecodedLength(response);
      if (!response.body) {
        const buffer = await response.arrayBuffer();
        post({ type: 'preopen-progress', url, loadedBytes: buffer.byteLength, totalBytes: buffer.byteLength });
        return buffer;
      }
      const reader = response.body.getReader();
      let bytes: Uint8Array<ArrayBuffer> = new Uint8Array(totalBytes ?? 0);
      let loadedBytes = 0;
      let lastReport = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const requiredBytes = loadedBytes + value.byteLength;
        if (requiredBytes > bytes.byteLength) bytes = growPreopenBuffer(bytes, requiredBytes, loadedBytes);
        bytes.set(value, loadedBytes);
        loadedBytes += value.byteLength;
        const now = Date.now();
        if (now - lastReport > 250) {
          lastReport = now;
          post({ type: 'preopen-progress', url, loadedBytes, totalBytes: totalBytes ?? 0 });
        }
      }
      post({ type: 'preopen-progress', url, loadedBytes, totalBytes: totalBytes ?? loadedBytes });
      const buffer = bytes.buffer as ArrayBuffer;
      return loadedBytes === bytes.byteLength ? buffer : buffer.slice(0, loadedBytes);
  })();
  inFlightPreopenBytes.set(url, request);
  try {
    return await request;
  } finally {
    if (inFlightPreopenBytes.get(url) === request) inFlightPreopenBytes.delete(url);
  }
}

async function buildPreopenDirectory(preopenFiles: PreopenFileSpec[] | undefined): Promise<PreopenDirectory> {
  const entries = new Map<string, File>();
  for (const spec of preopenFiles ?? []) {
    entries.set(spec.name, new File(await fetchPreopenBytes(spec.url)));
  }
  return new PreopenDirectory('.', entries);
}

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
