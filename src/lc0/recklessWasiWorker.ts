import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory, Fd, wasi } from '@bjorn3/browser_wasi_shim';

type OneShotWorkerRequest = {
  type: 'run';
  id: number;
  wasmUrl: string;
  commands: string[];
};

type PersistentWorkerRequest = {
  type: 'start-persistent';
  wasmUrl: string;
  inputBuffer: SharedArrayBuffer;
};

type WorkerRequest = OneShotWorkerRequest | PersistentWorkerRequest;

type WorkerResponse =
  | { type: 'result'; id: number; stdout: string[]; stderr: string[]; exitCode: number }
  | { type: 'error'; id: number; error: string }
  | { type: 'persistent-ready' }
  | { type: 'persistent-line'; stream: 'stdout' | 'stderr'; line: string }
  | { type: 'persistent-exit'; exitCode: number }
  | { type: 'persistent-error'; error: string };

const moduleCache = new Map<string, Promise<WebAssembly.Module>>();
const SHARED_STDIN_HEADER_INTS = 4;
const SHARED_STDIN_HEADER_BYTES = SHARED_STDIN_HEADER_INTS * Int32Array.BYTES_PER_ELEMENT;

function post(message: WorkerResponse): void {
  self.postMessage(message);
}

function isUsefulUciStdoutLine(line: string): boolean {
  return line === 'uciok' || line === 'readyok' || line.startsWith('bestmove') || (line.startsWith('info ') && line.includes(' pv '));
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
  if (!response.ok) throw new Error(`failed to fetch Reckless WASI module ${wasmUrl}: HTTP ${response.status}`);

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

async function runReckless(wasmUrl: string, commands: string[]): Promise<{ stdout: string[]; stderr: string[]; exitCode: number }> {
  const stdout: string[] = [];
  const stderr: string[] = [];
  const wasiInstance = new WASI(
    ['reckless', ...commands],
    [],
    [
      new OpenFile(new File([])),
      lineCollector(stdout, undefined, isUsefulUciStdoutLine),
      lineCollector(stderr),
      new PreopenDirectory('.', new Map()),
    ],
    { debug: false },
  );
  const instance = await WebAssembly.instantiate(await compileModule(wasmUrl), {
    wasi_snapshot_preview1: wasiInstance.wasiImport,
  });
  const exitCode = wasiInstance.start(instance as WebAssembly.Instance & { exports: { memory: WebAssembly.Memory; _start: () => unknown } });
  return { stdout, stderr, exitCode };
}

async function runPersistentReckless(wasmUrl: string, inputBuffer: SharedArrayBuffer): Promise<void> {
  const wasiInstance = new WASI(
    ['reckless'],
    [],
    [
      new SharedStdin(inputBuffer),
      lineCollector(null, (line) => post({ type: 'persistent-line', stream: 'stdout', line }), isUsefulUciStdoutLine),
      lineCollector(null, (line) => post({ type: 'persistent-line', stream: 'stderr', line })),
      new PreopenDirectory('.', new Map()),
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
    void runReckless(message.wasmUrl, message.commands)
      .then((result) => post({ type: 'result', id: message.id, ...result }))
      .catch((error) => post({ type: 'error', id: message.id, error: (error as Error).message }));
    return;
  }
  if (message.type === 'start-persistent') {
    void runPersistentReckless(message.wasmUrl, message.inputBuffer)
      .catch((error) => post({ type: 'persistent-error', error: (error as Error).message }));
  }
});
