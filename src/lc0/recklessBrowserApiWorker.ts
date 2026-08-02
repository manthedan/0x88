import { ConsoleStdout, File, OpenFile, PreopenDirectory, WASI } from '@bjorn3/browser_wasi_shim';

type SearchLineJson = {
  multipv: number;
  depth: number;
  scoreCp: number | null;
  mateIn: number | null;
  nodes: number;
  nps: number;
  pv: string[];
};

type SearchResultJson = {
  bestmove: string | null;
  elapsedMs: number;
  lines: SearchLineJson[];
};

type ApiExports = WebAssembly.Exports & {
  memory: WebAssembly.Memory;
  reckless_api_alloc(len: number): number;
  reckless_api_free_bytes(ptr: number, len: number, capacity: number): void;
  reckless_api_new(hashMb: number): number;
  reckless_api_new_with_network?: (hashMb: number, ptr: number, len: number) => number;
  reckless_api_global_error_ptr?: () => number;
  reckless_api_global_error_len?: () => number;
  reckless_api_free(handle: number): void;
  reckless_api_set_fen(handle: number, ptr: number, len: number): number;
  reckless_api_set_multipv(handle: number, multiPv: number): number;
  reckless_api_resize_hash(handle: number, hashMb: number): number;
  reckless_api_new_game(handle: number): number;
  reckless_api_search_depth(handle: number, depth: number): number;
  reckless_api_search_movetime(handle: number, ms: bigint): number;
  reckless_api_result_json_ptr(handle: number): number;
  reckless_api_result_json_len(handle: number): number;
  reckless_api_error_ptr(handle: number): number;
  reckless_api_error_len(handle: number): number;
};

type ApiMessage =
  | { type: 'prewarm'; id: number; wasmUrl: string; nnueUrl?: string; nnueExpectedBytes?: number; hashMb?: number }
  | { type: 'new-game'; id: number; wasmUrl: string; nnueUrl?: string; nnueExpectedBytes?: number; hashMb?: number }
  | {
      type: 'search';
      id: number;
      wasmUrl: string;
      nnueUrl?: string;
      nnueExpectedBytes?: number;
      hashMb?: number;
      fen: string;
      depth?: number;
      movetimeMs?: number;
      multipv?: number;
    }
  | { type: 'dispose' };

type ApiState = {
  wasmUrl: string;
  nnueUrl?: string;
  nnueExpectedBytes?: number;
  exports: ApiExports;
  handle: number;
  hashMb: number;
};

let state: ApiState | null = null;
const moduleCache = new Map<string, Promise<WebAssembly.Module>>();
const nnueCache = new Map<string, Promise<ArrayBuffer>>();
const INITIAL_NNUE_CAPACITY = 64 * 1024;
export const MAX_NNUE_BYTES = 1024 * 1024 * 1024;
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function nowMs(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function postOk(id: number, payload: Record<string, unknown> = {}): void {
  postMessage({ type: 'ok', id, ...payload });
}

function postError(id: number, error: unknown): void {
  postMessage({ type: 'error', id, error: error instanceof Error ? error.message : String(error) });
}

function postStatus(id: number, phase: string, details: Record<string, unknown> = {}): void {
  postMessage({ type: 'status', id, phase, ...details });
}

async function compileModule(wasmUrl: string, id: number): Promise<WebAssembly.Module> {
  const existing = moduleCache.get(wasmUrl);
  if (existing) {
    postStatus(id, 'wasm-module-cache-hit', { url: wasmUrl });
    return existing;
  }
  const started = nowMs();
  postStatus(id, 'wasm-fetch', { url: wasmUrl });
  const promise = fetch(wasmUrl, { cache: 'force-cache' })
    .then(async (response) => {
      if (!response.ok) throw new Error(`failed to fetch Reckless browser API module ${wasmUrl}: HTTP ${response.status}`);
      postStatus(id, 'wasm-compile', { url: wasmUrl, elapsedMs: nowMs() - started });
      try {
        return await WebAssembly.compileStreaming(response);
      } catch {
        // Avoid teeing every successful large response solely to preserve an
        // exceptional MIME fallback. Refetch only when streaming compilation
        // actually fails.
        const fallback = await fetch(wasmUrl, { cache: 'force-cache' });
        if (!fallback.ok) throw new Error(`failed to refetch Reckless browser API module ${wasmUrl}: HTTP ${fallback.status}`);
        return WebAssembly.compile(await fallback.arrayBuffer());
      }
    })
    .then((module) => {
      postStatus(id, 'wasm-ready', { url: wasmUrl, elapsedMs: nowMs() - started });
      return module;
    })
    .catch((error) => {
      moduleCache.delete(wasmUrl);
      throw error;
    });
  moduleCache.set(wasmUrl, promise);
  return promise;
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
    throw new Error(`invalid expected byte length for Reckless NNUE asset ${url}: ${String(expectedBytes)}`);
  }
  if (expectedBytes > MAX_NNUE_BYTES) {
    throw new Error(`expected byte length for Reckless NNUE asset ${url} exceeds the ${MAX_NNUE_BYTES}-byte hard maximum: ${expectedBytes}`);
  }
  return expectedBytes;
}

function validatedDecodedLength(response: Response, url: string, expectedBytes: number | undefined): number | undefined {
  const header = decodedLengthHeader(response);
  if (!header) return undefined;
  const decodedBytes = parsePositiveSafeInteger(header.raw);
  if (decodedBytes === undefined) {
    throw new Error(`invalid decoded byte length metadata for Reckless NNUE asset ${url}: ${header.name}=${JSON.stringify(header.raw)}`);
  }
  if (decodedBytes > MAX_NNUE_BYTES) {
    throw new Error(`decoded byte length metadata for Reckless NNUE asset ${url} exceeds the ${MAX_NNUE_BYTES}-byte hard maximum: ${decodedBytes}`);
  }
  if (expectedBytes !== undefined && decodedBytes !== expectedBytes) {
    throw new Error(`decoded byte length metadata mismatch for Reckless NNUE asset ${url}: got ${decodedBytes}, expected ${expectedBytes}`);
  }
  return decodedBytes;
}

function nnueCacheKey(url: string, expectedBytes: number | undefined): string {
  return `${url}\n${expectedBytes ?? ''}`;
}

function nnueLengthMismatch(url: string, actualBytes: number, expectedBytes: number): Error {
  return new Error(`Reckless NNUE asset ${url} decoded byte length mismatch: got ${actualBytes}, expected ${expectedBytes}`);
}

function nnueOverflow(url: string, requiredBytes: number, limitBytes: number): Error {
  return new Error(`Reckless NNUE asset ${url} exceeds its ${limitBytes}-byte download limit: received at least ${requiredBytes} decoded bytes`);
}

function initialNnueCapacity(decodedBytes: number | undefined, limitBytes: number): number {
  return decodedBytes ?? Math.min(INITIAL_NNUE_CAPACITY, limitBytes);
}

function growNnueBuffer(bytes: Uint8Array<ArrayBufferLike>, requiredBytes: number, loadedBytes: number, limitBytes: number): Uint8Array<ArrayBuffer> {
  let capacity = Math.max(bytes.byteLength, INITIAL_NNUE_CAPACITY);
  while (capacity < requiredBytes) capacity = Math.min(limitBytes, Math.max(requiredBytes, capacity * 2));
  const grown = new Uint8Array(capacity);
  grown.set(bytes.subarray(0, loadedBytes));
  return grown;
}

export async function readNnueResponseWithProgress(
  response: Response,
  id: number,
  phase: string,
  url: string,
  started: number,
  rawExpectedBytes?: number,
): Promise<ArrayBuffer> {
  const expectedBytes = validatedExpectedBytes(url, rawExpectedBytes);
  const decodedBytes = validatedDecodedLength(response, url, expectedBytes);
  const limitBytes = expectedBytes ?? decodedBytes ?? MAX_NNUE_BYTES;
  const totalBytes = expectedBytes ?? decodedBytes ?? 0;
  if (!response.body) {
    if (expectedBytes !== undefined) throw nnueLengthMismatch(url, 0, expectedBytes);
    throw new Error(`Reckless NNUE asset ${url} returned no response body`);
  }
  const reader = response.body.getReader();
  let out: Uint8Array<ArrayBuffer> = new Uint8Array(initialNnueCapacity(decodedBytes, limitBytes));
  let loadedBytes = 0;
  let lastProgressMs = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    const required = loadedBytes + value.byteLength;
    if (required > limitBytes) {
      await reader.cancel().catch(() => undefined);
      throw nnueOverflow(url, required, limitBytes);
    }
    if (required > out.byteLength) {
      out = growNnueBuffer(out, required, loadedBytes, limitBytes);
    }
    out.set(value, loadedBytes);
    loadedBytes = required;
    const current = nowMs();
    if (current - lastProgressMs > 250 || (totalBytes > 0 && loadedBytes >= totalBytes)) {
      lastProgressMs = current;
      postStatus(id, phase, { url, loadedBytes, totalBytes, elapsedMs: current - started });
    }
  }
  if (expectedBytes !== undefined && loadedBytes !== expectedBytes) {
    throw nnueLengthMismatch(url, loadedBytes, expectedBytes);
  }
  if (decodedBytes !== undefined && loadedBytes !== decodedBytes) {
    throw nnueLengthMismatch(url, loadedBytes, decodedBytes);
  }
  const bytes = loadedBytes === out.byteLength ? out.buffer : out.buffer.slice(0, loadedBytes);
  postStatus(id, `${phase}-ready`, { url, loadedBytes, totalBytes: totalBytes || loadedBytes, elapsedMs: nowMs() - started });
  return bytes;
}

export async function fetchNnue(nnueUrl: string, id: number, rawExpectedBytes?: number): Promise<ArrayBuffer> {
  const expectedBytes = validatedExpectedBytes(nnueUrl, rawExpectedBytes);
  const cacheKey = nnueCacheKey(nnueUrl, expectedBytes);
  const existing = nnueCache.get(cacheKey);
  if (existing) {
    postStatus(id, 'nnue-memory-cache-hit', { url: nnueUrl, totalBytes: expectedBytes });
    return existing;
  }
  const started = nowMs();
  postStatus(id, 'nnue-fetch', { url: nnueUrl, totalBytes: expectedBytes });
  const request = (async () => {
    const response = await fetch(nnueUrl, { cache: 'force-cache' });
    if (!response.ok) throw new Error(`failed to fetch Reckless NNUE asset ${nnueUrl}: HTTP ${response.status}`);
    return readNnueResponseWithProgress(response, id, 'nnue-fetch', nnueUrl, started, expectedBytes);
  })();
  nnueCache.set(cacheKey, request);
  try {
    return await request;
  } finally {
    if (nnueCache.get(cacheKey) === request) nnueCache.delete(cacheKey);
  }
}

function assertApiExports(exports: WebAssembly.Exports): asserts exports is ApiExports {
  for (const name of [
    'memory',
    'reckless_api_alloc',
    'reckless_api_free_bytes',
    'reckless_api_new',
    'reckless_api_free',
    'reckless_api_set_fen',
    'reckless_api_set_multipv',
    'reckless_api_resize_hash',
    'reckless_api_new_game',
    'reckless_api_search_depth',
    'reckless_api_search_movetime',
    'reckless_api_result_json_ptr',
    'reckless_api_result_json_len',
    'reckless_api_error_ptr',
    'reckless_api_error_len',
  ]) {
    if (!(name in exports)) throw new Error(`Reckless browser API export missing: ${name}`);
  }
}

function nullStdout(): ConsoleStdout {
  return new ConsoleStdout(() => undefined);
}

async function ensureState(wasmUrl: string, hashMb = 16, nnueUrl: string | undefined, nnueExpectedBytes: number | undefined, id: number): Promise<ApiState> {
  if (state && state.wasmUrl === wasmUrl && state.nnueUrl === nnueUrl && state.nnueExpectedBytes === nnueExpectedBytes) {
    if (state.hashMb !== hashMb) {
      check(state.exports, state.exports.reckless_api_resize_hash(state.handle, hashMb));
      state.hashMb = hashMb;
    }
    postStatus(id, 'engine-state-cache-hit', { url: wasmUrl, nnueUrl, totalBytes: nnueExpectedBytes });
    return state;
  }
  if (state) {
    state.exports.reckless_api_free(state.handle);
    state = null;
  }
  const modulePromise = compileModule(wasmUrl, id);
  const nnuePromise = nnueUrl ? fetchNnue(nnueUrl, id, nnueExpectedBytes) : undefined;
  const module = await modulePromise;
  const wasiInstance = new WASI(['reckless-browser-api'], [], [new OpenFile(new File([])), nullStdout(), nullStdout(), new PreopenDirectory('.', new Map())], {
    debug: false,
  });
  postStatus(id, 'wasm-instantiate', { url: wasmUrl });
  const instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasiInstance.wasiImport });
  wasiInstance.initialize(instance as WebAssembly.Instance & { exports: { memory: WebAssembly.Memory; _initialize?: () => unknown } });
  assertApiExports(instance.exports);
  const exports = instance.exports;
  const handle = nnueUrl
    ? await (async () => {
        if (!exports.reckless_api_new_with_network) throw new Error('Reckless browser API external-NNUE export missing: reckless_api_new_with_network');
        const bytes = new Uint8Array(await nnuePromise!);
        postStatus(id, 'nnue-copy', { url: nnueUrl, loadedBytes: bytes.byteLength, totalBytes: bytes.byteLength });
        return withBytes(exports, bytes, (ptr, len) => exports.reckless_api_new_with_network!(hashMb, ptr, len));
      })()
    : exports.reckless_api_new(hashMb);
  if (!handle) throw new Error(globalErrorString(exports) || 'Reckless browser API returned a null engine handle');
  state = { wasmUrl, nnueUrl, nnueExpectedBytes, exports, handle, hashMb };
  postStatus(id, 'ready', { url: wasmUrl, nnueUrl });
  return state;
}

function readBytes(exports: ApiExports, ptr: number, len: number): string {
  if (!ptr || !len) return '';
  return decoder.decode(new Uint8Array(exports.memory.buffer, ptr, len));
}

function errorString(exports: ApiExports, handle: number): string {
  return readBytes(exports, exports.reckless_api_error_ptr(handle), exports.reckless_api_error_len(handle)) || 'Reckless browser API call failed';
}

function check(exports: ApiExports, code: number): void {
  if (code !== 0) throw new Error(errorString(exports, state?.handle ?? 0));
}

function globalErrorString(exports: ApiExports): string {
  if (!exports.reckless_api_global_error_ptr || !exports.reckless_api_global_error_len) return '';
  return readBytes(exports, exports.reckless_api_global_error_ptr(), exports.reckless_api_global_error_len());
}

function withBytes<T>(exports: ApiExports, bytes: Uint8Array, fn: (ptr: number, len: number) => T): T {
  const ptr = exports.reckless_api_alloc(bytes.byteLength);
  if (!ptr) throw new Error('Reckless browser API allocation failed');
  new Uint8Array(exports.memory.buffer, ptr, bytes.byteLength).set(bytes);
  try {
    return fn(ptr, bytes.byteLength);
  } finally {
    exports.reckless_api_free_bytes(ptr, 0, bytes.byteLength);
  }
}

function withEncodedString<T>(exports: ApiExports, value: string, fn: (ptr: number, len: number) => T): T {
  const bytes = encoder.encode(value);
  const ptr = exports.reckless_api_alloc(bytes.byteLength);
  if (!ptr) throw new Error('Reckless browser API allocation failed');
  new Uint8Array(exports.memory.buffer, ptr, bytes.byteLength).set(bytes);
  try {
    return fn(ptr, bytes.byteLength);
  } finally {
    exports.reckless_api_free_bytes(ptr, 0, bytes.byteLength);
  }
}

function readResult(exports: ApiExports, handle: number): SearchResultJson {
  const json = readBytes(exports, exports.reckless_api_result_json_ptr(handle), exports.reckless_api_result_json_len(handle));
  return JSON.parse(json) as SearchResultJson;
}

async function handleMessage(message: ApiMessage): Promise<void> {
  if (message.type === 'dispose') {
    if (state) state.exports.reckless_api_free(state.handle);
    state = null;
    return;
  }
  const api = await ensureState(message.wasmUrl, message.hashMb ?? 16, message.nnueUrl, message.nnueExpectedBytes, message.id);
  if (message.type === 'prewarm') {
    postOk(message.id);
    return;
  }
  if (message.type === 'new-game') {
    check(api.exports, api.exports.reckless_api_new_game(api.handle));
    postOk(message.id);
    return;
  }
  check(api.exports, api.exports.reckless_api_resize_hash(api.handle, message.hashMb ?? 16));
  api.hashMb = message.hashMb ?? 16;
  check(api.exports, api.exports.reckless_api_set_multipv(api.handle, Math.max(1, Math.floor(message.multipv ?? 1))));
  withEncodedString(api.exports, message.fen, (ptr, len) => check(api.exports, api.exports.reckless_api_set_fen(api.handle, ptr, len)));
  if (message.movetimeMs && message.movetimeMs > 0) {
    check(api.exports, api.exports.reckless_api_search_movetime(api.handle, BigInt(Math.floor(message.movetimeMs))));
  } else {
    check(api.exports, api.exports.reckless_api_search_depth(api.handle, Math.max(1, Math.floor(message.depth ?? 4))));
  }
  postOk(message.id, { result: readResult(api.exports, api.handle) });
}

self.onmessage = (event: MessageEvent<ApiMessage>) => {
  void handleMessage(event.data).catch((error) => {
    const id = 'id' in event.data ? event.data.id : 0;
    postError(id, error);
  });
};
