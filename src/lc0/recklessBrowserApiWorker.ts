import { WASI, File, OpenFile, ConsoleStdout, PreopenDirectory } from '@bjorn3/browser_wasi_shim';

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
  | { type: 'prewarm'; id: number; wasmUrl: string; nnueUrl?: string; hashMb?: number }
  | { type: 'new-game'; id: number; wasmUrl: string; nnueUrl?: string; hashMb?: number }
  | { type: 'search'; id: number; wasmUrl: string; nnueUrl?: string; hashMb?: number; fen: string; depth?: number; movetimeMs?: number; multipv?: number }
  | { type: 'dispose' };

type ApiState = {
  wasmUrl: string;
  nnueUrl?: string;
  exports: ApiExports;
  handle: number;
  hashMb: number;
};

let state: ApiState | null = null;
const moduleCache = new Map<string, Promise<WebAssembly.Module>>();
const nnueCache = new Map<string, Promise<ArrayBuffer>>();
const encoder = new TextEncoder();
const decoder = new TextDecoder();

function postOk(id: number, payload: Record<string, unknown> = {}): void {
  postMessage({ type: 'ok', id, ...payload });
}

function postError(id: number, error: unknown): void {
  postMessage({ type: 'error', id, error: error instanceof Error ? error.message : String(error) });
}

async function compileModule(wasmUrl: string): Promise<WebAssembly.Module> {
  const existing = moduleCache.get(wasmUrl);
  if (existing) return existing;
  const promise = fetch(wasmUrl, { cache: 'force-cache' })
    .then(async (response) => {
      if (!response.ok) throw new Error(`failed to fetch Reckless browser API module ${wasmUrl}: HTTP ${response.status}`);
      try {
        return await WebAssembly.compileStreaming(response.clone());
      } catch {
        return WebAssembly.compile(await response.arrayBuffer());
      }
    })
    .catch((error) => {
      moduleCache.delete(wasmUrl);
      throw error;
    });
  moduleCache.set(wasmUrl, promise);
  return promise;
}

async function fetchNnue(nnueUrl: string): Promise<ArrayBuffer> {
  const existing = nnueCache.get(nnueUrl);
  if (existing) return existing;
  const promise = fetch(nnueUrl, { cache: 'force-cache' })
    .then(async (response) => {
      if (!response.ok) throw new Error(`failed to fetch Reckless NNUE asset ${nnueUrl}: HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .catch((error) => {
      nnueCache.delete(nnueUrl);
      throw error;
    });
  nnueCache.set(nnueUrl, promise);
  return promise;
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

async function ensureState(wasmUrl: string, hashMb = 16, nnueUrl?: string): Promise<ApiState> {
  if (state && state.wasmUrl === wasmUrl && state.nnueUrl === nnueUrl) {
    if (state.hashMb !== hashMb) {
      check(state.exports, state.exports.reckless_api_resize_hash(state.handle, hashMb));
      state.hashMb = hashMb;
    }
    return state;
  }
  if (state) state.exports.reckless_api_free(state.handle);
  const module = await compileModule(wasmUrl);
  const wasiInstance = new WASI(
    ['reckless-browser-api'],
    [],
    [new OpenFile(new File([])), nullStdout(), nullStdout(), new PreopenDirectory('.', new Map())],
    { debug: false },
  );
  const instance = await WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasiInstance.wasiImport });
  wasiInstance.initialize(instance as WebAssembly.Instance & { exports: { memory: WebAssembly.Memory; _initialize?: () => unknown } });
  assertApiExports(instance.exports);
  const exports = instance.exports;
  const handle = nnueUrl
    ? await (async () => {
      if (!exports.reckless_api_new_with_network) throw new Error('Reckless browser API external-NNUE export missing: reckless_api_new_with_network');
      const bytes = new Uint8Array(await fetchNnue(nnueUrl));
      return withBytes(exports, bytes, (ptr, len) => exports.reckless_api_new_with_network!(hashMb, ptr, len));
    })()
    : exports.reckless_api_new(hashMb);
  if (!handle) throw new Error(globalErrorString(exports) || 'Reckless browser API returned a null engine handle');
  state = { wasmUrl, nnueUrl, exports, handle, hashMb };
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
  const api = await ensureState(message.wasmUrl, message.hashMb ?? 16, message.nnueUrl);
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

onmessage = (event: MessageEvent<ApiMessage>) => {
  void handleMessage(event.data).catch((error) => {
    const id = 'id' in event.data ? event.data.id : 0;
    postError(id, error);
  });
};
