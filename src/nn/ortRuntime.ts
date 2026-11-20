import './ortConsoleFilter.ts';

export * from 'onnxruntime-web/webgpu';

import * as ort from 'onnxruntime-web/webgpu';
import { trackEngineRetirement } from '../lc0/engineRetirement.ts';
import { untilAborted } from '../lc0/modelCache.ts';
import { isOnnxModelParseError, OrtSessionFallbackError, ortErrorMessage } from '../lc0/onnxModelRecovery.ts';
import { supportsWasmRelaxedSimdIntegerDot } from '../lc0/wasmFeatures.ts';
import { foldOnnxDequantizeLinear } from './onnxDequantFold.ts';

(ort.env as unknown as { logLevel?: 'fatal' }).logLevel = 'fatal';

export type OrtExecutionProviderPreference = 'wasm' | 'webgpu' | 'webgpu,wasm' | 'auto';
export type OrtWasmSimdVariant = 'bundled' | 'fixed' | 'relaxed';
export type OrtWasmArtifactSelection = {
  variant: OrtWasmSimdVariant;
  mjsUrl?: string;
  wasmUrl?: string;
  artifactId?: string;
};

/**
 * ORT ships three threaded browser runtimes. The CPU pair is always the wasm
 * EP. The GPU pair is either Asyncify or JSPI, and glue and binary must match.
 *
 * - `asyncify`: WebGPU entrypoint instrumented for stack unwinding so wasm can
 *   suspend across GPU work. ~24MB. Fallback when JSPI is unavailable, and the
 *   pair the dev server keeps because the `onnxruntime-web/webgpu` bundle
 *   names this glue.
 * - `jspi`: the same native WebGPU EP built with Wasm JSPI instead of
 *   Asyncify. ~15MB. Loaded only from a production build, and only when
 *   `WebAssembly.Suspending` and `WebAssembly.promising` exist. ORT 1.27
 *   ships the matching glue as `ort-wasm-simd-threaded.jspi.mjs` (the
 *   `onnxruntime-web/jspi` JS entry is the same orchestrator with that
 *   filename baked in). The pthread bootstrap is inside the `.jspi.mjs`.
 * - `wasm`: the CPU-only entrypoint. Not asyncify-instrumented, ~13.5MB.
 *
 * Each glue module is its own Emscripten pthread bootstrap (it spawns
 * `new Worker(new URL(import.meta.url), { name: 'em-pthread' })`), so the one
 * we hand to ORT must keep its deployed URL for helper workers to re-import.
 * The plain glue refuses to instantiate the asyncify or JSPI binary (LinkError)
 * and vice versa. `scripts/ort_runtime_assets.mjs` stages all three pairs.
 * Auto-detect does not switch pairs when a file 404s, so the JSPI glue and
 * binary have to ship with the asyncify pair.
 */
export type OrtRuntimeArtifactKind = 'asyncify' | 'jspi' | 'wasm';
export type OrtJspiPreference = 'auto' | 'on' | 'off';

export const ORT_PTHREAD_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.asyncify.mjs';
export const ORT_PTHREAD_WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';
export const ORT_JSPI_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.jspi.mjs';
export const ORT_JSPI_WASM_FILE = 'ort-wasm-simd-threaded.jspi.wasm';
export const ORT_WASM_EP_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.mjs';
export const ORT_WASM_EP_WASM_FILE = 'ort-wasm-simd-threaded.wasm';

export const ORT_RUNTIME_ARTIFACT_FILES: Readonly<Record<OrtRuntimeArtifactKind, { mjs: string; wasm: string }>> = Object.freeze({
  asyncify: { mjs: ORT_PTHREAD_BOOTSTRAP_FILE, wasm: ORT_PTHREAD_WASM_FILE },
  jspi: { mjs: ORT_JSPI_BOOTSTRAP_FILE, wasm: ORT_JSPI_WASM_FILE },
  wasm: { mjs: ORT_WASM_EP_BOOTSTRAP_FILE, wasm: ORT_WASM_EP_WASM_FILE },
});

const ORT_JSPI_UNAVAILABLE = 'WebAssembly.Suspending or WebAssembly.promising is unavailable';
const IS_PROD = (import.meta as unknown as { env?: { PROD?: boolean } }).env?.PROD === true;
const FALSY_FLAGS = new Set(['0', 'false', 'no', 'off']);
const TRUTHY_FLAGS = new Set(['1', 'true', 'yes', 'on']);

/** JSPI glue wraps imports with `Suspending` and exports with `promising`. */
export function ortJspiApiAvailable(
  wasmGlobal: { Suspending?: unknown; promising?: unknown } | null | undefined = typeof WebAssembly === 'undefined'
    ? undefined
    : (WebAssembly as { Suspending?: unknown; promising?: unknown }),
): boolean {
  return typeof wasmGlobal?.Suspending === 'function' && typeof wasmGlobal?.promising === 'function';
}

export function resolveOrtJspiPreference(raw: string | null | undefined): OrtJspiPreference {
  const normalized = String(raw ?? '')
    .trim()
    .toLowerCase();
  if (FALSY_FLAGS.has(normalized) || normalized === 'asyncify') return 'off';
  if (TRUTHY_FLAGS.has(normalized) || normalized === 'jspi') return 'on';
  return 'auto';
}

/**
 * Which GPU binary to load. A non-production bundle cannot point its glue at
 * the JSPI module (the dev server keeps the asyncify import), so JSPI is a
 * built-client choice. Missing JSPI is always asyncify, including an explicit
 * `on`, so a browser without the API still boots.
 */
export function gpuOrtRuntimeArtifactForThread(env: { prod: boolean; jspiApiAvailable: boolean; preference: OrtJspiPreference }): 'jspi' | 'asyncify' {
  return env.prod && env.preference !== 'off' && env.jspiApiAvailable ? 'jspi' : 'asyncify';
}

export function coerceOrtRuntimeArtifactKind(kind: OrtRuntimeArtifactKind, jspiApiAvailable: boolean): { kind: OrtRuntimeArtifactKind; jspiFallback?: string } {
  if (kind === 'jspi' && !jspiApiAvailable) return { kind: 'asyncify', jspiFallback: ORT_JSPI_UNAVAILABLE };
  return { kind };
}

export function resolveOrtPthreadRuntimeUrls(
  runtimeBase = '/ort/',
  pageHref = typeof location === 'undefined' ? 'http://localhost/' : location.href,
  kind: OrtRuntimeArtifactKind = 'asyncify',
): { mjs: string; wasm: string } {
  const base = new URL(runtimeBase.endsWith('/') ? runtimeBase : `${runtimeBase}/`, pageHref);
  const files = ORT_RUNTIME_ARTIFACT_FILES[kind];
  return {
    mjs: new URL(files.mjs, base).href,
    wasm: new URL(files.wasm, base).href,
  };
}

export type OrtSessionAttempt = {
  at: string;
  providers: string[];
  ok: boolean;
  ms: number;
  error?: string;
};

export type OrtRuntimeFallback = {
  at: string;
  from: 'webgpu';
  to: 'wasm';
  reason: string;
};

export type OrtWebGpuAdapterDiagnostics = {
  ok: boolean;
  summary?: string;
  info?: unknown;
  features?: string[];
  /** gpuAdapterShaderF16 of the probed adapter; absent when unknowable. */
  shaderF16?: boolean;
  limits?: Record<string, number>;
  error?: string;
};

export type OrtWebGpuProfilingSummary = {
  enabled: boolean;
  status: 'disabled' | 'unavailable' | 'available';
  reason?: string;
  /** Callback events received, not a count of all GPU kernels executed. */
  eventCount: number;
  /** Null when no callback events were received; never interpret missing data as zero GPU cost. */
  kernelGpuMsTotal: number | null;
  topPrograms: Array<{ programName: string; count: number; gpuMs: number }> | null;
};

export type OrtWebGpuApiInstrumentationSummary = {
  enabled: boolean;
  installed: boolean;
  errors: string[];
  submitCount: number;
  submittedCommandBufferCount: number;
  mapAsyncCount: number;
  mapAsyncMsTotal: number;
  copyBufferToBufferCount: number;
  copyBufferToBufferBytes: number;
  createBufferCount: number;
  createBufferBytes: number;
  mapReadBufferCount: number;
  mapReadBufferBytes: number;
  computePipelineCreateCount: number;
  computePipelineCreateAsyncCount: number;
};

export type OrtWebGpuDiagnosticsSnapshot = {
  profiling: OrtWebGpuProfilingSummary;
  api: OrtWebGpuApiInstrumentationSummary;
};

export type OrtGraphOptimizationLevel = NonNullable<ort.InferenceSession.SessionOptions['graphOptimizationLevel']>;
export type OrtSessionDiagnostics = {
  /** LC0 worker's effective IO, when available (not merely the requested mode). */
  ioMode?: 'cpu' | 'gpu' | 'gpu-capture';
  fixedBatchSize?: number;
  graphOptimizationLevel: OrtGraphOptimizationLevel;
  dequantFoldEnabled: boolean;
  dequantFold?: OrtDequantFoldSummary;
};

export type OrtRuntimeDiagnostics = OrtSessionDiagnostics & {
  requestedEp: OrtExecutionProviderPreference;
  resolvedExecutionProviders: string[];
  describe: string;
  webgpuAvailable: boolean;
  secureContext?: boolean;
  crossOriginIsolated?: boolean;
  userAgent?: string;
  wasm: {
    numThreads?: number;
    proxy?: boolean;
    sharedArrayBuffer?: boolean;
    threadedAvailable?: boolean;
    simdVariant: OrtWasmSimdVariant;
    /** Which staged ORT runtime pair this thread loads (see OrtRuntimeArtifactKind). */
    runtimeArtifact: OrtRuntimeArtifactKind;
    jspiApiAvailable?: boolean;
    jspiPreference?: OrtJspiPreference;
    /** Set when JSPI was requested and this thread fell back to asyncify. */
    jspiFallback?: string;
    artifactId?: string;
    mjsUrl?: string;
    wasmUrl?: string;
    relaxedIntegerDotAvailable: boolean;
  };
  webgpuEnv?: { powerPreference?: string; profilingMode?: string; preferredOutputLocation?: string; apiInstrumentation?: boolean };
  adapter?: OrtWebGpuAdapterDiagnostics;
  sessions: { created: number; released: number; active: number };
  sessionAttempts: OrtSessionAttempt[];
  fallback?: OrtRuntimeFallback;
  webgpuDiagnostics?: OrtWebGpuDiagnosticsSnapshot;
};

export type OrtDequantFoldSummary = {
  enabled: boolean;
  foldedNodes: number;
  skippedNodes: number;
  removedInitializers: number;
  bytesBefore: number;
  bytesAfter: number;
  elapsedMs: number;
  /** Set when ORT rejected the folded model and the session was created from the original bytes. */
  discardedAfter?: string;
};

function browserParam(name: string): string | null {
  try {
    if (typeof location === 'undefined') return null;
    return new URLSearchParams(location.search).get(name);
  } catch {
    return null;
  }
}

function envValue(name: string): string | undefined {
  return globalThis.process?.env?.[name];
}

function debugParam(name: string): string | null {
  return browserParam(name) ?? envValue(`TINY_LEELA_${name.toUpperCase()}`) ?? null;
}

export function tinyLeelaDebugEnabled(area = 'latency'): boolean {
  const normalizedArea = area.toLowerCase();
  const direct = debugParam(`debug${normalizedArea[0].toUpperCase()}${normalizedArea.slice(1)}`) ?? debugParam(normalizedArea);
  if (direct !== null) return !FALSY_FLAGS.has(direct.toLowerCase());
  const tokens = [debugParam('debug'), debugParam('tlDebug')].flatMap((value) => (value ?? '').toLowerCase().split(/[,+\s]+/));
  return tokens.some((token) => TRUTHY_FLAGS.has(token) || token === 'all' || token === 'perf' || token === 'timing' || token === normalizedArea);
}

export function tinyLeelaNowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

export function tinyLeelaLogLatency(label: string, payload: Record<string, unknown>): void {
  if (!tinyLeelaDebugEnabled('latency')) return;
  console.info(
    `Centipawn latency: ${label}`,
    Object.fromEntries(Object.entries(payload).map(([key, value]) => [key, typeof value === 'number' ? Number(value.toFixed(2)) : value])),
  );
}

function diagFlagEnabled(disableNames: string[], enableNames: string[]): boolean {
  if (
    disableNames.some((name) => {
      const value = debugParam(name);
      return value !== null && FALSY_FLAGS.has(value.toLowerCase());
    })
  ) {
    return false;
  }
  return enableNames.some((name) => {
    const value = debugParam(name);
    return value !== null && !FALSY_FLAGS.has(value.toLowerCase());
  });
}

let forcedOrtExecutionProvider: OrtExecutionProviderPreference | null = null;
let forcedOrtDiagnosticsOptions: OrtRuntimeDiagnosticOptions | null = null;
let forcedOrtWasmArtifact: OrtWasmArtifactSelection = { variant: 'bundled' };
let forcedOrtWasmThreads: number | null = null;
let lockedOrtWasmArtifactKey: string | null = null;
let forcedOrtRuntimeArtifactKind: OrtRuntimeArtifactKind | null = null;
let lockedOrtRuntimeArtifactKind: OrtRuntimeArtifactKind | null = null;

export type OrtRuntimeDiagnosticOptions = {
  graphOptimizationLevel?: OrtGraphOptimizationLevel;
  /** Request JSEP kernel timestamp callbacks (unavailable on the native WebGPU EP in ORT 1.27). */
  webgpuProfiling?: boolean;
  /** Wrap browser WebGPU APIs before ORT initializes to count submits/maps/copies. */
  webgpuApiInstrumentation?: boolean;
  /** Ask ORT WebGPU to return GPU-backed outputs so tensor.getData() can be timed separately. */
  preferredOutputLocation?: 'cpu' | 'cpu-pinned' | 'gpu-buffer';
};

export function setRequestedOrtExecutionProviderForCurrentThread(value: OrtExecutionProviderPreference | null): void {
  forcedOrtExecutionProvider = value;
}

export function validateOrtWasmArtifactSelection(selection: OrtWasmArtifactSelection): OrtWasmArtifactSelection {
  const customPathCount = Number(Boolean(selection.mjsUrl)) + Number(Boolean(selection.wasmUrl));
  if (customPathCount === 1) throw new Error('custom ORT WASM artifacts require matching mjsUrl and wasmUrl');
  if (selection.variant === 'bundled' && customPathCount > 0) {
    throw new Error('the bundled ORT WASM variant cannot specify custom artifact URLs');
  }
  if (selection.variant !== 'bundled' && customPathCount !== 2) {
    throw new Error(`${selection.variant} ORT WASM variant requires matching mjsUrl and wasmUrl`);
  }
  return { ...selection };
}

export function setRequestedOrtWasmArtifactForCurrentThread(value: OrtWasmArtifactSelection | null): void {
  const next = value ? validateOrtWasmArtifactSelection(value) : ({ variant: 'bundled' } as const);
  if (lockedOrtWasmArtifactKey && ortWasmArtifactKey(next) !== lockedOrtWasmArtifactKey) {
    throw new Error('ORT WASM is already initialized with a different artifact; select the fallback in a fresh worker');
  }
  forcedOrtWasmArtifact = next;
}

export function requestedOrtWasmArtifact(): OrtWasmArtifactSelection {
  return { ...forcedOrtWasmArtifact };
}

export function setRequestedOrtWasmThreadsForCurrentThread(value: number | null): void {
  if (value !== null && (!Number.isInteger(value) || value < 1)) {
    throw new Error(`ORT WASM thread count must be a positive integer, received ${value}`);
  }
  forcedOrtWasmThreads = value;
}

function ortWasmArtifactKey(selection: OrtWasmArtifactSelection): string {
  return JSON.stringify([selection.variant, selection.mjsUrl ?? '', selection.wasmUrl ?? '', selection.artifactId ?? '']);
}

function lockRequestedOrtWasmArtifact(): void {
  const key = ortWasmArtifactKey(forcedOrtWasmArtifact);
  if (lockedOrtWasmArtifactKey && lockedOrtWasmArtifactKey !== key) {
    throw new Error('ORT WASM artifact selection changed after initialization; create a fresh worker');
  }
  lockedOrtWasmArtifactKey = key;
}

export function setOrtRuntimeDiagnosticOptionsForCurrentThread(options: OrtRuntimeDiagnosticOptions | null): void {
  forcedOrtDiagnosticsOptions = options;
}

let forcedOrtDequantFold: boolean | null = null;
let lastOrtDequantFold: OrtDequantFoldSummary | null = null;

/** Worker threads cannot see the page URL, so the page forwards its choice explicitly. */
export function setOrtDequantFoldForCurrentThread(value: boolean | null): void {
  forcedOrtDequantFold = value;
}

/**
 * Resolve an explicit model policy (Small/default callers keep folding). A worker pin
 * wins over the page/environment override, which wins over the model's default.
 * `?ortDequantFold=1/0` can force either arm for benchmarking on every model.
 */
export function ortDequantFoldEnabled(defaultEnabled = true): boolean {
  if (forcedOrtDequantFold !== null) return forcedOrtDequantFold;
  const raw = String(browserParam('ortDequantFold') ?? envValue('ORT_DEQUANT_FOLD') ?? '')
    .trim()
    .toLowerCase();
  if (FALSY_FLAGS.has(raw)) return false;
  if (TRUTHY_FLAGS.has(raw)) return true;
  return defaultEnabled;
}

export function lastOrtDequantFoldSummary(): OrtDequantFoldSummary | null {
  return lastOrtDequantFold;
}

function runningUnderNode(): boolean {
  return typeof globalThis.process?.versions?.node === 'string';
}

type OrtModelInput = string | Uint8Array | ArrayBuffer;

type PreparedOrtModelInput = {
  input: OrtModelInput;
  /** Lazy verified reload when available; raw input retained only for callers without reload provenance. */
  original?: OrtModelSource;
};

async function prepareOrtModelInput(modelPath: OrtModelInput, reloadModel?: () => Promise<OrtModelInput>): Promise<PreparedOrtModelInput> {
  if (!ortDequantFoldEnabled()) {
    lastOrtDequantFold = { enabled: false, foldedNodes: 0, skippedNodes: 0, removedInitializers: 0, bytesBefore: 0, bytesAfter: 0, elapsedMs: 0 };
    return { input: modelPath };
  }
  let bytes: Uint8Array;
  if (typeof modelPath === 'string') {
    // ORT reads file paths itself under Node; in a browser the string is a URL and fetching it
    // here is what ORT would have done anyway.
    if (runningUnderNode() || typeof fetch !== 'function') return { input: modelPath };
    try {
      const response = await fetch(modelPath);
      if (!response.ok) return { input: modelPath };
      bytes = new Uint8Array(await response.arrayBuffer());
    } catch {
      return { input: modelPath };
    }
  } else {
    bytes = modelPath instanceof Uint8Array ? modelPath : new Uint8Array(modelPath);
  }
  try {
    const result = foldOnnxDequantizeLinear(bytes);
    lastOrtDequantFold = {
      enabled: true,
      foldedNodes: result.foldedNodes,
      skippedNodes: result.skippedNodes,
      removedInitializers: result.removedInitializers,
      bytesBefore: result.bytesBefore,
      bytesAfter: result.bytesAfter,
      elapsedMs: result.elapsedMs,
    };
    return result.foldedNodes > 0 ? { input: result.bytes, original: reloadModel ?? bytes } : { input: bytes };
  } catch (err) {
    console.warn(`Centipawn: ORT dequantize fold skipped: ${err instanceof Error ? err.message : String(err)}`);
    lastOrtDequantFold = {
      enabled: true,
      foldedNodes: 0,
      skippedNodes: 0,
      removedInitializers: 0,
      bytesBefore: bytes.byteLength,
      bytesAfter: bytes.byteLength,
      elapsedMs: 0,
    };
    return { input: bytes };
  }
}

function discardDequantFold(reason: string): void {
  if (!lastOrtDequantFold) return;
  lastOrtDequantFold = { ...lastOrtDequantFold, foldedNodes: 0, removedInitializers: 0, bytesAfter: lastOrtDequantFold.bytesBefore, discardedAfter: reason };
}

function describeDequantFold(): string | undefined {
  const fold = lastOrtDequantFold;
  if (!fold?.enabled || fold.foldedNodes === 0) return undefined;
  return `dequant fold: ${fold.foldedNodes} DequantizeLinear folded (${(fold.bytesBefore / 1e6).toFixed(1)} -> ${(fold.bytesAfter / 1e6).toFixed(1)} MB, ${fold.elapsedMs.toFixed(0)} ms)`;
}

function normalizeEp(value: string | null | undefined): OrtExecutionProviderPreference {
  const raw = String(value ?? '')
    .trim()
    .toLowerCase();
  if (!raw || raw === 'auto') return 'auto';
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'webgpu,wasm' || raw === 'webgpu+wasm' || raw === 'gpu,wasm' || raw === 'gpu+wasm') return 'webgpu,wasm';
  return 'wasm';
}

/** The EP requested by the URL/env of this document or worker, ignoring per-call overrides. */
function ambientOrtExecutionProviderParam(): string | null | undefined {
  return (
    browserParam('ortEp') ?? browserParam('ep') ?? browserParam('executionProviders') ?? envValue('TINY_LEELA_ORT_EP') ?? envValue('ORT_EXECUTION_PROVIDERS')
  );
}

export function requestedOrtExecutionProvider(): OrtExecutionProviderPreference {
  return forcedOrtExecutionProvider ?? normalizeEp(ambientOrtExecutionProviderParam());
}

function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

let probedWebGpuAdapterUsable: boolean | null = null;
/** null = not probed or the adapter does not report features; false = adapter lacks shader-f16. */
let probedWebGpuShaderF16: boolean | null = null;

/**
 * Whether the adapter reports `shader-f16`. null when that is unknowable (no
 * adapter, no `features` set-like, a throwing getter): unknown keeps WebGPU.
 */
export function gpuAdapterShaderF16(adapter: unknown): boolean | null {
  if (!adapter || typeof adapter !== 'object') return null;
  try {
    const features = (adapter as { features?: { has?: (name: string) => boolean } }).features;
    return typeof features?.has === 'function' ? features.has('shader-f16') === true : null;
  } catch {
    return null;
  }
}

/**
 * The WebGPU EP is used only when the probed adapter can run f16 shaders.
 * The LC0 nets (and Maia3's fp16 fallback) are f16, and ORT compiles shaders
 * lazily: without `shader-f16` the session is created fine and the first run
 * fails ("Program Transpose requires f16 but the device does not support it"),
 * after createOrtSession's WASM fallback can no longer help. So an adapter
 * that reports no shader-f16 is treated like no adapter: auto and
 * webgpu,wasm resolve to WASM. f32-only models lose WebGPU on such devices too,
 * which is the safe side of that trade.
 */
function webgpuUsableForProviderSelection(): boolean {
  return webgpuAvailable() && probedWebGpuAdapterUsable !== false && probedWebGpuShaderF16 !== false;
}

export function resolvedOrtExecutionProviders(requested = requestedOrtExecutionProvider()): string[] {
  if (requested !== 'webgpu' && lockedOrtRuntimeArtifactKind === 'wasm') return ['wasm'];
  return selectOrtExecutionProviders(requested, webgpuUsableForProviderSelection());
}

function selectOrtExecutionProviders(requested: OrtExecutionProviderPreference, webgpuUsable: boolean): string[] {
  if (requested === 'webgpu') return ['webgpu'];
  if (requested !== 'wasm' && webgpuUsable) return ['webgpu', 'wasm'];
  return ['wasm'];
}

/** True when this document/worker was started with an explicit wasm-only EP pin. */
function explicitWasmOnlyOrtEpPin(): boolean {
  const raw = ambientOrtExecutionProviderParam();
  return raw !== null && raw !== undefined && String(raw).trim() !== '' && normalizeEp(raw) === 'wasm';
}

/**
 * Pin the ORT runtime pair for this thread. ORT's wasm artifact is
 * worker-global once initialized, so this is only honoured before the first
 * session is created; afterwards the locked kind wins and a conflicting pin
 * throws.
 */
export function setOrtRuntimeArtifactKindForCurrentThread(kind: OrtRuntimeArtifactKind | null): void {
  if (kind && lockedOrtRuntimeArtifactKind && kind !== lockedOrtRuntimeArtifactKind) {
    throw new Error(`ORT WASM is already initialized with the ${lockedOrtRuntimeArtifactKind} runtime artifact; select the ${kind} runtime in a fresh worker`);
  }
  forcedOrtRuntimeArtifactKind = kind;
}

/**
 * Whether this thread has already committed to a runtime pair. Callers that
 * want to *opportunistically* pin (a worker re-initialized with a different EP,
 * where the binary is already loaded and cannot change) use this to skip the
 * pin instead of taking the throw above.
 */
export function ortRuntimeArtifactKindIsLocked(): boolean {
  return lockedOrtRuntimeArtifactKind !== null;
}

/**
 * Which staged runtime pair this thread loads. Asyncify is required by the
 * WebGPU/JSEP path and is pure download + instrumentation overhead for
 * CPU-only sessions, so wasm-EP-only threads take the smaller CPU build.
 *
 * The choice must be made before ORT initializes and can never change
 * afterwards (the binary is worker-global), which is why "this session happens
 * to run on wasm" is not sufficient on its own:
 * - strict `ep=webgpu` keeps a GPU pair (JSPI on a production build that can
 *   use it, otherwise asyncify), so a GPU-less browser still fails with
 *   "WebGPU unavailable" instead of an opaque link error;
 * - anything that resolves to include `webgpu` keeps that same GPU pair, which
 *   also keeps the in-process WebGPU->WASM fallback in `createOrtSession` on
 *   one binary;
 * - a CPU-only resolution takes the small build only when escalating to WebGPU
 *   later in this same thread is impossible (`navigator.gpu` missing, i.e.
 *   Safari/Firefox) or has been ruled out by an explicit `?ep=wasm` / env pin.
 *   A programmatic `setRequestedOrtExecutionProviderForCurrentThread('wasm')`
 *   deliberately does NOT qualify: search workers switch EP per message and
 *   must keep the GPU-capable binary. Such a worker can still opt in via
 *   `setOrtRuntimeArtifactKindForCurrentThread('wasm')` before its first
 *   session.
 */
let ortJspiFallbackReason: string | undefined;

function requestedOrtJspiPreference(): OrtJspiPreference {
  return resolveOrtJspiPreference(browserParam('ortJspi') ?? envValue('ORT_JSPI') ?? envValue('TINY_LEELA_ORT_JSPI'));
}

export function ortRuntimeArtifactKindForCurrentThread(): OrtRuntimeArtifactKind {
  if (lockedOrtRuntimeArtifactKind) return lockedOrtRuntimeArtifactKind;
  // Node (no `location`) imports ORT's bundle with the Emscripten glue inlined,
  // which is the asyncify one; configureOrtRuntime feeds it the matching
  // binary and wasmPaths are never consulted, so the kind is fixed there.
  if (typeof location === 'undefined') return 'asyncify';
  // Checked BEFORE the forced kind: experimental builds carry explicit URLs and
  // are asyncify binaries, so reporting 'wasm' here would enable multi-threading
  // on asyncify (a measured 55% regression).
  if (forcedOrtWasmArtifact.variant !== 'bundled') return 'asyncify';
  if (forcedOrtRuntimeArtifactKind) {
    const coerced = coerceOrtRuntimeArtifactKind(forcedOrtRuntimeArtifactKind, ortJspiApiAvailable());
    ortJspiFallbackReason = coerced.jspiFallback;
    return coerced.kind;
  }
  if (
    requestedOrtExecutionProvider() !== 'webgpu' &&
    !resolvedOrtExecutionProviders().includes('webgpu') &&
    (!webgpuUsableForProviderSelection() || explicitWasmOnlyOrtEpPin())
  ) {
    return 'wasm';
  }
  const preference = requestedOrtJspiPreference();
  const available = ortJspiApiAvailable();
  const kind = gpuOrtRuntimeArtifactForThread({ prod: IS_PROD, jspiApiAvailable: available, preference });
  ortJspiFallbackReason = kind === 'asyncify' && preference !== 'off' && !available ? ORT_JSPI_UNAVAILABLE : undefined;
  return kind;
}

function lockOrtRuntimeArtifactKind(providers: string[]): OrtRuntimeArtifactKind {
  const kind = ortRuntimeArtifactKindForCurrentThread();
  if (kind === 'wasm' && providers.includes('webgpu')) {
    throw new Error('ORT WASM is initialized with the CPU-only artifact, which has no WebGPU support; request WebGPU in a fresh worker');
  }
  lockedOrtRuntimeArtifactKind = kind;
  return kind;
}

let lastOrtExecutionProviders: string[] | null = null;
const sessionAttempts: OrtSessionAttempt[] = [];
let lastOrtRuntimeFallback: OrtRuntimeFallback | undefined;
let createdOrtSessions = 0;
let releasedOrtSessions = 0;

export function describeOrtBackendConfig(): string {
  const requested = requestedOrtExecutionProvider();
  const resolved = (lastOrtExecutionProviders ?? resolvedOrtExecutionProviders()).join(',');
  const backend = requested === 'wasm' && resolved === 'wasm' ? 'wasm' : `${requested}->${resolved}`;
  const artifact = requestedOrtWasmArtifact();
  if (artifact.variant === 'bundled') return backend;
  return `${backend}[${artifact.variant}:${artifact.artifactId ?? 'custom'}]`;
}

function browserThreadedWasmAvailable(): boolean {
  if (runningUnderNode()) return true;
  return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined';
}

export type OrtWasmThreadContext = {
  /** Explicit request (`?ortThreads=`, env pin); null/'' means "use the default". */
  raw: string | number | null | undefined;
  isBrowserMainThread: boolean;
  isNode: boolean;
  /** A bundled worker chunk from a production build (not the dev server). */
  builtWorker: boolean;
  /** This thread is committed to the CPU-only runtime pair (no WebGPU possible). */
  cpuOnlyRuntimeArtifact: boolean;
  /** Cross-origin isolated + SharedArrayBuffer, i.e. threaded wasm can run at all. */
  threadedAvailable: boolean;
  autoThreads: number;
};

/**
 * Pure thread-budget policy, exported so `tests/ort_fallback_policy.test.mjs`
 * can pin it without a browser. 0 means "leave ORT's own default in place".
 */
export function resolveOrtWasmThreads(context: OrtWasmThreadContext): number {
  const raw = context.raw === undefined || context.raw === null || String(context.raw).trim() === '' ? null : context.raw;
  const autoBudget = context.cpuOnlyRuntimeArtifact && context.threadedAvailable ? context.autoThreads : 1;
  if (raw === null) {
    // Main thread and Node stay single-threaded; dev workers keep ORT's default;
    // built workers only fan out on the CPU-only pair (asyncify+threads regressed 55%).
    if (context.isBrowserMainThread || context.isNode) return 1;
    return context.builtWorker ? autoBudget : 0;
  }
  if (String(raw).toLowerCase() === 'auto') return autoBudget;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return 0;
  const requested = Math.floor(parsed);
  return context.isBrowserMainThread && requested > 1 && !context.threadedAvailable ? 1 : requested;
}

function requestedOrtWasmThreads(isBrowserMainThread: boolean, isNode: boolean): number {
  if (forcedOrtWasmThreads !== null) {
    return isBrowserMainThread && forcedOrtWasmThreads > 1 && !browserThreadedWasmAvailable() ? 1 : forcedOrtWasmThreads;
  }
  // Cap auto threads at 4 and half the machine: ORT is outside the CPU broker
  // (`src/lc0/resourceBroker.ts`), and 4 was the measured optimum on 10 cores.
  const hc = typeof navigator === 'undefined' ? 2 : Number(navigator.hardwareConcurrency ?? 2);
  const cores = Math.floor(Number.isFinite(hc) ? hc : 2);
  return resolveOrtWasmThreads({
    raw: browserParam('ortThreads') ?? browserParam('wasmThreads') ?? envValue('ORT_INTRA_OP_NUM_THREADS') ?? envValue('ORT_NUM_THREADS') ?? null,
    isBrowserMainThread,
    isNode,
    builtWorker: !isBrowserMainThread && !isNode && IS_PROD,
    cpuOnlyRuntimeArtifact: ortRuntimeArtifactKindForCurrentThread() === 'wasm',
    threadedAvailable: browserThreadedWasmAvailable(),
    autoThreads: Math.max(1, Math.min(4, Math.floor(cores / 2))),
  });
}

/** Worker init pins this option explicitly because its location does not contain the page query. */
export function requestedOrtGraphOptimizationLevel(): OrtGraphOptimizationLevel {
  const value = forcedOrtDiagnosticsOptions?.graphOptimizationLevel ?? browserParam('ortGraphOpt');
  return value === 'disabled' || value === 'basic' || value === 'extended' || value === 'layout' || value === 'all' ? value : 'all';
}

/** Effective page/thread diagnostics forwarded to dedicated ORT workers. */
export function requestedOrtRuntimeDiagnosticOptions(): OrtRuntimeDiagnosticOptions {
  return {
    graphOptimizationLevel: requestedOrtGraphOptimizationLevel(),
    preferredOutputLocation: requestedOrtPreferredOutputLocation(),
    webgpuProfiling: requestedOrtWebGpuProfiling(),
    webgpuApiInstrumentation: requestedOrtWebGpuApiInstrumentation(),
  };
}

export function collectOrtSessionDiagnostics(): OrtSessionDiagnostics {
  return {
    graphOptimizationLevel: requestedOrtGraphOptimizationLevel(),
    dequantFoldEnabled: ortDequantFoldEnabled(),
    ...(lastOrtDequantFold ? { dequantFold: { ...lastOrtDequantFold } } : {}),
  };
}

function requestedOrtPreferredOutputLocation(): OrtRuntimeDiagnosticOptions['preferredOutputLocation'] | undefined {
  if (forcedOrtDiagnosticsOptions?.preferredOutputLocation) return forcedOrtDiagnosticsOptions.preferredOutputLocation;
  const raw = browserParam('ortPreferredOutputLocation') ?? browserParam('preferredOutputLocation');
  if (raw === 'gpu-buffer' || raw === 'cpu-pinned' || raw === 'cpu') return raw;
  return diagFlagEnabled(['ortGpuOutputs'], ['ortGpuOutputs', 'ortReadbackProfile']) ? 'gpu-buffer' : undefined;
}

function requestedOrtWebGpuProfiling(): boolean {
  if (forcedOrtDiagnosticsOptions?.webgpuProfiling !== undefined) return forcedOrtDiagnosticsOptions.webgpuProfiling;
  return diagFlagEnabled(['ortWebGpuProfile', 'ortKernelProfile'], ['ortWebGpuProfile', 'ortKernelProfile', 'ortReadbackProfile']);
}

function requestedOrtWebGpuApiInstrumentation(): boolean {
  if (forcedOrtDiagnosticsOptions?.webgpuApiInstrumentation !== undefined) return forcedOrtDiagnosticsOptions.webgpuApiInstrumentation;
  return diagFlagEnabled(['ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace'], ['ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace', 'ortReadbackProfile']);
}

const ORT_LOG_LEVELS = ['verbose', 'info', 'warning', 'error', 'fatal'] as const;

function requestedOrtLogSeverityLevel(): 0 | 1 | 2 | 3 | 4 {
  const raw = browserParam('ortLogSeverity') ?? browserParam('ortLogLevel') ?? envValue('ORT_LOG_SEVERITY_LEVEL') ?? envValue('ORT_LOG_LEVEL');
  const normalized = String(raw ?? '')
    .trim()
    .toLowerCase();
  if (!normalized || normalized === 'fatal') return 4;
  if (normalized === 'verbose') return 0;
  if (normalized === 'info') return 1;
  if (normalized === 'warning' || normalized === 'warn') return 2;
  const parsed = Number(normalized);
  if (Number.isFinite(parsed)) {
    const value = Math.max(0, Math.min(4, Math.floor(parsed)));
    if (value === 0 || value === 1 || value === 2 || value === 3 || value === 4) return value;
  }
  return 4;
}

type OrtWebGpuProfileRecord = { programName: string; kernelName: string; kernelType: string; gpuMs: number };
const ortWebGpuProfileRecords: OrtWebGpuProfileRecord[] = [];
let ortWebGpuProfileEventCount = 0;
let ortWebGpuProfileTotalMs = 0;
let ortWebGpuProfilingConfigured = false;
let previousOrtWebGpuProfilingOnData: ((data: unknown) => void) | undefined;

function webGpuProfilingSummary(
  enabled: boolean,
  eventCount: number,
  totalMs: number,
  topPrograms: NonNullable<OrtWebGpuProfilingSummary['topPrograms']>,
): OrtWebGpuProfilingSummary {
  // ORT 1.27's only ondata invocation is in lib/wasm/jsep/backend-webgpu.ts.
  // Native initEp (wasm-core-impl.ts) does not wire it. No events is missing
  // attribution, not zero GPU work or evidence of a WASM fallback. Real events
  // remain authoritative, including previously collected events after disabling.
  const status = eventCount > 0 ? 'available' : enabled ? 'unavailable' : 'disabled';
  return {
    enabled,
    status,
    ...(status === 'available'
      ? {}
      : {
          reason:
            status === 'disabled'
              ? 'ORT WebGPU kernel profiling is disabled'
              : 'unavailable on native WebGPU EP: ORT 1.27 profiling.ondata is JSEP-only; no kernel events received',
        }),
    eventCount,
    kernelGpuMsTotal: eventCount > 0 ? Number(totalMs.toFixed(6)) : null,
    topPrograms: eventCount > 0 ? topPrograms : null,
  };
}

const WEBGPU_API_COUNTER_KEYS = [
  'submitCount',
  'submittedCommandBufferCount',
  'mapAsyncCount',
  'copyBufferToBufferCount',
  'copyBufferToBufferBytes',
  'createBufferCount',
  'createBufferBytes',
  'mapReadBufferCount',
  'mapReadBufferBytes',
  'computePipelineCreateCount',
  'computePipelineCreateAsyncCount',
] as const;

const webGpuApiStats: OrtWebGpuApiInstrumentationSummary = {
  enabled: false,
  installed: false,
  errors: [],
  submitCount: 0,
  submittedCommandBufferCount: 0,
  mapAsyncCount: 0,
  mapAsyncMsTotal: 0,
  copyBufferToBufferCount: 0,
  copyBufferToBufferBytes: 0,
  createBufferCount: 0,
  createBufferBytes: 0,
  mapReadBufferCount: 0,
  mapReadBufferBytes: 0,
  computePipelineCreateCount: 0,
  computePipelineCreateAsyncCount: 0,
};
let webGpuApiInstrumentationInstalled = false;

function wrapPrototypeMethod<T extends (...args: never[]) => unknown>(proto: unknown, method: string, wrap: (original: T) => T): boolean {
  if (!proto || typeof proto !== 'object') return false;
  const rec = proto as Record<string, unknown>;
  if (typeof rec[method] !== 'function') return false;
  try {
    rec[method] = wrap(rec[method] as T);
    return true;
  } catch (error) {
    webGpuApiStats.errors.push(`${method}: ${error instanceof Error ? error.message : String(error)}`);
    return false;
  }
}

export function getOrtWebGpuDiagnosticsSnapshot(): OrtWebGpuDiagnosticsSnapshot {
  const byProgram = new Map<string, { count: number; gpuMs: number }>();
  for (const record of ortWebGpuProfileRecords) {
    const value = byProgram.get(record.programName) ?? { count: 0, gpuMs: 0 };
    value.count += 1;
    value.gpuMs += record.gpuMs;
    byProgram.set(record.programName, value);
  }
  const topPrograms = Array.from(byProgram.entries())
    .map(([programName, value]) => ({ programName, count: value.count, gpuMs: Number(value.gpuMs.toFixed(6)) }))
    .sort((a, b) => b.gpuMs - a.gpuMs)
    .slice(0, 12);
  return {
    profiling: webGpuProfilingSummary(requestedOrtWebGpuProfiling(), ortWebGpuProfileEventCount, ortWebGpuProfileTotalMs, topPrograms),
    api: { ...webGpuApiStats, errors: [...webGpuApiStats.errors], mapAsyncMsTotal: Number(webGpuApiStats.mapAsyncMsTotal.toFixed(6)) },
  };
}

export async function waitForOrtWebGpuDiagnostics(): Promise<void> {
  if (!requestedOrtWebGpuProfiling()) return;
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

export function subtractOrtWebGpuDiagnosticsSnapshot(after: OrtWebGpuDiagnosticsSnapshot, before: OrtWebGpuDiagnosticsSnapshot): OrtWebGpuDiagnosticsSnapshot {
  const beforePrograms = new Map(before.profiling.topPrograms?.map((entry) => [entry.programName, entry]));
  const api: OrtWebGpuApiInstrumentationSummary = {
    ...after.api,
    mapAsyncMsTotal: Number((after.api.mapAsyncMsTotal - before.api.mapAsyncMsTotal).toFixed(6)),
  };
  for (const key of WEBGPU_API_COUNTER_KEYS) api[key] = after.api[key] - before.api[key];
  return {
    profiling: webGpuProfilingSummary(
      after.profiling.enabled,
      after.profiling.eventCount - before.profiling.eventCount,
      // Null cumulative totals have no events, so contribute nothing to a real-event delta.
      (after.profiling.kernelGpuMsTotal ?? 0) - (before.profiling.kernelGpuMsTotal ?? 0),
      (after.profiling.topPrograms ?? [])
        .map((entry) => {
          const prev = beforePrograms.get(entry.programName);
          return { programName: entry.programName, count: entry.count - (prev?.count ?? 0), gpuMs: Number((entry.gpuMs - (prev?.gpuMs ?? 0)).toFixed(6)) };
        })
        .filter((entry) => entry.count > 0 || entry.gpuMs > 0)
        .sort((a, b) => b.gpuMs - a.gpuMs)
        .slice(0, 12),
    ),
    api,
  };
}

function configureOrtRuntime(): { threads: number; logSeverityLevel: 0 | 1 | 2 | 3 | 4 } {
  const logSeverityLevel = requestedOrtLogSeverityLevel();
  (ort.env as unknown as { logLevel?: (typeof ORT_LOG_LEVELS)[number] }).logLevel = ORT_LOG_LEVELS[logSeverityLevel];
  const wasm = ort.env.wasm as unknown as {
    numThreads?: number;
    proxy?: boolean;
    simd?: boolean | 'fixed' | 'relaxed';
    wasmBinary?: ArrayBufferLike | Uint8Array;
    wasmPaths?: string | { wasm?: string; mjs?: string };
  };
  const isBrowserMainThread = typeof document !== 'undefined';
  const isNode = !isBrowserMainThread && runningUnderNode();
  // Node resolves `onnxruntime-web/webgpu` to its inlined asyncify glue; feed the matching binary.
  if (!isBrowserMainThread && !wasm.wasmBinary && forcedOrtWasmArtifact.variant === 'bundled') {
    const proc = globalThis.process as unknown as { cwd?: () => string; getBuiltinModule?: (name: string) => unknown } | undefined;
    const fs = (proc?.getBuiltinModule?.('node:fs') ?? proc?.getBuiltinModule?.('fs')) as
      | { existsSync?: (path: string) => boolean; readFileSync?: (path: string) => Uint8Array }
      | undefined;
    const wasmPath = proc?.cwd ? `${proc.cwd()}/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm` : '';
    if (wasmPath && fs?.existsSync?.(wasmPath) && fs.readFileSync) {
      wasm.wasmBinary = fs.readFileSync(wasmPath);
      wasm.wasmPaths = { wasm: wasmPath };
    }
  }
  if (forcedOrtWasmArtifact.variant === 'relaxed') {
    if (!supportsWasmRelaxedSimdIntegerDot()) {
      throw new Error('requested relaxed ORT WASM artifact requires i32x4.relaxed_dot_i8x16_i7x16_add support');
    }
    wasm.simd = 'relaxed';
  } else {
    wasm.simd = 'fixed';
  }
  if (forcedOrtWasmArtifact.variant !== 'bundled') {
    wasm.wasmBinary = undefined;
    wasm.wasmPaths = { mjs: forcedOrtWasmArtifact.mjsUrl!, wasm: forcedOrtWasmArtifact.wasmUrl! };
  } else if (!isNode && typeof location !== 'undefined') {
    // Production builds hand ORT explicit staged glue + wasm URLs so helper workers
    // re-import the exact pthread bootstrap; dev only pins the asyncify .wasm binary.
    wasm.wasmPaths =
      browserParam('ortWasmPath') ||
      (IS_PROD ? resolveOrtPthreadRuntimeUrls('/ort/', location.href, ortRuntimeArtifactKindForCurrentThread()) : { wasm: `/ort/${ORT_PTHREAD_WASM_FILE}` });
  }
  const threads = requestedOrtWasmThreads(isBrowserMainThread, isNode);
  if (threads > 0) wasm.numThreads = threads;
  // Keep proxy disabled on the main thread; users opt into pthread workers via ?ortThreads.
  if (isBrowserMainThread) wasm.proxy = false;
  webGpuApiStats.enabled = requestedOrtWebGpuApiInstrumentation();
  if (webGpuApiStats.enabled && !webGpuApiInstrumentationInstalled) {
    webGpuApiInstrumentationInstalled = true;
    const g = globalThis as unknown as Record<string, { prototype?: unknown } | undefined>;
    const installed = [
      wrapPrototypeMethod(
        g.GPUQueue?.prototype,
        'submit',
        (original: (buffers: unknown[]) => void) =>
          function submit(this: unknown, buffers: unknown[]) {
            webGpuApiStats.submitCount += 1;
            webGpuApiStats.submittedCommandBufferCount += Array.isArray(buffers) ? buffers.length : 0;
            return original.call(this, buffers);
          } as never,
      ),
      wrapPrototypeMethod(
        g.GPUBuffer?.prototype,
        'mapAsync',
        (original: (mode: number, offset?: number, size?: number) => Promise<void>) =>
          async function mapAsync(this: unknown, mode: number, offset?: number, size?: number) {
            const started = tinyLeelaNowMs();
            webGpuApiStats.mapAsyncCount += 1;
            try {
              return await original.call(this, mode, offset, size);
            } finally {
              webGpuApiStats.mapAsyncMsTotal += tinyLeelaNowMs() - started;
            }
          } as never,
      ),
      wrapPrototypeMethod(
        g.GPUCommandEncoder?.prototype,
        'copyBufferToBuffer',
        (original: (...args: unknown[]) => void) =>
          function copyBufferToBuffer(this: unknown, ...args: unknown[]) {
            webGpuApiStats.copyBufferToBufferCount += 1;
            const size = Number(args[4] ?? 0);
            if (Number.isFinite(size) && size > 0) webGpuApiStats.copyBufferToBufferBytes += size;
            return original.apply(this, args);
          } as never,
      ),
      wrapPrototypeMethod(
        g.GPUDevice?.prototype,
        'createBuffer',
        (original: (descriptor: { size?: number; usage?: number }) => unknown) =>
          function createBuffer(this: unknown, descriptor: { size?: number; usage?: number }) {
            const size = Number(descriptor?.size ?? 0);
            const usage = Number(descriptor?.usage ?? 0);
            webGpuApiStats.createBufferCount += 1;
            if (Number.isFinite(size) && size > 0) webGpuApiStats.createBufferBytes += size;
            const mapRead = (globalThis as unknown as { GPUBufferUsage?: { MAP_READ?: number } }).GPUBufferUsage?.MAP_READ ?? 1;
            if ((usage & mapRead) !== 0) {
              webGpuApiStats.mapReadBufferCount += 1;
              if (Number.isFinite(size) && size > 0) webGpuApiStats.mapReadBufferBytes += size;
            }
            return original.call(this, descriptor);
          } as never,
      ),
      wrapPrototypeMethod(
        g.GPUDevice?.prototype,
        'createComputePipeline',
        (original: (descriptor: unknown) => unknown) =>
          function createComputePipeline(this: unknown, descriptor: unknown) {
            webGpuApiStats.computePipelineCreateCount += 1;
            return original.call(this, descriptor);
          } as never,
      ),
      wrapPrototypeMethod(
        g.GPUDevice?.prototype,
        'createComputePipelineAsync',
        (original: (descriptor: unknown) => Promise<unknown>) =>
          function createComputePipelineAsync(this: unknown, descriptor: unknown) {
            webGpuApiStats.computePipelineCreateAsyncCount += 1;
            return original.call(this, descriptor);
          } as never,
      ),
    ];
    webGpuApiStats.installed = installed.some(Boolean);
    if (!webGpuApiStats.installed) webGpuApiStats.errors.push('No WebGPU prototypes were available to patch before ORT initialization');
  }
  const webgpu = ort.env.webgpu as unknown as
    | { powerPreference?: 'low-power' | 'high-performance'; profiling?: { mode?: 'off' | 'default'; ondata?: (data: unknown) => void } }
    | undefined;
  if (webgpu && requestedOrtExecutionProvider() !== 'wasm') webgpu.powerPreference = 'high-performance';
  if (webgpu && requestedOrtWebGpuProfiling()) {
    webgpu.profiling ??= {};
    webgpu.profiling.mode = 'default';
    if (!ortWebGpuProfilingConfigured) {
      previousOrtWebGpuProfilingOnData = webgpu.profiling.ondata;
      webgpu.profiling.ondata = (data: unknown) => {
        const rec = data as { programName?: unknown; kernelName?: unknown; kernelType?: unknown; startTime?: unknown; endTime?: unknown };
        const start = Number(rec.startTime ?? 0);
        const end = Number(rec.endTime ?? 0);
        const gpuMs = Number.isFinite(start) && Number.isFinite(end) && end >= start ? (end - start) / 1e6 : 0;
        ortWebGpuProfileEventCount += 1;
        ortWebGpuProfileTotalMs += gpuMs;
        ortWebGpuProfileRecords.push({
          programName: String(rec.programName ?? 'unknown'),
          kernelName: String(rec.kernelName ?? 'unknown'),
          kernelType: String(rec.kernelType ?? 'unknown'),
          gpuMs,
        });
        while (ortWebGpuProfileRecords.length > 4096) ortWebGpuProfileRecords.shift();
        previousOrtWebGpuProfilingOnData?.(data);
      };
      ortWebGpuProfilingConfigured = true;
    }
  }
  return { threads, logSeverityLevel };
}

export function sessionOptions(executionProviders = resolvedOrtExecutionProviders()): ort.InferenceSession.SessionOptions {
  const { threads, logSeverityLevel } = configureOrtRuntime();
  const opts: ort.InferenceSession.SessionOptions = {
    graphOptimizationLevel: requestedOrtGraphOptimizationLevel(),
    executionProviders,
    logSeverityLevel,
  };
  const preferredOutputLocation = requestedOrtPreferredOutputLocation();
  if (preferredOutputLocation && executionProviders.includes('webgpu')) opts.preferredOutputLocation = preferredOutputLocation;
  if (threads > 0) {
    opts.intraOpNumThreads = threads;
    opts.interOpNumThreads = 1;
  }
  return opts;
}

export function shouldFallbackToWasmAfterOrtFailure(requested: OrtExecutionProviderPreference, providers: string[]): boolean {
  return requested !== 'webgpu' && providers.includes('webgpu');
}

function recordSessionAttempt(providers: string[], ok: boolean, ms: number, error?: string) {
  sessionAttempts.push({ at: new Date().toISOString(), providers: [...providers], ok, ms, ...(error ? { error } : {}) });
  while (sessionAttempts.length > 32) sessionAttempts.shift();
}

function logOrtSessionReady(providers: string[], ms: number, note?: string) {
  const message = providers.includes('webgpu')
    ? 'Centipawn ORT: session ready with WebGPU provider requested/accepted.'
    : 'Centipawn ORT: session ready with WASM provider.';
  console.info(message, {
    requestedEp: requestedOrtExecutionProvider(),
    sessionProviders: providers,
    webgpuAvailable: webgpuAvailable(),
    describe: describeOrtBackendConfig(),
    runtimeArtifact: ortRuntimeArtifactKindForCurrentThread(),
    ms: Number(ms.toFixed(1)),
    ...(note ? { note } : {}),
  });
}

function webgpuNavigator(): { requestAdapter?: (opts?: unknown) => Promise<unknown> } | undefined {
  if (typeof navigator === 'undefined') return undefined;
  return (navigator as Navigator & { gpu?: { requestAdapter?: (opts?: unknown) => Promise<unknown> } }).gpu;
}

async function requestOrtWebGpuAdapter(): Promise<unknown> {
  return webgpuNavigator()?.requestAdapter?.({ powerPreference: 'high-performance' });
}

const GPU_LIMIT_KEYS = [
  'maxBindGroups',
  'maxBindingsPerBindGroup',
  'maxBufferSize',
  'maxComputeInvocationsPerWorkgroup',
  'maxComputeWorkgroupSizeX',
  'maxComputeWorkgroupSizeY',
  'maxComputeWorkgroupSizeZ',
  'maxComputeWorkgroupsPerDimension',
  'maxStorageBufferBindingSize',
  'maxStorageBuffersPerShaderStage',
  'maxUniformBufferBindingSize',
];

export async function collectOrtRuntimeDiagnostics(options: { probeAdapter?: boolean } = {}): Promise<OrtRuntimeDiagnostics> {
  configureOrtRuntime();
  const wasm = ort.env.wasm as unknown as { numThreads?: number; proxy?: boolean };
  const webgpu = ort.env.webgpu as unknown as { powerPreference?: string } | undefined;
  const artifact = requestedOrtWasmArtifact();
  const diag: OrtRuntimeDiagnostics = {
    ...collectOrtSessionDiagnostics(),
    requestedEp: requestedOrtExecutionProvider(),
    resolvedExecutionProviders: lastOrtExecutionProviders ?? resolvedOrtExecutionProviders(),
    describe: describeOrtBackendConfig(),
    webgpuAvailable: webgpuAvailable(),
    secureContext: typeof isSecureContext === 'undefined' ? undefined : isSecureContext,
    crossOriginIsolated: typeof crossOriginIsolated === 'undefined' ? undefined : crossOriginIsolated,
    userAgent: typeof navigator === 'undefined' ? undefined : navigator.userAgent,
    wasm: {
      numThreads: wasm.numThreads,
      proxy: wasm.proxy,
      sharedArrayBuffer: typeof SharedArrayBuffer !== 'undefined',
      threadedAvailable: browserThreadedWasmAvailable(),
      simdVariant: artifact.variant,
      runtimeArtifact: ortRuntimeArtifactKindForCurrentThread(),
      jspiApiAvailable: ortJspiApiAvailable(),
      jspiPreference: requestedOrtJspiPreference(),
      ...(ortJspiFallbackReason ? { jspiFallback: ortJspiFallbackReason } : {}),
      ...(artifact.artifactId ? { artifactId: artifact.artifactId } : {}),
      ...(artifact.mjsUrl ? { mjsUrl: artifact.mjsUrl } : {}),
      ...(artifact.wasmUrl ? { wasmUrl: artifact.wasmUrl } : {}),
      relaxedIntegerDotAvailable: supportsWasmRelaxedSimdIntegerDot(),
    },
    ...(webgpu
      ? {
          webgpuEnv: {
            powerPreference: webgpu.powerPreference,
            profilingMode: (webgpu as { profiling?: { mode?: string } }).profiling?.mode,
            preferredOutputLocation: requestedOrtPreferredOutputLocation(),
            apiInstrumentation: requestedOrtWebGpuApiInstrumentation(),
          },
        }
      : {}),
    sessions: { created: createdOrtSessions, released: releasedOrtSessions, active: Math.max(0, createdOrtSessions - releasedOrtSessions) },
    sessionAttempts: sessionAttempts.map((x) => ({ ...x, providers: [...x.providers] })),
    ...(lastOrtRuntimeFallback ? { fallback: { ...lastOrtRuntimeFallback } } : {}),
    webgpuDiagnostics: getOrtWebGpuDiagnosticsSnapshot(),
  };
  if (!options.probeAdapter || !webgpuAvailable()) return diag;
  try {
    const adapter = await webgpuNavigator()?.requestAdapter?.({ powerPreference: 'high-performance' });
    const rec = adapter && typeof adapter === 'object' ? (adapter as Record<string, unknown>) : undefined;
    let info: unknown = rec?.info;
    if (!info && typeof rec?.requestAdapterInfo === 'function') {
      try {
        info = await (rec.requestAdapterInfo as () => Promise<unknown>)();
      } catch {
        /* optional API */
      }
    }
    let features: string[] | undefined;
    if (rec?.features && typeof rec.features === 'object') {
      try {
        const items = Array.from(rec.features as Iterable<unknown>, (feature) => String(feature))
          .filter(Boolean)
          .sort();
        if (items.length) features = items;
      } catch {
        /* optional API */
      }
    }
    let limits: Record<string, number> | undefined;
    if (rec?.limits && typeof rec.limits === 'object') {
      const limitRec = rec.limits as Record<string, unknown>;
      const collected: Record<string, number> = {};
      for (const key of GPU_LIMIT_KEYS) {
        const raw = limitRec[key];
        if (typeof raw === 'number' && Number.isFinite(raw)) collected[key] = raw;
      }
      if (Object.keys(collected).length) limits = collected;
    }
    probedWebGpuAdapterUsable = !!adapter;
    probedWebGpuShaderF16 = gpuAdapterShaderF16(adapter);
    diag.resolvedExecutionProviders = resolvedOrtExecutionProviders();
    diag.describe = describeOrtBackendConfig();
    if (!adapter) {
      diag.adapter = { ok: false, error: 'navigator.gpu.requestAdapter returned null' };
      return diag;
    }
    const parts: string[] = [];
    if (rec) {
      for (const key of ['name', 'vendor', 'architecture', 'device', 'description']) {
        if (typeof rec[key] === 'string' && rec[key]) parts.push(`${key}=${rec[key]}`);
      }
      if (rec.info && typeof rec.info === 'object') {
        const infoRec = rec.info as Record<string, unknown>;
        for (const key of ['vendor', 'architecture', 'device', 'description']) {
          if (typeof infoRec[key] === 'string' && infoRec[key]) parts.push(`${key}=${infoRec[key]}`);
        }
      }
    }
    diag.adapter = {
      ok: true,
      summary: rec ? parts.join(' ') || Object.prototype.toString.call(adapter) : String(adapter),
      ...(info ? { info } : {}),
      ...(features ? { features } : {}),
      ...(probedWebGpuShaderF16 !== null ? { shaderF16: probedWebGpuShaderF16 } : {}),
      ...(limits ? { limits } : {}),
    };
  } catch (err) {
    probedWebGpuAdapterUsable = false;
    diag.resolvedExecutionProviders = resolvedOrtExecutionProviders();
    diag.describe = describeOrtBackendConfig();
    diag.adapter = { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  return diag;
}

/**
 * Settle `probedWebGpuAdapterUsable` before the artifact is chosen.
 *
 * `navigator.gpu` existing does not mean an adapter can be acquired (a
 * blocklisted GPU, a headless or software-only context). Provider and artifact
 * resolution are both synchronous and consult the probe result, but the only
 * thing that used to set it was `collectOrtRuntimeDiagnostics({probeAdapter})`,
 * which normal worker startup runs *after* the first session. Such a browser
 * therefore locked the 24 MB asyncify pair at one thread, tried WebGPU, failed,
 * and served the rest of the session from the large binary anyway.
 *
 * Runs at most once per thread, and only when the answer can change a decision:
 * a pinned `ep=wasm` never consults it, and strict `ep=webgpu` resolves to
 * `['webgpu']` regardless so it still surfaces "WebGPU unavailable" rather than
 * silently degrading.
 */
async function ensureWebGpuAdapterProbed(requested = requestedOrtExecutionProvider()): Promise<void> {
  if (probedWebGpuAdapterUsable !== null || !webgpuAvailable() || requested === 'wasm' || requested === 'webgpu') return;
  try {
    const adapter = await requestOrtWebGpuAdapter();
    probedWebGpuAdapterUsable = !!adapter;
    probedWebGpuShaderF16 = gpuAdapterShaderF16(adapter);
    if (probedWebGpuShaderF16 === false) console.info('Centipawn ORT: WebGPU adapter lacks shader-f16; ORT sessions run on WASM.');
    // gpuAdapterVendor never throws: unreadable metadata means "vendor unknown",
    // not "adapter unusable".
    if (adapter) probedWebGpuAdapterVendor = (await gpuAdapterVendor(adapter)) ?? '';
  } catch {
    probedWebGpuAdapterUsable = false;
  }
}

/**
 * `GPUAdapterInfo.vendor`, lower-cased; `adapter.info` first, then the older
 * `requestAdapterInfo()`. Never throws: a getter or API that throws just leaves
 * the vendor unknown (so the caller keeps WebGPU and the default model).
 */
export async function gpuAdapterVendor(adapter: unknown): Promise<string | undefined> {
  if (!adapter || typeof adapter !== 'object') return undefined;
  const rec = adapter as { info?: { vendor?: unknown }; requestAdapterInfo?: () => Promise<{ vendor?: unknown }> };
  let vendor: unknown;
  try {
    vendor = rec.info?.vendor;
  } catch {
    vendor = undefined;
  }
  if (typeof vendor !== 'string') {
    try {
      if (typeof rec.requestAdapterInfo === 'function') vendor = (await rec.requestAdapterInfo())?.vendor;
    } catch {
      /* optional API */
    }
  }
  return typeof vendor === 'string' && vendor.trim() ? vendor.trim().toLowerCase() : undefined;
}

/** Test hook: forget this thread's adapter probe so the next session probes again. */
export function resetWebGpuAdapterProbeForTests(): void {
  probedWebGpuAdapterUsable = null;
  probedWebGpuShaderF16 = null;
  probedWebGpuAdapterVendor = undefined;
}

/** undefined = not probed yet; '' = probed, no adapter or no vendor string. */
let probedWebGpuAdapterVendor: string | undefined;

async function webGpuAdapterVendor(): Promise<string | undefined> {
  if (probedWebGpuAdapterVendor === undefined) {
    probedWebGpuAdapterVendor = '';
    try {
      // Same request ORT makes (configureOrtRuntime sets powerPreference), so on a
      // dual-GPU machine this names the adapter the session will run on.
      const adapter = await requestOrtWebGpuAdapter();
      probedWebGpuAdapterVendor = (await gpuAdapterVendor(adapter)) ?? '';
    } catch {
      /* no adapter: vendor stays unknown */
    }
  }
  return probedWebGpuAdapterVendor || undefined;
}

export type OrtSessionPlacement = {
  /** Providers the next createOrtSession call will request, in order. */
  providers: string[];
  /** Lower-cased GPUAdapterInfo.vendor ('apple', 'nvidia', ...); only probed when providers include webgpu. */
  adapterVendor?: string;
};

/**
 * Where the next session on this thread will run, decided the same way
 * createOrtSession decides it. Lets a caller pick device-specific model bytes
 * before downloading them (src/lc0/lc0SmallModelSelection.ts).
 */
export async function resolveOrtSessionPlacement(requested = requestedOrtExecutionProvider()): Promise<OrtSessionPlacement> {
  await ensureWebGpuAdapterProbed(requested);
  const providers = resolvedOrtExecutionProviders(requested);
  if (!providers.includes('webgpu') || !webgpuAvailable()) return { providers };
  const adapterVendor = await webGpuAdapterVendor();
  return adapterVendor ? { providers, adapterVendor } : { providers };
}

export async function resolveFreshWorkerOrtSessionPlacement(requested: OrtExecutionProviderPreference): Promise<OrtSessionPlacement> {
  const adapter = requested === 'wasm' ? undefined : await requestOrtWebGpuAdapter().catch(() => undefined);
  const providers = selectOrtExecutionProviders(requested, !!adapter && gpuAdapterShaderF16(adapter) !== false);
  const adapterVendor = providers.includes('webgpu') ? await gpuAdapterVendor(adapter) : undefined;
  return adapterVendor ? { providers, adapterVendor } : { providers };
}

type OrtModelSource = OrtModelInput | (() => Promise<OrtModelInput>);

export type OrtSessionAttemptPlan = {
  input: OrtModelSource;
  providers: string[];
  /** 'wasm-fallback' = the explicit WASM attempt after the WebGPU attempts failed. */
  kind: 'primary' | 'unfolded' | 'wasm-fallback';
  note: (message: string | null) => string | undefined;
};

export type CreateOrtSessionOptions = {
  /** Cancel preparation waits and future native attempts; already-admitted creation is released before rejection. */
  signal?: AbortSignal;
  /**
   * Model for the WebGPU -> WASM fallback attempt, loaded only if that attempt
   * runs. Set when the primary model is a WebGPU-specific rewrite that is slower
   * on WASM; the default reuses the primary model's unfolded bytes.
   */
  wasmFallbackModel?: () => Promise<OrtModelInput>;
  /** Reacquire the same model through its verified loader, only if an unfolded/provider fallback needs it. */
  reloadModel?: () => Promise<OrtModelInput>;
  /**
   * Extra session options for the WebGPU attempts only (never the WASM
   * fallback), e.g. `enableGraphCapture` for GPU-resident IO (src/nn/ortGpuIo.ts).
   */
  webgpuSessionOptions?: Partial<ort.InferenceSession.SessionOptions>;
  /** Request WebGPU alone and never fall back to WASM: the caller handles failure. */
  webgpuOnly?: boolean;
  /** Let the verified loader repair metadata-trusted bytes before any provider/model fallback masks a parse error. */
  rejectModelParseErrors?: boolean | (() => boolean);
};

/**
 * Attempt order: folded model on the requested providers; the unfolded model on
 * the same providers if ORT rejected the folded rewrite; then, when the
 * requested provider preference allows it, WASM on `wasmFallbackModel` or else
 * the unfolded model. Exported for tests.
 *
 * With `wasmFallbackModel` the primary attempts request WebGPU only. Given
 * ['webgpu', 'wasm'], ORT itself drops WebGPU when its backend init fails and
 * resolves a WASM session on the primary (WebGPU-specific) bytes, so the WASM
 * model would never be used.
 */
export function planOrtSessionAttempts(
  prepared: { input: OrtModelInput; original?: OrtModelSource },
  providers: string[],
  wasmFallback: boolean,
  options: CreateOrtSessionOptions = {},
): OrtSessionAttemptPlan[] {
  const primaryProviders =
    options.wasmFallbackModel && wasmFallback && providers.includes('webgpu') ? providers.filter((provider) => provider !== 'wasm') : providers;
  const attempts: OrtSessionAttemptPlan[] = [{ input: prepared.input, providers: primaryProviders, kind: 'primary', note: () => describeDequantFold() }];
  if (prepared.original !== undefined) {
    attempts.push({
      input: prepared.original,
      providers: primaryProviders,
      kind: 'unfolded',
      note: (message) => `unfolded model after the folded one failed to load: ${message}`,
    });
  }
  if (wasmFallback) {
    attempts.push({
      input: options.wasmFallbackModel ?? prepared.original ?? options.reloadModel ?? prepared.input,
      providers: ['wasm'],
      kind: 'wasm-fallback',
      note: (message) => `fallback after WebGPU failure${options.wasmFallbackModel ? ' (WASM model)' : ''}: ${message}`,
    });
  }
  return attempts;
}

export function createOrtSession(modelPath: OrtModelInput, options: CreateOrtSessionOptions = {}): Promise<ort.InferenceSession> {
  // Async frames retain original argument registers even after reassignment.
  // Only this synchronous, consumed holder ever receives the raw argument.
  return createOrtSessionFromSource(() => {
    const input = modelPath;
    modelPath = undefined as never;
    return input;
  }, options);
}

async function createOrtSessionFromSource(takeModel: () => OrtModelInput, options: CreateOrtSessionOptions): Promise<ort.InferenceSession> {
  // Must precede resolvedOrtExecutionProviders(): both it and the artifact
  // selector read the probe result, and the locks below freeze that choice.
  await untilAborted(ensureWebGpuAdapterProbed(), options.signal);
  options.signal?.throwIfAborted();
  const resolved = resolvedOrtExecutionProviders();
  if (options.webgpuOnly && !resolved.includes('webgpu')) throw new Error('createOrtSession: webgpuOnly requested but WebGPU is not a resolved provider');
  const providers = options.webgpuOnly ? ['webgpu'] : resolved;
  // Both locks run before the first InferenceSession.create so the artifact is
  // decided (and frozen) before ORT initializes its worker-global wasm module;
  // the WebGPU->WASM fallback below then reuses the same binary.
  lockRequestedOrtWasmArtifact();
  lockOrtRuntimeArtifactKind(providers);
  let prepared: PreparedOrtModelInput | undefined = await prepareOrtModelInput(takeModel(), options.reloadModel);
  const wasmFallback = !options.webgpuOnly && shouldFallbackToWasmAfterOrtFailure(requestedOrtExecutionProvider(), providers);
  const attempts = planOrtSessionAttempts(prepared, providers, wasmFallback, options);
  prepared = undefined;
  let firstMessage: string | undefined;
  let lastMessage: string | null = null;
  for (let i = 0; i < attempts.length; i++) {
    const attempt = attempts[i];
    const t0 = typeof performance === 'undefined' ? Date.now() : performance.now();
    try {
      let source: OrtModelSource | undefined = attempt.input;
      attempt.input = undefined as never; // Consumed records must not pin an input through later attempts.
      let input: OrtModelInput | undefined = typeof source === 'function' ? await untilAborted(source(), options.signal) : source;
      source = undefined;
      options.signal?.throwIfAborted();
      const extra = attempt.providers.includes('webgpu') ? options.webgpuSessionOptions : undefined;
      const creating = ort.InferenceSession.create(input as never, { ...sessionOptions(attempt.providers), ...extra });
      input = undefined; // ORT owns its ingestion lifetime; do not keep an extra JS root until it returns.
      const session = await creating;
      const t1 = typeof performance === 'undefined' ? Date.now() : performance.now();
      createdOrtSessions += 1;
      if (options.signal?.aborted) {
        await trackEngineRetirement(releaseOrtSession(session));
        options.signal.throwIfAborted();
      }
      recordSessionAttempt(attempt.providers, true, t1 - t0);
      lastOrtExecutionProviders = attempt.providers;
      lastOrtRuntimeFallback =
        attempt.kind === 'wasm-fallback' ? { at: new Date().toISOString(), from: 'webgpu', to: 'wasm', reason: lastMessage ?? '' } : undefined;
      logOrtSessionReady(attempt.providers, t1 - t0, attempt.note(lastMessage));
      return session;
    } catch (err) {
      options.signal?.throwIfAborted();
      const t1 = typeof performance === 'undefined' ? Date.now() : performance.now();
      lastMessage = ortErrorMessage(err);
      recordSessionAttempt(attempt.providers, false, t1 - t0, lastMessage);
      const rejectParse = typeof options.rejectModelParseErrors === 'function' ? options.rejectModelParseErrors() : options.rejectModelParseErrors;
      const next = attempts[i + 1];
      if (!next || (rejectParse && isOnnxModelParseError(err))) {
        throw firstMessage === undefined ? err : new OrtSessionFallbackError(firstMessage, err);
      }
      firstMessage ??= lastMessage;
      if (next.kind !== 'wasm-fallback') {
        discardDequantFold(lastMessage);
        console.warn(`Centipawn: ORT session failed on the dequantize-folded model; retrying with the original bytes. ${lastMessage}`);
      } else {
        // A separate WASM model is loaded as-is (unfolded), so the fold summary no longer applies.
        if (typeof next.input === 'function') discardDequantFold(lastMessage);
        console.warn(`Centipawn: ORT WebGPU session failed; falling back to WASM. ${lastMessage}`);
      }
    }
  }
  throw new Error('Centipawn: ORT session creation exhausted all attempts');
}

export async function releaseOrtSession(session: ort.InferenceSession): Promise<void> {
  const maybe = session as ort.InferenceSession & { release?: () => Promise<void> | void };
  if (typeof maybe.release === 'function') await maybe.release();
  releasedOrtSessions += 1;
}

type OrtOutputTensor = Pick<ort.Tensor, 'location' | 'data' | 'getData' | 'dispose'>;

function ortTensorOnDevice(tensor: OrtOutputTensor): boolean {
  return tensor.location === 'gpu-buffer' || tensor.location === 'texture' || tensor.location === 'ml-tensor';
}

/**
 * Reads a session output on the CPU. With `preferredOutputLocation:
 * 'gpu-buffer'` (?ortGpuOutputs, ?preferredOutputLocation) ORT WebGPU returns
 * GPU-located outputs whose `.data` throws and whose GPU buffer lives until
 * the tensor is disposed; getData(true) downloads and then releases it.
 */
export async function ortTensorData(tensor: OrtOutputTensor): Promise<ort.Tensor['data']> {
  return ortTensorOnDevice(tensor) ? await tensor.getData(true) : tensor.data;
}

/** Releases the GPU buffers of outputs that were not (or not yet fully) read. CPU outputs are left untouched. */
export function releaseOrtGpuOutputs(outputs: Record<string, OrtOutputTensor>): void {
  for (const tensor of Object.values(outputs)) {
    if (!ortTensorOnDevice(tensor)) continue;
    try {
      tensor.dispose();
    } catch {
      /* still downloading: getData(true) releases it when the download completes */
    }
  }
}

/** Brings every GPU-located output to the CPU in place (so `.data` works) and releases its GPU buffer. */
export async function downloadOrtOutputs(outputs: Record<string, OrtOutputTensor>): Promise<void> {
  try {
    for (const tensor of Object.values(outputs)) if (ortTensorOnDevice(tensor)) await tensor.getData(true);
  } finally {
    releaseOrtGpuOutputs(outputs);
  }
}
