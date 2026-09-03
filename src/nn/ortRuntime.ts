import './ortConsoleFilter.ts';

export * from 'onnxruntime-web/webgpu';

import * as ort from 'onnxruntime-web/webgpu';
import { foldOnnxDequantizeLinear } from './onnxDequantFold.ts';
import { supportsWasmRelaxedSimdIntegerDot } from '../lc0/wasmFeatures.ts';

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
 * ORT ships two threaded browser runtimes and we stage both.
 *
 * - `asyncify`: the WebGPU/JSEP entrypoint. Instrumented for stack unwinding so
 *   the wasm side can suspend across GPU work. ~24MB.
 * - `wasm`: the CPU-only entrypoint. Not asyncify-instrumented, ~13.5MB, and
 *   everything an `executionProviders: ['wasm']` session needs.
 *
 * Both glue modules double as their own Emscripten pthread bootstrap (each
 * spawns `new Worker(new URL(import.meta.url), { name: 'em-pthread' })`), so
 * whichever one we hand to ORT must keep its deployed URL for helper workers to
 * re-import. Glue and binary must come from the same pair: the plain glue
 * refuses to instantiate the asyncify binary (LinkError) and vice versa.
 * `scripts/ort_runtime_assets.mjs` mirrors these names for the deploy-time
 * staging allowlist.
 */
export type OrtRuntimeArtifactKind = 'asyncify' | 'wasm';

export const ORT_PTHREAD_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.asyncify.mjs';
export const ORT_PTHREAD_WASM_FILE = 'ort-wasm-simd-threaded.asyncify.wasm';
export const ORT_WASM_EP_BOOTSTRAP_FILE = 'ort-wasm-simd-threaded.mjs';
export const ORT_WASM_EP_WASM_FILE = 'ort-wasm-simd-threaded.wasm';

export const ORT_RUNTIME_ARTIFACT_FILES: Readonly<Record<OrtRuntimeArtifactKind, { mjs: string; wasm: string }>> = Object.freeze({
  asyncify: { mjs: ORT_PTHREAD_BOOTSTRAP_FILE, wasm: ORT_PTHREAD_WASM_FILE },
  wasm: { mjs: ORT_WASM_EP_BOOTSTRAP_FILE, wasm: ORT_WASM_EP_WASM_FILE },
});

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
  limits?: Record<string, number>;
  error?: string;
};

export type OrtWebGpuProfilingSummary = {
  enabled: boolean;
  eventCount: number;
  kernelGpuMsTotal: number;
  topPrograms: Array<{ programName: string; count: number; gpuMs: number }>;
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

export type OrtRuntimeDiagnostics = {
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
  dequantFold?: OrtDequantFoldSummary;
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

function debugTokens(value: string | null | undefined): string[] {
  return String(value ?? '')
    .toLowerCase()
    .split(/[,+\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

export function tinyLeelaDebugEnabled(area = 'latency'): boolean {
  const normalizedArea = area.toLowerCase();
  const direct = debugParam(`debug${normalizedArea[0].toUpperCase()}${normalizedArea.slice(1)}`) ?? debugParam(normalizedArea);
  if (direct !== null) return !['0', 'false', 'no', 'off'].includes(String(direct).toLowerCase());
  const tokens = [...debugTokens(debugParam('debug')), ...debugTokens(debugParam('tlDebug'))];
  return tokens.some((token) => ['1', 'true', 'yes', 'on', 'all', 'perf', 'timing', normalizedArea].includes(token));
}

export function tinyLeelaNowMs(): number {
  return typeof performance === 'undefined' ? Date.now() : performance.now();
}

function roundMs(ms: number): number {
  return Number(ms.toFixed(2));
}

export function tinyLeelaLogLatency(label: string, payload: Record<string, unknown>): void {
  if (!tinyLeelaDebugEnabled('latency')) return;
  console.info(
    `Centipawn latency: ${label}`,
    Object.fromEntries(Object.entries(payload).map(([key, value]) => [key, typeof value === 'number' ? roundMs(value) : value])),
  );
}

function truthyParam(value: string | null | undefined): boolean {
  if (value === null || value === undefined) return false;
  return !['0', 'false', 'no', 'off'].includes(String(value).toLowerCase());
}

function falseyParam(value: string | null | undefined): boolean {
  if (value === null || value === undefined) return false;
  return ['0', 'false', 'no', 'off'].includes(String(value).toLowerCase());
}

function diagnosticParamValue(name: string): string | null | undefined {
  return browserParam(name) ?? envValue(`TINY_LEELA_${name.toUpperCase()}`);
}

function ortDiagnosticsParamEnabled(...names: string[]): boolean {
  return names.some((name) => truthyParam(diagnosticParamValue(name)));
}

function ortDiagnosticsParamDisabled(...names: string[]): boolean {
  return names.some((name) => falseyParam(diagnosticParamValue(name)));
}

let forcedOrtExecutionProvider: OrtExecutionProviderPreference | null = null;
let forcedOrtDiagnosticsOptions: OrtRuntimeDiagnosticOptions | null = null;
let forcedOrtWasmArtifact: OrtWasmArtifactSelection = { variant: 'bundled' };
let forcedOrtWasmThreads: number | null = null;
let lockedOrtWasmArtifactKey: string | null = null;
let forcedOrtRuntimeArtifactKind: OrtRuntimeArtifactKind | null = null;
let lockedOrtRuntimeArtifactKind: OrtRuntimeArtifactKind | null = null;

export type OrtRuntimeDiagnosticOptions = {
  /** Enable ORT WebGPU timestamp profiling and collect per-kernel program totals. */
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
 * Weight-only DequantizeLinear nodes are folded into plain initializers at load time unless
 * `?ortDequantFold=0` (or ORT_DEQUANT_FOLD=0) asks for the graph as shipped.
 */
export function ortDequantFoldEnabled(): boolean {
  if (forcedOrtDequantFold !== null) return forcedOrtDequantFold;
  const raw = String(browserParam('ortDequantFold') ?? envValue('ORT_DEQUANT_FOLD') ?? '')
    .trim()
    .toLowerCase();
  return !(raw === '0' || raw === 'false' || raw === 'off' || raw === 'no');
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
  /** The unfolded model, present only when `input` is a folded rewrite of it. */
  original?: OrtModelInput;
};

async function prepareOrtModelInput(modelPath: OrtModelInput): Promise<PreparedOrtModelInput> {
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
    return result.foldedNodes > 0 ? { input: result.bytes, original: bytes } : { input: bytes };
  } catch (err) {
    console.warn(`Centipawn: ORT dequantize fold skipped: ${err instanceof Error ? err.message : String(err)}`);
    lastOrtDequantFold = { enabled: true, foldedNodes: 0, skippedNodes: 0, removedInitializers: 0, bytesBefore: bytes.byteLength, bytesAfter: bytes.byteLength, elapsedMs: 0 };
    return { input: bytes };
  }
}

function discardDequantFold(reason: string): void {
  const fold = lastOrtDequantFold;
  if (!fold) return;
  lastOrtDequantFold = { ...fold, foldedNodes: 0, removedInitializers: 0, bytesAfter: fold.bytesBefore, discardedAfter: reason };
}

function describeDequantFold(): string | undefined {
  const fold = lastOrtDequantFold;
  if (!fold || !fold.enabled || fold.foldedNodes === 0) return undefined;
  return `dequant fold: ${fold.foldedNodes} DequantizeLinear folded (${(fold.bytesBefore / 1e6).toFixed(1)} -> ${(fold.bytesAfter / 1e6).toFixed(1)} MB, ${fold.elapsedMs.toFixed(0)} ms)`;
}

function normalizeEp(value: string | null | undefined): OrtExecutionProviderPreference {
  const raw = String(value ?? '')
    .trim()
    .toLowerCase();
  if (!raw) return 'auto';
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'webgpu,wasm' || raw === 'webgpu+wasm' || raw === 'gpu,wasm' || raw === 'gpu+wasm') return 'webgpu,wasm';
  if (raw === 'auto') return 'auto';
  return 'wasm';
}

/** The EP requested by the URL/env of this document or worker, ignoring per-call overrides. */
function ambientOrtExecutionProviderParam(): string | null | undefined {
  return (
    browserParam('ortEp') ?? browserParam('ep') ?? browserParam('executionProviders') ?? envValue('TINY_LEELA_ORT_EP') ?? envValue('ORT_EXECUTION_PROVIDERS')
  );
}

export function requestedOrtExecutionProvider(): OrtExecutionProviderPreference {
  if (forcedOrtExecutionProvider) return forcedOrtExecutionProvider;
  return normalizeEp(ambientOrtExecutionProviderParam());
}

function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!(navigator as unknown as { gpu?: unknown }).gpu;
}

let probedWebGpuAdapterUsable: boolean | null = null;

function webgpuUsableForProviderSelection(): boolean {
  return webgpuAvailable() && probedWebGpuAdapterUsable !== false;
}

export function resolvedOrtExecutionProviders(): string[] {
  const requested = requestedOrtExecutionProvider();
  if (requested === 'webgpu') return ['webgpu'];
  if (requested === 'webgpu,wasm') return webgpuUsableForProviderSelection() ? ['webgpu', 'wasm'] : ['wasm'];
  if (requested === 'auto') return webgpuUsableForProviderSelection() ? ['webgpu', 'wasm'] : ['wasm'];
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
 * - strict `ep=webgpu` always keeps asyncify, so a GPU-less browser still fails
 *   with "WebGPU unavailable" instead of an opaque link error;
 * - anything that resolves to include `webgpu` keeps asyncify, which also keeps
 *   the in-process WebGPU->WASM fallback in `createOrtSession` on one binary;
 * - a CPU-only resolution takes the small build only when escalating to WebGPU
 *   later in this same thread is impossible (`navigator.gpu` missing, i.e.
 *   Safari/Firefox) or has been ruled out by an explicit `?ep=wasm` / env pin.
 *   A programmatic `setRequestedOrtExecutionProviderForCurrentThread('wasm')`
 *   deliberately does NOT qualify: search workers switch EP per message and
 *   must keep the GPU-capable binary. Such a worker can still opt in via
 *   `setOrtRuntimeArtifactKindForCurrentThread('wasm')` before its first
 *   session.
 */
export function ortRuntimeArtifactKindForCurrentThread(): OrtRuntimeArtifactKind {
  if (lockedOrtRuntimeArtifactKind) return lockedOrtRuntimeArtifactKind;
  // Node (no `location`) imports ORT's bundle with the Emscripten glue inlined,
  // which is the asyncify one; configureNodeOrtWasmBinary feeds it the matching
  // binary and wasmPaths are never consulted, so the kind is fixed there.
  if (typeof location === 'undefined') return 'asyncify';
  // Checked BEFORE the forced kind: the experimental /ort-experimental/ builds
  // carry their own explicit URLs and are asyncify binaries, so what they are
  // is not a matter of preference. `?ortWasmVariant=fixed|relaxed` combined
  // with an `ep=wasm` pin would otherwise report the CPU-only kind while
  // actually loading asyncify, and the thread policy keys off this value --
  // which is exactly the asyncify-plus-threads combination measured as a 55%
  // regression.
  if (forcedOrtWasmArtifact.variant !== 'bundled') return 'asyncify';
  if (forcedOrtRuntimeArtifactKind) return forcedOrtRuntimeArtifactKind;
  if (requestedOrtExecutionProvider() === 'webgpu') return 'asyncify';
  if (resolvedOrtExecutionProviders().includes('webgpu')) return 'asyncify';
  // `webgpuUsableForProviderSelection`, not `webgpuAvailable`: a browser can
  // expose `navigator.gpu` and still fail to yield an adapter. Once the probe
  // has explicitly failed, provider resolution already drops WebGPU, so keying
  // the artifact off mere API presence would hand that browser the larger
  // asyncify pair and the single-thread policy for a GPU it can never use.
  // Strict `ep=webgpu` is unaffected: it returned asyncify above, so a GPU-less
  // browser still fails with "WebGPU unavailable" rather than a link error.
  if (!webgpuUsableForProviderSelection() || explicitWasmOnlyOrtEpPin()) return 'wasm';
  return 'asyncify';
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

function configureNodeOrtWasmBinary(wasm: { wasmBinary?: ArrayBufferLike | Uint8Array; wasmPaths?: string | { wasm?: string; mjs?: string } }): void {
  // Node resolves `onnxruntime-web/webgpu` to the bundle whose Emscripten glue
  // is inlined (the asyncify one), and glue and binary must match, so Node
  // always feeds it the asyncify binary regardless of the requested EP.
  if (typeof document !== 'undefined' || wasm.wasmBinary || forcedOrtWasmArtifact.variant !== 'bundled') return;
  const proc = globalThis.process as unknown as { cwd?: () => string; getBuiltinModule?: (name: string) => unknown } | undefined;
  const fs = (proc?.getBuiltinModule?.('node:fs') ?? proc?.getBuiltinModule?.('fs')) as
    | { existsSync?: (path: string) => boolean; readFileSync?: (path: string) => Uint8Array }
    | undefined;
  if (!proc?.cwd || !fs?.existsSync || !fs.readFileSync) return;
  const wasmPath = `${proc.cwd()}/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm`;
  if (fs.existsSync(wasmPath)) {
    wasm.wasmBinary = fs.readFileSync(wasmPath);
    // ORT-web 1.25's Node path may still consult wasmPaths in some import modes.
    wasm.wasmPaths = { wasm: wasmPath };
  }
}

function browserThreadedWasmAvailable(): boolean {
  const isNode = typeof globalThis.process?.versions?.node === 'string';
  if (isNode) return true;
  return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined';
}

/**
 * Thread count for `auto`. Capped at 4 and never more than half the machine:
 * ORT is not a participant in the CPU broker (`src/lc0/resourceBroker.ts`), so
 * a page that is also running a threaded Stockfish must not have every core
 * claimed here. 4 was also the measured optimum for the CPU-only LC0 path on a
 * 10-core machine (6 and 8 threads both regressed); see the sweep recorded on
 * `resolveOrtWasmThreads`.
 */
function defaultAutoThreads(): number {
  const hc = typeof navigator === 'undefined' ? 2 : Number(navigator.hardwareConcurrency ?? 2);
  const cores = Math.floor(Number.isFinite(hc) ? hc : 2);
  return Math.max(1, Math.min(4, Math.floor(cores / 2)));
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
  if (raw === null) {
    // Main thread and Node stay single-threaded: the UI thread must not fan out,
    // and Node's inlined glue has no pthread sidecar to re-import.
    if (context.isBrowserMainThread || context.isNode) return 1;
    // Dev worker: ORT picks its own default from the node_modules glue.
    if (!context.builtWorker) return 0;
    // Retested 2026-07-25 against the built `searchWorker` chunk served with
    // COOP/COEP: threaded ORT no longer deadlocks in a bundled worker. The
    // 2026-06-11 workaround (`2cc2f65`) predated the self-hosted pthread
    // sidecars (`180a4a4`, `0157f71`), which is what fixed it -- the helper
    // workers now re-import the staged glue at its own deployed URL instead of
    // re-executing our worker chunk. Measured on the CPU-only pair (median of
    // 20 evals, t1-256x10 qdq8, 10-core): 1t 61.3ms, 2t 43.4ms, 3t 42.2ms,
    // 4t 40.1ms, 6t 54.0ms, 8t 44.2ms; outputs matched the single-threaded run.
    // The budget stays 1 on the WebGPU-capable (asyncify) pair: with a WebGPU
    // session, numThreads=4 *regressed* the same workload from 11.6ms to
    // 17.9ms, and the artifact is worker-global, so only a thread that can
    // never escalate to WebGPU (Safari/Firefox, or an explicit `?ep=wasm` pin)
    // opts in.
    if (!context.cpuOnlyRuntimeArtifact) return 1;
    return context.threadedAvailable ? context.autoThreads : 1;
  }
  // `auto` means "choose sensibly for me", so it obeys the same artifact rule
  // as the default: on the WebGPU-capable asyncify pair the sensible choice is
  // 1, because threads there measured as a 55% regression. An explicit integer
  // below is left as a genuine override -- someone benchmarking asyncify at 4
  // threads is stating a number, not asking for a recommendation.
  if (String(raw).toLowerCase() === 'auto') {
    if (!context.cpuOnlyRuntimeArtifact) return 1;
    return context.threadedAvailable ? context.autoThreads : 1;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return 0;
  const requested = Math.floor(parsed);
  return context.isBrowserMainThread && requested > 1 && !context.threadedAvailable ? 1 : requested;
}

function requestedOrtWasmThreads(isBrowserMainThread: boolean, isNode: boolean): number {
  if (forcedOrtWasmThreads !== null) {
    return isBrowserMainThread && forcedOrtWasmThreads > 1 && !browserThreadedWasmAvailable() ? 1 : forcedOrtWasmThreads;
  }
  return resolveOrtWasmThreads({
    raw: browserParam('ortThreads') ?? browserParam('wasmThreads') ?? envValue('ORT_INTRA_OP_NUM_THREADS') ?? envValue('ORT_NUM_THREADS') ?? null,
    isBrowserMainThread,
    isNode,
    builtWorker:
      !isBrowserMainThread && !isNode && typeof document === 'undefined' && (import.meta as unknown as { env?: { PROD?: boolean } }).env?.PROD === true,
    cpuOnlyRuntimeArtifact: ortRuntimeArtifactKindForCurrentThread() === 'wasm',
    threadedAvailable: browserThreadedWasmAvailable(),
    autoThreads: defaultAutoThreads(),
  });
}

function browserOrtWasmPaths(kind: OrtRuntimeArtifactKind): string | Record<string, string> {
  const override = browserParam('ortWasmPath');
  if (override) return override;
  // Production builds: Vite/rolldown does not emit ORT's dynamic-import glue
  // (.mjs sidecars) into /assets/, so the runtime's import() resolves to the
  // SPA-fallback HTML and initWasm fails ("no available backend found").
  // Hand ORT explicit same-origin glue and wasm URLs instead. The glue module
  // is also the Emscripten pthread bootstrap, so preserving its deployed URL
  // lets helper workers re-import that exact staged module -- which holds for
  // both staged pairs, each of which bootstraps its own em-pthread workers.
  // `location` (not `document`) detects browser-ness so this also applies
  // inside workers.
  const builtBrowser = typeof location !== 'undefined' && (import.meta as unknown as { env?: { PROD?: boolean } }).env?.PROD === true;
  if (builtBrowser) return resolveOrtPthreadRuntimeUrls('/ort/', location.href, kind);
  // Dev server: a '/ort/' prefix is blocked for source-mode module imports,
  // so let the glue resolve from node_modules and only pin the .wasm binary.
  // (Empirically the jsep glue resolves its own binary in dev and tolerates
  // this asyncify-named pin; a jsep-named pin breaks its init instead.)
  // That glue is always ORT's own asyncify module -- the only entrypoint the
  // webgpu bundle names -- and glue and binary must match, so dev stays on the
  // asyncify binary for every EP. The CPU-only build is a production saving.
  return { wasm: `/ort/${ORT_PTHREAD_WASM_FILE}` };
}

function configureOrtWasmArtifact(
  wasm: {
    simd?: boolean | 'fixed' | 'relaxed';
    wasmBinary?: ArrayBufferLike | Uint8Array;
    wasmPaths?: string | { wasm?: string; mjs?: string };
  },
  isBrowserRuntime: boolean,
): void {
  const artifact = requestedOrtWasmArtifact();
  if (artifact.variant === 'relaxed') {
    if (!supportsWasmRelaxedSimdIntegerDot()) {
      throw new Error('requested relaxed ORT WASM artifact requires i32x4.relaxed_dot_i8x16_i7x16_add support');
    }
    wasm.simd = 'relaxed';
  } else {
    wasm.simd = 'fixed';
  }
  if (artifact.variant !== 'bundled') {
    wasm.wasmBinary = undefined;
    wasm.wasmPaths = { mjs: artifact.mjsUrl!, wasm: artifact.wasmUrl! };
  } else if (isBrowserRuntime) {
    wasm.wasmPaths = browserOrtWasmPaths(ortRuntimeArtifactKindForCurrentThread());
  }
}

function requestedOrtPreferredOutputLocation(): OrtRuntimeDiagnosticOptions['preferredOutputLocation'] | undefined {
  if (forcedOrtDiagnosticsOptions?.preferredOutputLocation) return forcedOrtDiagnosticsOptions.preferredOutputLocation;
  const raw = browserParam('ortPreferredOutputLocation') ?? browserParam('preferredOutputLocation');
  if (raw === 'gpu-buffer' || raw === 'cpu-pinned' || raw === 'cpu') return raw;
  if (ortDiagnosticsParamDisabled('ortGpuOutputs')) return undefined;
  if (ortDiagnosticsParamEnabled('ortGpuOutputs', 'ortReadbackProfile')) return 'gpu-buffer';
  return undefined;
}

function requestedOrtWebGpuProfiling(): boolean {
  if (forcedOrtDiagnosticsOptions?.webgpuProfiling !== undefined) return forcedOrtDiagnosticsOptions.webgpuProfiling;
  if (ortDiagnosticsParamDisabled('ortWebGpuProfile', 'ortKernelProfile')) return false;
  return ortDiagnosticsParamEnabled('ortWebGpuProfile', 'ortKernelProfile', 'ortReadbackProfile');
}

function requestedOrtWebGpuApiInstrumentation(): boolean {
  if (forcedOrtDiagnosticsOptions?.webgpuApiInstrumentation !== undefined) return forcedOrtDiagnosticsOptions.webgpuApiInstrumentation;
  if (ortDiagnosticsParamDisabled('ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace')) return false;
  return ortDiagnosticsParamEnabled('ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace', 'ortReadbackProfile');
}

function requestedOrtLogSeverityLevel(): 0 | 1 | 2 | 3 | 4 {
  const raw = browserParam('ortLogSeverity') ?? browserParam('ortLogLevel') ?? envValue('ORT_LOG_SEVERITY_LEVEL') ?? envValue('ORT_LOG_LEVEL');
  const normalized = String(raw ?? '')
    .trim()
    .toLowerCase();
  if (!normalized) return 4;
  if (normalized === 'verbose') return 0;
  if (normalized === 'info') return 1;
  if (normalized === 'warning' || normalized === 'warn') return 2;
  if (normalized === 'fatal') return 4;
  const parsed = Number(normalized);
  if (Number.isFinite(parsed)) {
    const value = Math.max(0, Math.min(4, Math.floor(parsed)));
    if (value === 0 || value === 1 || value === 2 || value === 3 || value === 4) return value;
  }
  return 4;
}

function requestedOrtLogLevelName(): 'verbose' | 'info' | 'warning' | 'error' | 'fatal' {
  const severity = requestedOrtLogSeverityLevel();
  if (severity <= 0) return 'verbose';
  if (severity === 1) return 'info';
  if (severity === 2) return 'warning';
  if (severity === 4) return 'fatal';
  return 'error';
}

type OrtWebGpuProfileRecord = { programName: string; kernelName: string; kernelType: string; gpuMs: number };
const ortWebGpuProfileRecords: OrtWebGpuProfileRecord[] = [];
let ortWebGpuProfileEventCount = 0;
let ortWebGpuProfileTotalMs = 0;
let ortWebGpuProfilingConfigured = false;
let previousOrtWebGpuProfilingOnData: ((data: unknown) => void) | undefined;

function recordOrtWebGpuProfileData(data: unknown): void {
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
}

function ortProfilingSummary(): OrtWebGpuProfilingSummary {
  const byProgram = new Map<string, { count: number; gpuMs: number }>();
  for (const record of ortWebGpuProfileRecords) {
    const value = byProgram.get(record.programName) ?? { count: 0, gpuMs: 0 };
    value.count += 1;
    value.gpuMs += record.gpuMs;
    byProgram.set(record.programName, value);
  }
  return {
    enabled: requestedOrtWebGpuProfiling(),
    eventCount: ortWebGpuProfileEventCount,
    kernelGpuMsTotal: Number(ortWebGpuProfileTotalMs.toFixed(6)),
    topPrograms: Array.from(byProgram.entries())
      .map(([programName, value]) => ({ programName, count: value.count, gpuMs: Number(value.gpuMs.toFixed(6)) }))
      .sort((a, b) => b.gpuMs - a.gpuMs)
      .slice(0, 12),
  };
}

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

function installWebGpuApiInstrumentation(): void {
  webGpuApiStats.enabled = requestedOrtWebGpuApiInstrumentation();
  if (!webGpuApiStats.enabled || webGpuApiInstrumentationInstalled) return;
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
          const gpuBufferUsage = (globalThis as unknown as { GPUBufferUsage?: { MAP_READ?: number } }).GPUBufferUsage;
          const mapRead = gpuBufferUsage?.MAP_READ ?? 1;
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

function configureOrtWebGpuProfiling(webgpu: { profiling?: { mode?: 'off' | 'default'; ondata?: (data: unknown) => void } } | undefined): void {
  if (!webgpu || !requestedOrtWebGpuProfiling()) return;
  webgpu.profiling ??= {};
  webgpu.profiling.mode = 'default';
  if (!ortWebGpuProfilingConfigured) {
    previousOrtWebGpuProfilingOnData = webgpu.profiling.ondata;
    webgpu.profiling.ondata = (data: unknown) => {
      recordOrtWebGpuProfileData(data);
      previousOrtWebGpuProfilingOnData?.(data);
    };
    ortWebGpuProfilingConfigured = true;
  }
}

function roundedApiStats(): OrtWebGpuApiInstrumentationSummary {
  return { ...webGpuApiStats, errors: [...webGpuApiStats.errors], mapAsyncMsTotal: Number(webGpuApiStats.mapAsyncMsTotal.toFixed(6)) };
}

export function getOrtWebGpuDiagnosticsSnapshot(): OrtWebGpuDiagnosticsSnapshot {
  return { profiling: ortProfilingSummary(), api: roundedApiStats() };
}

export async function waitForOrtWebGpuDiagnostics(): Promise<void> {
  if (!requestedOrtWebGpuProfiling()) return;
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
  await new Promise<void>((resolve) => setTimeout(resolve, 0));
}

export function subtractOrtWebGpuDiagnosticsSnapshot(after: OrtWebGpuDiagnosticsSnapshot, before: OrtWebGpuDiagnosticsSnapshot): OrtWebGpuDiagnosticsSnapshot {
  const beforePrograms = new Map(before.profiling.topPrograms.map((entry) => [entry.programName, entry]));
  return {
    profiling: {
      enabled: after.profiling.enabled,
      eventCount: after.profiling.eventCount - before.profiling.eventCount,
      kernelGpuMsTotal: Number((after.profiling.kernelGpuMsTotal - before.profiling.kernelGpuMsTotal).toFixed(6)),
      topPrograms: after.profiling.topPrograms
        .map((entry) => {
          const prev = beforePrograms.get(entry.programName);
          return { programName: entry.programName, count: entry.count - (prev?.count ?? 0), gpuMs: Number((entry.gpuMs - (prev?.gpuMs ?? 0)).toFixed(6)) };
        })
        .filter((entry) => entry.count > 0 || entry.gpuMs > 0)
        .sort((a, b) => b.gpuMs - a.gpuMs)
        .slice(0, 12),
    },
    api: {
      ...after.api,
      submitCount: after.api.submitCount - before.api.submitCount,
      submittedCommandBufferCount: after.api.submittedCommandBufferCount - before.api.submittedCommandBufferCount,
      mapAsyncCount: after.api.mapAsyncCount - before.api.mapAsyncCount,
      mapAsyncMsTotal: Number((after.api.mapAsyncMsTotal - before.api.mapAsyncMsTotal).toFixed(6)),
      copyBufferToBufferCount: after.api.copyBufferToBufferCount - before.api.copyBufferToBufferCount,
      copyBufferToBufferBytes: after.api.copyBufferToBufferBytes - before.api.copyBufferToBufferBytes,
      createBufferCount: after.api.createBufferCount - before.api.createBufferCount,
      createBufferBytes: after.api.createBufferBytes - before.api.createBufferBytes,
      mapReadBufferCount: after.api.mapReadBufferCount - before.api.mapReadBufferCount,
      mapReadBufferBytes: after.api.mapReadBufferBytes - before.api.mapReadBufferBytes,
      computePipelineCreateCount: after.api.computePipelineCreateCount - before.api.computePipelineCreateCount,
      computePipelineCreateAsyncCount: after.api.computePipelineCreateAsyncCount - before.api.computePipelineCreateAsyncCount,
    },
  };
}

function configureOrtRuntime() {
  const env = ort.env as unknown as { logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal' };
  env.logLevel = requestedOrtLogLevelName();
  const wasm = ort.env.wasm as unknown as {
    numThreads?: number;
    proxy?: boolean;
    simd?: boolean | 'fixed' | 'relaxed';
    wasmBinary?: ArrayBufferLike | Uint8Array;
    wasmPaths?: string | { wasm?: string; mjs?: string };
  };
  configureNodeOrtWasmBinary(wasm);
  const isBrowserMainThread = typeof document !== 'undefined';
  const isNode = typeof document === 'undefined' && !!globalThis.process?.versions?.node;
  const isBrowserRuntime = !isNode && typeof location !== 'undefined';
  configureOrtWasmArtifact(wasm, isBrowserRuntime);
  const threads = requestedOrtWasmThreads(isBrowserMainThread, isNode);
  if (threads > 0) wasm.numThreads = threads;
  if (isBrowserMainThread) {
    // Threaded ORT WASM requires cross-origin isolation / SharedArrayBuffer.
    // Keep proxy disabled; users can opt into pthread workers with ?ortThreads=auto or ?ortThreads=N.
    wasm.proxy = false;
  }
  installWebGpuApiInstrumentation();
  const webgpu = ort.env.webgpu as unknown as
    | { powerPreference?: 'low-power' | 'high-performance'; profiling?: { mode?: 'off' | 'default'; ondata?: (data: unknown) => void } }
    | undefined;
  if (webgpu && requestedOrtExecutionProvider() !== 'wasm') webgpu.powerPreference = 'high-performance';
  configureOrtWebGpuProfiling(webgpu);
}

export function sessionOptions(executionProviders = resolvedOrtExecutionProviders()): ort.InferenceSession.SessionOptions {
  configureOrtRuntime();
  const threads = requestedOrtWasmThreads(typeof document !== 'undefined', typeof document === 'undefined' && !!globalThis.process?.versions?.node);
  const opts: ort.InferenceSession.SessionOptions = { graphOptimizationLevel: 'all', executionProviders, logSeverityLevel: requestedOrtLogSeverityLevel() };
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
  const requested = requestedOrtExecutionProvider();
  const usesWebGpuProvider = providers.includes('webgpu');
  const message = usesWebGpuProvider
    ? 'Centipawn ORT: session ready with WebGPU provider requested/accepted.'
    : 'Centipawn ORT: session ready with WASM provider.';
  console.info(message, {
    requestedEp: requested,
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

function summarizeGpuAdapter(adapter: unknown): string {
  if (!adapter || typeof adapter !== 'object') return String(adapter);
  const rec = adapter as Record<string, unknown>;
  const parts: string[] = [];
  for (const key of ['name', 'vendor', 'architecture', 'device', 'description']) {
    if (typeof rec[key] === 'string' && rec[key]) parts.push(`${key}=${rec[key]}`);
  }
  if (rec.info && typeof rec.info === 'object') {
    const info = rec.info as Record<string, unknown>;
    for (const key of ['vendor', 'architecture', 'device', 'description']) {
      if (typeof info[key] === 'string' && info[key]) parts.push(`${key}=${info[key]}`);
    }
  }
  return parts.join(' ') || Object.prototype.toString.call(adapter);
}

function stringArrayFromSetLike(value: unknown): string[] | undefined {
  if (!value || typeof value !== 'object') return undefined;
  try {
    const iterable = value as Iterable<unknown>;
    const items = Array.from(iterable, (feature) => String(feature))
      .filter(Boolean)
      .sort();
    return items.length ? items : undefined;
  } catch {
    return undefined;
  }
}

function selectedGpuLimits(value: unknown): Record<string, number> | undefined {
  if (!value || typeof value !== 'object') return undefined;
  const rec = value as Record<string, unknown>;
  const keys = [
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
  const limits: Record<string, number> = {};
  for (const key of keys) {
    const raw = rec[key];
    if (typeof raw === 'number' && Number.isFinite(raw)) limits[key] = raw;
  }
  return Object.keys(limits).length ? limits : undefined;
}

export async function collectOrtRuntimeDiagnostics(options: { probeAdapter?: boolean } = {}): Promise<OrtRuntimeDiagnostics> {
  configureOrtRuntime();
  const wasm = ort.env.wasm as unknown as { numThreads?: number; proxy?: boolean };
  const webgpu = ort.env.webgpu as unknown as { powerPreference?: string } | undefined;
  const artifact = requestedOrtWasmArtifact();
  const diag: OrtRuntimeDiagnostics = {
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
    ...(lastOrtDequantFold ? { dequantFold: { ...lastOrtDequantFold } } : {}),
  };
  if (options.probeAdapter && webgpuAvailable()) {
    try {
      const adapter = await webgpuNavigator()?.requestAdapter?.({ powerPreference: 'high-performance' });
      const rec = adapter as Record<string, unknown> | null | undefined;
      let info: unknown = rec?.info;
      const requestAdapterInfo = rec?.requestAdapterInfo;
      if (!info && typeof requestAdapterInfo === 'function') {
        try {
          info = await (requestAdapterInfo as () => Promise<unknown>)();
        } catch {
          /* optional API */
        }
      }
      const adapterRec = adapter as Record<string, unknown> | null | undefined;
      const features = stringArrayFromSetLike(adapterRec?.features);
      const limits = selectedGpuLimits(adapterRec?.limits);
      probedWebGpuAdapterUsable = !!adapter;
      diag.resolvedExecutionProviders = resolvedOrtExecutionProviders();
      diag.describe = describeOrtBackendConfig();
      diag.adapter = adapter
        ? { ok: true, summary: summarizeGpuAdapter(adapter), ...(info ? { info } : {}), ...(features ? { features } : {}), ...(limits ? { limits } : {}) }
        : { ok: false, error: 'navigator.gpu.requestAdapter returned null' };
    } catch (err) {
      probedWebGpuAdapterUsable = false;
      diag.resolvedExecutionProviders = resolvedOrtExecutionProviders();
      diag.describe = describeOrtBackendConfig();
      diag.adapter = { ok: false, error: err instanceof Error ? err.message : String(err) };
    }
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
async function ensureWebGpuAdapterProbed(): Promise<void> {
  if (probedWebGpuAdapterUsable !== null) return;
  if (!webgpuAvailable()) return;
  const requested = requestedOrtExecutionProvider();
  if (requested === 'wasm' || requested === 'webgpu') return;
  try {
    probedWebGpuAdapterUsable = !!(await webgpuNavigator()?.requestAdapter?.({ powerPreference: 'high-performance' }));
  } catch {
    probedWebGpuAdapterUsable = false;
  }
}

export async function createOrtSession(modelPath: OrtModelInput): Promise<ort.InferenceSession> {
  // Must precede resolvedOrtExecutionProviders(): both it and the artifact
  // selector read the probe result, and the locks below freeze that choice.
  await ensureWebGpuAdapterProbed();
  const providers = resolvedOrtExecutionProviders();
  // Both locks run before the first InferenceSession.create so the artifact is
  // decided (and frozen) before ORT initializes its worker-global wasm module;
  // the WebGPU->WASM fallback below then reuses the same binary.
  lockRequestedOrtWasmArtifact();
  lockOrtRuntimeArtifactKind(providers);
  const prepared = await prepareOrtModelInput(modelPath);
  const wasmFallback = shouldFallbackToWasmAfterOrtFailure(requestedOrtExecutionProvider(), providers);
  // Attempt order: folded model on the requested providers; the unfolded model on the same
  // providers if ORT rejected the folded rewrite; then WASM on the unfolded model when the
  // requested provider preference allows a fallback.
  const attempts: { input: OrtModelInput; providers: string[]; note: (message: string | null) => string | undefined }[] = [
    { input: prepared.input, providers, note: () => describeDequantFold() },
  ];
  if (prepared.original !== undefined) {
    attempts.push({ input: prepared.original, providers, note: (message) => `unfolded model after the folded one failed to load: ${message}` });
  }
  if (wasmFallback) {
    attempts.push({ input: prepared.original ?? prepared.input, providers: ['wasm'], note: (message) => `fallback after WebGPU failure: ${message}` });
  }
  let lastMessage: string | null = null;
  for (let i = 0; i < attempts.length; i++) {
    const attempt = attempts[i];
    const t0 = typeof performance === 'undefined' ? Date.now() : performance.now();
    try {
      const session = await ort.InferenceSession.create(attempt.input as never, sessionOptions(attempt.providers));
      const t1 = typeof performance === 'undefined' ? Date.now() : performance.now();
      createdOrtSessions += 1;
      recordSessionAttempt(attempt.providers, true, t1 - t0);
      lastOrtExecutionProviders = attempt.providers;
      lastOrtRuntimeFallback =
        attempt.providers === providers ? undefined : { at: new Date().toISOString(), from: 'webgpu', to: 'wasm', reason: lastMessage ?? '' };
      logOrtSessionReady(attempt.providers, t1 - t0, attempt.note(lastMessage));
      return session;
    } catch (err) {
      const t1 = typeof performance === 'undefined' ? Date.now() : performance.now();
      lastMessage = err instanceof Error ? err.message : String(err);
      recordSessionAttempt(attempt.providers, false, t1 - t0, lastMessage);
      const next = attempts[i + 1];
      if (!next) throw err;
      if (next.input !== attempt.input) {
        discardDequantFold(lastMessage);
        console.warn(`Centipawn: ORT session failed on the dequantize-folded model; retrying with the original bytes. ${lastMessage}`);
      } else {
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
