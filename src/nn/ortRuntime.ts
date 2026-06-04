export * from 'onnxruntime-web/webgpu';
import * as ort from 'onnxruntime-web/webgpu';

export type OrtExecutionProviderPreference = 'wasm' | 'webgpu' | 'webgpu,wasm' | 'auto';

export type OrtSessionAttempt = {
  at: string;
  providers: string[];
  ok: boolean;
  ms: number;
  error?: string;
};

export type OrtWebGpuAdapterDiagnostics = {
  ok: boolean;
  summary?: string;
  info?: unknown;
  features?: string[];
  limits?: Record<string, number>;
  error?: string;
};

export type OrtRuntimeDiagnostics = {
  requestedEp: OrtExecutionProviderPreference;
  resolvedExecutionProviders: string[];
  describe: string;
  webgpuAvailable: boolean;
  secureContext?: boolean;
  crossOriginIsolated?: boolean;
  userAgent?: string;
  wasm: { numThreads?: number; proxy?: boolean; sharedArrayBuffer?: boolean; threadedAvailable?: boolean };
  webgpuEnv?: { powerPreference?: string };
  adapter?: OrtWebGpuAdapterDiagnostics;
  sessionAttempts: OrtSessionAttempt[];
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
  return String(value ?? '').toLowerCase().split(/[,+\s]+/).map((s) => s.trim()).filter(Boolean);
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
  console.info(`Tiny Leela latency: ${label}`, Object.fromEntries(Object.entries(payload).map(([key, value]) => [key, typeof value === 'number' ? roundMs(value) : value])));
}

let forcedOrtExecutionProvider: OrtExecutionProviderPreference | null = null;

export function setRequestedOrtExecutionProviderForCurrentThread(value: OrtExecutionProviderPreference | null): void {
  forcedOrtExecutionProvider = value;
}

function normalizeEp(value: string | null | undefined): OrtExecutionProviderPreference {
  const raw = String(value ?? '').trim().toLowerCase();
  if (!raw) return 'auto';
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'webgpu,wasm' || raw === 'webgpu+wasm' || raw === 'gpu,wasm' || raw === 'gpu+wasm') return 'webgpu,wasm';
  if (raw === 'auto') return 'auto';
  return 'wasm';
}

export function requestedOrtExecutionProvider(): OrtExecutionProviderPreference {
  if (forcedOrtExecutionProvider) return forcedOrtExecutionProvider;
  return normalizeEp(
    browserParam('ortEp')
      ?? browserParam('ep')
      ?? browserParam('executionProviders')
      ?? envValue('TINY_LEELA_ORT_EP')
      ?? envValue('ORT_EXECUTION_PROVIDERS')
  );
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
  if (requested === 'webgpu') return webgpuUsableForProviderSelection() ? ['webgpu'] : ['wasm'];
  if (requested === 'webgpu,wasm') return webgpuUsableForProviderSelection() ? ['webgpu', 'wasm'] : ['wasm'];
  if (requested === 'auto') return webgpuUsableForProviderSelection() ? ['webgpu', 'wasm'] : ['wasm'];
  return ['wasm'];
}

let lastOrtExecutionProviders: string[] | null = null;
const sessionAttempts: OrtSessionAttempt[] = [];

export function describeOrtBackendConfig(): string {
  const requested = requestedOrtExecutionProvider();
  const resolved = (lastOrtExecutionProviders ?? resolvedOrtExecutionProviders()).join(',');
  return requested === 'wasm' && resolved === 'wasm' ? 'wasm' : `${requested}->${resolved}`;
}

function configureNodeOrtWasmBinary(wasm: { wasmBinary?: ArrayBufferLike | Uint8Array; wasmPaths?: string | { wasm?: string } }): void {
  if (typeof document !== 'undefined' || wasm.wasmBinary) return;
  const proc = globalThis.process as unknown as { cwd?: () => string; getBuiltinModule?: (name: string) => unknown } | undefined;
  const fs = (proc?.getBuiltinModule?.('node:fs') ?? proc?.getBuiltinModule?.('fs')) as { existsSync?: (path: string) => boolean; readFileSync?: (path: string) => Uint8Array } | undefined;
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

function defaultAutoThreads(): number {
  const hc = typeof navigator === 'undefined' ? 2 : Number(navigator.hardwareConcurrency ?? 2);
  return Math.max(2, Math.min(4, Math.floor(Number.isFinite(hc) ? hc - 1 : 2)));
}

function requestedOrtWasmThreads(isBrowserMainThread: boolean, isNode: boolean): number {
  const raw = browserParam('ortThreads')
    ?? browserParam('wasmThreads')
    ?? envValue('ORT_INTRA_OP_NUM_THREADS')
    ?? envValue('ORT_NUM_THREADS')
    ?? (isBrowserMainThread || isNode ? '1' : '0');
  if (String(raw).toLowerCase() === 'auto') return browserThreadedWasmAvailable() ? defaultAutoThreads() : 1;
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) return 0;
  const requested = Math.floor(parsed);
  return isBrowserMainThread && requested > 1 && !browserThreadedWasmAvailable() ? 1 : requested;
}

function configureOrtRuntime() {
  const wasm = ort.env.wasm as unknown as { numThreads?: number; proxy?: boolean; wasmBinary?: ArrayBufferLike | Uint8Array; wasmPaths?: string | Record<string, string> };
  configureNodeOrtWasmBinary(wasm);
  if (typeof document !== 'undefined') wasm.wasmPaths = browserParam('ortWasmPath') ?? '/ort/';
  const isBrowserMainThread = typeof document !== 'undefined';
  const isNode = typeof document === 'undefined' && !!globalThis.process?.versions?.node;
  const threads = requestedOrtWasmThreads(isBrowserMainThread, isNode);
  if (threads > 0) wasm.numThreads = threads;
  if (isBrowserMainThread) {
    // Threaded ORT WASM requires cross-origin isolation / SharedArrayBuffer.
    // Keep proxy disabled; users can opt into pthread workers with ?ortThreads=auto or ?ortThreads=N.
    wasm.proxy = false;
  }
  const webgpu = ort.env.webgpu as unknown as { powerPreference?: 'low-power' | 'high-performance' } | undefined;
  if (webgpu && requestedOrtExecutionProvider() !== 'wasm') webgpu.powerPreference = 'high-performance';
}

export function sessionOptions(executionProviders = resolvedOrtExecutionProviders()): ort.InferenceSession.SessionOptions {
  configureOrtRuntime();
  const threads = requestedOrtWasmThreads(typeof document !== 'undefined', typeof document === 'undefined' && !!globalThis.process?.versions?.node);
  const opts: ort.InferenceSession.SessionOptions = { graphOptimizationLevel: 'all', executionProviders };
  if (threads > 0) {
    opts.intraOpNumThreads = threads;
    opts.interOpNumThreads = 1;
  }
  return opts;
}

function recordSessionAttempt(providers: string[], ok: boolean, ms: number, error?: string) {
  sessionAttempts.push({ at: new Date().toISOString(), providers: [...providers], ok, ms, ...(error ? { error } : {}) });
  while (sessionAttempts.length > 32) sessionAttempts.shift();
}

function logOrtSessionReady(providers: string[], ms: number, note?: string) {
  const requested = requestedOrtExecutionProvider();
  const usesWebGpuProvider = providers.includes('webgpu');
  const message = usesWebGpuProvider
    ? 'Tiny Leela ORT: session ready with WebGPU provider requested/accepted.'
    : 'Tiny Leela ORT: session ready with WASM provider.';
  console.info(message, {
    requestedEp: requested,
    sessionProviders: providers,
    webgpuAvailable: webgpuAvailable(),
    describe: describeOrtBackendConfig(),
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
    const items = Array.from(iterable, (feature) => String(feature)).filter(Boolean).sort();
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
    },
    ...(webgpu ? { webgpuEnv: { powerPreference: webgpu.powerPreference } } : {}),
    sessionAttempts: sessionAttempts.map((x) => ({ ...x, providers: [...x.providers] })),
  };
  if (options.probeAdapter && webgpuAvailable()) {
    try {
      const adapter = await webgpuNavigator()?.requestAdapter?.({ powerPreference: 'high-performance' });
      const rec = adapter as Record<string, unknown> | null | undefined;
      let info: unknown = rec?.info;
      const requestAdapterInfo = rec?.requestAdapterInfo;
      if (!info && typeof requestAdapterInfo === 'function') {
        try { info = await (requestAdapterInfo as () => Promise<unknown>)(); } catch { /* optional API */ }
      }
      const adapterRec = adapter as Record<string, unknown> | null | undefined;
      const features = stringArrayFromSetLike(adapterRec?.features);
      const limits = selectedGpuLimits(adapterRec?.limits);
      probedWebGpuAdapterUsable = !!adapter;
      diag.resolvedExecutionProviders = resolvedOrtExecutionProviders();
      diag.describe = describeOrtBackendConfig();
      diag.adapter = adapter ? { ok: true, summary: summarizeGpuAdapter(adapter), ...(info ? { info } : {}), ...(features ? { features } : {}), ...(limits ? { limits } : {}) } : { ok: false, error: 'navigator.gpu.requestAdapter returned null' };
    } catch (err) {
      probedWebGpuAdapterUsable = false;
      diag.resolvedExecutionProviders = resolvedOrtExecutionProviders();
      diag.describe = describeOrtBackendConfig();
      diag.adapter = { ok: false, error: err instanceof Error ? err.message : String(err) };
    }
  }
  return diag;
}

export async function createOrtSession(modelPath: string | Uint8Array | ArrayBuffer): Promise<ort.InferenceSession> {
  const providers = resolvedOrtExecutionProviders();
  const t0 = typeof performance === 'undefined' ? Date.now() : performance.now();
  try {
    const session = await ort.InferenceSession.create(modelPath as never, sessionOptions(providers));
    const t1 = typeof performance === 'undefined' ? Date.now() : performance.now();
    recordSessionAttempt(providers, true, t1 - t0);
    lastOrtExecutionProviders = providers;
    logOrtSessionReady(providers, t1 - t0);
    return session;
  } catch (err) {
    const t1 = typeof performance === 'undefined' ? Date.now() : performance.now();
    const message = err instanceof Error ? err.message : String(err);
    recordSessionAttempt(providers, false, t1 - t0, message);
    if (!providers.includes('webgpu')) throw err;
    console.warn(`Tiny Leela: ORT WebGPU session failed; falling back to WASM. ${message}`);
    const fallbackT0 = typeof performance === 'undefined' ? Date.now() : performance.now();
    const session = await ort.InferenceSession.create(modelPath as never, sessionOptions(['wasm']));
    const fallbackT1 = typeof performance === 'undefined' ? Date.now() : performance.now();
    recordSessionAttempt(['wasm'], true, fallbackT1 - fallbackT0);
    lastOrtExecutionProviders = ['wasm'];
    logOrtSessionReady(['wasm'], fallbackT1 - fallbackT0, `fallback after WebGPU failure: ${message}`);
    return session;
  }
}
