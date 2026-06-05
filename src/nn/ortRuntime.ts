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
  wasm: { numThreads?: number; proxy?: boolean; sharedArrayBuffer?: boolean; threadedAvailable?: boolean };
  webgpuEnv?: { powerPreference?: string; profilingMode?: string; preferredOutputLocation?: string; apiInstrumentation?: boolean };
  adapter?: OrtWebGpuAdapterDiagnostics;
  sessions: { created: number; released: number; active: number };
  sessionAttempts: OrtSessionAttempt[];
  webgpuDiagnostics?: OrtWebGpuDiagnosticsSnapshot;
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

function truthyParam(value: string | null | undefined): boolean {
  if (value === null || value === undefined) return false;
  return !['0', 'false', 'no', 'off'].includes(String(value).toLowerCase());
}

function ortDiagnosticsParamEnabled(...names: string[]): boolean {
  return names.some((name) => truthyParam(browserParam(name) ?? envValue(`TINY_LEELA_${name.toUpperCase()}`)));
}

let forcedOrtExecutionProvider: OrtExecutionProviderPreference | null = null;
let forcedOrtDiagnosticsOptions: OrtRuntimeDiagnosticOptions | null = null;

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

export function setOrtRuntimeDiagnosticOptionsForCurrentThread(options: OrtRuntimeDiagnosticOptions | null): void {
  forcedOrtDiagnosticsOptions = options;
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
let createdOrtSessions = 0;
let releasedOrtSessions = 0;

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

function browserOrtWasmPaths(): string | Record<string, string> {
  const override = browserParam('ortWasmPath');
  if (override) return override;
  // Give ORT a public .wasm sidecar while letting its JS glue resolve from the
  // bundled onnxruntime-web module. A plain '/ort/' prefix makes ORT dynamically
  // import '/ort/*.mjs' from Vite public/, which dev-server blocks for source imports.
  return { wasm: '/ort/ort-wasm-simd-threaded.asyncify.wasm' };
}

function requestedOrtPreferredOutputLocation(): OrtRuntimeDiagnosticOptions['preferredOutputLocation'] | undefined {
  if (forcedOrtDiagnosticsOptions?.preferredOutputLocation) return forcedOrtDiagnosticsOptions.preferredOutputLocation;
  const raw = browserParam('ortPreferredOutputLocation') ?? browserParam('preferredOutputLocation');
  if (raw === 'gpu-buffer' || raw === 'cpu-pinned' || raw === 'cpu') return raw;
  if (ortDiagnosticsParamEnabled('ortGpuOutputs', 'ortReadbackProfile')) return 'gpu-buffer';
  return undefined;
}

function requestedOrtWebGpuProfiling(): boolean {
  return !!forcedOrtDiagnosticsOptions?.webgpuProfiling || ortDiagnosticsParamEnabled('ortWebGpuProfile', 'ortKernelProfile', 'ortReadbackProfile');
}

function requestedOrtWebGpuApiInstrumentation(): boolean {
  return !!forcedOrtDiagnosticsOptions?.webgpuApiInstrumentation || ortDiagnosticsParamEnabled('ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace', 'ortReadbackProfile');
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
    wrapPrototypeMethod(g.GPUQueue?.prototype, 'submit', (original: (buffers: unknown[]) => void) => function submit(this: unknown, buffers: unknown[]) {
      webGpuApiStats.submitCount += 1;
      webGpuApiStats.submittedCommandBufferCount += Array.isArray(buffers) ? buffers.length : 0;
      return original.call(this, buffers);
    } as never),
    wrapPrototypeMethod(g.GPUBuffer?.prototype, 'mapAsync', (original: (mode: number, offset?: number, size?: number) => Promise<void>) => async function mapAsync(this: unknown, mode: number, offset?: number, size?: number) {
      const started = tinyLeelaNowMs();
      webGpuApiStats.mapAsyncCount += 1;
      try {
        return await original.call(this, mode, offset, size);
      } finally {
        webGpuApiStats.mapAsyncMsTotal += tinyLeelaNowMs() - started;
      }
    } as never),
    wrapPrototypeMethod(g.GPUCommandEncoder?.prototype, 'copyBufferToBuffer', (original: (...args: unknown[]) => void) => function copyBufferToBuffer(this: unknown, ...args: unknown[]) {
      webGpuApiStats.copyBufferToBufferCount += 1;
      const size = Number(args[4] ?? 0);
      if (Number.isFinite(size) && size > 0) webGpuApiStats.copyBufferToBufferBytes += size;
      return original.apply(this, args);
    } as never),
    wrapPrototypeMethod(g.GPUDevice?.prototype, 'createBuffer', (original: (descriptor: { size?: number; usage?: number }) => unknown) => function createBuffer(this: unknown, descriptor: { size?: number; usage?: number }) {
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
    } as never),
    wrapPrototypeMethod(g.GPUDevice?.prototype, 'createComputePipeline', (original: (descriptor: unknown) => unknown) => function createComputePipeline(this: unknown, descriptor: unknown) {
      webGpuApiStats.computePipelineCreateCount += 1;
      return original.call(this, descriptor);
    } as never),
    wrapPrototypeMethod(g.GPUDevice?.prototype, 'createComputePipelineAsync', (original: (descriptor: unknown) => Promise<unknown>) => function createComputePipelineAsync(this: unknown, descriptor: unknown) {
      webGpuApiStats.computePipelineCreateAsyncCount += 1;
      return original.call(this, descriptor);
    } as never),
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
      topPrograms: after.profiling.topPrograms.map((entry) => {
        const prev = beforePrograms.get(entry.programName);
        return { programName: entry.programName, count: entry.count - (prev?.count ?? 0), gpuMs: Number((entry.gpuMs - (prev?.gpuMs ?? 0)).toFixed(6)) };
      }).filter((entry) => entry.count > 0 || entry.gpuMs > 0).sort((a, b) => b.gpuMs - a.gpuMs).slice(0, 12),
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
  const wasm = ort.env.wasm as unknown as { numThreads?: number; proxy?: boolean; wasmBinary?: ArrayBufferLike | Uint8Array; wasmPaths?: string | Record<string, string> };
  configureNodeOrtWasmBinary(wasm);
  const isBrowserMainThread = typeof document !== 'undefined';
  const isNode = typeof document === 'undefined' && !!globalThis.process?.versions?.node;
  const isBrowserRuntime = !isNode && typeof location !== 'undefined';
  if (isBrowserRuntime) wasm.wasmPaths = browserOrtWasmPaths();
  const threads = requestedOrtWasmThreads(isBrowserMainThread, isNode);
  if (threads > 0) wasm.numThreads = threads;
  if (isBrowserMainThread) {
    // Threaded ORT WASM requires cross-origin isolation / SharedArrayBuffer.
    // Keep proxy disabled; users can opt into pthread workers with ?ortThreads=auto or ?ortThreads=N.
    wasm.proxy = false;
  }
  installWebGpuApiInstrumentation();
  const webgpu = ort.env.webgpu as unknown as { powerPreference?: 'low-power' | 'high-performance'; profiling?: { mode?: 'off' | 'default'; ondata?: (data: unknown) => void } } | undefined;
  if (webgpu && requestedOrtExecutionProvider() !== 'wasm') webgpu.powerPreference = 'high-performance';
  configureOrtWebGpuProfiling(webgpu);
}

export function sessionOptions(executionProviders = resolvedOrtExecutionProviders()): ort.InferenceSession.SessionOptions {
  configureOrtRuntime();
  const threads = requestedOrtWasmThreads(typeof document !== 'undefined', typeof document === 'undefined' && !!globalThis.process?.versions?.node);
  const opts: ort.InferenceSession.SessionOptions = { graphOptimizationLevel: 'all', executionProviders };
  const preferredOutputLocation = requestedOrtPreferredOutputLocation();
  if (preferredOutputLocation) opts.preferredOutputLocation = preferredOutputLocation;
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
    ...(webgpu ? { webgpuEnv: { powerPreference: webgpu.powerPreference, profilingMode: (webgpu as { profiling?: { mode?: string } }).profiling?.mode, preferredOutputLocation: requestedOrtPreferredOutputLocation(), apiInstrumentation: requestedOrtWebGpuApiInstrumentation() } } : {}),
    sessions: { created: createdOrtSessions, released: releasedOrtSessions, active: Math.max(0, createdOrtSessions - releasedOrtSessions) },
    sessionAttempts: sessionAttempts.map((x) => ({ ...x, providers: [...x.providers] })),
    webgpuDiagnostics: getOrtWebGpuDiagnosticsSnapshot(),
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
    createdOrtSessions += 1;
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
    createdOrtSessions += 1;
    recordSessionAttempt(['wasm'], true, fallbackT1 - fallbackT0);
    lastOrtExecutionProviders = ['wasm'];
    logOrtSessionReady(['wasm'], fallbackT1 - fallbackT0, `fallback after WebGPU failure: ${message}`);
    return session;
  }
}

export async function releaseOrtSession(session: ort.InferenceSession): Promise<void> {
  const maybe = session as ort.InferenceSession & { release?: () => Promise<void> | void };
  if (typeof maybe.release === 'function') await maybe.release();
  releasedOrtSessions += 1;
}
