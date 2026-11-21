export type OrtExecutionProviderPreference = 'wasm' | 'webgpu' | 'webgpu,wasm' | 'auto';
export type OrtGraphOptimizationLevel = 'disabled' | 'basic' | 'extended' | 'layout' | 'all';

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

export type OrtSessionDiagnostics = {
  /** LC0 worker's effective IO, when available (not merely the requested mode). */
  ioMode?: 'cpu' | 'gpu' | 'gpu-capture';
  fixedBatchSize?: number;
  graphOptimizationLevel: OrtGraphOptimizationLevel;
  dequantFoldEnabled: boolean;
  dequantFold?: OrtDequantFoldSummary;
};

export type OrtRuntimeDiagnosticOptions = {
  graphOptimizationLevel?: OrtGraphOptimizationLevel;
  /** Request JSEP kernel timestamp callbacks (unavailable on the native WebGPU EP in ORT 1.27). */
  webgpuProfiling?: boolean;
  /** Wrap browser WebGPU APIs before ORT initializes to count submits/maps/copies. */
  webgpuApiInstrumentation?: boolean;
  /** Ask ORT WebGPU to return GPU-backed outputs so tensor.getData() can be timed separately. */
  preferredOutputLocation?: 'cpu' | 'cpu-pinned' | 'gpu-buffer';
};

export type OrtSessionPlacement = {
  /** Primary provider followed by any fallback provider for the next createOrtSession call. */
  providers: string[];
  /** Lower-cased GPUAdapterInfo.vendor ('apple', 'nvidia', ...); only probed when providers include webgpu. */
  adapterVendor?: string;
};

export const ORT_TRUTHY_FLAGS = new Set(['1', 'true', 'yes', 'on']);
export const ORT_FALSY_FLAGS = new Set(['0', 'false', 'no', 'off']);

export function ortBrowserParam(name: string): string | null {
  try {
    if (typeof location === 'undefined') return null;
    return new URLSearchParams(location.search).get(name);
  } catch {
    return null;
  }
}

export function ortEnvValue(name: string): string | undefined {
  return globalThis.process?.env?.[name];
}

export function ortDebugParam(name: string): string | null {
  return ortBrowserParam(name) ?? ortEnvValue(`TINY_LEELA_${name.toUpperCase()}`) ?? null;
}

export function ortDiagFlagEnabled(disableNames: string[], enableNames: string[]): boolean {
  if (
    disableNames.some((name) => {
      const value = ortDebugParam(name);
      return value !== null && ORT_FALSY_FLAGS.has(value.toLowerCase());
    })
  ) {
    return false;
  }
  return enableNames.some((name) => {
    const value = ortDebugParam(name);
    return value !== null && !ORT_FALSY_FLAGS.has(value.toLowerCase());
  });
}

let forcedOrtExecutionProvider: OrtExecutionProviderPreference | null = null;
let forcedOrtDiagnosticsOptions: OrtRuntimeDiagnosticOptions | null = null;
let forcedOrtDequantFold: boolean | null = null;

export function setRequestedOrtExecutionProviderForCurrentThread(value: OrtExecutionProviderPreference | null): void {
  forcedOrtExecutionProvider = value;
}

export function setOrtRuntimeDiagnosticOptionsForCurrentThread(options: OrtRuntimeDiagnosticOptions | null): void {
  forcedOrtDiagnosticsOptions = options;
}

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
  const raw = String(ortBrowserParam('ortDequantFold') ?? ortEnvValue('ORT_DEQUANT_FOLD') ?? '')
    .trim()
    .toLowerCase();
  if (ORT_FALSY_FLAGS.has(raw)) return false;
  if (ORT_TRUTHY_FLAGS.has(raw)) return true;
  return defaultEnabled;
}

function normalizeOrtEp(value: string | null | undefined): OrtExecutionProviderPreference {
  const raw = String(value ?? '')
    .trim()
    .toLowerCase();
  if (!raw || raw === 'auto') return 'auto';
  if (raw === 'webgpu' || raw === 'gpu') return 'webgpu';
  if (raw === 'webgpu,wasm' || raw === 'webgpu+wasm' || raw === 'gpu,wasm' || raw === 'gpu+wasm') return 'webgpu,wasm';
  return 'wasm';
}

function ambientOrtExecutionProviderParam(): string | null | undefined {
  return (
    ortBrowserParam('ortEp') ??
    ortBrowserParam('ep') ??
    ortBrowserParam('executionProviders') ??
    ortEnvValue('TINY_LEELA_ORT_EP') ??
    ortEnvValue('ORT_EXECUTION_PROVIDERS')
  );
}

export function requestedOrtExecutionProvider(): OrtExecutionProviderPreference {
  return forcedOrtExecutionProvider ?? normalizeOrtEp(ambientOrtExecutionProviderParam());
}

export function explicitWasmOnlyOrtEpPin(): boolean {
  const raw = ambientOrtExecutionProviderParam();
  return raw !== null && raw !== undefined && String(raw).trim() !== '' && normalizeOrtEp(raw) === 'wasm';
}

export function selectOrtExecutionProviders(requested: OrtExecutionProviderPreference, webgpuUsable: boolean): string[] {
  if (requested === 'webgpu') return ['webgpu'];
  if (requested !== 'wasm' && webgpuUsable) return ['webgpu', 'wasm'];
  return ['wasm'];
}

/** Worker init pins this option explicitly because its location does not contain the page query. */
export function requestedOrtGraphOptimizationLevel(): OrtGraphOptimizationLevel {
  const value = forcedOrtDiagnosticsOptions?.graphOptimizationLevel ?? ortBrowserParam('ortGraphOpt');
  return value === 'disabled' || value === 'basic' || value === 'extended' || value === 'layout' || value === 'all' ? value : 'all';
}

export function requestedOrtPreferredOutputLocation(): OrtRuntimeDiagnosticOptions['preferredOutputLocation'] | undefined {
  if (forcedOrtDiagnosticsOptions?.preferredOutputLocation) return forcedOrtDiagnosticsOptions.preferredOutputLocation;
  const raw = ortBrowserParam('ortPreferredOutputLocation') ?? ortBrowserParam('preferredOutputLocation');
  if (raw === 'gpu-buffer' || raw === 'cpu-pinned' || raw === 'cpu') return raw;
  return ortDiagFlagEnabled(['ortGpuOutputs'], ['ortGpuOutputs', 'ortReadbackProfile']) ? 'gpu-buffer' : undefined;
}

export function requestedOrtWebGpuProfiling(): boolean {
  if (forcedOrtDiagnosticsOptions?.webgpuProfiling !== undefined) return forcedOrtDiagnosticsOptions.webgpuProfiling;
  return ortDiagFlagEnabled(['ortWebGpuProfile', 'ortKernelProfile'], ['ortWebGpuProfile', 'ortKernelProfile', 'ortReadbackProfile']);
}

export function requestedOrtWebGpuApiInstrumentation(): boolean {
  if (forcedOrtDiagnosticsOptions?.webgpuApiInstrumentation !== undefined) return forcedOrtDiagnosticsOptions.webgpuApiInstrumentation;
  return ortDiagFlagEnabled(['ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace'], ['ortMonkeyPatchWebGpu', 'ortWebGpuApiTrace', 'ortReadbackProfile']);
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

export function webgpuNavigator(): { requestAdapter?: (options?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<unknown> } | undefined {
  return typeof navigator !== 'undefined'
    ? (navigator as Navigator & { gpu?: { requestAdapter?: (options?: { powerPreference?: 'high-performance' | 'low-power' }) => Promise<unknown> } }).gpu
    : undefined;
}

export function webgpuAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

export async function requestOrtWebGpuAdapter(): Promise<unknown> {
  return webgpuNavigator()?.requestAdapter?.({ powerPreference: 'high-performance' });
}

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

export async function resolveFreshWorkerOrtSessionPlacement(requested: OrtExecutionProviderPreference): Promise<OrtSessionPlacement> {
  const adapter = requested === 'wasm' ? undefined : await requestOrtWebGpuAdapter().catch(() => undefined);
  const providers = selectOrtExecutionProviders(requested, !!adapter && gpuAdapterShaderF16(adapter) !== false);
  const adapterVendor = providers.includes('webgpu') ? await gpuAdapterVendor(adapter) : undefined;
  return adapterVendor ? { providers, adapterVendor } : { providers };
}

export async function probeOrtWebGpuShaderF16Support(): Promise<boolean> {
  try {
    if (!webgpuAvailable()) return false;
    const adapter = await requestOrtWebGpuAdapter();
    return !!adapter && gpuAdapterShaderF16(adapter) !== false;
  } catch {
    return false;
  }
}
