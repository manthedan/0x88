import { supportsWasmRelaxedSimd, supportsWasmSimd } from './wasmFeatures.ts';
export { supportsWasmRelaxedSimd, supportsWasmSimd } from './wasmFeatures.ts';

export type PlentyChessVariantKey = 'emscripten' | 'emscripten-sse41' | 'emscripten-relaxed' | 'custom';
export type PlentyChessAssetStatus = 'unknown' | 'checking' | 'present' | 'missing';

export interface PlentyChessVariant {
  key: PlentyChessVariantKey;
  label: string;
  /** Emscripten sidecar WASM URL. */
  wasmUrl: string;
  /** Emscripten JS glue URL for the currently-smoked browser worker path. */
  jsUrl: string;
  /** Emscripten preload data URL containing the processed NNUE. */
  dataUrl: string;
  sourceNetworkUrl?: string;
  note: string;
}

export const PLENTYCHESS_EMSCRIPTEN_JS_URL = '/plentychess/plentychess-emscripten.js';
export const PLENTYCHESS_EMSCRIPTEN_WASM_URL = '/plentychess/plentychess-emscripten.wasm';
export const PLENTYCHESS_EMSCRIPTEN_DATA_URL = '/plentychess/plentychess-emscripten.data';
export const PLENTYCHESS_EMSCRIPTEN_SSE41_JS_URL = '/plentychess/plentychess-emscripten-sse41.js';
export const PLENTYCHESS_EMSCRIPTEN_SSE41_WASM_URL = '/plentychess/plentychess-emscripten-sse41.wasm';
export const PLENTYCHESS_EMSCRIPTEN_SSE41_DATA_URL = '/plentychess/plentychess-emscripten-sse41.data';
export const PLENTYCHESS_EMSCRIPTEN_RELAXED_JS_URL = '/plentychess/plentychess-emscripten-relaxed-simd128.js';
export const PLENTYCHESS_EMSCRIPTEN_RELAXED_WASM_URL = '/plentychess/plentychess-emscripten-relaxed-simd128.wasm';
export const PLENTYCHESS_EMSCRIPTEN_RELAXED_DATA_URL = '/plentychess/plentychess-emscripten-relaxed-simd128.data';
export const PLENTYCHESS_MAIN_NETWORK = '0134-2r24-s0.bin';
export const PLENTYCHESS_SOURCE_NETWORK_URL = `https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/${PLENTYCHESS_MAIN_NETWORK}`;

const assetStatuses = new Map<string, PlentyChessAssetStatus>();
const assetChecks = new Map<string, Promise<PlentyChessAssetStatus>>();

// Current production only ships the smoked baseline Emscripten PlentyChess
// sidecars. The SSE4.1/relaxed variants remain selectable for local generated
// assets, but production should not issue doomed HEAD probes for known-unshipped
// files because devtools reports those handled fallbacks as 404 errors.
const DEPLOYED_PLENTYCHESS_URLS = new Set([
  PLENTYCHESS_EMSCRIPTEN_JS_URL,
  PLENTYCHESS_EMSCRIPTEN_WASM_URL,
  PLENTYCHESS_EMSCRIPTEN_DATA_URL,
]);

function isLocalDevelopmentOrigin(): boolean {
  if (typeof location === 'undefined') return true;
  return location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '::1' || location.hostname === '[::1]';
}

function shouldSkipKnownUnshippedProbe(variant: PlentyChessVariant): boolean {
  if (variant.key === 'custom' || isLocalDevelopmentOrigin()) return false;
  return assetUrls(variant).some((url) => url.startsWith('/plentychess/') && !DEPLOYED_PLENTYCHESS_URLS.has(url));
}

function sameOriginPlentyChessAsset(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    const base = typeof location !== 'undefined' ? location.origin : 'http://localhost';
    const url = new URL(raw, base);
    if (url.origin !== base || !url.pathname.startsWith('/plentychess/')) return undefined;
    return `${url.pathname}${url.search}`;
  } catch {
    return undefined;
  }
}

function assetUrls(variant: PlentyChessVariant): string[] {
  return [variant.jsUrl, variant.wasmUrl, variant.dataUrl];
}

function assetKey(variant: PlentyChessVariant): string {
  return assetUrls(variant).join('\n');
}

export const PLENTYCHESS_EMSCRIPTEN_VARIANT: PlentyChessVariant = {
  key: 'emscripten',
  label: 'PlentyChess Emscripten experimental',
  jsUrl: PLENTYCHESS_EMSCRIPTEN_JS_URL,
  wasmUrl: PLENTYCHESS_EMSCRIPTEN_WASM_URL,
  dataUrl: PLENTYCHESS_EMSCRIPTEN_DATA_URL,
  sourceNetworkUrl: PLENTYCHESS_SOURCE_NETWORK_URL,
  note: 'Smoked PlentyChess 7.0.66 single-thread Emscripten worker build with processed NNUE preloaded in .data.',
};

export const PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT: PlentyChessVariant = {
  key: 'emscripten-sse41',
  label: 'PlentyChess SSE4.1 Emscripten',
  jsUrl: PLENTYCHESS_EMSCRIPTEN_SSE41_JS_URL,
  wasmUrl: PLENTYCHESS_EMSCRIPTEN_SSE41_WASM_URL,
  dataUrl: PLENTYCHESS_EMSCRIPTEN_SSE41_DATA_URL,
  sourceNetworkUrl: PLENTYCHESS_SOURCE_NETWORK_URL,
  note: 'Default build plus -msse4.1: single-op convertEpi8Epi16 in the accumulator path (exact-equal semantics).',
};

export const PLENTYCHESS_EMSCRIPTEN_RELAXED_VARIANT: PlentyChessVariant = {
  key: 'emscripten-relaxed',
  label: 'PlentyChess Relaxed SIMD Emscripten',
  jsUrl: PLENTYCHESS_EMSCRIPTEN_RELAXED_JS_URL,
  wasmUrl: PLENTYCHESS_EMSCRIPTEN_RELAXED_WASM_URL,
  dataUrl: PLENTYCHESS_EMSCRIPTEN_RELAXED_DATA_URL,
  sourceNetworkUrl: PLENTYCHESS_SOURCE_NETWORK_URL,
  note: 'SSE4.1 build whose dpbusd helpers use the relaxed integer dot (exact: INPUT_QUANT=255/INPUT_SHIFT=9 keep activations in 0..127) and whose f32 tail is vectorized with relaxed madd. Requires WebAssembly Relaxed SIMD.',
};

export const PLENTYCHESS_VARIANTS: readonly PlentyChessVariant[] = [
  PLENTYCHESS_EMSCRIPTEN_VARIANT,
  PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT,
  PLENTYCHESS_EMSCRIPTEN_RELAXED_VARIANT,
];

export function normalizePlentyChessVariant(raw: string | null | undefined): PlentyChessVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  if (value === 'custom') return 'custom';
  if (value === 'emscriptenrelaxed' || value === 'relaxed' || value === 'relaxedsimd' || value === 'relaxedsimd128') return 'emscripten-relaxed';
  if (value === 'emscriptensse41' || value === 'sse41' || value === 'sse4') return 'emscripten-sse41';
  return 'emscripten';
}

export function plentyChessVariantUnsupportedReason(variant: PlentyChessVariant): string | null {
  if (variant.key === 'custom') return null;
  if (!supportsWasmSimd()) return 'requires WebAssembly SIMD';
  if (variant.key === 'emscripten-relaxed' && !supportsWasmRelaxedSimd()) return 'requires WebAssembly Relaxed SIMD';
  return null;
}

export function supportsPlentyChessVariant(variant: PlentyChessVariant): boolean {
  return plentyChessVariantUnsupportedReason(variant) === null;
}

// Promotion order: relaxed (relaxed dot + vectorized f32 tail) > sse41 >
// default. All variants are value-exact at fixed depth (40/40 parity), so
// this is a speed ladder gated by WebAssembly feature validation. Every
// bundled PlentyChess artifact requires baseline wasm SIMD; browsers without
// it keep the base option selected but disabled rather than attempting a load.
export function defaultPlentyChessVariantKey(): PlentyChessVariantKey {
  if (!supportsWasmSimd()) return 'emscripten';
  return supportsWasmRelaxedSimd() ? 'emscripten-relaxed' : 'emscripten-sse41';
}

/**
 * Resolve the feature-detected default against deployed assets: an unsupported
 * SIMD tier falls back to the base disabled option, a missing relaxed artifact
 * falls back to sse41, and a missing sse41 artifact falls back to the base
 * Emscripten build. Supported explicit selections are honored as-is.
 */
export async function resolveDefaultPlentyChessVariantAssetFallback(variant: PlentyChessVariant, explicit: boolean, onChange?: () => void): Promise<PlentyChessVariant> {
  if (!supportsPlentyChessVariant(variant)) return PLENTYCHESS_EMSCRIPTEN_VARIANT;
  if (explicit || variant.key === 'emscripten' || variant.key === 'custom') return variant;
  const status = await checkPlentyChessVariantAsset(variant, onChange);
  if (status !== 'missing') return variant;
  if (variant.key === 'emscripten-relaxed') {
    return resolveDefaultPlentyChessVariantAssetFallback(PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT, false, onChange);
  }
  return PLENTYCHESS_EMSCRIPTEN_VARIANT;
}

export function plentyChessVariantByKey(key: string): PlentyChessVariant {
  const normalized = normalizePlentyChessVariant(key);
  if (normalized === 'emscripten-sse41') return supportsWasmSimd() ? PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT : PLENTYCHESS_EMSCRIPTEN_VARIANT;
  if (normalized === 'emscripten-relaxed') return supportsWasmRelaxedSimd() ? PLENTYCHESS_EMSCRIPTEN_RELAXED_VARIANT : supportsWasmSimd() ? PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT : PLENTYCHESS_EMSCRIPTEN_VARIANT;
  if (normalized === 'custom') return {
    key: 'custom',
    label: 'PlentyChess Custom',
    jsUrl: PLENTYCHESS_EMSCRIPTEN_JS_URL,
    wasmUrl: PLENTYCHESS_EMSCRIPTEN_WASM_URL,
    dataUrl: PLENTYCHESS_EMSCRIPTEN_DATA_URL,
    sourceNetworkUrl: PLENTYCHESS_SOURCE_NETWORK_URL,
    note: 'Custom PlentyChess Emscripten JS URL.',
  };
  return PLENTYCHESS_EMSCRIPTEN_VARIANT;
}

export function hasExplicitPlentyChessVariant(params: URLSearchParams): boolean {
  return params.has('plentyChessJs') || params.has('plentyChessWasm') || params.has('plentyChessData') || params.has('plentyChessVariant') || params.has('plentychess');
}

export function plentyChessVariantFromParams(params: URLSearchParams): PlentyChessVariant {
  const customJsUrl = sameOriginPlentyChessAsset(params.get('plentyChessJs'));
  const customWasmUrl = sameOriginPlentyChessAsset(params.get('plentyChessWasm'));
  const customDataUrl = sameOriginPlentyChessAsset(params.get('plentyChessData'));
  if (customJsUrl || customWasmUrl || customDataUrl) {
    return {
      key: 'custom',
      label: 'PlentyChess Custom',
      jsUrl: customJsUrl ?? PLENTYCHESS_EMSCRIPTEN_JS_URL,
      wasmUrl: customWasmUrl ?? PLENTYCHESS_EMSCRIPTEN_WASM_URL,
      dataUrl: customDataUrl ?? PLENTYCHESS_EMSCRIPTEN_DATA_URL,
      sourceNetworkUrl: PLENTYCHESS_SOURCE_NETWORK_URL,
      note: 'Custom PlentyChess Emscripten sidecar URL(s) from query params.',
    };
  }
  return plentyChessVariantByKey(params.get('plentyChessVariant') ?? params.get('plentychess') ?? defaultPlentyChessVariantKey());
}

export function plentyChessVariantAssetStatus(variant: PlentyChessVariant): PlentyChessAssetStatus {
  return assetStatuses.get(assetKey(variant)) ?? 'unknown';
}

export function checkPlentyChessVariantAsset(variant: PlentyChessVariant, onChange?: () => void): Promise<PlentyChessAssetStatus> {
  const key = assetKey(variant);
  const current = assetStatuses.get(key);
  if (current === 'present' || current === 'missing') return Promise.resolve(current);
  const existing = assetChecks.get(key);
  if (existing) return existing;
  if (shouldSkipKnownUnshippedProbe(variant)) {
    assetStatuses.set(key, 'missing');
    onChange?.();
    return Promise.resolve('missing');
  }
  assetStatuses.set(key, 'checking');
  onChange?.();
  const promise = Promise.all(assetUrls(variant).map((url) => fetch(url, { method: 'HEAD', cache: 'no-store' }).then((response) => response.ok).catch(() => false)))
    .then((results) => (results.every(Boolean) ? 'present' : 'missing') as PlentyChessAssetStatus)
    .then((status) => {
      assetStatuses.set(key, status);
      assetChecks.delete(key);
      onChange?.();
      return status;
    });
  assetChecks.set(key, promise);
  return promise;
}
