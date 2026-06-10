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

export function supportsWasmRelaxedSimd(): boolean {
  if (typeof WebAssembly === 'undefined' || typeof WebAssembly.validate !== 'function') return false;
  // (module (func (result v128) (f32x4.relaxed_madd (f32x4.splat 1) (f32x4.splat 2) (f32x4.splat 3))))
  return WebAssembly.validate(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96,
    0, 1, 123, 3, 2, 1, 0, 10, 28, 1, 26, 0,
    67, 0, 0, 128, 63, 253, 19, 67, 0, 0, 0, 64,
    253, 19, 67, 0, 0, 64, 64, 253, 19, 253, 133,
    2, 11,
  ]));
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
  label: 'PlentyChess SSE4.1 Emscripten experimental',
  jsUrl: PLENTYCHESS_EMSCRIPTEN_SSE41_JS_URL,
  wasmUrl: PLENTYCHESS_EMSCRIPTEN_SSE41_WASM_URL,
  dataUrl: PLENTYCHESS_EMSCRIPTEN_SSE41_DATA_URL,
  sourceNetworkUrl: PLENTYCHESS_SOURCE_NETWORK_URL,
  note: 'Default build plus -msse4.1: single-op convertEpi8Epi16 in the accumulator path (exact-equal semantics).',
};

export const PLENTYCHESS_EMSCRIPTEN_RELAXED_VARIANT: PlentyChessVariant = {
  key: 'emscripten-relaxed',
  label: 'PlentyChess Relaxed SIMD Emscripten experimental',
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

export function defaultPlentyChessVariantKey(): PlentyChessVariantKey {
  return 'emscripten';
}

export function plentyChessVariantByKey(key: string): PlentyChessVariant {
  const normalized = normalizePlentyChessVariant(key);
  if (normalized === 'emscripten-sse41') return PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT;
  if (normalized === 'emscripten-relaxed') return supportsWasmRelaxedSimd() ? PLENTYCHESS_EMSCRIPTEN_RELAXED_VARIANT : PLENTYCHESS_EMSCRIPTEN_SSE41_VARIANT;
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
