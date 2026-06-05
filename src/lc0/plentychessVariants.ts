export type PlentyChessVariantKey = 'emscripten' | 'custom';
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
export const PLENTYCHESS_MAIN_NETWORK = '0134-2r24-s0.bin';
export const PLENTYCHESS_SOURCE_NETWORK_URL = `https://github.com/Yoshie2000/PlentyNetworks/releases/download/0134-2r24-s0/${PLENTYCHESS_MAIN_NETWORK}`;

const assetStatuses = new Map<string, PlentyChessAssetStatus>();
const assetChecks = new Map<string, Promise<PlentyChessAssetStatus>>();

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

export const PLENTYCHESS_VARIANTS: readonly PlentyChessVariant[] = [
  PLENTYCHESS_EMSCRIPTEN_VARIANT,
];

export function normalizePlentyChessVariant(raw: string | null | undefined): PlentyChessVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  if (value === 'custom') return 'custom';
  return 'emscripten';
}

export function defaultPlentyChessVariantKey(): PlentyChessVariantKey {
  return 'emscripten';
}

export function plentyChessVariantByKey(key: string): PlentyChessVariant {
  const normalized = normalizePlentyChessVariant(key);
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
  const customJsUrl = params.get('plentyChessJs');
  const customWasmUrl = params.get('plentyChessWasm');
  const customDataUrl = params.get('plentyChessData');
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
