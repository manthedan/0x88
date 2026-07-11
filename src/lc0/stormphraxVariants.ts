import { resolvePublicAssetUrl } from './assetUrls.ts';
import { isV0DeployProfile } from './engineCatalog.ts';
import { supportsWasmSimd } from './wasmFeatures.ts';

export type StormphraxVariantKey = 'emscripten' | 'custom';
export type StormphraxAssetStatus = 'unknown' | 'checking' | 'present' | 'missing';

export interface StormphraxVariant {
  key: StormphraxVariantKey;
  label: string;
  jsUrl: string;
  wasmUrl: string;
  dataUrl: string;
  sourceNetworkUrl?: string;
  note: string;
}

const STORMPHRAX_JS_PATH = isV0DeployProfile()
  ? '/artifacts/sha256/4c95a359f1c09760611d09ea6a3fa4a451407c699d14ef559940d37365a4bfec/stormphrax-emscripten.js'
  : '/stormphrax/stormphrax-emscripten.js';
const STORMPHRAX_WASM_PATH = isV0DeployProfile()
  ? '/artifacts/sha256/92149aeeeefbc7bac74ec9b6ebf24384599bf35e8f8f404bf26518f53f6f93e1/stormphrax-emscripten.wasm'
  : '/stormphrax/stormphrax-emscripten.wasm';
const STORMPHRAX_DATA_PATH = isV0DeployProfile()
  ? '/artifacts/sha256/04d651e078b7c7334709dbd772d40a23c0a5480e93e19521a03020c7d633f2cf/stormphrax-emscripten.data'
  : '/stormphrax/stormphrax-emscripten.data';

export const STORMPHRAX_EMSCRIPTEN_JS_URL = resolvePublicAssetUrl(STORMPHRAX_JS_PATH);
export const STORMPHRAX_EMSCRIPTEN_WASM_URL = resolvePublicAssetUrl(STORMPHRAX_WASM_PATH);
export const STORMPHRAX_EMSCRIPTEN_DATA_URL = resolvePublicAssetUrl(STORMPHRAX_DATA_PATH);
export const STORMPHRAX_MAIN_NETWORK = 'undertown.nnue';
export const STORMPHRAX_SOURCE_NETWORK_URL = `https://github.com/Ciekce/stormphrax-nets/releases/download/undertown/${STORMPHRAX_MAIN_NETWORK}`;

const assetStatuses = new Map<string, StormphraxAssetStatus>();
const assetChecks = new Map<string, Promise<StormphraxAssetStatus>>();

function isLocalDevelopmentOrigin(): boolean {
  if (typeof location === 'undefined') return true;
  return location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '::1' || location.hostname === '[::1]';
}

function sameOriginStormphraxAsset(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    const base = typeof location !== 'undefined' ? location.origin : 'http://localhost';
    const url = new URL(raw, base);
    if (url.origin !== base || !url.pathname.startsWith('/stormphrax/')) return undefined;
    return `${url.pathname}${url.search}`;
  } catch {
    return undefined;
  }
}

function assetUrls(variant: StormphraxVariant): string[] {
  return [variant.jsUrl, variant.wasmUrl, variant.dataUrl];
}

function assetKey(variant: StormphraxVariant): string {
  return assetUrls(variant).join('\n');
}

export const STORMPHRAX_EMSCRIPTEN_VARIANT: StormphraxVariant = {
  key: 'emscripten',
  label: 'Stormphrax 8',
  jsUrl: STORMPHRAX_EMSCRIPTEN_JS_URL,
  wasmUrl: STORMPHRAX_EMSCRIPTEN_WASM_URL,
  dataUrl: STORMPHRAX_EMSCRIPTEN_DATA_URL,
  sourceNetworkUrl: STORMPHRAX_SOURCE_NETWORK_URL,
  note: 'Stormphrax 8.0.0 single-thread WebAssembly SIMD Emscripten worker with the undertown NNUE preloaded in .data.',
};

export const STORMPHRAX_VARIANTS: readonly StormphraxVariant[] = [STORMPHRAX_EMSCRIPTEN_VARIANT];

export function normalizeStormphraxVariant(raw: string | null | undefined): StormphraxVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  return value === 'custom' ? 'custom' : 'emscripten';
}

export function stormphraxVariantUnsupportedReason(variant: StormphraxVariant): string | null {
  if (variant.key === 'custom') return null;
  return supportsWasmSimd() ? null : 'requires WebAssembly SIMD';
}

export function supportsStormphraxVariant(variant: StormphraxVariant): boolean {
  return stormphraxVariantUnsupportedReason(variant) === null;
}

export function defaultStormphraxVariantKey(): StormphraxVariantKey {
  return 'emscripten';
}

export function stormphraxVariantByKey(key: string): StormphraxVariant {
  if (normalizeStormphraxVariant(key) === 'custom') {
    return {
      ...STORMPHRAX_EMSCRIPTEN_VARIANT,
      key: 'custom',
      label: 'Stormphrax Custom',
      note: 'Custom Stormphrax Emscripten sidecar URLs.',
    };
  }
  return STORMPHRAX_EMSCRIPTEN_VARIANT;
}

export function hasExplicitStormphraxVariant(params: URLSearchParams): boolean {
  return params.has('stormphraxJs') || params.has('stormphraxWasm') || params.has('stormphraxData') || params.has('stormphraxVariant') || params.has('stormphrax');
}

export function stormphraxVariantFromParams(params: URLSearchParams): StormphraxVariant {
  const customJsUrl = sameOriginStormphraxAsset(params.get('stormphraxJs'));
  const customWasmUrl = sameOriginStormphraxAsset(params.get('stormphraxWasm'));
  const customDataUrl = sameOriginStormphraxAsset(params.get('stormphraxData'));
  if (customJsUrl || customWasmUrl || customDataUrl) {
    return {
      ...STORMPHRAX_EMSCRIPTEN_VARIANT,
      key: 'custom',
      label: 'Stormphrax Custom',
      jsUrl: customJsUrl ?? STORMPHRAX_EMSCRIPTEN_JS_URL,
      wasmUrl: customWasmUrl ?? STORMPHRAX_EMSCRIPTEN_WASM_URL,
      dataUrl: customDataUrl ?? STORMPHRAX_EMSCRIPTEN_DATA_URL,
      note: 'Custom Stormphrax Emscripten sidecar URL(s) from query params.',
    };
  }
  return stormphraxVariantByKey(params.get('stormphraxVariant') ?? params.get('stormphrax') ?? defaultStormphraxVariantKey());
}

export function stormphraxVariantAssetStatus(variant: StormphraxVariant): StormphraxAssetStatus {
  return assetStatuses.get(assetKey(variant)) ?? 'unknown';
}

export function checkStormphraxVariantAsset(variant: StormphraxVariant, onChange?: () => void): Promise<StormphraxAssetStatus> {
  const key = assetKey(variant);
  const current = assetStatuses.get(key);
  if (current === 'present' || current === 'missing') return Promise.resolve(current);
  const existing = assetChecks.get(key);
  if (existing) return existing.then((status) => { onChange?.(); return status; });
  if (!isLocalDevelopmentOrigin() && variant.key !== 'custom' && !supportsStormphraxVariant(variant)) {
    assetStatuses.set(key, 'missing');
    onChange?.();
    return Promise.resolve('missing');
  }
  assetStatuses.set(key, 'checking');
  const promise = Promise.all(assetUrls(variant).map((url) => fetch(url, { method: 'HEAD', cache: 'no-store' }).then((response) => response.ok).catch(() => false)))
    .then((results) => (results.every(Boolean) ? 'present' : 'missing') as StormphraxAssetStatus)
    .then((status) => {
      assetStatuses.set(key, status);
      assetChecks.delete(key);
      onChange?.();
      return status;
    });
  assetChecks.set(key, promise);
  onChange?.();
  return promise;
}

export async function resolveDefaultStormphraxVariantAssetFallback(variant: StormphraxVariant, _explicit: boolean, onChange?: () => void): Promise<StormphraxVariant> {
  if (!supportsStormphraxVariant(variant)) return variant;
  // There is no fallback tier yet, so asset status is informational. Do not
  // block Analysis/Arena initialization on three remote HEAD requests.
  void checkStormphraxVariantAsset(variant, onChange);
  return variant;
}
