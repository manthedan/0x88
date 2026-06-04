import { DEFAULT_RECKLESS_WASM_URL } from './recklessEngine.ts';

export type RecklessVariantKey = 'full' | 'simd' | 'lite' | 'browser-api' | 'custom';

export interface RecklessVariant {
  key: RecklessVariantKey;
  label: string;
  wasmUrl: string;
  note: string;
  backend?: 'wasi' | 'browser-api';
}

export type RecklessAssetStatus = 'unknown' | 'checking' | 'present' | 'missing';

const assetStatuses = new Map<string, RecklessAssetStatus>();
const assetChecks = new Map<string, Promise<RecklessAssetStatus>>();

export const RECKLESS_FULL_VARIANT: RecklessVariant = {
  key: 'full',
  label: 'Reckless Full',
  wasmUrl: DEFAULT_RECKLESS_WASM_URL,
  note: 'v60 full-size NNUE; strongest/current default, largest download.',
};

export const RECKLESS_SIMD_VARIANT: RecklessVariant = {
  key: 'simd',
  label: 'Reckless Full SIMD experimental',
  wasmUrl: '/reckless/reckless-simd128.wasm',
  note: 'Full-size NNUE built with wasm simd128 enabled; experimental opt-in benchmark target.',
};

export const RECKLESS_LITE_VARIANT: RecklessVariant = {
  key: 'lite',
  label: 'Reckless Lite experimental',
  wasmUrl: '/reckless/reckless-v53-l1-512.wasm',
  note: 'v53 L1=512 candidate; smaller/faster prototype, weaker and not shipped by default.',
};

export const RECKLESS_BROWSER_API_VARIANT: RecklessVariant = {
  key: 'browser-api',
  label: 'Reckless Full browser API experimental',
  wasmUrl: '/reckless/reckless-browser-api.wasm',
  note: 'Full-size NNUE with direct WASM exports; bypasses WASI/UCI text for lower adapter overhead.',
  backend: 'browser-api',
};

export const RECKLESS_VARIANTS = [RECKLESS_FULL_VARIANT, RECKLESS_SIMD_VARIANT, RECKLESS_BROWSER_API_VARIANT, RECKLESS_LITE_VARIANT] as const;

export function normalizeRecklessVariant(raw: string | null | undefined): RecklessVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _]/g, '-');
  if (value === 'lite' || value === 'small' || value === 'v53') return 'lite';
  if (value === 'api' || value === 'browser-api' || value === 'direct' || value === 'native') return 'browser-api';
  if (value === 'simd' || value === 'simd128' || value === 'full-simd') return 'simd';
  if (value === 'custom') return 'custom';
  return 'full';
}

export function recklessVariantByKey(key: RecklessVariantKey): RecklessVariant {
  if (key === 'lite') return RECKLESS_LITE_VARIANT;
  if (key === 'browser-api') return RECKLESS_BROWSER_API_VARIANT;
  if (key === 'simd') return RECKLESS_SIMD_VARIANT;
  if (key === 'custom') return { key: 'custom', label: 'Reckless Custom', wasmUrl: DEFAULT_RECKLESS_WASM_URL, note: 'Custom Reckless WASM URL.' };
  return RECKLESS_FULL_VARIANT;
}

export function recklessVariantFromParams(params: URLSearchParams): RecklessVariant {
  const customUrl = params.get('recklessWasm');
  if (customUrl) return { key: 'custom', label: 'Reckless Custom', wasmUrl: customUrl, note: 'Custom Reckless WASM URL from ?recklessWasm=…' };
  return recklessVariantByKey(normalizeRecklessVariant(params.get('recklessVariant') ?? params.get('reckless')));
}

export function recklessVariantAssetStatus(variant: RecklessVariant): RecklessAssetStatus {
  return assetStatuses.get(variant.wasmUrl) ?? 'unknown';
}

export function checkRecklessVariantAsset(variant: RecklessVariant, onChange?: () => void): Promise<RecklessAssetStatus> {
  const current = assetStatuses.get(variant.wasmUrl);
  if (current === 'present' || current === 'missing') return Promise.resolve(current);
  const existing = assetChecks.get(variant.wasmUrl);
  if (existing) return existing;
  assetStatuses.set(variant.wasmUrl, 'checking');
  const promise = fetch(variant.wasmUrl, { method: 'HEAD', cache: 'no-store' })
    .then((response) => (response.ok ? 'present' : 'missing') as RecklessAssetStatus)
    .catch(() => 'missing' as RecklessAssetStatus)
    .then((status) => {
      assetStatuses.set(variant.wasmUrl, status);
      assetChecks.delete(variant.wasmUrl);
      onChange?.();
      return status;
    });
  assetChecks.set(variant.wasmUrl, promise);
  return promise;
}
