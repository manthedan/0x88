import { DEFAULT_RECKLESS_WASM_URL } from './recklessEngine.ts';
import { resolvePublicAssetUrl } from './assetUrls.ts';
import { supportsWasmRelaxedSimd, supportsWasmSimd } from './wasmFeatures.ts';
export { supportsWasmRelaxedSimd, supportsWasmSimd } from './wasmFeatures.ts';

export type RecklessVariantKey = 'full' | 'simd' | 'relaxed-simd' | 'lite' | 'browser-api' | 'browser-api-simd' | 'browser-api-simd-external' | 'custom';

export interface RecklessVariant {
  key: RecklessVariantKey;
  label: string;
  wasmUrl: string;
  note: string;
  backend?: 'wasi' | 'browser-api';
  nnueUrl?: string;
}

export type RecklessAssetStatus = 'unknown' | 'checking' | 'present' | 'missing';

const assetStatuses = new Map<string, RecklessAssetStatus>();
const assetChecks = new Map<string, Promise<RecklessAssetStatus>>();
const recklessAsset = (path: string) => resolvePublicAssetUrl(path);

function assetKey(variant: RecklessVariant): string {
  return variant.nnueUrl ? `${variant.wasmUrl}\n${variant.nnueUrl}` : variant.wasmUrl;
}

export const RECKLESS_FULL_VARIANT: RecklessVariant = {
  key: 'full',
  label: 'Reckless Full scalar fallback',
  wasmUrl: DEFAULT_RECKLESS_WASM_URL,
  note: 'v60 full-size NNUE scalar WASI/UCI fallback for browsers without WebAssembly SIMD.',
};

export const RECKLESS_SIMD_VARIANT: RecklessVariant = {
  key: 'simd',
  label: 'Reckless Full SIMD',
  wasmUrl: recklessAsset('/reckless/reckless-simd128.wasm'),
  note: 'v60 full-size NNUE with integrated wasm simd128 backend; preferred default when supported.',
};

export const RECKLESS_RELAXED_SIMD_VARIANT: RecklessVariant = {
  key: 'relaxed-simd',
  label: 'Reckless Full Relaxed SIMD',
  wasmUrl: recklessAsset('/reckless/reckless-relaxed-simd128.wasm'),
  note: 'v60 full-size NNUE using the relaxed integer dot for dpbusd (exact: activations provably in 0..127). Default when the browser validates Relaxed SIMD; promoted on 60/60 fixed-depth parity and +24% NPS vs the old kernels.',
};

export const RECKLESS_LITE_VARIANT: RecklessVariant = {
  key: 'lite',
  label: 'Reckless Lite experimental',
  wasmUrl: recklessAsset('/reckless/reckless-v53-l1-512.wasm'),
  note: 'v53 L1=512 candidate; smaller/faster prototype, weaker and not shipped by default.',
};

export const RECKLESS_BROWSER_API_VARIANT: RecklessVariant = {
  key: 'browser-api',
  label: 'Reckless Full browser API experimental',
  wasmUrl: recklessAsset('/reckless/reckless-browser-api.wasm'),
  note: 'Full-size NNUE with direct WASM exports; bypasses WASI/UCI text for lower adapter overhead.',
  backend: 'browser-api',
};

export const RECKLESS_BROWSER_API_SIMD_VARIANT: RecklessVariant = {
  key: 'browser-api-simd',
  label: 'Reckless Full browser API SIMD experimental',
  wasmUrl: recklessAsset('/reckless/reckless-browser-api-simd128.wasm'),
  note: 'Direct browser API artifact combined with the integrated wasm simd128 NNUE backend.',
  backend: 'browser-api',
};

export const RECKLESS_BROWSER_API_SIMD_EXTERNAL_VARIANT: RecklessVariant = {
  key: 'browser-api-simd-external',
  label: 'Reckless Full browser API SIMD external NNUE experimental',
  wasmUrl: recklessAsset('/reckless/reckless-browser-api-simd128-external.wasm'),
  note: 'Direct browser API SIMD artifact with the full NNUE loaded as a separate cacheable asset.',
  backend: 'browser-api',
  nnueUrl: recklessAsset('/reckless/reckless-v60-7f587dfb.nnue'),
};

export const RECKLESS_VARIANTS = [RECKLESS_SIMD_VARIANT, RECKLESS_RELAXED_SIMD_VARIANT, RECKLESS_FULL_VARIANT, RECKLESS_BROWSER_API_VARIANT, RECKLESS_BROWSER_API_SIMD_VARIANT, RECKLESS_BROWSER_API_SIMD_EXTERNAL_VARIANT, RECKLESS_LITE_VARIANT] as const;

export function normalizeRecklessVariant(raw: string | null | undefined): RecklessVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _]/g, '-');
  if (value === 'lite' || value === 'small' || value === 'v53') return 'lite';
  if (value === 'api-simd-external' || value === 'browser-api-simd-external' || value === 'direct-simd-external' || value === 'native-simd-external' || value === 'external-simd') return 'browser-api-simd-external';
  if (value === 'api-simd' || value === 'browser-api-simd' || value === 'direct-simd' || value === 'native-simd') return 'browser-api-simd';
  if (value === 'api' || value === 'browser-api' || value === 'direct' || value === 'native') return 'browser-api';
  if (value === 'relaxed' || value === 'relaxed-simd' || value === 'relaxed-simd128' || value === 'full-relaxed-simd') return 'relaxed-simd';
  if (value === 'simd' || value === 'simd128' || value === 'full-simd') return 'simd';
  if (value === 'custom') return 'custom';
  return 'full';
}

export function defaultRecklessVariantKey(): RecklessVariantKey {
  // Promotion order: relaxed integer dot > simd128 > scalar. All three are
  // value-exact (60/60 fixed-depth parity), so selection is purely a speed
  // ladder gated by WebAssembly feature validation, with asset fallback in
  // resolveDefaultRecklessVariantAssetFallback.
  if (supportsWasmRelaxedSimd()) return 'relaxed-simd';
  return supportsWasmSimd() ? 'simd' : 'full';
}

export function recklessVariantByKey(key: RecklessVariantKey): RecklessVariant {
  if (key === 'lite') return RECKLESS_LITE_VARIANT;
  if (key === 'browser-api') return RECKLESS_BROWSER_API_VARIANT;
  if (key === 'browser-api-simd') return RECKLESS_BROWSER_API_SIMD_VARIANT;
  if (key === 'browser-api-simd-external') return RECKLESS_BROWSER_API_SIMD_EXTERNAL_VARIANT;
  if (key === 'relaxed-simd') return RECKLESS_RELAXED_SIMD_VARIANT;
  if (key === 'simd') return RECKLESS_SIMD_VARIANT;
  if (key === 'custom') return { key: 'custom', label: 'Reckless Custom', wasmUrl: DEFAULT_RECKLESS_WASM_URL, note: 'Custom Reckless WASM URL.' };
  return RECKLESS_FULL_VARIANT;
}

export function hasExplicitRecklessVariant(params: URLSearchParams): boolean {
  return params.has('recklessWasm') || params.has('recklessVariant') || params.has('reckless');
}

export function recklessVariantFromParams(params: URLSearchParams): RecklessVariant {
  const customUrl = params.get('recklessWasm');
  if (customUrl) return { key: 'custom', label: 'Reckless Custom', wasmUrl: customUrl, note: 'Custom Reckless WASM URL from ?recklessWasm=…' };
  const explicit = params.get('recklessVariant') ?? params.get('reckless');
  if (explicit) return recklessVariantByKey(normalizeRecklessVariant(explicit));
  return recklessVariantByKey(defaultRecklessVariantKey());
}

export async function resolveDefaultRecklessVariantAssetFallback(variant: RecklessVariant, explicit: boolean, onChange?: () => void): Promise<RecklessVariant> {
  if (variant.key === 'relaxed-simd') {
    if (!supportsWasmRelaxedSimd()) return supportsWasmSimd() ? RECKLESS_SIMD_VARIANT : RECKLESS_FULL_VARIANT;
    if (explicit) return variant;
    const status = await checkRecklessVariantAsset(variant, onChange);
    if (status !== 'missing') return variant;
    if (!supportsWasmSimd()) return RECKLESS_FULL_VARIANT;
    const simdStatus = await checkRecklessVariantAsset(RECKLESS_SIMD_VARIANT, onChange);
    return simdStatus === 'missing' ? RECKLESS_FULL_VARIANT : RECKLESS_SIMD_VARIANT;
  }
  if (explicit || variant.key !== 'simd') return variant;
  const status = await checkRecklessVariantAsset(variant, onChange);
  return status === 'missing' ? RECKLESS_FULL_VARIANT : variant;
}

export function recklessVariantAssetStatus(variant: RecklessVariant): RecklessAssetStatus {
  return assetStatuses.get(assetKey(variant)) ?? 'unknown';
}

export function checkRecklessVariantAsset(variant: RecklessVariant, onChange?: () => void): Promise<RecklessAssetStatus> {
  const key = assetKey(variant);
  const current = assetStatuses.get(key);
  if (current === 'present' || current === 'missing') return Promise.resolve(current);
  const existing = assetChecks.get(key);
  if (existing) return existing;
  assetStatuses.set(key, 'checking');
  const urls = [variant.wasmUrl, ...(variant.nnueUrl ? [variant.nnueUrl] : [])];
  const promise = Promise.all(urls.map((url) => fetch(url, { method: 'HEAD', cache: 'no-store' }).then((response) => response.ok).catch(() => false)))
    .then((results) => (results.every(Boolean) ? 'present' : 'missing') as RecklessAssetStatus)
    .then((status) => {
      assetStatuses.set(key, status);
      assetChecks.delete(key);
      onChange?.();
      return status;
    });
  assetChecks.set(key, promise);
  return promise;
}
