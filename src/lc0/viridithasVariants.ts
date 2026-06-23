import { supportsWasmRelaxedSimd, supportsWasmSimd } from './recklessVariants.ts';

export const DEFAULT_VIRIDITHAS_WASM_URL = '/viridithas/viridithas.wasm';

export type ViridithasAssetStatus = 'unknown' | 'checking' | 'ok' | 'missing';

export interface ViridithasVariant {
  key: 'default' | 'simd' | 'relaxed-simd' | 'custom';
  label: string;
  wasmUrl: string;
  note: string;
  assetStatus?: ViridithasAssetStatus;
}

export const VIRIDITHAS_DEFAULT_VARIANT: ViridithasVariant = {
  key: 'default',
  label: 'Viridithas scalar experimental',
  wasmUrl: DEFAULT_VIRIDITHAS_WASM_URL,
  note: 'Experimental patched Viridithas wasm32-wasip1 scalar build with one-shot, persistent, and batch benchmark modes.',
};

export const VIRIDITHAS_SIMD_VARIANT: ViridithasVariant = {
  key: 'simd',
  label: 'Viridithas SIMD',
  wasmUrl: '/viridithas/viridithas-simd128.wasm',
  note: 'Patched Viridithas wasm32-wasip1 build with wasm simd128 NNUE kernels. Default when the browser validates wasm SIMD but not Relaxed SIMD; 40/40 fixed-depth parity with scalar, 5.4-5.8x scalar NPS.',
};

export const VIRIDITHAS_RELAXED_SIMD_VARIANT: ViridithasVariant = {
  key: 'relaxed-simd',
  label: 'Viridithas Relaxed SIMD',
  wasmUrl: '/viridithas/viridithas-relaxed-simd128.wasm',
  note: 'Viridithas build using the relaxed integer dot for the L1 NNUE kernels (exact: QA=255/FT_SHIFT=9 keep activations in 0..127). Default when the browser validates Relaxed SIMD; +14% NPS over simd128 at 40/40 parity.',
};

const DEPLOYED_VIRIDITHAS_URLS = new Set([
  DEFAULT_VIRIDITHAS_WASM_URL,
  VIRIDITHAS_SIMD_VARIANT.wasmUrl,
]);

function isLocalDevelopmentOrigin(): boolean {
  if (typeof location === 'undefined') return true;
  return location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '::1' || location.hostname === '[::1]';
}

function shouldSkipKnownUnshippedProbe(variant: ViridithasVariant): boolean {
  if (variant.key === 'custom' || isLocalDevelopmentOrigin()) return false;
  return variant.wasmUrl.startsWith('/viridithas/') && !DEPLOYED_VIRIDITHAS_URLS.has(variant.wasmUrl);
}

// Promotion order: relaxed integer dot > simd128 > scalar. All variants are
// value-exact (40/40 fixed-depth parity), so this is a speed ladder gated by
// WebAssembly feature validation.
export function defaultViridithasVariantKey(): ViridithasVariant['key'] {
  if (supportsWasmRelaxedSimd()) return 'relaxed-simd';
  return supportsWasmSimd() ? 'simd' : 'default';
}

/**
 * Resolve the feature-detected default against deployed assets: a missing
 * relaxed artifact falls back to simd128, and a missing simd128 artifact
 * falls back to scalar. Explicit user selections are honored as-is.
 */
export async function resolveDefaultViridithasVariantAssetFallback(variant: ViridithasVariant, explicit: boolean, onChange?: () => void): Promise<ViridithasVariant> {
  if (explicit || variant.key === 'default' || variant.key === 'custom') return variant;
  const status = await checkViridithasVariantAsset(variant, onChange);
  if (status !== 'missing') return variant;
  if (variant.key === 'relaxed-simd' && supportsWasmSimd()) {
    return resolveDefaultViridithasVariantAssetFallback(VIRIDITHAS_SIMD_VARIANT, false, onChange);
  }
  return VIRIDITHAS_DEFAULT_VARIANT;
}

export const VIRIDITHAS_VARIANTS: readonly ViridithasVariant[] = [
  VIRIDITHAS_SIMD_VARIANT,
  VIRIDITHAS_RELAXED_SIMD_VARIANT,
  VIRIDITHAS_DEFAULT_VARIANT,
];

function sameOriginViridithasAsset(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    const base = typeof location !== 'undefined' ? location.origin : 'http://localhost';
    const url = new URL(raw, base);
    if (url.origin !== base || !url.pathname.startsWith('/viridithas/')) return undefined;
    return `${url.pathname}${url.search}`;
  } catch {
    return undefined;
  }
}


export function normalizeViridithasVariant(raw: string | null | undefined): ViridithasVariant['key'] {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  if (value === 'relaxedsimd' || value === 'relaxed' || value === 'relaxedsimd128') return 'relaxed-simd';
  if (value === 'simd' || value === 'simd128' || value === 'wasmsimd') return 'simd';
  if (value === 'scalar' || value === 'default') return 'default';
  if (value === 'custom') return 'custom';
  return 'default';
}

export function viridithasVariantByKey(key: string): ViridithasVariant {
  const normalized = normalizeViridithasVariant(key);
  if (normalized === 'relaxed-simd') return supportsWasmRelaxedSimd() ? VIRIDITHAS_RELAXED_SIMD_VARIANT : supportsWasmSimd() ? VIRIDITHAS_SIMD_VARIANT : VIRIDITHAS_DEFAULT_VARIANT;
  return VIRIDITHAS_VARIANTS.find((variant) => variant.key === normalized) ?? VIRIDITHAS_DEFAULT_VARIANT;
}

export function hasExplicitViridithasVariant(params: URLSearchParams): boolean {
  return params.has('viridithasWasm') || params.has('viridithas') || params.has('viridithasVariant');
}

export function viridithasVariantFromParams(params: URLSearchParams): ViridithasVariant {
  const customUrl = sameOriginViridithasAsset(params.get('viridithasWasm'));
  if (customUrl) return { key: 'custom', label: 'Viridithas Custom', wasmUrl: customUrl, note: 'Custom same-origin Viridithas WASM URL from ?viridithasWasm=…' };
  return viridithasVariantByKey(params.get('viridithas') ?? params.get('viridithasVariant') ?? defaultViridithasVariantKey());
}

export function viridithasVariantAssetStatus(variant: ViridithasVariant): ViridithasAssetStatus {
  return variant.assetStatus ?? 'unknown';
}

export async function checkViridithasVariantAsset(variant: ViridithasVariant, onChange?: () => void): Promise<ViridithasAssetStatus> {
  if (variant.assetStatus === 'ok' || variant.assetStatus === 'missing') return variant.assetStatus;
  if (shouldSkipKnownUnshippedProbe(variant)) {
    variant.assetStatus = 'missing';
    onChange?.();
    return variant.assetStatus;
  }
  variant.assetStatus = 'checking';
  onChange?.();
  try {
    const response = await fetch(variant.wasmUrl, { method: 'HEAD' });
    variant.assetStatus = response.ok ? 'ok' : 'missing';
  } catch {
    variant.assetStatus = 'missing';
  }
  onChange?.();
  return variant.assetStatus;
}
