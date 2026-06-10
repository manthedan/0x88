import { supportsWasmRelaxedSimd, supportsWasmSimd } from './recklessVariants.ts';
import { DEFAULT_VIRIDITHAS_WASM_URL } from './viridithasEngine.ts';

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
  label: 'Viridithas SIMD experimental',
  wasmUrl: '/viridithas/viridithas-simd128.wasm',
  note: 'Experimental patched Viridithas wasm32-wasip1 build with wasm simd128 NNUE kernels and one-shot, persistent, and batch benchmark modes.',
};

export const VIRIDITHAS_RELAXED_SIMD_VARIANT: ViridithasVariant = {
  key: 'relaxed-simd',
  label: 'Viridithas Relaxed SIMD experimental',
  wasmUrl: '/viridithas/viridithas-relaxed-simd128.wasm',
  note: 'Experimental Viridithas build using the relaxed integer dot for the L1 NNUE kernels (exact: QA=255/FT_SHIFT=9 keep activations in 0..127). Requires WebAssembly Relaxed SIMD.',
};

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

export function viridithasVariantFromParams(params: URLSearchParams): ViridithasVariant {
  const customUrl = sameOriginViridithasAsset(params.get('viridithasWasm'));
  if (customUrl) return { key: 'custom', label: 'Viridithas Custom', wasmUrl: customUrl, note: 'Custom same-origin Viridithas WASM URL from ?viridithasWasm=…' };
  // SIMD has the strongest/current smoke and benchmark evidence; scalar remains
  // available as an explicit compatibility fallback via ?viridithas=default.
  return viridithasVariantByKey(params.get('viridithas') ?? params.get('viridithasVariant') ?? 'simd');
}

export function viridithasVariantAssetStatus(variant: ViridithasVariant): ViridithasAssetStatus {
  return variant.assetStatus ?? 'unknown';
}

export async function checkViridithasVariantAsset(variant: ViridithasVariant, onChange?: () => void): Promise<ViridithasAssetStatus> {
  if (variant.assetStatus === 'ok' || variant.assetStatus === 'missing') return variant.assetStatus;
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
