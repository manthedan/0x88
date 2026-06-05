import { DEFAULT_VIRIDITHAS_WASM_URL } from './viridithasEngine.ts';

export type ViridithasAssetStatus = 'unknown' | 'checking' | 'ok' | 'missing';

export interface ViridithasVariant {
  key: 'default' | 'simd' | 'custom';
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

export const VIRIDITHAS_VARIANTS: readonly ViridithasVariant[] = [
  VIRIDITHAS_SIMD_VARIANT,
  VIRIDITHAS_DEFAULT_VARIANT,
];

export function normalizeViridithasVariant(raw: string | null | undefined): ViridithasVariant['key'] {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  if (value === 'simd' || value === 'simd128' || value === 'wasmsimd') return 'simd';
  if (value === 'scalar' || value === 'default') return 'default';
  if (value === 'custom') return 'custom';
  return 'default';
}

export function viridithasVariantByKey(key: string): ViridithasVariant {
  const normalized = normalizeViridithasVariant(key);
  return VIRIDITHAS_VARIANTS.find((variant) => variant.key === normalized) ?? VIRIDITHAS_DEFAULT_VARIANT;
}

export function viridithasVariantFromParams(params: URLSearchParams): ViridithasVariant {
  const customUrl = params.get('viridithasWasm');
  if (customUrl) return { key: 'custom', label: 'Viridithas Custom', wasmUrl: customUrl, note: 'Custom Viridithas WASM URL from ?viridithasWasm=…' };
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
