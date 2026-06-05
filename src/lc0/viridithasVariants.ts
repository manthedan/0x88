import { DEFAULT_VIRIDITHAS_WASM_URL } from './viridithasEngine.ts';

export type ViridithasAssetStatus = 'unknown' | 'checking' | 'ok' | 'missing';

export interface ViridithasVariant {
  key: 'default' | 'custom';
  label: string;
  wasmUrl: string;
  note: string;
  assetStatus?: ViridithasAssetStatus;
}

export const VIRIDITHAS_DEFAULT_VARIANT: ViridithasVariant = {
  key: 'default',
  label: 'Viridithas experimental',
  wasmUrl: DEFAULT_VIRIDITHAS_WASM_URL,
  note: 'Experimental patched Viridithas wasm32-wasip1 build; one-shot only.',
};

export function viridithasVariantFromParams(params: URLSearchParams): ViridithasVariant {
  const customUrl = params.get('viridithasWasm');
  if (customUrl) return { key: 'custom', label: 'Viridithas Custom', wasmUrl: customUrl, note: 'Custom Viridithas WASM URL from ?viridithasWasm=…' };
  return VIRIDITHAS_DEFAULT_VARIANT;
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
