import { DEFAULT_RECKLESS_WASM_URL } from './recklessEngine.ts';

export type RecklessVariantKey = 'full' | 'lite' | 'custom';

export interface RecklessVariant {
  key: RecklessVariantKey;
  label: string;
  wasmUrl: string;
  note: string;
}

export const RECKLESS_FULL_VARIANT: RecklessVariant = {
  key: 'full',
  label: 'Reckless Full',
  wasmUrl: DEFAULT_RECKLESS_WASM_URL,
  note: 'v60 full-size NNUE; strongest/current default, largest download.',
};

export const RECKLESS_LITE_VARIANT: RecklessVariant = {
  key: 'lite',
  label: 'Reckless Lite',
  wasmUrl: '/reckless/reckless-v53-l1-512.wasm',
  note: 'v53 L1=512 candidate; smaller/faster prototype, weaker and not shipped by default.',
};

export const RECKLESS_VARIANTS = [RECKLESS_FULL_VARIANT, RECKLESS_LITE_VARIANT] as const;

export function normalizeRecklessVariant(raw: string | null | undefined): RecklessVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _]/g, '-');
  if (value === 'lite' || value === 'small' || value === 'v53') return 'lite';
  if (value === 'custom') return 'custom';
  return 'full';
}

export function recklessVariantByKey(key: RecklessVariantKey): RecklessVariant {
  if (key === 'lite') return RECKLESS_LITE_VARIANT;
  if (key === 'custom') return { key: 'custom', label: 'Reckless Custom', wasmUrl: DEFAULT_RECKLESS_WASM_URL, note: 'Custom Reckless WASM URL.' };
  return RECKLESS_FULL_VARIANT;
}

export function recklessVariantFromParams(params: URLSearchParams): RecklessVariant {
  const customUrl = params.get('recklessWasm');
  if (customUrl) return { key: 'custom', label: 'Reckless Custom', wasmUrl: customUrl, note: 'Custom Reckless WASM URL from ?recklessWasm=…' };
  return recklessVariantByKey(normalizeRecklessVariant(params.get('recklessVariant') ?? params.get('reckless')));
}
