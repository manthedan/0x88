export type BerserkVariantKey = 'default' | 'simd' | 'custom';
export type BerserkAssetStatus = 'unknown' | 'checking' | 'present' | 'missing';

export interface BerserkVariant {
  key: BerserkVariantKey;
  label: string;
  wasmUrl: string;
  note: string;
  nnueUrl?: string;
  sourceNetworkUrl?: string;
}

export const BERSERK_DEFAULT_WASM_URL = '/berserk/berserk.wasm';
export const BERSERK_SIMD_WASM_URL = '/berserk/berserk-simd128.wasm';
export const BERSERK_MAIN_NETWORK = 'berserk-9b84c340af7e.nn';
export const BERSERK_DEFAULT_NNUE_URL = `/berserk/${BERSERK_MAIN_NETWORK}`;
export const BERSERK_SOURCE_NETWORK_URL = `https://github.com/jhonnold/berserk-networks/releases/download/networks/${BERSERK_MAIN_NETWORK}`;

const assetStatuses = new Map<string, BerserkAssetStatus>();
const assetChecks = new Map<string, Promise<BerserkAssetStatus>>();

function assetKey(variant: BerserkVariant): string {
  return [variant.wasmUrl, variant.nnueUrl].filter(Boolean).join('\n');
}

export function supportsBerserkWasmSimd(): boolean {
  if (typeof WebAssembly === 'undefined' || typeof WebAssembly.validate !== 'function') return false;
  // (module (func (result v128) i32.const 0 i8x16.splat))
  return WebAssembly.validate(new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1,
    5, 1, 96, 0, 1, 123, 3, 2, 1,
    0, 10, 8, 1, 6, 0, 65, 0, 253,
    15, 11,
  ]));
}

export const BERSERK_DEFAULT_VARIANT: BerserkVariant = {
  key: 'default',
  label: 'Berserk scalar experimental',
  wasmUrl: BERSERK_DEFAULT_WASM_URL,
  nnueUrl: BERSERK_DEFAULT_NNUE_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: 'Planned Berserk tag 14 scalar wasm32-wasip1 UCI build. Experimental until first browser smoke passes.',
};

export const BERSERK_SIMD_VARIANT: BerserkVariant = {
  key: 'simd',
  label: 'Berserk SIMD experimental',
  wasmUrl: BERSERK_SIMD_WASM_URL,
  nnueUrl: BERSERK_DEFAULT_NNUE_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: 'Planned Berserk tag 14 wasm simd128 UCI build. Enable only after scalar smoke and SIMD codegen validation.',
};

export const BERSERK_VARIANTS: readonly BerserkVariant[] = [
  BERSERK_DEFAULT_VARIANT,
  BERSERK_SIMD_VARIANT,
];

export function normalizeBerserkVariant(raw: string | null | undefined): BerserkVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  if (value === 'simd' || value === 'simd128' || value === 'wasmsimd') return 'simd';
  if (value === 'scalar' || value === 'default' || value === 'full') return 'default';
  if (value === 'custom') return 'custom';
  return 'default';
}

export function defaultBerserkVariantKey(): BerserkVariantKey {
  return supportsBerserkWasmSimd() ? 'simd' : 'default';
}

export function berserkVariantByKey(key: string): BerserkVariant {
  const normalized = normalizeBerserkVariant(key);
  if (normalized === 'simd') return BERSERK_SIMD_VARIANT;
  if (normalized === 'custom') return { key: 'custom', label: 'Berserk Custom', wasmUrl: BERSERK_DEFAULT_WASM_URL, nnueUrl: BERSERK_DEFAULT_NNUE_URL, sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL, note: 'Custom Berserk WASM URL.' };
  return BERSERK_DEFAULT_VARIANT;
}

export function hasExplicitBerserkVariant(params: URLSearchParams): boolean {
  return params.has('berserkWasm') || params.has('berserkVariant') || params.has('berserk');
}

export function berserkVariantFromParams(params: URLSearchParams): BerserkVariant {
  const customUrl = params.get('berserkWasm');
  const customNnueUrl = params.get('berserkNnue') ?? undefined;
  if (customUrl) {
    return {
      key: 'custom',
      label: 'Berserk Custom',
      wasmUrl: customUrl,
      nnueUrl: customNnueUrl ?? BERSERK_DEFAULT_NNUE_URL,
      sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
      note: 'Custom Berserk WASM URL from ?berserkWasm=…',
    };
  }
  const explicit = params.get('berserkVariant') ?? params.get('berserk');
  const variant = berserkVariantByKey(explicit ?? defaultBerserkVariantKey());
  if (!customNnueUrl) return variant;
  return {
    ...variant,
    nnueUrl: customNnueUrl,
    note: `${variant.note} External NNUE overridden by ?berserkNnue=…`,
  };
}

export async function resolveDefaultBerserkVariantAssetFallback(variant: BerserkVariant, explicit: boolean, onChange?: () => void): Promise<BerserkVariant> {
  if (explicit || variant.key !== 'simd') return variant;
  const status = await checkBerserkVariantAsset(variant, onChange);
  if (status !== 'missing') return variant;
  if (!variant.nnueUrl || variant.nnueUrl === BERSERK_DEFAULT_NNUE_URL) return BERSERK_DEFAULT_VARIANT;
  return {
    ...BERSERK_DEFAULT_VARIANT,
    nnueUrl: variant.nnueUrl,
    sourceNetworkUrl: variant.sourceNetworkUrl,
    note: `${BERSERK_DEFAULT_VARIANT.note} External NNUE overridden by ?berserkNnue=…`,
  };
}

export function berserkVariantAssetStatus(variant: BerserkVariant): BerserkAssetStatus {
  return assetStatuses.get(assetKey(variant)) ?? 'unknown';
}

export function checkBerserkVariantAsset(variant: BerserkVariant, onChange?: () => void): Promise<BerserkAssetStatus> {
  const key = assetKey(variant);
  const current = assetStatuses.get(key);
  if (current === 'present' || current === 'missing') return Promise.resolve(current);
  const existing = assetChecks.get(key);
  if (existing) return existing;
  assetStatuses.set(key, 'checking');
  onChange?.();
  const urls = [variant.wasmUrl, ...(variant.nnueUrl ? [variant.nnueUrl] : [])];
  const promise = Promise.all(urls.map((url) => fetch(url, { method: 'HEAD', cache: 'no-store' }).then((response) => response.ok).catch(() => false)))
    .then((results) => (results.every(Boolean) ? 'present' : 'missing') as BerserkAssetStatus)
    .then((status) => {
      assetStatuses.set(key, status);
      assetChecks.delete(key);
      onChange?.();
      return status;
    });
  assetChecks.set(key, promise);
  return promise;
}
