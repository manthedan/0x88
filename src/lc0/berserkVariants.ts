import { resolvePublicAssetUrl } from './assetUrls.ts';
import { supportsWasmRelaxedSimd, supportsWasmSimd } from './wasmFeatures.ts';

export type BerserkVariantKey = 'emscripten' | 'emscripten-simd' | 'emscripten-relaxed' | 'default' | 'simd' | 'custom';
export type BerserkAssetStatus = 'unknown' | 'checking' | 'present' | 'missing';

export interface BerserkVariant {
  key: BerserkVariantKey;
  label: string;
  /** WASI/SIMD candidate URL, or Emscripten sidecar WASM when jsUrl is set. */
  wasmUrl: string;
  note: string;
  /** Emscripten JS glue URL for the currently-smoked browser worker path. */
  jsUrl?: string;
  /** Emscripten preload data URL containing the NNUE. */
  dataUrl?: string;
  /** External NNUE URL for future WASI/custom paths. */
  nnueUrl?: string;
  sourceNetworkUrl?: string;
}

export const BERSERK_EMSCRIPTEN_JS_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten.js');
export const BERSERK_EMSCRIPTEN_WASM_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten.wasm');
/**
 * Canonical preload `.data` shared by every Berserk SIMD tier.
 *
 * Emscripten emits `<variant>.data` per build, but the packaged NNUE is
 * byte-identical across the scalar/simd128/relaxed builds (verified in
 * `scripts/build_berserk_emscripten.mjs`, which refuses to publish a variant
 * whose `.data` diverges). Pointing every variant at one URL means a relaxed ->
 * simd128 -> scalar fallback reuses the ~24 MB download already in the HTTP
 * cache instead of refetching it. Emscripten resolves the package through
 * `Module.locateFile`, so the name baked into each glue file is irrelevant.
 */
export const BERSERK_EMSCRIPTEN_DATA_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten.data');
export const BERSERK_EMSCRIPTEN_SIMD_JS_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten-simd128.js');
export const BERSERK_EMSCRIPTEN_SIMD_WASM_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten-simd128.wasm');
export const BERSERK_EMSCRIPTEN_RELAXED_JS_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten-relaxed-simd128.js');
export const BERSERK_EMSCRIPTEN_RELAXED_WASM_URL = resolvePublicAssetUrl('/berserk/berserk-emscripten-relaxed-simd128.wasm');
export const BERSERK_DEFAULT_WASM_URL = resolvePublicAssetUrl('/berserk/berserk.wasm');
export const BERSERK_SIMD_WASM_URL = resolvePublicAssetUrl('/berserk/berserk-simd128.wasm');
export const BERSERK_MAIN_NETWORK = 'berserk-9b84c340af7e.nn';
export const BERSERK_DEFAULT_NNUE_URL = resolvePublicAssetUrl(`/berserk/${BERSERK_MAIN_NETWORK}`);
export const BERSERK_SOURCE_NETWORK_URL = `https://github.com/jhonnold/berserk-networks/releases/download/networks/${BERSERK_MAIN_NETWORK}`;

/**
 * User-facing explanation for an absent Berserk artifact.
 *
 * Berserk is the one engine family whose generated artifacts are deliberately
 * NOT committed or deployed: the upstream NNUE in `jhonnold/berserk-networks`
 * has no resolved license/provenance, so this repository must not redistribute
 * it (see `docs/engine_artifact_distribution.md`). Every other engine ships its
 * binaries. A missing Berserk asset is therefore expected, not a broken deploy,
 * and the UI/engine errors should say so.
 */
export const BERSERK_ARTIFACT_BUILD_HINT = 'Berserk artifacts are not distributed with this site (upstream NNUE license/provenance is unresolved) — build locally with `npm run berserk:build-emscripten`.';

const assetStatuses = new Map<string, BerserkAssetStatus>();
const assetChecks = new Map<string, Promise<BerserkAssetStatus>>();

/**
 * Berserk artifact paths that a public deployment actually serves: none.
 *
 * Kept as an explicit empty set (rather than deleting the probe) so that a
 * future release which clears the network license can re-list the shipped
 * paths here and get probing back with no other changes. While it is empty,
 * `shouldSkipKnownUnshippedProbe` short-circuits every `/berserk/` probe on a
 * non-local origin: the status resolves to `missing` immediately, without
 * spending HEAD requests on files we know are absent.
 */
const DEPLOYED_BERSERK_PATHS = new Set<string>([]);

function isLocalDevelopmentOrigin(): boolean {
  if (typeof location === 'undefined') return true;
  return location.hostname === 'localhost' || location.hostname === '127.0.0.1' || location.hostname === '::1' || location.hostname === '[::1]';
}

function assetPathname(raw: string): string {
  try {
    const base = typeof location !== 'undefined' ? location.href : 'http://localhost/';
    return new URL(raw, base).pathname;
  } catch {
    return raw;
  }
}

function shouldSkipKnownUnshippedProbe(variant: BerserkVariant): boolean {
  if (variant.key === 'custom' || isLocalDevelopmentOrigin()) return false;
  return assetUrls(variant).some((url) => {
    const pathname = assetPathname(url);
    return pathname.startsWith('/berserk/') && !DEPLOYED_BERSERK_PATHS.has(pathname);
  });
}

function sameOriginBerserkAsset(raw: string | null | undefined): string | undefined {
  if (!raw) return undefined;
  try {
    const base = typeof location !== 'undefined' ? location.origin : 'http://localhost';
    const url = new URL(raw, base);
    if (url.origin !== base || !url.pathname.startsWith('/berserk/')) return undefined;
    return `${url.pathname}${url.search}`;
  } catch {
    return undefined;
  }
}


function assetUrls(variant: BerserkVariant): string[] {
  if (variant.jsUrl) return [variant.jsUrl, variant.wasmUrl, ...(variant.dataUrl ? [variant.dataUrl] : [])];
  return [variant.wasmUrl, ...(variant.nnueUrl ? [variant.nnueUrl] : [])];
}

function assetKey(variant: BerserkVariant): string {
  return assetUrls(variant).join('\n');
}

export function supportsBerserkWasmSimd(): boolean {
  return supportsWasmSimd();
}

export const BERSERK_EMSCRIPTEN_VARIANT: BerserkVariant = {
  key: 'emscripten',
  label: 'Berserk',
  jsUrl: BERSERK_EMSCRIPTEN_JS_URL,
  wasmUrl: BERSERK_EMSCRIPTEN_WASM_URL,
  dataUrl: BERSERK_EMSCRIPTEN_DATA_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: `Smoked Berserk tag 14 single-thread Emscripten worker build with tablebases disabled and NNUE preloaded in .data. ${BERSERK_ARTIFACT_BUILD_HINT}`,
};

export const BERSERK_EMSCRIPTEN_SIMD_VARIANT: BerserkVariant = {
  key: 'emscripten-simd',
  label: 'Berserk SIMD',
  jsUrl: BERSERK_EMSCRIPTEN_SIMD_JS_URL,
  wasmUrl: BERSERK_EMSCRIPTEN_SIMD_WASM_URL,
  dataUrl: BERSERK_EMSCRIPTEN_DATA_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: `Berserk tag 14 Emscripten build compiling the engine SSE4.1 NNUE path via -msse4.1 -msimd128 intrinsic emulation. 40/40 fixed-depth parity with scalar; ~3.8x scalar NPS in Node. ${BERSERK_ARTIFACT_BUILD_HINT}`,
};

export const BERSERK_EMSCRIPTEN_RELAXED_VARIANT: BerserkVariant = {
  key: 'emscripten-relaxed',
  label: 'Berserk Relaxed SIMD',
  jsUrl: BERSERK_EMSCRIPTEN_RELAXED_JS_URL,
  wasmUrl: BERSERK_EMSCRIPTEN_RELAXED_WASM_URL,
  dataUrl: BERSERK_EMSCRIPTEN_DATA_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: `SIMD Emscripten build whose m128 dpbusd helpers use the relaxed integer dot (exact: InputCReLU8 activations are in 0..127). Requires WebAssembly Relaxed SIMD. ${BERSERK_ARTIFACT_BUILD_HINT}`,
};

export const BERSERK_DEFAULT_VARIANT: BerserkVariant = {
  key: 'default',
  label: 'Berserk scalar WASI planned',
  wasmUrl: BERSERK_DEFAULT_WASM_URL,
  nnueUrl: BERSERK_DEFAULT_NNUE_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: 'Planned Berserk tag 14 scalar wasm32-wasip1 UCI build. Experimental until a WASI browser smoke passes.',
};

export const BERSERK_SIMD_VARIANT: BerserkVariant = {
  key: 'simd',
  label: 'Berserk SIMD WASI planned',
  wasmUrl: BERSERK_SIMD_WASM_URL,
  nnueUrl: BERSERK_DEFAULT_NNUE_URL,
  sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
  note: 'Planned Berserk tag 14 wasm simd128 UCI build. Enable only after scalar smoke and SIMD codegen validation.',
};

export const BERSERK_VARIANTS: readonly BerserkVariant[] = [
  BERSERK_EMSCRIPTEN_VARIANT,
  BERSERK_EMSCRIPTEN_SIMD_VARIANT,
  BERSERK_EMSCRIPTEN_RELAXED_VARIANT,
  BERSERK_DEFAULT_VARIANT,
  BERSERK_SIMD_VARIANT,
];

export function normalizeBerserkVariant(raw: string | null | undefined): BerserkVariantKey {
  const value = String(raw ?? '').toLowerCase().replace(/[ _-]+/g, '');
  if (value === 'emscriptenrelaxed' || value === 'relaxedsimd' || value === 'relaxed' || value === 'emscriptenrelaxedsimd128') return 'emscripten-relaxed';
  if (value === 'emscriptensimd' || value === 'emscriptensimd128' || value === 'jssimd') return 'emscripten-simd';
  if (value === 'emscripten' || value === 'js' || value === 'worker' || value === 'browser') return 'emscripten';
  if (value === 'simd' || value === 'simd128' || value === 'wasmsimd') return 'simd';
  if (value === 'scalar' || value === 'default' || value === 'wasi' || value === 'full') return 'default';
  if (value === 'custom') return 'custom';
  return 'emscripten';
}

// Promotion order: relaxed integer dot > SSE4.1-emulation simd128 > scalar
// Emscripten. All variants are value-exact (40/40 fixed-depth parity), so this
// is a speed ladder gated by WebAssembly feature validation.
export function defaultBerserkVariantKey(): BerserkVariantKey {
  if (supportsWasmRelaxedSimd()) return 'emscripten-relaxed';
  return supportsBerserkWasmSimd() ? 'emscripten-simd' : 'emscripten';
}

export function berserkVariantByKey(key: string): BerserkVariant {
  const normalized = normalizeBerserkVariant(key);
  if (normalized === 'emscripten-simd') return BERSERK_EMSCRIPTEN_SIMD_VARIANT;
  if (normalized === 'emscripten-relaxed') return supportsWasmRelaxedSimd() ? BERSERK_EMSCRIPTEN_RELAXED_VARIANT : supportsBerserkWasmSimd() ? BERSERK_EMSCRIPTEN_SIMD_VARIANT : BERSERK_EMSCRIPTEN_VARIANT;
  if (normalized === 'simd') return BERSERK_SIMD_VARIANT;
  if (normalized === 'default') return BERSERK_DEFAULT_VARIANT;
  if (normalized === 'custom') return {
    key: 'custom',
    label: 'Berserk Custom',
    wasmUrl: BERSERK_EMSCRIPTEN_WASM_URL,
    jsUrl: BERSERK_EMSCRIPTEN_JS_URL,
    dataUrl: BERSERK_EMSCRIPTEN_DATA_URL,
    sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
    note: 'Custom Berserk Emscripten JS URL.',
  };
  return BERSERK_EMSCRIPTEN_VARIANT;
}

export function hasExplicitBerserkVariant(params: URLSearchParams): boolean {
  return params.has('berserkJs') || params.has('berserkWasm') || params.has('berserkVariant') || params.has('berserk');
}

export function berserkVariantFromParams(params: URLSearchParams): BerserkVariant {
  const customJsUrl = sameOriginBerserkAsset(params.get('berserkJs'));
  const customWasmUrl = sameOriginBerserkAsset(params.get('berserkWasm'));
  const customDataUrl = sameOriginBerserkAsset(params.get('berserkData'));
  const customNnueUrl = sameOriginBerserkAsset(params.get('berserkNnue'));
  if (customJsUrl) {
    return {
      key: 'custom',
      label: 'Berserk Custom',
      jsUrl: customJsUrl,
      wasmUrl: customWasmUrl ?? BERSERK_EMSCRIPTEN_WASM_URL,
      dataUrl: customDataUrl ?? BERSERK_EMSCRIPTEN_DATA_URL,
      sourceNetworkUrl: BERSERK_SOURCE_NETWORK_URL,
      note: 'Custom Berserk Emscripten JS URL from ?berserkJs=…',
    };
  }
  const explicit = params.get('berserkVariant') ?? params.get('berserk');
  const variant = berserkVariantByKey(explicit ?? defaultBerserkVariantKey());
  if (!customNnueUrl || variant.jsUrl) return variant;
  return {
    ...variant,
    nnueUrl: customNnueUrl,
    note: `${variant.note} External NNUE overridden by ?berserkNnue=…`,
  };
}

/**
 * Every Berserk tier — including the scalar base — can legitimately be absent,
 * because the artifacts are intentionally untracked (see
 * `BERSERK_ARTIFACT_BUILD_HINT`). The base build stays the terminal fallback
 * since there is no lower tier, but it is probed too so callers report a
 * settled `missing` instead of an indefinite `unknown`; the load error carries
 * the build-locally hint.
 */
export async function resolveDefaultBerserkVariantAssetFallback(variant: BerserkVariant, explicit: boolean, onChange?: () => void): Promise<BerserkVariant> {
  if (explicit) return variant;
  if (variant.key !== 'simd' && variant.key !== 'emscripten-relaxed' && variant.key !== 'emscripten-simd') {
    if (variant.key === 'emscripten') await checkBerserkVariantAsset(variant, onChange);
    return variant;
  }
  const status = await checkBerserkVariantAsset(variant, onChange);
  if (status !== 'missing') return variant;
  if (variant.key === 'emscripten-relaxed' && supportsBerserkWasmSimd()) {
    return resolveDefaultBerserkVariantAssetFallback(BERSERK_EMSCRIPTEN_SIMD_VARIANT, false, onChange);
  }
  await checkBerserkVariantAsset(BERSERK_EMSCRIPTEN_VARIANT, onChange);
  return BERSERK_EMSCRIPTEN_VARIANT;
}

/**
 * Human-readable status line for a Berserk variant, suitable for a diagnostics
 * panel. A `missing` artifact is a documented state, not a deploy failure.
 */
export function berserkVariantAssetNote(variant: BerserkVariant): string {
  const status = berserkVariantAssetStatus(variant);
  if (status === 'missing') return BERSERK_ARTIFACT_BUILD_HINT;
  if (status === 'present') return 'Berserk artifacts found on this origin.';
  return 'Checking for locally built Berserk artifacts…';
}

export function berserkVariantAssetStatus(variant: BerserkVariant): BerserkAssetStatus {
  return assetStatuses.get(assetKey(variant)) ?? 'unknown';
}

export function checkBerserkVariantAsset(variant: BerserkVariant, onChange?: () => void): Promise<BerserkAssetStatus> {
  const key = assetKey(variant);
  const current = assetStatuses.get(key);
  if (current === 'present' || current === 'missing') return Promise.resolve(current);
  const existing = assetChecks.get(key);
  if (existing) return existing.then((status) => { onChange?.(); return status; });
  if (shouldSkipKnownUnshippedProbe(variant)) {
    assetStatuses.set(key, 'missing');
    onChange?.();
    return Promise.resolve('missing');
  }
  assetStatuses.set(key, 'checking');
  const promise = Promise.all(assetUrls(variant).map((url) => fetch(url, { method: 'HEAD', cache: 'no-store' }).then((response) => response.ok).catch(() => false)))
    .then((results) => (results.every(Boolean) ? 'present' : 'missing') as BerserkAssetStatus)
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
