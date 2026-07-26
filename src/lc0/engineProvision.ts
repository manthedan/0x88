// Shared CPU-engine construction for the play / arena / analysis pages.
//
// Each page keeps its own variant-selection policy (URL params, custom
// variants, async asset fallback) and its own engine cache, but the
// CONSTRUCTION — default options, constructor wiring, and the cache-key
// derivation those caches rely on — is single-sourced here. Before this
// module the three pages carried parallel copies that had already drifted
// (arena's reckless cache key included the backend, analysis's did not, so
// two backends of the same variant collided in the analysis cache).
import type { BrowserUciEngine } from './browserUciEngine.ts';
import { RecklessEngine } from './recklessEngine.ts';
import { defaultRecklessVariantKey, recklessVariantByKey, resolveDefaultRecklessVariantAssetFallback, type RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine, type ViridithasRuntimeOptions } from './viridithasEngine.ts';
import { defaultViridithasVariantKey, resolveDefaultViridithasVariantAssetFallback, viridithasVariantByKey, type ViridithasVariant } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import { BERSERK_ARTIFACT_BUILD_HINT, berserkVariantAssetStatus, berserkVariantByKey, defaultBerserkVariantKey, resolveDefaultBerserkVariantAssetFallback, type BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import { defaultPlentyChessVariantKey, plentyChessVariantByKey, resolveDefaultPlentyChessVariantAssetFallback, type PlentyChessVariant } from './plentychessVariants.ts';
import { StormphraxEngine } from './stormphraxEngine.ts';
import { defaultStormphraxVariantKey, resolveDefaultStormphraxVariantAssetFallback, stormphraxVariantByKey, type StormphraxVariant } from './stormphraxVariants.ts';

/**
 * Transposition-table size handed to every CPU engine at construction.
 *
 * The previous 16 MB matched nothing but habit: analysis searches run to
 * depth 12+ (see engineCatalog's per-surface strength defaults), where a
 * 16 MB table overwrites itself constantly and costs real strength. 64 MB
 * is the conservative-but-real step up:
 *  - Every Emscripten build we ship reserves INITIAL_MEMORY of 256 MB
 *    (berserk, plentychess) or 512 MB (stormphrax) with ALLOW_MEMORY_GROWTH
 *    and MAXIMUM_MEMORY=2 GB, so a 64 MB table lives inside the heap the
 *    module already reserved and never triggers a growth on its own.
 *  - Arena and analysis can hold several engine workers alive at once
 *    (see resourceBroker). At 64 MB, even eight concurrent engines add up to
 *    512 MB of table — large, but not the term that decides an OOM, since the
 *    reserved heaps already dominate. At 128+ MB per engine it would be.
 * Pages that want more for a single long-running analysis engine can pass
 * `hashMb` through setOptions; this is only the construction-time floor.
 */
export const DEFAULT_CPU_ENGINE_HASH_MB = 64;

/** Construction-time search defaults; pages re-tune per move via setOptions. */
const CPU_ENGINE_DEFAULTS = { depth: 4, hashMb: DEFAULT_CPU_ENGINE_HASH_MB } as const;

export function recklessCacheKey(variant: RecklessVariant): string {
  return `${variant.key}:${variant.wasmUrl}:${variant.nnueUrl ?? ''}:${variant.nnueExpectedBytes ?? ''}:${variant.backend ?? 'wasi'}`;
}

export function viridithasCacheKey(variant: ViridithasVariant): string {
  return `${variant.key}:${variant.wasmUrl}`;
}

export function berserkCacheKey(variant: BerserkVariant): string {
  return `${variant.key}:${variant.jsUrl ?? ''}:${variant.wasmUrl}:${variant.dataUrl ?? ''}`;
}

export function plentyChessCacheKey(variant: PlentyChessVariant): string {
  return `${variant.key}:${variant.jsUrl}:${variant.wasmUrl}:${variant.dataUrl}`;
}

export function stormphraxCacheKey(variant: StormphraxVariant): string {
  return `${variant.key}:${variant.jsUrl}:${variant.wasmUrl}:${variant.dataUrl}`;
}

export function createRecklessEngine(variant: RecklessVariant, onStatus?: () => void): RecklessEngine {
  return new RecklessEngine({ ...CPU_ENGINE_DEFAULTS }, variant.wasmUrl, {
    backend: variant.backend ?? 'wasi',
    nnueUrl: variant.nnueUrl,
    nnueExpectedBytes: variant.nnueExpectedBytes,
    ...(onStatus ? { onStatus } : {}),
  });
}

export function createViridithasEngine(variant: ViridithasVariant, runtimeOptions: ViridithasRuntimeOptions = {}): ViridithasEngine {
  return new ViridithasEngine({ ...CPU_ENGINE_DEFAULTS }, variant.wasmUrl, runtimeOptions);
}

export function createBerserkEngine(variant: BerserkVariant): BerserkEngine {
  return new BerserkEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createPlentyChessEngine(variant: PlentyChessVariant): PlentyChessEngine {
  return new PlentyChessEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createStormphraxEngine(variant: StormphraxVariant): StormphraxEngine {
  return new StormphraxEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export type DefaultBrowserUciFamily = 'reckless' | 'viridithas' | 'berserk' | 'plentychess' | 'stormphrax';

type DefaultBrowserUciProvisioner = (onStatus?: () => void) => Promise<BrowserUciEngine>;

/**
 * Default-variant provisioning used by Play. Analysis and Arena still resolve
 * explicit variants themselves, but a standard UCI family no longer needs a
 * new Play-specific construction branch.
 */
export const DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS: Record<DefaultBrowserUciFamily, DefaultBrowserUciProvisioner> = {
  reckless: async (onStatus) => createRecklessEngine(
    await resolveDefaultRecklessVariantAssetFallback(recklessVariantByKey(defaultRecklessVariantKey()), false),
    onStatus,
  ),
  viridithas: async () => createViridithasEngine(
    await resolveDefaultViridithasVariantAssetFallback(viridithasVariantByKey(defaultViridithasVariantKey()), false),
  ),
  // Berserk is the one family whose artifacts are never deployed (unresolved
  // upstream NNUE license). Every probe on a public origin therefore resolves
  // to `missing`, and constructing the engine anyway would spawn a worker whose
  // glue import is guaranteed to fail with an opaque network error. Fail fast
  // with the build-locally hint instead; callers already surface the message
  // (playBrowser renders "<engine> load failed: <message>").
  berserk: async () => {
    const variant = await resolveDefaultBerserkVariantAssetFallback(berserkVariantByKey(defaultBerserkVariantKey()), false);
    if (berserkVariantAssetStatus(variant) === 'missing') throw new Error(BERSERK_ARTIFACT_BUILD_HINT);
    return createBerserkEngine(variant);
  },
  plentychess: async () => createPlentyChessEngine(
    await resolveDefaultPlentyChessVariantAssetFallback(plentyChessVariantByKey(defaultPlentyChessVariantKey()), false),
  ),
  stormphrax: async () => createStormphraxEngine(
    await resolveDefaultStormphraxVariantAssetFallback(stormphraxVariantByKey(defaultStormphraxVariantKey()), false),
  ),
};

export function isDefaultBrowserUciFamily(family: string): family is DefaultBrowserUciFamily {
  return Object.hasOwn(DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS, family);
}

export function createDefaultBrowserUciEngine(family: DefaultBrowserUciFamily, onStatus?: () => void): Promise<BrowserUciEngine> {
  return DEFAULT_BROWSER_UCI_FAMILY_PROVISIONERS[family](onStatus);
}
