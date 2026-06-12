// Shared CPU-engine construction for the play / arena / analysis pages.
//
// Each page keeps its own variant-selection policy (URL params, custom
// variants, async asset fallback) and its own engine cache, but the
// CONSTRUCTION — default options, constructor wiring, and the cache-key
// derivation those caches rely on — is single-sourced here. Before this
// module the three pages carried parallel copies that had already drifted
// (arena's reckless cache key included the backend, analysis's did not, so
// two backends of the same variant collided in the analysis cache).
import { RecklessEngine } from './recklessEngine.ts';
import type { RecklessVariant } from './recklessVariants.ts';
import { ViridithasEngine } from './viridithasEngine.ts';
import type { ViridithasVariant } from './viridithasVariants.ts';
import { BerserkEngine } from './berserkEngine.ts';
import type { BerserkVariant } from './berserkVariants.ts';
import { PlentyChessEngine } from './plentychessEngine.ts';
import type { PlentyChessVariant } from './plentychessVariants.ts';

/** Construction-time search defaults; pages re-tune per move via setOptions. */
const CPU_ENGINE_DEFAULTS = { depth: 4, hashMb: 16 } as const;

export function recklessCacheKey(variant: RecklessVariant): string {
  return `${variant.key}:${variant.wasmUrl}:${variant.nnueUrl ?? ''}:${variant.backend ?? 'wasi'}`;
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

export function createRecklessEngine(variant: RecklessVariant, onStatus?: () => void): RecklessEngine {
  return new RecklessEngine({ ...CPU_ENGINE_DEFAULTS }, variant.wasmUrl, { backend: variant.backend ?? 'wasi', nnueUrl: variant.nnueUrl, ...(onStatus ? { onStatus } : {}) });
}

export function createViridithasEngine(variant: ViridithasVariant): ViridithasEngine {
  return new ViridithasEngine({ ...CPU_ENGINE_DEFAULTS }, variant.wasmUrl);
}

export function createBerserkEngine(variant: BerserkVariant): BerserkEngine {
  return new BerserkEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}

export function createPlentyChessEngine(variant: PlentyChessVariant): PlentyChessEngine {
  return new PlentyChessEngine({ ...CPU_ENGINE_DEFAULTS, threads: 1 }, variant.jsUrl, variant.wasmUrl, variant.dataUrl);
}
