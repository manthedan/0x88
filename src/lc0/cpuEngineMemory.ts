export type CpuEngineSurface = 'play' | 'arena' | 'analysis';

/**
 * Transposition-table budgets by surface.
 *
 * Play and Arena stay at 64 MB because they may keep multiple WASM engines
 * alive concurrently. Analysis gets 128 MB: its deeper searches benefit from
 * the extra table, while two simultaneous Stockfish variants or several
 * pooled CPU families still remain below the wasm32 memory ceiling.
 */
export const CPU_ENGINE_HASH_MB_BY_SURFACE: Readonly<Record<CpuEngineSurface, number>> = Object.freeze({
  play: 64,
  arena: 64,
  analysis: 128,
});

export const DEFAULT_CPU_ENGINE_HASH_MB = CPU_ENGINE_HASH_MB_BY_SURFACE.play;

export function cpuEngineHashMbForSurface(surface: CpuEngineSurface): number {
  return CPU_ENGINE_HASH_MB_BY_SURFACE[surface];
}
