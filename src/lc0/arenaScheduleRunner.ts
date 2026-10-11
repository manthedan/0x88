const CPU_FAMILIES = new Set(['sf', 'reckless', 'viridithas', 'berserk', 'plentychess', 'stormphrax']);

/** Neural seats stay serial even when their runtime falls back to WASM/CPU. */
export function arenaCpuOnlyPool(families: readonly string[]): boolean {
  return families.length >= 2 && families.every((family) => CPU_FAMILIES.has(family));
}

export function arenaGameConcurrency(environment: { hardwareConcurrency?: number; deviceMemory?: number }, requested = 'auto'): number {
  const cores = environment.hardwareConcurrency ?? 1;
  const memory = environment.deviceMemory ?? 0;
  // Unknown memory is treated as low, not as desktop headroom. K=2 is the
  // measured default; K=3 is an explicit opt-in, with two UI cores reserved.
  if (!Number.isFinite(memory) || memory <= 4 || !Number.isFinite(cores) || cores < 4) return 1;
  const cap = Math.min(memory < 16 ? 2 : 3, Math.floor((cores - 2) / 2));
  if (requested === 'auto') return cores >= 8 ? Math.min(2, cap) : 1;
  const value = Number(requested);
  return Number.isInteger(value) && value >= 1 ? Math.max(1, Math.min(cap, value)) : 1;
}

/**
 * Bounded schedule runner. Every game owns a child signal; a failure cancels
 * siblings and is propagated only after all games have settled their cleanup.
 * Completion order is deliberately independent from the stable schedule index.
 */
export async function runArenaSchedule<T>(options: {
  count: number;
  concurrency: number;
  signal: AbortSignal;
  play: (index: number, signal: AbortSignal) => Promise<T>;
  complete: (index: number, result: T) => void;
}): Promise<void> {
  const children = new Set<AbortController>();
  let next = 0;
  let failed = false;
  let failure: unknown;
  const cancel = () => {
    for (const child of children) child.abort();
  };
  options.signal.addEventListener('abort', cancel, { once: true });
  try {
    const lanes = Math.max(1, Math.min(options.count, Math.floor(options.concurrency)));
    await Promise.all(
      Array.from({ length: lanes }, async () => {
        while (!failed && !options.signal.aborted && next < options.count) {
          const index = next++;
          const child = new AbortController();
          children.add(child);
          try {
            const result = await options.play(index, child.signal);
            if (!child.signal.aborted) options.complete(index, result);
          } catch (error) {
            if (!failed) {
              failed = true;
              failure = error;
              cancel();
            }
          } finally {
            children.delete(child);
          }
        }
      }),
    );
    if (failed) throw failure;
  } finally {
    options.signal.removeEventListener('abort', cancel);
  }
}
