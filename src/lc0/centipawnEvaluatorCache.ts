import type { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';

type Loader = typeof createBrowserSquareformerRuntimeEvaluator;
type Request = {
  spec: Parameters<Loader>[0];
  options: Parameters<Loader>[1];
  maxEntries: number;
  label: string;
};

/** Shared loading and ownership: a retired load cannot delete or replace a newer entry. */
export function createCentipawnEvaluatorCache(load: Loader) {
  const entries = new Map<string, { promise: Promise<CachedEvaluator>; value?: CachedEvaluator }>();
  const destroy = (cached: CachedEvaluator) => {
    cached.clear();
    const inner = cached.inner as Evaluator & { destroy?: () => void };
    inner.destroy?.();
  };
  const remove = (key: string) => {
    const entry = entries.get(key);
    entries.delete(key);
    if (entry?.value) destroy(entry.value);
  };
  return {
    has(key: string): boolean {
      return entries.has(key);
    },
    get(key: string, request: Request, isActive: () => boolean): Promise<CachedEvaluator> {
      const existing = entries.get(key);
      if (existing) return existing.promise;
      const entry: { promise: Promise<CachedEvaluator>; value?: CachedEvaluator } = {
        promise: Promise.resolve()
          .then(async () => {
            const loaded = await load(request.spec, request.options);
            console.info(`[lc0-${request.options?.audit?.surface}] loaded Centipawn evaluator`, {
              requestedRuntime: loaded.requestedRuntime,
              resolvedRuntime: loaded.resolvedRuntime,
              runtimeConfigId: loaded.runtimeConfigId,
              manifestUrl: loaded.manifestUrl,
              fallbackReason: loaded.fallbackReason,
            });
            const cached = new CachedEvaluator(loaded.evaluator, {
              maxEntries: request.maxEntries,
              includeHistory: true,
              includeLegalMoves: true,
              label: request.label,
            });
            if (entries.get(key) !== entry || !isActive()) {
              destroy(cached);
              const error = new Error('Centipawn evaluator was disposed before it finished loading');
              error.name = 'AbortError';
              throw error;
            }
            entry.value = cached;
            return cached;
          })
          .catch((error) => {
            if (entries.get(key) === entry) entries.delete(key);
            throw error;
          }),
      };
      entries.set(key, entry);
      return entry.promise;
    },
    retain(keys: ReadonlySet<string>): void {
      for (const key of entries.keys()) if (!keys.has(key)) remove(key);
    },
    clear(): void {
      for (const key of entries.keys()) remove(key);
    },
  };
}
