import type { createBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import { CachedEvaluator, type Evaluator } from '../nn/evaluator.ts';
import { squareformerModelSettings } from '../nn/runtimeRegistry.ts';
import { loadRetirableResource } from './arenaResourceLoad.ts';
import { CENTIPAWN_NET_HISTORY_POSITIONS } from './engineCatalog.ts';
import { trackEngineRetirement } from './engineRetirement.ts';

type Loader = typeof createBrowserSquareformerRuntimeEvaluator;
type Request = {
  spec: Parameters<Loader>[0];
  options: Parameters<Loader>[1];
  maxEntries: number;
  label: string;
};

/** Shared loading and ownership: a retired load cannot delete or replace a newer entry. */
export function createCentipawnEvaluatorCache(load: Loader) {
  const entries = new Map<string, { promise: Promise<CachedEvaluator>; value?: CachedEvaluator; abort: AbortController }>();
  const disposeInner = async (evaluator: Evaluator) => {
    const inner = evaluator as Evaluator & { dispose?: () => void | Promise<void>; destroy?: () => void | Promise<void> };
    await (inner.dispose ? inner.dispose() : inner.destroy?.());
  };
  const destroy = async (cached: CachedEvaluator) => {
    cached.clear();
    await disposeInner(cached.inner);
  };
  const remove = (key: string) => {
    const entry = entries.get(key);
    entries.delete(key);
    if (!entry) return;
    entry.abort.abort();
    trackEngineRetirement(entry.value ? destroy(entry.value) : entry.promise.then(destroy, () => undefined));
  };
  return {
    has(key: string): boolean {
      return entries.has(key);
    },
    get(key: string, request: Request, isActive: () => boolean): Promise<CachedEvaluator> {
      const existing = entries.get(key);
      if (existing) return existing.promise;
      const abort = new AbortController();
      const external = request.options?.signal;
      if (external?.aborted) abort.abort(external.reason);
      else external?.addEventListener('abort', () => abort.abort(external.reason), { once: true, signal: abort.signal });
      const entry: { promise: Promise<CachedEvaluator>; value?: CachedEvaluator; abort: AbortController } = {
        abort,
        promise: Promise.resolve()
          .then(() =>
            loadRetirableResource(async () => {
              const loaded = await load(request.spec, { ...request.options, signal: abort.signal });
              return { ...loaded, dispose: () => disposeInner(loaded.evaluator) };
            }, abort.signal),
          )
          .then(async (loaded) => {
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
              maxHistoryPositions: squareformerModelSettings(loaded.modelId).nativeHistory ? undefined : CENTIPAWN_NET_HISTORY_POSITIONS,
            });
            if (entries.get(key) !== entry || !isActive()) {
              await trackEngineRetirement(destroy(cached));
              const error = new Error('Centipawn evaluator was disposed before it finished loading');
              error.name = 'AbortError';
              throw error;
            }
            entry.value = cached;
            return cached;
          })
          .catch((error) => {
            abort.abort();
            if (entries.get(key) === entry) entries.delete(key);
            throw error;
          }),
      };
      entries.set(key, entry);
      void entry.promise.catch(() => undefined);
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
