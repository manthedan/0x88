import { resolvePublicAssetUrl } from './assetUrls.ts';
import { acquireBigNetSearcher, BigNetSessionPool, type BigNetKey, detachBigNetSearcher } from './bigNetSessionPool.ts';
import { BIG_NETS, Bt4WorkerSearcher } from './bt4Engine.ts';
import { lc0SplitKUrl } from './lc0SmallModelSelection.ts';
import { validateSoakModelOverride } from './mobileSoak.ts';

/** Lab-only config selection; all allocation, model loading and search remain production code. */
export function createSoakBigNetBinding(key: BigNetKey, modelOverride?: string) {
  const config =
    modelOverride === undefined
      ? BIG_NETS[key]
      : {
          ...BIG_NETS[key],
          name: `${BIG_NETS[key].name} (lab override)`,
          modelUrl: resolvePublicAssetUrl(validateSoakModelOverride(modelOverride)),
          diagnosticModel: true as const,
        };
  // Separate pool namespace even if the override spells the shipped URL. Never alter shipping seats.
  const pool = modelOverride === undefined ? null : new BigNetSessionPool(() => new Bt4WorkerSearcher(config));
  return {
    config,
    acquire: () => (pool ? pool.acquire(key) : acquireBigNetSearcher(key)),
    retire: (searcher: Bt4WorkerSearcher) => (pool ? pool.detach(key, searcher) : detachBigNetSearcher(key, searcher)),
    modelRuntime(searcher?: Bt4WorkerSearcher): Record<string, unknown> {
      const registeredSplitKUrl = lc0SplitKUrl(config.modelUrl) ?? null;
      return {
        modelOverride: modelOverride !== undefined,
        requestedModelUrl: config.modelUrl,
        servedModelUrl: searcher?.servedModelUrl || null,
        deviceModelSelection: true,
        registeredSplitKUrl,
        splitKApplied: searcher?.servedModelUrl ? registeredSplitKUrl === searcher.servedModelUrl : null,
        splitKRule: registeredSplitKUrl
          ? 'Production Apple WebGPU rule enabled; see servedModelUrl for device/fallback outcome.'
          : 'No production split-K rewrite registered for this filename; model remains as requested.',
        integrity: 'Production manifest/hash verification remains required.',
      };
    },
  };
}
