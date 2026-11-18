import { resolveOrtSessionPlacement } from '../nn/ortRuntime.ts';
import type { BigNetConfig } from './bt4Engine.ts';
import { isMemoryConstrainedDevice, phoneReclaimSettings } from './deviceMemoryClass.ts';
import { awaitWorkerReclaim, terminateEngineWorker } from './engineRetirement.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from './lc0SmallModelSelection.ts';
import { inspectLc0ModelDownload, type Lc0ModelDownloadState, untilAborted } from './modelCache.ts';
import type { ModelPrefetchReply, ModelPrefetchRequest } from './modelPrefetchWorker.ts';

const defaultDeps = { placement: resolveOrtSessionPlacement, inspect: inspectLc0ModelDownload };

export type LargeNetDownloadState = Lc0ModelDownloadState & { mayDownloadFallback?: boolean; downloadBytesIncomplete?: boolean };

/** The model URL the production worker selects first (split-K on Apple WebGPU where listed). */
async function selectedLargeNetUrl(config: BigNetConfig, placement: typeof defaultDeps.placement): Promise<string> {
  const originalUrl = config.modelUrl;
  return lc0SplitKUrl(originalUrl) ? selectLc0OrtModelUrl(originalUrl, await placement()) : originalUrl;
}

/** Estimate the production worker's downloads without creating a session or
 * reading model bytes. Split-K load or WebGPU init failure can load the original.
 */
export async function inspectLargeNetDownload(config: BigNetConfig, signal?: AbortSignal, deps = defaultDeps): Promise<LargeNetDownloadState> {
  signal?.throwIfAborted();
  const originalUrl = config.modelUrl;
  const selectedUrl = await selectedLargeNetUrl(config, deps.placement);
  signal?.throwIfAborted();
  if (selectedUrl === originalUrl) return deps.inspect(originalUrl, { signal });

  const inspect = (url: string) => deps.inspect(url, { signal }).catch((): Lc0ModelDownloadState => ({ url, cached: false }));
  const [selected, original] = await Promise.all([inspect(selectedUrl), inspect(originalUrl)]);
  signal?.throwIfAborted();
  if (selected.cached && original.cached) return selected;
  if (selected.cached) return { ...original, mayDownloadFallback: true };
  // Caching split-K can evict the original before the worker tries its fallback.
  const { downloadBytes: selectedBytes } = selected;
  const { downloadBytes: originalBytes } = original;
  const downloadBytesIncomplete = selectedBytes === undefined || originalBytes === undefined;
  const downloadBytes = downloadBytesIncomplete ? (selectedBytes ?? originalBytes) : selectedBytes + originalBytes;
  return { ...selected, downloadBytes, mayDownloadFallback: true, downloadBytesIncomplete };
}

export type LargeNetPrefetchOptions = {
  /** Aborted when the owning searcher is disposed (Stop, seat swap, page hide). */
  signal: AbortSignal;
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
  /** The session load's Cache Storage options, so both name and size-check the same entry. */
  load: Pick<ModelPrefetchRequest, 'requestPersistentStorage' | 'minimumFreeBytesAfterCache'>;
};

/** 'cached': nothing to do; 'prefetched': Cache Storage now holds it; 'fallback': load it as before. */
export type LargeNetPrefetchResult = 'cached' | 'prefetched' | 'fallback';

const defaultPrefetchDeps = {
  ...defaultDeps,
  spawn: (): Worker => new Worker(new URL('./modelPrefetchWorker.ts', import.meta.url), { type: 'module' }),
  reclaimGapMs: () => phoneReclaimSettings().reclaimGapMs,
};

/**
 * Phone cold loads: download, verify and cache the net in a throwaway worker,
 * terminate it, then let the session worker load it like any cached net.
 * In one worker, a cold load left the download buffer, per-chunk garbage and
 * the SHA-256 and cache.put copies to GC, beside ORT's session create; on an
 * iPhone that pushed the tab past its 2048 MB limit, while cached loads
 * survived. Terminating the worker frees all of it before ORT starts.
 * The worker uses the shared loader, so integrity checks are unchanged.
 */
export async function prefetchLargeNetToCache(
  config: BigNetConfig,
  options: LargeNetPrefetchOptions,
  deps = defaultPrefetchDeps,
): Promise<LargeNetPrefetchResult> {
  const { signal } = options;
  signal.throwIfAborted();
  let modelUrl: string;
  try {
    modelUrl = await untilAborted(selectedLargeNetUrl(config, deps.placement), signal);
    // Byte-free probe: a warm net must not pay for a worker or a second cache read.
    if ((await deps.inspect(modelUrl, { signal })).cached) return 'cached';
  } catch (error) {
    signal.throwIfAborted();
    console.warn(`Lc0 ${config.name}: cache probe failed; loading without prefetch.`, error);
    return 'fallback';
  }
  const worker = deps.spawn();
  let cached = false;
  try {
    const reply = new Promise<boolean>((resolve, reject) => {
      worker.addEventListener('message', (event: MessageEvent<ModelPrefetchReply>) => {
        const data = event.data;
        if (data.type === 'progress') options.onProgress?.(data.loadedBytes, data.totalBytes);
        else if (data.type === 'done') resolve(data.cached);
        else reject(new Error(data.error));
      });
      worker.addEventListener('error', (event) => reject(new Error(event.message || 'model prefetch worker failed')));
      worker.postMessage({ type: 'prefetch', modelUrl, ...options.load } satisfies ModelPrefetchRequest);
    });
    cached = await untilAborted(reply, signal);
  } catch (error) {
    signal.throwIfAborted();
    // No Cache Storage or quota (rejected before downloading), or a network/integrity
    // failure: the session worker loads as before and reports its own error.
    console.warn(`Lc0 ${config.name}: prefetch failed; loading without it.`, error);
  } finally {
    terminateEngineWorker(worker);
  }
  // Counted from the termination above; 0 by default, a diagnostic override otherwise.
  await awaitWorkerReclaim(deps.reclaimGapMs(), signal, () => undefined);
  // A failed cache.put lost the bytes with the worker, so the fallback downloads again.
  // Rare: the quota was checked before the download.
  return cached ? 'prefetched' : 'fallback';
}

/** Phones with Cache Storage only. Elsewhere returns undefined, so callers add no await. */
export function phoneColdPrefetch(config: BigNetConfig, options: LargeNetPrefetchOptions): Promise<LargeNetPrefetchResult> | undefined {
  if (!isMemoryConstrainedDevice() || typeof caches === 'undefined') return undefined;
  return prefetchLargeNetToCache(config, options);
}
