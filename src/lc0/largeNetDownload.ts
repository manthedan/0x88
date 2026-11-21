import { type OrtExecutionProviderPreference, requestedOrtExecutionProvider, resolveFreshWorkerOrtSessionPlacement } from '../nn/ortConfig.ts';
import type { BigNetConfig } from './bt4Engine.ts';
import { isMemoryConstrainedDevice, phoneReclaimSettings } from './deviceMemoryClass.ts';
import { awaitWorkerReclaim, terminateEngineWorker } from './engineRetirement.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from './lc0ModelUrls.ts';
import { inspectLc0ModelDownload, type Lc0ModelDownloadState, type Lc0ModelLoadResult, lc0ModelCacheEntryExists, untilAborted } from './modelCache.ts';
import type { ModelPrefetchReply, ModelPrefetchRequest } from './modelPrefetchWorker.ts';

export function largeNetWorkerEp(): 'wasm' | 'auto' {
  return requestedOrtExecutionProvider() === 'wasm' ? 'wasm' : 'auto';
}

const defaultDeps = { placement: () => resolveFreshWorkerOrtSessionPlacement(largeNetWorkerEp()), inspect: inspectLc0ModelDownload };

export type LargeNetDownloadState = Lc0ModelDownloadState & { mayDownloadFallback?: boolean; downloadBytesIncomplete?: boolean };

/** Estimate the production worker's downloads without creating a session or
 * reading model bytes. Split-K load or WebGPU init failure can load the original.
 */
export async function inspectLargeNetDownload(config: BigNetConfig, signal?: AbortSignal, deps = defaultDeps): Promise<LargeNetDownloadState> {
  signal?.throwIfAborted();
  const originalUrl = config.modelUrl;
  const selectedUrl = lc0SplitKUrl(originalUrl) ? selectLc0OrtModelUrl(originalUrl, await deps.placement()) : originalUrl;
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
  /** The EP preference the session worker will be pinned to, so the prefetch picks its model file. */
  ep: OrtExecutionProviderPreference;
  onProgress?: (loadedBytes: number, totalBytes?: number) => void;
  /** The session load's Cache Storage options, so both name and size-check the same entry. */
  load: Pick<ModelPrefetchRequest, 'requestPersistentStorage' | 'minimumFreeBytesAfterCache'>;
};

/**
 * 'cached': already in Cache Storage; 'prefetched': in it now (a split-K net's worker may
 * have found it there); 'fallback': load as before.
 * 'memory': cache.put failed; `load` holds the verified bytes of `modelUrl` for the
 * session worker, so the net is still downloaded once.
 */
export type LargeNetPrefetchResult =
  | { status: 'fallback' }
  | { status: 'cached' | 'prefetched'; modelUrl: string }
  | { status: 'memory'; modelUrl: string; load: Lc0ModelLoadResult & { model: ArrayBuffer } };

const defaultPrefetchDeps = {
  cached: lc0ModelCacheEntryExists,
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
  const selectFor = lc0SplitKUrl(config.modelUrl) ? options.ep : undefined;
  if (!selectFor) {
    try {
      // Key presence only, no body read: a cached net must not pay for a worker or a second read.
      // The session worker's loader still verifies the entry (metadata, else a rehash).
      if (await untilAborted(deps.cached(config.modelUrl, { signal }), signal)) return { status: 'cached', modelUrl: config.modelUrl };
    } catch (error) {
      signal.throwIfAborted();
      console.warn(`Lc0 ${config.name}: cache probe failed; loading without prefetch.`, error);
      return { status: 'fallback' };
    }
  }
  const worker = deps.spawn();
  let result: LargeNetPrefetchResult = { status: 'fallback' };
  try {
    const reply = new Promise<LargeNetPrefetchResult>((resolve, reject) => {
      worker.addEventListener('message', (event: MessageEvent<ModelPrefetchReply>) => {
        const data = event.data;
        if (data.type === 'progress') options.onProgress?.(data.loadedBytes, data.totalBytes);
        else if (data.type === 'error') reject(new Error(data.error));
        else resolve(data.cached ? { status: 'prefetched', modelUrl: data.modelUrl } : { status: 'memory', modelUrl: data.modelUrl, load: data.load });
      });
      worker.addEventListener('messageerror', () => reject(new Error('model prefetch reply could not be received')));
      worker.addEventListener('error', (event) => reject(new Error(event.message || 'model prefetch worker failed')));
      const request: ModelPrefetchRequest = { type: 'prefetch', modelUrl: config.modelUrl, ...options.load };
      worker.postMessage(selectFor ? { ...request, selectFor } : request);
    });
    result = await untilAborted(reply, signal);
  } catch (error) {
    signal.throwIfAborted();
    // No Cache Storage or quota (rejected before downloading), a network/integrity
    // failure or an undeliverable reply: the session worker loads as before.
    console.warn(`Lc0 ${config.name}: prefetch failed; loading without it.`, error);
  } finally {
    terminateEngineWorker(worker);
  }
  // Counted from the termination above; 0 by default, a diagnostic override otherwise.
  await awaitWorkerReclaim(deps.reclaimGapMs(), signal, () => undefined);
  return result;
}

/** Phones with Cache Storage only. Elsewhere returns undefined, so callers add no await. */
export function phoneColdPrefetch(config: BigNetConfig, options: LargeNetPrefetchOptions): Promise<LargeNetPrefetchResult> | undefined {
  if (!isMemoryConstrainedDevice() || typeof caches === 'undefined') return undefined;
  return prefetchLargeNetToCache(config, options);
}
