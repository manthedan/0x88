// Throwaway worker for phone cold loads (largeNetDownload.ts prefetchLargeNetToCache).
// It imports the model cache, plus ORT's placement probe for nets with a split-K file;
// it never creates a session, and its owner terminates it after one reply.
import type { OrtExecutionProviderPreference } from '../nn/ortRuntime.ts';
import { type Lc0ModelLoadResult, lc0ModelCacheEntryExists, prefetchLc0ModelToCache } from './modelCache.ts';

export type ModelPrefetchRequest = {
  type: 'prefetch';
  /** The net's shipped model URL. */
  modelUrl: string;
  /** Set for nets with a split-K file: the EP the session worker will be sent, to pick the file it will load. */
  selectFor?: OrtExecutionProviderPreference;
  requestPersistentStorage?: boolean;
  minimumFreeBytesAfterCache?: number;
};

export type ModelPrefetchReply =
  | { type: 'progress'; loadedBytes: number; totalBytes?: number }
  /** The selected file is in Cache Storage (it was already, or is now). */
  | { type: 'done'; cached: true; modelUrl: string }
  /** cache.put failed: the verified bytes of `modelUrl`, transferred so they need not be downloaded again. */
  | { type: 'done'; cached: false; modelUrl: string; load: Lc0ModelLoadResult & { model: ArrayBuffer } }
  | { type: 'error'; error: string };

const post = (reply: ModelPrefetchReply, transfer: Transferable[] = []) => self.postMessage(reply, { transfer });

async function selectPrefetchModelUrl(modelUrl: string, selectFor: OrtExecutionProviderPreference | undefined): Promise<string> {
  if (!selectFor) return modelUrl;
  const [{ resolveOrtSessionPlacement }, { selectLc0OrtModelUrl }] = await Promise.all([import('../nn/ortRuntime.ts'), import('./lc0SmallModelSelection.ts')]);
  return selectLc0OrtModelUrl(modelUrl, await resolveOrtSessionPlacement(selectFor));
}

async function prefetch(request: ModelPrefetchRequest): Promise<void> {
  const { requestPersistentStorage, minimumFreeBytesAfterCache } = request;
  const modelUrl = await selectPrefetchModelUrl(request.modelUrl, request.selectFor);
  // Key presence only, no body read; the session worker's loader still verifies the entry.
  if (await lc0ModelCacheEntryExists(modelUrl)) return post({ type: 'done', cached: true, modelUrl });
  // ~2 MB steps, like the search worker, so a large net does not post once per chunk.
  let lastReported = -Infinity;
  const load = await prefetchLc0ModelToCache(modelUrl, {
    requestPersistentStorage,
    minimumFreeBytesAfterCache,
    onProgress: (loadedBytes, totalBytes) => {
      if (loadedBytes - lastReported < 2_000_000 && loadedBytes !== totalBytes) return;
      lastReported = loadedBytes;
      post({ type: 'progress', loadedBytes, totalBytes });
    },
  });
  if (load.mode === 'cache') return post({ type: 'done', cached: true, modelUrl });
  const model = load.model as ArrayBuffer; // cacheOnly never yields a URL
  post({ type: 'done', cached: false, modelUrl, load: { ...load, model } }, [model]);
}

self.addEventListener('message', (event: MessageEvent<ModelPrefetchRequest>) => {
  if (event.data.type !== 'prefetch') return;
  // Every failure, including a reply that cannot be posted (cloned or transferred),
  // ends in an error reply, so the owner falls back instead of waiting for dispose.
  prefetch(event.data).catch((error: unknown) => post({ type: 'error', error: error instanceof Error ? error.message : String(error) }));
});
