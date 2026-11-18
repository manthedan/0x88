// Throwaway worker for phone cold loads (largeNetDownload.ts prefetchLargeNetToCache).
// It imports only the model cache, never ORT; its owner terminates it after one reply.
import { type Lc0ModelLoadResult, prefetchLc0ModelToCache } from './modelCache.ts';

export type ModelPrefetchRequest = {
  type: 'prefetch';
  modelUrl: string;
  requestPersistentStorage?: boolean;
  minimumFreeBytesAfterCache?: number;
};

export type ModelPrefetchReply =
  | { type: 'progress'; loadedBytes: number; totalBytes?: number }
  | { type: 'done'; cached: true }
  /** cache.put failed: the verified bytes, transferred so they need not be downloaded again. */
  | { type: 'done'; cached: false; load: Lc0ModelLoadResult & { model: ArrayBuffer } }
  | { type: 'error'; error: string };

const post = (reply: ModelPrefetchReply, transfer: Transferable[] = []) => self.postMessage(reply, { transfer });

self.addEventListener('message', (event: MessageEvent<ModelPrefetchRequest>) => {
  const { type, modelUrl, requestPersistentStorage, minimumFreeBytesAfterCache } = event.data;
  if (type !== 'prefetch') return;
  // ~2 MB steps, like the search worker, so a large net does not post once per chunk.
  let lastReported = -Infinity;
  prefetchLc0ModelToCache(modelUrl, {
    requestPersistentStorage,
    minimumFreeBytesAfterCache,
    onProgress: (loadedBytes, totalBytes) => {
      if (loadedBytes - lastReported < 2_000_000 && loadedBytes !== totalBytes) return;
      lastReported = loadedBytes;
      post({ type: 'progress', loadedBytes, totalBytes });
    },
  }).then(
    (load) => {
      if (load.mode === 'cache') post({ type: 'done', cached: true });
      else {
        const model = load.model as ArrayBuffer; // cacheOnly never yields a URL
        post({ type: 'done', cached: false, load: { ...load, model } }, [model]);
      }
    },
    (error: unknown) => post({ type: 'error', error: error instanceof Error ? error.message : String(error) }),
  );
});
