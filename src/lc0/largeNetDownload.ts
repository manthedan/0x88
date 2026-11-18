import { resolveOrtSessionPlacement } from '../nn/ortRuntime.ts';
import type { BigNetConfig } from './bt4Engine.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from './lc0SmallModelSelection.ts';
import { inspectLc0ModelDownload, type Lc0ModelDownloadState } from './modelCache.ts';

const defaultDeps = { placement: resolveOrtSessionPlacement, inspect: inspectLc0ModelDownload };

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
