import { resolveOrtSessionPlacement } from '../nn/ortRuntime.ts';
import type { BigNetConfig } from './bt4Engine.ts';
import { lc0SplitKUrl, selectLc0OrtModelUrl } from './lc0SmallModelSelection.ts';
import { inspectLc0ModelDownload } from './modelCache.ts';

const defaultDeps = { placement: resolveOrtSessionPlacement, inspect: inspectLc0ModelDownload };

/** Match the production worker's model choice without creating a worker/session
 * or reading/downloading model bytes. The loader still verifies bytes at use.
 */
export async function inspectLargeNetDownload(config: BigNetConfig, signal?: AbortSignal, deps = defaultDeps) {
  signal?.throwIfAborted();
  const modelUrl = lc0SplitKUrl(config.modelUrl) ? selectLc0OrtModelUrl(config.modelUrl, await deps.placement()) : config.modelUrl;
  signal?.throwIfAborted();
  return deps.inspect(modelUrl, { signal });
}
