import type { createBrowserSquareformerRuntimeEvaluator as CreateBrowserSquareformerRuntimeEvaluator } from '../nn/browserRuntimeEvaluator.ts';
import type { createLc0OnnxEvaluatorFromUrl as CreateLc0OnnxEvaluatorFromUrl } from './lc0SmallModelSelection.ts';
import { untilAborted } from './modelCache.ts';

async function withImportedEvaluator<M, T>(load: () => Promise<M>, signal: AbortSignal | undefined, create: (runtime: M) => Promise<T>): Promise<T> {
  signal?.throwIfAborted();
  const runtime = await untilAborted(load(), signal);
  signal?.throwIfAborted();
  return create(runtime);
}

export const createBrowserSquareformerRuntimeEvaluator: typeof CreateBrowserSquareformerRuntimeEvaluator = (spec, options) =>
  withImportedEvaluator(
    () => import('../nn/browserRuntimeEvaluator.ts'),
    options?.signal,
    (runtime) => runtime.createBrowserSquareformerRuntimeEvaluator(spec, options),
  );

export const createLc0OnnxEvaluatorFromUrl: typeof CreateLc0OnnxEvaluatorFromUrl = (url, options, deps) =>
  withImportedEvaluator(
    () => import('./lc0SmallModelSelection.ts'),
    options?.load?.signal,
    (runtime) => runtime.createLc0OnnxEvaluatorFromUrl(url, options, deps),
  );
