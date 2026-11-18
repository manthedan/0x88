import type { Lc0ModelLoadResult } from './modelCache.ts';

export function ortErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export class OrtSessionFallbackError extends Error {
  constructor(firstMessage: string, lastError: unknown) {
    super(`${firstMessage} (fallback: ${ortErrorMessage(lastError)})`, { cause: lastError });
  }
}

/** Data failures only: a device loss, OOM or unsupported operator must not evict a good model. */
export function isOnnxModelParseError(error: unknown): boolean {
  if (error instanceof OrtSessionFallbackError) return isOnnxModelParseError(error.cause);
  if ((error as { name?: unknown } | null)?.name === 'AbortError') return false;
  const message = ortErrorMessage(error);
  if (/out of memory|bad_alloc|allocation failed|device (?:was )?lost/i.test(message)) return false;
  return /protobuf|modelproto|invalid[_ -]proto|ERROR_CODE:\s*7(?:\D|$)|(?:model|onnx).{0,80}pars(?:e|ing)|pars(?:e|ing).{0,80}(?:model|onnx)/i.test(message);
}

export function isMetadataTrustedModel(load: Pick<Lc0ModelLoadResult, 'cacheStatus' | 'verification' | 'sha256Valid'> | undefined): boolean {
  return load?.cacheStatus === 'hit' && load.verification === 'metadata' && load.sha256Valid === true;
}
