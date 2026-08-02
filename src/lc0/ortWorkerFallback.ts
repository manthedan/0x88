export function workerFallbackReplayAbort(requestType: unknown, searchCancellationRequested: boolean): Error | undefined {
  if (requestType !== 'search' || !searchCancellationRequested) return undefined;
  const error = new Error('LC0 search cancelled during ORT worker fallback');
  error.name = 'AbortError';
  return error;
}
