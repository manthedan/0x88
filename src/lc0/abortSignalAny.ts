// Safari before 17.4 does not implement AbortSignal.any().
export function abortSignalAny(signals: AbortSignal[]): AbortSignal {
  if (typeof AbortSignal.any === 'function') return AbortSignal.any(signals);
  const controller = new AbortController();
  const aborted = signals.find((signal) => signal.aborted);
  if (aborted) controller.abort(aborted.reason);
  else {
    for (const signal of signals) {
      signal.addEventListener('abort', () => controller.abort(signal.reason), { signal: controller.signal });
    }
  }
  return controller.signal;
}
