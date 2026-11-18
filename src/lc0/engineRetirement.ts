// Detach ownership synchronously, but do not overlap a replacement model load
// with the old session's asynchronous release (including across route changes).
// Release is best effort: a failed release still ends the wait, as before.
const retiring = new Set<Promise<void>>();
let lastWorkerRetirement = -Infinity;

export function terminateEngineWorker(worker: Worker | null | undefined): void {
  if (!worker) return;
  worker.terminate();
  lastWorkerRetirement = performance.now();
}

/** Recheck the deadline in case another worker terminated during the wait. */
export async function awaitWorkerReclaim(gapMs: number, signal: AbortSignal, onWait: () => void): Promise<void> {
  let announced = false;
  while (true) {
    signal.throwIfAborted();
    const remaining = gapMs - (performance.now() - lastWorkerRetirement);
    if (remaining <= 0 || gapMs === 0) return;
    if (!announced) onWait();
    announced = true;
    signal.throwIfAborted();
    await new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        clearTimeout(timer);
        reject(signal.reason);
      };
      const timer = setTimeout(() => {
        signal.removeEventListener('abort', onAbort);
        resolve();
      }, Math.ceil(remaining));
      signal.addEventListener('abort', onAbort, { once: true });
    });
  }
}

export function trackEngineRetirement(disposal: Promise<unknown>): Promise<void> {
  const boundary = disposal.then(
    () => undefined,
    (error: unknown) => console.warn('Engine release failed.', error),
  );
  retiring.add(boundary);
  void boundary.finally(() => retiring.delete(boundary));
  return boundary;
}

export function hasEngineRetirements(): boolean {
  return retiring.size > 0;
}

export async function awaitEngineRetirements(): Promise<void> {
  while (retiring.size) await Promise.all([...retiring]);
}
