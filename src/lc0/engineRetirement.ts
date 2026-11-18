// Detach ownership synchronously, but do not overlap a replacement model load
// with the old session's asynchronous release (including across route changes).
// Release is best effort: a failed release still ends the wait, as before.
const retiring = new Set<Promise<void>>();

/** Worker releases end by construction within disposeOrtWorker's 1500 ms ack
 * timeout plus a synchronous terminate(); only a stalled main-thread release
 * (an unaborted fetch, a GPU queue that never drains) reaches this cap. Past it,
 * the release stops holding the barrier so one stall cannot block every later
 * engine start. */
export const ENGINE_RETIREMENT_CAP_MS = 3000;
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

/** Only the admission barrier is capped; the returned promise is the real
 * release, for callers that must not allocate until it has finished. */
export function trackEngineRetirement(disposal: Promise<unknown>): Promise<void> {
  const released = disposal.then(
    () => undefined,
    (error: unknown) => console.warn('Engine release failed.', error),
  );
  let timer: ReturnType<typeof setTimeout> | undefined;
  const admission = Promise.race([
    released,
    new Promise<void>((resolve) => {
      timer = setTimeout(() => {
        console.warn(`Engine release still pending after ${ENGINE_RETIREMENT_CAP_MS} ms; starting the next engine anyway.`);
        resolve();
      }, ENGINE_RETIREMENT_CAP_MS);
    }),
  ]).finally(() => clearTimeout(timer));
  retiring.add(admission);
  void admission.finally(() => retiring.delete(admission));
  return released;
}

export function hasEngineRetirements(): boolean {
  return retiring.size > 0;
}

/** Waits only for releases registered before the call: a later one may be the
 * caller's own cleanup (a cancelled replacement), and waiting on it would deadlock. */
export async function awaitEngineRetirements(): Promise<void> {
  await Promise.all([...retiring]);
}
