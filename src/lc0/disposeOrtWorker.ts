/** Terminal worker handshake. Detach the owner synchronously before calling this.
 * The dedicated listener still observes the ack after the owner's transport is retired.
 * Timeout/error means best effort failed; termination remains the hard boundary.
 */
const closing = new WeakMap<Worker, Promise<void>>();
export function disposeOrtWorker(worker: Worker, timeoutMs = 1500): Promise<void> {
  const existing = closing.get(worker);
  if (existing) return existing;
  const promise = new Promise<void>((resolve, reject) => {
    let finished = false;
    let timer: ReturnType<typeof setTimeout>;
    const finish = (failure?: Error) => {
      if (finished) return;
      finished = true;
      clearTimeout(timer);
      worker.removeEventListener('message', message);
      worker.removeEventListener('error', failed);
      worker.removeEventListener('messageerror', failed);
      try {
        worker.terminate();
        if (failure) reject(failure);
        else resolve();
      } catch (error) {
        reject(error);
      }
    };
    const failed = (event: Event) => finish(new Error((event as ErrorEvent).message || 'ORT worker disposal transport failed'));
    const message = (event: MessageEvent) => {
      if (event.data?.id !== -1) return;
      if (event.data.type === 'disposed') finish();
      else if (event.data.type === 'error') finish(new Error(event.data.error ?? event.data.message ?? 'ORT worker release failed'));
    };
    worker.addEventListener('message', message);
    worker.addEventListener('error', failed);
    worker.addEventListener('messageerror', failed);
    timer = setTimeout(() => finish(), timeoutMs);
    try {
      worker.postMessage({ type: 'cancel' }); // Out-of-band: drain a running search before queued release.
      worker.postMessage({ type: 'dispose', id: -1 });
    } catch {
      finish();
    }
  });
  closing.set(worker, promise);
  return promise;
}
