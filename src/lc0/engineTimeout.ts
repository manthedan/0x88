/** Bound engine work, including non-abortable cold startup, without holding its caller's lease. */
export async function withEngineTimeout<T>(label: string, parentSignal: AbortSignal, timeoutMs: number, task: (signal: AbortSignal) => Promise<T>): Promise<T> {
  if (parentSignal.aborted) throw new DOMException('Analysis aborted', 'AbortError');
  const controller = new AbortController();
  const timeoutMessage = `${label} timed out after ${Math.round(timeoutMs / 1000)}s`;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let abortReject: ((error: DOMException) => void) | null = null;
  const abortPromise = new Promise<never>((_, reject) => {
    abortReject = reject;
  });
  const onAbort = () => {
    // Claim the cancellation first, as the deadline does: a task that settles
    // from its own abort listener must not turn Cancel into a partial result.
    abortReject?.(new DOMException('Analysis aborted', 'AbortError'));
    controller.abort();
  };
  parentSignal.addEventListener('abort', onAbort, { once: true });
  try {
    const taskPromise = task(controller.signal);
    taskPromise.catch(() => undefined);
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        // Claim the deadline before abort listeners can settle the task with
        // a partial success or AbortError (which Analysis treats as Cancel).
        reject(new Error(timeoutMessage));
        controller.abort();
      }, timeoutMs);
    });
    return await Promise.race([taskPromise, timeoutPromise, abortPromise]);
  } finally {
    if (timer) clearTimeout(timer);
    parentSignal.removeEventListener('abort', onAbort);
  }
}
