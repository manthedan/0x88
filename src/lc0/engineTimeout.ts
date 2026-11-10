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
    controller.abort();
    abortReject?.(new DOMException('Analysis aborted', 'AbortError'));
  };
  parentSignal.addEventListener('abort', onAbort, { once: true });
  try {
    const taskPromise = task(controller.signal);
    taskPromise.catch(() => undefined);
    const timeoutPromise = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        controller.abort();
        reject(new Error(timeoutMessage));
      }, timeoutMs);
    });
    return await Promise.race([taskPromise, timeoutPromise, abortPromise]);
  } finally {
    if (timer) clearTimeout(timer);
    parentSignal.removeEventListener('abort', onAbort);
  }
}
