function queueAbortError(): Error {
  const error = new Error('Queued task aborted');
  error.name = 'AbortError';
  return error;
}

/**
 * Runs tasks one at a time, in call order.
 *
 * A task whose signal aborts while it waits for its turn leaves at once with
 * the queue's AbortError, without waiting for the task ahead of it; the tasks
 * behind it keep their order. A task that has started is never interrupted by
 * the queue: cancelling running work is the task's own job.
 */
export class ExclusiveQueue {
  private tail: Promise<void> = Promise.resolve();
  private readonly abortError: () => Error;

  constructor(abortError: () => Error = queueAbortError) {
    this.abortError = abortError;
  }

  async run<T>(task: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    const previous = this.tail;
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    this.tail = previous.then(() => gate);
    try {
      await this.waitForTurn(previous, signal);
    } catch (error) {
      release();
      throw error;
    }
    try {
      return await task();
    } finally {
      release();
    }
  }

  /** Settles once every task queued so far has finished. */
  idle(): Promise<void> {
    return this.tail;
  }

  private waitForTurn(previous: Promise<void>, signal?: AbortSignal): Promise<void> {
    if (!signal) return previous;
    if (signal.aborted) return Promise.reject(this.abortError());
    return new Promise<void>((resolve, reject) => {
      const onAbort = () => reject(this.abortError());
      signal.addEventListener('abort', onAbort, { once: true });
      previous.then(resolve).finally(() => signal.removeEventListener('abort', onAbort));
    });
  }
}
